# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MCore-facing utilities for the DSv4 THD context-parallel path.

This module owns CP row mapping, boundary exchange, compressor-input layout,
and indexer top-k metadata. It reuses MCore's fused MLA RoPE and calls the
retained compaction kernel; ``csa.py`` calls final-index lowering directly.
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.fusions.fused_mla_yarn_rope_apply import fused_mla_rope_inplace
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

from . import cp_layout_kernels
from .fused_sparse_attention import indexer_topk

# =============================================================================
# RoPE Wrappers
# =============================================================================


def _thd_cp_position_ids(
    cu_seqlens_padded: torch.Tensor, global_start: int, local_rows: int
) -> torch.Tensor:
    """Map a consecutive CP row interval to positions within packed sequences."""
    global_rows = torch.arange(
        int(global_start),
        int(global_start) + int(local_rows),
        dtype=cu_seqlens_padded.dtype,
        device=cu_seqlens_padded.device,
    )
    sequence_ids = torch.bucketize(
        global_rows, cu_seqlens_padded[1:], out_int32=True, right=True
    ).clamp_max(cu_seqlens_padded.shape[0] - 2)
    sequence_starts = cu_seqlens_padded[sequence_ids]
    sequence_ends = cu_seqlens_padded[sequence_ids + 1]
    valid_rows = (global_rows >= sequence_starts) & (global_rows < sequence_ends)
    return torch.where(valid_rows, global_rows - sequence_starts, 0)


def apply_thd_cp_local_rope_fused(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply fused non-interleaved RoPE to local THD CP rows."""
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    if inverse:
        # The fused kernel is in-place, but sparse-attention backward needs its original output.
        rope_input = rope_input.clone()
    output = fused_mla_rope_inplace(
        rope_input,
        cos,
        sin,
        nope_dim,
        pos_dim,
        cu_seqlens_q=cu_seqlens_padded,
        inverse=inverse,
        remove_interleaving=True,
        position_ids=position_ids,
    )
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


def apply_thd_cp_local_rope_unfused(
    x: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    cu_seqlens_padded: torch.Tensor,
    global_start: int,
    config,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply unfused RoPE to a consecutive interval of packed CP rows."""
    position_ids = _thd_cp_position_ids(cu_seqlens_padded, global_start, x.shape[0])
    freqs = torch.index_select(rotary_pos_emb, 0, position_ids.long())

    squeezed_batch = x.ndim == 4 and x.shape[1] == 1
    squeezed_head = x.ndim == 2
    rope_input = x.squeeze(1) if squeezed_batch else x
    rope_input = rope_input.unsqueeze(1) if squeezed_head else rope_input
    content, rotary = torch.split(rope_input, [nope_dim, pos_dim], dim=-1)
    rotary = _apply_rotary_pos_emb_bshd(
        rotary,
        freqs,
        rotary_interleaved=config.rotary_interleaved,
        mscale=1.0,
        mla_rotary_interleaved=True,
        inverse=inverse,
        mla_output_remove_interleaving=True,
    )
    output = torch.cat((content, rotary), dim=-1)
    if squeezed_batch:
        return output.unsqueeze(1)
    if squeezed_head:
        return output.squeeze(1)
    return output


# =============================================================================
# Boundary Hidden Exchange
# =============================================================================


class _LeftBoundaryExchange(torch.autograd.Function):
    """Exchange fixed left-boundary windows and scatter gradients back to senders."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group: torch.distributed.ProcessGroup):
        """Receive fixed left-boundary hidden rows needed by this CP rank."""
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DSv4 CP boundary exchange requires local rows >= D_window: "
                f"local_rows={tensor.shape[0]}, D_window={d_window}."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        """Send boundary gradients back to ranks that own those hidden rows."""
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        d_window = ctx.d_window
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        if cp_rank > 0:
            send_grad = grad_boundary.contiguous()
            ops.append(
                dist.P2POp(
                    dist.isend, send_grad, dist.get_global_rank(cp_group, cp_rank - 1), cp_group
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(
                    dist.irecv, recv_grad, dist.get_global_rank(cp_group, cp_rank + 1), cp_group
                )
            )
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        if cp_rank + 1 < cp_size:
            grad_input[-d_window:] = recv_grad
        return grad_input, None, None


def exchange_cp_boundary_hidden(
    hidden_states: torch.Tensor,
    compress_ratio: int,
    csa_window_size: int,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Exchange hidden-state rows immediately left of this rank's token block."""
    d_comp = 8 if compress_ratio == 4 else compress_ratio if compress_ratio > 1 else 0
    d_window = max(int(csa_window_size), d_comp)
    hidden_flat = hidden_states.view(hidden_states.shape[0], -1)
    boundary_hidden = _LeftBoundaryExchange.apply(hidden_flat, d_window, cp_group)
    return boundary_hidden.reshape((d_window,) + tuple(hidden_states.shape[1:]))


# =============================================================================
# Compressed Metadata And Compressor Inputs
# =============================================================================


def prepare_cp_compressor_input(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    cp_size: int,
    ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-capacity compressor input for this rank's token block.

    Returns:
        ``hidden_compact``: rank-local compressor input, shape
            ``(compact_group_capacity * ratio, ...)``.
        ``compressed_group_ids``: original per-sequence compressed group id for each
            compact group, shape ``(compact_group_capacity,)``. For example,
            with ``ratio=4``, ``comp_id=3`` maps to RoPE position ``12``.
        ``seq_to_rank_row``: map from global sequence-major compressed rows
            to their canonical rank-major all-gather rows.
            If rank 0 owns ``A0, A1`` and rank 1 owns ``B0, B1``, with four
            slots per rank, logical rows ``[A0, A1, B0, B1]`` are stored as
            ``[A0, A1, pad, pad | B0, B1, pad, pad]`` and map to ``[0, 1, 4, 5]``.
    """
    cp_size = int(cp_size)
    ratio = int(ratio)
    d_comp = 8 if ratio == 4 else ratio
    global_start = int(global_start)
    l_local = hidden_local.shape[0]
    group_alignment = 32 // math.gcd(32, ratio)
    c_cap = max(1, (l_local + d_comp) // ratio)
    c_cap = ((c_cap + group_alignment - 1) // group_alignment) * group_alignment
    hidden_compact, compressed_group_ids = cp_layout_kernels.CompressorInputCompact.apply(
        hidden_local, boundary_hidden, cu_seqlens, global_start, ratio, d_comp, c_cap
    )

    # A compressed group belongs to the rank containing its last token. From
    # that rank's first visible compressed row, its fixed-capacity
    # rank-major slot follows directly; no (seq, comp, valid) tensors or repack
    # kernel are needed.
    seq_major_rows = (l_local * cp_size) // ratio
    logical_rows = torch.arange(seq_major_rows, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    n_seq = cu_seqlens.shape[0] - 1
    seq_ids = torch.bucketize(
        logical_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    comp_ids = logical_rows - cu_seqlens_compressed[seq_ids]
    group_last_rows = cu_seqlens[seq_ids] + (comp_ids + 1) * ratio - 1
    owner_ranks = torch.div(group_last_rows, l_local, rounding_mode="floor").clamp_(0, cp_size - 1)

    rank_starts = torch.arange(cp_size, dtype=cu_seqlens.dtype, device=cu_seqlens.device) * l_local
    first_seq_ids = torch.bucketize(
        rank_starts, cu_seqlens[1:], out_int32=True, right=True
    ).clamp_max(n_seq - 1)
    first_comp_ids = torch.div(
        (rank_starts - d_comp - cu_seqlens[first_seq_ids]).clamp_min_(0) + ratio - 1,
        ratio,
        rounding_mode="floor",
    )
    first_logical_rows = cu_seqlens_compressed[first_seq_ids] + first_comp_ids
    rank_slots = logical_rows - first_logical_rows[owner_ranks]
    rank_rows = owner_ranks * compressed_group_ids.shape[0] + rank_slots
    seq_to_rank_row = torch.where(logical_rows < cu_seqlens_compressed[-1], rank_rows, -1).to(
        torch.int32
    )
    return hidden_compact, compressed_group_ids, seq_to_rank_row


@torch.compile
def _build_cp_indexer_layout(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    local_rows: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the indexer's packed local-Q/full-K metadata."""
    # Each real Q segment intersects its sequence with this rank's row interval,
    # while K keeps the sequence's full compressed segment. The final synthetic
    # segment holds CP capacity padding and has zero K rows. Causal offsets
    # restore each non-empty local Q segment's position in the original sequence.
    global_end = global_start + local_rows
    zero = torch.zeros((1,), dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)
    local_starts = cu_seqlens_q[:-1].clamp_min(global_start)
    local_ends = cu_seqlens_q[1:].clamp_max(global_end)
    q_lens = (local_ends - local_starts).clamp_min(0)
    q_prefix = torch.cumsum(q_lens, dim=0, dtype=torch.int32)
    padding_q = (global_end - cu_seqlens_q[-1].clamp_min(global_start)).clamp_min(0)
    cu_q_topk = torch.cat((zero, q_prefix, (q_prefix[-1] + padding_q).view(1)))
    cu_k_topk = torch.cat((cu_seqlens_compressed, cu_seqlens_compressed[-1:]))
    q_causal_offsets = torch.cat(
        (torch.where(q_lens > 0, local_starts - cu_seqlens_q[:-1], 0), zero)
    )
    return cu_q_topk, cu_k_topk, q_causal_offsets


def _plan_h64_litetopk_tile_slices(local_rows: int) -> tuple[tuple[int, int], ...]:
    """Cover the latest possible Q rows with qualified 4096/4032 tiles.

    Keeping an unsupported remainder at the beginning is intentional: later causal rows are
    more likely to expose HOT12K and therefore dispatch LiteTopK.  A 4032 tail absorbs all but at
    most 4095 leading rows when the local shard is not an exact multiple of 4096.
    """
    local_rows = int(local_rows)
    if local_rows <= 0:
        return ()
    if local_rows % 4096 == 0:
        return tuple((start, start + 4096) for start in range(0, local_rows, 4096))
    if local_rows < 4032:
        return ()
    tail_start = local_rows - 4032
    first_tiled_row = tail_start % 4096
    tiles = [(start, start + 4096) for start in range(first_tiled_row, tail_start, 4096)]
    tiles.append((tail_start, local_rows))
    return tuple(tiles)


def _plan_stock_fused_q_tile_slices(
    start: int, end: int, tile_rows: int = 4096
) -> tuple[tuple[int, int], ...]:
    """Split an official fused-selector Q interval into memory-bounded tiles.

    The stock selector materializes an FP32 ``[Q, compressed_K]`` score matrix.
    Capping Q at 4096 bounds that temporary to 2/3/4 GiB per rank for global
    512K/768K/1M inputs under CP8, while preserving the official kernel, packed
    sequence metadata, row order, and exact concatenated result.
    """
    start, end, tile_rows = int(start), int(end), int(tile_rows)
    if start < 0 or end < start:
        raise ValueError(f"invalid stock fused Q interval [{start}, {end})")
    if tile_rows <= 0:
        raise ValueError(f"stock fused Q tile size must be positive, got {tile_rows}")
    return tuple(
        (tile_start, min(tile_start + tile_rows, end))
        for tile_start in range(start, end, tile_rows)
    )


def compute_cp_indexer_topk(
    q_indexer_local: torch.Tensor,
    weights_indexer_local: torch.Tensor,
    k_indexer_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    use_fused: bool,
    cp_size: int = 1,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    litetopk_state_key: object = None,
    litetopk_request_key: object = None,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """Return local top-k and its local-Q/full-K packed layout."""
    del cp_group  # Compatibility with callers sharing the stock CP layout API.
    topk_width = int(topk_width)
    if topk_width == 0 or k_indexer_seq_major.shape[0] == 0:
        return None, None
    max_seqlen_kv = int(max_seqlen_q) // int(ratio)
    if max_seqlen_kv == 0:
        return None, None

    global_start = int(global_start)
    l_local = q_indexer_local.shape[0]
    if weights_indexer_local.shape[0] != l_local:
        raise RuntimeError(
            "DSv4 CP indexer top-k expects weights rows to be "
            f"{l_local}, got {weights_indexer_local.shape[0]}."
        )

    cu_q_topk, cu_k_topk, q_causal_offsets = _build_cp_indexer_layout(
        cu_seqlens_q, cu_seqlens_compressed, global_start, l_local
    )

    def finish(topk: torch.Tensor):
        # Only a complete, unpadded CP1 request may publish local carry.
        cp_carry_layout_supported = (
            cp_size == 1
            and int(global_start) == 0
            and cu_seqlens_q.numel() == 2
            and cu_seqlens_compressed.numel() == 2
            and int(max_seqlen_q) == int(l_local)
            and int(max_seqlen_q) % int(ratio) == 0
            and int(k_indexer_seq_major.shape[0]) == int(max_seqlen_q) // int(ratio)
        )
        if litetopk_state_key is not None and cp_carry_layout_supported:
            from megatron.core.transformer.experimental_attention_variant import (
                dsa_litetopk_kernels,
            )

            dsa_litetopk_kernels.observe_reference_topk(
                topk,
                state_key=litetopk_state_key,
                request_key=litetopk_request_key,
                sequence_length=int(k_indexer_seq_major.shape[0]),
                cp_size=cp_size,
                use_fp4=True,
            )
        return topk, (cu_q_topk, cu_k_topk, q_causal_offsets)

    def run_stock_fused_interval(start: int, end: int) -> torch.Tensor:
        tile_cu_q, tile_cu_k, tile_q_offsets = _build_cp_indexer_layout(
            cu_seqlens_q, cu_seqlens_compressed, global_start + int(start), int(end) - int(start)
        )
        tile_topk, _ = indexer_topk(
            q_indexer_local[start:end].contiguous(),
            k_indexer_seq_major,
            weights_indexer_local[start:end].contiguous(),
            topk=topk_width,
            ratio=ratio,
            indexer_softmax_scale=indexer_softmax_scale,
            cu_seqlens_q=tile_cu_q,
            cu_seqlens_kv=tile_cu_k,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_kv=int(max_seqlen_kv),
            q_causal_offsets=tile_q_offsets,
        )
        return tile_topk

    def run_stock_fused_range(
        start: int, end: int, *, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        slices = _plan_stock_fused_q_tile_slices(start, end)
        if not slices:
            # Preserve the pre-tiling empty-Q behavior instead of concatenating
            # an empty list and changing the official kernel's contract.
            result = run_stock_fused_interval(start, end)
            if output is not None:
                output.copy_(result)
                return output
            return result
        if output is not None:
            expected_shape = (end - start, topk_width)
            if (
                tuple(output.shape) != expected_shape
                or output.dtype != torch.int32
                or output.device != q_indexer_local.device
                or not output.is_contiguous()
            ):
                raise RuntimeError(
                    "stock fused interval output must be contiguous int32 "
                    f"{expected_shape} on {q_indexer_local.device}"
                )
            for tile_start, tile_end in slices:
                output[tile_start - start : tile_end - start].copy_(
                    run_stock_fused_interval(tile_start, tile_end)
                )
            return output
        pieces = [run_stock_fused_interval(tile_start, tile_end) for tile_start, tile_end in slices]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)

    if use_fused and litetopk_state_key is not None:
        # Keep the H64 production dependency lazy so the stock CSA path does not import Triton or
        # the external source-pinned adapter.  A decline preserves the existing fused/unfused
        # selector exactly.
        from megatron.core.transformer.experimental_attention_variant import csa_litetopk_kernels

        litetopk = csa_litetopk_kernels.run_cp_indexer_topk(
            q_indexer_local,
            k_indexer_seq_major,
            weights_indexer_local,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_compressed=cu_seqlens_compressed,
            global_start=global_start,
            ratio=ratio,
            topk_width=topk_width,
            indexer_softmax_scale=indexer_softmax_scale,
            max_seqlen_q=max_seqlen_q,
            cp_size=cp_size,
            state_key=litetopk_state_key,
            request_key=litetopk_request_key,
        )
        if litetopk is not None:
            return finish(litetopk)

        # Whole CP-local shards are normally much larger than one qualified production Q shape.
        # Pack K once in the adapter, dispatch every eligible late tile, and use the unchanged
        # official fused selector for each remaining contiguous interval. If no LiteTopK tile
        # dispatches, the same official selector runs through memory-bounded Q tiles below.
        tiled_fn = getattr(csa_litetopk_kernels, "run_cp_indexer_topk_tiles", None)
        tile_slices = _plan_h64_litetopk_tile_slices(l_local)
        if use_fused and tiled_fn is not None and l_local not in (4096, 4032) and tile_slices:
            # Let LiteTopK and any stock gaps write directly into the final tensor.  This avoids
            # a second full-Q allocation and torch.cat after the selector has already produced
            # every admitted row.
            tiled_output = torch.empty(
                (l_local, topk_width), dtype=torch.int32, device=q_indexer_local.device
            )
            tile_results = tiled_fn(
                q_indexer_local,
                k_indexer_seq_major,
                weights_indexer_local,
                tile_slices=tile_slices,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_compressed=cu_seqlens_compressed,
                global_start=global_start,
                ratio=ratio,
                topk_width=topk_width,
                indexer_softmax_scale=indexer_softmax_scale,
                max_seqlen_q=max_seqlen_q,
                cp_size=cp_size,
                state_key=litetopk_state_key,
                request_key=litetopk_request_key,
                output=tiled_output,
            )
            if tile_results is not None:
                if len(tile_results) != len(tile_slices):
                    raise RuntimeError(
                        "H64 tiled LiteTopK returned a result count that does not match its Q plan"
                    )
                unresolved_start = 0
                for (start, end), tile_result in zip(tile_slices, tile_results):
                    if tile_result is None:
                        continue
                    expected_shape = (end - start, topk_width)
                    if (
                        tuple(tile_result.shape) != expected_shape
                        or tile_result.dtype != torch.int32
                        or tile_result.device != q_indexer_local.device
                        or not tile_result.is_contiguous()
                    ):
                        raise RuntimeError(
                            "H64 tiled LiteTopK output contract mismatch: "
                            f"expected shape={expected_shape}, dtype=int32, "
                            f"device={q_indexer_local.device}; "
                            f"got shape={tuple(tile_result.shape)}, "
                            f"dtype={tile_result.dtype}, device={tile_result.device}"
                        )
                    if unresolved_start < start:
                        run_stock_fused_range(
                            unresolved_start, start, output=tiled_output[unresolved_start:start]
                        )
                    target = tiled_output[start:end]
                    if tile_result.data_ptr() != target.data_ptr():
                        target.copy_(tile_result)
                    unresolved_start = end
                if unresolved_start < l_local:
                    run_stock_fused_range(
                        unresolved_start, l_local, output=tiled_output[unresolved_start:l_local]
                    )
                return finish(tiled_output)

    if not use_fused:
        global_rows = torch.arange(
            global_start,
            global_start + l_local,
            dtype=cu_seqlens_q.dtype,
            device=cu_seqlens_q.device,
        )
        sequence_ids = torch.bucketize(
            global_rows, cu_seqlens_q[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_q.shape[0] - 2)
        positions = global_rows - cu_seqlens_q[sequence_ids]
        visible_k = torch.minimum(
            torch.div(positions + 1, int(ratio), rounding_mode="floor"),
            cu_seqlens_compressed[sequence_ids + 1] - cu_seqlens_compressed[sequence_ids],
        ).clamp_min(0)
        valid_q = (global_rows >= cu_seqlens_q[sequence_ids]) & (
            global_rows < cu_seqlens_q[sequence_ids + 1]
        )

        k_rows = torch.arange(
            k_indexer_seq_major.shape[0],
            dtype=cu_seqlens_compressed.dtype,
            device=cu_seqlens_compressed.device,
        )
        k_sequence_ids = torch.bucketize(
            k_rows, cu_seqlens_compressed[1:], out_int32=True, right=True
        ).clamp_max(cu_seqlens_compressed.shape[0] - 2)
        k_positions = k_rows - cu_seqlens_compressed[k_sequence_ids]
        output = torch.full(
            (l_local, topk_width), -1, dtype=torch.int32, device=q_indexer_local.device
        )
        selected_width = min(topk_width, k_indexer_seq_major.shape[0])
        for start in range(0, l_local, 128):
            end = min(start + 128, l_local)
            scores = torch.einsum(
                "rhd,kd->rhk", q_indexer_local[start:end].float(), k_indexer_seq_major.float()
            )
            scores = torch.relu(scores) * weights_indexer_local[start:end].float().unsqueeze(-1)
            scores = scores.sum(dim=1) * float(indexer_softmax_scale)
            valid_k = (
                (k_sequence_ids.unsqueeze(0) == sequence_ids[start:end].unsqueeze(1))
                & (k_positions.unsqueeze(0) < visible_k[start:end].unsqueeze(1))
                & valid_q[start:end].unsqueeze(1)
            )
            scores = scores.masked_fill(~valid_k, float("-inf"))
            values, rows = torch.topk(scores, selected_width, dim=-1)
            local_rows = k_positions[rows].to(torch.int32)
            output[start:end, :selected_width] = torch.where(torch.isfinite(values), local_rows, -1)
        return finish(output)

    topk = run_stock_fused_range(0, l_local)
    return finish(topk)
