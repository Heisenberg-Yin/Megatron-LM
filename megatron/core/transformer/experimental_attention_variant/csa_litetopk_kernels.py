# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""H64/FP4 LiteTopK adapter for DSv4 ratio-4 indexers with complete local queries.

This module keeps the production source's real FP4 semantics: E2M1 values, four
UE8M0 scales per 128-wide row, HOT12288 pair gathering, and the legacy contiguous
``MERGE_CAP`` candidate slab.  It intentionally does not claim the paged
``logical_cap=S`` contract used by the H32/FP8 adapter.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from megatron.core.transformer.experimental_attention_variant import dsa_litetopk_kernels

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU reference remains available for static/unit tests.
    triton = None
    tl = None

_LOGGER = logging.getLogger(__name__)

_INDEXER_HEADS = 64
_INDEXER_HEAD_DIM = 128
_PACKED_FP4_DIM = 64
_PACKED_FP4_RECORD_BYTES = 68
_CACHE_BLOCK_SIZE = dsa_litetopk_kernels._CACHE_BLOCK_SIZE
_INDEX_TOPK = 512
_COMPRESS_RATIO = 4
_QUALIFIED_QUERY_LENGTHS = frozenset((4096, 4032))
_HOT_PREFIX = 12288
_MAX_COMPRESSED_SEQUENCE_LENGTH = 1 << 20

_FP4_WORKSPACES: dict[tuple[str, int], dict[str, Tensor | int]] = {}


def _plan_group_tiles() -> int:
    """Return the qualified number of equal-shape Q tiles sharing one K plan."""
    value = int(os.environ.get("MEGATRON_LITETOPK_H64_PLAN_GROUP_TILES", "1"))
    if value not in (1, 2, 4, 8):
        raise ValueError(
            "MEGATRON_LITETOPK_H64_PLAN_GROUP_TILES must be 1, 2, 4, or 8, " f"got {value}"
        )
    return value


def _rolling_tile_seed_enabled() -> bool:
    return os.environ.get("MEGATRON_LITETOPK_H64_ROLLING_SEED", "1") == "1"


def _startup_query_row() -> int:
    """Return the first permitted original-Q row, independent of C4 key coordinates."""
    name = "MEGATRON_LITETOPK_H64_STARTUP_Q"
    raw = os.environ.get(name, "0")
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a nonnegative multiple of 4096, got {raw!r}") from None
    if value < 0 or value % 4096:
        raise ValueError(f"{name} must be a nonnegative multiple of 4096, got {raw!r}")
    return value


def _decline(reason: str) -> None:
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("LiteTopK CSA selector declined: %s", reason)


if triton is not None:

    @triton.jit
    def _select_group_value(group, v0, v1, v2, v3):
        return tl.where(group == 0, v0, tl.where(group == 1, v1, tl.where(group == 2, v2, v3)))

    @triton.jit
    def _ceil_ue8m0_exp(x):
        bits = x.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) & 0xFF
        mantissa = bits & 0x7FFFFF
        exponent += mantissa != 0
        return tl.minimum(tl.maximum(exponent, 1), 254)

    @triton.jit
    def _fp4_e2m1_code(x):
        absolute = tl.minimum(tl.abs(x), 6.0)
        code = (absolute > 0.25).to(tl.uint8)
        code += (absolute > 0.75).to(tl.uint8)
        code += (absolute > 1.25).to(tl.uint8)
        code += (absolute > 1.75).to(tl.uint8)
        code += (absolute > 2.5).to(tl.uint8)
        code += (absolute > 3.5).to(tl.uint8)
        code += (absolute > 5.0).to(tl.uint8)
        sign = ((x < 0) & (code != 0)).to(tl.uint8)
        return code | (sign << 3)

    @triton.jit
    def _quantize_fp4_indexer_kernel(
        source, destination, scales, BLOCK_N: tl.constexpr, GROUP_N: tl.constexpr
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        values = tl.load(source + row * BLOCK_N + offsets).to(tl.float32)
        absolute = tl.abs(values)

        amax0 = tl.max(tl.where(offsets < GROUP_N, absolute, 0.0), axis=0)
        amax1 = tl.max(
            tl.where((GROUP_N <= offsets) & (offsets < 2 * GROUP_N), absolute, 0.0), axis=0
        )
        amax2 = tl.max(
            tl.where((2 * GROUP_N <= offsets) & (offsets < 3 * GROUP_N), absolute, 0.0), axis=0
        )
        amax3 = tl.max(tl.where(3 * GROUP_N <= offsets, absolute, 0.0), axis=0)

        exp0 = _ceil_ue8m0_exp(tl.maximum(amax0 / 6.0, 1.0e-4))
        exp1 = _ceil_ue8m0_exp(tl.maximum(amax1 / 6.0, 1.0e-4))
        exp2 = _ceil_ue8m0_exp(tl.maximum(amax2 / 6.0, 1.0e-4))
        exp3 = _ceil_ue8m0_exp(tl.maximum(amax3 / 6.0, 1.0e-4))
        tl.store(scales + row, exp0 | (exp1 << 8) | (exp2 << 16) | (exp3 << 24))

        pair_offsets = tl.arange(0, BLOCK_N // 2)
        offsets0 = pair_offsets * 2
        offsets1 = offsets0 + 1
        group0 = offsets0 // GROUP_N
        group1 = offsets1 // GROUP_N
        scale_exp0 = _select_group_value(group0, exp0, exp1, exp2, exp3)
        scale_exp1 = _select_group_value(group1, exp0, exp1, exp2, exp3)
        scale0 = (scale_exp0 << 23).to(tl.float32, bitcast=True)
        scale1 = (scale_exp1 << 23).to(tl.float32, bitcast=True)
        code0 = _fp4_e2m1_code(tl.load(source + row * BLOCK_N + offsets0).to(tl.float32) / scale0)
        code1 = _fp4_e2m1_code(tl.load(source + row * BLOCK_N + offsets1).to(tl.float32) / scale1)
        tl.store(destination + row * (BLOCK_N // 2) + pair_offsets, code0 | (code1 << 4))


def _quantize_fp4_indexer_tensor_reference(value: Tensor) -> Tuple[Tensor, Tensor]:
    """CPU/reference implementation of the pinned SGLang MXFP4 quantizer."""
    if value.shape[-1] != _INDEXER_HEAD_DIM:
        raise ValueError(f"FP4 indexer quantization requires D={_INDEXER_HEAD_DIM}")
    rows = value.contiguous().reshape(-1, _INDEXER_HEAD_DIM).float()
    grouped = rows.reshape(-1, 4, 32)
    scale = grouped.abs().amax(dim=-1).div(6.0).clamp_min(1.0e-4).contiguous()
    scale_bits = scale.view(torch.int32)
    exponent = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp_(1, 254)
    power_of_two_scale = (exponent << 23).contiguous().view(torch.float32)

    normalized = grouped / power_of_two_scale.unsqueeze(-1)
    absolute = normalized.abs().clamp_max(6.0)
    code = torch.zeros_like(absolute, dtype=torch.uint8)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        code.add_(absolute > threshold)
    code |= ((normalized < 0) & (code != 0)).to(torch.uint8) << 3
    code = code.reshape(-1, _INDEXER_HEAD_DIM)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous().view(torch.int8)
    packed_scale = (
        exponent[:, 0] | (exponent[:, 1] << 8) | (exponent[:, 2] << 16) | (exponent[:, 3] << 24)
    ).contiguous()
    return packed, packed_scale


def _quantize_fp4_indexer_tensor(value: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize BF16/FP16 rows to packed E2M1 values and UE8M0 scales."""
    if value.shape[-1] != _INDEXER_HEAD_DIM:
        raise ValueError(f"FP4 indexer quantization requires D={_INDEXER_HEAD_DIM}")
    value_2d = value.contiguous().reshape(-1, _INDEXER_HEAD_DIM)
    if not value.is_cuda:
        return _quantize_fp4_indexer_tensor_reference(value_2d)
    if triton is None:
        raise RuntimeError("the production H64 LiteTopK quantizer requires Triton on CUDA")
    packed = torch.empty(
        (value_2d.shape[0], _PACKED_FP4_DIM), dtype=torch.int8, device=value.device
    )
    scales = torch.empty((value_2d.shape[0],), dtype=torch.int32, device=value.device)
    if value_2d.shape[0] > 0:
        _quantize_fp4_indexer_kernel[(value_2d.shape[0],)](
            value_2d, packed, scales, BLOCK_N=_INDEXER_HEAD_DIM, GROUP_N=32
        )
    return packed, scales


def _supports_device(value: Tensor) -> bool:
    return (
        value.is_cuda
        and torch.cuda.get_device_capability(value.device)[0] == 10
        and not torch.cuda.is_current_stream_capturing()
    )


def _stream_id(device: torch.device) -> int:
    """Return a stable execution-stream identity without touching CUDA for CPU tests."""
    return int(torch.cuda.current_stream(device).cuda_stream) if device.type == "cuda" else 0


def _workspace(sequence_length: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # The cache is overwritten in-place for every indexer layer.  Key CUDA storage by
    # stream as well as device so re-entrant/overlapped forwards cannot overwrite a K
    # tensor that another stream's selector is still consuming.
    key = (str(device), _stream_id(device))
    entry = _FP4_WORKSPACES.get(key)
    if entry is None or int(entry["capacity"]) < sequence_length:
        capacity = ((int(sequence_length) + 4095) // 4096) * 4096
        block_capacity = capacity // _CACHE_BLOCK_SIZE
        entry = {
            "capacity": capacity,
            "cache": torch.empty(
                (block_capacity, _CACHE_BLOCK_SIZE, _PACKED_FP4_RECORD_BYTES),
                dtype=torch.uint8,
                device=device,
            ),
            "k": torch.empty((capacity, _PACKED_FP4_DIM), dtype=torch.uint8, device=device),
            "scale": torch.empty((capacity, 4), dtype=torch.uint8, device=device),
            "blocks": torch.arange(block_capacity, dtype=torch.int32, device=device),
        }
        _FP4_WORKSPACES[key] = entry
    active_blocks = (int(sequence_length) + _CACHE_BLOCK_SIZE - 1) // _CACHE_BLOCK_SIZE
    cache = entry["cache"][:active_blocks]
    destination_k = entry["k"][:sequence_length]
    destination_scale = entry["scale"][:sequence_length]
    block_table = entry["blocks"][:active_blocks].view(1, active_blocks)
    return cache, destination_k, destination_scale, block_table


def _pack_fp4_cache(key: Tensor, cache: Tensor) -> None:
    packed, scales = _quantize_fp4_indexer_tensor(key)
    cache_values, cache_scales = dsa_litetopk_kernels._block_major_cache_views(
        cache, _PACKED_FP4_DIM
    )
    dsa_litetopk_kernels._copy_block_major_rows(cache_values, packed.view(torch.uint8))
    dsa_litetopk_kernels._copy_block_major_rows(
        cache_scales, scales.view(torch.uint8).reshape(key.shape[0], 4)
    )


def _build_single_request_c4_bounds(
    *,
    query_length: int,
    sequence_length: int,
    global_start: int,
    ratio: int,
    max_seqlen_q: int,
    cu_seqlens_q: Tensor,
    cu_seqlens_compressed: Tensor,
    device: torch.device,
) -> Optional[tuple[Tensor, Tensor, int]]:
    """Build local-Q causal bounds in one sequence's compressed-C4 coordinates."""
    if ratio != _COMPRESS_RATIO:
        _decline(f"production H64 route requires compression ratio {_COMPRESS_RATIO}")
        return None
    if cu_seqlens_q.numel() != 2 or cu_seqlens_compressed.numel() != 2:
        _decline("production H64 route currently supports one packed request")
        return None
    if sequence_length != int(max_seqlen_q) // ratio:
        _decline("compressed K length must equal floor(max_seqlen_q / 4)")
        return None
    if global_start < 0 or global_start + query_length > int(max_seqlen_q):
        _decline("local CP query interval lies outside the single packed request")
        return None
    common_end = (int(global_start) + 1) // ratio
    if common_end < _HOT_PREFIX or common_end > sequence_length:
        _decline(f"every H64 row must expose the exact HOT{_HOT_PREFIX} prefix")
        return None
    starts = torch.zeros(query_length, dtype=torch.int32, device=device)
    ends = torch.arange(
        int(global_start) + 1,
        int(global_start) + query_length + 1,
        dtype=torch.int32,
        device=device,
    )
    ends = torch.div(ends, ratio, rounding_mode="floor").clamp_max_(sequence_length)
    return starts, ends.contiguous(), common_end


def current_request_key() -> object:
    """Share the ordinary-DSA scheduler request identity with the CSA adapter."""
    return dsa_litetopk_kernels.current_request_key()


def run_cp_indexer_topk_tiles(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    *,
    tile_slices: Sequence[tuple[int, int]],
    cu_seqlens_q: Tensor,
    cu_seqlens_compressed: Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    cp_size: int,
    state_key: object,
    request_key: object = None,
    output: Optional[Tensor] = None,
) -> Optional[tuple[Optional[Tensor], ...]]:
    """Run one or more eligible CP Q tiles while packing the shared K only once.

    ``None`` means that no tile dispatched, so the caller must preserve its original
    one-shot stock-selector fallback.  Once at least one tile dispatches, the returned
    tuple is aligned with ``tile_slices``; an individual ``None`` denotes a tile rejected
    during the side-effect-free HOT/cap preflight and asks the caller to run the official
    selector for just that interval.

    ``output`` optionally supplies the caller-owned final ``[Q, topk]`` tensor.  Admitted
    tiles write directly into their slices, so a full CP shard does not pay one sort and one
    concatenation after every selector launch.  Unresolved slices are left untouched for the
    caller's stock fallback.

    Successful tiles publish local carry at the configured group boundary. The caller also
    publishes from the final mixed output. Neither operation communicates across ranks.

    ``MEGATRON_LITETOPK_H64_STARTUP_Q`` optionally delays selection to a zero-based original-Q
    row aligned to 4096. A tile is admitted only when ``global_start + tile_start`` reaches
    that boundary; earlier tiles keep the stock fallback. The default zero retains existing
    HOT visibility, compressed-K crossover, candidate capacity and carry checks unchanged.
    """
    startup_q = _startup_query_row()
    if state_key is None or cp_size != 1:
        _decline("DSv4 LiteTopK requires a stable state key and CP1")
        return None
    normalized_tiles = tuple((int(start), int(end)) for start, end in tile_slices)
    if not normalized_tiles:
        _decline("the H64 tiled route requires at least one Q interval")
        return None
    if q_indexer.ndim != 3 or k_indexer.ndim != 2 or weights.ndim != 2:
        _decline("expected q=[Q,64,128], k=[S,128], weights=[Q,64]")
        return None
    total_query_length, heads, head_dim = q_indexer.shape
    sequence_length = int(k_indexer.shape[0])
    if (heads, head_dim) != (_INDEXER_HEADS, _INDEXER_HEAD_DIM):
        _decline(f"expected DSv4 indexer H={_INDEXER_HEADS}, D={_INDEXER_HEAD_DIM}")
        return None
    if tuple(k_indexer.shape[1:]) != (_INDEXER_HEAD_DIM,) or tuple(weights.shape) != (
        total_query_length,
        _INDEXER_HEADS,
    ):
        _decline("K or head-gate shape does not match Q")
        return None
    if int(topk_width) != _INDEX_TOPK:
        _decline(f"production DSv4 route requires top-k={_INDEX_TOPK}")
        return None
    if output is not None and (
        tuple(output.shape) != (total_query_length, _INDEX_TOPK)
        or output.dtype != torch.int32
        or output.device != q_indexer.device
        or not output.is_contiguous()
    ):
        _decline("caller-owned H64 output must be contiguous int32 [Q,512] on the Q device")
        return None
    previous_end = -1
    for start, end in normalized_tiles:
        query_length = end - start
        if start < 0 or end > total_query_length or start >= end or start < previous_end:
            _decline("H64 Q tile intervals must be ordered, non-overlapping, and in range")
            return None
        if query_length not in _QUALIFIED_QUERY_LENGTHS:
            _decline(f"unqualified Megatron H64 query length Q={query_length}")
            return None
        previous_end = end
    if sequence_length > _MAX_COMPRESSED_SEQUENCE_LENGTH:
        _decline("compressed sequence exceeds the production 1M-row source limit")
        return None
    if (
        q_indexer.device != k_indexer.device
        or q_indexer.device != weights.device
        or not _supports_device(q_indexer)
    ):
        _decline("Q, K, and weights must share an eager SM100 CUDA device")
        return None

    production = dsa_litetopk_kernels._load_production_litetopk()
    merge_cap = int(getattr(production, "MERGE_CAP", -1))
    # The upstream default stops at 768K original tokens (C4). Its slab ABI
    # accepts a per-call capacity, so cover the remaining CP ranks at 1M too.
    # Respect an explicit memory budget and keep automatic growth bounded to
    # the validated 1M original-context configuration. Allocation below still
    # uses only this rank's admitted causal extent.
    if merge_cap == 196608 and "SGLANG_LITETOPK_MERGE_CAP" not in os.environ:
        merge_cap = max(merge_cap, min(sequence_length, 262144))
    if merge_cap < max(16384, 32 * _INDEX_TOPK):
        _decline(f"legacy H64 candidate slab is too small: MERGE_CAP={merge_cap}")
        return None
    if sequence_length < int(production.production_min_s(True)):
        _decline("compressed sequence is below the qualified FP4 crossover")
        return None
    for start, end in normalized_tiles:
        query_length = end - start
        if not bool(production.supports_fused_query_len(query_length, use_fp4=True)):
            _decline(f"unsupported production FP4 query length Q={query_length}")
            return None

    raw_bounds_by_tile = tuple(
        (
            _build_single_request_c4_bounds(
                query_length=end - start,
                sequence_length=sequence_length,
                global_start=global_start + start,
                ratio=ratio,
                max_seqlen_q=max_seqlen_q,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_compressed=cu_seqlens_compressed,
                device=q_indexer.device,
            )
            if int(global_start) + start >= startup_q
            else None
        )
        for start, end in normalized_tiles
    )
    # The FP4 source still uses its legacy contiguous candidate slab.  Exact-once emits at
    # most one candidate per causal-visible logical key, so max(ends) is a conservative
    # complete-coverage bound for a zero-based single request.  Reject only the intervals
    # whose own visible extent can exceed MERGE_CAP; using full packed S here needlessly
    # disabled safe early CP intervals at a 1M-token original context.
    visible_ends_by_tile = tuple(
        min(sequence_length, (int(global_start) + end) // int(ratio))
        for _start, end in normalized_tiles
    )
    bounds_by_tile = tuple(
        bounds if bounds is not None and visible_end <= merge_cap else None
        for bounds, visible_end in zip(raw_bounds_by_tile, visible_ends_by_tile)
    )
    if not any(bounds is not None for bounds in bounds_by_tile):
        _decline("no H64 tile has both HOT12K visibility and complete legacy MERGE_CAP coverage")
        return None
    selector_cap = max(
        16384,
        32 * _INDEX_TOPK,
        max(
            visible_end
            for bounds, visible_end in zip(bounds_by_tile, visible_ends_by_tile)
            if bounds is not None
        ),
    )
    dsa_litetopk_kernels._ensure_request_state(
        production, q_indexer.device, sequence_length, request_key
    )

    with torch.no_grad():
        output_buffer = (
            output
            if output is not None
            else torch.empty(
                (total_query_length, _INDEX_TOPK), dtype=torch.int32, device=q_indexer.device
            )
        )
        cache, destination_k, destination_scale, block_table = _workspace(
            sequence_length, q_indexer.device
        )
        _pack_fp4_cache(k_indexer, cache)
        k_scales = destination_scale.view(torch.int32).reshape(sequence_length)
        # Megatron CSA uses contiguous CP shards.  Adjacent local rows therefore remain an
        # ordinary recent-time window; the source's CP division is for interleaved shards.
        carry_recent_rows = int(production.carry_recent_rows_for_cp(1))
        results: list[Optional[Tensor]] = [None] * len(normalized_tiles)
        group_tiles = _plan_group_tiles()
        rolling_seed = _rolling_tile_seed_enabled()
        # Group-local carry must not mutate the scheduler-visible layer state until the caller
        # has assembled the complete mixed stock/LiteTopK result.  A private key lets the fused
        # winner-map/vote path warm later groups while preserving atomic fallback semantics.
        temporary_state_key = ("megatron-h64-tile-group", state_key, _stream_id(q_indexer.device))
        temporary_hot_key = (str(q_indexer.device), temporary_state_key)
        temporary_seed_ready = False

        def clear_temporary_seed() -> None:
            production._HOT_CARRY.pop(temporary_hot_key, None)

        # The key is deliberately stable per layer/stream so production carry-vote
        # scratch is reused instead of growing with every transient Q allocation.
        # Remove any prior private publication before starting this atomic dispatch.
        clear_temporary_seed()

        try:
            tile_index = 0
            while tile_index < len(normalized_tiles):
                start, end = normalized_tiles[tile_index]
                bounds = bounds_by_tile[tile_index]
                if bounds is None:
                    tile_index += 1
                    continue
                query_length = end - start

                # A pair permutation and its full-K gather are immutable while the HOT seed
                # stays fixed. Reuse them across adjacent equal-shape tiles. The first tile's
                # causal endpoint is a valid lower bound for every later tile; every selector
                # call still receives that tile's exact per-row ends.
                group_end = tile_index + 1
                while group_end < len(normalized_tiles) and group_end - tile_index < group_tiles:
                    next_start, next_end = normalized_tiles[group_end]
                    if (
                        bounds_by_tile[group_end] is None
                        or next_start != normalized_tiles[group_end - 1][1]
                        or next_end - next_start != query_length
                    ):
                        break
                    group_end += 1

                _starts, _ends, group_common_end = bounds
                plan_hot_key = temporary_state_key if temporary_seed_ready else state_key
                plan = production.prepare_permuted_gather(
                    cache,
                    destination_k,
                    destination_scale,
                    block_table,
                    sequence_length=sequence_length,
                    query_length=query_length,
                    num_reqs=1,
                    common_end=group_common_end,
                    window_start=0,
                    hot_key=plan_hot_key,
                )
                if plan is None:
                    _decline("production H64 HOT12K pair-plan/gather declined")
                    return None
                if temporary_seed_ready:
                    # The plan has enqueued the event dependency and consumed the private
                    # carry. Drop the publication now; bounded vote scratch remains reusable.
                    clear_temporary_seed()
                    temporary_seed_ready = False

                has_later_group = any(
                    later_bounds is not None for later_bounds in bounds_by_tile[group_end:]
                )
                for current_index in range(tile_index, group_end):
                    current_start, current_end = normalized_tiles[current_index]
                    current_bounds = bounds_by_tile[current_index]
                    assert current_bounds is not None
                    starts, ends, _current_common_end = current_bounds
                    visible_end = visible_ends_by_tile[current_index]
                    q_fp4_flat, q_scales_flat = _quantize_fp4_indexer_tensor(
                        q_indexer[current_start:current_end]
                    )
                    q_fp4 = q_fp4_flat.view(query_length, _INDEXER_HEADS, _PACKED_FP4_DIM)
                    q_scales = q_scales_flat.view(query_length, _INDEXER_HEADS)
                    scaled_weights = (
                        weights[current_start:current_end]
                        .float()
                        .mul(float(indexer_softmax_scale))
                        .contiguous()
                    )
                    tile_output = output_buffer[current_start:current_end]
                    publish_group_seed = (
                        rolling_seed and current_index == group_end - 1 and has_later_group
                    )
                    selector_hot_key = (
                        temporary_state_key
                        if temporary_seed_ready or publish_group_seed
                        else state_key
                    )
                    dispatched = production.try_large_exact_once_chunk(
                        q_fp4,
                        destination_k,
                        k_scales,
                        scaled_weights,
                        starts,
                        ends,
                        tile_output,
                        _INDEX_TOPK,
                        permuted_plan=plan,
                        num_reqs=1,
                        ke_min_hint=group_common_end,
                        hot_key=selector_hot_key,
                        ks_common_hint=0,
                        carry_extent_hint=visible_end,
                        carry_recent_rows_hint=carry_recent_rows,
                        q_sf=q_scales,
                        _carry_io=publish_group_seed,
                        # Allocate only this rank's admitted causal extent. MERGE_CAP
                        # remains the global correctness ceiling used by preflight.
                        cap=selector_cap,
                    )
                    if not dispatched:
                        _decline("production H64 FP4 exact-once selector declined")
                        return None
                    # The downstream attention-index builder consumes an unordered set of
                    # logical compressed IDs. Sorting all 512 winners by ID was only
                    # adapter-side canonicalization and added a Q-proportional kernel.
                    results[current_index] = tile_output
                    if publish_group_seed:
                        # Carry I/O can be disabled or strided. Successful dispatch is not
                        # itself proof that a private seed was published.
                        temporary_seed_ready = temporary_hot_key in production._HOT_CARRY
                tile_index = group_end
            return tuple(results) if any(result is not None for result in results) else None
        finally:
            clear_temporary_seed()


def run_cp_indexer_topk(
    q_indexer: Tensor,
    k_indexer: Tensor,
    weights: Tensor,
    *,
    cu_seqlens_q: Tensor,
    cu_seqlens_compressed: Tensor,
    global_start: int,
    ratio: int,
    topk_width: int,
    indexer_softmax_scale: float,
    max_seqlen_q: int,
    cp_size: int,
    state_key: object,
    request_key: object = None,
) -> Optional[Tensor]:
    """Run the pinned production H64 FP4 selector for one eligible CP Q tile."""
    results = run_cp_indexer_topk_tiles(
        q_indexer,
        k_indexer,
        weights,
        tile_slices=((0, int(q_indexer.shape[0])),),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_compressed=cu_seqlens_compressed,
        global_start=global_start,
        ratio=ratio,
        topk_width=topk_width,
        indexer_softmax_scale=indexer_softmax_scale,
        max_seqlen_q=max_seqlen_q,
        cp_size=cp_size,
        state_key=state_key,
        request_key=request_key,
    )
    return None if results is None else results[0]


__all__ = ["current_request_key", "run_cp_indexer_topk", "run_cp_indexer_topk_tiles"]
