# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FP8 GLM scorer with bounded SM-wave score blocks and unsorted exact-tie TopK."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def score_chunk_rows(keys: int, *, num_sms: int) -> int:
    """Plan BLOCK_Q4 SM waves within the fixed 2 GiB FP32 score budget.

    If one complete wave does not fit, preserve the four-aligned maximum.
    The supplied SM count is CPU metadata; this function does not query a device.
    """
    if keys <= 0:
        raise ValueError('Key count must be positive.')
    if type(num_sms) is not int or num_sms <= 0:
        raise ValueError('sm-wave requires a positive integer num_sms.')
    maximum = max(4, ((2**31 // (4 * keys)) // 4) * 4)
    wave_rows = 4 * num_sms
    return maximum // wave_rows * wave_rows if maximum >= wave_rows else maximum


@triton.jit
def _canonicalize_kernel(
    SELECTED,
    ENDS,
    OUTPUT,
    LENGTHS,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    SELECTED_STRIDE: tl.constexpr,
    OUTPUT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    selected = tl.load(
        SELECTED + row.to(tl.int64) * SELECTED_STRIDE + columns, columns < INPUT_WIDTH, -1
    )
    end = tl.load(ENDS + row)
    valid = (columns < INPUT_WIDTH) & (selected >= 0) & (selected < end)
    count = tl.sum(valid.to(tl.int32), axis=0)
    destinations = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    # These stores have disjoint destinations: the compacted valid prefix
    # is [0,count), and only the tail [count,OUTPUT_WIDTH) is initialized.
    # Filling the entire row before a scatter would require synchronization.
    tl.store(
        OUTPUT + row.to(tl.int64) * OUTPUT_STRIDE + columns,
        -1,
        (columns >= count) & (columns < OUTPUT_WIDTH),
    )
    tl.store(OUTPUT + row.to(tl.int64) * OUTPUT_STRIDE + destinations, selected, valid)
    tl.store(LENGTHS + row, count)


def canonicalize_into(
    selected: torch.Tensor,
    ends: torch.Tensor,
    output: torch.Tensor,
    lengths: torch.Tensor,
    *,
    keys: int,
) -> None:
    """Stably pack valid IDs and a -1 tail without sorting."""
    rows, width = selected.shape
    if (
        selected.dtype != torch.int32
        or ends.dtype != torch.int32
        or output.dtype != torch.int32
        or lengths.dtype != torch.int32
        or ends.shape != (rows,)
        or lengths.shape != (rows,)
        or output.shape[0] != rows
        or not 0 < width <= output.shape[1] <= 2048
        or not 0 < keys < 2**31
        or any(t.device != selected.device for t in (ends, output, lengths))
        or not all(t.is_cuda for t in (selected, ends, output, lengths))
        or selected.stride(1) != 1
        or output.stride(1) != 1
        or not ends.is_contiguous()
        or not lengths.is_contiguous()
    ):
        raise ValueError('Expected compatible CUDA int32 row-major TopK buffers.')
    if rows:
        _canonicalize_kernel[(rows,)](
            selected,
            ends,
            output,
            lengths,
            width,
            output.shape[1],
            selected.stride(0),
            output.stride(0),
            triton.next_power_of_2(output.shape[1]),
            num_warps=4,
        )


@torch.no_grad()
def dense_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score all GLM queries using fixed 2 GiB SM-wave blocks and exact-tie TopK.

    Callers supply zero starts and causal exclusive ends. The full-prompt
    attention constructs these bounds; LiteTopK preserves them for prefix
    fallback. Ends are clamped to K before scoring and valid-prefix packing.
    This internal path does not add a tensor-content synchronization.
    """
    import deep_gemm

    from megatron.core.transformer.experimental_attention_variant import dsa_litetopk_kernels

    if (
        q.ndim != 3
        or k.ndim != 2
        or q.shape[1:] != (32, 128)
        or k.shape[1] != 128
        or weights.shape != q.shape[:2]
        or starts.shape != (q.shape[0],)
        or ends.shape != (q.shape[0],)
        or not 0 < topk <= 2048
        or k.shape[0] >= 2**31
        or not q.is_cuda
        or any(t.device != q.device for t in (k, weights, starts, ends))
    ):
        raise ValueError('Expected GLM CUDA Q[M,32,128], K[S,128], and compatible metadata.')
    rows, keys = q.shape[0], k.shape[0]
    output = torch.empty((rows, topk), device=q.device, dtype=torch.int32)
    lengths = torch.empty(rows, device=q.device, dtype=torch.int32)
    if not rows:
        return output, lengths
    if not keys:
        output.fill_(-1)
        lengths.zero_()
        return output, lengths
    from glm_exact_tie_topk import exact_tie_topk

    topk_selector = exact_tie_topk
    k_data, k_scale = dsa_litetopk_kernels._quantize_fp8_per_row(k)
    chunk = score_chunk_rows(
        keys, num_sms=torch.cuda.get_device_properties(q.device).multi_processor_count
    )
    width = min(topk, keys)
    for first in range(0, rows, chunk):
        last = min(first + chunk, rows)
        q_data, scale = dsa_litetopk_kernels._quantize_fp8_per_row(q[first:last])
        w = (weights[first:last].float() * scale).contiguous()
        bounds_start = starts[first:last].int().contiguous()
        bounds_end = ends[first:last].clamp(0, keys).int().contiguous()
        scores = deep_gemm.fp8_fp4_mqa_logits(
            (q_data, None), (k_data, k_scale), w, bounds_start, bounds_end, clean_logits=False
        )
        selected = topk_selector(scores, bounds_end, top_k=width, next_n=1, return_val=False)[
            'indices'
        ]
        canonicalize_into(selected, bounds_end, output[first:last], lengths[first:last], keys=keys)
        del selected, scores
    return output, lengths
