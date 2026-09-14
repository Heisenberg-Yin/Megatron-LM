# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Complete-query GLM LiteTopK with Native startup188416 and serial Q1776 tiles."""

from __future__ import annotations

from contextlib import contextmanager

import torch

_STARTUP = 188416
_TILE_ROWS = 1776


def wave_tile_slices(sequence_length: int) -> tuple[tuple[int, int], ...]:
    """Return local Q1776 suffix tiles starting exactly at original query 188416."""
    if type(sequence_length) is not int or sequence_length <= 0 or sequence_length % 4:
        raise ValueError('sequence_length must be a positive multiple of four')
    return tuple(
        (first, min(first + _TILE_ROWS, sequence_length))
        for first in range(_STARTUP, sequence_length, _TILE_ROWS)
    )


@contextmanager
def experimental_paged_shapes(module, query_lengths):
    """Temporarily admit exact requested shapes for an explicit eager experiment.

    Intended for the single-threaded benchmark selector path. Concurrent
    selector dispatch in the same process is unsupported; this context is not
    a production configuration API and does not widen device/kernel guards.
    """
    shapes = frozenset(query_lengths)
    if not shapes or any(type(q) is not int or q <= 0 or q % 4 for q in shapes):
        raise ValueError('Experimental paged query lengths must be positive multiples of four')
    name = '_EXPERIMENTAL_FP8_CP_PAGED_QUERY_LENS'
    original = getattr(module, name)
    setattr(module, name, frozenset(original) | shapes)
    try:
        yield
    finally:
        setattr(module, name, original)


@torch.no_grad()
def local_wave_topk(q, k, weights, starts, ends, topk, *, state_key, request_key):
    """Select a Native prefix and local Q1776 LiteTopK suffix without sorting.

    Every TP rank owns the complete Q and K. Each eight-tile group publishes
    its final 1536 query rows as HOT carry. A shorter terminal tile publishes
    min(tail_rows,1536); no later group in this layer consumes that carry.
    The next full-layer invocation explicitly replaces the seed.
    """
    from glm_indexer_canonical import preserve_topk_indices
    from glm_native_indexer import dense_topk

    from megatron.core.transformer.experimental_attention_variant import (
        dsa_litetopk_kernels as lite,
    )

    rows = int(q.shape[0])
    sequence_length = int(k.shape[0])
    if rows != sequence_length:
        raise ValueError('Wave tiles require every prompt row in one complete local segment')
    if topk != 2048 or starts.numel() != rows or ends.numel() != rows:
        raise ValueError('Wave tiles require TopK2048 and complete bounds')
    module = lite._load_production_litetopk()
    if sequence_length < int(module.production_min_s(False)) or rows <= _STARTUP:
        indices, lengths = dense_topk(q, k, weights, starts, ends, topk)
        return indices, lengths, 0
    tiles = wave_tile_slices(rows)
    if state_key is None or request_key is None or module.CP_GLOBAL_CARRY:
        raise ValueError('Wave selection requires stable keys and no CP global carry')

    # Validate the same generic causal-window contract as the ordinary helper.
    # This transfer is outside the per-tile loop and never changes starts/ends.
    bounds = lite._normalize_h32_tiled_bounds(starts, ends, tiles, rows, rows, q.device)
    if bounds is None or any(bound is None for bound in bounds):
        raise ValueError('Wave suffix requires a common HOT12K causal window for every tile')
    bootstrap_extent = int(ends[_STARTUP - 1].item())
    if bootstrap_extent != _STARTUP or bounds[0][3] != _STARTUP + 1:
        raise ValueError('Wave experiment requires the complete ordinary causal prompt')

    lite._ensure_request_state(module, q.device, sequence_length, request_key)
    module.release_pair_swap_workspace(q.device)
    indices = torch.empty((rows, topk), device=q.device, dtype=torch.int32)
    lengths = torch.empty(rows, device=q.device, dtype=torch.int32)
    stock_indices, stock_lengths = dense_topk(
        q[:_STARTUP],
        k[:bootstrap_extent].contiguous(),
        weights[:_STARTUP],
        starts[:_STARTUP],
        ends[:_STARTUP],
        topk,
    )
    indices[:_STARTUP], lengths[:_STARTUP] = stock_indices, stock_lengths
    del stock_indices, stock_lengths
    module.stash_carry(
        state_key,
        indices[_STARTUP - 1536 : _STARTUP],
        bootstrap_extent,
        min_index=0,
        recent_rows_hint=int(module.carry_recent_rows_for_cp(1)),
    )
    with experimental_paged_shapes(module, (last - first for first, last in tiles)):
        results = lite.run_fused_qk_topk_tiles(
            q[:, None],
            k[:, None],
            weights[:, None],
            topk,
            starts,
            ends,
            8192,
            tile_slices=tiles,
            cp_size=1,
            query_partition_size=1,
            publish_local_carry=True,
            canonicalize_indices=preserve_topk_indices,
            experimental_tile_query_length=_TILE_ROWS,
            state_key=state_key,
            request_key=request_key,
        )
    if results is None or len(results) != len(tiles) or any(result is None for result in results):
        raise RuntimeError('Experimental wave selector declined an eligible LiteTopK tile')
    for (first, last), (idx, ln) in zip(tiles, results):
        indices[first:last] = idx.reshape(last - first, topk)
        lengths[first:last] = ln.reshape(-1)
    return indices, lengths, rows - _STARTUP
