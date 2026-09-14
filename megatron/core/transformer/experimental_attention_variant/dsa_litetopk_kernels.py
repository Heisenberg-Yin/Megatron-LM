# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Production LiteTopK adapter for the ordinary-DSA indexer on SM100.

This module deliberately does not contain another selector implementation. It adapts MCore's
contiguous BF16 ordinary-DSA tensors to the external LiteTopK ABI used by the
SGLang prefill path. The production implementation owns the HOT12288 pair-swap, exact-once
suffix scan, paged candidate arena (logical capacity ``S``), overflow continuation, and
winner-only carry publication. Keeping those operations in one CUDA implementation prevents
the old Megatron prototype from silently drifting back to an 8K seed or a fixed candidate cap.

The source module is supplied explicitly through ``MEGATRON_LITETOPK_PRODUCTION_PATH``,
which may name ``litetopk.py`` or its checkout root. ABI and device checks are mandatory;
selecting and pinning the external source revision belongs to the caller. Only CP1 is supported.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Hashable, Optional, Sequence, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams

_LOGGER = logging.getLogger(__name__)

_INDEXER_HEADS = 32
_INDEXER_HEAD_DIM = 128
_INDEX_TOPK = 2048
_HOT_PREFIX = 12288
_MAX_SEQUENCE_LENGTH = 1 << 20
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)
_PACKED_FP8_RECORD_BYTES = _INDEXER_HEAD_DIM + 4
_CACHE_BLOCK_SIZE = 64
_K_QUANTIZE_ROWS = int(os.environ.get("MEGATRON_LITETOPK_K_QUANTIZE_ROWS", "65536"))
_H32_DIRECT_TILE_QUERY_LENGTH = 2048
_H32_DIRECT_TAIL_QUERY_LENGTH = 2040

_REQUIRED_SOURCE_FILES = ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh")
_REQUIRED_PRODUCTION_ATTRS = (
    "prepare_permuted_gather",
    "try_large_exact_once_chunk",
    "stash_carry",
    "retire_if_carry_extent_rollback",
    "supports_fused_query_len",
    "production_min_s",
    "carry_recent_rows_for_cp",
    "fp8_large_q_row_slices",
    "_supports_direct_paged_query",
    "_retire_request_state",
    "_HOT_CARRY",
    "HOTONLY",
    "HOTSAMPLE",
    "_dsa_source_id",
    "release_pair_swap_workspace",
    "_CARRY_IO_ENV",
    "_CARRY_EVERY",
)

_PRODUCTION_MODULE: Optional[ModuleType] = None
_FP8_WORKSPACES: dict[str, dict[str, Tensor | int]] = {}
_ACTIVE_REQUEST_KEYS: dict[str, Hashable] = {}
_ACTIVE_REQUEST_OBJECTS: dict[str, object] = {}
_REQUEST_KEY: ContextVar[object] = ContextVar("megatron_litetopk_request_key", default=None)


def _decline(reason: str) -> None:
    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("LiteTopK DSA selector declined: %s", reason)


def _configure_production_environment() -> None:
    """Translate Megatron knobs before the source module snapshots its environment."""
    aliases = {
        "MEGATRON_LITETOPK_PRODUCTION_MIN_S": "SGLANG_LITETOPK_PRODUCTION_MIN_S",
        "MEGATRON_LITETOPK_ROW_TILES": "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES",
        "MEGATRON_LITETOPK_PAGED_POOL_PAGES_PER_ROW": ("SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW"),
        "MEGATRON_LITETOPK_CHECK": "SGLANG_LITETOPK_CHECK",
        "MEGATRON_LITETOPK_SO": "SGLANG_LITETOPK_SO",
        "MEGATRON_LITETOPK_SO_SHA256": "SGLANG_LITETOPK_SO_SHA256",
        "MEGATRON_LITETOPK_BUILD": "SGLANG_LITETOPK_BUILD",
        "MEGATRON_LITETOPK_CARRY_DEBUG": "SGLANG_LITETOPK_CARRY_DEBUG",
    }
    for megatron_name, source_name in aliases.items():
        value = os.environ.get(megatron_name)
        if value is not None:
            os.environ.setdefault(source_name, value)

    # Preserve the external adapter defaults; launch options can override them explicitly.
    defaults = {
        "SGLANG_LITETOPK": "1",
        "SGLANG_LITETOPK_PAGED_CANDIDATES": "1",
        "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "32",
        "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES": "2",
        "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q": "1",
        "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK": "1",
        # A full-model evaluation may enter above the crossover without a preceding scheduler
        # chunk. Identity HOT12K remains exact because the suffix is still scanned exactly once.
        "SGLANG_LITETOPK_COLDSTART_IDENTITY": "1",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def _source_path_from_env() -> Optional[Path]:
    configured = os.environ.get("MEGATRON_LITETOPK_PRODUCTION_PATH")
    if not configured:
        return None
    root = Path(configured).expanduser().resolve()
    candidates = (
        root,
        root / "python/sglang/srt/layers/attention/dsa/litetopk.py",
        root / "sglang/srt/layers/attention/dsa/litetopk.py",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "litetopk.py":
            return candidate
    return root


def production_dependency_issues() -> list[str]:
    """Report missing external sources and DeepGEMM without building CUDA code."""
    issues = []
    source_path = _source_path_from_env()
    if source_path is None:
        issues.append("MEGATRON_LITETOPK_PRODUCTION_PATH pointing to external litetopk.py")
    elif not source_path.is_file():
        issues.append(f"LiteTopK source not found at {source_path}")
    else:
        kernel_dir = source_path.parent / "litetopk_kernels"
        missing = [name for name in _REQUIRED_SOURCE_FILES if not (kernel_dir / name).is_file()]
        if missing:
            issues.append("LiteTopK CUDA sources missing: " + ", ".join(missing))
    try:
        deep_gemm_spec = importlib.util.find_spec("deep_gemm")
    except (ImportError, ModuleNotFoundError, ValueError):
        deep_gemm_spec = None
    if deep_gemm_spec is None:
        issues.append("DeepGEMM with fp8_fp4_mqa_logits and CUDA headers")
    return issues


def _validate_production_module(module: ModuleType) -> None:
    missing = [name for name in _REQUIRED_PRODUCTION_ATTRS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "production LiteTopK module is missing required ABI attributes: " + ", ".join(missing)
        )
    state_attributes = {"_HOT_CARRY", "HOTONLY", "HOTSAMPLE", "_CARRY_IO_ENV", "_CARRY_EVERY"}
    invalid = [
        name
        for name in _REQUIRED_PRODUCTION_ATTRS
        if name not in state_attributes and not callable(getattr(module, name))
    ]
    if invalid:
        raise RuntimeError("LiteTopK ABI functions are not callable: " + ", ".join(invalid))
    if not bool(getattr(module, "ENABLED", False)):
        raise RuntimeError("production LiteTopK module was imported with SGLANG_LITETOPK disabled")
    if int(getattr(module, "HOT_PREFIX", -1)) != _HOT_PREFIX:
        raise RuntimeError(
            f"production LiteTopK must use HOT{_HOT_PREFIX}, got "
            f"HOT{getattr(module, 'HOT_PREFIX', None)}"
        )
    if not bool(getattr(module, "PAGED_CANDIDATES", False)):
        raise RuntimeError("Megatron production LiteTopK requires paged candidates")
    if bool(getattr(module, "CP_GLOBAL_CARRY", False)):
        raise RuntimeError("This complete-query adapter requires SGLANG_LITETOPK_CP_GLOBAL_CARRY=0")
    if not bool(module.HOTONLY) or int(module.HOTSAMPLE) < _HOT_PREFIX:
        raise RuntimeError("Local carry requires the production HOT12K vote path")
    import deep_gemm

    if not hasattr(deep_gemm, "fp8_fp4_mqa_logits"):
        raise RuntimeError("DeepGEMM runtime does not expose fp8_fp4_mqa_logits")


def _load_production_litetopk() -> ModuleType:
    """Load the explicitly supplied external implementation and validate its ABI."""
    global _PRODUCTION_MODULE
    if _PRODUCTION_MODULE is not None:
        return _PRODUCTION_MODULE
    _configure_production_environment()
    issues = production_dependency_issues()
    if issues:
        raise ImportError("LiteTopK requires: " + "; ".join(issues))
    source_path = _source_path_from_env()
    assert source_path is not None
    module_name = "megatron_external_litetopk"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LiteTopK source {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        _validate_production_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _PRODUCTION_MODULE = module
    return module


def _quantize_fp8_per_row(value: Tensor) -> Tuple[Tensor, Tensor]:
    """Quantize the last dimension to E4M3 and return a positive FP32 row scale."""
    value_fp32 = value.float()
    scale = value_fp32.abs().amax(dim=-1).div(_FP8_MAX)
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    quantized = value_fp32.div(scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    return quantized.to(_FP8_DTYPE).contiguous(), scale.contiguous()


def _canonicalize_topk_indices(out_idx: Tensor) -> Tensor:
    """Sort IDs per row without materializing 16K rows of int64 argsort."""
    canonical = torch.empty_like(out_idx)
    for start in range(0, int(out_idx.shape[0]), _H32_DIRECT_TILE_QUERY_LENGTH):
        end = min(start + _H32_DIRECT_TILE_QUERY_LENGTH, int(out_idx.shape[0]))
        sorted_values = out_idx[start:end].sort(dim=-1).values
        canonical[start:end].copy_(sorted_values)
        del sorted_values
    return canonical.contiguous().unsqueeze(0)


def _supports_device(value: Tensor) -> bool:
    return (
        value.is_cuda
        and torch.cuda.get_device_capability(value.device)[0] == 10
        and not torch.cuda.is_current_stream_capturing()
    )


def _block_major_cache_views(cache: Tensor, value_bytes: int) -> tuple[Tensor, Tensor]:
    """View one production cache block as its K region followed by its scale region."""
    expected_record_bytes = int(value_bytes) + 4
    if (
        cache.ndim != 3
        or cache.size(1) != _CACHE_BLOCK_SIZE
        or cache.size(2) != expected_record_bytes
        or not cache.is_contiguous()
    ):
        raise ValueError(
            "LiteTopK cache must be contiguous "
            f"[blocks,{_CACHE_BLOCK_SIZE},{expected_record_bytes}]"
        )
    flat_blocks = cache.view(cache.size(0), _CACHE_BLOCK_SIZE * expected_record_bytes)
    values = flat_blocks[:, : _CACHE_BLOCK_SIZE * value_bytes].view(
        cache.size(0), _CACHE_BLOCK_SIZE, value_bytes
    )
    scales = flat_blocks[:, _CACHE_BLOCK_SIZE * value_bytes :].view(
        cache.size(0), _CACHE_BLOCK_SIZE, 4
    )
    return values, scales


def _copy_block_major_rows(destination: Tensor, source: Tensor, start_row: int = 0) -> None:
    """Copy contiguous rows into an aligned block-major destination view."""
    start_row = int(start_row)
    if start_row < 0 or start_row % _CACHE_BLOCK_SIZE != 0:
        raise ValueError(f"block-major copy start must align to {_CACHE_BLOCK_SIZE} rows")
    if source.ndim != 2 or destination.ndim != 3 or source.size(1) != destination.size(2):
        raise ValueError("block-major source/destination row widths must match")
    first_block = start_row // _CACHE_BLOCK_SIZE
    full_blocks, tail_rows = divmod(int(source.size(0)), _CACHE_BLOCK_SIZE)
    if full_blocks:
        destination[first_block : first_block + full_blocks].copy_(
            source[: full_blocks * _CACHE_BLOCK_SIZE].view(
                full_blocks, _CACHE_BLOCK_SIZE, source.size(1)
            )
        )
    if tail_rows:
        tail_block = first_block + full_blocks
        destination[tail_block].zero_()
        destination[tail_block, :tail_rows].copy_(source[-tail_rows:])


def _workspace(sequence_length: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return reusable block-major cache, gather, scale, and page-table storage."""
    key = str(device)
    entry = _FP8_WORKSPACES.get(key)
    if entry is None or int(entry["capacity"]) < sequence_length:
        capacity = ((int(sequence_length) + 4095) // 4096) * 4096
        block_capacity = capacity // _CACHE_BLOCK_SIZE
        entry = {
            "capacity": capacity,
            "cache": torch.empty(
                (block_capacity, _CACHE_BLOCK_SIZE, _PACKED_FP8_RECORD_BYTES),
                dtype=torch.uint8,
                device=device,
            ),
            "k": torch.empty((capacity, _INDEXER_HEAD_DIM), dtype=_FP8_DTYPE, device=device),
            "scale": torch.empty((capacity, 4), dtype=torch.uint8, device=device),
            "blocks": torch.arange(block_capacity, dtype=torch.int32, device=device),
        }
        _FP8_WORKSPACES[key] = entry
    active_blocks = (int(sequence_length) + _CACHE_BLOCK_SIZE - 1) // _CACHE_BLOCK_SIZE
    cache = entry["cache"][:active_blocks]
    dst_k = entry["k"][:sequence_length]
    dst_scale = entry["scale"][:sequence_length]
    block_table = entry["blocks"][:active_blocks].view(1, active_blocks)
    return cache, dst_k, dst_scale, block_table


def _pack_fp8_cache(k: Tensor, cache_u8: Tensor) -> None:
    """Pack BF16 K into aligned blocks: all FP8 values, then all FP32 scales."""
    rows = int(k.shape[0])
    if _K_QUANTIZE_ROWS <= 0:
        raise ValueError("MEGATRON_LITETOPK_K_QUANTIZE_ROWS must be positive")
    cache_values, cache_scales = _block_major_cache_views(cache_u8, _INDEXER_HEAD_DIM)
    chunk_rows = max(_CACHE_BLOCK_SIZE, (_K_QUANTIZE_ROWS // _CACHE_BLOCK_SIZE) * _CACHE_BLOCK_SIZE)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        k_fp8, k_scale = _quantize_fp8_per_row(k[start:end, 0])
        _copy_block_major_rows(cache_values, k_fp8.view(torch.uint8), start)
        _copy_block_major_rows(
            cache_scales, k_scale.view(torch.uint8).reshape(end - start, 4), start
        )


def _normalize_request_key(request_key: object) -> Optional[Hashable]:
    if request_key is None:
        return None
    if isinstance(request_key, (str, bytes, int, float)):
        return ("value", request_key)
    return ("identity", id(request_key))


@contextmanager
def request_scope(request_key: object):
    """Give every DSA layer/chunk in one external request a stable reset identity."""
    if request_key is None:
        raise ValueError("LiteTopK request_scope requires a non-None request key")
    token = _REQUEST_KEY.set(request_key)
    try:
        yield
    finally:
        _REQUEST_KEY.reset(token)


def current_request_key() -> object:
    """Return the key installed by :func:`request_scope`, if any."""
    return _REQUEST_KEY.get()


def _ensure_request_state(
    module: ModuleType, device: torch.device, sequence_length: int, request_key: object
) -> None:
    """Reset carry/scratch on extent rollback or an explicit request identity change."""
    retired_for_rollback = bool(
        module.retire_if_carry_extent_rollback(device, int(sequence_length))
    )
    normalized = _normalize_request_key(request_key)
    if normalized is None:
        return
    device_key = str(device)
    previous = _ACTIVE_REQUEST_KEYS.get(device_key)
    if previous is not None and previous != normalized and not retired_for_rollback:
        module._retire_request_state(device, release_scratch=True)
    _ACTIVE_REQUEST_KEYS[device_key] = normalized
    _ACTIVE_REQUEST_OBJECTS[device_key] = request_key


def reset_request_state(device: Optional[torch.device | str | int] = None) -> None:
    """Explicitly retire all LiteTopK carry and scratch at a scheduler request boundary."""
    module = _load_production_litetopk()
    resolved = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    module._retire_request_state(resolved, release_scratch=True)
    _ACTIVE_REQUEST_KEYS.pop(str(resolved), None)
    _ACTIVE_REQUEST_OBJECTS.pop(str(resolved), None)


def _uses_paged_logical_cap(module: ModuleType, query_length: int) -> bool:
    """Reject shapes that would silently fall back to the legacy fixed-cap slab."""
    slices = tuple(module.fp8_large_q_row_slices(int(query_length)))
    if len(slices) > 1:
        return True
    direct = getattr(module, "_supports_direct_paged_query", None)
    return bool(direct is not None and direct(int(query_length), use_fp4=False))


def _plan_h32_litetopk_tile_slices(
    local_rows: int, cp_size: int, *, query_partition_size: int | None = None
) -> tuple[tuple[int, int], ...]:
    """Cover a complete CP1 query with local 2048/2040 tiles.

    An unsupported remainder is deliberately left at the beginning. Later rows have larger
    causal visibility and are therefore more likely to satisfy the exact HOT12K precondition.
    ``query_partition_size=1`` explicitly selects complete-Q execution on each rank. These
    tiles execute serially on that rank; they do not split queries across GPUs.
    """
    local_rows = int(local_rows)
    if int(cp_size) != 1 or query_partition_size != 1 or local_rows <= 4096:
        return ()
    if local_rows % _H32_DIRECT_TILE_QUERY_LENGTH == 0:
        return tuple(
            (start, start + _H32_DIRECT_TILE_QUERY_LENGTH)
            for start in range(0, local_rows, _H32_DIRECT_TILE_QUERY_LENGTH)
        )
    if local_rows < _H32_DIRECT_TAIL_QUERY_LENGTH:
        return ()

    tail_start = local_rows - _H32_DIRECT_TAIL_QUERY_LENGTH
    first_tiled_row = tail_start % _H32_DIRECT_TILE_QUERY_LENGTH
    tiles = [
        (start, start + _H32_DIRECT_TILE_QUERY_LENGTH)
        for start in range(first_tiled_row, tail_start, _H32_DIRECT_TILE_QUERY_LENGTH)
    ]
    tiles.append((tail_start, local_rows))
    return tuple(tiles)


def _h32_plan_group_tiles() -> int:
    value = int(os.environ.get("MEGATRON_LITETOPK_H32_PLAN_GROUP_TILES", "8"))
    if value not in (1, 2, 4, 8):
        raise ValueError("MEGATRON_LITETOPK_H32_PLAN_GROUP_TILES must be 1, 2, 4, or 8")
    return value


def _normalize_h32_tiled_bounds(
    starts: Tensor,
    ends: Tensor,
    tile_slices: Sequence[tuple[int, int]],
    num_queries: int,
    sk: int,
    device: torch.device,
) -> Optional[tuple[Optional[tuple[Tensor, Tensor, int, int]], ...]]:
    """Normalize H32 tile bounds with one small host synchronization for the whole group."""
    if (
        starts is None
        or ends is None
        or starts.numel() != num_queries
        or ends.numel() != num_queries
    ):
        _decline("starts/ends must contain one bound per query row")
        return None

    starts_i32 = starts.reshape(-1).to(device=device, dtype=torch.int32).contiguous()
    ends_i32 = (
        ends.reshape(-1).to(device=device, dtype=torch.int32).clamp(min=0, max=sk).contiguous()
    )
    normalized_tiles = tuple((int(start), int(end)) for start, end in tile_slices)
    if not normalized_tiles:
        return None

    summaries = []
    for start, end in normalized_tiles:
        tile_starts = starts_i32[start:end]
        tile_ends = ends_i32[start:end]
        summaries.append(torch.stack((tile_starts.amin(), tile_starts.amax(), tile_ends.amin())))

    # All reductions above stay asynchronous on CUDA; transfer only three integers per tile.
    summary_values = torch.stack(summaries, dim=0).cpu().tolist()
    bounds_by_tile: list[Optional[tuple[Tensor, Tensor, int, int]]] = []
    for (start, end), (window_min, window_max, common_end) in zip(normalized_tiles, summary_values):
        window_start = int(window_min)
        common_end = int(common_end)
        if (
            window_start != int(window_max)
            # The full-local-Q CP route currently publishes the final stock/Lite result through
            # ``observe_reference_topk``, whose carry metadata uses identity key positions.  Do
            # not admit a shifted key window until that common minimum is carried through the
            # mixed-result contract as well.
            or window_start != 0
            or window_start + _HOT_PREFIX > common_end
        ):
            bounds_by_tile.append(None)
            continue
        bounds_by_tile.append(
            (starts_i32[start:end], ends_i32[start:end], window_start, common_end)
        )
    return tuple(bounds_by_tile)


def _recent_valid_carry_rows(
    logical: Tensor, *, recent_rows: int, min_index: int, sequence_length: int
) -> Tensor:
    """Validate only the bounded tail consumed by CP-local carry voting."""
    validation_rows = min(int(logical.shape[0]), int(recent_rows))
    logical = logical[-validation_rows:].to(dtype=torch.int32)
    valid_rows = ((logical >= min_index) & (logical < sequence_length)).all(dim=-1)
    return logical[valid_rows][-recent_rows:]


def observe_reference_topk(
    indices: Tensor,
    *,
    state_key: object,
    request_key: object = None,
    sequence_length: int,
    cp_size: int = 1,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_partition_mode: Optional[str] = None,
    next_sequence_length: Optional[int] = None,
    min_index: int = 0,
    use_fp4: bool = False,
) -> None:
    """Publish HOT12K carry asynchronously from an official fallback result."""
    del cp_group, cp_partition_mode
    if state_key is None or request_key is None or not indices.is_cuda or cp_size != 1:
        return
    module = _load_production_litetopk()
    _ensure_request_state(module, indices.device, sequence_length, request_key)
    min_s = int(module.production_min_s(bool(use_fp4)))
    if sequence_length < min_s and (
        next_sequence_length is None or int(next_sequence_length) < min_s
    ):
        return
    if indices.ndim == 3:
        if indices.size(0) != 1:
            return
        logical = indices[0]
    elif indices.ndim == 2:
        logical = indices
    else:
        return
    if logical.numel() == 0:
        return
    recent_rows = int(module.carry_recent_rows_for_cp(1))
    # Carry consumes only the recent query window. Bound validation to that tail instead of creating
    # gigabyte-scale temporary masks over the complete mixed LiteTopK/stock result. A padded tail
    # with no valid rows simply declines carry publication; it never contaminates the next request.
    logical = _recent_valid_carry_rows(
        logical, recent_rows=recent_rows, min_index=min_index, sequence_length=sequence_length
    )
    if logical.numel() == 0:
        return
    module.stash_carry(
        state_key,
        logical,
        int(sequence_length),
        min_index=int(min_index),
        next_sequence_length=next_sequence_length,
        recent_rows_hint=recent_rows,
    )


def run_fused_qk_topk_tiles(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk: int,
    starts: Tensor,
    ends: Tensor,
    block_size: int,
    *,
    tile_slices: Sequence[tuple[int, int]],
    use_relu: bool = True,
    use_local_indexer_varlen: bool = False,
    single_packed_thd_sequence: bool = False,
    local_packed_cp_rank: int = 0,
    local_packed_cp_query_start: int = 0,
    local_packed_cp_query_len: Optional[int] = None,
    packed_seq_params: Optional["PackedSeqParams"] = None,
    cp_size: int = 1,
    query_partition_size: int | None = None,
    publish_local_carry: bool = False,
    canonicalize_indices: Callable[[Tensor], Tensor] | None = None,
    experimental_tile_query_length: int | None = None,
    state_key: object = None,
    request_key: object = None,
) -> Optional[tuple[Optional[Tuple[Tensor, Tensor]], ...]]:
    """Run qualified direct-paged H32 tiles while packing the shared K exactly once.

    A returned tuple is aligned with ``tile_slices``. ``None`` in one slot means that interval
    failed the side-effect-free HOT/window preflight and must use the ordinary selector. A top-level
    ``None`` means the complete LiteTopK group declined at runtime and the caller must execute one
    full-Q stock selector. By default carry I/O is disabled for every tile; the caller publishes
    one carry only after all LiteTopK and stock intervals have produced the final result.
    Explicit CP1 callers may set publish_local_carry after bootstrapping a same-layer HOT from
    preceding stock rows. Only each group's last tile then publishes, with its actual causal
    extent and no CP communication. ``query_partition_size=1`` means every rank processes the
    complete query; local serial tiles do not introduce cross-rank query/sequence parallelism.
    An optional ``canonicalize_indices`` must return contiguous signed-int32 ``[1, Q, 2048]``
    indices preserving each row's membership. The default uses the existing ascending torch
    sort. A caller may provide an original-order formatter after validating that its attention
    consumer accepts that order; this callback does not change lengths or selection semantics.
    ``experimental_tile_query_length`` is an explicit unqualified CP1/full-Q experiment.
    It accepts that primary tile length and a final positive four-aligned shorter tail;
    callers must separately admit those shapes in a scoped experimental source adapter.
    The default keeps the existing 2048/2040 production shape gate unchanged.
    """
    del (
        block_size,
        local_packed_cp_rank,
        local_packed_cp_query_start,
        local_packed_cp_query_len,
        packed_seq_params,
    )
    if experimental_tile_query_length is not None:
        if (
            type(experimental_tile_query_length) is not int
            or experimental_tile_query_length <= 0
            or experimental_tile_query_length % 4
            or int(cp_size) != 1
            or query_partition_size != 1
        ):
            raise ValueError(
                "Experimental H32 tiles require a positive four-aligned Q and complete-Q CP1"
            )
    if not use_relu:
        _decline("the production kernel implements ReLU(q @ k) scoring only")
        return None
    if int(cp_size) != 1 or query_partition_size != 1:
        _decline("H32 tiles require a complete local query and CP1")
        return None
    if state_key is None or request_key is None:
        _decline("H32 tiles require stable request and layer keys")
        return None
    if use_local_indexer_varlen:
        _decline("H32 tiled LiteTopK is not yet qualified for packed CP metadata")
        return None
    if q.ndim != 4 or k.ndim != 3 or weights.ndim != 3:
        _decline("expected q=[sq,b,h,d], k=[sk,b,d], weights=[sq,b,h]")
        return None

    sq, batch, heads, head_dim = q.shape
    sk = int(k.size(0))
    if sq != sk:
        _decline("Complete-Q CP1 LiteTopK requires all query and key rows from the same prompt")
        return None
    if sq == 0 or batch != 1 or k.size(1) != 1 or weights.size(1) != 1:
        _decline("production LiteTopK requires one non-empty request")
        return None
    if heads != _INDEXER_HEADS or head_dim != _INDEXER_HEAD_DIM:
        _decline(f"expected GLM indexer shape H={_INDEXER_HEADS}, D={_INDEXER_HEAD_DIM}")
        return None
    if tuple(k.shape[1:]) != (1, _INDEXER_HEAD_DIM) or tuple(weights.shape) != (
        sq,
        1,
        _INDEXER_HEADS,
    ):
        _decline("K or weight shape does not match Q")
        return None
    if q.device != k.device or q.device != weights.device or not _supports_device(q):
        _decline("Q, K, and weights must share an eager SM100 CUDA device")
        return None
    if sk > _MAX_SEQUENCE_LENGTH or int(index_topk) != _INDEX_TOPK:
        _decline(f"production GLM route requires S<=1M and top-k={_INDEX_TOPK}")
        return None

    normalized_tiles = tuple((int(start), int(end)) for start, end in tile_slices)
    if not normalized_tiles:
        _decline("the H32 tiled route requires at least one Q interval")
        return None
    previous_end = normalized_tiles[0][0]
    for tile_index, (start, end) in enumerate(normalized_tiles):
        query_length = end - start
        if experimental_tile_query_length is None:
            supported_tile = query_length in (
                _H32_DIRECT_TILE_QUERY_LENGTH,
                _H32_DIRECT_TAIL_QUERY_LENGTH,
            ) and (
                query_length != _H32_DIRECT_TAIL_QUERY_LENGTH
                or tile_index + 1 == len(normalized_tiles)
            )
        else:
            supported_tile = (
                0 < query_length <= experimental_tile_query_length
                and query_length % 4 == 0
                and (
                    query_length == experimental_tile_query_length
                    or tile_index + 1 == len(normalized_tiles)
                )
            )
        if start < 0 or end > sq or start >= end or start != previous_end or not supported_tile:
            _decline(
                "H32 Q tiles must be contiguous, in range, and satisfy the default "
                "2048/2040 layout or the explicitly requested experimental tile layout"
            )
            return None
        previous_end = end

    module = _load_production_litetopk()
    if sk < int(module.production_min_s(False)):
        _decline("sequence is below the qualified production crossover")
        return None
    for start, end in normalized_tiles:
        query_length = end - start
        if not bool(module.supports_fused_query_len(query_length, use_fp4=False)):
            _decline(f"unsupported production FP8 query length Q={query_length}")
            return None
        if not _uses_paged_logical_cap(module, query_length):
            _decline(f"Q={query_length} is not on the direct-paged logical_cap=S route")
            return None

    bounds_by_tile = _normalize_h32_tiled_bounds(starts, ends, normalized_tiles, sq, sk, q.device)
    if bounds_by_tile is None or not any(bounds is not None for bounds in bounds_by_tile):
        _decline("no H32 tile exposes one common, four-aligned HOT12K window")
        return None
    _ensure_request_state(module, q.device, sk, request_key)
    local_carry_extents = None
    if publish_local_carry:
        if module.CP_GLOBAL_CARRY or not module._CARRY_IO_ENV or module._CARRY_EVERY != 1:
            raise ValueError(
                "Local query carry requires enabled per-group carry and no CP collectives"
            )
        # One bounded metadata transfer; never use full S as a partial-query carry extent.
        local_carry_extents = (
            torch.stack([ends.reshape(-1)[end - 1] for _start, end in normalized_tiles])
            .cpu()
            .tolist()
        )

    with torch.no_grad():
        cache_u8, k_fp8, k_scale_u8, block_table = _workspace(sk, q.device)
        _pack_fp8_cache(k, cache_u8)
        k_scale = k_scale_u8.view(torch.float32).reshape(sk)
        carry_recent_rows = int(module.carry_recent_rows_for_cp(1))
        results: list[Optional[Tuple[Tensor, Tensor]]] = [None] * len(normalized_tiles)
        group_tiles = _h32_plan_group_tiles()
        tile_index = 0
        while tile_index < len(normalized_tiles):
            start, end = normalized_tiles[tile_index]
            bounds = bounds_by_tile[tile_index]
            if bounds is None:
                tile_index += 1
                continue
            query_length = end - start

            # The pair permutation and full-S gathered K are immutable while
            # the seed stays fixed.  Reuse one plan for up to eight adjacent
            # Q=2048 calls; the first tile's causal endpoint is a valid lower
            # bound for every later tile, while each selector still receives
            # its exact per-row ends.
            group_end = tile_index + 1
            window_start = bounds[2]
            group_common_end = bounds[3]
            while group_end < len(normalized_tiles) and group_end - tile_index < group_tiles:
                next_start, next_end = normalized_tiles[group_end]
                next_bounds = bounds_by_tile[group_end]
                if (
                    next_bounds is None
                    or next_start != normalized_tiles[group_end - 1][1]
                    or next_end - next_start != query_length
                    or next_bounds[2] != window_start
                    or next_bounds[3] < group_common_end
                ):
                    break
                group_end += 1

            if publish_local_carry:
                carry = module._HOT_CARRY.get((str(q.device), state_key))
                if (
                    carry is None
                    or len(carry) < 4
                    or int(carry[1]) > group_common_end
                    or int(carry[3]) < window_start
                    or carry[0].numel() < _HOT_PREFIX
                ):
                    raise RuntimeError("Local query carry needs a valid preceding stock/group HOT")
            plan = module.prepare_permuted_gather(
                cache_u8,
                k_fp8,
                k_scale_u8,
                block_table,
                sequence_length=sk,
                query_length=query_length,
                num_reqs=1,
                common_end=group_common_end,
                window_start=window_start,
                hot_key=state_key,
            )
            if plan is None:
                _decline("production HOT12K pair-plan/direct-paged gather declined")
                return None

            for current_index in range(tile_index, group_end):
                current_start, current_end = normalized_tiles[current_index]
                current_bounds = bounds_by_tile[current_index]
                assert current_bounds is not None
                starts_i32, ends_i32, _window_start, _common_end = current_bounds
                q_fp8, q_scale = _quantize_fp8_per_row(q[current_start:current_end, 0])
                scaled_weights = (
                    weights[current_start:current_end, 0].float().mul(q_scale).contiguous()
                )
                out_idx = torch.empty(
                    (query_length, _INDEX_TOPK), dtype=torch.int32, device=q.device
                )
                dispatched = module.try_large_exact_once_chunk(
                    q_fp8,
                    k_fp8,
                    k_scale,
                    scaled_weights,
                    starts_i32,
                    ends_i32,
                    out_idx,
                    _INDEX_TOPK,
                    permuted_plan=plan,
                    num_reqs=1,
                    ke_min_hint=group_common_end,
                    hot_key=state_key,
                    ks_common_hint=window_start,
                    carry_extent_hint=(
                        int(local_carry_extents[current_index]) if publish_local_carry else sk
                    ),
                    carry_recent_rows_hint=carry_recent_rows,
                    _carry_io=publish_local_carry and current_index + 1 == group_end,
                    # ``cap`` is intentionally omitted: Q=2048/2040 uses paged logical_cap=S.
                )
                if not dispatched:
                    _decline("production H32 direct-paged exact-once selector declined")
                    return None

                canonical = (
                    _canonicalize_topk_indices(out_idx)
                    if canonicalize_indices is None
                    else canonicalize_indices(out_idx)
                )
                lengths = torch.full(
                    (1, query_length), _INDEX_TOPK, dtype=torch.int32, device=q.device
                )
                results[current_index] = (canonical, lengths)
            tile_index = group_end
        return tuple(results)


__all__ = [
    "current_request_key",
    "observe_reference_topk",
    "production_dependency_issues",
    "request_scope",
    "reset_request_state",
    "run_fused_qk_topk_tiles",
]
