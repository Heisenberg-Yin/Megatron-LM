# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek FP4 dense indexer for Raw and LiteTopK's startup fallback.

Scores use the original DeepGEMM FP32 path in at most 1 GiB chunks. The caller
installs the exact-tie selector. Output retains the selector's unsorted slots;
CSA scans all slots and compacts any invalid -1 holes before attention.
"""

from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

import torch


@torch.no_grad()
def dense_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    *,
    key_cache: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select unsorted causal IDs using FP4 Q/K and a versioned K cache.

    The scoped caller supplies zero starts and causal-prefix ends. Lengths count
    valid entries; they do not describe a contiguous valid prefix in this map.
    """
    import deep_gemm

    from megatron.core.transformer.experimental_attention_variant import (
        csa_litetopk_kernels,
        dsa_cudnn_kernels,
    )

    assert q.ndim == 3 and k.ndim == 2 and q.shape[-1] == k.shape[-1] == 128
    assert weights.shape == q.shape[:2] and starts.numel() == ends.numel() == q.shape[0]
    dsa_cudnn_kernels._ensure_dsa_namespace()
    sk = k.shape[0]
    width = min(topk, sk)
    if key_cache.get('tensor') is k and key_cache.get('version') == k._version:
        k_data, k_scale = key_cache['quantized']
    else:
        k_data, k_scale = csa_litetopk_kernels._quantize_fp4_indexer_tensor(k)
    key_cache.update(tensor=k, version=k._version, quantized=(k_data, k_scale))
    chunk = max(4, ((2**30 // (4 * sk)) // 4) * 4)
    indices, lengths = [], []
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        q_data, q_scale = csa_litetopk_kernels._quantize_fp4_indexer_tensor(q[start:end])
        q_data = q_data.view(end - start, q.shape[1], 64)
        q_scale = q_scale.view(end - start, q.shape[1])
        w = weights[start:end].float().contiguous()
        bounds_start = starts[start:end].int().contiguous()
        bounds_end = ends[start:end].clamp(0, sk).int().contiguous()
        scores = deep_gemm.fp8_fp4_mqa_logits(
            (q_data, q_scale), (k_data, k_scale), w, bounds_start, bounds_end, clean_logits=True
        )
        selected = dsa_cudnn_kernels._cudnn_dsa.indexer_top_k_wrapper(
            scores, bounds_end, top_k=width, next_n=1, return_val=False
        )["indices"]
        valid = (selected >= 0) & (selected < bounds_end[:, None])
        selected = selected.masked_fill(~valid, -1).int()
        if width < topk:
            selected = torch.nn.functional.pad(selected, (0, topk - width), value=-1)
        indices.append(selected)
        lengths.append(valid.sum(-1).int())
        del scores
    return torch.cat(indices), torch.cat(lengths)


@contextmanager
def matched_quantized_baseline() -> Iterator[None]:
    """Install the FP4 baseline for the benchmark's single-request CP1 layout."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    key_cache = {}

    def csa_topk(
        q_indexer,
        k_indexer,
        weights,
        topk,
        ratio=4,
        indexer_softmax_scale=1.0,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        q_causal_offsets=None,
        **kwargs,
    ):
        assert cu_seqlens_q is not None and cu_seqlens_q.numel() in (2, 3)
        assert cu_seqlens_kv.numel() == cu_seqlens_q.numel()
        assert q_causal_offsets.numel() == cu_seqlens_q.numel() - 1
        positions = torch.arange(q_indexer.shape[0], device=q_indexer.device)
        ends = ((positions + q_causal_offsets[0] + 1) // ratio).clamp(max=k_indexer.shape[0]).int()
        return dense_topk(
            q_indexer,
            k_indexer,
            weights.float() * indexer_softmax_scale,
            torch.zeros_like(ends),
            ends,
            topk,
            key_cache=key_cache,
        )

    with patch.object(cp_utils, "indexer_topk", csa_topk):
        yield
