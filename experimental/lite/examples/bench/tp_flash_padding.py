# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded-query padding adapter for FlashMLA's supported 64-head prefill."""

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils.fused_sparse_attention import (
    _csa_fwd_flash_mla as native_sparse_forward,
)


def padded_sparse_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    indexer_topk: int = 0,
    *,
    reuse_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Pad only a bounded query chunk to FlashMLA's supported head count."""
    assert not torch.is_grad_enabled() or not q.requires_grad
    assert q.shape[1] <= 64 and indexer_topk == 0
    heads = q.shape[1]
    if reuse_query:
        # The caller must relinquish Q. Each chunk is first copied to the
        # padded input; its output can then overwrite only those consumed rows.
        assert not torch.is_grad_enabled() and not q.requires_grad
        assert q.is_contiguous() and q.shape[-1] == d_v and heads < 64
        out = q
    else:
        out = torch.empty((q.shape[0], heads, d_v), dtype=q.dtype, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    sink = None if attn_sink is None else torch.nn.functional.pad(attn_sink, (0, 64 - heads))
    for begin in range(0, q.shape[0], 4096):
        stop = min(begin + 4096, q.shape[0])
        padded = torch.nn.functional.pad(q[begin:stop], (0, 0, 0, 64 - heads))
        result, logsum, _ = native_sparse_forward(
            padded,
            kv,
            topk_idxs[begin:stop],
            softmax_scale,
            d_v=d_v,
            attn_sink=sink,
            topk_length=None if topk_length is None else topk_length[begin:stop],
        )
        out[begin:stop] = result[:, :heads]
        lse[begin:stop] = logsum[:, :heads]
    return out, lse, None
