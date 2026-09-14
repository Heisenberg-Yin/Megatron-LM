# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Experimental inference-only sparse attention for TP-local query heads."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse(
    Q,
    KV,
    IDX,
    SINK,
    LENGTH,
    OUT,
    LSE,
    H: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    NK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    q = tl.load(Q + row * H * D + h[:, None] * D + d[None, :], h[:, None] < H, 0)
    m = (
        tl.load(SINK + h, h < H, 0).to(tl.float32)
        if HAS_SINK
        else tl.full((BH,), -1.0e20, tl.float32)
    )
    norm = tl.full((BH,), 1.0 if HAS_SINK else 0.0, tl.float32)
    acc = tl.zeros((BH, D), tl.float32)
    length = tl.load(LENGTH + row) if HAS_LENGTH else K
    for start in range(tl.cdiv(K, BK)):
        slots = start * BK + tl.arange(0, BK)
        idx = tl.load(IDX + row * K + slots, slots < K, -1).to(tl.int64)
        valid = (idx >= 0) & (idx < NK) & (slots < length)
        k = tl.load(KV + idx[:, None] * D + d[None, :], valid[:, None], 0)
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
        scores = tl.where(valid[None, :], scores, -1.0e20)
        new_m = tl.maximum(m, tl.max(scores, 1))
        alpha = tl.exp(m - new_m)
        prob = tl.where(valid[None, :], tl.exp(scores - new_m[:, None]), 0.0)
        acc = acc * alpha[:, None] + tl.dot(prob.to(k.dtype), k)
        norm = norm * alpha + tl.sum(prob, 1)
        m = new_m
    out = acc / tl.maximum(norm[:, None], 1.0e-20)
    tl.store(OUT + row * H * D + h[:, None] * D + d[None, :], out, h[:, None] < H)
    tl.store(LSE + row * H + h, tl.log(norm) + m, h < H)


def sparse_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    indexer_topk: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Compute short-window attention for TP-local heads in BF16."""
    if torch.is_grad_enabled() and q.requires_grad:
        raise RuntimeError('TP sparse kernel is inference-only')
    assert indexer_topk == 0 and d_v == q.shape[-1] == 512 and q.shape[1] <= 16
    q, kv, indices = q.contiguous(), kv.contiguous(), topk_idxs.contiguous()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    _sparse[(q.shape[0],)](
        q,
        kv,
        indices,
        attn_sink,
        topk_length,
        out,
        lse,
        q.shape[1],
        512,
        indices.shape[-1],
        kv.shape[0],
        softmax_scale,
        attn_sink is not None,
        topk_length is not None,
        16,
        64,
        num_warps=8,
    )
    return out, lse, None
