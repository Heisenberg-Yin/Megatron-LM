# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference mHC fusion preserving the local model's normalization formula."""

import torch
import triton
import triton.language as tl

from megatron.lite.primitive.modules.attention.hca import HyperConnection

compiled_pre = torch.compile(
    HyperConnection.forward,
    fullgraph=True,
    options={'emulate_precision_casts': True, 'triton.cudagraphs': False},
)


@triton.jit
def _post(X, RES, POST, COMB, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < D
    dtype = RES.dtype.element_ty
    # One CTA owns all four streams for its columns: no in-place read/write race.
    r0 = tl.load(RES + (row * 4 + 0) * D + col, valid, 0).to(tl.float32)
    r1 = tl.load(RES + (row * 4 + 1) * D + col, valid, 0).to(tl.float32)
    r2 = tl.load(RES + (row * 4 + 2) * D + col, valid, 0).to(tl.float32)
    r3 = tl.load(RES + (row * 4 + 3) * D + col, valid, 0).to(tl.float32)
    x = tl.load(X + row * D + col, valid, 0).to(tl.float32)
    for stream in tl.static_range(4):
        c0 = tl.load(COMB + row * 16 + stream * 4 + 0).to(tl.float32)
        c1 = tl.load(COMB + row * 16 + stream * 4 + 1).to(tl.float32)
        c2 = tl.load(COMB + row * 16 + stream * 4 + 2).to(tl.float32)
        c3 = tl.load(COMB + row * 16 + stream * 4 + 3).to(tl.float32)
        mixed = tl.fma(c3, r3, tl.fma(c2, r2, tl.fma(c1, r1, c0 * r0)))
        mixed = mixed.to(dtype).to(tl.float32)
        p = tl.load(POST + row * 4 + stream).to(tl.float32)
        placed = (p * x).to(dtype).to(tl.float32)
        tl.store(RES + (row * 4 + stream) * D + col, placed + mixed, valid)


def post_inplace(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> None:
    """Combine four residual streams after reading each token completely."""
    assert not torch.is_grad_enabled()
    assert residual.shape[-2] == 4 and residual.dtype == torch.bfloat16
    assert all(t.is_contiguous() for t in (x, residual, post, comb))
    n, d = x.numel() // x.shape[-1], x.shape[-1]
    _post[(n, triton.cdiv(d, 512))](x, residual, post, comb, d, 512, enable_fp_fusion=False)


def install_fast_mhc(model: torch.nn.Module) -> None:
    """Install compiled mixing and the fused in-place residual operation."""

    def mix_and_norm(hc, norm, hidden):
        n, b, _, d = hidden.shape
        out = torch.empty((n, b, d), device=hidden.device, dtype=hidden.dtype)
        post = torch.empty((n, b, 4), device=hidden.device, dtype=hidden.dtype)
        comb = torch.empty((n, b, 4, 4), device=hidden.device, dtype=hidden.dtype)
        for begin in range(0, n, model.chunk_size):
            stop = min(n, begin + model.chunk_size)
            x, p, c = compiled_pre(hc, hidden[begin:stop])
            out[begin:stop] = norm(x)
            post[begin:stop], comb[begin:stop] = p, c
        return out, post, comb

    model.mix_and_norm = mix_and_norm
    model.post_inplace = post_inplace
