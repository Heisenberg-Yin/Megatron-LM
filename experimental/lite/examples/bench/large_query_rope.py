# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward-only partial RoPE without full-size multiply/stack temporaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope(
    X,
    COS,
    SIN,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    Q_MAJOR: tl.constexpr,
    CS: tl.constexpr,
    CD: tl.constexpr,
    SS: tl.constexpr,
    SD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    dim = offsets % D
    row = offsets // D
    position = row // H if Q_MAJOR else row % L
    rotated = dim >= D - R
    pair = (dim - (D - R)) // 2
    first = row * D + D - R + 2 * pair
    a = tl.load(X + first, valid & rotated, 0).to(tl.float32)
    b = tl.load(X + first + 1, valid & rotated, 0).to(tl.float32)
    c = tl.load(COS + position * CS + pair * CD, valid & rotated, 0).to(tl.float32)
    s = tl.load(SIN + position * SS + pair * SD, valid & rotated, 0).to(tl.float32)
    dtype = X.dtype.element_ty
    # Eager PyTorch rounds each multiply before the add/subtract. Preserve
    # those roundings rather than introducing an FMA or a higher precision formula.
    ac = (a * c).to(dtype).to(tl.float32)
    bs = (b * s).to(dtype).to(tl.float32)
    bc = (b * c).to(dtype).to(tl.float32)
    ass = (a * s).to(dtype).to(tl.float32)
    value = tl.where((dim - (D - R)) % 2 == 0, ac - bs, bc + ass)
    tail = tl.load(X + offsets, valid & ~rotated, 0).to(tl.float32)
    tl.store(OUT + offsets, tl.where(rotated, value, tail), valid)


def fused_partial_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_head_dim: int
) -> torch.Tensor:
    """Apply partial RoPE with the original intermediate BF16/FP16 roundings."""
    if torch.is_grad_enabled():
        raise RuntimeError('Large-query fused RoPE supports inference only')
    assert x.ndim == 4 and x.shape[0] == 1 and x.stride(-1) == 1
    assert x.dtype in (torch.bfloat16, torch.float16)
    _, heads, length, dim = x.shape
    assert 0 < rope_head_dim <= dim and rope_head_dim % 2 == 0
    q_major = x.stride(-3) == dim and x.stride(-2) == heads * dim
    h_major = x.stride(-2) == dim and x.stride(-3) == length * dim
    assert q_major or h_major
    assert cos.shape[-2] == sin.shape[-2] == length
    out = torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
    _rope[(triton.cdiv(x.numel(), 2048),)](
        x,
        cos,
        sin,
        out,
        x.numel(),
        heads,
        length,
        dim,
        rope_head_dim,
        q_major,
        cos.stride(-2),
        cos.stride(-1),
        sin.stride(-2),
        sin.stride(-1),
        2048,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return out
