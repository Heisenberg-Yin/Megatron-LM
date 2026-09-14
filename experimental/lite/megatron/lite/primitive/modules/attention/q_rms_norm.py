# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference Q RMS normalization without full-size FP32 activation buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(X, Y, R, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    # The original graph rounds the reciprocal RMS before multiplying Q.
    rounded_r = r.to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * D + col, x * rounded_r, col < D)
    tl.store(R + row, r)


def q_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Fuse no-grad contiguous Q rows; preserve the PyTorch autograd path."""
    if (
        not torch.is_grad_enabled()
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and 0 < x.shape[-1] <= 4096
        and x.numel()
    ):
        y = torch.empty_like(x)
        rows = x.numel() // x.shape[-1]
        r = torch.empty(rows, device=x.device, dtype=torch.float32)
        _forward[(rows,)](
            x, y, r, x.shape[-1], eps, triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False
        )
        return y
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
