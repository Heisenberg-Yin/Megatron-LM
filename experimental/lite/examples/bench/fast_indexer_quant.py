# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Batch FP4 indexer rows per CTA with the exact native quantization rule."""

import torch
import triton
import triton.language as tl

from megatron.core.transformer.experimental_attention_variant.csa_litetopk_kernels import (
    _ceil_ue8m0_exp,
    _fp4_e2m1_code,
)


@triton.jit
def _quantize(X, OUT, SCALE, ROWS, B: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    cols = tl.arange(0, 128)
    x = tl.load(X + rows[:, None] * 128 + cols[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
    grouped = tl.reshape(x, (B, 4, 32))
    amax = tl.max(tl.abs(grouped), 2)
    exponent = _ceil_ue8m0_exp(tl.maximum(amax / 6.0, 1e-4))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    codes = _fp4_e2m1_code(grouped / scale[:, :, None])
    lo, hi = tl.split(tl.reshape(codes, (B, 64, 2)))
    packed = lo | (hi << 4)
    tl.store(OUT + rows[:, None] * 64 + tl.arange(0, 64)[None, :], packed, rows[:, None] < ROWS)
    shifts = tl.arange(0, 4) * 8
    packed_scale = tl.sum(exponent << shifts[None, :], 1)
    tl.store(SCALE + rows, packed_scale, rows < ROWS)


def quantize_fp4(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the original E2M1/UE8M0 rule with multiple rows per CTA."""
    assert value.is_cuda and value.shape[-1] == 128
    x = value.contiguous().reshape(-1, 128)
    packed = torch.empty((x.shape[0], 64), device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[0], device=x.device, dtype=torch.int32)
    if x.shape[0]:
        _quantize[(triton.cdiv(x.shape[0], 32),)](x, packed, scales, x.shape[0], 32)
    return packed, scales


def install_fast_indexer_quant() -> None:
    """Install the shared Raw/LiteTopK FP4 indexer quantizer."""
    from megatron.core.transformer.experimental_attention_variant import csa_litetopk_kernels

    csa_litetopk_kernels._quantize_fp4_indexer_tensor = quantize_fp4
