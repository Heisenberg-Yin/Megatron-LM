# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP-sharded native MXFP4 checkpoint experts with grouped FP8 x FP4 GEMM."""

import torch
import triton
import triton.language as tl

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt
from megatron.lite.primitive.parallel.state import ParallelState


@triton.jit
def _quantize(X, OUT, SCALE, GROUPS, GROUP: tl.constexpr):
    groups = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + offsets, groups[:, None] < GROUPS, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    scale = tl.exp2(tl.ceil(tl.log2(amax / 448.0)))
    tl.store(OUT + offsets, x / scale[:, None], groups[:, None] < GROUPS)
    tl.store(SCALE + groups, scale, groups < GROUPS)


def quantize_activation(x: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows to FP8 with power-of-two scales for grouped GEMM."""
    assert x.ndim == 2 and x.is_contiguous() and x.shape[1] % group == 0
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((x.shape[0], x.shape[1] // group), device=x.device, dtype=torch.float32)
    groups = x.numel() // group
    _quantize[(triton.cdiv(groups, 32),)](x, out, scale, groups, group)
    return out, scale


class GroupedFP4Linear(torch.nn.Module):
    """Grouped native FP4 weights with runtime FP8 activation quantization."""

    def __init__(self, experts: int, out_features: int, in_features: int):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            'weight',
            torch.empty(experts, out_features, in_features // 2, device='cuda', dtype=torch.int8),
        )
        self.register_buffer(
            'scale',
            torch.empty(
                experts, out_features, in_features // 32, device='cuda', dtype=torch.float32
            ),
        )
        self.layout = None

    def finalize(self) -> None:
        """Pack block scales into the DeepGEMM TMA layout."""
        import deep_gemm

        self.scale = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(self.scale)

    def forward(
        self, x: torch.Tensor, splits: list[int] | None, is_first_microbatch: bool | None = None
    ) -> torch.Tensor:
        """Multiply the explicitly assigned expert layout without a reduction."""
        import deep_gemm

        assert not torch.is_grad_enabled() and self.layout is not None
        a, scale = quantize_activation(x)
        scale = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(scale)
        out = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a, scale),
            (self.weight, self.scale),
            out,
            self.layout,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
        )
        return out


class DeepGemmTPExperts(torch.nn.Module):
    """Load ETP8 native FP4 expert shards; install_fixed_capacity supplies forward."""

    def __init__(
        self, config: DeepseekV4Config, ps: ParallelState, checkpoint: str, layer_idx: int
    ):
        super().__init__()
        assert ps.ep_size == 1 and ps.etp_size == 8
        self.num_local_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size // ps.etp_size
        self.swiglu_limit = config.swiglu_limit
        self.etp_group = ps.etp_group
        self.defer_reduce = False
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        self.fc1 = GroupedFP4Linear(e, 2 * i, h)
        self.fc2 = GroupedFP4Linear(e, h, i)
        rank = ps.etp_rank
        with ckpt.SafeTensorReader(checkpoint) as reader:
            for expert in range(e):
                prefix = f'layers.{layer_idx}.ffn.experts.{expert}'
                for part, name in enumerate(('w1', 'w3')):
                    weight = reader.get_tensor(f'{prefix}.{name}.weight')
                    scale = reader.get_tensor(f'{prefix}.{name}.scale')
                    assert weight.dtype == torch.int8
                    assert scale.dtype in (torch.uint8, torch.float8_e8m0fnu)
                    self.fc1.weight[expert, part * i : (part + 1) * i].copy_(
                        weight[rank * i : (rank + 1) * i]
                    )
                    self.fc1.scale[expert, part * i : (part + 1) * i].copy_(
                        ckpt._scale_to_float(scale[rank * i : (rank + 1) * i])
                    )
                weight = reader.get_tensor(f'{prefix}.w2.weight')
                scale = reader.get_tensor(f'{prefix}.w2.scale')
                self.fc2.weight[expert].copy_(weight[:, rank * i // 2 : (rank + 1) * i // 2])
                self.fc2.scale[expert].copy_(
                    ckpt._scale_to_float(scale[:, rank * i // 32 : (rank + 1) * i // 32])
                )
        self.fc1.finalize()
        self.fc2.finalize()
