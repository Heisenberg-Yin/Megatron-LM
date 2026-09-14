# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MoE expert compute: SwiGLU fusions, _AllReduceETP, and Experts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

import torch  # pyright: ignore[reportMissingImports]
import torch.distributed as dist  # pyright: ignore[reportMissingImports]
import torch.nn as nn  # pyright: ignore[reportMissingImports]
import transformer_engine.pytorch as te  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.kernels.swiglu import (
    bias_swiglu_impl,
    clamped_swiglu,
    clamped_weighted_swiglu,
    weighted_bias_swiglu_impl,
)
from megatron.lite.primitive.modules.lora import (
    LoraConfig,
    SharedGroupedLinearLoRA,
    normalize_lora_config,
)
from megatron.lite.primitive.parallel import ParallelState
from megatron.lite.primitive.recompute import CheckpointWithoutOutput
from megatron.lite.primitive.utils import ensure_divisible

__all__ = ["BlockFP8Experts", "Experts", "_AllReduceETP"]


@contextmanager
def _expert_nvtx_range(name: str):
    if os.environ.get("MEGATRON_LITE_EP_EXPERT_NVTX") != "1" or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def swiglu_with_probs(
    y: torch.Tensor, probs: torch.Tensor | None, swiglu_limit: float = 0.0
) -> torch.Tensor:
    """SwiGLU with optional expert probability scaling."""
    if swiglu_limit > 0:
        if not torch.is_grad_enabled():
            if probs is not None:
                return clamped_weighted_swiglu(y, probs, swiglu_limit)
            return clamped_swiglu(y, swiglu_limit)
        gate, up = y.chunk(2, dim=-1)
        up = torch.clamp(up.float(), min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate.float(), max=swiglu_limit)
        out = torch.nn.functional.silu(gate) * up
        if probs is not None:
            out = out * probs
        return out.to(dtype=y.dtype)
    if probs is not None:
        return weighted_bias_swiglu_impl(y, bias=None, weights=probs)
    return bias_swiglu_impl(y, bias=None)


class _AllReduceETP(torch.autograd.Function):
    """AllReduce with proper autograd: grad(AllReduce) = AllReduce."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class BlockFP8Experts(nn.Module):
    """Checkpoint block-FP8 weight storage for forward-only expert subclasses.

    This module is intentionally separate from :class:`Experts`.  It owns one
    stacked FP8 weight and one compact FP32 128x128 scale tensor per grouped
    linear, so constructing it never allocates the BF16 routed-expert weights.
    The GLM benchmark loads each local expert directly into a slice of these
    buffers and calls :meth:`finalize_checkpoint_load_` once loading completes.
    Its subclass supplies dispatch and grouped GEMM computation.

    DeepGEMM's SM100 kernels consume power-of-two (UE8M0) scales.  Official
    GLM-5 checkpoints contain ordinary FP32 block scales, so finalization
    requantizes one expert at a time.  This bounds temporary memory to one
    expert instead of materializing a second stacked copy of the layer.
    """

    block_shape = (128, 128)

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        if fp8:
            raise ValueError("BlockFP8Experts is independent of TE FP8 training.")
        if moe_act_recompute:
            raise ValueError("BlockFP8Experts is forward-only and does not support recompute.")
        if ps.etp_size != 1:
            raise ValueError("BlockFP8Experts currently requires expert tensor parallel size 1.")
        lora = normalize_lora_config(lora_config)
        if lora.enabled:
            raise ValueError("BlockFP8Experts does not support LoRA.")
        if not torch.cuda.is_available():
            raise RuntimeError("BlockFP8Experts requires a CUDA device and DeepGEMM.")

        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)
        block_m, block_k = self.block_shape
        if self.hidden_size % block_k or self.intermediate_size % block_m:
            raise ValueError(
                "BlockFP8Experts requires hidden and intermediate sizes divisible by 128, "
                f"got hidden={self.hidden_size}, intermediate={self.intermediate_size}."
            )

        # Allocate on the already-selected rank-local GPU. The _apply override
        # lets callers move this module without expanding FP8/FP32 buffers
        # when they convert surrounding model parameters to BF16.
        device = torch.device("cuda", torch.cuda.current_device())
        fc1_out = self.intermediate_size * 2
        self.register_buffer(
            "fc1_weight",
            torch.empty(
                self.num_local_experts,
                fc1_out,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )
        self.register_buffer(
            "fc1_scale",
            torch.empty(
                self.num_local_experts,
                fc1_out // block_m,
                self.hidden_size // block_k,
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "fc2_weight",
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
        )
        self.register_buffer(
            "fc2_scale",
            torch.empty(
                self.num_local_experts,
                self.hidden_size // block_m,
                self.intermediate_size // block_k,
                dtype=torch.float32,
                device=device,
            ),
        )
        self._loaded_experts: set[int] = set()
        self._checkpoint_finalized = False
        self._use_ue8m0 = torch.cuda.get_device_capability(device)[0] >= 10

    def _apply(self, fn, recurse: bool = True):
        """Move buffers without allowing ``Module.to(dtype=...)`` to expand FP8."""
        del recurse  # BlockFP8Experts has no child modules or parameters.
        probe = torch.empty(0, dtype=torch.uint8, device=self.fc1_weight.device)
        target_device = fn(probe).device
        for name, value in self._buffers.items():
            if value is not None and value.device != target_device:
                self._buffers[name] = value.to(device=target_device)
        return self

    @staticmethod
    def _validate_checkpoint_tensor(
        name: str,
        tensor: torch.Tensor,
        expected_shape: tuple[int, ...],
        expected_dtype: torch.dtype,
    ) -> None:
        if tuple(tensor.shape) != expected_shape or tensor.dtype != expected_dtype:
            raise ValueError(
                f"Invalid block-FP8 checkpoint tensor {name}: expected "
                f"shape={expected_shape}, dtype={expected_dtype}, got "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}."
            )

    @torch.no_grad()
    def load_checkpoint_expert_(
        self,
        local_idx: int,
        *,
        gate_weight: torch.Tensor,
        gate_scale: torch.Tensor,
        up_weight: torch.Tensor,
        up_scale: torch.Tensor,
        down_weight: torch.Tensor,
        down_scale: torch.Tensor,
    ) -> None:
        """Direct-copy one checkpoint expert without a BF16 intermediate."""
        if not 0 <= local_idx < self.num_local_experts:
            raise IndexError(f"local expert index {local_idx} is out of range")
        block_m, block_k = self.block_shape
        i, h = self.intermediate_size, self.hidden_size
        self._validate_checkpoint_tensor("gate_weight", gate_weight, (i, h), torch.float8_e4m3fn)
        self._validate_checkpoint_tensor("up_weight", up_weight, (i, h), torch.float8_e4m3fn)
        self._validate_checkpoint_tensor("down_weight", down_weight, (h, i), torch.float8_e4m3fn)
        self._validate_checkpoint_tensor(
            "gate_scale", gate_scale, (i // block_m, h // block_k), torch.float32
        )
        self._validate_checkpoint_tensor(
            "up_scale", up_scale, (i // block_m, h // block_k), torch.float32
        )
        self._validate_checkpoint_tensor(
            "down_scale", down_scale, (h // block_m, i // block_k), torch.float32
        )

        # copy_ performs the CPU-to-GPU transfer directly into the final slice;
        # concatenating gate/up here would introduce a second full expert copy.
        self.fc1_weight[local_idx, :i].copy_(gate_weight)
        self.fc1_weight[local_idx, i:].copy_(up_weight)
        self.fc1_scale[local_idx, : i // block_m].copy_(gate_scale)
        self.fc1_scale[local_idx, i // block_m :].copy_(up_scale)
        self.fc2_weight[local_idx].copy_(down_weight)
        self.fc2_scale[local_idx].copy_(down_scale)
        self._loaded_experts.add(local_idx)
        self._checkpoint_finalized = False

    @torch.no_grad()
    def finalize_checkpoint_load_(self) -> None:
        """Prepare checkpoint scales for the current DeepGEMM architecture."""
        expected = set(range(self.num_local_experts))
        if self._loaded_experts != expected:
            missing = sorted(expected - self._loaded_experts)
            raise RuntimeError(f"Cannot finalize BlockFP8Experts; missing experts {missing}.")
        if self._checkpoint_finalized:
            return

        if self._use_ue8m0:
            import deep_gemm  # Lazy: the default BF16 Experts path has no dependency.

            block_m, block_k = self.block_shape
            for expert_idx in range(self.num_local_experts):
                for weight, scale in (
                    (self.fc1_weight[expert_idx], self.fc1_scale[expert_idx]),
                    (self.fc2_weight[expert_idx], self.fc2_scale[expert_idx]),
                ):
                    expanded_scale = scale.repeat_interleave(block_m, dim=0).repeat_interleave(
                        block_k, dim=1
                    )
                    dequantized = weight.float() * expanded_scale
                    requantized, ue8m0_scale = deep_gemm.per_block_cast_to_fp8(
                        dequantized, use_ue8m0=True
                    )
                    weight.copy_(requantized)
                    scale.copy_(ue8m0_scale)
                    del expanded_scale, dequantized, requantized, ue8m0_scale

        self._checkpoint_finalized = True


class Experts(nn.Module):

    def __init__(
        self,
        config: Any,
        ps: ParallelState,
        *,
        fp8: bool = False,
        moe_act_recompute: bool = False,
        lora_config: LoraConfig | dict | None = None,
    ):
        super().__init__()
        self.num_local_experts = ensure_divisible(config.num_experts, ps.ep_size)
        self.fp8 = fp8
        self.moe_act_recompute = moe_act_recompute
        self.etp_group = ps.etp_group if ps.etp_size > 1 else None
        self.swiglu_limit = float(getattr(config, "swiglu_limit", 0.0) or 0.0)

        self.fc1 = te.GroupedLinear(
            self.num_local_experts,
            config.hidden_size,
            config.moe_intermediate_size * 2 // ps.etp_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        self.fc2 = te.GroupedLinear(
            self.num_local_experts,
            config.moe_intermediate_size // ps.etp_size,
            config.hidden_size,
            bias=False,
            params_dtype=torch.bfloat16,
        )
        lora = normalize_lora_config(lora_config)
        self.fc1_lora: SharedGroupedLinearLoRA | None = None
        self.fc2_lora: SharedGroupedLinearLoRA | None = None
        if lora.enabled and lora.targets_module("linear_fc1"):
            self.fc1_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.hidden_size,
                config.moe_intermediate_size * 2 // ps.etp_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if lora.enabled and lora.targets_module("linear_fc2"):
            self.fc2_lora = SharedGroupedLinearLoRA(
                self.num_local_experts,
                config.moe_intermediate_size // ps.etp_size,
                config.hidden_size,
                lora.rank,
                alpha=lora.alpha,
                dropout=lora.dropout,
            )
        if ps.tp_size > 1 and ps.ep_size == 1 and ps.etp_size == 1:
            tp_group = ps.tp_group
            for module in (self.fc1, self.fc2, self.fc1_lora, self.fc2_lora):
                if module is None:
                    continue
                for param in module.parameters():

                    def _ar(grad, g=tp_group):
                        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=g)
                        return grad

                    param.register_hook(_ar)

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        tokens_per_expert_list: list[int] | None = None,
    ) -> torch.Tensor:
        m_splits = (
            tokens_per_expert.tolist()
            if tokens_per_expert_list is None
            else list(tokens_per_expert_list)
        )
        pad_mask = None
        if self.fp8:
            x, permuted_probs, m_splits, pad_mask = self._fp8_pad(x, permuted_probs, m_splits)

        etp_real_len = x.shape[0]
        if self.etp_group is not None:
            max_len = torch.tensor([etp_real_len], device=x.device, dtype=torch.int64)
            dist.all_reduce(max_len, op=dist.ReduceOp.MAX, group=self.etp_group)
            max_len = int(max_len.item())
            if etp_real_len < max_len:
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            max_len - etp_real_len, x.shape[1], dtype=x.dtype, device=x.device
                        ),
                    ],
                    dim=0,
                )
                if permuted_probs is not None:
                    permuted_probs = torch.cat(
                        [
                            permuted_probs,
                            torch.zeros(
                                max_len - etp_real_len, dtype=permuted_probs.dtype, device=x.device
                            ),
                        ],
                        dim=0,
                    )
                m_splits = list(m_splits)
                m_splits[-1] += max_len - etp_real_len

        probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
        with _expert_nvtx_range("ep_experts.forward"):
            if self.moe_act_recompute and probs is not None:
                act_ckpt = CheckpointWithoutOutput(preserve_rng_state=True)
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = act_ckpt.checkpoint(swiglu_with_probs, fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)
                act_ckpt.discard_output_and_register_recompute(out)
            else:
                fc1_out = self.fc1(x, m_splits)
                if self.fc1_lora is not None:
                    fc1_out = fc1_out + self.fc1_lora(x, m_splits)
                h = swiglu_with_probs(fc1_out, probs, self.swiglu_limit)
                out = self.fc2(h, m_splits)
                if self.fc2_lora is not None:
                    out = out + self.fc2_lora(h, m_splits)

        if self.etp_group is not None:
            out = _AllReduceETP.apply(out, self.etp_group)
            out = out[:etp_real_len]

        if pad_mask is not None:
            out = out[pad_mask]
        return out

    @staticmethod
    def _fp8_pad(x, permuted_probs, m_splits):
        padded = [(s + 15) // 16 * 16 for s in m_splits]
        if padded == m_splits:
            return x, permuted_probs, m_splits, None
        device, dtype = x.device, x.dtype
        total_padded = sum(padded)
        x_pad = torch.zeros(total_padded, x.size(1), device=device, dtype=dtype)
        mask = torch.zeros(total_padded, dtype=torch.bool, device=device)
        probs_pad = None
        if permuted_probs is not None:
            probs_pad = torch.zeros(total_padded, device=device, dtype=permuted_probs.dtype)
        src_off, dst_off = 0, 0
        for real, pad in zip(m_splits, padded, strict=True):
            x_pad[dst_off : dst_off + real] = x[src_off : src_off + real]
            mask[dst_off : dst_off + real] = True
            if probs_pad is not None:
                probs_pad[dst_off : dst_off + real] = permuted_probs[src_off : src_off + real]
            src_off += real
            dst_off += pad
        return x_pad, probs_pad, padded, mask
