# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Native GLM block-FP8 experts for the TP8/EP8/ETP1 prefill benchmark.

The checkpoint loading and Blackwell scale conversion are shared with MLite's
BlockFP8Experts. Only activation quantization and already-padded dispatch are
specialized here. Each EP rank owns complete experts; there is no expert tensor
sharding or output collective inside this module.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from deepgemm_tp_experts import quantize_activation

from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
from megatron.lite.primitive.modules.experts import BlockFP8Experts, swiglu_with_probs


class GroupedBlockFP8Linear(torch.nn.Module):
    """Grouped FP8 linear with 128x128 checkpoint scales and explicit layout."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scale", scale)
        self.in_features = weight.shape[2]
        self.out_features = weight.shape[1]
        self.layout: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, *, has_inactive: bool = True) -> torch.Tensor:
        """Compute active expert tiles; inactive layout -1 rows remain zero."""
        import deep_gemm

        if torch.is_grad_enabled() or self.layout is None:
            raise RuntimeError("GroupedBlockFP8Linear needs no-grad and an explicit expert layout.")
        if not isinstance(has_inactive, bool):
            raise TypeError("has_inactive must be a CPU bool supplied by the dispatcher.")
        a, scale = quantize_activation(x.contiguous(), group=128)
        # The installed grouped kernel writes negative-layout tiles despite
        # its generic scheduler's skip check. Give every tile a legal weight
        # index and explicitly clear inactive output before activation/amax.
        # Exact-capacity dispatch avoids this spare computation except for an
        # all-empty local dispatch, which retains one legal 128-row tile.
        # An exact, nonempty dispatch already proves that every row has a
        # nonnegative expert ID. Reuse that CPU-known fact without a GPU read.
        grouped_layout = self.layout.clamp_min(0) if has_inactive else self.layout
        out = torch.empty(x.shape[0], self.out_features, device=x.device, dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a, scale), (self.weight, self.scale), out, grouped_layout
        )
        if has_inactive:
            out.masked_fill_(self.layout[:, None] < 0, 0)
        return out


class GlmTPExperts(BlockFP8Experts):
    """Load this EP rank's native FP8 experts and consume GPU-described dispatch.

    Args:
        config: GLM model configuration, including the global expert count.
        ps: Parallel state with EP expert ownership and ETP size one.
        checkpoint: Local HF checkpoint directory.
        layer_idx: Original checkpoint transformer layer index.
        reader: Optional already-open checkpoint reader owned by the caller.

    The inherited finalizer converts official FP32 block scales to Blackwell's
    UE8M0 scales by requantizing the decoded native FP8 weights once at load.
    This is the same conversion used by the existing MLite GLM FP8 backend.
    """

    def __init__(
        self,
        config: Any,
        ps: Any,
        checkpoint: str,
        layer_idx: int,
        *,
        reader: SafeTensorReader | None = None,
    ) -> None:
        if ps.etp_size != 1:
            raise ValueError("GlmTPExperts requires ETP1 and partitions experts over EP.")
        super().__init__(config, ps)
        self.expert_start = ps.ep_rank * self.num_local_experts
        with nullcontext(reader) if reader is not None else SafeTensorReader(checkpoint) as src:
            for local_idx in range(self.num_local_experts):
                global_idx = self.expert_start + local_idx
                prefix = f"model.layers.{layer_idx}.mlp.experts.{global_idx}"
                tensors = {}
                for short, projection in (
                    ("gate", "gate_proj"),
                    ("up", "up_proj"),
                    ("down", "down_proj"),
                ):
                    name = f"{prefix}.{projection}.weight"
                    tensors[f"{short}_weight"] = src.get_tensor(name)
                    tensors[f"{short}_scale"] = src.get_tensor(f"{name}_scale_inv")
                self.load_checkpoint_expert_(local_idx, **tensors)
        self.finalize_checkpoint_load_()
        self.fc1 = GroupedBlockFP8Linear(self.fc1_weight, self.fc1_scale)
        self.fc2 = GroupedBlockFP8Linear(self.fc2_weight, self.fc2_scale)

    def forward_fixed_capacity(
        self,
        x: torch.Tensor,
        padded_counts: torch.Tensor,
        permuted_probs: torch.Tensor | None = None,
        *,
        layout: torch.Tensor,
        has_inactive: bool = True,
    ) -> torch.Tensor:
        """Return local routed output without host count reads or collectives.

        The dispatcher supplies 128-row aligned counts and layout. Unused
        capacity has layout -1 and zero probability. The caller unpermutes this
        output and then performs the EP reduction once over original tokens.
        Only exact nonempty dispatch may supply ``has_inactive=False``; this
        skips redundant layout conversion and output masking in both GEMMs.
        """
        if torch.is_grad_enabled():
            raise RuntimeError("GlmTPExperts is a forward-only benchmark module.")
        if (
            x.ndim != 2
            or x.shape[1] != self.hidden_size
            or x.shape[0] % 128
            or tuple(layout.shape) != (x.shape[0],)
            or layout.dtype != torch.int32
            or padded_counts.numel() != self.num_local_experts
        ):
            raise ValueError("Expected aligned expert input, local counts, and int32 row layout.")
        self.fc1.layout = self.fc2.layout = layout
        try:
            first = self.fc1(x, has_inactive=has_inactive)
            probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
            hidden = swiglu_with_probs(first, probs, self.swiglu_limit)
            del first
            return self.fc2(hidden, has_inactive=has_inactive)
        finally:
            self.fc1.layout = self.fc2.layout = None
