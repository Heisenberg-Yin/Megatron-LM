# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Forward-only GLM expert parallelism for replicated TP activations.

TP and EP use the same eight ranks.  Attention returns the complete token
sequence on every rank, so each EP rank can dispatch directly to its owned
experts without first exchanging the already-replicated input.  The routed
expert output and the TP-sharded shared-expert output use one all-reduce.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn as nn
import triton
import triton.language as tl
from transformer_engine.pytorch.permutation import _moe_permute_mask_map

from megatron.lite.model.glm5.lite.model import _router_linear, _swiglu
from megatron.lite.primitive.utils.moe import topk_routing_with_score_function, unpermute


@triton.jit
def _local_routing_maps(
    SCORES,
    INDICES,
    ROUTING,
    PROBS,
    N: tl.constexpr,
    K: tl.constexpr,
    LOCAL_E: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(INDICES + offset, offset < N * K, -1) - START
    score = tl.load(SCORES + offset, offset < N * K, 0)
    local = (offset < N * K) & (expert >= 0) & (expert < LOCAL_E)
    destination = offset // K * LOCAL_E + expert
    tl.store(ROUTING + destination, 1, local)
    tl.store(PROBS + destination, score, local)


@triton.jit
def _padded_expert_layout(ENDS, OUT, E: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    expert_ids = tl.arange(0, B)
    ends = tl.load(ENDS + expert_ids, expert_ids < E, 2147483647)
    expert = tl.sum((block * 128 >= ends).to(tl.int32), 0)
    # Mark unused storage explicitly. The expert backend handles these rows;
    # the installed grouped GEMM cannot be relied on to skip negative layouts.
    expert = tl.where(expert < E, expert, -1)
    tl.store(OUT + block * 128 + tl.arange(0, 128), expert)


class LocalEPDispatch(NamedTuple):
    """Local expert input and metadata needed to restore replicated tokens."""

    hidden: torch.Tensor
    padded_counts: torch.Tensor
    probabilities: torch.Tensor
    row_map: torch.Tensor
    pad_offsets: torch.Tensor
    layout: torch.Tensor
    has_inactive: bool = True


@torch.no_grad()
def compact_glm_router(
    router: nn.Module, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve native FP32 GLM routing while sorting only the selected experts."""
    logits = _router_linear(hidden, router.gate.weight, None, torch.float32)
    logits = logits.view(-1, router.num_experts)
    scores, indices = topk_routing_with_score_function(
        logits,
        router.topk,
        use_pre_softmax=router.use_pre_softmax,
        num_groups=router.num_groups,
        group_topk=router.group_topk,
        score_function="sigmoid",
        expert_bias=router.expert_bias,
        scaling_factor=router.scaling_factor or None,
        fused=False,
        dense_output=True,
    )
    indices, order = indices.sort(dim=-1)
    return scores.gather(1, order), indices


@torch.no_grad()
def local_ep_dispatch(
    hidden: torch.Tensor,
    scores: torch.Tensor,
    indices: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
) -> LocalEPDispatch:
    """Permute owned experts using exact 128-row-aligned capacity, without drops.

    Read the padded count once on the host. If this rank owns no selected rows,
    retain one inactive tile so all ranks follow the same collective path.
    """
    if hidden.ndim != 2 or indices.ndim != 2 or scores.shape != indices.shape:
        raise ValueError("Expected hidden [tokens, hidden] and scores/indices [tokens, topk].")
    if hidden.shape[0] != indices.shape[0] or num_local_experts <= 0 or expert_start < 0:
        raise ValueError("Invalid token count or expert ownership range.")
    tokens, topk = indices.shape
    if tokens == 0:
        raise ValueError("An EP dispatch chunk must contain at least one token.")
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    scores, indices = scores.contiguous(), indices.contiguous()
    routing = torch.zeros(tokens, num_local_experts, device=hidden.device, dtype=torch.bool)
    probabilities = torch.zeros(tokens, num_local_experts, device=hidden.device, dtype=scores.dtype)
    _local_routing_maps[(triton.cdiv(tokens * topk, 256),)](
        scores, indices, routing, probabilities, tokens, topk, num_local_experts, expert_start, 256
    )
    counts = routing.sum(0)
    padded = triton.cdiv(counts, 128) * 128
    padding = padded - counts
    offsets = padding.cumsum(0) - padding
    padded_capacity = int(padded.sum().item())
    capacity = max(padded_capacity, 128)
    # A positive padded total assigns every row to an expert; only the empty
    # local dispatch needs the backend's inactive-row clamp and output mask.
    has_inactive = padded_capacity == 0
    output, row_map, permuted_probs = _moe_permute_mask_map.apply(
        hidden, routing, capacity, probabilities, offsets
    )
    ends = padded.cumsum(0)
    layout = torch.empty(capacity, device=hidden.device, dtype=torch.int32)
    _padded_expert_layout[(capacity // 128,)](
        ends, layout, num_local_experts, triton.next_power_of_2(num_local_experts)
    )
    return LocalEPDispatch(output, padded, permuted_probs, row_map, offsets, layout, has_inactive)


class GlmEPForward(nn.Module):
    """Compute GLM routed and shared experts under TP8/EP8 with replicated input.

    Args:
        router: Native GLM router with checkpoint gate and FP32 correction bias.
        experts: EP-local expert module exposing ``forward_fixed_capacity``.
        shared_expert: Native TP-local shared expert, or None when absent.
        ps: Explicit parallel state; TP and EP must span the same ranks.
        chunk_size: Maximum tokens per dispatch; callers may already pass chunks.
    """

    def __init__(
        self,
        router: nn.Module,
        experts: nn.Module,
        shared_expert: nn.Module | None,
        ps,
        *,
        chunk_size: int = 65536,
    ) -> None:
        super().__init__()
        if ps.tp_size != ps.ep_size or ps.etp_size != 1 or ps.cp_size != 1:
            raise ValueError("GLM replicated EP requires TP=EP, ETP=1, and CP=1.")
        if ps.tp_rank != ps.ep_rank:
            raise ValueError("The TP and EP rank order must match for merged output reduction.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if router.num_experts != experts.num_local_experts * ps.ep_size:
            raise ValueError("Router expert count does not match the EP ownership partition.")
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert
        self.ps = ps
        self.chunk_size = chunk_size
        self.expert_start = ps.ep_rank * experts.num_local_experts

    def _forward_chunk(self, hidden: torch.Tensor) -> torch.Tensor:
        scores, indices = compact_glm_router(self.router, hidden)
        dispatch = local_ep_dispatch(
            hidden, scores, indices, self.expert_start, self.experts.num_local_experts
        )
        expert_output = self.experts.forward_fixed_capacity(
            dispatch.hidden,
            dispatch.padded_counts,
            dispatch.probabilities,
            layout=dispatch.layout,
            has_inactive=dispatch.has_inactive,
        )
        output = unpermute(
            expert_output,
            dispatch.row_map,
            restore_shape=hidden.shape,
            fused=True,
            pad_offsets=dispatch.pad_offsets,
        )
        if self.shared_expert is not None:
            # These modules already own TP-local weights.  Calling the native
            # SharedExpert.forward would apply sequence-parallel collectives.
            shared = self.shared_expert.down(_swiglu(self.shared_expert.gate_up(hidden)))
            output.add_(shared)
        if self.ps.ep_size > 1:
            dist.all_reduce(output, group=self.ps.ep_group)
        return output

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the complete expert output with the input shape on every rank."""
        if torch.is_grad_enabled():
            raise RuntimeError("GlmEPForward is a forward-only benchmark module.")
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        if flat.shape[0] <= self.chunk_size:
            return self._forward_chunk(flat).view(shape)
        output = torch.empty_like(flat)
        for start in range(0, flat.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, flat.shape[0])
            output[start:end].copy_(self._forward_chunk(flat[start:end]))
        return output.view(shape)
