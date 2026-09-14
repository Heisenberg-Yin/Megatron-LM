# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference EP1 padding with a CPU-known capacity and GPU-only expert layout."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from transformer_engine.pytorch.permutation import _moe_permute_mask_map

from megatron.lite.primitive.modules.experts import swiglu_with_probs
from megatron.lite.primitive.utils.moe import unpermute


@triton.jit
def _layout(ENDS, OUT, E: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    e = tl.arange(0, B)
    ends = tl.load(ENDS + e, e < E, 2147483647)
    expert = tl.minimum(tl.sum((block * 128 >= ends).to(tl.int32), 0), E - 1)
    tl.store(OUT + block * 128 + tl.arange(0, 128), expert)


def expert_layout(counts: torch.Tensor, capacity: int) -> torch.Tensor:
    """Build aligned expert IDs from GPU counts without a CPU scalar read."""
    assert capacity % 128 == 0
    ends = counts.cumsum(0)
    layout = torch.empty(capacity, device=counts.device, dtype=torch.int32)
    _layout[(capacity // 128,)](
        ends, layout, counts.numel(), triton.next_power_of_2(counts.numel())
    )
    return layout


def permute_capacity(
    hidden: torch.Tensor, scores: torch.Tensor, indices: torch.Tensor, experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute into a safe CPU-known capacity, retaining zero padding."""
    assert not torch.is_grad_enabled()
    n, topk = indices.shape
    routing_map = torch.zeros(n, experts, device=hidden.device, dtype=torch.bool)
    routing_map.scatter_(1, indices, True)
    probs = torch.zeros(n, experts, device=hidden.device, dtype=scores.dtype)
    probs.scatter_(1, indices, scores)
    counts = routing_map.sum(0)
    padded = triton.cdiv(counts, 128) * 128
    padding = padded - counts
    offsets = padding.cumsum(0) - padding
    # At most 127 extra rows per expert. Any excess capacity is zero-filled and
    # assigned to the last expert; no host scalar read or dynamic allocation.
    capacity = triton.cdiv(n * topk + experts * 127, 128) * 128
    output, row_map, permuted_probs = _moe_permute_mask_map.apply(
        hidden, routing_map, capacity, probs, offsets
    )
    return output, padded, permuted_probs, row_map, offsets


def install_fixed_capacity(model: torch.nn.Module) -> None:
    """Install EP1 fixed-capacity routing and grouped FP4 expert forward."""
    for layer in model.layers.values():
        dispatcher, experts = layer.mlp.dispatcher, layer.mlp.experts
        assert dispatcher.ep_size == 1 and dispatcher.moe_permute_fusion

        def dispatch(hidden, scores, indices, _d=dispatcher):
            output, counts, probs, row_map, offsets = permute_capacity(
                hidden, scores, indices, _d.num_experts
            )
            _d._row_id_map, _d._restore_shape = row_map, hidden.shape
            _d._e2e_pad_offsets, _d._local_tpe_list = offsets, None
            return output, counts, probs

        def combine(output, _d=dispatcher):
            result = unpermute(
                output,
                _d._row_id_map,
                restore_shape=_d._restore_shape,
                fused=True,
                pad_offsets=_d._e2e_pad_offsets,
            )
            _d._row_id_map = _d._restore_shape = _d._e2e_pad_offsets = None
            return result

        def forward(x, counts, permuted_probs=None, tokens_per_expert_list=None, _e=experts):
            assert not torch.is_grad_enabled()
            _e.fc1.layout = _e.fc2.layout = expert_layout(counts, x.shape[0])
            first = _e.fc1(x, None)
            probs = permuted_probs.unsqueeze(-1) if permuted_probs is not None else None
            h = swiglu_with_probs(first, probs, _e.swiglu_limit)
            del first
            out = _e.fc2(h, None)
            _e.fc1.layout = _e.fc2.layout = None
            if not _e.defer_reduce:
                dist.all_reduce(out, group=_e.etp_group)
            return out

        dispatcher._dispatch_local = dispatch
        dispatcher._combine_local = combine
        experts.forward = forward
