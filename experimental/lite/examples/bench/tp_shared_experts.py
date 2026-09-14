# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shard shared experts over ETP and reduce shared+routed output together."""

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.lite.primitive.modules.experts import swiglu_with_probs


def shared_shard(
    shared: torch.nn.Module, rank: int, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice the shared intermediate dimension using the ETP rank."""
    gate, up = shared.gate_up.weight.chunk(2, dim=0)
    assert gate.shape[0] % size == 0
    return (
        torch.cat((gate.chunk(size, 0)[rank], up.chunk(size, 0)[rank]), 0).contiguous(),
        shared.down.weight.chunk(size, 1)[rank].contiguous(),
    )


def shared_forward(
    x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, limit: float
) -> torch.Tensor:
    """Compute the local shared-expert contribution without reducing it."""
    return F.linear(swiglu_with_probs(F.linear(x, gate_up), None, limit), down)


@torch.no_grad()
def install_tp_shared(model: torch.nn.Module) -> None:
    """Reduce routed and shared contributions together after local combination."""
    assert model.ps.ep_size == 1 and model.ps.etp_size == 8
    for layer in model.layers.values():
        mlp = layer.mlp
        shared, experts, dispatcher = mlp.shared_experts, mlp.experts, mlp.dispatcher
        if shared is None:
            continue
        gate_up, down = shared_shard(shared, model.ps.etp_rank, model.ps.etp_size)
        shared.register_buffer('_tp_gate_up', gate_up)
        shared.register_buffer('_tp_down', down)
        original_shared, original_combine, original_mlp = (
            shared.forward,
            dispatcher.combine,
            mlp.forward,
        )

        def forward_shared(x, _s=shared, _e=experts, _original=original_shared):
            if not _e.defer_reduce:
                return _original(x)
            assert not torch.is_grad_enabled()
            return shared_forward(x, _s._tp_gate_up, _s._tp_down, _s.swiglu_limit)

        def combine(x, _d=dispatcher, _e=experts, _original=original_combine):
            if not _e.defer_reduce:
                return _original(x)
            return _d._combine_local(x)

        def forward_mlp(x, *, input_ids=None, _e=experts, _original=original_mlp):
            out = _original(x, input_ids=input_ids)
            if _e.defer_reduce:
                dist.all_reduce(out, group=_e.etp_group)
            return out

        shared.forward = forward_shared
        dispatcher.combine = combine
        mlp.forward = forward_mlp
