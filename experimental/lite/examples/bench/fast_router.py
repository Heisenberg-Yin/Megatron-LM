# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Keep the native router scores and tie handling without dense map round trips."""

import torch

from megatron.lite.primitive.utils.moe import topk_routing_with_score_function


def compact_router(router: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Preserve the original router scores and ascending expert-ID combine order."""
    assert not torch.is_grad_enabled() and router.router_replay is None
    logits = router.gate(x).view(-1, router.num_experts)
    probs, indices = topk_routing_with_score_function(
        logits,
        router.topk,
        score_function=router.score_function,
        expert_bias=router.expert_bias.to(logits.dtype),
        scaling_factor=(router.scaling_factor or None),
        fused=False,
        dense_output=True,
    )
    indices, order = torch.sort(indices, dim=-1)
    return probs.gather(1, order).to(logits.dtype), indices


def install_fast_router(model: torch.nn.Module) -> None:
    """Replace learned-router forwards for this inference model only."""
    for layer in model.layers.values():
        router = layer.mlp.gate
        original = router.forward

        def forward(x, _router=router, _original=original):
            if torch.is_grad_enabled() or _router.router_replay is not None:
                return _original(x)
            return compact_router(_router, x)

        router.forward = forward
