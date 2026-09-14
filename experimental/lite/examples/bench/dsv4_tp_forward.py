# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP-local attention and checkpoint loading for the full DeepSeek prompt benchmark.

This benchmark does not enable unsupported training/export paths in the runtime.
The compressor, indexer and hyper-connections remain replicated; the full-model
entry point installs TP-sharded shared experts separately.
"""

import copy

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite import checkpoint as ckpt
from megatron.lite.primitive.modules.attention.csa import CompressedSparseAttention
from megatron.lite.primitive.parallel.state import ParallelState


class TPAttention(CompressedSparseAttention):
    """TP-local heads/output projection with replicated compressed KV and indexer."""

    def __init__(self, config: DeepseekV4Config, *, layer_idx: int, ps: ParallelState):
        local = copy.copy(config)
        assert config.num_attention_heads % ps.tp_size == 0
        assert config.o_groups % ps.tp_size == 0
        local.num_attention_heads //= ps.tp_size
        local.o_groups //= ps.tp_size
        super().__init__(local, layer_idx=layer_idx, ps=ps)

    def _project_context(self, context, cos, sin):
        out = super()._project_context(context, cos, sin)
        if torch.is_grad_enabled():
            raise RuntimeError('Experimental TP attention supports inference only')
        if self.ps.tp_size > 1:
            dist.all_reduce(out, group=self.ps.tp_group)
        return out


def shard_spec(name: str, ps: ParallelState) -> tuple[int, int, int] | None:
    """Return the checkpoint shard dimension, size, and local rank."""
    if name in ('embed_tokens.embedding.weight', 'lm_head.col.linear.weight'):
        return 0, ps.tp_size, ps.tp_rank
    if '.mlp.experts.fc1.' in name:
        return 0, ps.etp_size, ps.etp_rank
    if '.mlp.experts.fc2.' in name:
        return 1, ps.etp_size, ps.etp_rank
    for suffix, dim in [('wq_b.weight', 0), ('wo_a.weight', 0), ('wo_b.weight', 1), ('sinks', 0)]:
        if name.endswith('.self_attn.self_attn.' + suffix):
            return dim, ps.tp_size, ps.tp_rank
    return None


@torch.no_grad()
def load_shards(
    holder: torch.nn.Module,
    path: str,
    config: DeepseekV4Config,
    ps: ParallelState,
    layer_map: dict[int, int] | None = None,
    skip_experts: bool = False,
) -> None:
    """Copy real checkpoint weights, slicing TP attention and ETP experts."""
    with ckpt.SafeTensorReader(path) as reader:
        for name, target in holder.state_dict().items():
            if skip_experts and '.mlp.experts.' in name:
                continue
            if ckpt._is_native_metadata_key(name):
                continue
            global_name = ckpt.to_global_layer_name(name, {} if layer_map is None else layer_map)
            names = ckpt._hf_names_for_state_key(
                ckpt._to_global_expert_name(global_name, config, ps), config
            )
            assert names and all(ckpt._has(reader, n) for n in names), name
            destinations = target.chunk(2, dim=0) if len(names) == 2 else (target,)
            spec = shard_spec(name, ps)
            for dest, hf_name in zip(destinations, names, strict=True):
                if spec is None or spec[1] == 1:
                    scale_name = ckpt._scale_name_for_hf_name(hf_name)
                    scale = reader.get_tensor(scale_name) if ckpt._has(reader, scale_name) else None
                    ckpt._copy_param(dest, reader.get_tensor(hf_name), scale=scale)
                    continue
                dim, size, rank = spec
                shape = list(dest.shape)
                shape[dim] *= size
                source = reader.get_tensor(hf_name).cuda()
                scale_name = ckpt._scale_name_for_hf_name(hf_name)
                if ckpt._has(reader, scale_name):
                    source = ckpt._dequantize_scaled_tensor(
                        source, reader.get_tensor(scale_name).cuda(), torch.Size(shape)
                    )
                assert tuple(source.shape) == tuple(shape), (name, source.shape, shape)
                dest.copy_(source.chunk(size, dim=dim)[rank])
                del source
            assert torch.isfinite(target).all(), name
