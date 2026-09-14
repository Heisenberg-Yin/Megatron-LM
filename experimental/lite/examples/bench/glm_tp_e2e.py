# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full resident GLM-5.2 forward with pure TP8/EP8 and no query/token sharding.

The measured scope is input token IDs through the final next-token logits.
Every one of the checkpoint's 78 transformer layers executes on each request.
Every TP rank processes all prompt queries and all router token rows.
This experimental inference entry does not export a persistent decoding cache.
"""

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from glm_ep_dispatch import GlmEPForward
from glm_tp_attention import GlmTPAttention, load_attention
from glm_tp_experts import GlmTPExperts

from megatron.lite.model.glm5.config import Glm5Config
from megatron.lite.model.glm5.lite.checkpoint import _get, _lm_head_name, _text_prefix
from megatron.lite.model.glm5.lite.model import Glm5SigmoidTopKRouter, SharedExpert, _swiglu
from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader
from megatron.lite.primitive.modules.attention.litetopk import (
    litetopk_request_scope,
    reset_litetopk_request_state,
)
from megatron.lite.primitive.parallel import VocabParallelEmbedding, VocabParallelOutput
from megatron.lite.primitive.parallel.state import init_parallel
from megatron.lite.runtime.contracts import ParallelConfig


def copy_weight(parameter, reader, name, *, shard=None, rank=0, size=8):
    """Load one checkpoint weight, dequantizing non-expert FP8 to BF16."""
    weight = _get(reader, name)
    if shard is not None:
        weight = weight.chunk(size, dim=shard)[rank]
    assert parameter.shape == weight.shape, (name, parameter.shape, weight.shape)
    parameter.copy_(weight)


class LocalDenseMLP(torch.nn.Module):
    """Dense GLM FFN with TP-sharded matrices and replicated sequence input."""

    def __init__(self, config, ps, reader, prefix):
        super().__init__()
        self.ps = ps
        local_ffn = config.intermediate_size // ps.tp_size
        self.gate_up = torch.nn.Linear(config.hidden_size, 2 * local_ffn, bias=False)
        self.down = torch.nn.Linear(local_ffn, config.hidden_size, bias=False)
        self.cuda().bfloat16()
        for j, name in enumerate(('gate_proj', 'up_proj')):
            copy_weight(
                self.gate_up.weight[j * local_ffn : (j + 1) * local_ffn],
                reader,
                f'{prefix}.{name}.weight',
                shard=0,
                rank=ps.tp_rank,
            )
        copy_weight(
            self.down.weight, reader, f'{prefix}.down_proj.weight', shard=1, rank=ps.tp_rank
        )

    def forward(self, x):
        output = self.down(_swiglu(self.gate_up(x)))
        dist.all_reduce(output, group=self.ps.tp_group)
        return output


class GlmPromptLayer(torch.nn.Module):
    def __init__(self, config, ps, checkpoint, reader, prefix, layer_idx, chunk_size):
        super().__init__()
        hf_prefix = f'{prefix}.layers.{layer_idx}'
        self.input_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cuda().bfloat16()
        copy_weight(self.input_layernorm.weight, reader, f'{hf_prefix}.input_layernorm.weight')
        copy_weight(
            self.post_attention_layernorm.weight,
            reader,
            f'{hf_prefix}.post_attention_layernorm.weight',
        )
        self.self_attention = GlmTPAttention(config, ps, layer_idx, chunk_size=chunk_size)
        load_attention(self.self_attention, reader, f'{hf_prefix}.self_attn')
        if config.is_moe_layer(layer_idx):
            router = (
                Glm5SigmoidTopKRouter(
                    config, ps, router_bias_rate=0.0, compute_aux_loss=False, use_pre_softmax=True
                )
                .cuda()
                .bfloat16()
            )
            copy_weight(router.gate.weight, reader, f'{hf_prefix}.mlp.gate.weight')
            copy_weight(router.expert_bias, reader, f'{hf_prefix}.mlp.gate.e_score_correction_bias')
            shared = SharedExpert(config, ps).cuda().bfloat16()
            shared_prefix = f'{hf_prefix}.mlp.shared_experts'
            local_ffn = config.n_shared_experts * config.moe_intermediate_size // ps.tp_size
            for j, name in enumerate(('gate_proj', 'up_proj')):
                copy_weight(
                    shared.gate_up.linear.weight[j * local_ffn : (j + 1) * local_ffn],
                    reader,
                    f'{shared_prefix}.{name}.weight',
                    shard=0,
                    rank=ps.tp_rank,
                )
            copy_weight(
                shared.down.linear.weight,
                reader,
                f'{shared_prefix}.down_proj.weight',
                shard=1,
                rank=ps.tp_rank,
            )
            experts = GlmTPExperts(config, ps, checkpoint, layer_idx, reader=reader)
            self.mlp = GlmEPForward(router, experts, shared, ps, chunk_size=65536)
        else:
            self.mlp = LocalDenseMLP(config, ps, reader, f'{hf_prefix}.mlp')


class FullGlmPromptModel(torch.nn.Module):
    def __init__(self, config, ps, checkpoint, chunk_size):
        super().__init__()
        self.config, self.ps, self.chunk_size = config, ps, chunk_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
        self.lm_head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)
        self.lm_head.col.linear.sp = False
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cuda().bfloat16()
        self.layers = torch.nn.ModuleDict()
        with SafeTensorReader(checkpoint) as reader:
            prefix = _text_prefix(reader)
            copy_weight(
                self.embed_tokens.embedding.weight,
                reader,
                f'{prefix}.embed_tokens.weight',
                shard=0,
                rank=ps.tp_rank,
            )
            copy_weight(
                self.lm_head.col.linear.weight,
                reader,
                _lm_head_name(reader, prefix),
                shard=0,
                rank=ps.tp_rank,
            )
            copy_weight(self.norm.weight, reader, f'{prefix}.norm.weight')
            for i in range(config.num_hidden_layers):
                self.layers[str(i)] = GlmPromptLayer(
                    config, ps, checkpoint, reader, prefix, i, chunk_size
                )
                if ps.tp_rank == 0:
                    print(
                        json.dumps(
                            {
                                'loaded_layer': i,
                                'resident_gib': torch.cuda.memory_allocated() / 2**30,
                            }
                        ),
                        flush=True,
                    )
        self.requires_grad_(False).eval()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Execute every layer and return the final next-token logits."""
        assert not torch.is_grad_enabled()
        length = ids.shape[1]
        h = torch.empty(
            (length, 1, self.config.hidden_size), device=ids.device, dtype=torch.bfloat16
        )
        for start in range(0, length, self.chunk_size):
            stop = min(length, start + self.chunk_size)
            h[start:stop] = self.embed_tokens(ids[:, start:stop])
        share_state = {}
        for layer in self.layers.values():
            x = torch.empty_like(h)
            for start in range(0, length, self.chunk_size):
                stop = min(length, start + self.chunk_size)
                x[start:stop] = layer.input_layernorm(h[start:stop])
            y = layer.self_attention(x, indexer_share_state=share_state)
            del x
            h.add_(y)
            del y
            for start in range(0, length, 65536):
                stop = min(length, start + 65536)
                y = layer.mlp(layer.post_attention_layernorm(h[start:stop]))
                h[start:stop].add_(y)
                del y
        last = self.norm(h[-1:])
        del h, share_state
        return self.lm_head.gather(self.lm_head(last))


@torch.no_grad()
def main() -> None:
    """Benchmark full-resident TP8/EP8 prefill with replicated prompt rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--litetopk-source', required=True, help='External litetopk.py')
    parser.add_argument('--native-exact-tie-source', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--lengths',
        nargs='+',
        type=int,
        choices=[262144, 524288, 786432, 1048576],
        default=[262144, 524288, 786432, 1048576],
    )
    parser.add_argument(
        '--arms', nargs='+', choices=['raw', 'litetopk'], default=['raw', 'litetopk']
    )
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error('Require positive repeats and nonnegative warmups.')
    os.environ.update(
        {
            'MEGATRON_LITETOPK_PRODUCTION_PATH': str(Path(args.litetopk_source).resolve()),
            'GLM_EXACT_TIE_SOURCE': str(Path(args.native_exact_tie_source).resolve()),
            'SGLANG_LITETOPK_H32_TIE_POLICY': 'logical-id-desc',
            'SGLANG_LITETOPK_H32_SCORE_POLICY': 'native-fp32',
            'SGLANG_LITETOPK_TIERED_SEED_12K': '1',
            'SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW': '13',
            'MEGATRON_LITETOPK_H32_PLAN_GROUP_TILES': '8',
        }
    )
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2)
    dist.init_process_group('nccl', device_id=torch.cuda.current_device())
    ps = init_parallel(ParallelConfig(tp=8, ep=8, etp=1, cp=1))
    if dist.get_world_size() != 8:
        raise ValueError('This GLM benchmark requires eight GPUs.')
    config = Glm5Config.from_hf(args.checkpoint)
    if config.num_hidden_layers != 78:
        raise ValueError('Expected the complete 78-layer GLM-5.2 checkpoint.')
    model = FullGlmPromptModel(config, ps, args.checkpoint, 16384)
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'model': 'GLM-5.2',
        'layers': 78,
        'tp': 8,
        'ep': 8,
        'cp': 1,
        'scope': 'full forward; final next-token logits',
        'numerical_qualified': False,
        'warmups': args.warmups,
        'repeats': args.repeats,
        'results': [],
    }
    for length in args.lengths:
        gen = torch.Generator(device='cuda').manual_seed(20260912)
        ids = torch.randint(0, config.vocab_size, (1, length), device='cuda', generator=gen)
        row = {'length': length}
        for arm in args.arms:
            reset_litetopk_request_state(torch.device('cuda', torch.cuda.current_device()))
            for layer in model.layers.values():
                layer.self_attention.arm = arm
            samples = []
            for step in range(args.warmups + args.repeats):
                dist.barrier()
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                with litetopk_request_scope((length, arm, step)):
                    begin.record()
                    logits = model(ids)
                    end.record()
                end.synchronize()
                elapsed = torch.tensor(begin.elapsed_time(end) / 1000, device='cuda')
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=ps.tp_group)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('Nonfinite final logits.')
                if step >= args.warmups:
                    samples.append(elapsed.item())
                    if ps.tp_rank == 0:
                        torch.save(
                            logits.cpu(),
                            args.output.parent / f'{length}-{arm}-{step-args.warmups}.pt',
                        )
                if ps.tp_rank == 0:
                    print(
                        json.dumps(
                            {'length': length, 'arm': arm, 'step': step, 'seconds': elapsed.item()}
                        ),
                        flush=True,
                    )
                del logits
            row[arm] = {'seconds': median(samples), 'samples': samples}
            if 'raw' in row and 'litetopk' in row:
                row['speedup'] = row['raw']['seconds'] / row['litetopk']['seconds']
            if ps.tp_rank == 0:
                args.output.write_text(
                    json.dumps({**report, 'results': report['results'] + [row]}, indent=2)
                )
        report['results'].append(row)
        del ids
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
