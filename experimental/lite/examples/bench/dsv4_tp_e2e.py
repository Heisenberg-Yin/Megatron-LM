# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4-Flash checkpoint prefill on TP8/ETP8/EP1/CP1.

Run with torchrun on eight Blackwell GPUs. This experimental inference entry
loads all 43 layers and measures token IDs through the final next-token logits.
It uses native FP4 experts/indexer inputs, BF16 sparse attention, and unsorted
Raw/LiteTopK index maps. It does not implement a persistent decoding KV cache.
"""

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Iterator
from unittest.mock import patch

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from deepgemm_tp_experts import DeepGemmTPExperts
from dsv4_tp_forward import TPAttention, load_shards
from fast_indexer_quant import install_fast_indexer_quant
from fast_mhc import install_fast_mhc
from fast_router import install_fast_router
from fixed_capacity_padding import install_fixed_capacity
from glm_exact_tie_topk import exact_tie_topk
from large_query_rope import fused_partial_rope
from quantized_dense_indexer import matched_quantized_baseline
from streamed_dense_attention import install_streamed_dense_attention
from tp_flash_padding import padded_sparse_forward
from tp_shared_experts import install_tp_shared
from tp_sparse_forward import sparse_forward

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite import model as model_module
from megatron.lite.primitive.modules.attention import csa as csa_module
from megatron.lite.primitive.modules.attention.litetopk import (
    litetopk_request_scope,
    reset_litetopk_request_state,
)
from megatron.lite.primitive.modules.attention.mhc import MultiHeadHyperConnectionHead
from megatron.lite.primitive.parallel import VocabParallelEmbedding, VocabParallelOutput
from megatron.lite.primitive.parallel.state import ParallelState, init_parallel
from megatron.lite.primitive.parallel.thd import _make_packed_seq_params
from megatron.lite.primitive.utils import build_fp8_recipe
from megatron.lite.runtime.contracts import ParallelConfig

LENGTHS = (262144, 524288, 786432, 1048576)
CHUNK_ROWS = 131072
STARTUP_ROWS = 49152
CANDIDATE_CAPACITY = 60000


class FullPromptModel(torch.nn.Module):
    """Resident full checkpoint with the retained inference compute paths."""

    def __init__(self, config: DeepseekV4Config, ps: ParallelState, checkpoint: str):
        super().__init__()
        self.config, self.ps, self.chunk_size = config, ps, CHUNK_ROWS
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, ps)
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head = MultiHeadHyperConnectionHead(
            config.hidden_size, config.hc_mult, config.hc_eps
        )
        self.lm_head = VocabParallelOutput(config.vocab_size, config.hidden_size, ps)
        self.lm_head.col.linear.sp = False
        self.cuda().bfloat16()
        load_shards(self, checkpoint, config, ps, layer_map={})
        self.layers = torch.nn.ModuleDict()
        for i in range(config.num_hidden_layers):
            layer = model_module.DeepseekV4Layer(config, ps, i, fp8=True).cuda().bfloat16()
            layer.mlp.dispatcher.moe_permute_fusion = True
            layer.self_attn.self_attn.attention_backend = 'fused'
            holder = torch.nn.Module()
            holder.layers = torch.nn.ModuleDict({str(i): layer})
            load_shards(holder, checkpoint, config, ps, layer_map={i: i}, skip_experts=True)
            layer.mlp.experts = DeepGemmTPExperts(config, ps, checkpoint, i)
            self.layers[str(i)] = layer
            if ps.tp_rank == 0:
                print(json.dumps({'loaded_layer': i}), flush=True)
        install_fast_router(self)
        install_fixed_capacity(self)
        install_tp_shared(self)
        install_fast_mhc(self)
        self.eval()

    def forward(self, ids: torch.Tensor, positions: torch.Tensor, packed) -> torch.Tensor:
        """Compute final-token logits without retaining full vocabulary outputs."""
        if torch.is_grad_enabled():
            raise RuntimeError('This full-prompt benchmark supports inference only')
        embedding = self.embed_tokens(ids)
        h = embedding.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        del embedding
        for layer in self.layers.values():
            x, post, comb = self.mix_and_norm(layer.attn_hc, layer.input_layernorm, h)
            y = layer.self_attn(x, position_ids=positions, packed_seq_params=packed)
            del x
            self.post_inplace(y, h, post, comb)
            del y, post, comb
            x, post, comb = self.mix_and_norm(layer.ffn_hc, layer.post_attention_layernorm, h)
            layer.mlp.experts.defer_reduce = True
            for begin in range(0, h.shape[0], self.chunk_size):
                stop = min(h.shape[0], begin + self.chunk_size)
                y = layer.mlp(x[begin:stop], input_ids=ids[:, begin:stop].T.contiguous())
                self.post_inplace(y, h[begin:stop], post[begin:stop], comb[begin:stop])
            del x, post, comb, y
            layer.mlp.experts.defer_reduce = False
        last = self.norm(self.hc_head(h[-1:]))
        del h
        return self.lm_head.gather(self.lm_head(last))


def _install_attention() -> None:
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        cp_layout_kernels,
        fused_sparse_attention,
    )

    def hybrid_sparse(q, kv, indices, scale, **kwargs):
        fn = padded_sparse_forward if indices.shape[-1] > 128 else sparse_forward
        if fn is padded_sparse_forward and q.shape[0] >= 1048576:
            kwargs['reuse_query'] = True
        return fn(q, kv, indices, scale, **kwargs)

    eager_rope = csa_module.apply_partial_rope

    def bounded_rope(x, cos, sin, rope_head_dim):
        if (
            not torch.is_grad_enabled()
            and x.ndim == 4
            and x.shape[0] == 1
            and rope_head_dim > 0
            and x.stride(-1) == 1
            and (
                (x.stride(-3) == x.shape[-1] and x.stride(-2) == x.shape[1] * x.shape[-1])
                or (x.stride(-2) == x.shape[-1] and x.stride(-3) == x.shape[-2] * x.shape[-1])
            )
        ):
            return fused_partial_rope(x, cos, sin, rope_head_dim)
        return eager_rope(x, cos, sin, rope_head_dim)

    fused_sparse_attention._csa_fwd_flash_mla = hybrid_sparse
    csa_module.apply_partial_rope = bounded_rope
    install_streamed_dense_attention(csa_module, cp_layout_kernels)
    install_fast_indexer_quant()
    model_module.CompressedSparseAttention = TPAttention


@contextmanager
def _selector_scope() -> Iterator[dict]:
    from megatron.core.transformer.experimental_attention_variant import (
        dsa_cudnn_kernels,
        dsa_litetopk_kernels,
    )

    production = dsa_litetopk_kernels._load_production_litetopk()
    original_prep = production._prep_bufs
    original_production = production.try_large_exact_once_chunk
    state = {'selector_calls': 0}
    buffers = {}

    def prep(*args, **kwargs):
        result = original_prep(*args, **kwargs)
        buffers['current'] = result
        return result

    def select(*args, **kwargs):
        capacity = min(kwargs['cap'], CANDIDATE_CAPACITY)
        kwargs['cap'] = capacity
        dispatched = original_production(*args, **kwargs)
        if dispatched:
            q = args[0].shape[0]
            current = buffers['current']
            counts, status = current['cc'][:q], current['status'][:q]
            torch._assert_async(
                ((status == 0) & (counts <= capacity) & (counts >= 512)).all(),
                'candidate slab overflow or invalid selector result',
            )
            state['selector_calls'] += 1
        return dispatched

    dsa_cudnn_kernels._ensure_dsa_namespace()
    with (
        patch.object(production, '_prep_bufs', prep),
        patch.object(production, 'try_large_exact_once_chunk', select),
        patch.object(dsa_cudnn_kernels._cudnn_dsa, 'indexer_top_k_wrapper', exact_tie_topk),
        matched_quantized_baseline(),
    ):
        yield state


def main() -> None:
    """Load the checkpoint once and measure both selectors on identical prompts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--litetopk-source', required=True)
    parser.add_argument('--native-exact-tie-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--lengths', nargs='+', type=int, choices=LENGTHS, default=list(LENGTHS))
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    # Zero warmups and one repeat are allowed for dependency smoke tests only.
    if args.warmups < 0 or args.repeats < 1 or len(set(args.lengths)) != len(args.lengths):
        parser.error('Use nonnegative warmups, positive repeats, and unique lengths')
    args.fp8_recipe = 'mxfp8'
    os.environ.update(
        MEGATRON_LITETOPK_PRODUCTION_PATH=str(Path(args.litetopk_source).expanduser().resolve()),
        MEGATRON_LITETOPK_H64_STARTUP_Q='0',
        GLM_NATIVE_TOPK_MODE='exact-ties',
        GLM_EXACT_TIE_SOURCE=str(Path(args.native_exact_tie_source).expanduser().resolve()),
        NVTE_FLASH_ATTN='0',
        NVTE_FUSED_ATTN='1',
        NVTE_UNFUSED_ATTN='0',
    )
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', device_id=torch.cuda.current_device())
    ps = init_parallel(ParallelConfig(tp=8, etp=8, cp=1, ep=1))
    assert (
        dist.get_world_size() == 8
        and ps.tp_size == ps.etp_size == 8
        and ps.cp_size == ps.ep_size == 1
    )
    _install_attention()
    config = DeepseekV4Config.from_hf(args.checkpoint)
    assert config.num_hidden_layers == 43 and config.n_shared_experts > 0
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.no_grad():
            model = FullPromptModel(config, ps, args.checkpoint)
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.cuda.empty_cache()
    recipe = build_fp8_recipe(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        config=vars(args),
        scope='full_model_prompt_to_next_token_logits',
        layers=43,
        tp=8,
        etp=8,
        ep=1,
        cp=1,
        chunk_rows=CHUNK_ROWS,
        effective_startup_q=STARTUP_ROWS,
        candidate_capacity=CANDIDATE_CAPACITY,
        indexer_precision='FP4 input, FP32 scores, exact ties',
        expert_precision='FP8 activations x native FP4 checkpoint',
        sparse_attention_precision='BF16',
        final_index_sort=False,
        persistent_decode_cache=False,
        dense_score_budget_bytes=2**30,
        results=[],
    )

    def save():
        if ps.tp_rank == 0:
            output.write_text(json.dumps(report, indent=2) + '\n')

    with torch.no_grad(), _selector_scope() as counts:
        for length in args.lengths:
            gen = torch.Generator(device='cuda').manual_seed(20260912)
            ids = torch.randint(0, config.vocab_size, (1, length), device='cuda', generator=gen)
            positions = torch.arange(length, device='cuda')[None]
            cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
            packed = _make_packed_seq_params(
                cu_seqlens_padded=cu, max_seqlen=length, cp_size=1, cp_rank=0, cp_group=ps.cp_group
            )
            for arm in ('raw', 'litetopk'):
                reset_litetopk_request_state()
                for layer in model.layers.values():
                    layer.self_attn.self_attn.attention_backend = (
                        'fused' if arm == 'raw' else 'litetopk'
                    )
                rows = []
                for step in range(args.warmups + args.repeats):
                    dist.barrier()
                    torch.cuda.synchronize()
                    before = dict(counts)
                    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    with (
                        litetopk_request_scope(('full-prompt', length, arm, 0, step)),
                        te.fp8_autocast(enabled=True, fp8_recipe=recipe),
                    ):
                        start.record()
                        logits = model(ids, positions, packed)
                        end.record()
                    end.synchronize()
                    row = dict(
                        step=step,
                        forward_ms=start.elapsed_time(end),
                        selector={key: counts[key] - before[key] for key in counts},
                    )
                    if arm == 'litetopk':
                        assert row['selector']['selector_calls'] > 0, row
                    else:
                        assert row['selector']['selector_calls'] == 0, row
                    assert (
                        logits.shape == (1, 1, config.vocab_size) and torch.isfinite(logits).all()
                    )
                    rows.append(row)
                    if ps.tp_rank == 0:
                        print(json.dumps(dict(length=length, arm=arm, **row)), flush=True)
                        torch.save(
                            logits.float().cpu(),
                            output.with_suffix(f'.{length}.{arm}.step{step}.logits.pt'),
                        )
                    del logits
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, rows)
                report['results'].append(
                    dict(
                        length=length,
                        arm=arm,
                        median_forward_s=median(
                            max(rank[j]['forward_ms'] for rank in gathered) / 1000
                            for j in range(args.warmups, args.warmups + args.repeats)
                        ),
                        rows=gathered,
                    )
                )
                save()
            del ids, positions, packed
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
