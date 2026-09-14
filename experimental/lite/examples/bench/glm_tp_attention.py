# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Inference-only GLM attention with TP heads and a replicated full-Q indexer.

Each TP rank computes every query row against the complete K sequence. Only
the main attention heads are partitioned; no indexer query collective is used.
Projection and sparse-attention temporaries are chunked locally on each GPU.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.lite.model.glm5.lite.checkpoint import _get
from megatron.lite.primitive.kernels import dsa_kernels
from megatron.lite.primitive.modules.attention.dsa import DynamicSparseAttention, rotate_activation


def apply_checkpoint_rope(
    x: torch.Tensor, positions: torch.Tensor, theta: float, *, interleaved: bool
) -> torch.Tensor:
    """Apply the checkpoint's rotary pairing with FP32 rotary arithmetic."""
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    phase = positions.float()[:, None] * inv[None]
    cos, sin = phase.cos(), phase.sin()
    while cos.ndim < x.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    xf = x.float()
    if interleaved:
        even, odd = xf[..., 0::2], xf[..., 1::2]
        output = torch.empty_like(xf)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = odd * cos + even * sin
    else:
        first, second = xf.chunk(2, dim=-1)
        output = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return output.to(x.dtype)


def attention_weight_shard(name: str, weight: torch.Tensor, rank: int, size: int) -> torch.Tensor:
    """Map complete checkpoint head blocks onto one TP rank."""
    if name in ('q_b_proj.weight', 'kv_b_proj.weight'):
        return weight.chunk(size, dim=0)[rank].contiguous()
    if name == 'o_proj.weight':
        return weight.chunk(size, dim=1)[rank].contiguous()
    return weight


@torch.no_grad()
def load_attention(attention: GlmTPAttention, reader, hf_prefix: str) -> None:
    """Load TP head shards; retain replicated Q/K down-projections and indexer."""
    for name, target in attention.named_parameters():
        weight = _get(reader, f'{hf_prefix}.{name}')
        weight = attention_weight_shard(name, weight, attention.ps.tp_rank, attention.ps.tp_size)
        if target.shape != weight.shape:
            raise ValueError(f'{hf_prefix}.{name}: {tuple(weight.shape)} != {tuple(target.shape)}')
        target.copy_(weight)


class GlmTPAttention(DynamicSparseAttention):
    """Full-query TP8 prefill with shared index maps and no decode cache."""

    def __init__(self, config, ps, layer_idx: int, chunk_size: int = 16384):
        if ps.cp_size != 1 or config.num_attention_heads % ps.tp_size:
            raise ValueError('GLM experimental attention requires CP1 and head-divisible TP')
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads // ps.tp_size,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            index_n_heads=config.index_n_heads,
            index_head_dim=config.index_head_dim,
            index_topk=config.index_topk,
            rms_norm_eps=config.rms_norm_eps,
            # Local SGLang GLM uses the model RMS epsilon for both latent norms.
            latent_rms_norm_eps=config.rms_norm_eps,
            rope_interleaved=config.rope_interleave,
            indexer_layer_norm_eps=config.indexer_layer_norm_eps,
            indexer_rope_interleaved=config.indexer_rope_interleave,
            indexer_rope_first=config.indexer_rope_first,
            indexer_use_hadamard=config.indexer_use_hadamard,
            indexer_type=config.indexer_type(layer_idx),
        )
        self.ps, self.config, self.layer_idx = ps, config, layer_idx
        self.chunk_size, self.arm = chunk_size, 'raw'
        self.cuda().bfloat16().eval()
        if self.indexer is not None:
            self.indexer.k_norm.float()

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, *, indexer=False) -> torch.Tensor:
        return apply_checkpoint_rope(
            x,
            positions,
            self.config.rope_theta,
            interleaved=(
                self.config.indexer_rope_interleave if indexer else self.config.rope_interleave
            ),
        )

    def _index_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dim = self.qk_rope_head_dim
        if self.indexer.rope_first:
            output = torch.cat(
                (self._rope(x[..., :dim], positions, indexer=True), x[..., dim:]), -1
            )
        else:
            output = torch.cat(
                (x[..., :-dim], self._rope(x[..., -dim:], positions, indexer=True)), -1
            )
        return rotate_activation(output) if self.indexer.use_hadamard else output

    def _select(self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor):
        from glm_native_indexer import dense_topk

        n = x.shape[0]
        topk = min(n, self.index_topk)
        ends = (positions + 1).to(torch.int32)
        if n <= self.index_topk:
            indices = torch.arange(n, device=x.device, dtype=torch.int32)[None].expand(n, -1)
            indices = indices.masked_fill(indices >= ends[:, None], -1).contiguous()
            lengths = ends.clone()
        else:
            k = torch.empty((n, self.indexer.head_dim), device=x.device, dtype=x.dtype)
            for first in range(0, n, self.chunk_size):
                last = min(first + self.chunk_size, n)
                k[first:last] = self._index_rope(
                    self.indexer.k_norm(self.indexer.wk(x[first:last]).float()).to(x.dtype),
                    positions[first:last],
                )
            q = torch.empty(
                (n, self.indexer.num_heads, self.indexer.head_dim), device=x.device, dtype=x.dtype
            )
            weights = torch.empty((n, self.indexer.num_heads), device=x.device, dtype=torch.float32)
            for first in range(0, n, self.chunk_size):
                last = min(first + self.chunk_size, n)
                q[first:last] = self._index_rope(
                    self.indexer.wq_b(q_resid[first:last]).view(
                        -1, self.indexer.num_heads, self.indexer.head_dim
                    ),
                    positions[first:last],
                )
                weights[first:last] = (
                    F.linear(x[first:last].float(), self.indexer.weights_proj.weight.float())
                    * self.indexer.num_heads**-0.5
                    * self.indexer_softmax_scale
                )
            starts = torch.zeros_like(ends)
            if self.arm == 'raw':
                indices, lengths = dense_topk(q, k, weights, starts, ends, topk)
            elif self.arm == 'litetopk':
                indices, lengths = self._select_litetopk(q, k, weights, starts, ends, topk)
            else:
                raise ValueError(f'Unknown indexer arm: {self.arm}')
            del q, k, weights
        return indices, lengths

    def _select_litetopk(self, q, k, weights, starts, ends, topk):
        from glm_litetopk_wave import local_wave_topk

        from megatron.core.transformer.experimental_attention_variant import (
            dsa_litetopk_kernels as lite,
        )

        indices, lengths, _ = local_wave_topk(
            q,
            k,
            weights,
            starts,
            ends,
            topk,
            state_key=self._litetopk_state_key,
            request_key=lite.current_request_key(),
        )
        return indices, lengths

    def _sparse(self, q, kv, indices, lengths):
        if indices.shape[1] == 2048:
            from glm_trtllm_sparse import sparse_forward

            return sparse_forward(q, kv, indices, self.softmax_scale, topk_length=lengths)
        heads = q.shape[1]
        output = torch.empty((*q.shape[:2], self.kv_lora_rank), device=q.device, dtype=q.dtype)
        for first in range(0, q.shape[0], 2048):
            last = min(first + 2048, q.shape[0])
            padded = F.pad(q[first:last], (0, 0, 0, 64 - heads)) if heads < 64 else q[first:last]
            out, _, _ = dsa_kernels._dsa_fwd_flash_mla(
                padded,
                kv,
                indices[first:last],
                self.softmax_scale,
                d_v=self.kv_lora_rank,
                attn_sink=None,
                topk_length=lengths[first:last],
            )
            output[first:last] = out[:, :heads]
        return output

    @torch.no_grad()
    def forward(self, x, packed_seq_params=None, indexer_share_state=None, position_ids=None):
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError('GLM TP prefill requires SBHD with batch size one')
        if packed_seq_params is not None:
            raise ValueError('GLM TP adapter currently accepts one unpacked prompt')
        x = x[:, 0]
        n = x.shape[0]
        positions = (
            torch.arange(n, device=x.device) if position_ids is None else position_ids.reshape(-1)
        )
        q_resid = torch.empty((n, self.q_lora_rank), device=x.device, dtype=x.dtype)
        kv = torch.empty(
            (n, self.kv_lora_rank + self.qk_rope_head_dim), device=x.device, dtype=x.dtype
        )
        for first in range(0, n, self.chunk_size):
            last = min(first + self.chunk_size, n)
            q_resid[first:last] = self.q_a_layernorm(self.q_a_proj(x[first:last]))
            latent, pe = self.kv_a_proj_with_mqa(x[first:last]).split(
                (self.kv_lora_rank, self.qk_rope_head_dim), -1
            )
            kv[first:last, : self.kv_lora_rank] = self.kv_a_layernorm(latent)
            kv[first:last, self.kv_lora_rank :] = self._rope(pe, positions[first:last])
        if self.indexer is not None:
            if indexer_share_state is not None:
                # This layer replaces the previous map. At 1M, retaining it
                # during full-Q selection would consume another 8 GiB.
                indexer_share_state.pop('glm_tp_indices', None)
            indices, lengths = self._select(x, q_resid, positions)
            if indexer_share_state is not None:
                indexer_share_state['glm_tp_indices'] = (n, indices, lengths)
        else:
            entry = (
                None if indexer_share_state is None else indexer_share_state.get('glm_tp_indices')
            )
            if entry is None or entry[0] != n:
                raise RuntimeError('Shared GLM indexer needs same-request full-layer indices')
            _, indices, lengths = entry
        output = torch.empty_like(x)
        if indices.shape[1] == 2048:
            from glm_trtllm_sparse import prepare_kv

            kv = prepare_kv(kv)
        k_weight, v_weight = self._split_kv_b_weights()
        for first in range(0, n, self.chunk_size):
            last = min(first + self.chunk_size, n)
            query = self.q_b_proj(q_resid[first:last]).view(-1, self.num_heads, self.qk_head_dim)
            nope, pe = query.split((self.qk_nope_head_dim, self.qk_rope_head_dim), -1)
            nope = torch.einsum('shd,hdr->shr', nope, k_weight)
            query = torch.cat((nope, self._rope(pe, positions[first:last])), -1).contiguous()
            context = self._sparse(query, kv, indices[first:last], lengths[first:last])
            value = torch.einsum('shr,hvr->shv', context, v_weight).reshape(last - first, -1)
            output[first:last] = self.o_proj(value)
        if self.ps.tp_size > 1:
            dist.all_reduce(output, group=self.ps.tp_group)
        return output[:, None]
