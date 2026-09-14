# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Full-prompt GLM H8 sparse attention with unit-scale saturating FP8 Q/KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_WORKSPACES = {}
_WORKSPACE_BYTES = 384 * 1024 * 1024


@triton.jit
def _cast_pad_fp8(INPUT, OUTPUT, N: tl.constexpr, PADDED_N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(INPUT + offsets, offsets < N, 0).to(tl.float32)
    values = tl.minimum(tl.maximum(values, -448.0), 448.0)
    tl.store(OUTPUT + offsets, values, offsets < PADDED_N)


@torch.no_grad()
def quantize_unit_fp8(tensor: torch.Tensor, *, padded_rows: int | None = None) -> torch.Tensor:
    """Round contiguous BF16 values to saturated E4M3, optionally zero-padding rows."""
    if not tensor.is_cuda or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
        raise ValueError('FP8 conversion requires a contiguous CUDA BF16 tensor.')
    if tensor.ndim < 1:
        raise ValueError('FP8 conversion requires a row dimension.')
    rows = tensor.shape[0] if padded_rows is None else int(padded_rows)
    if rows < tensor.shape[0]:
        raise ValueError('Padded rows cannot truncate input rows.')
    output = torch.empty((rows, *tensor.shape[1:]), dtype=torch.float8_e4m3fn, device=tensor.device)
    if output.numel():
        _cast_pad_fp8[(triton.cdiv(output.numel(), 2048),)](
            tensor, output, tensor.numel(), output.numel(), 2048
        )
    return output


@dataclass(frozen=True)
class PreparedGlmKV:
    """One layer's complete prompt KV, viewed as physical 64-token pages."""

    pages: torch.Tensor
    sequence_length: int


@torch.no_grad()
def prepare_kv(kv: torch.Tensor) -> PreparedGlmKV:
    """Prepare [N,576] full KV once; no per-row or block FP8 scale metadata is used."""
    if kv.ndim != 2 or kv.shape[1] != 576 or not kv.shape[0]:
        raise ValueError('GLM sparse KV must have shape [N,576] with N>0.')
    if not kv.is_cuda or kv.dtype != torch.bfloat16 or not kv.is_contiguous():
        raise ValueError('GLM sparse KV must be contiguous CUDA BF16 input.')
    tokens = kv.shape[0]
    padded_tokens = triton.cdiv(tokens, 64) * 64
    storage = quantize_unit_fp8(kv, padded_rows=padded_tokens)
    return PreparedGlmKV(storage.view(-1, 1, 64, 576), tokens)


def get_workspace(device: torch.device) -> torch.Tensor:
    """Reuse a zero-initialized workspace per CUDA stream, shared across layers."""
    device = torch.device(device)
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return _WORKSPACES[key]


@torch.no_grad()
def sparse_forward(
    q: torch.Tensor,
    kv: PreparedGlmKV,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    topk_length: torch.Tensor,
    workspace: torch.Tensor | None = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute [rows,8,512] from H8 Q and full-prompt physical token IDs.

    ``indices`` is [rows,2048] int32. Its first ``topk_length[row]`` entries
    must be the selected, valid physical token IDs in [0,N); remaining slots
    are inactive. The caller supplies the causal selection: this interface
    does not derive global query positions or change/reorder selected IDs.
    ``topk_length`` is the sparse KV length, not the full prompt length.

    The API treats each query row as a length-one request sharing full KV.
    chunk_size only limits this GPU's local launch; no Q rows leave the GPU.
    """
    if q.ndim != 3 or tuple(q.shape[1:]) != (8, 576):
        raise ValueError('GLM TRTLLM sparse attention requires Q=[rows,8,576].')
    if not q.is_cuda or q.dtype != torch.bfloat16 or not q.is_contiguous():
        raise ValueError('GLM sparse Q must be contiguous CUDA BF16 input.')
    if kv.pages.device != q.device:
        raise ValueError('Prepared KV precision/device is incompatible with Q.')
    if tuple(indices.shape) != (q.shape[0], 2048):
        raise ValueError('GLM TRTLLM sparse attention requires indices=[rows,2048].')
    for name, tensor in (('indices', indices), ('topk_length', topk_length)):
        if tensor.dtype != torch.int32 or tensor.device != q.device or not tensor.is_contiguous():
            raise ValueError(f'{name} must be contiguous CUDA int32 on the query device.')
    if tuple(topk_length.shape) != (q.shape[0],):
        raise ValueError('topk_length must contain one length per query row.')
    if not 1 <= chunk_size <= 65536:
        raise ValueError('Local sparse launch chunk_size must be between 1 and 65536.')
    output = torch.empty((q.shape[0], 8, 512), dtype=torch.bfloat16, device=q.device)
    if not q.shape[0]:
        return output
    if workspace is None:
        workspace = get_workspace(q.device)
    if (
        workspace.dtype != torch.uint8
        or workspace.device != q.device
        or not workspace.is_contiguous()
    ):
        raise ValueError('TRTLLM workspace must be contiguous CUDA uint8 on the query device.')
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    for first in range(0, q.shape[0], chunk_size):
        last = min(first + chunk_size, q.shape[0])
        query = quantize_unit_fp8(q[first:last])
        trtllm_batch_decode_with_kv_cache_mla(
            query=query[:, None],
            kv_cache=kv.pages,
            workspace_buffer=workspace,
            qk_nope_head_dim=192,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=indices[first:last, None],
            seq_lens=topk_length[first:last],
            max_seq_len=kv.sequence_length,
            sparse_mla_top_k=2048,
            out=output[first:last, None],
            bmm1_scale=float(softmax_scale),
            bmm2_scale=1.0,
            backend='trtllm-gen',
            enable_pdl=False,
            skip_softmax_threshold_scale_factor=None,
        )
    return output
