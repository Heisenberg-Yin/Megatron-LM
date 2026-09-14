# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CP1 prompt inference without a full L x (window + L/ratio) index table."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from tp_flash_padding import padded_sparse_forward


@triton.jit
def _indices(
    OUT,
    LENGTHS,
    MAP,
    BEGIN,
    ROWS,
    LOCAL_ROWS,
    BOUNDARY,
    WINDOW,
    RATIO,
    COMPRESSED,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = off // WIDTH, off % WIDTH
    valid = row < ROWS
    q = BEGIN + row
    window_count = tl.minimum(q + 1, WINDOW)
    compressed_count = tl.minimum((q + 1) // RATIO, COMPRESSED)
    length = window_count + compressed_count
    is_window = col < window_count
    compressed_id = col - window_count
    mapped = tl.load(
        MAP + compressed_id, valid & ~is_window & (compressed_id < compressed_count), other=-1
    )
    value = tl.where(
        is_window,
        BOUNDARY + q - window_count + 1 + col,
        tl.where(mapped >= 0, BOUNDARY + LOCAL_ROWS + mapped, -1),
    )
    tl.store(OUT + off, tl.where(col < length, value, -1), valid)
    tl.store(LENGTHS + row, length, valid & (col == 0))


@dataclass
class DenseAttentionPlan:
    """Materialize dense compressed-attention indices only for each query chunk."""

    local_rows: int
    boundary: int
    window: int
    ratio: int
    compressed_width: int
    seq_to_rank_row: torch.Tensor

    def indices(self, begin: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct causal attention slots for the requested query chunk."""
        rows, width = stop - begin, self.window + self.compressed_width
        out = torch.empty((rows, width), device=self.seq_to_rank_row.device, dtype=torch.int32)
        lengths = torch.empty(rows, device=out.device, dtype=torch.int32)
        _indices[(triton.cdiv(out.numel(), 1024),)](
            out,
            lengths,
            self.seq_to_rank_row,
            begin,
            rows,
            self.local_rows,
            self.boundary,
            self.window,
            self.ratio,
            self.compressed_width,
            width,
            1024,
        )
        return out, lengths

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor, sink: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Reuse each query chunk as output after FlashMLA consumes it."""
        assert not torch.is_grad_enabled() and not query.requires_grad
        assert query.shape == (self.local_rows, 8, 512) and query.is_contiguous()
        for begin in range(0, self.local_rows, 4096):
            stop = min(begin + 4096, self.local_rows)
            indices, lengths = self.indices(begin, stop)
            padded_sparse_forward(
                query[begin:stop],
                kv,
                indices,
                scale,
                attn_sink=sink,
                topk_length=lengths,
                reuse_query=True,
            )
        return query.reshape(self.local_rows, -1)


def install_streamed_dense_attention(csa_module: object, layout_module: object) -> None:
    """Only the experimental full-prompt entry installs this CP1 inference adapter."""
    original_indices = layout_module.build_attention_indices
    original_attention = csa_module.csa_sparse_attn

    def build(
        cu_seqlens,
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        compressed_width,
        compressed_topk=None,
        **kw,
    ):
        if (
            not torch.is_grad_enabled()
            and global_start == 0
            and l_local >= 1048576
            and cu_seqlens.numel() == 2
            and ratio == 128
            and compressed_topk is None
            and not kw.get('for_indexer_loss', False)
        ):
            return (
                DenseAttentionPlan(
                    l_local, d_window, window_size, ratio, compressed_width, kw['seq_to_rank_row']
                ),
                None,
                None,
            )
        return original_indices(
            cu_seqlens,
            global_start,
            l_local,
            d_window,
            window_size,
            ratio,
            compressed_width,
            compressed_topk,
            **kw,
        )

    def attention(query, kv, sink, indices, scale, **kw):
        if isinstance(indices, DenseAttentionPlan):
            assert kw.get('is_thd') and kw.get('topk_length') is None
            assert kw.get('indexer_topk', 0) == 0
            return indices.forward(query, kv, sink, scale)
        return original_attention(query, kv, sink, indices, scale, **kw)

    layout_module.build_attention_indices = build
    csa_module.csa_sparse_attn = attention
