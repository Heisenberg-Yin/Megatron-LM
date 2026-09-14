# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch

_TOPK = 2048


def preserve_topk_indices(out_idx: torch.Tensor) -> torch.Tensor:
    """Return a ``[1, Q, 2048]`` view without sorting or allocating device storage.

    Args:
        out_idx: Contiguous CUDA int32 ``[Q, 2048]`` tensor. Its original
            order, storage offset, values and storage ownership are retained.

    Returns:
        A view sharing the input storage; no IDs are filtered or reordered.
        The caller must ensure its attention consumer accepts this order.

    Raises:
        ValueError: The input is not contiguous CUDA int32 ``[Q, 2048]``.
    """
    if (
        out_idx.ndim != 2
        or out_idx.shape[1] != _TOPK
        or out_idx.dtype != torch.int32
        or not out_idx.is_cuda
        or not out_idx.is_contiguous()
    ):
        raise ValueError('Expected contiguous CUDA int32 [Q, 2048].')
    return out_idx.unsqueeze(0)
