# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load the explicit external FP32 exact-tie selector without sorting its output."""

import importlib
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch


@lru_cache(maxsize=1)
def _load_selector() -> Callable:
    source = Path(os.environ['GLM_EXACT_TIE_SOURCE']).expanduser().resolve(strict=True)
    for filename in (
        'block_scan.py',
        'indexer_top_k_varlen_util.py',
        'indexer_top_k_decode_varlen.py',
    ):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    name = '_megatron_exact_tie'
    package = ModuleType(name)
    package.__path__ = [str(source)]
    package.__package__ = name
    package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
    sys.modules[name] = package
    return importlib.import_module(f'{name}.indexer_top_k_decode_varlen').cute_dsl_topk_wrapper


def exact_tie_topk(
    input_values: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    top_k: int,
    next_n: int = 1,
    return_val: bool = False,
) -> dict:
    """Select on the original padded-row scores, without copying or sorting.

    The GLM caller clamps ``seq_lens`` to [0, keys]; this bridge checks metadata
    without a device-to-host synchronization. Candidate scratch buffers remain
    owned by its wrapper and are allocated on this same ambient launch stream.
    """
    if (
        not isinstance(input_values, torch.Tensor)
        or not isinstance(seq_lens, torch.Tensor)
        or input_values.ndim != 2
        or input_values.dtype != torch.float32
        or input_values.shape[0] <= 0
        or not 0 < input_values.shape[1] <= 1048576
        or not input_values.is_cuda
        or input_values.stride(1) != 1
        or input_values.stride(0) < input_values.shape[1]
        or seq_lens.shape != (input_values.shape[0],)
        or seq_lens.dtype != torch.int32
        or seq_lens.device != input_values.device
        or not seq_lens.is_contiguous()
        or type(top_k) is not int
        or not 0 < top_k <= 2048
        or type(next_n) is not int
        or next_n != 1
        or type(return_val) is not bool
    ):
        raise ValueError(
            'Expected CUDA FP32 scores [rows,keys<=1M], int32 row lengths, TopK<=2048 and next_n=1'
        )
    selector = _load_selector()
    with torch.cuda.device(input_values.device):
        stream = torch.cuda.current_stream(input_values.device)
        input_values.record_stream(stream)
        seq_lens.record_stream(stream)
        indices, values = selector(input_values, seq_lens, top_k, next_n, return_val=return_val)
        # Record any tensor outputs before rejecting malformed metadata so an
        # exception cannot free asynchronously written CUDA storage too soon.
        for output in (indices, values):
            if isinstance(output, torch.Tensor) and output.is_cuda:
                output.record_stream(stream)
        if (
            not isinstance(indices, torch.Tensor)
            or indices.shape != (input_values.shape[0], top_k)
            or indices.dtype != torch.int32
            or indices.device != input_values.device
            or (not return_val and values is not None)
            or (
                return_val
                and (
                    not isinstance(values, torch.Tensor)
                    or values.shape != indices.shape
                    or values.dtype != torch.float32
                    or values.device != input_values.device
                )
            )
        ):
            raise ValueError('Exact-tie candidate returned incompatible output metadata')
        return {'indices': indices, 'values': values}
