# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Explicit request-lifecycle API for Megatron-Lite LiteTopK.

``attention_backend_override="litetopk"`` is intentionally not enough on its
own: every logical request must run inside ``litetopk_request_scope`` with a
key that is unique for that request and stable across all of its layers/chunks.
Schedulers should additionally call ``reset_litetopk_request_state`` when a
request is cancelled or otherwise ends before another keyed request can retire
its carry state.

Typical use::

    with litetopk_request_scope(scheduler_request_id):
        runtime.forward_backward(handle, batch, loss_fn, forward_only=True)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch


@contextmanager
def litetopk_request_scope(request_key: object) -> Iterator[None]:
    """Install one request identity for all LiteTopK-enabled model layers.

    ``request_key`` must be non-``None`` and must not be reused for a different
    logical request.  Value types (for example an integer scheduler id) compare
    by value; other objects compare by identity in the production adapter.
    """
    from megatron.core.transformer.experimental_attention_variant.dsa_litetopk_kernels import (
        request_scope,
    )

    with request_scope(request_key):
        yield


def current_litetopk_request_key() -> object:
    """Return the request key installed in the current context, if any."""
    from megatron.core.transformer.experimental_attention_variant.dsa_litetopk_kernels import (
        current_request_key,
    )

    return current_request_key()


def reset_litetopk_request_state(device: Optional[torch.device | str | int] = None) -> None:
    """Retire LiteTopK carry/scratch after cancellation or explicit teardown."""
    from megatron.core.transformer.experimental_attention_variant.dsa_litetopk_kernels import (
        reset_request_state,
    )

    reset_request_state(device)


__all__ = ["current_litetopk_request_key", "litetopk_request_scope", "reset_litetopk_request_state"]
