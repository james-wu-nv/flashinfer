"""Concrete paged-prefill backends and the factory the controller uses.

Each backend owns one complete implementation behind the same three calls:
``preflight(meta)`` (cheap, allocation-free, may raise the typed
``_BackendPlanUnsupportedError`` for a batch it cannot run), ``plan(meta,
derived)`` and ``run(q, k_cache, v_cache, *, out, lse)``.  Nothing outside a
backend knows its native dialect or LSE format.
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet

import torch

from ._capabilities import (
    CAPABILITIES,
    MIN_DENSE_PAGE_SIZE,
    PagedAttentionCapabilities,
    _BackendPlanUnsupportedError,
)
from .cudnn_backend import _CudnnBackend
from .fa_backend import _FaBackend
from .trtllm_gen_backend import _TrtllmGenBackend

_FACTORIES: Dict[str, Callable] = {
    "fa2": lambda dev, layout, ws, cap: _FaBackend(dev, layout, ws, "fa2", cap),
    "fa3": lambda dev, layout, ws, cap: _FaBackend(dev, layout, ws, "fa3", cap),
    "cudnn": lambda dev, layout, ws, cap: _CudnnBackend(dev, layout, ws, cap),
    "trtllm-gen": lambda dev, layout, ws, cap: _TrtllmGenBackend(dev, layout, ws),
    "cake": lambda dev, layout, ws, cap: _TrtllmGenBackend(dev, layout, ws, "cake"),
}
_CLASSES: Dict[str, type] = {
    "fa2": _FaBackend,
    "fa3": _FaBackend,
    "cudnn": _CudnnBackend,
    "trtllm-gen": _TrtllmGenBackend,
    "cake": _TrtllmGenBackend,
}


def derived_needs(name: str) -> FrozenSet[str]:
    """The derived forms backend ``name`` reads (``_planning.DERIVED_FORMS``
    names, declared as ``DERIVED_NEEDS`` on the backend class).  An adapter
    detail, so it lives here and not in the capability table."""
    return _CLASSES[name].DERIVED_NEEDS


def workspace_bound(name: str, **geometry) -> int:
    """Conservative scratch-workspace bytes backend ``name`` may carve out of
    the shared buffer for any plan within ``geometry`` (the keyword set of
    ``_controller.workspace_requirements``: heads, head dims, kv dtype, batch
    size, total query tokens, max_q_len, need_lse, graph mode, SM count,
    opt-in shared memory).  Each backend documents its own formula; like
    ``derived_needs`` this is an adapter detail, not a capability."""
    return _CLASSES[name].workspace_bound(name, **geometry)


def make_backend(
    name: str,
    device: torch.device,
    kv_layout: str,
    workspace: torch.Tensor,
    *,
    graph_capacity=None,
):
    """Construct the backend ``name`` (a key of ``CAPABILITIES``).

    ``graph_capacity`` (a ``_graph.GraphCapacity``) is set in CUDA-graph mode so
    backends that keep their own metadata storage (the generated-FA wrapper,
    cuDNN's LSE gather indices) can reserve it at the capacity up front.
    """
    return _FACTORIES[name](device, kv_layout, workspace, graph_capacity)


__all__ = [
    "CAPABILITIES",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "_BackendPlanUnsupportedError",
    "derived_needs",
    "make_backend",
    "workspace_bound",
]
