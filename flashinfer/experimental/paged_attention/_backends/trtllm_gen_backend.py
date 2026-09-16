"""trtllm-gen backend: trtllm_batch_context_with_kv_cache (and, with
``backend="cake"``, the separately versioned Cake FMHA product behind the same
front door — one class, the product name is the only difference).

Dialect: the unified form natively (this is where the canonical form came
from); the only derivations are cum_kv_seq_lens and the bmm scale fold
(bmm1 = sm_scale for unquantized, bmm2 = 1.0).  Attention sinks are a native
run-time argument.

Buffers: the 128 MB workspace is ordinary softmax-stats/scratch and is the
shared one every backend runs on (the legacy wrapper's trtllm-gen branch uses
its shared float workspace the same way).  The kernel's multi-CTA KV
*counters* are the only thing that must be zero-initialized; they live in a
separate KB-sized buffer this backend owns and passes explicitly, sized at
plan time from (batch, heads, SM count).  The kernel self-resets the counters
after every launch, so one zeroing at allocation suffices — and passing it
avoids the fresh ``torch.zeros`` the one-shot function would otherwise
allocate on every ``run()``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .._contracts import LN2, PlanMetadata
from .._planning import Derived
from ._capabilities import _BackendPlanUnsupportedError

# Page sizes the trtllm-gen paged context kernel is shipped for (the
# capability table admits a verified subset of these).
_KERNEL_PAGE_SIZES = frozenset({16, 32, 64, 128, 256, 512, 1024})


class _TrtllmGenBackend:
    def __init__(self, device, kv_layout, workspace, backend: str = "trtllm-gen"):
        self.name = backend  # "trtllm-gen" or "cake"
        self._workspace = workspace  # the controller's shared scratch
        self._kv_layout = kv_layout
        self._device = device
        self._sm_count: Optional[int] = None
        self._counter: Optional[torch.Tensor] = None  # zeroed multi-CTA KV counters
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None

    def preflight(self, meta: PlanMetadata) -> None:
        """Batch-specific checks; typed unsupported only, no allocation."""
        if meta.page_size not in _KERNEL_PAGE_SIZES:
            raise _BackendPlanUnsupportedError(
                f"{self.name} has no paged context kernel for page_size "
                f"{meta.page_size} (shipped: {sorted(_KERNEL_PAGE_SIZES)})"
            )
        if not meta.causal and meta.window_left >= 0:
            raise _BackendPlanUnsupportedError(
                f"{self.name} has no non-causal sliding-window context kernel "
                "(window_left >= 0 requires causal=True)"
            )

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        from ....utils import (
            _get_trtllm_gen_multi_ctas_kv_counter_buffer,
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )

        # Page-table ABI: the launcher takes the row stride from
        # block_tables.size(-1) and the kernel walks a raw int32 pointer, so a
        # narrow VIEW of a wider table (row stride != width) is read as a
        # packed (b, width) array — accepted and silently wrong (31.9% of
        # elements off in the sibling probe).  Reject before any state moves.
        bt = meta.block_tables
        assert bt is not None  # needs_dense contract
        if not bt.is_contiguous():  # row stride == width, unit inner stride
            raise ValueError(
                "trtllm-gen requires a contiguous (batch, width) block_tables, got "
                f"shape {tuple(bt.shape)} with strides {tuple(bt.stride())}: the "
                "kernel walks it as a packed int32 array with row stride == width, "
                "so a narrow view of a wider table is silently misread — pass "
                "block_tables[:, :width].contiguous() (or the full-width table; "
                "extra columns past max_kv_len are fine)"
            )
        if self._sm_count is None:
            self._sm_count = get_device_sm_count(self._device)
        need = get_trtllm_gen_multi_ctas_kv_counter_bytes(
            meta.batch_size, meta.num_qo_heads, self._sm_count
        )
        counter = self._counter
        if counter is None or counter.numel() < need:
            # grows monotonically; stable across CUDA-graph re-plans, whose
            # batch size (and hence size) the controller holds fixed
            counter = _get_trtllm_gen_multi_ctas_kv_counter_buffer(
                meta.batch_size, meta.num_qo_heads, self._sm_count, self._device
            )
        # publish only after the allocation above succeeded
        self._counter = counter
        self._meta, self._derived = meta, derived

    def run(
        self,
        q,
        k_cache,
        v_cache,
        *,
        out=None,
        lse=None,
        sm_scale: float,
        k_scale=None,
        v_scale=None,
        sinks=None,
    ):
        from ....prefill import trtllm_batch_context_with_kv_cache

        meta, derived = self._meta, self._derived
        assert meta is not None and derived is not None
        assert meta.block_tables is not None  # needs_dense contract
        result = trtllm_batch_context_with_kv_cache(
            q,
            (k_cache, v_cache),
            self._workspace,
            meta.block_tables,
            meta.kv_seq_lens,
            meta.max_q_len,
            meta.max_kv_len,
            sm_scale
            * (k_scale if k_scale is not None else 1.0),  # bmm1 (k descale folds in)
            v_scale if v_scale is not None else 1.0,  # bmm2 (v descale)
            meta.batch_size,
            meta.qo_indptr,
            derived.cum_kv_seq_lens,
            window_left=meta.window_left,
            kv_layout=self._kv_layout,
            causal=meta.causal,
            out=out,
            lse=lse,
            return_lse=meta.need_lse,
            # (num_qo_heads,) fp32; the kernel folds it into the softmax
            # denominator and its LSE (measured on B200: matches
            # logaddexp(lse, sink) to 2e-6)
            sinks=sinks,
            multi_ctas_kv_counter_buffer=self._counter,
            backend=self.name,
        )
        if meta.need_lse:
            out_t, lse_t = result
            if meta.lse_mode == "basee":
                lse_t.mul_(LN2)  # trtllm-gen / cake emit base-2; one fold
            return out_t, lse_t
        return result, None


__all__ = ["_TrtllmGenBackend"]
