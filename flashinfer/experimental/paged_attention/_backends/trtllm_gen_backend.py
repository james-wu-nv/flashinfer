"""trtllm-gen backend: trtllm_batch_context_with_kv_cache.

Dialect: the unified form natively (this is where the canonical form came
from); the only derivations are cum_kv_seq_lens and the bmm scale fold
(bmm1 = sm_scale for unquantized, bmm2 = 1.0).

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


class _TrtllmGenBackend:
    name = "trtllm-gen"

    def __init__(self, device, kv_layout, workspace):
        self._workspace = workspace  # the controller's shared scratch
        self._kv_layout = kv_layout
        self._device = device
        self._sm_count: Optional[int] = None
        self._counter: Optional[torch.Tensor] = None  # zeroed multi-CTA KV counters
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        from ....utils import (
            _get_trtllm_gen_multi_ctas_kv_counter_buffer,
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
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
            multi_ctas_kv_counter_buffer=self._counter,
        )
        if meta.need_lse:
            out_t, lse_t = result
            if meta.lse_mode == "basee":
                lse_t.mul_(LN2)  # trtllm-gen emits base-2; one fold
            return out_t, lse_t
        return result, None


__all__ = ["_TrtllmGenBackend"]
