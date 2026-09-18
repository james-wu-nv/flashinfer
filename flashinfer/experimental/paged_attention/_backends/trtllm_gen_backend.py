"""trtllm-gen backend: trtllm_batch_context_with_kv_cache (and, with
``backend="cake"``, the separately versioned Cake FMHA product behind the same
front door — one class, the product name is the only difference).

Dialect: the unified form natively (this is where the canonical form came
from); the only derivations are cum_kv_seq_lens and the bmm scale fold
(bmm1 = sm_scale * k_scale, bmm2 = v_scale; 1.0 for an omitted scale).
Attention sinks are a native run-time argument.

Buffers: the controller's scratch workspace (the 128 MiB per-device default
or the caller's buffer) is ordinary softmax-stats/scratch and is the shared
one every backend runs on (the legacy wrapper's trtllm-gen branch uses its
shared float workspace the same way).  The kernel's multi-CTA KV
*counters* are the only thing that must be zero-initialized; they live in a
separate KB-sized buffer this backend owns and passes explicitly, sized at
plan time from (batch, heads, SM count).  The kernel self-resets the counters
after every launch, so one zeroing at allocation suffices — and passing it
avoids the fresh ``torch.zeros`` the one-shot function would otherwise
allocate on every ``run()``.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .._contracts import LN2, PlanMetadata
from .._planning import FORM_BLOCK_TABLES, FORM_CUM_KV_SEQ_LENS, Derived
from ._capabilities import _BackendPlanUnsupportedError, _workspace_too_small

# Page sizes the trtllm-gen paged context kernel is shipped for (the
# capability table declares the same set, each measured on B200).
_KERNEL_PAGE_SIZES = frozenset({16, 32, 64, 128, 256, 512, 1024})

# Scratch the context launcher carves out of the shared workspace
# (csrc/trtllm_fmha_kernel_launcher.cu, trtllm_paged_attention_launcher):
# only with an LSE output, the softmax-stats buffer of
# float2[num_qo_heads, batch_size, round_up(max_q_len, 256)] plus a 1 MiB
# guard (kTrtllmGenSoftmaxStatsGuardBytes), 16-byte aligned.  The Context
# mode takes no multi-CTA scratch; its counters live in the backend's own
# buffer.  In graph mode batch_size and max_q_len are the capacity values.
_SOFTMAX_STATS_GUARD_BYTES = 1 << 20


def _softmax_stats_bytes(num_qo_heads: int, batch_size: int, max_q_len: int) -> int:
    slots = num_qo_heads * batch_size * (-(-max_q_len // 256) * 256)
    return (8 * slots + _SOFTMAX_STATS_GUARD_BYTES + 15) // 16 * 16


class _TrtllmGenBackend:
    # the canonical form natively, plus cumulative KV lengths and the dense table
    DERIVED_NEEDS = frozenset({FORM_CUM_KV_SEQ_LENS, FORM_BLOCK_TABLES})

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
        if self.name == "cake" and int(meta.kv_seq_lens_cpu.min()) == 0:
            # Measured on B200 (cake_fmha context, bf16, page 16): a request
            # with kv_len == 0 never returns — the device hangs (ledger M19).
            # Padding rows are legal input, so cake declines the batch and
            # backend="auto" moves on to trtllm-gen / cuDNN / fa2.
            raise _BackendPlanUnsupportedError(
                "cake cannot run a batch with a kv_len == 0 request (the kernel "
                "hangs on an empty KV range); padding rows need trtllm-gen, "
                "cudnn or fa2"
            )
        # Page-table ABI: the launcher takes the row stride from
        # block_tables.size(-1) and the kernel walks a raw int32 pointer, so a
        # narrow VIEW of a wider table (row stride != width) is read as a
        # packed (b, width) array — accepted and silently wrong (31.9% of
        # elements off in the sibling probe).  Declining here (typed) lets
        # backend="auto" fall through to a candidate that walks the table by
        # its strides; an explicit backend surfaces it as a ValueError.
        bt = meta.block_tables
        if bt is not None and not bt.is_contiguous():
            raise _BackendPlanUnsupportedError(
                f"{self.name} requires a contiguous (batch, width) block_tables, "
                f"got shape {tuple(bt.shape)} with strides {tuple(bt.stride())}: "
                "the kernel walks it as a packed int32 array with row stride == "
                "width, so a narrow view of a wider table is silently misread — "
                "pass block_tables[:, :width].contiguous() (or the full-width "
                "table; extra columns past max_kv_len are fine)"
            )

        need = self.workspace_need(meta)
        if need > self._workspace.numel():
            raise ValueError(
                _workspace_too_small(self.name, need, self._workspace.numel())
            )

    @staticmethod
    def workspace_bound(
        name: str,
        *,
        num_qo_heads: int,
        batch_size: int,
        max_q_len: int,
        need_lse: bool,
        **_unused: Any,
    ) -> int:
        """Scratch bytes the context launcher allocates (exact: the softmax
        stats depend on heads, batch size and max_q_len only); 0 without LSE."""
        return (
            _softmax_stats_bytes(num_qo_heads, batch_size, max_q_len) if need_lse else 0
        )

    def workspace_need(self, meta: PlanMetadata) -> int:
        """Scratch bytes ``run()`` will carve out for this plan."""
        if not meta.need_lse:
            return 0
        return _softmax_stats_bytes(meta.num_qo_heads, meta.batch_size, meta.max_q_len)

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        from ....utils import (
            _get_trtllm_gen_multi_ctas_kv_counter_buffer,
            get_device_sm_count,
            get_trtllm_gen_multi_ctas_kv_counter_bytes,
        )

        assert meta.block_tables is not None  # needs_dense contract
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
        # KV ABI: the launcher takes ONE set of pool strides from the K cache
        # (csrc/trtllm_fmha_kernel_launcher.cu: kv_stride_* = key_cache.stride(..),
        # assigned to both kStride* and vStride*), so a V pool with its own
        # page/head/token strides is read with K's and comes out wrong (ledger
        # M16, measured max err 2.67 where cuDNN/fa2 are exact).  Same head
        # dims here, so equal shapes must mean equal strides.
        if v_cache.stride() != k_cache.stride():
            raise ValueError(
                f"{self.name} reads the V cache with the K cache's strides, so "
                "k_cache and v_cache must share one layout; got K strides "
                f"{tuple(k_cache.stride())} vs V strides {tuple(v_cache.stride())} "
                "(independent K/V pools with different page strides are "
                "supported by the fa2 and cudnn backends)"
            )
        result = trtllm_batch_context_with_kv_cache(
            q,
            (k_cache, v_cache),
            self._workspace,
            meta.block_tables,
            meta.kv_seq_lens,
            meta.max_q_len,
            meta.max_kv_len,
            # bmm1 = softmax scale with k_scale folded in, bmm2 = the output
            # scale (v_scale); the kernel applies both for any KV dtype
            # (float KV k = 0.5 / v = 2.0 measured against the oracle on B200)
            sm_scale * (k_scale if k_scale is not None else 1.0),
            v_scale if v_scale is not None else 1.0,
            meta.batch_size,
            meta.qo_indptr,
            derived.require(FORM_CUM_KV_SEQ_LENS),
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
