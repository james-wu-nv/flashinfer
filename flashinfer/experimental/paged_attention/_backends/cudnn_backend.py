"""cuDNN backend: cudnn_batch_prefill_with_kv_cache (post-#3921 tokens mode).

Dialect: token-unit indptr as batch offsets (units="tokens"), per-request
lens as (b,1,1,1).  Native LSE is base-2 padded (b, max_q, h) (natural-log
before #4663); this backend gathers it to packed (tokens, h).
"""

from __future__ import annotations

from typing import Optional

import torch

from .._contracts import PlanMetadata
from .._planning import FORM_BLOCK_TABLES, FORM_Q_SEQ_LENS, Derived
from ._capabilities import _BackendPlanUnsupportedError


class _CudnnBackend:
    name = "cudnn"
    # per-request query lengths for the padding mask + the dense page table
    DERIVED_NEEDS = frozenset({FORM_Q_SEQ_LENS, FORM_BLOCK_TABLES})

    def __init__(self, device, kv_layout, workspace, graph_capacity=None):
        # The cuDNN graph is built from k/v_cache.stride(), so NHD storage is
        # presented as a zero-copy permuted view with HND logical dim order.
        self._permute_kv = kv_layout == "NHD"
        # CUDA-graph mode: the LSE gather indices keep the capacity's row
        # count, so the captured gather reads stable storage when a smaller
        # batch is re-planned (rows past the batch gather a valid, unused entry)
        self._rows: Optional[int] = (
            graph_capacity.total_q_tokens if graph_capacity is not None else None
        )
        self._workspace = workspace.view(torch.int8)
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        self._block_tables: Optional[torch.Tensor] = None  # width-exact view
        self._native_lse: Optional[torch.Tensor] = None
        self._batch_ids: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None
        self._device = device
        # cuDNN takes fp8 dequant scales as (1,1,1,1) GPU tensors; scales are
        # per-layer constants, so cache one tensor per distinct value (no H2D
        # on the hot path after the first call).
        self._scale_tensors: dict = {}

    def _scale_tensor(self, value):
        if value is None:
            return None
        t = self._scale_tensors.get(value)
        if t is None:
            t = torch.tensor([value], dtype=torch.float32, device=self._device).view(
                1, 1, 1, 1
            )
            self._scale_tensors[value] = t
        return t

    def preflight(self, meta: PlanMetadata) -> None:
        """Batch-specific checks; typed unsupported only, no allocation."""
        from ....cudnn import prefill as cudnn_prefill

        if not cudnn_prefill.CUDNN_AVAILABLE:
            raise _BackendPlanUnsupportedError(
                "cudnn-frontend python package not importable"
            )

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # Page-table ABI: cuDNN requires the table's page dimension to equal
        # ceil(max_sequence_kv / page_size) exactly (CUDNN_STATUS_BAD_PARAM
        # otherwise), while engines hand over capacity-width tables.  The
        # graph is stride-driven, so a narrow VIEW keeping the caller's row
        # stride is correct (sibling probe) — take it here, never copy.  In
        # graph mode the view is of the reserved buffer, so its pointer is as
        # stable as the buffer's.
        bt = meta.block_tables
        assert bt is not None  # needs_dense contract
        width = (meta.max_kv_len + meta.page_size - 1) // meta.page_size
        if bt.shape[1] > 1 and bt.stride(1) != 1:
            raise ValueError(
                "cudnn requires block_tables rows to be unit-stride along the page "
                f"dimension, got strides {tuple(bt.stride())} for shape "
                f"{tuple(bt.shape)} — pass a row-major (batch, width) table (a "
                "column slice [:, :w] of a wider row-major table is fine)"
            )
        if bt.shape[1] < width:
            raise ValueError(
                f"block_tables has {bt.shape[1]} page columns but max_kv_len "
                f"{meta.max_kv_len} at page_size {meta.page_size} needs {width}"
            )
        block_tables = bt[:, :width] if bt.shape[1] != width else bt
        # The LSE-gather indices and the native stats buffer are static per
        # plan (qo_indptr and batch size are fixed here) — precompute them so
        # run() stays a single indexed lookup on the hot path.
        native_lse = batch_ids = pos = None
        if meta.need_lse:
            dev = meta.qo_indptr.device
            rows = meta.total_q_tokens if self._rows is None else self._rows
            token = torch.arange(rows, device=dev, dtype=torch.int64)
            bounds = meta.qo_indptr[1:].to(torch.int64)
            # rows past the batch (graph mode) clamp to the last request's
            # first padded slot: in range, never read by callers
            new_batch_ids = torch.searchsorted(bounds, token, right=True).clamp_(
                max=meta.batch_size - 1
            )
            new_pos = (token - meta.qo_indptr.to(torch.int64)[new_batch_ids]).clamp_(
                0, meta.max_q_len - 1
            )
            # Keep the storage stable when the shapes repeat (CUDA-graph
            # re-plan): refill in place instead of rebinding new tensors.
            batch_ids, pos, native_lse = self._batch_ids, self._pos, self._native_lse
            if batch_ids is None or batch_ids.shape != new_batch_ids.shape:
                batch_ids, pos = new_batch_ids, new_pos
            else:
                batch_ids.copy_(new_batch_ids)
                pos.copy_(new_pos)
            lse_shape = (meta.batch_size, meta.max_q_len, meta.num_qo_heads)
            if native_lse is None or tuple(native_lse.shape) != lse_shape:
                native_lse = torch.empty(*lse_shape, device=dev, dtype=torch.float32)
        # publish only after every allocation above succeeded
        self._meta, self._derived = meta, derived
        self._block_tables = block_tables
        self._native_lse, self._batch_ids, self._pos = native_lse, batch_ids, pos

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
        from ....cudnn import cudnn_batch_prefill_with_kv_cache

        meta, derived = self._meta, self._derived
        assert meta is not None and derived is not None
        assert self._block_tables is not None  # needs_dense contract
        assert sinks is None  # capability-excluded (supports_sinks=False)
        b = meta.batch_size
        if self._permute_kv:
            k_cache = k_cache.permute(0, 2, 1, 3)
            v_cache = v_cache.permute(0, 2, 1, 3)
        out_t, lse_t = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            sm_scale,
            self._workspace,
            max_token_per_sequence=meta.max_q_len,
            max_sequence_kv=meta.max_kv_len,
            actual_seq_lens_q=derived.require(FORM_Q_SEQ_LENS).view(b, 1, 1, 1),
            actual_seq_lens_kv=meta.kv_seq_lens.view(b, 1, 1, 1),
            block_tables=self._block_tables,  # width == ceil(max_kv / page)
            causal=meta.causal,
            k_scale=self._scale_tensor(k_scale),
            v_scale=self._scale_tensor(v_scale),
            return_lse=meta.need_lse,
            # native stats are natural-log: basee costs nothing, base2 one fold
            lse_base="e" if meta.lse_mode == "basee" else "2",
            batch_offsets_q=meta.qo_indptr,
            batch_offsets_units="tokens",
            out=out,
            lse=self._native_lse,
        )
        if not meta.need_lse:
            return out_t, None
        # padded (b, max_q, h) -> packed (tokens, h), using the plan-time
        # precomputed gather indices (zero sync). The base was selected above.
        # Graph mode gathers capacity rows; q's rows are what the caller sees.
        packed = lse_t[self._batch_ids, self._pos, :][: q.shape[0]]
        if lse is not None:
            lse.copy_(packed)
            packed = lse
        return out_t, packed


__all__ = ["_CudnnBackend"]
