"""cuDNN backend: cudnn_batch_prefill_with_kv_cache (post-#3921 tokens mode).

Dialect: token-unit indptr as batch offsets (units="tokens"), per-request
lens as (b,1,1,1).  LSE: cuDNN's native stats are natural-log; with
``batch_offsets_stats`` it writes them packed ``(tokens, h)`` straight into
the caller's buffer (ledger M12), otherwise padded ``(b, max_q, h)`` and this
backend gathers them to packed.  Which path applies is decided by a one-time
feature probe per device (``_packed_lse_supported``), not a version compare,
with one measured exception: at ``max_q_len == 1`` cuDNN (9.25 / frontend
1.29, B200) writes no stats at all when a ragged stats offset is set, so that
case uses the padded layout, which for max_q_len == 1 is byte-identical to
the packed one (a view, still no gather).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .._contracts import PlanMetadata
from .._planning import FORM_BLOCK_TABLES, FORM_Q_SEQ_LENS, Derived
from ._capabilities import _BackendPlanUnsupportedError

# device -> can this cuDNN write ragged (packed) softmax stats on the paged path?
_PACKED_LSE_SUPPORTED: Dict[torch.device, bool] = {}


def _packed_lse_supported(device: torch.device, workspace: torch.Tensor) -> bool:
    """Feature probe (once per device): build and run the paged SDPA graph
    with ``batch_offsets_stats`` on a two-request toy problem.

    A frontend without ragged stats offsets, or a backend that rejects them
    for this engine configuration, fails at graph build/execute with an
    exception; that selects the gather fallback.  The probe issues no host
    sync (plan() stays zero-sync): the numerical agreement of the packed stats
    with the reference is pinned by tests/experimental on the supported
    versions, not re-checked here.
    """
    hit = _PACKED_LSE_SUPPORTED.get(device)
    if hit is not None:
        return hit
    from ....cudnn import cudnn_batch_prefill_with_kv_cache
    from ....cudnn.prefill import _cudnn_supports_direct_seqlens

    ok = False
    # the ragged-offset multipliers this relies on belong to the direct
    # (cu_seq_len) path the library gates by version; probe only there
    if _cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True):
        try:
            # max_token_per_sequence 3 (> 1): the s_qo == 1 case is routed to
            # the padded layout by plan() and is not what this probes
            h, d, page = 2, 128, 16
            g = torch.Generator(device=device).manual_seed(0)
            q = torch.randn(5, h, d, dtype=torch.bfloat16, device=device, generator=g)
            k = torch.randn(
                2, h, page, d, dtype=torch.bfloat16, device=device, generator=g
            )
            v = torch.randn(
                2, h, page, d, dtype=torch.bfloat16, device=device, generator=g
            )
            qo_indptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
            q_lens = torch.tensor([3, 2], dtype=torch.int32, device=device)
            kv_lens = torch.tensor([5, 4], dtype=torch.int32, device=device)
            table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
            lse = torch.empty(5, h, dtype=torch.float32, device=device)
            cudnn_batch_prefill_with_kv_cache(
                q,
                k,
                v,
                1.0 / math.sqrt(d),
                workspace,
                max_token_per_sequence=3,
                max_sequence_kv=5,
                actual_seq_lens_q=q_lens.view(2, 1, 1, 1),
                actual_seq_lens_kv=kv_lens.view(2, 1, 1, 1),
                block_tables=table,
                causal=True,
                return_lse=True,
                lse_base="e",
                batch_offsets_q=qo_indptr,
                batch_offsets_stats=qo_indptr,
                batch_offsets_units="tokens",
                out=torch.empty_like(q),
                lse=lse,
            )
            ok = True
        except Exception:  # noqa: BLE001 - any failure means "not supported here"
            ok = False
    _PACKED_LSE_SUPPORTED[device] = ok
    return ok


class _CudnnBackend:
    name = "cudnn"
    # per-request query lengths for the padding mask + the dense page table
    DERIVED_NEEDS = frozenset({FORM_Q_SEQ_LENS, FORM_BLOCK_TABLES})

    def __init__(self, device, kv_layout, workspace, graph_capacity=None):
        # The cuDNN graph is built from k/v_cache.stride(), so NHD storage is
        # presented as a zero-copy permuted view with HND logical dim order.
        self._permute_kv = kv_layout == "NHD"
        # CUDA-graph mode: the LSE gather indices (fallback path) keep the
        # capacity's row count, so the captured gather reads stable storage
        # when a smaller batch is re-planned (rows past the batch gather a
        # valid, unused entry)
        self._rows: Optional[int] = (
            graph_capacity.total_q_tokens if graph_capacity is not None else None
        )
        self._workspace = workspace.view(torch.int8)
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        self._block_tables: Optional[torch.Tensor] = None  # width-exact view
        # "direct": batch_offsets_stats; "view": max_q_len == 1, padded ==
        # packed; "gather": padded native stats + plan-time gather indices
        self._lse_path = "gather"
        # gather-fallback state
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

    @property
    def lse_written_packed(self) -> bool:
        """True when cuDNN writes the packed LSE directly (no gather)."""
        return self._lse_path != "gather"

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

        native_lse = batch_ids = pos = None
        path = "gather"
        if meta.need_lse:
            if meta.max_q_len == 1:
                path = "view"  # padded (b, 1, h) is the packed (b, h) buffer
            elif _packed_lse_supported(self._device, self._workspace):
                # plan() is never inside a graph capture, so the one-time
                # probe (a graph build + execute on a toy problem) belongs here
                path = "direct"
        if meta.need_lse and path == "gather":
            # Gather fallback: the LSE-gather indices and the native padded
            # stats buffer are static per plan (qo_indptr and batch size are
            # fixed here) - precompute them so run() stays a single indexed
            # lookup on the hot path.
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
        self._lse_path = path
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
        stats_offsets = None
        if not meta.need_lse:
            lse_buf = None
        elif self._lse_path == "gather":
            lse_buf = self._native_lse
        else:
            # the caller's packed (tokens, h) buffer is written directly:
            # "direct" through token-unit ragged stats offsets, "view" because
            # at max_q_len == 1 the padded (b, 1, h) layout is the packed one
            lse_buf = (
                lse
                if lse is not None
                else torch.empty(
                    q.shape[0], meta.num_qo_heads, dtype=torch.float32, device=q.device
                )
            )
            if self._lse_path == "direct":
                stats_offsets = meta.qo_indptr
            else:
                lse_buf = lse_buf[:b].view(b, 1, meta.num_qo_heads)
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
            batch_offsets_stats=stats_offsets,
            batch_offsets_units="tokens",
            out=out,
            lse=lse_buf,
        )
        if not meta.need_lse:
            return out_t, None
        if self._lse_path == "direct":
            return out_t, lse_t
        if self._lse_path == "view":
            # hand back the caller's own tensor object, not a view of it
            return out_t, (
                lse if lse is not None else lse_t.view(-1, meta.num_qo_heads)
            )
        # gather fallback: padded (b, max_q, h) -> packed (tokens, h), using
        # the plan-time precomputed gather indices (zero sync).  Graph mode
        # gathers capacity rows; q's rows are what the caller sees.
        packed = lse_t[self._batch_ids, self._pos, :][: q.shape[0]]
        if lse is not None:
            lse.copy_(packed)
            packed = lse
        return out_t, packed


__all__ = ["_CudnnBackend"]
