"""cuDNN backend: cudnn_batch_prefill_with_kv_cache (the tokens mode of PR 3921).

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

Execution: plan() stages a ``_CudnnPrefillPlan`` (the plan-owned metadata
views, graph descriptor keys and static bindings the legacy wrapper also
builds) and run() goes through a prepared ``CudnnPrefillGraph`` with
``matches_plan`` / ``run_planned``, rebinding only Q / K / V / O / Stats per
call.  The staged declaration is the one ``cudnn_batch_prefill_with_kv_cache``
resolves from the same arguments, so both reach the same cached cuDNN graph.
That functional entry remains the fallback where no plan can be staged (a
cuDNN without the direct token-offset path) and the packed-LSE probe.

Declared shape: an eager plan declares a shape class rather than the batch's
exact maxes, so the plans of a serving loop share one cuDNN graph
(``_CudnnBackend._declared_bounds``), and binds a table of its own at the
class width, each request's own pages, instead of the caller's
(``_CudnnBackend._own_pages_table``).  CUDA-graph mode declares its capacity
through the controller's reserved table.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

from .._contracts import PlanMetadata
from .._planning import FORM_BLOCK_TABLES, FORM_Q_SEQ_LENS, Derived
from ._capabilities import _BackendPlanUnsupportedError

# device -> can this cuDNN write ragged (packed) softmax stats on the paged path?
_PACKED_LSE_SUPPORTED: Dict[torch.device, bool] = {}

# cuDNN sizes its own workspace when the SDPA graph is built
# (graph.get_workspace_size()) and the number is not derivable from the
# library.  Measured on cuDNN 9.25 / frontend 1.29 / B200 for prefill,
# decode, ragged and (192, 128) shapes up to 8192 query tokens: 0 or 8960
# bytes.  On cuDNN 9.27 / frontend 1.32 the frontend's SM100 prefill engine
# prefers a plan that takes more where the buffer has the room (2.4 to 16.9
# MB measured up to batch 4, declared (512, 8192)); a buffer of this
# allowance gets a plan within its size instead, which the graph preparation
# selects under the buffer's limit (1408 to 3072 bytes for those shapes).
# This allowance stands in for those measurements in the bound; it is not a
# proof.
_CUDNN_WORKSPACE_ALLOWANCE = 1 << 20


def _probe_packed_lse(device: torch.device, workspace: torch.Tensor) -> None:
    """Build and run the paged SDPA graph with ``batch_offsets_stats`` on a
    two-request toy problem (max_token_per_sequence 3 > 1: the s_qo == 1 case
    is routed to the padded layout by plan() and is not what this probes).

    No host sync and no blocking copy: the random inputs are generated on
    the device and the five small int32 arrays travel in one pinned staging
    tensor with an asynchronous upload (plan() stays zero-sync even the
    first time it runs on a device).
    """
    from ....cudnn import cudnn_batch_prefill_with_kv_cache

    h, d, page = 2, 128, 16
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(5, h, d, dtype=torch.bfloat16, device=device, generator=g)
    k = torch.randn(2, h, page, d, dtype=torch.bfloat16, device=device, generator=g)
    v = torch.randn(2, h, page, d, dtype=torch.bfloat16, device=device, generator=g)
    # qo_indptr [0, 3, 5] | q_lens [3, 2] | kv_lens [5, 4] | block_tables [[0], [1]]
    host = torch.tensor([0, 3, 5, 3, 2, 5, 4, 0, 1], dtype=torch.int32, pin_memory=True)
    ints = host.to(device, non_blocking=True)
    qo_indptr = ints[0:3]
    lse = torch.empty(5, h, dtype=torch.float32, device=device)
    cudnn_batch_prefill_with_kv_cache(
        q,
        k,
        v,
        1.0 / math.sqrt(d),
        workspace,
        max_token_per_sequence=3,
        max_sequence_kv=5,
        actual_seq_lens_q=ints[3:5].view(2, 1, 1, 1),
        actual_seq_lens_kv=ints[5:7].view(2, 1, 1, 1),
        block_tables=ints[7:9].view(2, 1),
        causal=True,
        return_lse=True,
        lse_base="ln",
        batch_offsets_q=qo_indptr,
        batch_offsets_stats=qo_indptr,
        batch_offsets_units="tokens",
        out=torch.empty_like(q),
        lse=lse,
    )


def _packed_lse_supported(device: torch.device, workspace: torch.Tensor) -> bool:
    """Feature probe (once per device): can this cuDNN write ragged (packed)
    softmax stats on the paged path?  See :func:`_probe_packed_lse`.

    The answer is the backend's: a graph it declines raises
    ``cudnn.cudnnGraphNotSupportedError`` (the frontend's NOT_SUPPORTED
    type) and selects the gather fallback for the process lifetime.  Any
    other exception -- a CUDA failure, a bad handle, a sync-debug trip, a
    bug in the probe itself -- is not an answer about the feature: it
    propagates and nothing is cached, so a transient failure costs one
    plan instead of silently routing every later LSE plan of this device to
    the slower gather path.  The numerical agreement of the packed stats
    with the reference is pinned by tests/experimental/paged_attention on
    the supported versions, not re-checked here.
    """
    hit = _PACKED_LSE_SUPPORTED.get(device)
    if hit is not None:
        return hit
    from ....cudnn.prefill import _cudnn_supports_direct_seqlens

    ok = False
    # The ragged-offset multipliers this relies on belong to the direct
    # token-offset path the library gates by version; probe only there.
    # This is the mixed-form gate (cuDNN 9.25), one version above the gate a
    # plan is staged under (_stage_cudnn_plan: 9.24 for explicit lengths).
    # On 9.24 the entry would take the direct path for the toy problem too,
    # but this backend's ragged stats offset has not been run on that
    # version: an error other than the one caught below would fail every
    # LSE plan that reaches the probe, and a probe that returned would
    # select packed stats no test of this package has compared with the
    # reference there.  Unprobed, a 9.24 plan keeps the padded stats and the
    # gather.
    if _cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True):
        import cudnn

        try:
            _probe_packed_lse(device, workspace)
            ok = True
        except cudnn.cudnnGraphNotSupportedError:
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
        # the table cuDNN binds: ceil(declared KV bound / page_size) columns
        self._block_tables: Optional[torch.Tensor] = None
        # (max_q, max_kv) declared to cuDNN for the published plan
        self._declared: Tuple[int, int] = (0, 0)
        # "direct": batch_offsets_stats; "view": max_q_len == 1, padded ==
        # packed; "gather": padded native stats + plan-time gather indices
        self._lse_path = "gather"
        # gather-fallback state
        self._native_lse: Optional[torch.Tensor] = None
        self._batch_ids: Optional[torch.Tensor] = None
        self._pos: Optional[torch.Tensor] = None
        self._device = device
        # Prepared path.  plan() stages the plan-owned cuDNN metadata, graph
        # descriptor keys and static bindings (None: no plan can be staged on
        # this cuDNN and run() takes the functional entry).  The
        # cuDNN graph is also keyed on the q / K / V layouts and the scale
        # float, which only run() sees, so run() fetches it once per
        # signature and keeps it in the short most-recent-first history
        # _reuse_cudnn_prefill_preparation maintains: four entries, of which
        # _prepared (the last one used, checked first) is the head, not a
        # fifth.  Alternating between up to four scales, layouts or
        # declarations therefore does not go back to the process-wide graph
        # cache; with more, each miss costs about one functional call.  The
        # history outlives re-plans; the workspace its graphs were sized
        # against is fixed for the life of this instance.
        self._cudnn_plan: Optional[Any] = None
        self._lse_base = "log2"
        self._prepared: Optional[Any] = None
        self._prepared_history: List[Any] = []

    def preflight(self, meta: PlanMetadata) -> None:
        """Batch-specific checks; typed unsupported only, no allocation."""
        from ....cudnn import prefill as cudnn_prefill

        if not cudnn_prefill.CUDNN_AVAILABLE:
            raise _BackendPlanUnsupportedError(
                "cudnn-frontend python package not importable"
            )

    @staticmethod
    def workspace_bound(name: str, **_unused: Any) -> int:
        """Allowance for cuDNN's graph workspace (measured, see module note)."""
        return _CUDNN_WORKSPACE_ALLOWANCE

    def workspace_need(self, meta: PlanMetadata) -> int:
        """Unknown before the graph is built; cuDNN checks its workspace itself."""
        return 0

    @property
    def lse_written_packed(self) -> bool:
        """True when cuDNN writes the packed LSE directly (no gather)."""
        return self._lse_path != "gather"

    def _declared_bounds(self, meta: PlanMetadata) -> Tuple[int, int]:
        """``(graph_q, graph_kv)``: the per-request bounds declared to cuDNN.

        cuDNN finalizes one execution plan per declared shape (about 5 ms),
        and in a serving loop the longest KV grows every step, so the batch's
        exact maxes would declare a new shape on every step.  An eager plan
        declares a shape class instead; the per-request lengths stay the
        actual ones and mask the live tokens.

        - KV: the next power of two, whatever the paging form and the width
          of the caller's table: cuDNN binds a table of the class width that
          this backend builds (``_own_pages_table``).
        - Q, from 64 tokens up and with a GQA group that divides 128: 64
          stays 64, anything longer rounds up to whole ``128 // group``-token
          query tiles, at least 128 (whole powers of two would hide an
          underfilled tile grid).  Below 64 the bound stays exact: the
          ``max_q_len == 1`` LSE "view" layout and cuDNN's single-token
          paths depend on the exact value.

        These are the formulas of the legacy wrapper's auto path
        (``_cudnn_paged_bounds`` in flashinfer/prefill.py), so in the region
        that path admits (a longest query of 64 tokens or more, a GQA group
        of 1 to 16) both declare the same shapes, except through a caller
        table narrower than the KV class, where that path keeps both bounds
        exact.  For a shorter query its formula would declare 64, where the
        bound stays exact here.  That path applies them on compute capability
        (10, 0) and (10, 3) only and declares the plan's own maxes elsewhere,
        as it does for an explicit cuDNN backend; this backend applies them
        on every architecture.

        CUDA-graph mode keeps the capacity's values: a captured graph serves
        every later batch of its bucket and cannot follow one batch's class.
        """
        graph_q, graph_kv = meta.max_q_len, meta.max_kv_len
        if self._rows is not None:
            return graph_q, graph_kv
        graph_kv = 1 << (graph_kv - 1).bit_length()
        group, ungrouped = divmod(meta.num_qo_heads, meta.num_kv_heads)
        if graph_q > 64 and not ungrouped and 128 % group == 0:
            q_tile = 128 // group
            graph_q = max(128, -(-graph_q // q_tile) * q_tile)
        return graph_q, graph_kv

    @staticmethod
    def _own_pages_table(
        meta: PlanMetadata, table: torch.Tensor, width: int
    ) -> torch.Tensor:
        """The table an eager plan binds: ``width`` columns per request, its
        own pages followed by its last live page again.  A new contiguous
        ``(batch, width)`` tensor built on the device from the plan's dense
        ``table`` -- the caller's, or the one derived from flat page ids --
        with no host sync (five small kernels).

        cuDNN never sees ``table`` itself.  The KV class reaches past the
        columns the caller's ``max_kv_len`` covers, which were never
        initialised for this batch, and cuDNN does load pages through the
        columns past a request's own: one of its backend engines fetches
        whole 128-token KV tiles, so up to ``128 / page_size - 1`` columns
        past a request's last page, and multiplies the masked (zero)
        probabilities with what it fetched.  A page id outside the pool is
        harmless there, but a pool page holding NaN / Inf reaches the output
        as ``0 * NaN`` (measured with NaN-filled pages on cuDNN 9.25 and
        9.27, SM100).  The request's own last page is one whose content the
        in-page tail contract already requires to be finite;
        ``_planning.derive`` clamps the table it derives from flat page ids
        the same way.

        Only each request's own pages are read from ``table``: the column
        index is clamped to the request's last live page, which lies within
        the ``ceil(max_kv_len / page_size)`` columns plan() requires of it.
        ``table`` may therefore be narrower than the class, and what the
        caller left past a request's pages is not copied.  That last page
        comes from the device lengths, which nothing validates (the host
        checks hold their CPU mirror to ``max_kv_len``), so it is also
        capped at the last column of ``table``: a device length past its
        mirror repeats that column, where an index past the table would be
        a device-side assert that ends every later CUDA call of the process.

        One layout for every source.  The table's shape and strides are part
        of the cuDNN graph key (``_prefill_descriptor_key`` in
        flashinfer/cudnn/prefill.py), so a view that kept the caller's row
        stride would build one graph per caller table width; with this one a
        caller table of any width or row stride and a table derived from
        flat page ids reach one graph per (batch, class).

        CUDA-graph mode binds the controller's reserved table under its
        capacity bound instead (a capture keeps that pointer), so there the
        caller's columns do reach cuDNN; and where the bound is not a whole
        number of 128-token tiles the same engine also loads pool page 0 for
        the rest of the last tile, whatever the table names.  A KV class is
        whole tiles from 128 tokens up, and no such load was measured below.
        """
        # A request's last live page is the number of page ends below its
        # length, counted over the columns both tables have: 0 for a kv_len 0
        # row, which repeats its first column, and never past ``table``.
        # (kv_len - 1) // page_size costs the same two kernels without the
        # cap.
        page = meta.page_size
        ends = torch.arange(
            page,
            min(table.shape[1], width) * page,
            page,
            device=table.device,
            dtype=meta.kv_seq_lens.dtype,
        )
        last = torch.bucketize(meta.kv_seq_lens, ends)
        cols = torch.arange(width, device=table.device, dtype=torch.int64)
        return torch.gather(table, 1, torch.minimum(cols, last.unsqueeze(1)))

    def _stage_cudnn_plan(
        self,
        meta: PlanMetadata,
        derived: Derived,
        block_tables: torch.Tensor,
        path: str,
        declared: Tuple[int, int],
    ) -> Optional[Any]:
        """Plan-time half of the prepared path: the ``_CudnnPrefillPlan`` that
        run() executes through, or None where only the functional entry can
        serve the plan.

        The metadata is the one ``cudnn_batch_prefill_with_kv_cache`` resolves
        per call from the arguments run() hands it on the fallback path (the
        same ``declared`` bounds, the same length / offset / table tensors, a
        stats offset only on the direct LSE path), so both compute the same
        graph descriptor key and share one cached cuDNN graph.  Nothing is
        written to the backend here; plan() publishes the result with the
        rest.
        """
        from ....cudnn import prefill as cudnn_prefill

        # The library's own gate for the form staged here, token-unit offsets
        # next to explicit per-request Q and KV lengths (cuDNN 9.24 for f16 /
        # bf16; the helper the legacy wrapper's plan() and the entry's own
        # resolution ask), read as a module attribute on every plan because
        # tests steer it.  Below it the entry converts the offsets to element
        # units on each call; a plan staged once would freeze that copy while
        # the offsets keep changing in place (graph mode), so run() keeps
        # calling the entry there.
        if not cudnn_prefill._cudnn_supports_paged_token_offsets(
            meta.q_dtype, explicit_q=True, explicit_kv=True
        ):
            return None
        b = meta.batch_size
        metadata = cudnn_prefill._PrefillMetadata(
            *declared,
            meta.causal,
            meta.need_lse,
            o_data_type=meta.q_dtype,
            actual_seq_lens_q=derived.require(FORM_Q_SEQ_LENS).view(b, 1, 1, 1),
            actual_seq_lens_kv=meta.kv_seq_lens.view(b, 1, 1, 1),
            block_tables=block_tables,  # width == ceil(graph_kv / page)
            batch_offsets_q=meta.qo_indptr,
            batch_offsets_o=meta.qo_indptr,
            batch_offsets_stats=meta.qo_indptr if path == "direct" else None,
        ).resolve_from_plan(
            meta.q_dtype,
            meta.num_qo_heads,
            meta.num_kv_heads,
            meta.head_dim_qk,
            meta.head_dim_vo,
        )
        if not metadata._token_offsets:
            # resolved to the element-offset conversion after all
            return None
        # The device is the workspace's (indexed): the plan compares it with
        # every tensor it binds.  With the previous plan at hand, prepare()
        # returns that same object when nothing it binds moved -- graph mode,
        # where every tensor above is reserved storage, so a captured execute
        # keeps the pointers it baked in -- and otherwise reuses its
        # descriptor keys when only the pointers changed (eager re-plans with
        # fresh metadata tensors of one layout).
        return cudnn_prefill._CudnnPrefillPlan.prepare(
            metadata, meta.q_dtype, self._workspace.device, self._cudnn_plan
        )

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # Page-table ABI: cuDNN requires the table's page dimension to equal
        # ceil(max_sequence_kv / page_size) exactly (CUDNN_STATUS_BAD_PARAM
        # otherwise), while engines hand over capacity-width tables and the
        # table derived from flat page ids has the batch's own width.
        bt = meta.block_tables
        assert bt is not None  # needs_dense contract
        # once per plan: the table width, the staged metadata, the fallback
        # call's maxes and the padded stats buffer all follow this one pair
        graph_q, graph_kv = self._declared_bounds(meta)
        page = meta.page_size
        width = (graph_kv + page - 1) // page
        # what the batch's own pages take; the declared width in graph mode
        own_width = (meta.max_kv_len + page - 1) // page
        if bt.shape[1] > 1 and bt.stride(1) != 1:
            raise ValueError(
                "cudnn requires block_tables rows to be unit-stride along the page "
                f"dimension, got strides {tuple(bt.stride())} for shape "
                f"{tuple(bt.shape)} — pass a row-major (batch, width) table (a "
                "column slice [:, :w] of a wider row-major table is fine)"
            )
        # The controller rejects such a table before any backend sees it.
        # Kept for the eager table, which would otherwise repeat the last
        # column for the missing ones instead of failing.
        if bt.shape[1] < own_width:
            raise ValueError(
                f"block_tables has {bt.shape[1]} page columns but max_kv_len "
                f"{meta.max_kv_len} at page_size {page} needs {own_width}"
            )
        if self._rows is None:
            # Eager: the class's columns, built from each request's own pages
            # (see _own_pages_table).  A new tensor per plan, so a plan that
            # fails later leaves the published one's table as it was.
            block_tables = self._own_pages_table(meta, bt, width)
        else:
            # Graph mode: the reserved table itself, the caller's columns
            # included.  The reserved table has the declared width
            # (GraphCapacity pairs table_width with max_kv_len;
            # reserve_dense_table uses the same width), so this takes the
            # table itself; never copy: a capture keeps the buffer's pointer.
            block_tables = bt[:, :width] if bt.shape[1] != width else bt

        native_lse = batch_ids = pos = None
        path = "gather"
        if meta.need_lse:
            if meta.max_q_len == 1:
                # padded (b, 1, h) is the packed (b, h) buffer only while every
                # request has exactly one token: a q_len 0 row (vLLM's padded
                # query_start_loc tail) shifts the packed rows after it, and
                # in graph mode a later re-plan may introduce one while the
                # captured graph keeps the layout it was captured with -- so
                # the view is eager-only and needs total tokens == batch size.
                # The direct path is not an option here (cuDNN 9.25 writes no
                # stats at max_q_len 1 with a ragged stats offset); gather it is.
                if self._rows is None and meta.total_q_tokens == meta.batch_size:
                    path = "view"
            elif _packed_lse_supported(self._device, self._workspace):
                # plan() is never inside a graph capture, so the one-time
                # probe (a graph build + execute on a toy problem) belongs here
                path = "direct"
        # Stage before the gather indices below are refilled in place: staging
        # can raise, and those indices are backend-owned storage a captured
        # run keeps reading, outside the controller's rollback of the
        # reserved buffers.
        cudnn_plan = self._stage_cudnn_plan(
            meta, derived, block_tables, path, (graph_q, graph_kv)
        )
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
                0, graph_q - 1
            )
            batch_ids, pos, native_lse = self._batch_ids, self._pos, self._native_lse
            # padded to the declared bound: the shape cuDNN writes
            lse_shape = (meta.batch_size, graph_q, meta.num_qo_heads)
            if native_lse is None or tuple(native_lse.shape) != lse_shape:
                native_lse = torch.empty(*lse_shape, device=dev, dtype=torch.float32)
            # Keep the storage stable when the shapes repeat (CUDA-graph
            # re-plan): refill in place instead of rebinding new tensors.
            # Last, after the allocation above: an eager re-plan with the
            # same token count but another (batch, max_q) refills the
            # previous plan's indices, which must not happen in a plan that
            # then fails and leaves that previous plan published.
            if batch_ids is None or batch_ids.shape != new_batch_ids.shape:
                batch_ids, pos = new_batch_ids, new_pos
            else:
                batch_ids.copy_(new_batch_ids)
                pos.copy_(new_pos)
        # publish only after every allocation above succeeded
        self._meta, self._derived = meta, derived
        self._block_tables = block_tables
        self._declared = (graph_q, graph_kv)
        self._lse_path = path
        self._native_lse, self._batch_ids, self._pos = native_lse, batch_ids, pos
        self._cudnn_plan = cudnn_plan
        # native stats are natural-log: basee costs nothing, base2 one fold
        # (none where cuDNN builds base-2 stats natively)
        self._lse_base = "ln" if meta.lse_mode == "basee" else "log2"

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
        # K / V scales on the f16 / bf16 SDPA graph: the graph has no descale
        # tensors, so k_scale folds into attn_scale -- a host float in the
        # cuDNN graph key, hence part of a prepared graph's signature (one
        # prepared graph per value; the staged plan carries no scale binding)
        # -- and v_scale multiplies the output below, over the caller's whole
        # buffer so a captured graph scales every row the capacity may hold.
        if k_scale is not None:
            sm_scale = sm_scale * k_scale
        plan = self._cudnn_plan
        if plan is None:
            # No staged plan (see _stage_cudnn_plan): the functional entry
            # resolves its metadata, graph and bindings on every call.
            from ....cudnn import cudnn_batch_prefill_with_kv_cache

            out_t, lse_t = cudnn_batch_prefill_with_kv_cache(
                q,
                k_cache,
                v_cache,
                sm_scale,
                self._workspace,
                max_token_per_sequence=self._declared[0],
                max_sequence_kv=self._declared[1],
                actual_seq_lens_q=derived.require(FORM_Q_SEQ_LENS).view(b, 1, 1, 1),
                actual_seq_lens_kv=meta.kv_seq_lens.view(b, 1, 1, 1),
                block_tables=self._block_tables,  # width == ceil(graph_kv / page)
                causal=meta.causal,
                return_lse=meta.need_lse,
                lse_base=self._lse_base,
                batch_offsets_q=meta.qo_indptr,
                batch_offsets_stats=stats_offsets,
                batch_offsets_units="tokens",
                out=out,
                lse=lse_buf,
            )
        else:
            # Hot path: compare this call's q / K / V layouts and scale float
            # with the last prepared graph, then rebind Q, K, V, O and Stats
            # next to the plan's static bindings and execute.  The functional
            # entry instead re-resolves the metadata, computes the graph key
            # twice and builds a new graph wrapper and binding dict on every
            # call, for the same kernels.  A miss -- the first run of a
            # signature, another scale or K/V stride layout, a re-plan with a
            # new declaration -- goes through this instance's history and
            # then the process-wide graph cache, which builds only a graph it
            # has never seen (as the functional entry would).
            if out is None:
                out = torch.empty(
                    q.shape[0],
                    meta.num_qo_heads,
                    meta.head_dim_vo,
                    dtype=q.dtype,
                    device=q.device,
                )
            need_lse, lse_base = meta.need_lse, self._lse_base
            prepared = self._prepared
            if prepared is None or not prepared.matches_plan(
                q, k_cache, v_cache, sm_scale, plan, need_lse, lse_base=lse_base
            ):
                from ....cudnn.prefill import _reuse_cudnn_prefill_preparation

                prepared = self._prepared = _reuse_cudnn_prefill_preparation(
                    self._prepared_history,
                    q,
                    k_cache,
                    v_cache,
                    sm_scale,
                    self._workspace,
                    plan=plan,
                    return_lse=need_lse,
                    lse_base=lse_base,
                )
            out_t, lse_t = prepared.run_planned(
                q,
                k_cache,
                v_cache,
                out,
                lse_buf,
                self._workspace,
                plan=plan,
                lse_base=lse_base,
            )
        if v_scale is not None and v_scale != 1.0:
            out_t.mul_(v_scale)
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
