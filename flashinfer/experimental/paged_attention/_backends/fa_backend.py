"""fa2/fa3 backend: the existing BatchPrefillWithPagedKVCacheWrapper.

Dialect: CSR page metadata + host arrays for the split-KV scheduler.  The
host arrays are the pinned int32 views the derivation layer computed from the
mirrors (``FORM_HOST_ARRAYS``), so the wrapper's ``non_blocking`` uploads of
them are real asynchronous copies; from pageable memory they would each be a
blocking staging copy (ledger M6, the vLLM pin_host_range_buf lesson).  The
generated FA wrapper holds its own plan state, so this backend keeps one
wrapper instance as stable storage and re-plans it in place.

Launch: plan() goes through the wrapper's plan(), run() does not go through
its run().  That run() hands the module's ``paged_run`` a positional argument
list which changes only with the plan, so plan() stages the list from the
wrapper's state (``_stage_paged_run``) and run() fills the entries only a
call knows and launches the module itself.  The wrapper's run() stays the
path of every plan the staging does not model (the AttentionSink variant
among them).

Features: ``logits_soft_cap`` and ``custom_mask`` go to the wrapper's plan()
(the kernel's own soft-cap variant; the mask in the legacy flattened
per-request layout, ANDed here with the causal / sliding-window envelope
because MaskMode.CUSTOM replaces the kernel's causal mask).  Attention sinks
are the ``AttentionSink`` JIT variant (``BatchAttentionWithAttentionSinkWrapper``):
the default wrapper's ``run(sinks=)`` reaches the kernel only on the
trtllm-gen path, and its standard fa2 / fa3 adapter refuses a sink tensor
(``NotImplementedError``), so a sink plan selects the variant wrapper here and
passes the sink tensor as the variant's additional run() argument.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .._contracts import LN2, PlanMetadata
from .._planning import FORM_HOST_ARRAYS, FORM_KV_PAGE_INDICES, Derived
from ._capabilities import _BackendPlanUnsupportedError, _workspace_too_small


def _envelope_mask(custom_mask: torch.Tensor, meta: PlanMetadata) -> torch.Tensor:
    """AND the caller's flattened mask with the causal / window envelope.

    Device ops only, from the device metadata the plan already holds: no
    host-to-device copy (the mirrors are never uploaded here) and no sync
    (``repeat_interleave`` gets its ``output_size`` from the mask, whose
    length the contract pinned to ``sum(q_len_i * kv_len_i)``).  Positions
    follow the oracle: query ``p`` of a request sits at absolute KV position
    ``kv_len - q_len + p``.
    """
    if not meta.causal and meta.window_left < 0:
        return custom_mask
    dev = custom_mask.device
    total = custom_mask.numel()
    q_lens = meta.qo_indptr.diff().to(torch.int64)
    kv_lens = meta.kv_seq_lens.to(torch.int64)
    sizes = q_lens * kv_lens
    starts = torch.cumsum(sizes, 0) - sizes
    req = torch.repeat_interleave(
        torch.arange(meta.batch_size, device=dev), sizes, output_size=total
    )
    off = torch.arange(total, device=dev) - starts[req]
    kv_len = kv_lens[req]
    q_pos = torch.div(off, kv_len, rounding_mode="floor")
    kv_pos = off - q_pos * kv_len
    diag = kv_len - q_lens[req] + q_pos
    allowed = custom_mask
    if meta.causal:
        allowed = allowed & (kv_pos <= diag)
    if meta.window_left >= 0:
        allowed = allowed & (kv_pos >= diag - meta.window_left)
    return allowed


# ---------------------------------------------------------------------------
# Scratch workspace the fa2 planner carves out of the shared buffer
# (include/flashinfer/attention/scheduler.cuh: PrefillPlanImpl and
# PrefillSplitQOKVIndptr).  Only the split-KV path touches the float
# workspace, with two 16-byte-aligned allocations (both sizes are multiples
# of 64, so alignment adds nothing):
#
#     rows  = ceil(padded_batch_size * cta_tile_q / gqa_group)
#     tmp_v = num_qo_heads * rows * head_dim_vo * 4
#     tmp_s = num_qo_heads * rows * 4
#
# (cta_tile_q counts packed (row, head) indices, so one work item spans
# cta_tile_q / gqa_group query rows; the planner reserves per query row.)
#
# padded_batch_size counts (request, q-tile, kv-chunk) work items:
#   eager: new_batch_size, which the kv-chunk binary search keeps at or under
#          max_grid_size / num_kv_heads whenever split_kv is on (max_grid_size
#          = 2 * SM count; split_kv is off, and no float scratch is used, when
#          even one chunk per request does not fit that grid);
#   graph: max(max_grid_size / num_kv_heads,
#              ceil(total_rows * gqa_group / cta_tile_q) + batch_size - 1),
#          total_rows being the wrapper's row budget (the graph capacity), and
#          split_kv is always on.
# cta_tile_q is FA2DetermineCtaTileQ (include/flashinfer/utils.cuh) of the
# average (eager) or maximum (graph) packed query length -- a power of two in
# {16, 32, 64, 128}, so the byte count is monotone in it.  The SM90 (fa3)
# planner allocates from the wrapper-owned int workspace only.
# ---------------------------------------------------------------------------
_FA2_BLOCKS_PER_SM = 2  # num_blocks_per_sm in PrefillPlanImpl


def _align16(n: int) -> int:
    return (n + 15) // 16 * 16


def _dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _fa2_cta_tile_q(
    packed_qo_len: int,
    head_dim_vo: int,
    head_dim_qk: int,
    kv_dtype_bytes: int,
    smem_optin: Optional[int],
) -> Tuple[int, ...]:
    """FA2DetermineCtaTileQ on the host: the tiles it can return for this
    input.  One entry, except that the short-query branch consults the
    device's opt-in shared memory; when that is unknown both outcomes are
    returned (the caller takes the max for a bound, the min for a check).
    The pre-Ampere branch is not modelled (fa2 is declared for sm_80+)."""
    if head_dim_vo >= 512:
        return (16,) if packed_qo_len <= 32 else (32,)
    if head_dim_qk >= 512:
        return (16,)
    if packed_qo_len > 64 and head_dim_vo < 256:
        return (128,)
    if packed_qo_len > 16:
        return (64,)
    if smem_optin is None:
        return (16, 64)
    q_tile_smem = 16 * head_dim_qk * 2
    kv_step_smem = (head_dim_qk + head_dim_vo) * 16 * 4 * kv_dtype_bytes
    return (64,) if q_tile_smem + kv_step_smem > smem_optin else (16,)


def _fa2_tile_ceiling(head_dim_vo: int, head_dim_qk: int) -> int:
    """Largest cta_tile_q FA2DetermineCtaTileQ can return for these head dims."""
    if head_dim_vo >= 512:
        return 32
    if head_dim_qk >= 512:
        return 16
    return 128 if head_dim_vo < 256 else 64


def _fa2_split_bytes(
    num_qo_heads: int,
    gqa: int,
    padded_batch_size: int,
    cta_tile_q: int,
    head_dim_vo: int,
) -> int:
    rows = num_qo_heads * -(-padded_batch_size * cta_tile_q // gqa)
    return _align16(rows * head_dim_vo * 4) + _align16(rows * 4)


def _fa2_graph_tiles(
    total_rows: int, batch_size: int, gqa: int, cta_tile_q: int
) -> int:
    return -(-total_rows * gqa // cta_tile_q) + batch_size - 1


# ---------------------------------------------------------------------------
# Positions, in the argument list _FaBackend._stage_paged_run stages, of the
# entries only run() knows.  The list is positional because the module's
# ``paged_run`` is (flashinfer/prefill.py, get_batch_prefill_module), and its
# order is the legacy wrapper's, which upstream extends from time to time:
# tests/experimental/paged_attention/test_paged_attention_fa_prepared.py runs
# one planned wrapper through its own run() and through the staged list and
# compares what reaches ``paged_run``, so a change there fails that test
# instead of shifting an argument here.
# ---------------------------------------------------------------------------
_ARG_Q, _ARG_K_CACHE, _ARG_V_CACHE = 3, 4, 5
_ARG_OUT, _ARG_LSE = 10, 11
_ARG_SM_SCALE = 23


class _FaBackend:
    # CSR page ids on device; the indptr / last-page lengths / KV lengths
    # travel as pinned host arrays (the wrapper uploads them itself)
    DERIVED_NEEDS = frozenset({FORM_KV_PAGE_INDICES, FORM_HOST_ARRAYS})

    def __init__(
        self, device, kv_layout, workspace, backend: str = "fa2", graph_capacity=None
    ):
        self.name = backend
        self._device = device
        self._kv_layout = kv_layout
        self._workspace = workspace
        self._graph_capacity = graph_capacity
        props = torch.cuda.get_device_properties(device)
        self._sm_count = int(props.multi_processor_count)
        self._smem_optin: Optional[int] = getattr(
            props, "shared_memory_per_block_optin", None
        )
        from ....utils import device_support_pdl

        # a fact of the device (the wrapper's run() looks it up on every call)
        self._enable_pdl = device_support_pdl(workspace.device)
        # The default-variant wrapper is the stable storage for plain plans;
        # attention sinks need the AttentionSink JIT variant, whose module is
        # specialized per (dtypes, head dims, sliding window), so those
        # wrappers are built on first use and kept alongside it.
        self._wrapper = self._make_wrapper(None)
        self._sink_wrappers: Dict[Tuple, Any] = {}
        # per wrapper: the stream point after its last schedule upload (see plan)
        self._upload_events: Dict[int, torch.cuda.Event] = {}
        self._active = self._wrapper
        # the active plan's direct launch (_stage_paged_run), None when run()
        # must go through the wrapper's own run()
        self._staged: Optional[Tuple[Any, list, torch.dtype]] = None
        self._lse_mode = "none"
        self._use_sinks = False
        self._total_q_tokens = 0
        self._head_dim_vo = 0

    def _graph_bufs(self) -> Dict[str, torch.Tensor]:
        # The wrapper's own CUDA-graph protocol: it copies each plan's CSR
        # metadata into these reserved buffers, so the captured kernel keeps
        # reading valid pointers across re-plans.
        cap = self._graph_capacity
        if cap is None:
            return {}
        b = cap.batch_size
        i32 = dict(dtype=torch.int32, device=self._device)
        return dict(
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(b + 1, **i32),
            paged_kv_indptr_buf=torch.zeros(b + 1, **i32),
            paged_kv_indices_buf=torch.zeros(cap.flat_capacity, **i32),
            paged_kv_last_page_len_buf=torch.zeros(b, **i32),
        )

    def _make_wrapper(self, sink_key: Optional[Tuple]):
        if sink_key is None:
            from ....prefill import BatchPrefillWithPagedKVCacheWrapper

            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._workspace,
                self._kv_layout,
                backend=self.name,
                **self._graph_bufs(),
            )
        else:
            from ....attention._core import BatchAttentionWithAttentionSinkWrapper

            q_dtype, kv_dtype, head_dim_qk, head_dim_vo, window_left = sink_key
            wrapper = BatchAttentionWithAttentionSinkWrapper(
                self._workspace,
                self._kv_layout,
                backend=self.name,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                window_left=window_left,
                **self._graph_bufs(),
            )
        if self._graph_capacity is not None:
            # The wrapper fixes its row budget from the first plan it sees and
            # afterwards accepts any total_num_rows <= that budget (its CPU
            # scheduler plans for the budget; the kernel reads the live row
            # count from the int workspace).  Seeding it with the capacity
            # lets the first batch be smaller than the bucket.  It is the
            # wrapper's plan() that reads it, so this poke stays for as long
            # as plans go through the wrapper (only run() bypasses it).
            wrapper._max_total_num_rows = self._graph_capacity.total_q_tokens
        return wrapper

    @staticmethod
    def _sink_key(meta: PlanMetadata) -> Tuple:
        # the variant module only distinguishes windowed / unwindowed
        return (
            meta.q_dtype,
            meta.kv_dtype,
            meta.head_dim_qk,
            meta.head_dim_vo,
            -1 if meta.window_left < 0 else 0,
        )

    def preflight(self, meta: PlanMetadata) -> None:
        """Batch-specific checks, no allocation: typed unsupported for what
        another candidate may still serve (the workspace check included); the
        one plain ValueError at the end is not a fallback case."""
        if self.name == "fa3":
            from ....utils import is_sm90a_supported

            if not is_sm90a_supported(self._device):
                raise _BackendPlanUnsupportedError(
                    f"fa3 needs SM90a and CUDA >= 12.3; {self._device} does not qualify"
                )
        else:
            self._check_workspace(meta)
        if meta.use_sinks:
            # The AttentionSink variant owns the softmax update: it has no
            # soft-cap hook, and its combination with MaskMode.CUSTOM is not
            # verified by any suite.  fp8 KV with sinks IS verified (B200,
            # 2026-09-18: e4m3 / e5m2 at D128 / D256 match the sink-aware
            # oracle to 4e-3; k_scale folds into the launch scale below).
            if meta.logits_soft_cap is not None:
                raise _BackendPlanUnsupportedError(
                    f"{self.name} AttentionSink kernel variant has no logits soft cap"
                )
            if meta.custom_mask is not None:
                raise _BackendPlanUnsupportedError(
                    f"{self.name} attention sinks with a custom mask are not verified"
                )
        if meta.custom_mask is not None and self._graph_capacity is not None:
            # Not a fallback case (no other backend takes custom masks): the
            # wrapper's reserved packed-mask storage (custom_mask_buf /
            # mask_indptr_buf) is not wired into the graph capacity yet.
            raise ValueError(
                "custom_mask with use_cuda_graph=True is follow-up work for the "
                "fa backends (reserved packed-mask storage is not part of the "
                "graph capacity yet); plan the masked batch on an eager instance"
            )

    # ------------------------- scratch workspace -------------------------

    @staticmethod
    def workspace_bound(
        name: str,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int,
        kv_dtype: torch.dtype,
        batch_size: int,
        total_q_tokens: int,
        use_cuda_graph: bool,
        sm_count: int,
        smem_optin: Optional[int],
        **_unused: Any,
    ) -> int:
        """Upper bound on the float-workspace bytes the fa2 planner allocates
        for any batch within the geometry (see the module formulas); 0 for fa3.

        Graph mode is exact given the capacity (the planner pads to the same
        maximum); eager mode bounds the work items by the grid and the tile
        by its ceiling, so it is loose by the ratio of the chosen tile to the
        ceiling and of the actual work items to the grid.
        """
        if name != "fa2":
            return 0
        gqa = num_qo_heads // num_kv_heads
        grid = _FA2_BLOCKS_PER_SM * sm_count // num_kv_heads
        if not use_cuda_graph:
            tile = _fa2_tile_ceiling(head_dim_vo, head_dim_qk)
            return _fa2_split_bytes(num_qo_heads, gqa, grid, tile, head_dim_vo)
        tiles = _fa2_cta_tile_q(
            (total_q_tokens - batch_size + 1) * gqa,
            head_dim_vo,
            head_dim_qk,
            _dtype_bytes(kv_dtype),
            smem_optin,
        )
        return max(
            _fa2_split_bytes(
                num_qo_heads,
                gqa,
                max(grid, _fa2_graph_tiles(total_q_tokens, batch_size, gqa, t)),
                t,
                head_dim_vo,
            )
            for t in tiles
        )

    def workspace_need(self, meta: PlanMetadata) -> int:
        """Float-workspace bytes the fa2 planner allocates for THIS batch: the
        planner's own arithmetic (PrefillSplitQOKVIndptr) replayed on the host
        mirrors, no device access.  0 for fa3.  Where the tile is ambiguous
        (unknown shared-memory limit) the smaller outcome is taken, so a
        check built on this never rejects a batch the planner would accept.
        """
        if self.name != "fa2":
            return 0
        gqa = meta.num_qo_heads // meta.num_kv_heads
        page = meta.page_size
        grid = _FA2_BLOCKS_PER_SM * self._sm_count // meta.num_kv_heads
        qo = meta.qo_indptr_cpu.numpy().astype(np.int64)
        packed = (qo[1:] - qo[:-1]) * gqa
        kv = meta.kv_seq_lens_cpu.numpy().astype(np.int64)
        pages = (kv + page - 1) // page
        batch = meta.batch_size
        cap = self._graph_capacity
        kv_bytes = _dtype_bytes(meta.kv_dtype)
        if cap is not None:
            rows = cap.total_q_tokens
            tiles = _fa2_cta_tile_q(
                (rows - batch + 1) * gqa,
                meta.head_dim_vo,
                meta.head_dim_qk,
                kv_bytes,
                self._smem_optin,
            )
        else:
            tiles = _fa2_cta_tile_q(
                int(packed.sum()) // batch,
                meta.head_dim_vo,
                meta.head_dim_qk,
                kv_bytes,
                self._smem_optin,
            )
        min_chunk = max(128 // page, 1)
        need = None
        for tile in tiles:
            q_tiles = -(-packed // tile)
            if meta.window_left >= 0:
                eff = np.minimum(-(-(meta.window_left + tile) // page), pages)
            else:
                eff = pages
            eff = np.maximum(eff, 1)
            max_pages = int(eff.max())
            low, high = min_chunk, max_pages
            while low < high:
                mid = (low + high) // 2
                if int((q_tiles * (-(-eff // mid))).sum()) > grid:
                    low = mid + 1
                else:
                    high = mid
            if cap is None:
                if not low < max_pages:
                    return 0  # no split: the planner touches no float scratch
                padded = int((q_tiles * (-(-eff // low))).sum())
            else:
                padded = max(
                    grid, _fa2_graph_tiles(cap.total_q_tokens, batch, gqa, tile)
                )
            size = _fa2_split_bytes(
                meta.num_qo_heads, gqa, padded, tile, meta.head_dim_vo
            )
            need = size if need is None else min(need, size)
        return need

    def _check_workspace(self, meta: PlanMetadata) -> None:
        """Decline (typed) a batch whose planner allocation cannot fit the
        workspace, before the planner itself overflows (kernel-side
        RuntimeError): auto moves on to a candidate that fits, a named fa2
        surfaces it as a ValueError naming the bytes.  ``workspace_need`` is a
        lower bound on the planner's allocation, so the decline is never
        spurious.  The O(1) bound settles almost every plan; the per-batch
        replay runs only when the bound exceeds the buffer."""
        avail = self._workspace.numel()
        cap = self._graph_capacity
        bound = self.workspace_bound(
            self.name,
            num_qo_heads=meta.num_qo_heads,
            num_kv_heads=meta.num_kv_heads,
            head_dim_qk=meta.head_dim_qk,
            head_dim_vo=meta.head_dim_vo,
            kv_dtype=meta.kv_dtype,
            batch_size=meta.batch_size,
            total_q_tokens=cap.total_q_tokens
            if cap is not None
            else meta.total_q_tokens,
            use_cuda_graph=cap is not None,
            sm_count=self._sm_count,
            smem_optin=self._smem_optin,
        )
        if bound <= avail:
            return
        need = self.workspace_need(meta)
        if need > avail:
            raise _BackendPlanUnsupportedError(_workspace_too_small(need, avail))

    # --------------------------- direct launch ---------------------------

    def _stage_paged_run(
        self, wrapper, meta: PlanMetadata, alibi_slopes: Optional[torch.Tensor]
    ) -> Optional[Tuple[Any, list, torch.dtype]]:
        """The launch ``wrapper.run()`` would make for the plan the wrapper now
        holds, as ``(module, paged_run arguments, output dtype)`` with the
        ``_ARG_*`` entries left for run(); None when that run() would not take
        its plain path, and run() here then calls it.

        The list is BatchPrefillWithPagedKVCacheWrapper.run() from
        ``run_args = [`` to ``self._cached_module.paged_run(*run_args)``, the
        branch without a custom JIT module, entry for entry and read from the
        same attributes; that code is the same for fa2 and fa3 (the two differ
        inside ``paged_run``).  Where the wrapper reads the call's tensors the
        plan's value stands in, because the controller rejects a call that
        differs from the plan before it reaches the backend: the page size
        (``k_cache``'s shape) and the head count and device of ``q`` that key
        the cached ALiBi slopes (``alibi_slopes``, which plan() looks up).

        Attribute reads and one list, no device allocation: this runs after
        the wrapper's plan has committed, where a failure would leave the
        previous plan's list published next to the new plan's schedule.
        """
        from ....prefill import BatchPrefillWithPagedKVCacheWrapper
        from ....utils import MaskMode, TensorLayout

        w = wrapper
        # Every way that run() leaves the plain path, one test each:
        if (
            # a subclass owns its run(): the AttentionSink variant folds the
            # scales into its positional arguments before delegating
            type(w) is not BatchPrefillWithPagedKVCacheWrapper
            # the batch-invariant class takes the whole call
            or w._batch_invariant is not None
            # cute-dsl / cute-dsl-prims / cudnn / trtllm-gen launch elsewhere
            or w._backend not in ("fa2", "fa3")
            # a custom JIT module gets its tail from prepare_jit_additional_args
            or w._jit_module is not None
            # NVFP4 (uint8) and per-(token, head) scaled KV pass kv_cache_sf
            or w._cached_kv_data_type == torch.uint8
            or getattr(w, "_use_token_head_sf", False)
            # the specialized SM90 fp8-KV verify kernel may take the call
            or w._eagle_verify_fp8kv_sm90_state is not None
            # multi-item scoring switches the mask mode
            or w._prefix_len_ptr is not None
            # an fp8 q reads its scale tensors from positional *args
            or w._cached_q_data_type in (torch.float8_e4m3fn, torch.float8_e5m2)
            # nothing planned to launch (that run() asserts on both)
            or w._plan_info is None
            or w._cached_module is None
        ):
            return None
        if w._custom_mask_buf is not None or w._variant_owns_mask:
            mask_mode = MaskMode.CUSTOM.value
        elif w._causal:
            mask_mode = MaskMode.CAUSAL.value
        else:
            mask_mode = MaskMode.NON_CAUSAL.value
        args = [
            w._float_workspace_buffer,
            w._int_workspace_buffer,
            w._plan_info,
            None,  # _ARG_Q
            None,  # _ARG_K_CACHE
            None,  # _ARG_V_CACHE
            w._qo_indptr_buf,
            w._paged_kv_indptr_buf,
            w._paged_kv_indices_buf,
            w._paged_kv_last_page_len_buf,
            None,  # _ARG_OUT
            None,  # _ARG_LSE
            mask_mode,
            TensorLayout[w._kv_layout].value,
            w._window_left,
            self._enable_pdl,
            w._custom_mask_buf,
            w._mask_indptr_buf,
            alibi_slopes,
            w._prefix_len_ptr,
            w._token_pos_in_items_ptr,
            w._max_item_len_ptr,
            0.0 if w._logits_soft_cap is None else w._logits_soft_cap,
            None,  # _ARG_SM_SCALE
            None,  # fp8 q scale tensor (positional *args of an fp8 q)
            None,  # fp8 k scale tensor
            None,  # fp8 v scale tensor
            1.0 if w._rope_scale is None else w._rope_scale,
            1e4 if w._rope_theta is None else w._rope_theta,
            w._token_pos_in_items_len,
            w._workspace_size,
            w._num_qo_heads,
            w._num_kv_heads,
            w._block_tables,
            w._kv_lens_buffer,
            meta.page_size,
            w._max_q_len,
            w._max_kv_len,
            w._batch_size,
            w._qo_indptr_buf,
            w._paged_kv_indptr_buf,
            None,  # sinks (paged_run refuses one on fa2 / fa3; trtllm-gen only)
            None,  # NVFP4 key block scales
            None,  # NVFP4 value block scales
            None,  # skip_softmax_threshold_scale_factor
            True,  # uses_shared_paged_kv_idx
            None,  # use_fp16_softmax
            None,  # uses_spcompress
            w._trtllm_gen_multi_ctas_kv_counter_buffer,
        ]
        return w._cached_module, args, w._cached_o_data_type

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # The kernel walks kv_page_indices as a raw int32 pointer bounded by
        # the page indptr; a strided view would be silently misread.  In graph
        # mode this is the reserved (contiguous) buffer, so only the eager CSR
        # form can fail here — reject before the wrapper's plan moves state.
        idx = derived.require(FORM_KV_PAGE_INDICES)
        if not idx.is_contiguous():
            raise ValueError(
                f"{self.name} requires a contiguous kv_page_indices, got strides "
                f"{tuple(idx.stride())}: the kernel walks the flat page-id list as a "
                "packed int32 array — pass kv_page_indices.contiguous()"
            )
        custom_mask = (
            _envelope_mask(meta.custom_mask, meta)
            if meta.custom_mask is not None
            else None
        )
        if meta.use_sinks:
            key = self._sink_key(meta)
            wrapper = self._sink_wrappers.get(key)
            if wrapper is None:
                wrapper = self._make_wrapper(key)
                self._sink_wrappers[key] = wrapper
        else:
            wrapper = self._wrapper
        # The ALiBi slopes of the staged launch (one buffer per head count and
        # device, cached for the process; the wrapper's run() looks it up on
        # every call) are allocated on first use, so the lookup can fail.  It
        # is made here, before the wrapper's plan moves any state: once that
        # plan has returned, nothing that can fail may run before publication,
        # or run() would launch the previous plan's staged list against the
        # new plan's schedule.  A sink plan is not staged.
        alibi_slopes = None
        if not meta.use_sinks:
            from ....utils import _get_cache_alibi_slopes_buf

            alibi_slopes = _get_cache_alibi_slopes_buf(
                meta.num_qo_heads, wrapper.device
            )
        # The wrapper's C++ plan writes the kernel schedule into ONE pinned
        # host buffer of its own and uploads it with an asynchronous copy on
        # the current stream.  Rewriting that host buffer before the previous
        # upload executed would hand the earlier enqueued — or captured and
        # replayed — run the later batch's schedule (request/tile indices,
        # merge offsets), silently.  Wait for the previous upload first: a
        # query when the GPU has passed it (the common case), a bounded wait
        # when the host runs more than one step ahead of the device.
        upload = self._upload_events.get(id(wrapper))
        if upload is not None and not upload.query():
            upload.synchronize()
        # Host arrays: pinned views the derivation layer already computed, so
        # nothing is allocated or derived here and every upload the wrapper
        # issues from them is asynchronous.  seq_lens=, max_token_per_sequence=
        # and max_sequence_kv= hand the wrapper values it would otherwise
        # recompute from the host arrays (the KV max is an O(batch) Python
        # loop there; the fa kernels never read the wrapper's copy of it).
        wrapper.plan(
            derived.require("qo_indptr_host"),
            derived.require("kv_page_indptr_host"),
            idx,
            derived.require("kv_last_page_len_host"),
            meta.num_qo_heads,
            meta.num_kv_heads,
            meta.head_dim_qk,
            meta.page_size,
            head_dim_vo=meta.head_dim_vo,
            # with a custom mask the wrapper selects MaskMode.CUSTOM and the
            # causal envelope is already folded into the mask above
            custom_mask=custom_mask,
            causal=meta.causal,
            window_left=meta.window_left,
            logits_soft_cap=meta.logits_soft_cap,  # None -> 0.0 (off) in the wrapper
            q_data_type=meta.q_dtype,
            kv_data_type=meta.kv_dtype,
            seq_lens=derived.require("kv_seq_lens_host"),
            max_token_per_sequence=meta.max_q_len,
            max_sequence_kv=meta.max_kv_len,
        )
        if upload is None:
            upload = self._upload_events[id(wrapper)] = torch.cuda.Event()
        upload.record(torch.cuda.current_stream(self._device))
        staged = self._stage_paged_run(wrapper, meta, alibi_slopes)
        # publish only after the wrapper's plan returned -- the launch staged
        # from it included, as one object: a failed re-plan leaves the
        # previous plan's list (and the tensors it holds) in place, and no
        # run() sees one plan's arguments next to another's
        self._active = wrapper
        self._staged = staged
        self._lse_mode = meta.lse_mode
        self._total_q_tokens = meta.total_q_tokens
        self._head_dim_vo = meta.head_dim_vo
        self._use_sinks = meta.use_sinks

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
        need_lse = self._lse_mode != "none"
        if self.name == "fa3" and v_cache.stride() != k_cache.stride():
            # The SM90 binding (csrc/batch_prefill_sm90.cu) takes one set of
            # page/token strides for K and V and asserts they match; a V pool
            # laid out differently from K would trip the native check or be
            # misread.  fa2 and cuDNN walk each pool by its own strides.
            raise ValueError(
                "fa3 requires k_cache and v_cache to share one layout (equal page "
                f"and token strides); got K strides {tuple(k_cache.stride())} vs V "
                f"strides {tuple(v_cache.stride())} — independent K/V pools with "
                "different strides are supported by the fa2 and cudnn backends"
            )
        rows, n = q.shape[0], self._total_q_tokens
        if rows != n:
            # Graph mode: q/out/lse are the capacity-sized capture buffers, and
            # the wrapper insists on q.shape[0] == qo_indptr[-1].  The launch
            # gets the batch's row prefix on either path below — the same
            # storage, so a captured graph keeps reading the same pointers —
            # and the caller's buffers are returned whole.
            if out is None:
                out = torch.empty(
                    rows, q.shape[1], self._head_dim_vo, dtype=q.dtype, device=q.device
                )
            if lse is None and need_lse:
                lse = torch.empty(
                    rows, q.shape[1], dtype=torch.float32, device=q.device
                )
            q_v, out_v = q[:n], out[:n]
            lse_v = lse[:n] if lse is not None else None
        else:
            q_v, out_v, lse_v = q, out, lse
        # k_scale (fp8 dequant scale, or a multiplier of a float K cache)
        # folds into the softmax scale, which is what the legacy wrapper does
        # for every KV dtype (sm_scale *= k_scale before the launch); doing it
        # here covers the AttentionSink variant too, whose run() has no
        # k_scale argument and would otherwise drop it silently.  Soft cap,
        # sinks and the LSE therefore see the scaled logits, as K * k_scale
        # would give.  v_scale is applied below, over the caller's whole buffer.
        if k_scale is not None:
            sm_scale = sm_scale * k_scale
        staged = self._staged
        if staged is not None:
            # Hot path, one call per attention layer: launch the planned
            # module with the list plan() staged.  The wrapper's run() reaches
            # the same paged_run with the same arguments, after a second
            # API-decorator layer, a repeat of the checks the controller has
            # made and a rebuild of this list from the wrapper's attributes:
            # 7.7 of the 22.7 us one unified fa2 run() spent on the host
            # (x86 host of a B200, batch 1, q 128 / kv 8192, 2026-11-10; the
            # legacy wrapper's own run(): 16.8 us), 10-20% of a short kernel.
            # A copy is filled, because the staged list would otherwise keep
            # the caller's tensors alive between calls.  The alternative, and
            # the path of every plan _stage_paged_run declines, is the
            # wrapper's run() below.
            module, args, o_dtype = staged
            if out_v is None:
                # zeros, in the plan's output dtype, as the wrapper allocates
                out_v = torch.zeros(
                    n, q.shape[1], self._head_dim_vo, dtype=o_dtype, device=q.device
                )
            if lse_v is None and need_lse:
                lse_v = torch.empty(n, q.shape[1], dtype=torch.float32, device=q.device)
            args = args.copy()
            args[_ARG_Q] = q_v
            args[_ARG_K_CACHE] = k_cache
            args[_ARG_V_CACHE] = v_cache
            args[_ARG_OUT] = out_v
            args[_ARG_LSE] = lse_v
            args[_ARG_SM_SCALE] = sm_scale
            module.paged_run(*args)
            # as the wrapper's run() returns
            r: Any = (out_v, lse_v) if need_lse else out_v
        elif self._use_sinks:
            # AttentionSink variant: the sink tensor and sm_scale are the
            # module's additional run() arguments (positional, in that order);
            # sm_scale already carries k_scale (folded above), v_scale is
            # applied below over the whole buffer
            r = self._active.run(
                q_v,
                (k_cache, v_cache),
                sinks,
                sm_scale,
                out=out_v,
                lse=lse_v,
                return_lse=need_lse,
            )
        else:
            # A plain plan the staging declined.  The generated-FA wrapper
            # reads sm_scale from plan-time state and its run() has no
            # override, while the kernel takes it as a launch arg; setting it
            # here keeps sm_scale a per-run (per-layer) value on this path as
            # well.  The direct launch passes it as the argument it is and
            # leaves the wrapper's attribute alone.
            self._active._sm_scale = sm_scale
            r = self._active.run(
                q_v,
                (k_cache, v_cache),
                out=out_v,
                lse=lse_v,
                return_lse=need_lse,
            )
        # Post-processing (fp8 V descale, LSE base fold) runs over the WHOLE
        # caller buffer, never the batch's row prefix: under CUDA-graph
        # capture the prefix is frozen at the first batch's row count, and a
        # later re-plan that grows the live rows would leave the new rows
        # unscaled / in base 2 (review R1).  Rows past the batch are
        # unspecified by contract, so scaling them is harmless.
        out_full = out if out is not None else (r[0] if need_lse else r)
        if v_scale is not None and v_scale != 1.0:
            out_full.mul_(v_scale)  # the wrapper would scale the row prefix only
        if not need_lse:
            return out_full, None
        _, lse_t = r
        lse_full = lse if lse is not None else lse_t
        if self._lse_mode == "basee":
            lse_full.mul_(LN2)  # FA kernels emit base-2 (exp2 softmax); one fold
        return out_full, lse_full


__all__ = ["_FaBackend"]
