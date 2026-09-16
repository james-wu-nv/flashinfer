"""fa2/fa3 backend: the existing BatchPrefillWithPagedKVCacheWrapper.

Dialect: CSR page metadata + host arrays for the split-KV scheduler.  The
host arrays are the pinned int32 views the derivation layer computed from the
mirrors (``FORM_HOST_ARRAYS``), so the wrapper's ``non_blocking`` uploads of
them are real asynchronous copies; from pageable memory they would each be a
blocking staging copy (ledger M6, the vLLM pin_host_range_buf lesson).  The
generated FA wrapper holds its own plan state, so this backend keeps one
wrapper instance as stable storage and re-plans it in place.

Features: ``logits_soft_cap`` and ``custom_mask`` go to the wrapper's plan()
(the kernel's own soft-cap variant; the mask in the legacy flattened
per-request layout, ANDed here with the causal / sliding-window envelope
because MaskMode.CUSTOM replaces the kernel's causal mask).  Attention sinks
are the ``AttentionSink`` JIT variant (``BatchAttentionWithAttentionSinkWrapper``):
the default wrapper's ``run(sinks=)`` is forwarded to the kernel only on the
trtllm-gen path, so a sink plan selects the variant wrapper here and passes
the sink tensor as the variant's additional run() argument.
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
#     tmp_v = num_qo_heads * padded_batch_size * cta_tile_q * head_dim_vo * 4
#     tmp_s = num_qo_heads * padded_batch_size * cta_tile_q * 4
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
    num_qo_heads: int, padded_batch_size: int, cta_tile_q: int, head_dim_vo: int
) -> int:
    rows = num_qo_heads * padded_batch_size * cta_tile_q
    return _align16(rows * head_dim_vo * 4) + _align16(rows * 4)


def _fa2_graph_tiles(
    total_rows: int, batch_size: int, gqa: int, cta_tile_q: int
) -> int:
    return -(-total_rows * gqa // cta_tile_q) + batch_size - 1


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
        # The default-variant wrapper is the stable storage for plain plans;
        # attention sinks need the AttentionSink JIT variant, whose module is
        # specialized per (dtypes, head dims, sliding window), so those
        # wrappers are built on first use and kept alongside it.
        self._wrapper = self._make_wrapper(None)
        self._sink_wrappers: Dict[Tuple, Any] = {}
        # per wrapper: the stream point after its last schedule upload (see plan)
        self._upload_events: Dict[int, torch.cuda.Event] = {}
        self._active = self._wrapper
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
            # lets the first batch be smaller than the bucket; this poke goes
            # away when the backend calls the FA module directly.
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
        """Batch-specific checks; typed unsupported only, no allocation."""
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
            # soft-cap hook, and its combination with MaskMode.CUSTOM and
            # with fp8 KV dequantization is not verified by any suite.
            if meta.logits_soft_cap is not None:
                raise _BackendPlanUnsupportedError(
                    f"{self.name} AttentionSink kernel variant has no logits soft cap"
                )
            if meta.custom_mask is not None:
                raise _BackendPlanUnsupportedError(
                    f"{self.name} attention sinks with a custom mask are not verified"
                )
            if meta.kv_dtype != meta.q_dtype:
                raise _BackendPlanUnsupportedError(
                    f"{self.name} attention sinks with an fp8 KV cache are not verified"
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
            return _fa2_split_bytes(num_qo_heads, grid, tile, head_dim_vo)
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
            size = _fa2_split_bytes(meta.num_qo_heads, padded, tile, meta.head_dim_vo)
            need = size if need is None else min(need, size)
        return need

    def _check_workspace(self, meta: PlanMetadata) -> None:
        """Reject a batch whose planner allocation cannot fit the workspace
        before the planner itself overflows (kernel-side RuntimeError).  The
        O(1) bound settles almost every plan; the per-batch replay runs only
        when the bound exceeds the buffer."""
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
            raise ValueError(_workspace_too_small(self.name, need, avail))

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
        # publish only after the wrapper's plan returned
        self._active = wrapper
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
        rows, n = q.shape[0], self._total_q_tokens
        if rows != n:
            # Graph mode: q/out/lse are the capacity-sized capture buffers, and
            # the wrapper insists on q.shape[0] == qo_indptr[-1].  Hand it the
            # batch's row prefix — the same storage, so a captured graph keeps
            # reading the same pointers — and return the caller's buffers whole.
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
        if self._use_sinks:
            # AttentionSink variant: the sink tensor and sm_scale are the
            # module's additional run() arguments (positional, in that order)
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
            # The generated-FA wrapper reads sm_scale from plan-time state and
            # its run() has no override, while the kernel takes it as a launch
            # arg.  Setting it here keeps sm_scale a per-run (per-layer)
            # value; this poke goes away when the backend calls the FA module
            # directly.
            self._active._sm_scale = sm_scale
            r = self._active.run(
                q_v,
                (k_cache, v_cache),
                k_scale=k_scale,  # fp8 KV: folded into the softmax scale by the wrapper
                v_scale=v_scale,  # fp8 KV: applied to the output by the kernel path
                out=out_v,
                lse=lse_v,
                return_lse=need_lse,
            )
        if not need_lse:
            return (out if out is not None else r), None
        out_t, lse_t = r
        if self._lse_mode == "basee":
            lse_t.mul_(LN2)  # FA kernels emit base-2 (exp2 softmax); one fold
        return (out if out is not None else out_t), (lse if lse is not None else lse_t)


__all__ = ["_FaBackend"]
