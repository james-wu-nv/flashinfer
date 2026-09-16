"""Plan/run controller for the unified paged-prefill API.

Owns the public lifecycle behind ``flashinfer.prefill``:
validation, level-1 pinning against a ``Resolution``, level-2 choice within
it (a preference-ordered walk in which only the typed
``_BackendPlanUnsupportedError`` from a candidate's ``preflight()`` moves on to
the next candidate), derivation, and transactional publication of the planned
backend. It does not own any backend dialect (``_backends/``) or selection
policy (``_selection.py``).

Publication is transactional at this layer: ``plan()`` only swaps the
published metadata, derived forms, and active backend after the candidate
backend's own ``plan()`` returned, so a failed re-plan leaves the previous
plan runnable. The generated-FA backend re-plans its wrapper in place, so a
failure *inside* that wrapper's plan is the one path where backend-internal
state may already have moved — the same caveat the MLA design documents for
its generated backends.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ._backends import (
    CAPABILITIES,
    _BackendPlanUnsupportedError,
    derived_needs,
    make_backend,
    workspace_bound,
)
from ._contracts import (
    PagedAttentionMetadata,
    PlanMetadata,
    Resolution,
    _expect,
    _expect_custom_mask,
    _expect_lse_mode,
    _expect_pinned_device,
    _expect_window_left,
    _normalize_logits_soft_cap,
    resolve_config_key,
)
from ._graph import GraphBuffers, GraphCapacity, Transaction
from ._planning import FORM_BLOCK_TABLES, Derived
from ._selection import resolve_paged_attention

# ---------------------------------------------------------------------------
# Scratch workspace.  Every backend runs on plain scratch (the role of the
# legacy wrappers' caller-owned float_workspace_buffer; trtllm-gen carves its
# softmax-stats/scratch regions out of the same kind of buffer).  An engine
# holds one controller per CUDA-graph bucket, so the workspace is ONE lazily
# allocated pool per device shared by every instance, unless the caller passes
# its own (e.g. the buffer it already shares with legacy wrappers).  Sharing
# follows the legacy wrappers' rule: instances sharing a workspace must not
# run concurrently on different streams.
#
# Default size.  The fa2 split-KV planner is the only consumer whose need
# grows with the geometry (formulas in _backends/fa_backend.py); the legacy
# wrappers' documented 128 MiB overflows it on an ordinary prefill shape
# (ledger M14: one request, q = kv = 2048, 32 query / 8 KV heads, head_dim
# 128, graph mode needs 129 MiB).  512 MiB is the upper end of what the
# engines settled on for the same kernels (vLLM 394 MiB, SGLang 384 MiB and
# 512 MiB for Qwen2/3); on a B200 (148 SMs) it covers, for 32/8 heads at
# head_dim 128, eager prefill of any batch (75 MiB), decode graph buckets up
# to 1024 requests (322 MiB) and 4096-token graph prefill (258 MiB).  It is
# one allocation per device.  Anything larger -- an 8192-token single-request
# graph capture (516 MiB), MQA with 32 query heads (516 MiB eager) -- is a
# caller-owned buffer sized by workspace_requirements(); plan() rejects a
# buffer the planner would overflow before any kernel-side error.
# ---------------------------------------------------------------------------
_WORKSPACE_BYTES = 512 * 1024 * 1024
_shared_workspaces: Dict[torch.device, torch.Tensor] = {}
_shared_workspaces_lock = threading.Lock()


def _shared_workspace(device: torch.device) -> torch.Tensor:
    with _shared_workspaces_lock:
        ws = _shared_workspaces.get(device)
        if ws is None:
            ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            _shared_workspaces[device] = ws
        return ws


def workspace_requirements(
    capacity: GraphCapacity,
    *,
    device: Optional[torch.device] = None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: Optional[int] = None,
    q_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
    kv_layout: str = "HND",
    causal: bool = True,
    need_lse: bool = True,
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    custom_mask: bool = False,
    use_sinks: bool = False,
    use_cuda_graph: bool = True,
    backend: Union[str, Resolution] = "auto",
) -> int:
    """Conservative scratch-workspace bytes for every plan within ``capacity``.

    The maximum over the backends that resolve for the configuration (or the
    one named / the pinned ``Resolution``) of each backend's own bound
    (``_backends.workspace_bound``); an engine sizes ``workspace_buffer``
    with it before capture instead of discovering an overflow in a planner.
    Zero-sync and tensor-free; needs the target ``device`` for its SM count
    and opt-in shared memory (the default is the current CUDA device).
    """
    _expect(
        isinstance(capacity, GraphCapacity),
        "workspace_requirements() takes a GraphCapacity (the batch geometry: "
        f"batch size, total query tokens, maxes, page size), got {type(capacity).__name__}",
    )
    dev = torch.device(device) if device is not None else torch.device("cuda")
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    head_dim_vo = head_dim_vo if head_dim_vo is not None else head_dim_qk
    kv_dtype = kv_dtype if kv_dtype is not None else q_dtype
    if isinstance(backend, Resolution):
        candidates = backend.backends
    else:
        candidates = resolve_paged_attention(
            device=dev,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            page_size=capacity.page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=capacity.kv_input_form,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
            sinks=use_sinks,
            backend=backend,
        ).backends
    props = torch.cuda.get_device_properties(dev)
    geometry = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        kv_dtype=kv_dtype,
        page_size=capacity.page_size,
        batch_size=capacity.batch_size,
        total_q_tokens=capacity.total_q_tokens,
        max_q_len=capacity.max_q_len,
        max_kv_len=capacity.max_kv_len,
        need_lse=need_lse,
        window_left=window_left,
        use_cuda_graph=use_cuda_graph,
        sm_count=int(props.multi_processor_count),
        smem_optin=getattr(props, "shared_memory_per_block_optin", None),
    )
    return max(workspace_bound(name, **geometry) for name in candidates)


def _validate_workspace_buffer(buf: Any, device: torch.device) -> torch.Tensor:
    """Caller-supplied scratch workspace -> flat uint8 view on ``device``."""
    _expect(
        isinstance(buf, torch.Tensor),
        f"workspace_buffer must be a torch.Tensor, got {type(buf).__name__}",
    )
    _expect(
        buf.device == device,
        f"workspace_buffer lives on {buf.device} but this instance is bound to {device}",
    )
    _expect(
        buf.dim() == 1 and buf.is_contiguous(),
        "workspace_buffer must be a contiguous 1-D byte tensor, got shape "
        f"{tuple(buf.shape)}",
    )
    _expect(
        buf.dtype in (torch.uint8, torch.int8),
        f"workspace_buffer must be uint8/int8, got {buf.dtype}",
    )
    return buf.view(torch.uint8)


# ---------------------------------------------------------------------------
# Frozen contract of a captured graph.  A captured graph keeps launching the
# kernels the first graph-mode plan chose, with the kernel variant (causal,
# window, dtypes, head dims, LSE base, ...) baked in.  Every backend would
# accept a later plan() that changed one of those and the graph would silently
# replay stale results (mla-alignment F1: causal=True captured, causal=False
# re-planned and accepted).  So the first successful graph-mode plan freezes
# the chosen backend plus every PlanMetadata field that is not a per-batch
# value; a semantic kwarg added to PlanMetadata later is frozen automatically.
# ---------------------------------------------------------------------------
_PER_BATCH_FIELDS = frozenset(
    {
        "qo_indptr",
        "kv_seq_lens",
        "block_tables",
        # per-batch mask values (a tensor; graph mode rejects custom masks in
        # the fa preflight until the mask storage joins the capacity contract)
        "custom_mask",
        "qo_indptr_cpu",
        "kv_seq_lens_cpu",
        "batch_size",
        "max_q_len",
        "max_kv_len",
    }
)


def _semantic_fields() -> Tuple[str, ...]:
    """PlanMetadata fields a captured graph depends on: all but the per-batch values."""
    return tuple(
        f.name
        for f in dataclasses.fields(PlanMetadata)
        if f.name not in _PER_BATCH_FIELDS
    )


def _frozen_contract(backend: str, meta: PlanMetadata) -> Dict[str, Any]:
    contract: Dict[str, Any] = {
        "backend": backend,
        "dense_table": meta.block_tables is not None,
    }
    for name in _semantic_fields():
        contract[name] = getattr(meta, name)
    return contract


def _check_frozen_contract(frozen: Dict[str, Any], current: Dict[str, Any]) -> None:
    for key, want in frozen.items():
        got = current.get(key)
        if got != want:
            raise ValueError(
                f"CUDA graph re-plan: {key} changed from {want!r} (captured) to "
                f"{got!r}; the captured graph would keep launching the kernels "
                "planned for the old configuration — construct a new "
                "PagedAttention for the new configuration and recapture"
            )


class PagedAttentionController:
    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        graph_capacity: Optional[GraphCapacity] = None,
        use_cuda_graph: bool = False,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        dev = torch.device(device) if device is not None else torch.device("cuda")
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev
        self._backends: Dict[Any, Any] = {}
        # the caller's scratch buffer, or (lazily, on the first plan) the
        # per-device shared one — never a per-instance allocation
        self._workspace: Optional[torch.Tensor] = (
            _validate_workspace_buffer(workspace_buffer, dev)
            if workspace_buffer is not None
            else None
        )
        # CUDA-graph mode: reserved storage (_graph.py) sized by an explicit
        # GraphCapacity here, or by the first plan when only use_cuda_graph=True
        if graph_capacity is not None:
            _expect(
                isinstance(graph_capacity, GraphCapacity),
                "graph_capacity must be a GraphCapacity, got "
                f"{type(graph_capacity).__name__}",
            )
            use_cuda_graph = True
        self._use_cuda_graph = use_cuda_graph
        self._graph: Optional[GraphBuffers] = (
            GraphBuffers(graph_capacity, dev) if graph_capacity is not None else None
        )
        # what the first successful graph-mode plan froze (_frozen_contract),
        # and the plan() kwargs update() re-issues from it
        self._frozen: Optional[Dict[str, Any]] = None
        self._frozen_plan_kwargs: Optional[Dict[str, Any]] = None
        # published plan state (swapped together, only on success)
        self._planned = False
        self._backend_name: Optional[str] = None
        self._resolution: Optional[Resolution] = None
        # plan-time selection trace: (backend, phase, reason) per candidate tried
        self._trace: List[Tuple[str, str, str]] = []
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
        # flat form only: the page ids the plan was made from (reserved storage
        # in graph mode), independent of whether the chosen backend reads them
        self._kv_page_indices: Optional[torch.Tensor] = None
        self._active = None

    @property
    def backend(self) -> Optional[str]:
        """Name of the backend chosen by the last successful plan()."""
        return self._backend_name

    @property
    def resolution(self) -> Optional[Resolution]:
        return self._resolution

    def _scratch_workspace(self) -> torch.Tensor:
        if self._workspace is None:
            self._workspace = _shared_workspace(self.device)
        return self._workspace

    def _device_binding(self) -> Tuple[int, int, int]:
        """``(cc_major, cc_minor, device_index)`` of the device this instance runs on."""
        props = torch.cuda.get_device_properties(self.device)
        return int(props.major), int(props.minor), int(self.device.index)

    # ------------------------------ plan ------------------------------

    def plan(
        self,
        metadata: PagedAttentionMetadata,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: Optional[int] = None,
        q_dtype: torch.dtype,
        kv_dtype: Optional[torch.dtype] = None,
        kv_layout: str = "HND",
        causal: bool = True,
        window_left: int = -1,
        lse_mode: str = "none",
        logits_soft_cap: Optional[float] = None,
        custom_mask: Optional[torch.Tensor] = None,
        use_sinks: bool = False,
        backend: Union[str, Resolution] = "auto",
    ) -> None:
        _expect(
            isinstance(metadata, PagedAttentionMetadata),
            "metadata must be a PagedAttentionMetadata (build it with "
            ".dense(...) or .csr(...))",
        )
        _expect(
            metadata.device == self.device,
            f"metadata lives on {metadata.device} but this instance is bound to "
            f"{self.device}",
        )
        head_dim_vo = head_dim_vo if head_dim_vo is not None else head_dim_qk
        kv_dtype = kv_dtype if kv_dtype is not None else q_dtype
        _expect(
            kv_layout in ("HND", "NHD"),
            f"kv_layout must be 'HND' or 'NHD', got {kv_layout!r}",
        )
        kv_input_form = metadata.kv_input_form
        _expect_window_left(window_left)
        _expect_lse_mode(lse_mode)
        need_lse = lse_mode != "none"
        if causal:
            metadata.validate_causal_envelope()
        logits_soft_cap = _normalize_logits_soft_cap(logits_soft_cap)
        if custom_mask is not None:
            q_lens = metadata.qo_indptr_cpu.diff().to(torch.int64)
            _expect_custom_mask(
                custom_mask,
                self.device,
                int((q_lens * metadata.kv_seq_lens_cpu.to(torch.int64)).sum()),
            )
        use_sinks = bool(use_sinks)

        if isinstance(backend, Resolution):
            # Level-1 pinning (proposal §5.3): verify the plan config AND the
            # device binding match what the engine resolved at init, then
            # choose within the set.
            want = resolve_config_key(
                num_qo_heads,
                num_kv_heads,
                head_dim_qk,
                head_dim_vo,
                q_dtype,
                kv_dtype,
                metadata.page_size,
                kv_layout,
                causal,
                need_lse,
                window_left,
                kv_input_form,
                logits_soft_cap,
                custom_mask is not None,
                use_sinks,
                backend.max_q_len,  # the hint is the Resolution's own; checked per batch below
                *self._device_binding(),
            )
            _expect(
                backend.config[:-1] == want[:-1],
                "plan() arguments do not match the pinned Resolution "
                f"(resolved {backend.config[:-1]}, got {want[:-1]}) — re-run "
                "resolve_paged_attention() with the new configuration",
            )
            _expect_pinned_device(backend.device_binding, want[-1])
            resolution = backend
        else:
            resolution = resolve_paged_attention(
                device=self.device,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                page_size=metadata.page_size,
                kv_layout=kv_layout,
                causal=causal,
                need_lse=need_lse,
                window_left=window_left,
                kv_input_form=kv_input_form,
                logits_soft_cap=logits_soft_cap,
                custom_mask=custom_mask is not None,
                sinks=use_sinks,
                backend=backend,
            )
        # Plan-time choice within the pinned set (level 2): walk the
        # candidates in preference order.  A candidate may decline THIS batch
        # from its cheap preflight() with the typed unsupported signal, in
        # which case the next member is tried; every other exception (invalid
        # input, OOM, JIT failure) propagates unchanged, and an explicit
        # backend string never falls back.  The walk finishes before any
        # reserved-buffer write in graph mode.  The autotune hook (proposal
        # §5.4) would rank the survivors here, keyed on bucketed
        # (total_q_tokens, max_kv_len).
        explicit = isinstance(backend, str) and backend != "auto"
        gb: Optional[GraphBuffers] = None
        # no instance plans under capture: an eager plan would bake this
        # batch's uploads and a fresh backend plan into the caller's graph
        _expect_not_capturing("plan()")
        if self._use_cuda_graph:
            # Graph mode: backends only ever see the reserved storage, so the
            # pointers a captured graph baked in stay valid across re-plans.
            # The buffers stay a local until publication below: a first plan
            # that fails in the backend must not leave a capacity behind.
            if self._graph is None:
                gb = GraphBuffers(GraphCapacity.from_metadata(metadata), self.device)
            else:
                gb = self._graph
                stream = torch.cuda.current_stream(self.device)
                _expect(
                    gb.stream is None or stream == gb.stream,
                    "CUDA graph re-plan on a different stream than the first "
                    "graph-mode plan: the staging copies must be ordered before "
                    "the replay — plan()/update() and graph.replay() on one "
                    "stream (the stream of the first graph-mode plan)",
                )
            gb.preflight(metadata)
            # Capacity substitution: the captured kernels were planned with
            # the capacity maxes and keep reading them, so backends see those
            # rather than this batch's own (validated <= in preflight); the
            # derived dense table is sized to the reserved one likewise.
            max_q_len, max_kv_len = gb.capacity.max_q_len, gb.capacity.max_kv_len
        else:
            max_q_len, max_kv_len = metadata.max_q_len, metadata.max_kv_len
        if resolution.max_q_len is not None:
            # The Resolution's candidate order was chosen for batches with at
            # most ``max_q_len`` query tokens per request (the caller's hint);
            # a batch — in graph mode the capacity the kernels are planned
            # with — above it breaks that promise, so it is refused here,
            # before any reserved-storage write, rather than run in the
            # wrong order.
            _expect(
                max_q_len <= resolution.max_q_len,
                f"max_q_len {max_q_len} of this "
                f"{'graph capacity' if gb is not None else 'batch'} exceeds the "
                f"pinned Resolution's hint max_q_len={resolution.max_q_len} (its "
                "candidate order is for batches with at most that many query "
                "tokens per request) — resolve_paged_attention() without the "
                "hint, or with a larger one, for these batches",
            )

        trace: List[Tuple[str, str, str]] = []
        chosen = None
        frozen_backend = self._frozen["backend"] if self._frozen is not None else None
        if frozen_backend is not None and frozen_backend not in resolution.backends:
            _check_frozen_contract(
                {"backend": frozen_backend}, {"backend": resolution.backends[0]}
            )
        for name in resolution.backends:
            if frozen_backend is not None and name != frozen_backend:
                # a captured graph launches the frozen backend's kernels; skip
                # every other candidate before constructing or reserving
                # anything for it
                trace.append(
                    (name, "frozen", "not the backend the first graph-mode plan froze")
                )
                continue
            # derive exactly the forms this candidate declared it reads (cached
            # on the metadata per need set and width; in graph mode the dense
            # table derived from flat page ids is sized to the reserved one)
            needs = derived_needs(name)
            fresh = metadata.derived(
                needs=needs,
                max_kv_len=gb.capacity.max_kv_len if gb is not None else None,
            )
            if gb is not None:
                if FORM_BLOCK_TABLES in needs:
                    # flat form: only a backend that reads the dense table pays
                    # for it, and only once (no-op in the dense form)
                    gb.reserve_dense_table()
                derived = gb.derived_view(needs=needs, fresh=fresh)
                qo_indptr, kv_seq_lens = gb.qo_indptr, gb.kv_seq_lens
                block_tables = (
                    gb.block_tables
                    if (FORM_BLOCK_TABLES in needs or metadata.block_tables is not None)
                    else None
                )
            else:
                derived = fresh
                qo_indptr, kv_seq_lens = metadata.qo_indptr, metadata.kv_seq_lens
                # dense table given by the caller or derived (None where truly absent)
                block_tables = (
                    metadata.block_tables
                    if metadata.block_tables is not None
                    else derived.block_tables
                )
            meta = PlanMetadata(
                qo_indptr=qo_indptr,
                kv_seq_lens=kv_seq_lens,
                block_tables=block_tables,
                kv_input_form=kv_input_form,
                page_size=metadata.page_size,
                max_q_len=max_q_len,
                max_kv_len=max_kv_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                causal=causal,
                window_left=window_left,
                kv_layout=kv_layout,
                lse_mode=lse_mode,
                batch_size=metadata.batch_size,
                qo_indptr_cpu=metadata.qo_indptr_cpu,
                kv_seq_lens_cpu=metadata.kv_seq_lens_cpu,
                logits_soft_cap=logits_soft_cap,
                custom_mask=custom_mask,
                use_sinks=use_sinks,
            )
            key = (name, kv_layout)
            candidate = self._backends.get(key)
            if candidate is None:
                # construction only (no plan state); cached so a candidate
                # that keeps declining is not rebuilt on every plan — except
                # while a first graph-mode plan's capacity is still unpublished:
                # a backend built against it (reserved CSR buffers, row budget)
                # must go with the capacity if that plan fails, so it is cached
                # at publication below instead
                candidate = make_backend(
                    name,
                    self.device,
                    kv_layout,
                    self._scratch_workspace(),
                    graph_capacity=gb.capacity if gb is not None else None,
                )
                if gb is None or self._graph is not None:
                    self._backends[key] = candidate
            try:
                candidate.preflight(meta)
            except _BackendPlanUnsupportedError as exc:
                trace.append((name, "preflight", str(exc)))
                if explicit:
                    raise ValueError(
                        f"backend {name!r} cannot plan this batch: {exc}"
                    ) from exc
                continue
            trace.append((name, "preflight", "accepted"))
            chosen = (name, candidate, meta, derived, fresh)
            break
        if chosen is None:
            detail = "; ".join(f"{n}: {r}" for n, _, r in trace)
            raise ValueError(
                "no pinned candidate can plan this batch "
                f"(candidates {list(resolution.backends)}: {detail})"
            )
        name, candidate, meta, derived, fresh = chosen
        key = (name, kv_layout)

        # graph mode: the frozen contract is checked before any reserved
        # buffer is written (the staging copies run inside the transaction)
        contract = _frozen_contract(name, meta) if gb is not None else None
        if self._frozen is not None:
            _check_frozen_contract(self._frozen, contract)

        if gb is not None:
            # stage the new batch into reserved storage; any failure below
            # (including inside the backend's own plan) restores every buffer
            transaction = Transaction(gb.targets(metadata, fresh))
            with transaction:
                candidate.plan(meta, derived)
                transaction.commit()
        else:
            candidate.plan(meta, derived)

        # publish — nothing above mutated the published state (the backend
        # cache holds constructed objects, not plan state)
        if gb is not None:
            if gb.stream is None:
                gb.stream = torch.cuda.current_stream(self.device)
            self._graph = gb
            self._frozen = contract
            self._frozen_plan_kwargs = {
                k: v for k, v in contract.items() if k in _PLAN_PARAMS
            }
            # update() re-plans within the pinned resolution narrowed to the
            # frozen backend: no level-1 re-resolution per step, and explain()
            # keeps the resolve-time exclusion reasons
            self._frozen_plan_kwargs["backend"] = dataclasses.replace(
                resolution,
                backends=(name,),
                excluded={
                    **resolution.excluded,
                    **{
                        other: "not the backend the first graph-mode plan froze"
                        for other in resolution.backends
                        if other != name
                    },
                },
            )
        self._backends[key] = candidate
        self._active = candidate
        self._backend_name = name
        self._resolution = resolution
        self._trace = trace
        self._meta = meta
        self._derived = derived
        if metadata.kv_page_indices is not None:
            self._kv_page_indices = (
                gb.kv_page_indices if gb is not None else metadata.kv_page_indices
            )
        self._planned = True

    # ----------------------------- update -----------------------------

    def update(self, metadata: PagedAttentionMetadata) -> None:
        """Graph-mode re-plan with the frozen semantic kwargs.

        Routes through :meth:`plan` with the kwargs the first graph-mode plan
        froze (backend included), so it stages into the reserved storage,
        preflights against the capacity and rolls back on failure exactly as
        a re-plan does; there is no second staging path.
        """
        _expect_not_capturing("update()")
        if not self._use_cuda_graph:
            raise RuntimeError(
                "update() is the CUDA-graph re-plan: construct "
                "PagedAttention(graph_capacity=...) or PagedAttention("
                "use_cuda_graph=True); eager instances re-plan with plan()"
            )
        if self._frozen_plan_kwargs is None:
            raise RuntimeError(
                "update() called before a successful plan(): the first plan() "
                "freezes the backend and the semantic kwargs update() reuses"
            )
        self.plan(metadata, **self._frozen_plan_kwargs)

    # ------------------------------ run -------------------------------

    def run(
        self,
        q: torch.Tensor,
        kv_cache: Sequence[torch.Tensor],
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _expect(self._planned, "run() called before plan() — call plan() first")
        m = self._meta
        assert m is not None and self._active is not None
        _expect(
            isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2,
            "kv_cache must be a (k_cache, v_cache) pair of paged tensors "
            "(pages, num_kv_heads, page_size, head_dim) [HND]",
        )
        k_cache, v_cache = kv_cache
        # Device placement: every backend launches on self.device and takes
        # raw pointers; a tensor elsewhere is a fault or a silent misread.
        for name, t in (
            ("q", q),
            ("k_cache", k_cache),
            ("v_cache", v_cache),
            ("out", out),
            ("lse", lse),
        ):
            _expect(
                t is None or t.device == self.device,
                f"{name} lives on {getattr(t, 'device', None)} but this instance "
                f"is bound to {self.device}",
            )
        layout = m.kv_layout
        h_pos, ps_pos = (1, 2) if layout == "HND" else (2, 1)
        shape_word = (
            "(pages, H, page_size, D)"
            if layout == "HND"
            else "(pages, page_size, H, D)"
        )
        for name, t, dim_ in (
            ("k_cache", k_cache, m.head_dim_qk),
            ("v_cache", v_cache, m.head_dim_vo),
        ):
            _expect(t.dim() == 4, f"{name} must be 4-D paged {shape_word} [{layout}]")
            swapped_hint = (
                f" — dims 1/2 look transposed: the plan declared kv_layout="
                f"{layout!r} {shape_word}; permute(0, 2, 1, 3).contiguous() or "
                "re-plan with the other kv_layout"
                if (
                    t.shape[ps_pos] == m.num_kv_heads
                    and t.shape[h_pos] == m.page_size
                    and m.num_kv_heads != m.page_size
                )
                else ""
            )
            _expect(
                t.shape[h_pos] == m.num_kv_heads
                and t.shape[ps_pos] == m.page_size
                and t.shape[3] == dim_,
                f"{name} shape {tuple(t.shape)} does not match plan "
                f"(kv_layout={layout}, H={m.num_kv_heads}, "
                f"page_size={m.page_size}, D={dim_})"
                f"{swapped_hint}",
            )
            _expect(
                t.dtype == m.kv_dtype,
                f"{name} dtype {t.dtype} != planned {m.kv_dtype}",
            )
            # Same ABI as q below: the bindings pass page/head/token strides
            # only, so a non-unit head_dim stride would be silently misread.
            _expect(
                t.stride(-1) == 1,
                f"{name} must be dense along head_dim (stride(-1) == 1), got "
                f"strides {tuple(t.stride())} — the kernels address the KV pool "
                "by page/head/token stride only; pass a view whose last dim is "
                "unit-stride or a packed copy",
            )
        _expect(
            q.dim() == 3, "q must be packed (total_q_tokens, num_qo_heads, head_dim)"
        )
        _expect(
            q.shape[1] == m.num_qo_heads and q.shape[2] == m.head_dim_qk,
            f"q shape {tuple(q.shape)} does not match plan "
            f"(H={m.num_qo_heads}, D={m.head_dim_qk})",
        )
        _expect(q.dtype == m.q_dtype, f"q dtype {q.dtype} != planned {m.q_dtype}")
        total = m.total_q_tokens
        if self._graph is not None:
            # graph mode: q/out/lse are the capture buffers, at most the
            # capacity's rows; the smallest row count seen bounds later
            # batches (GraphBuffers.preflight), since a graph captured on
            # these buffers cannot reach past them
            rows = self._graph.capacity.total_q_tokens
            _expect(
                total <= q.shape[0] <= rows,
                f"q has {q.shape[0]} tokens; a graph-mode instance needs this "
                f"batch's {total} tokens and at most its capacity of {rows}",
            )
        else:
            _expect(
                q.shape[0] == total,
                f"q has {q.shape[0]} tokens but qo_indptr sums to {total}",
            )
        # Query ABI shared by every backend: the FA2/FA3 and trtllm-gen bindings
        # pass only the token and head strides (the kernels assume a dense
        # head_dim), so a non-unit inner stride is silently misread (native
        # probe: max abs error 1.5 / 1.3 against 0.0025); cuDNN rejects it in
        # graph construction.  See tests/experimental/test_paged_attention_strides.py.
        _expect(
            q.stride(-1) == 1,
            f"q must be dense along head_dim (stride(-1) == 1), got strides "
            f"{tuple(q.stride())} — every backend addresses q by token/head "
            "stride only; pass a view whose last dim is unit-stride (a head "
            "slice of a fused QKV buffer is fine) or a packed copy",
        )
        cap = CAPABILITIES[self._backend_name]
        if cap.requires_contiguous_q and not q.is_contiguous():
            strided = [
                n for n, c in CAPABILITIES.items() if not c.requires_contiguous_q
            ]
            raise ValueError(
                f"backend {self._backend_name!r} requires packed q, got strides "
                f"{tuple(q.stride())}: its kernel reads q as a packed (T, H, D) "
                "array, so a head slice of a fused QKV buffer (token stride > H*D) "
                "is misaddressed — pass a packed copy or pin a strided-capable "
                f"backend ({'/'.join(strided)})"
            )
        if out is not None:
            _expect(
                tuple(out.shape) == (q.shape[0], m.num_qo_heads, m.head_dim_vo)
                and out.is_contiguous(),
                "out must be contiguous (total_q_tokens, num_qo_heads, head_dim_vo)",
            )
            _expect(
                out.dtype == m.q_dtype,
                f"out must match q dtype {m.q_dtype}, got {out.dtype} — "
                "allocate with torch.empty(..., dtype=q.dtype, device=q.device)",
            )
        if lse is not None:
            _expect(
                m.need_lse,
                "lse= buffer passed but the plan has lse_mode='none' — "
                "plan(lse_mode='base2' or 'basee') or drop the lse= argument",
            )
            _expect(
                tuple(lse.shape) == (q.shape[0], m.num_qo_heads)
                and lse.dtype == torch.float32
                and lse.is_contiguous(),
                "lse must be contiguous fp32 (total_q_tokens, num_qo_heads) — "
                "the LSE contract is packed fp32 in the planned base for every "
                "backend",
            )
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(m.head_dim_qk)
        _expect(
            isinstance(sm_scale, float) and math.isfinite(sm_scale) and sm_scale > 0,
            f"sm_scale must be a positive finite host float, got {sm_scale!r}",
        )
        kv_is_fp8 = m.kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        for nm, sc in (("k_scale", k_scale), ("v_scale", v_scale)):
            if sc is None:
                continue
            _expect(
                kv_is_fp8,
                f"{nm} given but the plan's kv_dtype is {m.kv_dtype}; per-tensor "
                "KV scales apply to fp8 KV caches only",
            )
            _expect(
                isinstance(sc, float) and math.isfinite(sc) and sc > 0,
                f"{nm} must be a positive finite host float (dequant = fp8 * scale), "
                f"got {sc!r}",
            )
        if m.use_sinks:
            _expect(
                sinks is not None,
                "the plan declared use_sinks=True but run() got no sinks= tensor "
                "(the sink-aware kernel variant needs one per call)",
            )
            _expect(
                isinstance(sinks, torch.Tensor)
                and tuple(sinks.shape) == (m.num_qo_heads,)
                and sinks.dtype == torch.float32
                and sinks.device == q.device
                and sinks.is_contiguous(),
                f"sinks must be a contiguous fp32 ({m.num_qo_heads},) tensor on "
                f"{q.device} (one extra softmax logit per query head), got "
                + (
                    f"shape {tuple(sinks.shape)}, {sinks.dtype}, {sinks.device}"
                    if isinstance(sinks, torch.Tensor)
                    else type(sinks).__name__
                ),
            )
        else:
            _expect(
                sinks is None,
                "sinks= passed but the plan did not declare use_sinks=True — "
                "plan(use_sinks=True) selects the sink-aware kernel variant",
            )
        if self._graph is not None:
            gb = self._graph
            gb.rows = q.shape[0] if gb.rows is None else min(gb.rows, q.shape[0])
        return self._active.run(
            q,
            k_cache,
            v_cache,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sinks=sinks,
        )

    @property
    def selection_trace(self) -> Tuple[Tuple[str, str, str], ...]:
        """``(backend, phase, reason)`` for every candidate the last plan() tried."""
        return tuple(self._trace)

    def explain(self) -> str:
        _expect(self._planned, "explain() called before plan() — call plan() first")
        assert self._resolution is not None
        lines = [f"chosen: {self._backend_name}", "plan-time trace:"]
        for name, phase, reason in self._trace:
            lines.append(f"  {name} [{phase}]: {reason}")
        lines.append(self._resolution.explain())
        return "\n".join(lines)

    # --------------------------- read-only trace -----------------------

    def trace_context(self) -> Dict[str, Any]:
        """Facts of the LAST SUCCESSFUL plan that a trace (``fi_trace``) needs.

        Read-only: no device-to-host copy, no kernel launch, no change to the
        selection or the published plan.  The tensors are the ones the planned
        kernels read — in CUDA-graph mode the reserved storage — so a trace
        taken between re-plans describes what a replay computes, and a failed
        re-plan leaves the previous successful plan visible here.

        Exactly one paging form is present, matching the plan's input form:
        ``block_tables`` for the dense form (the caller's table, or its copy
        in reserved storage), ``kv_page_indices`` for the flat form (the live
        prefix of the caller's flat page-id list — or its reserved copy — its
        length computed from the host mirrors; this is the plan's source even
        when the chosen backend reads a dense table derived from it).
        ``logits_soft_cap`` / ``has_custom_mask`` / ``use_sinks`` are the
        plan's feature knobs.  ``backend`` / ``excluded_backends`` /
        ``graph_capacity`` are provenance, not part of a trace's mathematical
        identity.
        """
        _expect(
            self._planned,
            "trace_context() called before plan() — trace a planned instance "
            "(call plan() first)",
        )
        m, res = self._meta, self._resolution
        assert m is not None and res is not None
        page = m.page_size
        block_tables = kv_page_indices = None
        if m.kv_input_form == "block_tables":
            block_tables = m.block_tables
        else:
            assert self._kv_page_indices is not None
            live = int(torch.sum((m.kv_seq_lens_cpu + page - 1) // page))
            kv_page_indices = self._kv_page_indices[:live]
        return {
            "qo_indptr": m.qo_indptr,
            "kv_seq_lens": m.kv_seq_lens,
            "block_tables": block_tables,
            "kv_page_indices": kv_page_indices,
            "kv_input_form": m.kv_input_form,
            "page_size": page,
            "kv_layout": m.kv_layout,
            "num_qo_heads": m.num_qo_heads,
            "num_kv_heads": m.num_kv_heads,
            "head_dim_qk": m.head_dim_qk,
            "head_dim_vo": m.head_dim_vo,
            "q_dtype": m.q_dtype,
            "kv_dtype": m.kv_dtype,
            "causal": m.causal,
            "window_left": m.window_left,
            "lse_mode": m.lse_mode,
            "logits_soft_cap": m.logits_soft_cap,
            "has_custom_mask": m.custom_mask is not None,
            "use_sinks": m.use_sinks,
            "max_q_len": m.max_q_len,
            "max_kv_len": m.max_kv_len,
            "batch_size": m.batch_size,
            "total_q_tokens": m.total_q_tokens,
            "qo_indptr_cpu": m.qo_indptr_cpu,
            "kv_seq_lens_cpu": m.kv_seq_lens_cpu,
            # provenance — not identity
            "backend": self._backend_name,
            "excluded_backends": dict(res.excluded),
            "graph_capacity": self._graph.capacity if self._graph is not None else None,
        }


# plan() keyword arguments update() re-issues from the frozen contract; the
# remaining frozen keys (kv_input_form, page_size, dense_table) come from the
# metadata and the chosen backend and are checked, not passed
_PLAN_PARAMS = frozenset(
    inspect.signature(PagedAttentionController.plan).parameters
) - {"self", "metadata"}


def _expect_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} cannot run during CUDA graph capture: plan or update before "
            "capture, replay inside it"
        )


__all__ = ["PagedAttentionController", "workspace_requirements"]
