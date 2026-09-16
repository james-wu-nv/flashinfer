"""Plan/run controller for the unified paged-prefill API.

Owns the public lifecycle behind ``flashinfer.prefill``:
validation, level-1 pinning against a ``Resolution``, level-2 choice within
it, derivation, and transactional publication of the planned backend. It does
not own any backend dialect (``_backends/``) or selection policy
(``_selection.py``).

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
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from ._backends import CAPABILITIES, make_backend
from ._contracts import (
    PagedAttentionMetadata,
    PlanMetadata,
    Resolution,
    _expect,
    _expect_lse_mode,
    _expect_window_left,
    resolve_config_key,
)
from ._graph import GraphBuffers, GraphCapacity, Transaction
from ._planning import Derived, validate_causal_envelope
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
# ---------------------------------------------------------------------------
_WORKSPACE_BYTES = 128 * 1024 * 1024  # the legacy wrappers' documented default
_shared_workspaces: Dict[torch.device, torch.Tensor] = {}
_shared_workspaces_lock = threading.Lock()


def _shared_workspace(device: torch.device) -> torch.Tensor:
    with _shared_workspaces_lock:
        ws = _shared_workspaces.get(device)
        if ws is None:
            ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
            _shared_workspaces[device] = ws
        return ws


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
        self._meta: Optional[PlanMetadata] = None
        self._derived: Optional[Derived] = None
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
            validate_causal_envelope(metadata.qo_indptr_cpu, metadata.kv_seq_lens_cpu)

        if isinstance(backend, Resolution):
            # Level-1 pinning (proposal §5.3): verify the plan config matches
            # what the engine resolved at init, then choose within the set.
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
            )
            _expect(
                backend.config == want,
                "plan() arguments do not match the pinned Resolution "
                f"(resolved {backend.config}, got {want}) — re-run "
                "resolve_paged_attention() with the new configuration",
            )
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
                backend=backend,
            )
        # Plan-time choice within the pinned set (level 2).  The prototype
        # takes the heuristic head; the autotune hook (proposal §5.4) would
        # consult its cache here, keyed on bucketed (total_q_tokens, max_kv_len).
        name = resolution.chosen
        needs_dense = CAPABILITIES[name].needs_dense

        if self._use_cuda_graph:
            _expect_not_capturing("plan()")
            # Graph mode: backends only ever see the reserved storage, so the
            # pointers a captured graph baked in stay valid across re-plans.
            # The buffers stay a local until publication below: a first plan
            # that fails in the backend must not leave a capacity behind.
            if self._graph is None:
                gb = GraphBuffers(GraphCapacity.from_metadata(metadata), self.device)
            else:
                gb = self._graph
            gb.preflight(metadata)
            cap = gb.capacity
            if needs_dense:
                # flat form: only a backend that reads the dense table pays
                # for it, and only once (no-op in the dense form)
                gb.reserve_dense_table()
            # Capacity substitution: the captured kernels were planned with
            # the capacity maxes and keep reading them, so backends see those
            # rather than this batch's own (validated <= in preflight); the
            # derived dense table is sized to the reserved one likewise.
            max_q_len, max_kv_len = cap.max_q_len, cap.max_kv_len
            fresh = metadata.derived(needs_dense=needs_dense, max_kv_len=cap.max_kv_len)
            transaction = Transaction(gb.targets(metadata, fresh))
            derived = gb.derived_view(needs_dense=needs_dense)
            qo_indptr, kv_seq_lens = gb.qo_indptr, gb.kv_seq_lens
            block_tables = (
                gb.block_tables
                if (needs_dense or metadata.block_tables is not None)
                else None
            )
        else:
            gb = None
            max_q_len, max_kv_len = metadata.max_q_len, metadata.max_kv_len
            fresh = metadata.derived(needs_dense=needs_dense)
            transaction = None
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
        )

        # graph mode: the frozen contract is checked before any reserved
        # buffer is written (the staging copies run inside the transaction)
        contract = _frozen_contract(name, meta) if gb is not None else None
        if self._frozen is not None:
            _check_frozen_contract(self._frozen, contract)

        key = (name, kv_layout)
        candidate = self._backends.get(key)
        if candidate is None:
            candidate = make_backend(
                name,
                self.device,
                kv_layout,
                self._scratch_workspace(),
                graph_capacity=gb.capacity if gb is not None else None,
            )
        if transaction is not None:
            # stage the new batch into reserved storage; any failure below
            # (including inside the backend's own plan) restores every buffer
            with transaction:
                candidate.plan(meta, derived)
                transaction.commit()
        else:
            candidate.plan(meta, derived)

        # publish — nothing above mutated the published state
        if gb is not None:
            self._graph = gb
            self._frozen = contract
            self._frozen_plan_kwargs = {
                k: v for k, v in contract.items() if k in _PLAN_PARAMS
            }
            self._frozen_plan_kwargs["backend"] = name
        self._backends[key] = candidate
        self._active = candidate
        self._backend_name = name
        self._resolution = resolution
        self._meta = meta
        self._derived = derived
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
            # graph mode: q/out/lse are the capture buffers, sized to the
            # capacity; rows past this batch are neither read nor written
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
        cap = CAPABILITIES[self._backend_name]
        if cap.requires_contiguous_q and not q.is_contiguous():
            raise ValueError(
                f"backend {self._backend_name!r} requires contiguous packed q "
                "(token-unit addressing assumes packed THD); call "
                ".contiguous() or pin a strided-capable backend (fa2/fa3)"
            )
        if out is not None:
            _expect(
                tuple(out.shape) == (q.shape[0], m.num_qo_heads, m.head_dim_vo)
                and out.is_contiguous(),
                "out must be contiguous (total_q_tokens, num_qo_heads, head_dim_vo)",
            )
            _expect(
                out.dtype == m.q_dtype and out.device == q.device,
                f"out must match q dtype/device ({m.q_dtype}, {q.device}), "
                f"got ({out.dtype}, {out.device}) — allocate with "
                "torch.empty(..., dtype=q.dtype, device=q.device)",
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
                and lse.is_contiguous()
                and lse.device == q.device,
                "lse must be contiguous fp32 (total_q_tokens, num_qo_heads) "
                f"on {q.device} — the LSE contract is packed fp32 in the planned "
                "base for every backend",
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
        return self._active.run(
            q,
            k_cache,
            v_cache,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def explain(self) -> str:
        _expect(self._planned, "explain() called before plan() — call plan() first")
        assert self._resolution is not None
        return f"chosen: {self._backend_name}\n{self._resolution.explain()}"


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


__all__ = ["PagedAttentionController"]
