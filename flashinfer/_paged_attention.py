"""Paged attention (PagedAttention) — EXPERIMENTAL public entry point.

Working prototype of ``PAGED_PREFILL_UNIFICATION_PROPOSAL.md``. This module is
the thin core entry point required by ``flashinfer/experimental/README.md``:
public signatures, the experimental marker, and a deferred handoff to
``flashinfer.experimental.paged_attention``, which owns contracts, planning,
selection, and the per-backend modules. Everything dispatches to *existing*
kernels — there is no new kernel.

Layers (see proposal §"Architecture" and the MLA precedent in
``docs/design_docs/batch_mla_backend_architecture.md``):

    entry       this module (exported from flashinfer.prefill):
                resolve_paged_attention() + PagedAttention + PagedAttentionMetadata
    controller  experimental.paged_attention._controller  (plan/run lifecycle)
    backends    experimental.paged_attention._backends    (fa2/fa3, cudnn, trtllm-gen, cake)
    kernels     the existing wrapper / standalone functions (untouched)

Design rules enforced (each traces to a documented failure mode):

1.  ONE canonical metadata form — token-unit ``qo_indptr``, per-request
    ``kv_seq_lens``, a dense ``block_tables`` or flat ``kv_page_indices``, and
    REQUIRED host maxes.  Everything any backend wants (CSR page indices for
    fa2/fa3, cumulative KV lens for trtllm-gen, ``(b,1,1,1)`` lens for cuDNN)
    is derived internally.
2.  Closed input set + loud errors.  Combinations outside the contract raise
    ``ValueError`` with the fix in the message.  We never guess a layout
    (cuDNN issue 3800 is what guessing looks like).
3.  Reject-or-correct.  Anything this API returns must match the reference
    semantics; anything it cannot address must raise.  The companion fuzzer
    (``tests/experimental/test_paged_attention_fuzzer.py``) enforces exactly
    this property with randomized valid and corrupted inputs.
4.  Two-level selection.  ``resolve_paged_attention()`` is a static, tensor-free
    query usable at engine init (before pool allocation / graph capture);
    passing the returned ``Resolution`` to ``plan(backend=...)`` pins the
    candidate set — plan() verifies the config matches and may only choose
    within it.
5.  One output contract.  LSE base is declared at plan time (``lse_mode`` =
    ``"none"`` / ``"base2"`` / ``"basee"``, the Batch MLA vocabulary), shape
    ``(total_q_tokens, num_qo_heads)``, fp32 — backends deliver the declared
    base natively where free (cuDNN: natural log) and with one fold otherwise;
    padded native stats are gathered inside the backend, not by callers.
6.  Layer constants are run-time.  ``sm_scale`` is a ``run()`` argument, so one
    plan serves layers with different scales; ``window_left`` stays plan-time
    because it selects a compiled kernel variant on the FA backends.

Prototype simplifications (documented, not hidden):
- dtypes: fp16/bf16 activations; an fp8 (e4m3 or e5m2) KV cache with
  per-tensor ``k_scale`` / ``v_scale`` at ``run()`` where the capability
  table declares it (fa2).  The same two scales multiply a fp16/bf16 KV
  cache on every backend (folded into the softmax scale / the output).
  fp8 Q and nvfp4 are undeclared axes.
- Heuristic order is a static per-arch table bucketed by the optional
  ``max_q_len`` hint of ``resolve_paged_attention`` (seeded from the B200
  sweep in the WP-K report; proposal §5.2).  Autotune hook (§5.4) is not
  wired: ``auto`` never times anything.
- ``logits_soft_cap`` / ``custom_mask`` / ``use_sinks`` are explicit
  capability axes (fa2: all three; fa3: soft cap + sinks; trtllm-gen and
  cake: sinks; cuDNN: none) — a backend lacking a requested feature is excluded, never
  silently bypassed.  Custom masks under CUDA-graph mode are follow-up work.

CUDA-graph lifecycle — three stages (``experimental/paged_attention/_graph.py``):

1.  Construct.  ``PagedAttention(device, graph_capacity=GraphCapacity(...))``
    reserves the metadata storage a captured ``run()`` reads, sized by the
    capacity (batch size, total query tokens, host maxes, page size, block
    table width or flat page-id capacity) — one instance per graph bucket.
    ``use_cuda_graph=True`` is the compatibility form: the capacity is
    inferred from the first plan.  A first plan that fails installs nothing.
2.  First ``plan()`` freezes.  The first successful graph-mode plan fixes the
    backend and every semantic ``plan()`` argument (causal, window, LSE base,
    dtypes, heads, head dims, layout, paging form, page size): the captured
    kernels bake them in.  A later ``plan()`` that changes one is rejected
    before anything is written — construct a new instance and recapture.
3.  ``update(metadata)`` per step.  Only the batch changes.  It is staged
    into the reserved storage (rolled back on failure) and must fit the
    capacity: batch size, paging form and page size exact; total query
    tokens, host maxes and flat page-id length at most the capacity's.
    Capacity substitution: backends are planned with, and the captured
    kernels keep reading, the CAPACITY ``max_q_len`` / ``max_kv_len``; a
    batch's own maxes are only validated against them.  ``q`` / ``out`` /
    ``lse`` are the capture buffers, at most the capacity's rows; the
    smallest row count a ``run()`` sees bounds every later batch (a graph
    captured on those buffers cannot reach past them).  Rows past the batch
    are not read; ``out`` / ``lse`` rows past it are unspecified and may be
    written (a backend's whole-buffer post-processing, cuDNN's base
    conversion, trtllm-gen's ``-inf`` fill).  ``plan()`` and ``update()`` are
    rejected while the current stream is capturing, and after the first
    graph-mode plan they must run on that plan's stream — the stream the
    graph is replayed on, so the staging copies precede the replay.
    ``sm_scale`` /
    ``k_scale`` / ``v_scale`` are launch scalars: a captured graph keeps the
    values it was captured with.

Paging metadata comes in exactly one of two forms (never both):
- dense ``block_tables (b, max_pages)`` — vLLM-native; page_size >= 8
  (denser would blow the table up; token-CSR engines use the other form);
- flat ``kv_page_indices`` — sglang-style token/CSR-native, any page_size
  >= 1.  The page-unit indptr and last-page lengths are NOT accepted: they
  are derivable from ``kv_seq_lens`` + ``page_size``, and accepting them
  would create a second truth (see the PR 3921 mask-divergence class).
  Backends that need the dense table (cudnn, trtllm-gen, cake) get it
  derived by a zero-sync gather when page_size >= 8, and are
  capability-excluded below that.
- Trusted inputs (documented, not validated): host mirrors must match the
  device tensors; ``block_tables`` VALUES (page ids) must be in-pool —
  checking them costs a device-side pass the hot path cannot pay; a debug
  mode (FLASHINFER_VALIDATE_INPUTS-style) is the production answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import torch

from .api_logging import flashinfer_experimental_api
from .trace.templates.paged_attention import paged_attention_trace_dispatch

if TYPE_CHECKING:  # pragma: no cover — types only; the package is imported lazily
    from .experimental.paged_attention import (
        GraphCapacity,
        PagedAttentionMetadata,
        Resolution,
    )

__all__ = [
    "resolve_paged_attention",
    "PagedAttention",
]

_FEATURE = "PagedAttention"

# Value types users receive from / hand back to this API. Resolved lazily so
# that importing core never loads the experimental package.
_LAZY_EXPORTS = {
    "PagedAttentionMetadata": "PagedAttentionMetadata",
    "GraphCapacity": "GraphCapacity",
    "Resolution": "Resolution",
    "PagedAttentionCapabilities": "PagedAttentionCapabilities",
    "BackendCapability": "PagedAttentionCapabilities",  # pre-rename alias
    "CAPABILITIES": "CAPABILITIES",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        from .experimental import paged_attention

        value = getattr(paged_attention, _LAZY_EXPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


@flashinfer_experimental_api(feature=_FEATURE)
def resolve_paged_attention(
    *,
    device: Optional[torch.device] = None,
    cc_major: Optional[int] = None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: Optional[int] = None,
    q_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
    page_size: int,
    kv_layout: str = "HND",
    causal: bool = True,
    need_lse: bool = False,
    window_left: int = -1,
    kv_input_form: str = "block_tables",
    logits_soft_cap: Optional[float] = None,
    custom_mask: bool = False,
    sinks: bool = False,
    max_q_len: Optional[int] = None,
    backend: str = "auto",
) -> "Resolution":
    """Static backend resolution — no plan state, no tensors.

    Callable at engine init, before the KV pool is allocated and before any
    CUDA graph is captured (vLLM decides its cudagraph mode and Q dtype at
    that point).  Returns an ordered candidate set plus a reason for every
    excluded backend; raises ``ValueError`` when nothing can run.  Pass the
    result to :meth:`PagedAttention.plan` as ``backend=`` to pin the
    candidate set.

    - ``device`` / ``cc_major``: the Resolution is pinned to ``device`` (its
      full compute capability and index; the default is the current CUDA
      device).  With only ``cc_major`` it is pinned to that major and to no
      device, so it can be prepared off-device.
    - ``logits_soft_cap`` (``cap * tanh(score / cap)``; ``None``/``0`` = off),
      ``custom_mask`` and ``sinks``: the features the plans will request.  A
      backend that cannot apply a requested feature is excluded with the
      reason — ``auto`` never drops a feature silently.
    - ``max_q_len``: optional shape hint — the batches planned with this
      Resolution have at most this many query tokens per request (1 for
      plain decode, the draft length for speculative decode; leave ``None``
      for prefill or mixed batches).  It picks the ``auto`` candidate ORDER
      from a small per-architecture table seeded from measurements, never
      the candidate set: on sm_100 (B200) the paged context kernels behind
      trtllm-gen / cake are 1.0-9.5x slower than fa2 at ``max_q_len <= 16``
      (B=32, kv=4096, q=1: 513 vs 92 us; B=1, kv=16384, q=1: 209 vs 22 us)
      and ahead from 64-256 query tokens per request depending on kv_len,
      so hints ``<= 16`` put fa2 first and larger hints (or none) keep
      today's order.  ``auto`` remains a static selection made
      here, not autotuning: nothing is timed and ``plan()`` never reorders.
      The hint is pinned in the Resolution (``explain()`` shows it, and two
      Resolutions of one model configuration with different hints have
      different ``config`` keys), and a ``plan()`` whose batch — in
      CUDA-graph mode, whose capacity — has a larger ``max_q_len`` is
      rejected with a ``ValueError``; hold one Resolution per query-length
      bucket, as vLLM's decode / prefill split does.
    """
    from .experimental.paged_attention import resolve_paged_attention as _resolve

    return _resolve(
        device=device,
        cc_major=cc_major,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=need_lse,
        window_left=window_left,
        kv_input_form=kv_input_form,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        sinks=sinks,
        max_q_len=max_q_len,
        backend=backend,
    )


class PagedAttention:
    """Paged attention over the existing fa2/fa3, cuDNN, trtllm-gen and cake
    kernels behind one contract (experimental).

    Usage (engine-shaped; see ``prototype_demo_paged_attention.py``)::

        res = resolve_paged_attention(cc_major=9, num_qo_heads=8, num_kv_heads=2,
                                      head_dim_qk=128, q_dtype=torch.bfloat16,
                                      page_size=16, need_lse=True)
        # ... engine allocates its KV pool in res.kv_layout, sets dtypes ...
        attn = PagedAttention(device)
        for step in engine:
            md = PagedAttentionMetadata.dense(qo_indptr, kv_seq_lens, block_tables,
                                              page_size=16, max_q_len=..., max_kv_len=...,
                                              qo_indptr_cpu=..., kv_seq_lens_cpu=...)
            attn.plan(md, num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
                      q_dtype=torch.bfloat16, causal=True, lse_mode="base2",
                      backend=res)          # pinned candidate set (or a string)
            for layer in model:
                out, lse = attn.run(q, (k_cache, v_cache), sm_scale=layer.scale)

    One CUDA-graph bucket (the module docstring describes the lifecycle; the
    ``Resolution`` was resolved with ``need_lse=True``, so the plan must keep
    ``lse_mode != "none"``, which is part of the pinned config)::

        cap = GraphCapacity(batch_size=256, total_q_tokens=1024, max_q_len=4,
                            max_kv_len=8192 * 16, page_size=16, table_width=8192)
        nbytes = PagedAttention.workspace_requirements(
            cap, device=device, num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
            q_dtype=torch.bfloat16)          # scratch every plan in cap fits
        attn = PagedAttention(device, graph_capacity=cap,
                              workspace_buffer=torch.empty(nbytes, dtype=torch.uint8,
                                                           device=device))
        attn.plan(md0, num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
                  q_dtype=torch.bfloat16, causal=True, lse_mode="base2",
                  backend=res)                           # freezes backend + semantics
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            attn.run(q_buf, (k_cache, v_cache), out=out_buf, lse=lse_buf)
        for step in engine:
            attn.update(md_step)   # only the batch changes; must fit ``cap``
            g.replay()
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        graph_capacity: Optional["GraphCapacity"] = None,
        use_cuda_graph: bool = False,
        workspace_buffer: Optional[torch.Tensor] = None,
    ):
        """
        - ``graph_capacity``: a :class:`GraphCapacity` — the capture shapes of
          one CUDA-graph bucket (batch size, total query tokens, host maxes,
          page size, block-table width or flat page-id capacity).  The
          reserved metadata storage is allocated here, sized by it, so a
          captured ``run()`` can be re-planned and replayed; a plan that does
          not fit the capacity is rejected before anything is written, and a
          plan that fails midway restores the previous plan's buffers.  In
          the flat ``kv_page_indices`` form the dense block table is reserved
          only if the chosen backend needs it (never at ``page_size < 8``).
          Use one instance per graph bucket; this is the recommended form.
        - ``use_cuda_graph``: graph mode with the capacity inferred from the
          FIRST plan (compatibility form of ``graph_capacity``).
        - ``workspace_buffer``: optional caller-owned scratch workspace
          (contiguous 1-D uint8 or int8 on ``device``) that every backend's
          kernels run on — pass the buffer the engine already shares with its
          legacy wrappers, sized with :meth:`workspace_requirements` for the
          largest geometry it will plan.  By default every instance on a
          device shares one lazily allocated library-owned 512 MiB pool (the
          upper end of the engines' own defaults for these kernels), so
          holding one instance per graph bucket costs no workspace per
          bucket.  ``plan()`` raises ``ValueError`` naming the required bytes
          when a batch's planner allocation would not fit the buffer in use.
          Instances sharing a workspace must not run concurrently on
          different streams; pass a private buffer where that isolation is
          needed.
        """
        from .experimental.paged_attention import PagedAttentionController

        self._impl = PagedAttentionController(
            device,
            graph_capacity=graph_capacity,
            use_cuda_graph=use_cuda_graph,
            workspace_buffer=workspace_buffer,
        )

    @property
    def device(self) -> torch.device:
        return self._impl.device

    @property
    def backend(self) -> Optional[str]:
        """Name of the backend chosen by the last successful ``plan()``."""
        return self._impl.backend

    @staticmethod
    @flashinfer_experimental_api(feature=_FEATURE)
    def workspace_requirements(
        capacity: "GraphCapacity",
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
        backend: Union[str, "Resolution"] = "auto",
    ) -> int:
        """Scratch-workspace bytes that cover every plan within ``capacity``.

        A conservative upper bound on what the planners of the backends that
        can run this configuration carve out of ``workspace_buffer``, for an
        engine to size a caller-owned buffer (or deduct from its KV-pool
        budget) at init, before any capture.  Tensor-free and sync-free.

        - ``capacity``: the batch geometry as a :class:`GraphCapacity` (batch
          size, total query tokens, ``max_q_len`` / ``max_kv_len``, page size,
          paging form) — the bucket an instance will be built with, or the
          largest batch an eager instance will plan.
        - ``device``: the target device; its SM count bounds the split-KV
          work items (default: the current CUDA device).
        - model configuration and feature flags as for :meth:`plan`
          (``need_lse`` defaults to ``True`` so the bound covers plans with
          and without an LSE output).
        - ``use_cuda_graph``: ``True`` (default) bounds a graph-mode instance
          planned with ``capacity``, which also covers any eager plan that
          fits it; ``False`` gives the tighter eager-only bound, which does
          not depend on the token counts.
        - ``backend``: ``"auto"`` takes the maximum over every backend that
          resolves for the configuration; a name or a ``Resolution`` asks
          about that candidate set.

        Formulas (per backend; ``H`` query heads, ``H_kv`` KV heads, ``G =
        H / H_kv``, ``D`` = ``head_dim_vo``, ``SM`` = SM count, ``B`` =
        batch size, ``N`` = total query tokens, ``T`` = the fa2 query tile,
        a power of two in 16..128 chosen from the packed query length):

        - **fa2** (dominates): the split-KV planner allocates
          ``4 * H * P * T * (D + 1)`` bytes with ``P`` work items.  Graph
          mode: ``P = max(2 * SM // H_kv, ceil(N * G / T) + B - 1)`` — exact
          for the capacity.  Eager mode: ``P <= 2 * SM // H_kv`` and ``T`` is
          bounded by its ceiling (128 below ``D = 256``, else 64), so the
          bound is independent of ``N`` and loose by the tile and work-item
          ratios.  Example (B200, 148 SMs, 32/8 heads, ``D = 128``): one
          request of 2048 tokens in graph mode needs 129 MiB; eager prefill
          of any batch needs at most 75 MiB.
        - **trtllm-gen / cake**: with an LSE output, softmax stats of
          ``8 * H * B * round_up(max_q_len, 256)`` bytes plus a 1 MiB guard;
          nothing otherwise.
        - **cuDNN**: sizes its own graph workspace; measured 0-8960 bytes
          across shapes, carried as a 1 MiB allowance (not derived).
        - **fa3**: nothing from this buffer (its planner uses the wrapper's
          own int workspace).

        The reserved metadata storage, the generated-FA wrapper's 8 MiB int
        workspace and its pinned mirror are per-instance costs outside this
        buffer and not included.
        """
        from .experimental.paged_attention import (
            workspace_requirements as _workspace_requirements,
        )

        return _workspace_requirements(
            capacity,
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
            use_sinks=use_sinks,
            use_cuda_graph=use_cuda_graph,
            backend=backend,
        )

    @flashinfer_experimental_api(feature=_FEATURE)
    def plan(
        self,
        metadata: "PagedAttentionMetadata",
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
        backend: Union[str, "Resolution"] = "auto",
    ) -> "PagedAttention":
        """Plan one batch.

        - ``metadata``: a :class:`PagedAttentionMetadata` built once per step
          with ``.dense(...)`` (vLLM-style block table, page_size >= 8) or
          ``.csr(...)`` (sglang-style flat page ids, any page_size >= 1).  It
          carries the device tensors, the required host maxes and the optional
          CPU mirrors; construction validates once, so plan() is zero-sync and
          several plans over the same batch derive their metadata once.
        - ``num_qo_heads / num_kv_heads / head_dim_qk / head_dim_vo / q_dtype /
          kv_dtype / kv_layout``: the static model configuration.
        - ``causal``: also enforces ``q_len_i <= kv_len_i`` per request, except
          for padding rows (``kv_len_i == 0``, see
          :class:`PagedAttentionMetadata`): those are legal and read no KV
          page; the row's output and LSE are unspecified by contract and MAY
          BE LEFT UNWRITTEN (fa2/fa3 and cuDNN write a zero row and an LSE of
          -inf, trtllm-gen leaves the rows untouched; cake declines such
          batches with the typed signal because its kernel hangs on an empty
          KV range) — never read a padding row.
        - ``window_left``: sliding-window size (-1 = unlimited); backends
          without window support are capability-excluded.  Plan-time because
          it selects a compiled kernel variant on the FA backends.
        - ``lse_mode``: ``"none"``, ``"base2"`` or ``"basee"`` — the base of
          the LSE ``run()`` returns, delivered natively where the backend can
          and with one fold otherwise.
        - ``logits_soft_cap``: softmax logits soft cap ``cap * tanh(score /
          cap)`` applied to the scaled scores (Gemma-2 / Grok style);
          ``None`` or ``0`` = off.  Plan-time because it selects a compiled
          kernel variant.
        - ``custom_mask``: flattened boolean attention mask, the per-request
          ``(q_len_i, kv_len_i)`` masks flattened row-major and concatenated
          in request order (``sum(q_len_i * kv_len_i)`` elements, ``True`` =
          may attend), on this instance's device.  It is ANDed into the
          causal / sliding-window envelope, so ``causal=True`` plus a mask
          never widens the envelope.  Not yet supported together with
          ``use_cuda_graph=True``.
        - ``use_sinks``: declare that ``run()`` will pass per-head attention
          sinks (the backend plans its sink-aware kernel variant); ``run()``
          then requires ``sinks=``.
        - ``backend``: a backend name, ``"auto"``, or a ``Resolution`` from
          :func:`resolve_paged_attention` — the latter pins the candidate set
          decided at engine init (plan() verifies the config, including the
          three feature flags, matches).  Backends that cannot apply a
          requested feature are excluded with a reason, never silently.

        Publication is transactional: a failing ``plan()`` leaves the previous
        plan runnable (see ``experimental/paged_attention/_controller.py`` for
        the one generated-FA caveat).  In CUDA-graph mode the new batch is
        staged into reserved storage and rolled back on failure; it must fit
        the :class:`GraphCapacity` and keep the semantic arguments the first
        graph-mode plan froze (use :meth:`update` to pass only the metadata).
        """
        self._impl.plan(
            metadata,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            lse_mode=lse_mode,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
            use_sinks=use_sinks,
            backend=backend,
        )
        return self

    @flashinfer_experimental_api(feature=_FEATURE)
    def update(self, metadata: "PagedAttentionMetadata") -> "PagedAttention":
        """Re-plan a captured graph's next batch (CUDA-graph mode only).

        Takes only the per-step metadata: the backend and every semantic
        ``plan()`` argument were frozen by the first successful graph-mode
        ``plan()`` and are reused here.  The new batch must fit the
        :class:`GraphCapacity` (same batch size, paging form and page size;
        at most the capacity's total query tokens, host maxes and flat page-id
        length); it is staged into the reserved storage and rolled back on
        failure, so a failed ``update()`` leaves the previous batch
        replayable.  Call it outside capture, on the stream the graph is
        replayed on; then ``graph.replay()``.

        Raises ``RuntimeError`` when the instance is not in graph mode, when
        no ``plan()`` has succeeded yet, or when the current stream is
        capturing.  It re-issues ``plan()`` with the frozen arguments and the
        frozen backend name, so a ``plan()`` spelled that way is equivalent;
        ``update()`` is the engine-facing spelling (sglang's
        ``fast_prefill_plan`` role).
        """
        self._impl.update(metadata)
        return self

    @flashinfer_experimental_api(feature=_FEATURE, trace=paged_attention_trace_dispatch)
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
        """Run the planned batch.

        - ``q``: ``(total_q_tokens, num_qo_heads, head_dim_qk)``, dtype as
          planned, dense along ``head_dim`` (``q.stride(-1) == 1``).  fa2/fa3
          and trtllm-gen address any such view (e.g. the head slice
          ``qkv[:, :num_qo_heads]`` of a fused QKV projection); cuDNN and
          cake need packed storage and reject other layouts.  Nothing is
          copied here.  In CUDA-graph mode ``q`` is the capture buffer and
          may carry up to the capacity's rows; rows past the batch are
          neither read nor written.
        - ``kv_cache``: ``(k_cache, v_cache)`` pair, each paged in the planned
          layout — HND ``(pages, num_kv_heads, page_size, head_dim)`` or NHD
          ``(pages, page_size, num_kv_heads, head_dim)``.
        - ``out``: optional preallocated output, contiguous
          ``(q.shape[0], num_qo_heads, head_dim_vo)``, dtype == q dtype.
        - ``lse``: optional preallocated LSE buffer, contiguous fp32
          ``(q.shape[0], num_qo_heads)``; requires ``lse_mode != "none"``.
        - ``sm_scale``: softmax scale for this call (default
          ``1/sqrt(head_dim_qk)``); a per-layer value, so one plan serves
          layers with different scales.
        - ``k_scale`` / ``v_scale``: per-tensor scales of the K and V caches,
          positive host floats so the call stays sync-free; an omitted scale
          means no scaling (1.0).  For an fp8 ``kv_dtype`` they are the
          dequantization scales (``real = fp8_value * scale``); for a fp16 /
          bf16 cache they multiply K and V (the legacy wrapper's semantics),
          which every backend applies alike: ``k_scale`` folds into the
          softmax scale (so ``logits_soft_cap``, ``sinks`` and the returned
          LSE see the scaled logits, as ``K * k_scale`` would give) and
          ``v_scale`` multiplies the output.  Per-layer values like
          ``sm_scale``; under a captured graph the captured values replay.
        - ``sinks``: per-head attention sinks, a contiguous fp32
          ``(num_qo_heads,)`` device tensor: head ``h`` gets one extra logit
          ``sinks[h]`` in its softmax denominator with no value contribution
          (the returned LSE includes it).  Required iff the plan declared
          ``use_sinks=True``; a per-layer value like ``sm_scale``.

        Returns ``(out, lse)``; ``lse`` is packed ``(total_q_tokens,
        num_qo_heads)`` fp32 in the planned base — identical for every backend
        (``None`` when ``lse_mode="none"``).  Rows of padding requests
        (``kv_len == 0``) are unspecified by contract and may be left
        unwritten (fa2/fa3 and cuDNN write a zero row and an LSE of -inf,
        trtllm-gen leaves them untouched; cake declines such batches at plan
        time); never read them.

        Tracing: ``flashinfer.fi_trace(attn.run, q=q, kv_cache=(k, v))`` on a
        planned instance (or ``FLASHINFER_TRACE_DUMP=1`` during ``run()``)
        exports a flashinfer-bench definition whose identity is the plan's
        paging form, KV layout, causal / window / LSE settings and geometry,
        with the plan-owned metadata read from the last successful ``plan()``.
        A plan that uses ``logits_soft_cap``, a custom mask or sinks refuses
        to trace, because the definition does not encode those yet.
        """
        return self._impl.run(
            q,
            kv_cache,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            sinks=sinks,
        )

    def explain(self) -> str:
        """Chosen backend, the plan-time trace (every candidate tried and why
        it was accepted or declined) and the resolve-time exclusion reasons.
        Raises ``ValueError`` before the first successful ``plan()``."""
        return self._impl.explain()

    def _trace_context(self) -> Dict[str, Any]:
        """Read-only facts of the last successful ``plan()`` for ``fi_trace``.

        Private: consumed by the trace template bound to :meth:`run`.  Raises
        ``ValueError`` before the first successful plan; never syncs.
        """
        return self._impl.trace_context()
