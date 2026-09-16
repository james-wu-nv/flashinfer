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
    backends    experimental.paged_attention._backends    (fa2/fa3, cudnn, trtllm-gen)
    kernels     the existing wrapper / standalone functions (untouched)

Design rules enforced (each traces to a documented failure mode):

1.  ONE canonical metadata form — token-unit ``qo_indptr``, per-request
    ``kv_seq_lens``, a dense ``block_tables`` or flat ``kv_page_indices``, and
    REQUIRED host maxes.  Everything any backend wants (CSR page indices for
    fa2/fa3, cumulative KV lens for trtllm-gen, ``(b,1,1,1)`` lens for cuDNN)
    is derived internally.
2.  Closed input set + loud errors.  Combinations outside the contract raise
    ``ValueError`` with the fix in the message.  We never guess a layout
    (cuDNN issue #3800 is what guessing looks like).
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
- dtypes: fp16/bf16 activations; an fp8 (e4m3) KV cache with per-tensor
  ``k_scale`` / ``v_scale`` at ``run()`` where the capability table declares
  it (fa2).  fp8 Q and nvfp4 are undeclared axes.
- Heuristic order is a static per-arch placeholder, to be seeded from the
  benchmark suite (proposal §5.2).  Autotune hook (§5.4) is not wired.
- ``sinks`` / custom masks / soft-cap are absent capability axes.

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
    ``lse`` are the capture buffers, sized to the capacity's rows; rows past
    the batch are neither read nor written.  ``plan()`` and ``update()`` are
    rejected while the current stream is capturing.  ``sm_scale`` /
    ``k_scale`` / ``v_scale`` are launch scalars: a captured graph keeps the
    values it was captured with.

Paging metadata comes in exactly one of two forms (never both):
- dense ``block_tables (b, max_pages)`` — vLLM-native; page_size >= 8
  (denser would blow the table up; token-CSR engines use the other form);
- flat ``kv_page_indices`` — sglang-style token/CSR-native, any page_size
  >= 1.  The page-unit indptr and last-page lengths are NOT accepted: they
  are derivable from ``kv_seq_lens`` + ``page_size``, and accepting them
  would create a second truth (see the #3921 mask-divergence class).
  Backends that need the dense table (cudnn, trtllm-gen) get it derived by
  a zero-sync gather when page_size >= 8, and are capability-excluded below
  that.
- Trusted inputs (documented, not validated): host mirrors must match the
  device tensors; ``block_tables`` VALUES (page ids) must be in-pool —
  checking them costs a device-side pass the hot path cannot pay; a debug
  mode (FLASHINFER_VALIDATE_INPUTS-style) is the production answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

import torch

from .api_logging import flashinfer_experimental_api

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
    backend: str = "auto",
) -> "Resolution":
    """Static backend resolution — no plan state, no tensors.

    Callable at engine init, before the KV pool is allocated and before any
    CUDA graph is captured (vLLM decides its cudagraph mode and Q dtype at
    that point).  Returns an ordered candidate set plus a reason for every
    excluded backend; raises ``ValueError`` when nothing can run.  Pass the
    result to :meth:`PagedAttention.plan` as ``backend=`` to pin the
    candidate set.
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
        backend=backend,
    )


class PagedAttention:
    """Paged attention over the existing fa2/fa3, cuDNN and trtllm-gen kernels
    behind one contract (experimental).

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

    One CUDA-graph bucket (the module docstring describes the lifecycle)::

        cap = GraphCapacity(batch_size=256, total_q_tokens=1024, max_q_len=4,
                            max_kv_len=8192 * 16, page_size=16, table_width=8192)
        attn = PagedAttention(device, graph_capacity=cap)
        attn.plan(md0, num_qo_heads=8, num_kv_heads=2, head_dim_qk=128,
                  q_dtype=torch.bfloat16, backend=res)   # freezes backend + semantics
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
          (contiguous 1-D uint8 on ``device``; the legacy wrappers' 128 MB
          convention) that every backend's kernels run on — pass the buffer
          the engine already shares with its legacy wrappers.  By default
          every instance on a device shares one lazily allocated
          library-owned pool, so holding one instance per graph bucket costs
          no workspace per bucket.  Instances sharing a workspace must not run
          concurrently on different streams; pass a private buffer where that
          isolation is needed.
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
        - ``causal``: also enforces ``q_len_i <= kv_len_i`` per request.
        - ``window_left``: sliding-window size (-1 = unlimited); backends
          without window support are capability-excluded.  Plan-time because
          it selects a compiled kernel variant on the FA backends.
        - ``lse_mode``: ``"none"``, ``"base2"`` or ``"basee"`` — the base of
          the LSE ``run()`` returns, delivered natively where the backend can
          and with one fold otherwise.
        - ``backend``: a backend name, ``"auto"``, or a ``Resolution`` from
          :func:`resolve_paged_attention` — the latter pins the candidate set
          decided at engine init (plan() verifies the config matches).

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
        capturing.  A ``plan()`` with the same frozen arguments is equivalent;
        ``update()`` is the engine-facing spelling (sglang's
        ``fast_prefill_plan`` role).
        """
        self._impl.update(metadata)
        return self

    @flashinfer_experimental_api(feature=_FEATURE)
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
        """Run the planned batch.

        - ``q``: ``(total_q_tokens, num_qo_heads, head_dim_qk)``, dtype as
          planned, dense along ``head_dim`` (``q.stride(-1) == 1``).  fa2/fa3
          and trtllm-gen address any such view (e.g. the head slice
          ``qkv[:, :num_qo_heads]`` of a fused QKV projection); cuDNN needs
          packed storage and rejects other layouts.  Nothing is copied here.
        - ``kv_cache``: ``(k_cache, v_cache)`` pair, each paged in the planned
          layout — HND ``(pages, num_kv_heads, page_size, head_dim)`` or NHD
          ``(pages, page_size, num_kv_heads, head_dim)``.
        - ``out``: optional preallocated output, contiguous
          ``(total_q_tokens, num_qo_heads, head_dim_vo)``, dtype == q dtype.
        - ``lse``: optional preallocated LSE buffer, contiguous fp32
          ``(total_q_tokens, num_qo_heads)``; requires ``lse_mode != "none"``.
        - ``sm_scale``: softmax scale for this call (default
          ``1/sqrt(head_dim_qk)``); a per-layer value, so one plan serves
          layers with different scales.
        - ``k_scale`` / ``v_scale``: per-tensor dequantization scales for an
          fp8 KV cache (``dequant = fp8_value * scale``), host floats so the
          call stays sync-free; only valid when the plan's ``kv_dtype`` is fp8.

        Returns ``(out, lse)``; ``lse`` is packed ``(total_q_tokens,
        num_qo_heads)`` fp32 in the planned base — identical for every backend.
        """
        return self._impl.run(
            q,
            kv_cache,
            out=out,
            lse=lse,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    def explain(self) -> str:
        """Chosen backend, the plan-time trace (every candidate tried and why
        it was accepted or declined) and the resolve-time exclusion reasons."""
        return self._impl.explain()
