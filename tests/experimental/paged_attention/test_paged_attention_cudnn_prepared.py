"""Prepared-plan path of the unified cuDNN backend (``backend="cudnn"`` pinned).

``plan()`` stages a ``flashinfer.cudnn.prefill._CudnnPrefillPlan`` and
``run()`` executes a prepared cuDNN graph through ``matches_plan`` /
``run_planned``; the functional entry ``cudnn_batch_prefill_with_kv_cache``
remains the fallback where no plan can be staged.  Both declare the same
shapes, so this is a host-path change only.  Pinned here, against the fp32
paged-attention oracle and without timing thresholds:

1. Output and LSE (``base2`` / ``basee`` / none) for causal and non-causal,
   HND and NHD, GQA and MHA, ragged per-request lengths.
2. The prepared path is the one taken where the library gate holds: a staged
   plan after ``plan()``, and a ``run()`` that never reaches the functional
   entry and repeats none of the plan-time host work (no metadata resolve,
   no plan staging, no binding map or descriptor key rebuilt; the execute is
   ``run_planned``).  The functional fallback is taken and still correct
   without a staged plan.
3. ``sm_scale`` / ``k_scale`` / ``v_scale`` supplied only to ``run()`` and
   changed between runs of one plan.
4. Eager re-plans with changed lengths and changed page ids; through page
   tables of other widths, row strides and owners (derived from flat page
   ids or the caller's); across LSE modes, masks and LSE layouts; a rejected
   re-plan and one that fails after staging.
5. CUDA-graph mode: capture one run, ``update()`` with changed lengths and
   page ids, replay -- on the prepared path, on the functional fallback and
   with cuDNN's backend engines alone (provider control), in a small bucket
   and in one large enough for the frontend's own prefill engines; NHD,
   non-causal and run-time scales; two captures of one instance; a capture
   that is the instance's first run.  A staging failure leaves the captured
   state intact.
6. Single-token batches (every ``q_len == 1``) and batches mixing 1 with
   longer or empty queries: the three LSE layouts of the backend.
7. q / out / lse views with a storage offset.
8. Both paths reach one cached cuDNN graph: the same graph key and graph
   object across layouts, masks, dtypes, head dims, paging forms, LSE
   layouts and graph mode, and no second build on an isolated graph cache.

The backend's private state is read through ``attn._impl._active``, as the
neighbouring cuDNN tests do.
"""

import collections
import functools
import math

import pytest
import torch

import flashinfer.cudnn as fi_cudnn
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.experimental.paged_attention._backends import cudnn_backend
from flashinfer.prefill import PagedAttention

from .test_paged_attention_coverage import (
    assert_canaries,
    assert_matches,
    build_problem,
    canary_buffers,
    plan,
    reference,
    run,
)
from .test_paged_attention_prototype import (
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not cudnn_prefill.CUDNN_AVAILABLE,
        reason="cudnn-frontend python package not available",
    ),
]

DEVICE = "cuda:0"
PAGE = 16
HEAD_DIM = 128
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

# One ragged batch for the eager tests: a one-token request, a full-prefill
# request with q == kv, two prefixes.
_RAGGED = ([5, 17, 1, 9], [40, 17, 70, 33])
# A batch large enough for the frontend's own prefill engines where the stack
# has them (HND, causal); the smaller shapes here stay on cuDNN's backend
# engines.  No test asserts which provider serves it by default; only the
# backend-engine control checks the engine it selected.
_LARGE = ([256, 512, 1], [2048, 4096, 3000])

# The three LSE layouts of the backend, by query partition (KV lengths
# _LSE_LAYOUT_KV): the layout an eager plan takes where this cuDNN writes
# ragged stats.
_LSE_LAYOUT_KV = [33, 70, 16, 48, 5, 20]
_LSE_LAYOUTS = {
    "all_one": ([1, 1, 1, 1, 1, 1], "view"),  # decode shape: padded stats == packed
    "one_and_longer": ([1, 9, 1, 4, 1, 1], "direct"),  # one-token next to longer
    "one_and_empty": ([1, 0, 1, 1, 0, 1], "gather"),  # max_q 1, padded != packed
}


def _problem(
    q_lens,
    kv_lens,
    *,
    seed,
    heads=(8, 2),
    kv_layout="HND",
    input_form="block_tables",
    table_width=None,
    pool=None,
    dtype=torch.bfloat16,
    head_dim_qk=HEAD_DIM,
):
    """``build_problem`` at head dims (128, 128) / page 16 / bf16 unless
    overridden.  ``pool`` fixes the number of pages in the K/V pool, so
    batches with different lengths share the K/V shapes (part of the cuDNN
    graph signature, and the size of a capture's static K/V buffers)."""
    pages = sum(-(-int(kv) // PAGE) for kv in kv_lens)
    return build_problem(
        q_lens,
        kv_lens,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim_qk=head_dim_qk,
        head_dim_vo=HEAD_DIM,
        page_size=PAGE,
        dtype=dtype,
        seed=seed,
        kv_layout=kv_layout,
        input_form=input_form,
        table_width=table_width,
        pool_slack=8 if pool is None else pool - pages,
        device=DEVICE,
    )


def _prepared_expected():
    """The gate the backend stages its plan under: per-request KV lengths next
    to token-unit query offsets (cuDNN 9.25 / cudnn-frontend 1.27)."""
    return cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True)


def _require_prepared():
    if not _prepared_expected():
        pytest.skip("this cuDNN / frontend has no direct paged seq-lens path")


def _plan_cudnn(p, *, causal=True, lse_mode="base2", attn=None, **attn_kwargs):
    """Resolve-or-skip, then plan ``p`` on cudnn; returns (attn, backend)."""
    _resolve_or_skip(p, "cudnn", causal=causal, need_lse=lse_mode != "none")
    attn = attn or PagedAttention(torch.device(DEVICE), **attn_kwargs)
    plan(attn, p, "cudnn", causal=causal, lse_mode=lse_mode)
    assert attn.backend == "cudnn"
    return attn, attn._impl._active


def _assert_staged(backend):
    """A staged plan exactly where the gate holds; returns whether it is."""
    staged = backend._cudnn_plan is not None
    assert staged == _prepared_expected()
    return staged


def _answer_packed_lse_probe():
    """The one-time packed-LSE probe of this process runs the functional
    entry from the first LSE plan with max_q_len > 1.  Answer it here, for a
    test whose first plan is not such a plan and that forbids or counts the
    entry before a later one."""
    _plan_cudnn(_problem(*_RAGGED, seed=100), lse_mode="base2")


def _forbid_functional_entry(monkeypatch):
    """After plan() (whose one-time packed-LSE probe uses the entry): any
    later call of the functional entry fails the test."""

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "run() reached cudnn_batch_prefill_with_kv_cache although a plan was staged"
        )

    monkeypatch.setattr(fi_cudnn, "cudnn_batch_prefill_with_kv_cache", forbidden)


def _spy_functional_entry(monkeypatch):
    real = fi_cudnn.cudnn_batch_prefill_with_kv_cache
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(fi_cudnn, "cudnn_batch_prefill_with_kv_cache", spy)
    return calls


def _count_preparations(monkeypatch):
    """What ``prepare_cudnn_batch_prefill`` returned, call by call: the
    prepared path makes one call per graph signature its history does not
    hold, the functional entry one per run."""
    real = cudnn_prefill.prepare_cudnn_batch_prefill
    fetched = []

    def counted(*args, **kwargs):
        prepared = real(*args, **kwargs)
        fetched.append(prepared)
        return prepared

    monkeypatch.setattr(cudnn_prefill, "prepare_cudnn_batch_prefill", counted)
    return fetched


def _count_host_work(monkeypatch):
    """Call counts of the host work that belongs to plan() or to the first
    run() of a graph signature, and that the functional entry repeats on
    every call: resolving the metadata, staging a plan, building the binding
    map, computing the graph descriptor key -- plus which execute served the
    call (``run_planned`` with the plan's static bindings, or the per-call
    ``run`` of the functional entry).  ``matches_plan`` computes the runtime
    key on every prepared run by design and is not counted."""
    counts = collections.Counter()

    def counted(name, real):
        @functools.wraps(real)
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return wrapper

    metadata_cls = cudnn_prefill._PrefillMetadata
    plan_cls = cudnn_prefill._CudnnPrefillPlan
    graph_cls = cudnn_prefill.CudnnPrefillGraph
    monkeypatch.setattr(
        metadata_cls,
        "resolve_from_plan",
        counted("_PrefillMetadata.resolve_from_plan", metadata_cls.resolve_from_plan),
    )
    monkeypatch.setattr(
        plan_cls,
        "prepare",
        classmethod(counted("_CudnnPrefillPlan.prepare", plan_cls.prepare.__func__)),
    )
    for name in ("_prefill_plan_bindings", "_prefill_descriptor_key"):
        monkeypatch.setattr(
            cudnn_prefill, name, counted(name, getattr(cudnn_prefill, name))
        )
    for name in ("run_planned", "run"):
        monkeypatch.setattr(
            graph_cls,
            name,
            counted(f"CudnnPrefillGraph.{name}", getattr(graph_cls, name)),
        )
    return counts


def _assert_hot_path(counts, runs, *, first_of_signature=False):
    """The ``runs`` run() calls since ``counts.clear()`` did only the per-call
    work of the prepared path: one ``run_planned`` each and nothing that
    plan() staged.  The first run of a graph signature also fetches the cuDNN
    graph, which computes its cache key."""
    seen = dict(counts)
    if first_of_signature:
        seen.pop("_prefill_descriptor_key", None)
    assert seen == {"CudnnPrefillGraph.run_planned": runs}


def _no_staged_plan(monkeypatch):
    monkeypatch.setattr(
        cudnn_backend._CudnnBackend,
        "_stage_cudnn_plan",
        lambda self, meta, derived, block_tables, path: None,
    )


def _isolate_graph_cache(monkeypatch):
    """Give the test a private, real cuDNN graph cache and return the list of
    graphs handed out from it: a graph another test built cannot hide a build
    here, and the engine each graph selected can be inspected."""
    build = cudnn_prefill.cudnn.graph_cache(key_fn=cudnn_prefill._sdpa_prefill_key_fn)(
        cudnn_prefill._build_prefill_graph.__wrapped__
    )
    graphs = []

    def record(*args, **kwargs):
        result = build(*args, **kwargs)
        graphs.append(result[0])
        return result

    monkeypatch.setattr(cudnn_prefill, "_build_prefill_graph", record)
    return graphs


def _backend_engines_only(monkeypatch):
    """Provider control: leave the frontend no engine candidate of its own, so
    cuDNN's backend engines must serve every graph built from here on.
    Returns the predicate that tells a backend engine."""
    engines = pytest.importorskip("cudnn.engines")
    if not hasattr(cudnn_prefill.cudnn.pygraph, "_candidate_engines"):
        pytest.skip("this cudnn-frontend has no frontend engine candidates")
    monkeypatch.setattr(
        cudnn_prefill.cudnn.pygraph, "_candidate_engines", lambda self: []
    )
    return engines.is_backend_engine


def _warm_up(attn, q, kv, out, lse, **run_kwargs):
    """Two eager runs on a side stream (cuDNN graph build)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            attn.run(q, kv, out=out, lse=lse, **run_kwargs)
    torch.cuda.current_stream().wait_stream(side)


def _capture(attn, q, kv, out, lse, **run_kwargs):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn.run(q, kv, out=out, lse=lse, **run_kwargs)
    return graph


def _warm_and_capture(attn, q, kv, out, lse, **run_kwargs):
    """Warm up on a side stream, then capture one run."""
    _warm_up(attn, q, kv, out, lse, **run_kwargs)
    return _capture(attn, q, kv, out, lse, **run_kwargs)


def _assert_rows_match(
    out, lse, p, *, lse_mode, causal=True, sm_scale=None, k_scale=None, v_scale=None
):
    """Oracle on the first ``total_q_tokens`` rows of capture-sized buffers.
    The scales are the ones run() was given: ``k_scale`` is a factor of the
    softmax scale, ``v_scale`` of the output."""
    n = p["q"].shape[0]
    if k_scale is not None:
        sm_scale = (SM_SCALE if sm_scale is None else sm_scale) * k_scale
    if v_scale is not None:
        p = dict(p, v_ref=p["v_ref"].float() * v_scale)
    ref_out, ref_lse = reference(
        p,
        causal=causal,
        sm_scale=sm_scale,
        lse_base="e" if lse_mode == "basee" else "2",
    )
    torch.testing.assert_close(out[:n].float(), ref_out, **OUT_TOL)
    if lse_mode != "none":
        torch.testing.assert_close(lse[:n], ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# 1 + 2: numbers through the prepared path, and that it is the path taken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("heads", [(8, 2), (4, 4)], ids=["gqa", "mha"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
def test_prepared_path_matches_oracle(monkeypatch, causal, kv_layout, heads, lse_mode):
    """Ragged batch (a one-token request, a full-prefill request with
    q == kv, two prefixes).  Two runs: the first fetches the cuDNN graph, the
    second reuses the prepared one; neither reaches the functional entry."""
    p = _problem(*_RAGGED, seed=101, heads=heads, kv_layout=kv_layout)
    attn, backend = _plan_cudnn(p, causal=causal, lse_mode=lse_mode)
    staged = _assert_staged(backend)
    if staged:
        _forbid_functional_entry(monkeypatch)
    for _ in range(2):
        out, lse = run(attn, p)
        assert_matches(out, lse, p, causal=causal, lse_mode=lse_mode)
    if staged:
        assert backend._prepared is not None
        assert backend._prepared in backend._prepared_history


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("batch", list(_LSE_LAYOUTS))
def test_run_does_no_plan_time_host_work(monkeypatch, batch, kv_layout, lse_mode):
    """The controller contract of run() on the prepared path, as call counts
    (not timings): plan() staged the metadata, the descriptor keys and the
    static bindings, and no run() resolves, stages or rebuilds any of them,
    nor executes through the functional entry's per-call ``run``.  A run that
    did would give the host saving back and still match the oracle.  Checked
    on the first run of the signature (which fetches the graph) and on warm
    runs with and without caller buffers and with another ``v_scale``, for
    the three LSE layouts."""
    _require_prepared()
    p = _problem(_LSE_LAYOUTS[batch][0], _LSE_LAYOUT_KV, seed=107, kv_layout=kv_layout)
    attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
    assert backend._cudnn_plan is not None
    counts = _count_host_work(monkeypatch)
    _forbid_functional_entry(monkeypatch)

    out, lse = run(attn, p)
    _assert_hot_path(counts, 1, first_of_signature=True)
    assert_matches(out, lse, p, lse_mode=lse_mode)

    kv = (p["k_cache"], p["v_cache"])
    out_buf = torch.empty_like(out)
    lse_buf = None if lse is None else torch.empty_like(lse)
    counts.clear()
    out, lse = attn.run(p["q"], kv)
    assert_matches(out, lse, p, lse_mode=lse_mode)
    out, lse = attn.run(p["q"], kv, out=out_buf, lse=lse_buf)
    assert out is out_buf and lse is lse_buf
    assert_matches(out, lse, p, lse_mode=lse_mode)
    attn.run(p["q"], kv, out=out_buf, lse=lse_buf, v_scale=2.0)
    _assert_hot_path(counts, 3)


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
def test_functional_fallback_without_a_staged_plan(monkeypatch, kv_layout, lse_mode):
    """No staged plan: every run() calls the functional entry and is correct.
    The choice is per plan: one instance alternates between the two paths as
    it is re-planned, and neither leaves state that breaks the other.  The
    host-work counters see what the entry repeats per call, which is what
    they must stay at zero for on the prepared path."""
    p = _problem(*_RAGGED, seed=102, kv_layout=kv_layout)
    attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
    staged = _assert_staged(backend)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode=lse_mode)
    prepared = backend._prepared

    with monkeypatch.context() as m:
        _no_staged_plan(m)
        _plan_cudnn(p, lse_mode=lse_mode, attn=attn)
        assert backend._cudnn_plan is None
        calls = _spy_functional_entry(m)
        counts = _count_host_work(m)
        for n in (1, 2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode=lse_mode)
            assert len(calls) == n
        assert backend._prepared is prepared  # untouched by the fallback
        assert counts["CudnnPrefillGraph.run"] == 2
        assert counts["CudnnPrefillGraph.run_planned"] == 0
        assert counts["_PrefillMetadata.resolve_from_plan"]
        assert counts["_prefill_plan_bindings"]
        assert counts["_prefill_descriptor_key"]

    _plan_cudnn(p, lse_mode=lse_mode, attn=attn)
    assert (backend._cudnn_plan is not None) == staged
    if staged:
        _forbid_functional_entry(monkeypatch)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode=lse_mode)


def test_functional_fallback_below_the_direct_seqlens_gate(monkeypatch):
    """The library gate forced off (what an older cuDNN / frontend reports):
    nothing is staged and the functional entry serves the call."""
    p = _problem(*_RAGGED, seed=103)
    monkeypatch.setattr(
        cudnn_prefill,
        "_cudnn_supports_direct_seqlens",
        lambda dtype, *, mixed=False: False,
    )
    attn, backend = _plan_cudnn(p, lse_mode="none")
    assert backend._cudnn_plan is None
    calls = _spy_functional_entry(monkeypatch)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="none")
    assert len(calls) == 1 and backend._prepared is None


@pytest.mark.parametrize("first", ["functional", "prepared"])
@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
def test_prepared_and_functional_paths_share_one_cudnn_graph(
    monkeypatch, first, lse_mode
):
    """Same declaration, same graph key: on an isolated graph cache, whichever
    path runs second builds no cuDNN graph (both orders)."""
    _require_prepared()
    p = _problem([6, 14, 3], [50, 14, 80], seed=104)
    _resolve_or_skip(p, "cudnn", need_lse=lse_mode != "none")
    _isolate_graph_cache(monkeypatch)

    def builds(functional):
        with monkeypatch.context() as m:
            if functional:
                _no_staged_plan(m)
            # plan first: its one-time packed-LSE probe builds a toy graph
            attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
            assert (backend._cudnn_plan is None) == functional
            before = cudnn_prefill._prefill_graph_builds
            for _ in range(2):
                out, lse = run(attn, p)
                assert_matches(out, lse, p, lse_mode=lse_mode)
            return cudnn_prefill._prefill_graph_builds - before

    order = (True, False) if first == "functional" else (False, True)
    first_builds, second_builds = (builds(functional) for functional in order)
    assert first_builds >= 1  # the isolated cache started empty
    assert second_builds == 0


# What the declaration of test_both_paths_fetch_the_same_cudnn_graph varies
# besides the LSE mode and the LSE layout: _problem() / plan() overrides.
_IDENTITY_VARIANTS = {
    "hnd_causal": dict(),
    "nhd": dict(kv_layout="NHD"),
    "noncausal": dict(causal=False),
    "csr": dict(input_form="page_indices"),  # the dense table is derived
    "fp16": dict(dtype=torch.float16),
    "d192": dict(head_dim_qk=192),
    # capacity maxes instead of the batch's, reserved metadata buffers, and
    # padded stats for every max_q_len == 1 batch
    "graph": dict(use_cuda_graph=True),
}


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("batch", list(_LSE_LAYOUTS))
@pytest.mark.parametrize("variant", list(_IDENTITY_VARIANTS))
def test_both_paths_fetch_the_same_cudnn_graph(monkeypatch, variant, batch, lse_mode):
    """The staged declaration is the one the functional entry resolves from
    the same arguments: both paths ask the graph cache for the same key and
    get the same cuDNN graph object, with the same stats base and no shape
    override -- hence the same kernel.  Compared as the prepared graph each
    path fetched, across the declarations the backend can make."""
    _require_prepared()
    overrides = dict(_IDENTITY_VARIANTS[variant])
    causal = overrides.pop("causal", True)
    attn_kwargs = (
        dict(use_cuda_graph=True) if overrides.pop("use_cuda_graph", False) else {}
    )
    p = _problem(_LSE_LAYOUTS[batch][0], _LSE_LAYOUT_KV, seed=108, **overrides)

    def fetch(functional):
        with monkeypatch.context() as m:
            if functional:
                _no_staged_plan(m)
            # a fresh instance per path; plan before counting: the one-time
            # packed-LSE probe of this process fetches a toy graph
            attn, backend = _plan_cudnn(
                p, causal=causal, lse_mode=lse_mode, **attn_kwargs
            )
            assert (backend._cudnn_plan is None) == functional
            fetched = _count_preparations(m)
            out, lse = run(attn, p)
            assert_matches(out, lse, p, causal=causal, lse_mode=lse_mode)
            assert len(fetched) == 1
            if not functional:
                assert fetched[0] is backend._prepared
            return fetched[0]

    prepared, functional = fetch(False), fetch(True)
    assert functional.key == prepared.key
    assert functional.graph is prepared.graph
    for name in (
        "override_cache",
        "return_lse",
        "stats_head_stride",
        "requested_stats_head_stride",
        "lse_base",
        "stats_use_log2",
    ):
        assert getattr(functional, name) == getattr(prepared, name), name
    assert prepared.override_cache is None  # the exact declaration, no class


@pytest.mark.parametrize("lse_mode", ["none", "base2"])
def test_prepared_run_allocates_nothing_with_caller_buffers(lse_mode):
    """With out= / lse= supplied, a warm run() on the prepared path makes no
    device allocation (cuDNN works in the instance's workspace)."""
    _require_prepared()
    p = _problem(*_RAGGED, seed=105)
    attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
    if lse_mode != "none" and not backend.lse_written_packed:
        pytest.skip("this cuDNN takes the gather fallback, which allocates")
    dev = torch.device(DEVICE)
    out = torch.empty(
        p["q"].shape[0], p["num_qo_heads"], HEAD_DIM, dtype=p["dtype"], device=dev
    )
    lse = (
        torch.empty(p["q"].shape[0], p["num_qo_heads"], dtype=torch.float32, device=dev)
        if lse_mode != "none"
        else None
    )
    kv = (p["k_cache"], p["v_cache"])
    attn.run(p["q"], kv, out=out, lse=lse)  # warm: graph build
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    attn.run(p["q"], kv, out=out, lse=lse)
    torch.cuda.synchronize(dev)
    allocations = torch.cuda.memory_stats(dev)["allocation.all.allocated"] - before
    assert allocations == 0, f"run() made {allocations} device allocation(s)"
    assert_matches(out, lse, p, lse_mode=lse_mode)


# ---------------------------------------------------------------------------
# 3: run-time scales
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lens", [_RAGGED, _LARGE], ids=["small", "large"])
def test_run_time_scales_change_without_replan(monkeypatch, lens):
    """One plan; sm_scale, k_scale and v_scale exist only at run() and change
    from run to run.  k_scale reaches the logits (and the LSE) as a factor of
    the softmax scale, v_scale multiplies the output and leaves the LSE
    alone.  On the prepared path each distinct softmax scale is one graph
    signature: prepared once, then found again in the instance history (three
    scales here, within any history worth keeping), and no run repeats
    plan-time work.  ``large`` is a shape the frontend's own prefill engines
    take where the stack has them."""
    p = _problem(*lens, seed=106)
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    counts = _count_host_work(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    s2 = 1.7 * SM_SCALE
    # (run() kwargs, the softmax scale the kernel must see, V factor,
    #  preparations so far on the prepared path)
    steps = [
        (dict(), SM_SCALE, 1.0, 1),
        (dict(), SM_SCALE, 1.0, 1),
        (dict(sm_scale=s2), s2, 1.0, 2),
        (dict(sm_scale=s2, k_scale=0.5), s2 * 0.5, 1.0, 3),
        (dict(v_scale=2.0), SM_SCALE, 2.0, 3),  # v_scale is not in the signature
        (dict(sm_scale=s2, k_scale=0.5, v_scale=2.0), s2 * 0.5, 2.0, 3),
        (dict(sm_scale=s2), s2, 1.0, 3),  # earlier scales again: history hits
        (dict(), SM_SCALE, 1.0, 3),
    ]
    unscaled = None
    for kwargs, scale, v_factor, prepared_so_far in steps:
        prepared_before = len(preparations)
        counts.clear()
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), **kwargs)
        ref_out, _ = assert_matches(
            out,
            lse,
            dict(p, v_ref=p["v_ref"].float() * v_factor),
            sm_scale=scale,
            lse_mode="base2",
        )
        if unscaled is None:
            unscaled = ref_out
        elif kwargs:
            assert not torch.allclose(ref_out, unscaled)  # the scale matters here
        if staged:
            assert len(preparations) == prepared_so_far, kwargs
            _assert_hot_path(
                counts, 1, first_of_signature=prepared_so_far > prepared_before
            )


# ---------------------------------------------------------------------------
# 4: eager re-plans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
def test_eager_replan_with_changed_lengths_and_page_ids(monkeypatch, input_form):
    """One instance, fresh metadata tensors per plan.  ``a`` and ``b`` share a
    declaration (batch, maxes, table width, pool) but not their per-request
    lengths or page ids; ``c`` declares another shape.  Every run matches its
    own batch, and on the prepared path a cuDNN graph is prepared once per
    declaration, also when coming back to an earlier one; the repeated run of
    a plan does no plan-time work."""
    form = dict(input_form=input_form, pool=24)
    a = _problem([7, 16, 3], [64, 30, 90], seed=111, **form)
    b = _problem([16, 2, 9], [90, 41, 17], seed=112, **form)
    c = _problem([4, 30], [200, 31], seed=113, **form)
    assert a["max_q_len"] == b["max_q_len"] and a["max_kv_len"] == b["max_kv_len"]
    assert a["block_tables"].shape == b["block_tables"].shape
    assert not torch.equal(a["block_tables"], b["block_tables"])

    attn, backend = _plan_cudnn(a, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    counts = _count_host_work(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    for p, prepared_so_far in ((a, 1), (b, 1), (c, 2), (a, 2), (b, 2), (c, 2)):
        _plan_cudnn(p, lse_mode="base2", attn=attn)
        for _ in range(2):  # repeated run() of one plan
            counts.clear()
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode="base2")
        if staged:
            assert len(preparations) == prepared_so_far
            _assert_hot_path(counts, 1)


def test_table_width_and_stride_transitions(monkeypatch):
    """One instance through the page-table layouts an engine hands over: an
    exact-width table, a capacity-width one (the backend binds a narrow view
    that keeps the wide row stride), the caller's own row-stride view, a
    wider declaration, then the earlier layouts again.  The table's shape and
    strides are part of the graph signature: every layout is correct, and on
    the prepared path a layout seen before prepares nothing."""
    capacity = 9
    wide = _problem([7, 16, 3], [64, 30, 90], seed=115, table_width=capacity, pool=30)
    width = -(-wide["max_kv_len"] // PAGE)
    assert width < capacity
    exact = dict(wide, block_tables=wide["block_tables"][:, :width].contiguous())
    view = dict(wide, block_tables=wide["block_tables"][:, :width])
    assert view["block_tables"].stride(0) == capacity
    longer = _problem(
        [7, 16, 3], [64, 30, 140], seed=116, table_width=capacity, pool=30
    )
    assert -(-longer["max_kv_len"] // PAGE) == capacity

    attn, backend = _plan_cudnn(exact, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    steps = ((exact, 1), (wide, 2), (view, 2), (longer, 3), (exact, 3), (view, 3))
    for p, prepared_so_far in steps:
        _plan_cudnn(p, lse_mode="base2", attn=attn)
        out, lse = run(attn, p)
        assert_matches(out, lse, p, lse_mode="base2")
        if staged:
            assert len(preparations) == prepared_so_far


def test_eager_replan_across_page_table_owners(monkeypatch):
    """One instance alternating between a dense table the controller derives
    from flat page ids and caller-supplied tables: a wider one (a narrow view
    with the wide row stride is bound) and one of the exact width.  Every
    step changes the page ids and the per-request lengths, so a binding left
    over from the previous owner's table would read the wrong pages.  The
    derived table and the caller's exact-width one have one layout, hence one
    prepared graph; the wide table's row stride is a second one."""
    same = dict(pool=30)  # one batch size, max_q 16, max_kv 90: 6 table columns
    flat_a = _problem(
        [7, 16, 3], [64, 30, 90], seed=117, input_form="page_indices", **same
    )
    wide_b = _problem([16, 2, 9], [90, 41, 17], seed=118, table_width=11, **same)
    flat_c = _problem(
        [4, 9, 16], [33, 90, 60], seed=119, input_form="page_indices", **same
    )
    exact_d = _problem([16, 16, 1], [16, 90, 77], seed=120, **same)
    assert wide_b["block_tables"].shape[1] == 11
    assert exact_d["block_tables"].shape[1] == 6

    attn, backend = _plan_cudnn(flat_a, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    steps = (
        (flat_a, 1),
        (wide_b, 2),
        (flat_c, 2),
        (exact_d, 2),
        (flat_a, 2),
        (wide_b, 2),
        (exact_d, 2),
    )
    for p, prepared_so_far in steps:
        _plan_cudnn(p, lse_mode="base2", attn=attn)
        for _ in range(2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode="base2")
        if staged:
            assert len(preparations) == prepared_so_far


@pytest.mark.parametrize("first", [0, 2], ids=["from_none", "from_basee"])
def test_replan_across_lse_modes_masks_and_lse_layouts(monkeypatch, first):
    """One instance re-planned through every LSE mode, both masks and the
    three LSE layouts.  The LSE base, the LSE layout with its gather state
    and the prepared graphs are instance state that outlives a plan: each
    plan must publish its own, whatever the instance planned first (the
    sequence is entered at two points), and two runs of each plan match its
    oracle."""
    batches = {
        name: _problem(q_lens, _LSE_LAYOUT_KV, seed=150 + i)
        for i, (name, (q_lens, _)) in enumerate(_LSE_LAYOUTS.items())
    }
    plans = [
        ("one_and_longer", "none", True),
        ("one_and_longer", "base2", True),
        ("one_and_longer", "basee", True),
        ("one_and_longer", "base2", False),
        ("all_one", "basee", True),
        ("one_and_empty", "base2", True),
        ("one_and_longer", "none", False),
        ("one_and_empty", "basee", False),
        ("all_one", "base2", True),
        ("one_and_longer", "basee", False),
        ("all_one", "none", True),
        ("one_and_longer", "base2", True),
    ]
    _answer_packed_lse_probe()
    attn = None
    for batch, lse_mode, causal in plans[first:] + plans[:first]:
        p = batches[batch]
        attn, backend = _plan_cudnn(p, causal=causal, lse_mode=lse_mode, attn=attn)
        if _assert_staged(backend):
            _forbid_functional_entry(monkeypatch)
        for _ in range(2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, causal=causal, lse_mode=lse_mode)


def test_rejected_replan_keeps_the_staged_plan(monkeypatch):
    """A plan the backend rejects (page dimension not unit-stride) publishes
    nothing: the staged plan and the prepared graph stay, and the next run()
    is correct without preparing anything."""
    p = _problem([7, 16, 3], [64, 30, 90], seed=114)
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    staged = _assert_staged(backend)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")
    kept_plan, kept_graph = backend._cudnn_plan, backend._prepared

    table = p["block_tables"].t().contiguous().t()  # same values, stride(1) == b
    assert table.stride(1) != 1 and torch.equal(table, p["block_tables"])
    with pytest.raises(ValueError, match="unit-stride along the page"):
        plan(attn, dict(p, block_tables=table), "cudnn", lse_mode="base2")
    assert backend._cudnn_plan is kept_plan and backend._prepared is kept_graph

    preparations = _count_preparations(monkeypatch)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")
    if staged:
        assert not preparations


def test_eager_replan_failing_after_staging_keeps_the_gather_indices(monkeypatch):
    """Eager, padded-stats (gather) LSE path.  ``b`` has as many query tokens
    as ``a`` but another (batch, max_q), so its plan would refill ``a``'s
    gather indices in place and needs a new padded stats buffer.  When that
    allocation fails the plan publishes nothing, ``a`` stays planned, and its
    indices must still be ``a``'s: ``b``'s would gather the wrong stats rows
    (here they also index past ``a``'s batch)."""
    dev = torch.device(DEVICE)
    monkeypatch.setitem(cudnn_backend._PACKED_LSE_SUPPORTED, dev, False)
    a = _problem([7, 16, 3], [64, 30, 90], seed=126)
    b = _problem([10, 4, 2, 10], [40, 20, 9, 64], seed=127)
    assert a["q"].shape[0] == b["q"].shape[0]
    attn, backend = _plan_cudnn(a, lse_mode="base2")
    assert not backend.lse_written_packed
    out, lse = run(attn, a)
    assert_matches(out, lse, a, lse_mode="base2")
    batch_ids, pos = backend._batch_ids.clone(), backend._pos.clone()

    stats_shape = (4, b["max_q_len"], b["num_qo_heads"])
    real_empty = torch.empty

    def failing_empty(*args, **kwargs):
        if args == stats_shape:
            raise RuntimeError("injected allocation failure")
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", failing_empty)
        with pytest.raises(RuntimeError, match="injected allocation failure"):
            plan(attn, b, "cudnn", lse_mode="base2")
    assert torch.equal(backend._batch_ids, batch_ids)
    assert torch.equal(backend._pos, pos)
    out, lse = run(attn, a)
    assert_matches(out, lse, a, lse_mode="base2")

    # and the instance still takes that plan
    _plan_cudnn(b, lse_mode="base2", attn=attn)
    assert not torch.equal(backend._batch_ids, batch_ids)
    out, lse = run(attn, b)
    assert_matches(out, lse, b, lse_mode="base2")


# ---------------------------------------------------------------------------
# 5: CUDA-graph mode
# ---------------------------------------------------------------------------

# Graph buckets: the _problem() shape every batch of the bucket shares, and
# three batches.  The later batches change the query partition, the KV
# lengths and every page id, and use fewer tokens than the first (captured)
# one; the second has a request without query tokens.
_GRAPH_BUCKETS = {
    # batch 4, at most 45 query tokens, max_q 20, table width 8 (max_kv 128),
    # 40 pool pages; the third batch has a padding row (no query, no KV)
    "small": (
        dict(table_width=8, pool=40),
        (
            ([9, 20, 4, 12], [100, 37, 128, 64]),
            ([20, 0, 14, 6], [41, 90, 14, 119]),
            ([3, 12, 20, 0], [128, 12, 77, 0]),
        ),
    ),
    # batch 3, at most 769 query tokens, max_q 512, table width 256 (max_kv
    # 4096), 600 pool pages: _LARGE and two siblings, the last with a
    # full-prefill request (q == kv)
    "large": (
        dict(table_width=256, pool=600),
        (
            _LARGE,
            ([512, 0, 200], [3000, 4096, 777]),
            ([100, 1, 512], [4096, 1500, 512]),
        ),
    ),
}
_RUN_SCALES = dict(sm_scale=1.7 * SM_SCALE, k_scale=0.5, v_scale=2.0)


def _graph_batches(bucket, *, seed, **overrides):
    shape, batches = _GRAPH_BUCKETS[bucket]
    return [
        _problem(*lens, seed=seed + i, **shape, **overrides)
        for i, lens in enumerate(batches)
    ]


def _capture_buffers(p, lse_mode):
    dev = torch.device(DEVICE)
    q, k, v = p["q"].clone(), p["k_cache"].clone(), p["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p["num_qo_heads"], HEAD_DIM, dtype=q.dtype, device=dev
    )
    lse = (
        torch.empty(q.shape[0], p["num_qo_heads"], dtype=torch.float32, device=dev)
        if lse_mode != "none"
        else None
    )
    return q, k, v, out, lse


def _load(buffers, p):
    """Swap the CONTENTS the captured graph reads to batch ``p``."""
    q, k, v = buffers[:3]
    q[: p["q"].shape[0]].copy_(p["q"])
    k.copy_(p["k_cache"])
    v.copy_(p["v_cache"])
    torch.cuda.synchronize()


def _clear(buffers):
    """Wipe out / lse, so an earlier launch's result cannot stand in for the
    next one's."""
    out, lse = buffers[3:]
    out.zero_()
    if lse is not None:
        lse.fill_(float("nan"))


def _replay(graph, buffers):
    _clear(buffers)
    graph.replay()
    torch.cuda.synchronize()


def _update(attn, p):
    """update() must not synchronize: it is the per-step host path."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        attn.update(make_metadata(p))
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _capture_update_replay(
    monkeypatch, batches, *, lse_mode, arm, causal=True, scales=None
):
    """Plan ``batches[0]``, capture one run, replay; then update() with each
    later batch and replay the same capture, and run the updated plan eagerly
    too.  Arms:

    - ``any_engine``: the prepared path where the gate holds.  update() keeps
      the very plan object the capture was made with (its bindings are the
      reserved buffers), no cuDNN graph is prepared after the capture, and
      the eager run after an update does no plan-time work.
    - ``backend``: the same with cuDNN's backend engines alone, on the
      private graph cache (provider control; the engines are checked).
    - ``functional``: no staged plan from the first plan on and through every
      update(), so the capture and every eager run go through the functional
      entry -- the path below the library gate.
    """
    scales = scales or {}
    p1 = batches[0]
    if arm == "functional":
        _no_staged_plan(monkeypatch)
    # plan before the provider control: the one-time packed-LSE probe of this
    # process must not be answered (and cached) under it
    attn, backend = _plan_cudnn(
        p1, causal=causal, lse_mode=lse_mode, use_cuda_graph=True
    )
    staged = arm != "functional" and _assert_staged(backend)
    captured_plan = backend._cudnn_plan
    assert (captured_plan is not None) == staged
    graphs = _isolate_graph_cache(monkeypatch)
    is_backend_engine = _backend_engines_only(monkeypatch) if arm == "backend" else None
    entry_calls = _spy_functional_entry(monkeypatch)
    buffers = _capture_buffers(p1, lse_mode)
    q, k, v, out, lse = buffers
    graph = _warm_and_capture(attn, q, (k, v), out, lse, **scales)
    assert graphs  # the graph under capture came from the private cache
    if is_backend_engine is not None:
        assert all(
            is_backend_engine(g.get_engine_and_knobs_at_index(g._plan_index)[0])
            for g in graphs
        )
    # the two warm-up runs and the captured one
    assert len(entry_calls) == (0 if staged else 3)
    preparations = _count_preparations(monkeypatch)
    counts = _count_host_work(monkeypatch)

    def check(p):
        _assert_rows_match(out, lse, p, lse_mode=lse_mode, causal=causal, **scales)

    _replay(graph, buffers)
    check(p1)
    for p in batches[1:]:
        _load(buffers, p)
        _update(attn, p)
        assert backend._cudnn_plan is captured_plan
        _replay(graph, buffers)
        check(p)
        # the updated plan also serves an eager run of the same instance
        _clear(buffers)
        counts.clear()
        entry_calls.clear()
        attn.run(q, (k, v), out=out, lse=lse, **scales)
        check(p)
        if staged:
            assert not entry_calls
            _assert_hot_path(counts, 1)
        else:
            assert len(entry_calls) == 1
    if staged:
        assert not preparations


@pytest.mark.parametrize("arm", ["any_engine", "backend", "functional"])
@pytest.mark.parametrize("bucket", list(_GRAPH_BUCKETS))
@pytest.mark.parametrize(
    "lse_mode,input_form",
    [
        ("none", "block_tables"),
        ("base2", "block_tables"),
        ("basee", "block_tables"),
        ("base2", "page_indices"),  # the dense table is derived and reserved
    ],
)
def test_cuda_graph_update_with_changed_lengths_and_page_ids(
    monkeypatch, lse_mode, input_form, bucket, arm
):
    """Plan, capture one run, replay; then update() twice with changed
    lengths, query partitions and page ids and replay the same capture (see
    ``_capture_update_replay`` for the arms).  In the ``large`` bucket the
    ``any_engine`` arm may run on the frontend's own prefill engine, which
    the ``backend`` arm excludes."""
    batches = _graph_batches(bucket, seed=121, input_form=input_form)
    assert not torch.equal(batches[0]["block_tables"], batches[1]["block_tables"])
    _capture_update_replay(monkeypatch, batches, lse_mode=lse_mode, arm=arm)


@pytest.mark.parametrize("arm", ["any_engine", "functional"])
@pytest.mark.parametrize(
    "kv_layout,causal,scales",
    [
        ("NHD", True, None),
        ("HND", False, None),
        ("NHD", False, None),
        ("HND", True, _RUN_SCALES),
        ("NHD", False, _RUN_SCALES),
    ],
    ids=["nhd", "noncausal", "nhd_noncausal", "scales", "nhd_noncausal_scales"],
)
def test_cuda_graph_layouts_masks_and_run_time_scales(
    monkeypatch, kv_layout, causal, scales, arm
):
    """The same capture / update() / replay lifetime with an NHD cache (the
    capture binds permuted K/V views), without the causal mask, and with
    sm_scale / k_scale / v_scale given to the captured run: the capture keeps
    its scales across update(), as it keeps its buffers."""
    batches = _graph_batches("small", seed=131, kv_layout=kv_layout)
    _capture_update_replay(
        monkeypatch, batches, lse_mode="base2", arm=arm, causal=causal, scales=scales
    )


def test_cuda_graph_two_captures_with_different_scales(monkeypatch):
    """One instance, two captures: the second with other run-time scales, so
    with another prepared cuDNN graph, which replaces the first as the
    instance's last prepared one.  Both captures stay valid: after every
    update() each replays the new batch with its own scales, in either
    order, and nothing more is prepared -- not by the replays, and not by
    eager runs with either scale (both graphs are in the history)."""
    p1, p2, p3 = _graph_batches("small", seed=141)
    attn, backend = _plan_cudnn(p1, lse_mode="base2", use_cuda_graph=True)
    staged = _assert_staged(backend)
    captures = []
    for scales in ({}, _RUN_SCALES):
        buffers = _capture_buffers(p1, "base2")
        q, k, v, out, lse = buffers
        graph = _warm_and_capture(attn, q, (k, v), out, lse, **scales)
        captures.append((graph, buffers, scales))
    preparations = _count_preparations(monkeypatch)
    if staged:
        assert len(backend._prepared_history) == 2
        _forbid_functional_entry(monkeypatch)

    for step, p in enumerate((p1, p2, p3)):
        if step:
            for _, buffers, _ in captures:
                _load(buffers, p)
            _update(attn, p)
        for graph, buffers, scales in captures[:: -1 if step % 2 else 1]:
            _replay(graph, buffers)
            _assert_rows_match(*buffers[3:], p, lse_mode="base2", **scales)
    for _, buffers, scales in captures:
        q, k, v, out, lse = buffers
        _clear(buffers)
        attn.run(q, (k, v), out=out, lse=lse, **scales)
        _assert_rows_match(out, lse, p3, lse_mode="base2", **scales)
    if staged:
        assert not preparations


def test_cuda_graph_capture_is_the_instances_first_run():
    """The lifecycle rule is one eager run() per instance before capture.
    This pins what skipping it does where another instance of the same bucket
    already built the cuDNN graph (layers sharing one shape): the captured
    run is this instance's first, on the prepared path a miss of the (empty)
    history that fetches the graph from the process-wide cache under
    capture.  As with the functional entry, it builds nothing there, and the
    capture survives update() like a warmed one."""
    p1, p2, _ = _graph_batches("small", seed=144)
    buffers = _capture_buffers(p1, "base2")
    q, k, v, out, lse = buffers
    warm, _ = _plan_cudnn(p1, lse_mode="base2", use_cuda_graph=True)
    _warm_up(warm, q, (k, v), out, lse)

    attn, backend = _plan_cudnn(p1, lse_mode="base2", use_cuda_graph=True)
    staged = _assert_staged(backend)
    assert backend._prepared is None
    builds = cudnn_prefill._prefill_graph_builds
    graph = _capture(attn, q, (k, v), out, lse)
    assert cudnn_prefill._prefill_graph_builds == builds
    if staged:
        assert backend._prepared is not None
    _replay(graph, buffers)
    _assert_rows_match(out, lse, p1, lse_mode="base2")
    _load(buffers, p2)
    _update(attn, p2)
    _replay(graph, buffers)
    _assert_rows_match(out, lse, p2, lse_mode="base2")


def test_cuda_graph_staging_failure_keeps_the_captured_state(monkeypatch):
    """An update() whose cuDNN staging raises must leave everything a captured
    run reads as it was: the reserved buffers (the controller's rollback) and
    the backend-owned LSE gather indices, which the backend refills in place
    only after staging.  Pinned on the padded-stats (gather) LSE path, the
    one with such indices."""
    dev = torch.device(DEVICE)
    monkeypatch.setitem(cudnn_backend._PACKED_LSE_SUPPORTED, dev, False)
    p1, p2, _ = _graph_batches("small", seed=124)
    attn, backend = _plan_cudnn(p1, lse_mode="base2", use_cuda_graph=True)
    _assert_staged(backend)
    assert not backend.lse_written_packed
    buffers = _capture_buffers(p1, "base2")
    q, k, v, out, lse = buffers
    graph = _warm_and_capture(attn, q, (k, v), out, lse)
    kept_plan = backend._cudnn_plan
    batch_ids, pos = backend._batch_ids.clone(), backend._pos.clone()

    def failing(meta, derived, block_tables, path):
        raise RuntimeError("injected staging failure")

    with monkeypatch.context() as m:
        m.setattr(backend, "_stage_cudnn_plan", failing)
        with pytest.raises(RuntimeError, match="injected staging failure"):
            attn.update(make_metadata(p2))
    assert backend._cudnn_plan is kept_plan
    assert torch.equal(backend._batch_ids, batch_ids)
    assert torch.equal(backend._pos, pos)
    _replay(graph, buffers)
    _assert_rows_match(out, lse, p1, lse_mode="base2")

    # and the instance still takes a valid update
    _load(buffers, p2)
    attn.update(make_metadata(p2))
    assert not torch.equal(backend._batch_ids, batch_ids)  # the partition changed
    _replay(graph, buffers)
    _assert_rows_match(out, lse, p2, lse_mode="base2")


# ---------------------------------------------------------------------------
# 6: single-token batches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("heads", [(8, 2), (4, 4)], ids=["gqa", "mha"])
@pytest.mark.parametrize("batch", list(_LSE_LAYOUTS))
def test_single_token_batches(monkeypatch, batch, heads, lse_mode):
    """The three LSE layouts of the backend on the prepared path, each written
    into (or copied to) the caller's packed buffer."""
    q_lens, layout = _LSE_LAYOUTS[batch]
    p = _problem(q_lens, _LSE_LAYOUT_KV, seed=131, heads=heads)
    attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
    staged = _assert_staged(backend)
    if staged:
        _forbid_functional_entry(monkeypatch)
    lse_buf = None
    if lse_mode != "none":
        # "direct" needs this cuDNN's ragged stats (a one-time probe); without
        # them the backend gathers and only the numbers are checked
        if layout != "direct":
            assert backend.lse_written_packed == (layout == "view")
        lse_buf = torch.full(
            (p["q"].shape[0], heads[0]),
            float("nan"),
            dtype=torch.float32,
            device=DEVICE,
        )
    for _ in range(2):
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), lse=lse_buf)
        assert lse is lse_buf
        assert_matches(out, lse, p, lse_mode=lse_mode)


# ---------------------------------------------------------------------------
# 7: storage offsets
# ---------------------------------------------------------------------------


def test_q_out_lse_with_storage_offsets(monkeypatch):
    """q, out and lse as packed views into larger allocations (non-zero
    storage offset): the rows around q are NaN and the memory around out /
    lse holds canaries.  Pointers are not part of a graph signature, so the
    packed q itself then runs on the same prepared graph."""
    p = _problem(*_RAGGED, seed=141)
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    total = p["q"].shape[0]
    pad = 3
    q_big = torch.full(
        (total + 2 * pad, *p["q"].shape[1:]),
        float("nan"),
        dtype=p["dtype"],
        device=DEVICE,
    )
    q_big[pad : pad + total] = p["q"]
    q_view = q_big[pad : pad + total]
    out_big, out_view, lse_big, lse_view = canary_buffers(p)
    assert q_view.is_contiguous() and q_view.storage_offset() > 0
    assert out_view.storage_offset() > 0 and lse_view.storage_offset() > 0

    kv = (p["k_cache"], p["v_cache"])
    out, lse = attn.run(q_view, kv, out=out_view, lse=lse_view)
    assert out is out_view and lse is lse_view
    assert_matches(out, lse, p, lse_mode="base2")
    assert_canaries(out_big, out_view, 64)
    assert_canaries(lse_big, lse_view, 64)

    out, lse = attn.run(p["q"], kv)
    assert_matches(out, lse, p, lse_mode="base2")
    if staged:
        assert len(preparations) == 1
