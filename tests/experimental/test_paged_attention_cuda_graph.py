"""CUDA-graph re-plan protocol for PagedAttention (experimental).

Contract under test (``PagedAttention(use_cuda_graph=True)``):

1. ``run()`` planned once can be captured into a ``torch.cuda.CUDAGraph``.
2. A later ``plan()`` with a new batch that FITS the capacity re-fills the
   reserved storage; replaying the captured graph then computes the new batch
   — no re-capture.  Batch size, paging form, page size and dense table width
   are exact; total query tokens, host maxes and flat page-id length are upper
   bounds (the kernels were planned with the capacity values: capacity
   substitution).
3. A plan outside the capacity is rejected before anything is written.
4. A plan that fails midway (here: the causal envelope) leaves the previously
   published plan runnable and the reserved buffers untouched — replay still
   produces the previous batch's answer.  This holds for a failure at ANY
   staging copy: Python never calls ``__exit__`` when ``__enter__`` raises,
   so ``Transaction.__enter__`` restores the destinations it already wrote.
5. Re-plan is sync-free when the host mirrors are supplied.
6. A FIRST plan that fails (backend construction or the backend's own plan)
   installs nothing: the next plan may fix any shape, including the batch
   size.
7. The first successful graph-mode plan freezes the semantic contract (the
   backend and every non-per-batch ``PlanMetadata`` field); a later plan that
   changes any of them is rejected before anything is written, because the
   captured graph would keep launching the old kernels.
8. ``PagedAttention(graph_capacity=GraphCapacity(...))`` allocates the
   reserved storage at construction; in the flat ``kv_page_indices`` form the
   dense block table is reserved only for a backend that reads it (never at
   ``page_size < 8``, where it would be ``(batch, max_context)``).
9. ``update(metadata)`` is the graph-mode re-plan with the frozen semantic
   kwargs: same staging path as ``plan()``, rejected before a plan, in eager
   mode, and during capture (as is a graph-mode ``plan()``).
"""

import dataclasses
import gc

import pytest
import torch

from flashinfer.experimental.paged_attention._graph import GraphBuffers, Transaction
from flashinfer.prefill import (
    GraphCapacity,
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    BACKENDS,
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SHAPE = dict(
    batch_size=4,
    max_q=32,
    max_kv=256,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)


def _sibling_batch(p, seed, *, shrink=False):
    """A second batch with the SAME capture shapes as ``p`` (batch size, total
    query tokens, table width, maxes) but different per-request lengths, a
    different page permutation and fresh K/V/q contents — what a serving engine
    replays into one graph bucket.  With ``shrink=True`` the sibling has FEWER
    total query tokens and smaller actual maxes (the capacity-substitution
    case); the table width is unchanged."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(p["device"])
    b = p["kv_seq_lens_cpu"].shape[0]
    q_lens = p["qo_indptr_cpu"].diff()
    if shrink:
        q_lens = torch.clamp(q_lens // 2, min=1)
        kv_cap = p["max_kv_len"] // 2
    else:
        q_lens = q_lens[torch.randperm(b, generator=g)]  # same multiset -> same total
        kv_cap = p["max_kv_len"] - 1
    kv_lens = torch.minimum(
        q_lens + torch.randint(0, kv_cap, (b,), generator=g),
        torch.tensor(kv_cap, dtype=torch.int32),
    ).to(torch.int32)
    kv_lens = torch.maximum(kv_lens, q_lens)  # causal envelope
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    page = p["page_size"]
    pages = (kv_lens + page - 1) // page
    width = p["block_tables"].shape[1]
    assert int(pages.max()) <= width
    pool = p["k_cache"].shape[0]
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    bt = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages[i])
        bt[i, :n] = perm[off : off + n]
        off += n
    kv_page_indices = torch.cat([bt[i, : int(pages[i])] for i in range(b)])
    q = torch.randn(
        int(qo_indptr_cpu[-1]), *p["q"].shape[1:], dtype=p["q"].dtype, device=dev
    )
    k = torch.randn_like(p["k_cache"])
    v = torch.randn_like(p["v_cache"])
    return dict(
        p,
        q=q,
        k_cache=k,
        v_cache=v,
        k_ref=k,
        v_ref=v,
        qo_indptr=qo_indptr_cpu.to(dev),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(dev),
        kv_seq_lens_cpu=kv_lens,
        block_tables=bt.to(dev),
        kv_page_indices=kv_page_indices.to(torch.int32).to(dev),
        max_q_len=int(q_lens.max()) if shrink else p["max_q_len"],
        max_kv_len=int(kv_lens.max()) if shrink else p["max_kv_len"],
    )


def _plan(attn, p, backend):
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend=backend,
    )


def _plan_with(attn, p, **overrides):
    kwargs = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    kwargs.update(overrides)
    attn.plan(make_metadata(p), **kwargs)


def _graph_capacity_of(p, *, extra_rows=0, extra_q=0):
    """Explicit capacity for one make_problem() batch (dense form), optionally
    with headroom; ``max_kv_len`` is the table's full extent, as the width
    rule requires."""
    width = int(p["block_tables"].shape[1])
    return GraphCapacity(
        batch_size=int(p["kv_seq_lens_cpu"].shape[0]),
        total_q_tokens=int(p["qo_indptr_cpu"][-1]) + extra_rows,
        max_q_len=p["max_q_len"] + extra_q,
        max_kv_len=width * p["page_size"],
        page_size=p["page_size"],
        table_width=width,
    )


def _reference(p):
    return reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_capture_replan_replay(backend):
    p1 = make_problem(seed=41, **_SHAPE)
    _resolve_or_skip(p1, backend)
    p2 = _sibling_batch(p1, seed=42)
    dev = torch.device(p1["device"])

    attn = PagedAttention(dev, use_cuda_graph=True)
    # static input/output storage the graph will read and write
    q = p1["q"].clone()
    k = p1["k_cache"].clone()
    v = p1["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p1["num_qo_heads"], dtype=torch.float32, device=dev)

    _plan(attn, p1, backend)
    # warm up on a side stream (module load / graph build), then capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)

    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)

    # re-plan the sibling batch into the reserved storage (sync-free), swap the
    # tensor CONTENTS the graph reads, replay: the captured graph computes p2
    q.copy_(p2["q"])
    k.copy_(p2["k_cache"])
    v.copy_(p2["v_cache"])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _plan(attn, p2, backend)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    g.replay()
    torch.cuda.synchronize()
    ref_out2, ref_lse2 = _reference(p2)
    torch.testing.assert_close(out.float(), ref_out2, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse2, **LSE_TOL)
    assert not torch.allclose(ref_out, ref_out2)  # the two batches really differ


@pytest.mark.parametrize("backend", BACKENDS)
def test_replan_rejects_capture_shape_drift(backend):
    p1 = make_problem(seed=43, **_SHAPE)
    _resolve_or_skip(p1, backend)
    attn = PagedAttention(torch.device(p1["device"]), use_cuda_graph=True)
    _plan(attn, p1, backend)
    smaller = make_problem(seed=44, **dict(_SHAPE, batch_size=3))
    with pytest.raises(ValueError, match="CUDA graph re-plan: batch_size"):
        _plan(attn, smaller, backend)
    # the published plan is intact
    assert attn.backend is not None
    attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_failed_replan_restores_previous_plan(backend):
    p1 = make_problem(seed=45, **_SHAPE)
    _resolve_or_skip(p1, backend)
    dev = torch.device(p1["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p1["q"].clone(), p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, backend)
    attn.run(q, (k, v), out=out, lse=lse)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)

    # a sibling batch that violates the causal envelope on one request
    # (q_len > kv_len): metadata builds fine, plan() must fail AFTER the
    # metadata is accepted and BEFORE anything is published
    bad = _sibling_batch(p1, seed=46)
    kv_bad = bad["kv_seq_lens_cpu"].clone()
    i = int(bad["qo_indptr_cpu"].diff().argmax())
    kv_bad[i] = max(1, int(bad["qo_indptr_cpu"].diff()[i]) - 1)
    bad["kv_seq_lens_cpu"] = kv_bad
    bad["kv_seq_lens"] = kv_bad.to(dev)
    with pytest.raises(ValueError, match="causal masking requires"):
        _plan(attn, bad, backend)

    # and a failure INSIDE the transaction (the backend's own plan raising
    # after the reserved buffers were already overwritten) must roll them back
    good_sibling = _sibling_batch(p1, seed=47)
    active = attn._impl._active
    real_plan = active.plan

    def boom(meta, derived):
        raise RuntimeError("injected backend plan failure")

    active.plan = boom
    try:
        with pytest.raises(RuntimeError, match="injected"):
            _plan(attn, good_sibling, backend)
    finally:
        active.plan = real_plan

    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


class _FakeBuffer:
    """CPU stand-in for a reserved tensor: records writes, fails on demand."""

    def __init__(self, value, fail=False):
        self.value, self.fail = value, fail

    def clone(self):
        return _FakeBuffer(self.value)

    def copy_(self, other, **kwargs):
        if self.fail:
            raise RuntimeError("injected copy failure")
        self.value = other.value


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_transaction_enter_restores_earlier_copies(fail_at):
    """A copy that fails inside ``Transaction.__enter__`` must restore every
    destination written before it: Python does not call ``__exit__`` when
    ``__enter__`` raises, so the rollback has to happen right there."""
    dsts = [_FakeBuffer(i, fail=(i == fail_at)) for i in range(3)]
    srcs = [_FakeBuffer(10 + i) for i in range(3)]
    with (
        pytest.raises(RuntimeError, match="injected"),
        Transaction(list(zip(dsts, srcs, strict=True))),
    ):
        pass
    assert [d.value for d in dsts] == [0, 1, 2]


# dense form: qo_indptr, kv_seq_lens, q_seq_lens, cum_kv_seq_lens,
# kv_page_indptr, block_tables, kv_page_indices
_DENSE_STAGING_POSITIONS = 7


@pytest.mark.parametrize("fail_at", range(_DENSE_STAGING_POSITIONS))
def test_failed_staging_copy_restores_previous_plan(fail_at):
    """Inject a synchronous copy failure (a source of the wrong shape) at each
    staging position of a graph re-plan: every reserved buffer must hold the
    previous plan's contents afterwards, the published plan must be the
    previous one, and the captured graph must still compute the previous
    batch."""
    p1 = make_problem(seed=48, **_SHAPE)
    _resolve_or_skip(p1, "fa2")
    dev = torch.device(p1["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p1["q"].clone(), p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, "fa2")
    attn.run(q, (k, v), out=out, lse=lse)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.synchronize()

    gb = attn._impl._graph
    meta_before = attn._impl._meta
    reserved = {
        name: getattr(gb, name).clone()
        for name in (
            "qo_indptr",
            "kv_seq_lens",
            "q_seq_lens",
            "cum_kv_seq_lens",
            "kv_page_indptr",
            "block_tables",
            "kv_page_indices",
        )
    }
    real_targets = gb.targets

    def bad_targets(metadata, fresh):
        pairs = real_targets(metadata, fresh)
        assert len(pairs) == _DENSE_STAGING_POSITIONS
        dst, _ = pairs[fail_at]
        wrong = torch.zeros(dst.numel() + 1, dtype=dst.dtype, device=dst.device)
        pairs[fail_at] = (dst, wrong)  # copy_ raises synchronously (no broadcast)
        return pairs

    gb.targets = bad_targets
    try:
        with pytest.raises(RuntimeError):
            _plan(attn, _sibling_batch(p1, seed=49), "fa2")
    finally:
        del gb.targets
    torch.cuda.synchronize()

    for name, snap in reserved.items():
        assert torch.equal(getattr(gb, name), snap), f"{name} not restored"
    assert attn._impl._meta is meta_before
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("stage", ["construct", "plan"])
def test_failed_first_plan_leaves_no_capacity(monkeypatch, stage):
    """A first graph-mode plan that fails while building the backend or inside
    the backend's own plan must not install a capacity: the next plan, with a
    different batch size, must succeed (mla-alignment F3: the capacity of a
    graph that was never captured locked the instance)."""
    from flashinfer.experimental.paged_attention import _controller
    from flashinfer.experimental.paged_attention._backends import fa_backend

    p2 = make_problem(seed=50, **_SHAPE)
    _resolve_or_skip(p2, "fa2")
    attn = PagedAttention(torch.device(p2["device"]), use_cuda_graph=True)

    failed = []
    if stage == "construct":
        real = _controller.make_backend

        def fail_once(*args, **kwargs):
            if not failed:
                failed.append(stage)
                raise RuntimeError("injected backend construction failure")
            return real(*args, **kwargs)

        monkeypatch.setattr(_controller, "make_backend", fail_once)
    else:
        real_plan = fa_backend._FaBackend.plan

        def fail_once(self, meta, derived):
            if not failed:
                failed.append(stage)
                raise RuntimeError("injected backend plan failure")
            return real_plan(self, meta, derived)

        monkeypatch.setattr(fa_backend._FaBackend, "plan", fail_once)

    with pytest.raises(RuntimeError, match="injected"):
        _plan(attn, p2, "fa2")
    assert failed == [stage]
    assert attn.backend is None
    assert attn._impl._graph is None, "a failed first plan installed a capacity"

    p3 = make_problem(seed=51, **dict(_SHAPE, batch_size=3))
    _plan(attn, p3, "fa2")  # a different batch size is still free to choose
    assert attn._impl._graph.capacity.batch_size == 3
    out, lse = attn.run(p3["q"], (p3["k_cache"], p3["v_cache"]))
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p3)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_frozen_contract_covers_every_semantic_field():
    """The frozen set is derived from PlanMetadata, so a semantic kwarg added
    later is frozen automatically; only the per-batch values are exempt."""
    from flashinfer.experimental.paged_attention import _controller
    from flashinfer.experimental.paged_attention._contracts import PlanMetadata

    names = {f.name for f in dataclasses.fields(PlanMetadata)}
    semantic = set(_controller._semantic_fields())
    assert semantic == names - _controller._PER_BATCH_FIELDS
    assert semantic >= {
        "kv_input_form",
        "page_size",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim_qk",
        "head_dim_vo",
        "q_dtype",
        "kv_dtype",
        "causal",
        "window_left",
        "kv_layout",
        "lse_mode",
    }


# (frozen field named in the error, plan() override that changes it)
_SEMANTIC_DRIFT = [
    ("causal", dict(causal=False)),  # mla-alignment F1
    ("window_left", dict(window_left=16)),
    ("lse_mode", dict(lse_mode="basee")),
    ("q_dtype", dict(q_dtype=torch.float16)),
    ("head_dim_qk", dict(head_dim_qk=64)),
    ("num_qo_heads", dict(num_qo_heads=4)),
    ("kv_layout", dict(kv_layout="NHD")),
    ("backend", dict(backend="cudnn")),
]


@pytest.mark.parametrize(
    "field,drift", _SEMANTIC_DRIFT, ids=[field for field, _ in _SEMANTIC_DRIFT]
)
def test_graph_replan_rejects_semantic_drift(field, drift):
    """Capture with one configuration, re-plan the SAME metadata with one
    semantic kwarg changed: the plan must be rejected naming the field, and
    the captured graph must still replay the captured configuration.  Before
    the frozen contract the causal=False re-plan was accepted and replay
    returned the stale causal result (mla-alignment F1)."""
    p = make_problem(seed=52, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    if field == "backend":
        _resolve_or_skip(p, drift["backend"])
    dev = torch.device(p["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p["q"].clone(), p["k_cache"].clone(), p["v_cache"].clone()
    out = torch.empty(
        q.shape[0], p["num_qo_heads"], p["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan_with(attn, p)
    attn.run(q, (k, v), out=out, lse=lse)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    g.replay()
    torch.cuda.synchronize()
    before_out, before_lse = out.clone(), lse.clone()

    with pytest.raises(ValueError, match=rf"CUDA graph re-plan: {field} changed"):
        _plan_with(attn, p, **drift)

    # nothing moved: the published plan and the reserved buffers still
    # describe the captured configuration, and replay is still correct for it
    assert attn.backend == "fa2"
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, before_out) and torch.equal(lse, before_lse)
    ref_out, ref_lse = _reference(p)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_graph_capacity_validation():
    dense = dict(
        batch_size=2, total_q_tokens=8, max_q_len=4, max_kv_len=64, page_size=16
    )
    cap = GraphCapacity(**dense, table_width=4)
    assert cap.flat_capacity == 8 and cap.dense_table_width == 4
    with pytest.raises(ValueError, match="requires table_width"):
        GraphCapacity(**dense)
    with pytest.raises(ValueError, match="too narrow"):
        GraphCapacity(**dict(dense, max_kv_len=65), table_width=4)
    with pytest.raises(ValueError, match="CUDNN_STATUS_BAD_PARAM"):
        GraphCapacity(**dense, table_width=5)  # wider than ceil(64 / 16)
    assert GraphCapacity(**dict(dense, max_kv_len=49), table_width=4).max_kv_len == 49
    with pytest.raises(ValueError, match="leave it unset"):
        GraphCapacity(**dense, table_width=4, flat_capacity=9)
    with pytest.raises(ValueError, match="page_size"):
        GraphCapacity(**dict(dense, page_size=1), table_width=64)  # dense floor
    with pytest.raises(ValueError, match="total_q_tokens"):
        GraphCapacity(**dict(dense, total_q_tokens=1), table_width=4)
    with pytest.raises(ValueError, match="max_q_len"):
        GraphCapacity(**dict(dense, max_q_len=9), table_width=4)
    with pytest.raises(ValueError, match="positive host int"):
        GraphCapacity(**dict(dense, batch_size=0), table_width=4)

    flat = dict(dense, page_size=1, kv_input_form="page_indices")
    csr = GraphCapacity(**flat, flat_capacity=128)
    assert csr.dense_table_width == 64 and csr.table_width is None
    with pytest.raises(ValueError, match="requires flat_capacity"):
        GraphCapacity(**flat)
    with pytest.raises(ValueError, match="do not pass table_width"):
        GraphCapacity(**flat, flat_capacity=128, table_width=64)
    with pytest.raises(ValueError, match="cannot hold"):
        GraphCapacity(**flat, flat_capacity=63)
    with pytest.raises(ValueError, match="kv_input_form"):
        GraphCapacity(**dict(dense, kv_input_form="csr"), table_width=4)


def test_explicit_capacity_allocates_at_construction():
    """With graph_capacity= the reserved storage exists before the first plan,
    the first plan publishes that same storage, and a batch outside the
    capacity is rejected against it."""
    p = make_problem(seed=53, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    cap = _graph_capacity_of(p)
    attn = PagedAttention(dev, graph_capacity=cap)
    gb = attn._impl._graph
    assert gb is not None and gb.capacity == cap
    assert tuple(gb.block_tables.shape) == (cap.batch_size, cap.table_width)
    assert attn.backend is None

    _plan(attn, p, "fa2")
    assert attn._impl._graph is gb
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)

    other = make_problem(seed=54, **dict(_SHAPE, batch_size=3))
    with pytest.raises(ValueError, match="CUDA graph re-plan: batch_size"):
        _plan(attn, other, "fa2")
    with pytest.raises(ValueError, match="must be a GraphCapacity"):
        PagedAttention(dev, graph_capacity=dict(batch_size=4))


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen"])
def test_csr_graph_instance_reserves_dense_table_only_for_dense_backends(backend):
    """Flat page-id form at page_size 16: the dense table is reserved lazily
    at the first plan, only if the chosen backend reads it; the reserved
    table then carries the derived dense form through capture and re-plan."""
    p = make_problem(seed=55, **dict(_SHAPE, input_form="page_indices"))
    _resolve_or_skip(p, backend)
    dev = torch.device(p["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    _plan(attn, p, backend)
    gb = attn._impl._graph
    needs_dense = backend in ("cudnn", "trtllm-gen")
    if needs_dense:
        width = (p["max_kv_len"] + p["page_size"] - 1) // p["page_size"]
        assert tuple(gb.block_tables.shape) == (gb.capacity.batch_size, width)
    else:
        assert gb.block_tables is None
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize()
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        None,
        p["page_size"],
        True,
        kv_page_indices=p["kv_page_indices"],
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_dense_graph_instance_mirrors_the_caller_width_table():
    p = make_problem(seed=56, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    attn = PagedAttention(torch.device(p["device"]), use_cuda_graph=True)
    _plan(attn, p, "fa2")
    assert tuple(attn._impl._graph.block_tables.shape) == tuple(p["block_tables"].shape)


def test_csr_page_size_1_graph_instance_reserves_no_dense_table():
    """sglang shape: page_size=1 flat page ids, one 128K request in a batch of
    64.  A dense (batch, max_kv) table would be 32 MiB per bucket; a
    CSR-native backend must not pay for it (findings ledger M1)."""
    dev = torch.device("cuda:0")
    b, long_kv = 64, 128 * 1024
    kv_lens = torch.ones(b, dtype=torch.int32)
    kv_lens[0] = long_kv
    qo_indptr_cpu = torch.arange(b + 1, dtype=torch.int32)  # one query token each
    n_pages = int(kv_lens.sum())  # page_size=1: one page per token
    kv_page_indices = torch.arange(n_pages, dtype=torch.int32, device=dev)
    semantic = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
    )
    try:
        resolve_paged_attention(
            device=dev,
            page_size=1,
            need_lse=True,
            kv_input_form="page_indices",
            backend="fa2",
            **{k: v for k, v in semantic.items() if k not in ("causal", "lse_mode")},
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")

    def metadata():
        return PagedAttentionMetadata.csr(
            qo_indptr_cpu.to(dev),
            kv_lens.to(dev),
            kv_page_indices,
            page_size=1,
            max_q_len=1,
            max_kv_len=long_kv,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_lens,
        )

    # one-time costs (JIT module, per-device shared workspace) on a warm instance
    warm = PagedAttention(dev, use_cuda_graph=True)
    warm.plan(metadata(), backend="fa2", **semantic)
    torch.cuda.synchronize(dev)
    gc.collect()
    before = torch.cuda.memory_allocated(dev)

    attn = PagedAttention(dev, use_cuda_graph=True)
    attn.plan(metadata(), backend="fa2", **semantic)
    torch.cuda.synchronize(dev)
    gc.collect()
    growth = torch.cuda.memory_allocated(dev) - before

    assert attn._impl._graph.block_tables is None
    dense_bytes = b * long_kv * 4
    assert growth < dense_bytes // 2, (
        f"a page_size=1 CSR graph instance grew device memory by "
        f"{growth / 2**20:.1f} MiB; the dense (batch, max_kv) table it must not "
        f"reserve is {dense_bytes / 2**20:.0f} MiB"
    )


def test_inferred_dense_capacity_widens_max_kv_to_the_table():
    """use_cuda_graph=True: the inferred dense capacity spans the whole table
    (max_kv_len = width * page_size), so later batches may use any context
    the table can address and cuDNN's width rule holds."""
    p = make_problem(seed=61, **_SHAPE)
    md = make_metadata(p)
    cap = GraphCapacity.from_metadata(md)
    width = int(p["block_tables"].shape[1])
    assert cap.table_width == width
    assert cap.max_kv_len == width * p["page_size"] >= p["max_kv_len"]
    assert (cap.total_q_tokens, cap.max_q_len) == (md.total_q_tokens, p["max_q_len"])
    csr = GraphCapacity.from_metadata(make_metadata(dict(p, input_form="page_indices")))
    assert csr.max_kv_len == p["max_kv_len"] and csr.table_width is None
    assert csr.flat_capacity == int(p["kv_page_indices"].shape[0])


def test_replan_capacity_bounds():
    """Preflight rules: exact batch size / form / page size / dense width;
    total tokens, maxes and flat length bounded by the capacity."""
    dev = torch.device("cuda:0")
    page = 16

    def dense_md(q_lens, kv_lens, max_q, max_kv, width=4):
        q_lens = torch.tensor(q_lens, dtype=torch.int32)
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
        qo = torch.cat([torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0).int()])
        bt = torch.arange(len(kv_lens) * width, dtype=torch.int32).view(-1, width)
        return PagedAttentionMetadata.dense(
            qo.to(dev),
            kv_lens.to(dev),
            bt.to(dev),
            page_size=page,
            max_q_len=max_q,
            max_kv_len=max_kv,
            qo_indptr_cpu=qo,
            kv_seq_lens_cpu=kv_lens,
        )

    cap = GraphCapacity(
        batch_size=2,
        total_q_tokens=8,
        max_q_len=4,
        max_kv_len=64,
        page_size=page,
        table_width=4,
    )
    gb = GraphBuffers(cap, dev)
    gb.preflight(dense_md([4, 4], [64, 64], 4, 64))  # exactly the capacity
    gb.preflight(dense_md([1, 2], [10, 20], 2, 20))  # under it in every bound
    with pytest.raises(ValueError, match="total_q_tokens 9 exceeds"):
        gb.preflight(dense_md([4, 5], [64, 64], 5, 64))
    with pytest.raises(ValueError, match="max_q_len 5 exceeds"):
        gb.preflight(dense_md([3, 3], [64, 64], 5, 64))  # over-claimed max
    with pytest.raises(ValueError, match="block_tables width 5"):
        gb.preflight(dense_md([1, 1], [10, 10], 1, 10, width=5))
    with pytest.raises(ValueError, match="batch_size 3 differs"):
        gb.preflight(dense_md([1, 1, 1], [10, 10, 10], 1, 10))

    def csr_md(kv_lens, max_kv, n_ids):
        kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
        qo = torch.arange(len(kv_lens) + 1, dtype=torch.int32)
        return PagedAttentionMetadata.csr(
            qo.to(dev),
            kv_lens.to(dev),
            torch.arange(n_ids, dtype=torch.int32, device=dev),
            page_size=page,
            max_q_len=1,
            max_kv_len=max_kv,
            qo_indptr_cpu=qo,
            kv_seq_lens_cpu=kv_lens,
        )

    flat = GraphCapacity(
        batch_size=2,
        total_q_tokens=8,
        max_q_len=4,
        max_kv_len=64,
        page_size=page,
        kv_input_form="page_indices",
        flat_capacity=8,
    )
    gb = GraphBuffers(flat, dev)
    assert gb.block_tables is None
    gb.preflight(csr_md([64, 64], 64, 8))
    gb.preflight(csr_md([10, 20], 20, 3))  # fewer page ids than reserved
    with pytest.raises(ValueError, match="max_kv_len 65 exceeds"):
        gb.preflight(csr_md([10, 20], 65, 8))
    with pytest.raises(ValueError, match="kv_page_indices has 9 entries"):
        gb.preflight(csr_md([64, 64], 64, 9))
    with pytest.raises(ValueError, match="kv_input_form"):
        gb.preflight(dense_md([1, 1], [10, 10], 1, 10))


@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_capture_replan_smaller_batch_replay(backend, input_form):
    """Capacity semantics: a re-plan may use fewer total query tokens and
    smaller actual maxes than the capture.  Backends keep seeing the capacity
    maxes (capacity substitution), so the captured kernels read consistent
    parameters and replay matches the oracle on the batch's rows."""
    p1 = make_problem(seed=57, **dict(_SHAPE, input_form=input_form))
    _resolve_or_skip(p1, backend)
    p2 = _sibling_batch(p1, seed=58, shrink=True)
    total1, total2 = int(p1["qo_indptr_cpu"][-1]), int(p2["qo_indptr_cpu"][-1])
    assert total2 < total1
    assert p2["max_q_len"] < p1["max_q_len"] and p2["max_kv_len"] < p1["max_kv_len"]
    dev = torch.device(p1["device"])

    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p1["q"].clone(), p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        total1, p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(total1, p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, backend)
    cap = attn._impl._graph.capacity
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)

    q[:total2].copy_(p2["q"])
    k.copy_(p2["k_cache"])
    v.copy_(p2["v_cache"])
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _plan(attn, p2, backend)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    meta = attn._impl._meta
    assert (meta.max_q_len, meta.max_kv_len) == (cap.max_q_len, cap.max_kv_len)
    assert meta.total_q_tokens == total2
    g.replay()
    torch.cuda.synchronize()
    ref_out2, ref_lse2 = _reference(p2)
    torch.testing.assert_close(out[:total2].float(), ref_out2, **OUT_TOL)
    torch.testing.assert_close(lse[:total2], ref_lse2, **LSE_TOL)

    # an eager run on the capacity-sized buffers is accepted; too few rows are not
    out.zero_()
    attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:total2].float(), ref_out2, **OUT_TOL)
    with pytest.raises(ValueError, match="tokens"):
        attn.run(q[: total2 - 1], (k, v))


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_capacity_headroom_replay(backend):
    """An explicit GraphCapacity larger than the first batch: capture buffers
    are sized to the capacity, the first and the re-planned batch both use
    less, and a batch outside the capacity is rejected before anything is
    written."""
    p1 = make_problem(seed=59, **_SHAPE)
    _resolve_or_skip(p1, backend)
    p2 = _sibling_batch(p1, seed=60, shrink=True)
    total1, total2 = int(p1["qo_indptr_cpu"][-1]), int(p2["qo_indptr_cpu"][-1])
    dev = torch.device(p1["device"])
    cap = _graph_capacity_of(p1, extra_rows=16, extra_q=8)
    attn = PagedAttention(dev, graph_capacity=cap)

    rows = cap.total_q_tokens
    q = torch.zeros(
        rows, p1["num_qo_heads"], p1["head_dim_qk"], dtype=p1["dtype"], device=dev
    )
    q[:total1] = p1["q"]
    k, v = p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        rows, p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(rows, p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, backend)
    meta = attn._impl._meta
    assert (meta.max_q_len, meta.max_kv_len) == (cap.max_q_len, cap.max_kv_len)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p1)
    torch.testing.assert_close(out[:total1].float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse[:total1], ref_lse, **LSE_TOL)

    q[:total2].copy_(p2["q"])
    k.copy_(p2["k_cache"])
    v.copy_(p2["v_cache"])
    _plan(attn, p2, backend)
    g.replay()
    torch.cuda.synchronize()
    ref_out2, ref_lse2 = _reference(p2)
    torch.testing.assert_close(out[:total2].float(), ref_out2, **OUT_TOL)
    torch.testing.assert_close(lse[:total2], ref_lse2, **LSE_TOL)

    md = make_metadata(p1)
    over = PagedAttentionMetadata.dense(
        md.qo_indptr,
        md.kv_seq_lens,
        md.block_tables,
        page_size=md.page_size,
        max_q_len=cap.max_q_len + 1,  # over-claimed beyond the capacity
        max_kv_len=md.max_kv_len,
        qo_indptr_cpu=md.qo_indptr_cpu,
        kv_seq_lens_cpu=md.kv_seq_lens_cpu,
    )
    with pytest.raises(ValueError, match="max_q_len .* exceeds"):
        attn.plan(
            over,
            num_qo_heads=p1["num_qo_heads"],
            num_kv_heads=p1["num_kv_heads"],
            head_dim_qk=p1["head_dim_qk"],
            q_dtype=p1["dtype"],
            causal=True,
            lse_mode="base2",
            backend=backend,
        )
    g.replay()  # the rejected plan left the p2 plan in place
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:total2].float(), ref_out2, **OUT_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_update_sibling_replay(backend):
    """capture -> update(sibling) -> replay matches the oracle, for a
    same-shape sibling and for a smaller one; update() takes no semantic
    kwargs and reuses the frozen ones."""
    p1 = make_problem(seed=62, **_SHAPE)
    _resolve_or_skip(p1, backend)
    dev = torch.device(p1["device"])
    total1 = int(p1["qo_indptr_cpu"][-1])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p1["q"].clone(), p1["k_cache"].clone(), p1["v_cache"].clone()
    out = torch.empty(
        total1, p1["num_qo_heads"], p1["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(total1, p1["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p1, backend)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)

    for seed, shrink in ((63, False), (64, True)):
        p2 = _sibling_batch(p1, seed=seed, shrink=shrink)
        total2 = int(p2["qo_indptr_cpu"][-1])
        q[:total2].copy_(p2["q"])
        k.copy_(p2["k_cache"])
        v.copy_(p2["v_cache"])
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            assert attn.update(make_metadata(p2)) is attn
        finally:
            torch.cuda.set_sync_debug_mode("default")
        g.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = _reference(p2)
        torch.testing.assert_close(out[:total2].float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse[:total2], ref_lse, **LSE_TOL)
    assert attn.backend == attn._impl._frozen["backend"]


def test_update_before_plan_and_outside_graph_mode_raise():
    p = make_problem(seed=65, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    md = make_metadata(p)
    with pytest.raises(RuntimeError, match="before a successful plan"):
        PagedAttention(dev, use_cuda_graph=True).update(md)
    with pytest.raises(RuntimeError, match="before a successful plan"):
        PagedAttention(dev, graph_capacity=_graph_capacity_of(p)).update(md)
    eager = PagedAttention(dev)
    _plan(eager, p, "fa2")
    with pytest.raises(RuntimeError, match="CUDA-graph re-plan"):
        eager.update(md)


def test_plan_and_update_during_capture_raise():
    """Neither plan() nor update() may run while the current stream is
    capturing: their metadata copies and backend planning would be recorded
    into the graph instead of executed."""
    p = make_problem(seed=66, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    q, k, v = p["q"], p["k_cache"], p["v_cache"]
    out = torch.empty(
        q.shape[0], p["num_qo_heads"], p["head_dim_vo"], dtype=q.dtype, device=dev
    )
    lse = torch.empty(q.shape[0], p["num_qo_heads"], dtype=torch.float32, device=dev)
    _plan(attn, p, "fa2")
    attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.synchronize()
    md = make_metadata(_sibling_batch(p, seed=67))
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
        with pytest.raises(RuntimeError, match="during CUDA graph capture"):
            attn.update(md)
        with pytest.raises(RuntimeError, match="during CUDA graph capture"):
            _plan(attn, p, "fa2")
    # the capture itself is intact and computes the planned batch
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = _reference(p)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
