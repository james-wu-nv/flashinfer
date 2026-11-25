"""Measured selection of PagedAttention (experimental): ``backend="autotune"``.

The candidate set is the ``auto`` one.  plan() prepares every candidate that
accepts the batch and the first run() selects one: the cached result for the
planned workload, or, inside ``flashinfer.autotune(True)``, a profile on the
caller's own tensors.  A miss outside the context raises.  This is the MLA
wrapper's ``backend="autotune"`` on the unified lifecycle.
Profiling is made deterministic by ranking the candidates through a patched
timer; every candidate still launches once for real before profiling.
"""

import dataclasses

import pytest
import torch

import flashinfer
from flashinfer.autotuner import AutoTuner
from flashinfer.experimental.paged_attention import GraphCapacity
from flashinfer.experimental.paged_attention import _auto_policy
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .paged_attention_reference import scratch_workspace
from .test_paged_attention_prototype import (
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
    reference_paged_prefill,
)

_CFG = dict(
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    q_dtype=torch.bfloat16,
    page_size=16,
    causal=True,
    need_lse=True,
)

_SHAPE = dict(
    batch_size=3,
    max_q=48,
    max_kv=256,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# --------------------------------------------------------------------------
# resolution and workload key (no GPU)
# --------------------------------------------------------------------------


def test_autotune_resolves_the_auto_set_and_says_so():
    auto = resolve_paged_attention(cc_major=10, **_CFG)
    tuned = resolve_paged_attention(cc_major=10, backend="autotune", **_CFG)
    assert tuned.autotune and not auto.autotune
    assert tuned.backends == auto.backends and tuned.excluded == auto.excluded
    # a policy, not configuration: the drift key is the same
    assert tuned.config == auto.config
    assert "selection: autotune" in tuned.explain()
    assert "selection: auto (policy order)" in auto.explain()
    with pytest.raises(ValueError, match="'auto' or 'autotune'"):
        resolve_paged_attention(cc_major=10, backend="fastest", **_CFG)


def _workload(**changes):
    facts = dict(
        batch_size=3,
        total_q_tokens=100,
        max_q_len=48,
        max_kv_len=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=16,
        kv_layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        causal=True,
        window_left=-1,
        logits_soft_cap=None,
        custom_mask=False,
        use_sinks=False,
        use_cuda_graph=False,
    )
    key = dict(
        config=("cfg", torch.bfloat16),
        digest="d0",
        workspace_bytes=1024,
        device=torch.device("cuda:0"),
        candidates=("fa2", "trtllm-gen"),
    )
    for name, value in changes.items():
        (facts if name in facts else key)[name] = value
    return _auto_policy.autotune_workload(facts=_auto_policy._PlanFacts(**facts), **key)


@needs_cuda
def test_out_is_required(tuner):
    p, res, md, order = _setup(139)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    attn.plan(md, backend=res, **_plan_kw(p))
    with pytest.raises(ValueError, match="out must be provided"):
        attn.run(p["q"], (p["k_cache"], p["v_cache"]))


def test_workload_key_is_the_planned_workload():
    base = _workload()
    assert base == _workload()
    # nothing is bucketed: any planned fact, length, the workspace size, the
    # device or the prepared candidates makes another workload
    for name, value in (
        ("batch_size", 4),
        ("total_q_tokens", 101),
        ("max_q_len", 49),
        ("max_kv_len", 257),
        ("use_cuda_graph", True),
        ("config", ("other", torch.bfloat16)),
        ("digest", "d1"),
        ("workspace_bytes", 2048),
        ("device", torch.device("cuda:1")),
        ("candidates", ("fa2", "cudnn")),
    ):
        assert _workload(**{name: value}) != base, name

    # host values only: the key survives a round trip through str (file cache)
    def flat(value):
        if isinstance(value, tuple):
            for item in value:
                yield from flat(item)
        else:
            yield value

    assert all(isinstance(v, (str, int, bool)) for v in flat(base))


def test_metadata_digest_reads_lengths_and_page_ids():
    lens = torch.tensor([3, 5], dtype=torch.int32)
    pages = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    def digest(lens=lens, pages=pages, name="block_tables"):
        return _auto_policy.metadata_digest(
            (("kv_seq_lens", lens), (name, pages), ("kv_page_indices", None))
        )

    assert digest() == digest(lens.clone(), pages.clone())
    assert digest(lens=lens + 1) != digest()
    assert digest(pages=pages.flip(0)) != digest()  # the page ids count
    assert digest(pages=pages.reshape(4)) != digest()
    assert digest(name="kv_page_indices") != digest()


# --------------------------------------------------------------------------
# GPU: measured choice, cache reuse, graph mode
# --------------------------------------------------------------------------


@pytest.fixture
def tuner(monkeypatch):
    # isolate decisions without clearing another test's process-global cache
    instance = AutoTuner(warmup=1, repeat=2)
    monkeypatch.setattr(AutoTuner, "_instance", instance)
    return instance


def _rank(tuner, monkeypatch, winner):
    """Profile with a fake timer that makes ``winner`` the fastest; returns
    the set of candidate names the tuner timed."""
    timed = set()

    def fake_profile(runner, inputs, tactic, tuning_config, **kwargs):
        timed.add(runner.name)
        return 1.0 if runner.name == winner else 2.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", fake_profile)
    return timed


def _forbidden(*args, **kwargs):
    raise AssertionError("must not be called")


def _plan_kw(p):
    return dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
    )


def _check(p, out, lse):
    flat = p.get("input_form") == "page_indices"
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        None if flat else p["block_tables"],
        p["page_size"],
        True,
        kv_page_indices=p.get("kv_page_indices"),
    )
    rows = ref_out.shape[0]
    torch.testing.assert_close(out[:rows].float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse[:rows], ref_lse, **LSE_TOL)


def _setup(seed, input_form="block_tables"):
    p = make_problem(seed=seed, input_form=input_form, **_SHAPE)
    res = _resolve_or_skip(p, "autotune")
    if len(res.backends) < 2:
        pytest.skip("needs two runnable candidates to measure a choice")
    md = make_metadata(p)
    # the order the auto policy walks this batch in: what autotune measures
    # over, and what it falls back to
    auto = PagedAttention(scratch_workspace(torch.device(p["device"])))
    auto.plan(md, backend=dataclasses.replace(res, autotune=False), **_plan_kw(p))
    order = auto._impl.selection_order
    assert set(order) == set(res.backends)
    return p, res, md, order


def _run(attn, p):
    """run() with caller-owned outputs: autotune requires ``out``."""
    rows = p["q"].shape[0]
    out = torch.empty(
        rows,
        p["num_qo_heads"],
        p["head_dim_qk"],
        dtype=p["dtype"],
        device=p["q"].device,
    )
    lse = torch.empty(rows, p["num_qo_heads"], device=p["q"].device)
    return attn.run(p["q"], (p["k_cache"], p["v_cache"]), out=out, lse=lse)


def _selected(attn):
    """The candidate an autotune plan launches (``backend`` stays "autotune")."""
    return attn._impl._backend_name


def _autotune_lines(attn):
    return [t for t in attn._impl.selection_trace if t[1] == "autotune"]


@needs_cuda
def test_first_run_measures_every_accepting_candidate(tuner, monkeypatch):
    p, res, md, order = _setup(101)
    winner = order[-1]  # not the heuristic head: the measurement decides
    timed = _rank(tuner, monkeypatch, winner)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True):
        attn.plan(md, backend=res, **_plan_kw(p))
        # every candidate accepted and planned; the head is provisional
        accepted = [n for n, ph, r in attn._impl.selection_trace if r == "accepted"]
        assert accepted == list(order)
        assert attn.backend == "autotune"
        assert "autotune selects among" in attn.explain()
        out, lse = _run(attn, p)
    assert timed == set(order)
    assert attn.backend == "autotune" and _selected(attn) == winner
    assert f"chosen: autotune (selected {winner})" in attn.explain()
    assert _autotune_lines(attn)[-1] == (
        winner,
        "autotune",
        f"measured over {list(order)}",
    )
    assert "selects among" not in attn.explain()
    _check(p, out, lse)
    # the choice lasts for the plan: later runs do not consult the tuner
    monkeypatch.setattr(tuner, "search_cache", _forbidden)
    out2, lse2 = _run(attn, p)
    assert torch.equal(out, out2) and torch.equal(lse, lse2)


@needs_cuda
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_real_profiling_times_every_candidate(tuner, graph):
    """No patched timer: every candidate must be timed by the tuner's own
    cold-L2 profiler (a failing profile is scored as an infinitely slow
    tactic, which would silently leave the heuristic head)."""
    p, res, md, order = _setup(151)
    dev = torch.device(p["device"])
    attn = (
        PagedAttention(
            scratch_workspace(dev), graph_capacity=GraphCapacity.from_metadata(md)
        )
        if graph
        else PagedAttention(scratch_workspace(dev))
    )
    with flashinfer.autotune(True):
        attn.plan(md, backend=res, **_plan_kw(p))
        out, lse = _run(attn, p)
    assert not tuner.stats.failed_profiling_count.get(_auto_policy._OP)
    assert _autotune_lines(attn)[-1] == (
        _selected(attn),
        "autotune",
        f"measured over {list(order)}",
    )
    assert len(tuner.profiling_cache) == 1
    _check(p, out, lse)


@needs_cuda
def test_caller_buffers_are_written_once_by_the_winner(tuner, monkeypatch):
    p, res, md, order = _setup(103)
    _rank(tuner, monkeypatch, order[-1])
    rows, h, d = p["q"].shape[0], p["num_qo_heads"], p["head_dim_qk"]
    out = torch.full((rows, h, d), float("nan"), dtype=p["dtype"], device=p["q"].device)
    lse = torch.full((rows, h), float("nan"), device=p["q"].device)
    seen = []

    def spy(backend_cls):
        real = backend_cls.run

        def run(self, q, k, v, *, out=None, lse=None, **kw):
            seen.append((self.name, out is not None and out.data_ptr()))
            return real(self, q, k, v, out=out, lse=lse, **kw)

        monkeypatch.setattr(backend_cls, "run", run)

    from flashinfer.experimental.paged_attention._backends import _CLASSES

    for backend_cls in {_CLASSES[n] for n in order}:
        spy(backend_cls)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True):
        attn.plan(md, backend=res, **_plan_kw(p))
        attn.run(p["q"], (p["k_cache"], p["v_cache"]), out=out, lse=lse)
    caller = [name for name, ptr in seen if ptr == out.data_ptr()]
    assert caller == [order[-1]], "profiling wrote the caller's out"
    _check(p, out, lse)


@needs_cuda
@pytest.mark.parametrize("source", ["memory", "disk"])
def test_a_cached_result_selects_without_profiling(
    tuner, monkeypatch, tmp_path, source
):
    p, res, md, order = _setup(107)
    winner = order[-1]
    _rank(tuner, monkeypatch, winner)
    cache = str(tmp_path / "paged-attention-autotune.json")
    first = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True, cache=cache):
        first.plan(md, backend=res, **_plan_kw(p))
        _run(first, p)
    assert _selected(first) == winner

    if source == "disk":
        tuner = AutoTuner(warmup=1, repeat=2)
        monkeypatch.setattr(AutoTuner, "_instance", tuner)
        assert not tuner.profiling_cache
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    # the same workload on a fresh instance, by name, with other query values
    q = dict(p, q=p["q"] * 0.5)
    second = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(False, cache=cache if source == "disk" else None):
        second.plan(make_metadata(q), backend="autotune", **_plan_kw(q))
        # the plan still prepares every candidate; the first run() selects
        assert second._impl._tuning is not None and second.backend == "autotune"
        out, lse = _run(second, q)
    assert second.backend == "autotune" and _selected(second) == winner
    assert "selects among" not in second.explain()
    assert _autotune_lines(second)[-1] == (winner, "autotune", "cached measurement")
    _check(q, out, lse)


@needs_cuda
def test_a_miss_outside_the_context_raises(tuner, monkeypatch):
    p, res, md, order = _setup(113)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    attn.plan(md, backend=res, **_plan_kw(p))
    with pytest.raises(RuntimeError, match="No cached PagedAttention autotune result"):
        _run(attn, p)
    # the selection stays open: a warm-up inside the context settles it
    assert attn._impl._tuning is not None
    monkeypatch.undo()
    with flashinfer.autotune(True):
        _check(p, *_run(attn, p))
    assert attn._impl._tuning is None


@needs_cuda
def test_another_workload_is_not_a_cache_hit(tuner, monkeypatch):
    p, res, md, order = _setup(109)
    _rank(tuner, monkeypatch, order[-1])
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True):
        attn.plan(md, backend=res, **_plan_kw(p))
        _run(attn, p)
    other = make_problem(seed=211, **dict(_SHAPE, batch_size=2))
    attn.plan(make_metadata(other), backend=res, **_plan_kw(other))
    with pytest.raises(RuntimeError, match="No cached PagedAttention autotune result"):
        _run(attn, other)


@needs_cuda
def test_skipped_profiling_is_an_error(tuner):
    p, res, md, order = _setup(127)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True, skip_ops={_auto_policy._OP}):
        attn.plan(md, backend=res, **_plan_kw(p))
        with pytest.raises(RuntimeError, match="produced no measured selection"):
            _run(attn, p)
    assert not tuner.profiling_cache


@needs_cuda
def test_profiling_is_refused_under_capture(tuner, monkeypatch):
    p, res, md, order = _setup(131)
    attn = PagedAttention(scratch_workspace(torch.device(p["device"])))
    with flashinfer.autotune(True):
        attn.plan(md, backend=res, **_plan_kw(p))
        with monkeypatch.context() as capturing:
            capturing.setattr(
                _auto_policy.torch.cuda, "is_current_stream_capturing", lambda: True
            )
            with pytest.raises(
                RuntimeError, match="requires a prepared selection before CUDA graph"
            ):
                _run(attn, p)
        # the choice stays open; the next eager run settles it
        assert attn._impl._tuning is not None
        _check(p, *_run(attn, p))
    assert attn._impl._tuning is None


@needs_cuda
@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
@pytest.mark.parametrize("pick", ["head", "last"])
def test_graph_mode_freezes_the_measured_winner(tuner, monkeypatch, input_form, pick):
    # flat page ids: the candidates read different derived forms (a dense
    # table for trtllm-gen / cudnn, the CSR page indptr for fa2), all staged
    # once into the reserved storage
    p, res, md, order = _setup(137, input_form)
    winner = order[0] if pick == "head" else order[-1]
    _rank(tuner, monkeypatch, winner)
    dev = torch.device(p["device"])
    cap = GraphCapacity.from_metadata(md)
    g = PagedAttention(scratch_workspace(dev), graph_capacity=cap)
    q_buf = p["q"].clone()
    out_buf = torch.empty(
        q_buf.shape[0],
        p["num_qo_heads"],
        p["head_dim_qk"],
        dtype=p["dtype"],
        device=dev,
    )
    lse_buf = torch.empty(q_buf.shape[0], p["num_qo_heads"], device=dev)
    with flashinfer.autotune(True):
        g.plan(md, backend=res, **_plan_kw(p))
        with pytest.raises(RuntimeError, match="autotune selection is open"):
            g.plan(md, backend=res, **_plan_kw(p))
        # eager warm-up before capture settles and freezes the choice
        g.run(q_buf, (p["k_cache"], p["v_cache"]), out=out_buf, lse=lse_buf)
    assert g.backend == "autotune" and _selected(g) == winner
    assert g._impl._frozen["backend"] == winner
    _check(p, out_buf, lse_buf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.run(q_buf, (p["k_cache"], p["v_cache"]), out=out_buf, lse=lse_buf)
    # a re-plan keeps the frozen winner, no new measurement
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    out_buf.zero_()
    g.plan(md, backend=res, **_plan_kw(p))
    assert g._impl.selection_order == (winner,)
    assert g.backend == "autotune" and _selected(g) == winner
    graph.replay()
    torch.cuda.synchronize()
    _check(p, out_buf, lse_buf)


@needs_cuda
def test_graph_mode_cached_result_freezes_at_the_first_run(tuner, monkeypatch):
    p, res, md, order = _setup(149)
    winner = order[-1]
    _rank(tuner, monkeypatch, winner)
    dev = torch.device(p["device"])
    cap = GraphCapacity.from_metadata(md)
    with flashinfer.autotune(True):
        first = PagedAttention(scratch_workspace(dev), graph_capacity=cap)
        first.plan(md, backend=res, **_plan_kw(p))
        _run(first, p)
    monkeypatch.setattr(tuner, "choose_one", _forbidden)
    second = PagedAttention(
        scratch_workspace(dev), graph_capacity=dataclasses.replace(cap)
    )
    second.plan(md, backend=res, **_plan_kw(p))
    with pytest.raises(RuntimeError, match="autotune selection is open"):
        second.plan(md, backend=res, **_plan_kw(p))
    _check(p, *_run(second, p))
    assert _selected(second) == winner and second._impl._frozen["backend"] == winner
    second.plan(md, backend=res, **_plan_kw(p))
    _check(p, *_run(second, p))
