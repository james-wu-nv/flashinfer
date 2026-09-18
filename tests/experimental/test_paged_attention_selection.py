"""Selection contract of PagedAttention (experimental): device binding of a
``Resolution`` and the typed plan-time fallback within its candidate set.

Everything that needs no kernel runs without a GPU (explicit ``cc_major``
resolution, monkeypatched device properties); the controller-side checks
need one CUDA device.
"""

import types

import pytest
import torch

from flashinfer.experimental.paged_attention import (
    HEURISTIC_ORDER,
    GraphCapacity,
    Resolution,
    heuristic_order,
)
from flashinfer.experimental.paged_attention._backends import (
    cudnn_backend,
    fa_backend,
    trtllm_gen_backend,
)
from flashinfer.experimental.paged_attention._backends._capabilities import (
    _BackendPlanUnsupportedError,
)
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .test_paged_attention_prototype import (
    _resolve_or_skip,
    make_metadata,
    make_problem,
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
    batch_size=2,
    max_q=16,
    max_kv=64,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# --------------------------------------------------------------------------
# device binding (resolve_config_key ends with (cc_major, cc_minor, index))
# --------------------------------------------------------------------------


def test_explicit_cc_major_is_pinned_to_the_major_only():
    res = resolve_paged_attention(cc_major=9, **_CFG)
    assert res.device_binding == (9, None, None)
    assert "not pinned to a device" in res.explain()
    # fa3 is admitted from the capability table + toolkit level alone: no
    # device probe can run without a device (the old probe asked the CURRENT
    # device, which is the bug this pins)
    assert "fa3" in res.backends or "fa3 requires CUDA" in res.excluded.get("fa3", "")


def test_probes_answer_for_the_target_device(monkeypatch):
    """The fa3 probe must be asked about the device being resolved, not the
    current device; the binding records that device's full capability."""
    seen = {}

    def fake_props(dev):
        assert torch.device(dev) == torch.device("cuda", 1)
        return types.SimpleNamespace(major=9, minor=0)

    def fake_sm90a(dev):
        seen["device"] = dev
        return True

    import flashinfer.utils as fi_utils

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_props)
    monkeypatch.setattr(fi_utils, "is_sm90a_supported", fake_sm90a)
    res = resolve_paged_attention(device=torch.device("cuda", 1), **_CFG)
    assert seen["device"] == torch.device("cuda", 1)
    assert res.device_binding == (9, 0, 1)
    assert res.chosen == "fa3"
    assert "sm_90, cuda:1" in res.explain()


def test_cc_major_must_agree_with_the_device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda dev: types.SimpleNamespace(major=10, minor=0),
    )
    with pytest.raises(ValueError, match="contradicts device"):
        resolve_paged_attention(device=torch.device("cuda", 0), cc_major=9, **_CFG)


def test_non_cuda_device_is_rejected():
    with pytest.raises(ValueError, match="must be a CUDA device"):
        resolve_paged_attention(device=torch.device("cpu"), **_CFG)


@needs_cuda
def test_resolution_records_the_resolved_device():
    dev = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(dev)
    res = resolve_paged_attention(device=dev, **_CFG)
    assert res.device_binding == (props.major, props.minor, 0)
    # neither device nor cc_major: the current device, still recorded
    res2 = resolve_paged_attention(**_CFG)
    assert res2.device_binding == (
        props.major,
        props.minor,
        torch.cuda.current_device(),
    )


@needs_cuda
def test_plan_rejects_a_resolution_pinned_elsewhere():
    """A Resolution from another device index or another compute capability
    must be refused with a message naming both sides; a major-only pin from
    an explicit cc_major is accepted on any device of that major."""
    p = make_problem(seed=61, **_SHAPE)
    res = _resolve_or_skip(p, "auto")
    major, minor, index = res.device_binding
    md = make_metadata(p)
    attn = PagedAttention(torch.device(p["device"]))
    plan_kw = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
    )

    def rebind(binding):
        return Resolution(
            backends=res.backends,
            excluded=res.excluded,
            kv_layout=res.kv_layout,
            config=res.config[:-1] + (binding,),
        )

    with pytest.raises(ValueError, match=f"cuda:{index + 1}"):
        attn.plan(md, backend=rebind((major, minor, index + 1)), **plan_kw)
    with pytest.raises(ValueError, match="compute capability"):
        attn.plan(md, backend=rebind((major + 1, minor, index)), **plan_kw)
    with pytest.raises(ValueError, match="compute capability"):
        attn.plan(md, backend=rebind((major, minor + 1, index)), **plan_kw)
    # major-only pin (explicit cc_major) is accepted on this device
    attn.plan(md, backend=rebind((major, None, None)), **plan_kw)
    assert attn.backend in res.backends
    # and the exact binding of course
    attn.plan(md, backend=res, **plan_kw)
    # semantic drift is still reported as such (not as a device problem)
    with pytest.raises(ValueError, match="pinned Resolution"):
        attn.plan(md, backend=res, **dict(plan_kw, num_kv_heads=p["num_qo_heads"]))


# --------------------------------------------------------------------------
# typed plan-time fallback within the pinned candidate set
# --------------------------------------------------------------------------

_BACKEND_CLASS = {
    "fa2": fa_backend._FaBackend,
    "fa3": fa_backend._FaBackend,
    "cudnn": cudnn_backend._CudnnBackend,
    "trtllm-gen": trtllm_gen_backend._TrtllmGenBackend,
    "cake": trtllm_gen_backend._TrtllmGenBackend,
}


def _inject_preflight(monkeypatch, name, exc):
    def preflight(self, meta):
        if self.name == name:
            raise exc
        return real(self, meta)

    real = _BACKEND_CLASS[name].preflight
    monkeypatch.setattr(_BACKEND_CLASS[name], "preflight", preflight)


def _plan_kw(p):
    return dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
    )


@needs_cuda
def test_typed_unsupported_moves_to_the_next_candidate(monkeypatch):
    p = make_problem(seed=71, **_SHAPE)
    res = _resolve_or_skip(p, "auto")
    if len(res.backends) < 2:
        pytest.skip("needs two runnable candidates")
    first, second = res.backends[0], res.backends[1]
    _inject_preflight(
        monkeypatch,
        first,
        _BackendPlanUnsupportedError("injected: cannot do this batch"),
    )
    attn = PagedAttention(torch.device(p["device"]))
    attn.plan(make_metadata(p), backend=res, **_plan_kw(p))
    assert attn.backend == second
    trace = attn._impl.selection_trace
    assert trace[0] == (first, "preflight", "injected: cannot do this batch")
    assert trace[1] == (second, "preflight", "accepted")
    text = attn.explain()
    assert f"chosen: {second}" in text and "injected" in text
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()


@needs_cuda
def test_other_exceptions_from_a_candidate_are_not_swallowed(monkeypatch):
    p = make_problem(seed=73, **_SHAPE)
    res = _resolve_or_skip(p, "auto")
    _inject_preflight(monkeypatch, res.backends[0], ValueError("boom: not a fallback"))
    attn = PagedAttention(torch.device(p["device"]))
    with pytest.raises(ValueError, match="boom: not a fallback"):
        attn.plan(make_metadata(p), backend=res, **_plan_kw(p))
    assert attn.backend is None
    _inject_preflight(monkeypatch, res.backends[0], RuntimeError("plain runtime error"))
    with pytest.raises(RuntimeError, match="plain runtime error"):
        attn.plan(make_metadata(p), backend=res, **_plan_kw(p))


@needs_cuda
def test_explicit_backend_never_falls_back(monkeypatch):
    p = make_problem(seed=79, **_SHAPE)
    _resolve_or_skip(p, "fa2")
    _inject_preflight(monkeypatch, "fa2", _BackendPlanUnsupportedError("declined"))
    attn = PagedAttention(torch.device(p["device"]))
    with pytest.raises(ValueError, match="'fa2' cannot plan this batch: declined"):
        attn.plan(make_metadata(p), backend="fa2", **_plan_kw(p))
    # a Resolution whose every member declines: ValueError with the trace
    res = _resolve_or_skip(p, "fa2")
    with pytest.raises(ValueError, match="no pinned candidate.*fa2: declined"):
        attn.plan(make_metadata(p), backend=res, **_plan_kw(p))
    assert attn.backend is None


@needs_cuda
def test_preflight_runs_before_any_reserved_buffer_write(monkeypatch):
    """Graph mode: a batch every candidate declines must leave the reserved
    storage untouched (the walk happens before the staging transaction)."""
    p = make_problem(seed=83, **_SHAPE)
    res = _resolve_or_skip(p, "auto")
    for name in res.backends:
        _inject_preflight(monkeypatch, name, _BackendPlanUnsupportedError("declined"))
    # An explicit capacity allocates the reserved storage at construction;
    # with the inferred form a failed first plan publishes no capacity at all
    # (the graph-lifecycle rule), which is checked in the cuda_graph suite.
    md = make_metadata(p)
    attn = PagedAttention(
        torch.device(p["device"]), graph_capacity=GraphCapacity.from_metadata(md)
    )
    with pytest.raises(ValueError, match="no pinned candidate"):
        attn.plan(md, backend=res, **_plan_kw(p))
    gb = attn._impl._graph
    assert gb is not None
    for buf in (gb.qo_indptr, gb.kv_seq_lens, gb.block_tables, gb.kv_page_indices):
        assert not buf.any(), "reserved storage was written before preflight"


# --------------------------------------------------------------------------
# feature axes: soft cap / custom mask / sinks are selection facts
# --------------------------------------------------------------------------


def test_features_exclude_backends_with_reasons():
    """`auto` must reject backends that cannot apply a requested feature
    (and say so) instead of dropping the feature silently."""
    res = resolve_paged_attention(cc_major=10, logits_soft_cap=30.0, **_CFG)
    assert res.excluded["cudnn"] == "logits soft cap not supported"
    assert res.excluded["trtllm-gen"] == "logits soft cap not supported"
    assert "fa2" in res.backends
    assert res.config[-5:-2] == (30.0, False, False)

    res = resolve_paged_attention(cc_major=9, custom_mask=True, **_CFG)
    assert res.excluded["fa3"] == "custom attention mask not supported"
    assert res.excluded["cudnn"] == "custom attention mask not supported"
    assert res.backends == ("fa2",)
    assert res.config[-5:-2] == (None, True, False)

    res = resolve_paged_attention(cc_major=10, sinks=True, **_CFG)
    assert res.excluded["cudnn"] == "attention sinks not supported"
    assert set(res.backends) == {"trtllm-gen", "cake", "fa2"}
    assert res.config[-5:-2] == (None, False, True)

    # a configuration only cuDNN could run + a feature cuDNN lacks: loud
    with pytest.raises(ValueError, match="cudnn: logits soft cap not supported"):
        resolve_paged_attention(
            cc_major=10,
            logits_soft_cap=50.0,
            **dict(_CFG, head_dim_qk=192, head_dim_vo=128),
        )


def test_logits_soft_cap_is_normalized_and_validated():
    off = resolve_paged_attention(cc_major=10, logits_soft_cap=0.0, **_CFG)
    plain = resolve_paged_attention(cc_major=10, **_CFG)
    assert off.config == plain.config  # 0 means off, like the legacy wrappers
    assert "cudnn" in off.backends
    for bad in (-1.0, float("nan"), float("inf"), "30"):
        with pytest.raises(ValueError, match="logits_soft_cap"):
            resolve_paged_attention(cc_major=10, logits_soft_cap=bad, **_CFG)


def test_trtllm_gen_noncausal_is_declared_but_not_with_a_window():
    """supports_noncausal=True is measured (see the capability comment); the
    one non-causal configuration the kernel lacks - a sliding window - is
    excluded at resolve rather than failing at plan."""
    res = resolve_paged_attention(cc_major=10, **dict(_CFG, causal=False))
    assert "trtllm-gen" in res.backends
    # no backend is left: trtllm-gen / cake have no kernel, cuDNN no window,
    # and the fa2 kernel computes it wrong (M20) so it is declared unsupported
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        resolve_paged_attention(cc_major=10, **dict(_CFG, causal=False, window_left=16))
    detail = str(ei.value)
    assert "trtllm-gen: sliding window with non-causal attention not supported" in detail
    assert "fa2: sliding window with non-causal attention not supported" in detail
    assert "cudnn" in detail  # no window at all


# --------------------------------------------------------------------------
# the max_q_len hint: order bucket on this device, pinned, checked by plan()
# --------------------------------------------------------------------------


def _hint_bound_or_skip(cc_major):
    buckets = HEURISTIC_ORDER.get(cc_major, ())
    if len(buckets) < 2:
        pytest.skip(f"sm_{cc_major}x has a single preference order (no q_len bucket)")
    return buckets[0][0]


def _runnable(res, order):
    return tuple(n for n in order if n in res.backends)


@needs_cuda
def test_max_q_len_hint_changes_the_order_on_this_device():
    """Below the bucket bound the hinted Resolution ranks fa2 first; above it,
    and without a hint, the order is today's.  The hint is pinned in the key
    and shown by explain(); the candidate set and exclusions do not move."""
    dev = torch.device("cuda", 0)
    cc_major = torch.cuda.get_device_properties(dev).major
    bound = _hint_bound_or_skip(cc_major)
    plain = resolve_paged_attention(device=dev, **_CFG)
    small = resolve_paged_attention(device=dev, max_q_len=1, **_CFG)
    edge = resolve_paged_attention(device=dev, max_q_len=bound, **_CFG)
    above = resolve_paged_attention(device=dev, max_q_len=bound + 1, **_CFG)
    if len(plain.backends) < 2:
        pytest.skip("needs two runnable candidates to observe an order")

    assert plain.backends == _runnable(plain, heuristic_order(cc_major))
    assert small.backends == _runnable(small, heuristic_order(cc_major, 1))
    assert edge.backends == small.backends
    assert above.backends == plain.backends
    assert set(small.backends) == set(plain.backends)
    assert small.excluded == plain.excluded
    if "fa2" in plain.backends:
        assert small.chosen == "fa2"
        assert small.backends != plain.backends
    assert small.max_q_len == 1 and plain.max_q_len is None
    assert small.config != plain.config and small.config[:-2] == plain.config[:-2]
    assert small.device_binding == plain.device_binding
    assert (
        "max_q_len hint: 1 (order for batches with at most 1 query" in small.explain()
    )
    assert "max_q_len hint: none (default order)" in plain.explain()


@needs_cuda
def test_no_hint_keeps_the_default_order_and_plan_honours_the_hint():
    p = make_problem(seed=89, uniform_q1=True, **dict(_SHAPE, max_q=1))
    res_plain = _resolve_or_skip(p, "auto")
    dev = torch.device(p["device"])
    cc_major = torch.cuda.get_device_properties(dev).major
    assert res_plain.backends == _runnable(res_plain, heuristic_order(cc_major))
    _hint_bound_or_skip(cc_major)
    res_hint = resolve_paged_attention(
        device=dev,
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        page_size=p["page_size"],
        causal=True,
        need_lse=True,
        max_q_len=1,
    )
    attn = PagedAttention(dev)
    attn.plan(make_metadata(p), backend=res_hint, **_plan_kw(p))
    # plan() walks the hinted order: the first candidate that accepts the batch
    assert attn.backend == res_hint.backends[0]
    assert "max_q_len hint: 1" in attn.explain()
    # the same batch under the plain Resolution follows today's order
    attn.plan(make_metadata(p), backend=res_plain, **_plan_kw(p))
    assert attn.backend == res_plain.backends[0]
    assert "max_q_len hint: none" in attn.explain()
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()


@needs_cuda
def test_plan_rejects_a_batch_above_the_hint():
    """The hint is a promise about the batches: a batch (eager) or a graph
    capacity whose max_q_len exceeds it is refused before any state moves;
    a batch within the hint plans normally."""
    p = make_problem(seed=97, **_SHAPE)  # max_q 16
    _resolve_or_skip(p, "auto")
    dev = torch.device(p["device"])
    _hint_bound_or_skip(torch.cuda.get_device_properties(dev).major)

    def resolve(hint):
        return resolve_paged_attention(
            device=dev,
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            q_dtype=p["dtype"],
            page_size=p["page_size"],
            causal=True,
            need_lse=True,
            max_q_len=hint,
        )

    md = make_metadata(p)
    attn = PagedAttention(dev)
    with pytest.raises(ValueError, match=r"max_q_len \d+ of this batch exceeds"):
        attn.plan(md, backend=resolve(1), **_plan_kw(p))
    assert attn.backend is None
    attn.plan(md, backend=resolve(p["max_q_len"]), **_plan_kw(p))
    assert attn.backend is not None
    # graph mode: the CAPACITY is what the kernels are planned with, so it is
    # what the hint is checked against, and nothing is reserved on rejection
    cap = GraphCapacity.from_metadata(md)
    g = PagedAttention(dev, graph_capacity=cap)
    with pytest.raises(ValueError, match="of this graph capacity exceeds"):
        g.plan(md, backend=resolve(1), **_plan_kw(p))
    assert g.backend is None
    gb = g._impl._graph
    for buf in (gb.qo_indptr, gb.kv_seq_lens, gb.block_tables, gb.kv_page_indices):
        assert buf is None or not buf.any(), (
            "reserved storage written before the hint check"
        )
    g.plan(md, backend=resolve(cap.max_q_len), **_plan_kw(p))
    assert g.backend is not None


def test_max_q_len_hint_is_keyed_and_validated_without_a_gpu():
    """CPU variant (explicit cc_major): the hint slot sits right before the
    device binding, distinct hints give distinct keys, and the value is
    validated like the other host scalars."""
    plain = resolve_paged_attention(cc_major=10, **_CFG)
    one = resolve_paged_attention(cc_major=10, max_q_len=1, **_CFG)
    big = resolve_paged_attention(cc_major=10, max_q_len=4096, **_CFG)
    assert plain.config[-2] is None and one.config[-2] == 1 and big.config[-2] == 4096
    assert len({plain.config, one.config, big.config}) == 3
    assert one.backends[0] == "fa2" and plain.chosen == big.chosen == "trtllm-gen"
    assert set(one.backends) == set(plain.backends) == set(big.backends)
    for bad in (0, -3, 2.0, False, "8"):
        with pytest.raises(ValueError, match="max_q_len must be None"):
            resolve_paged_attention(cc_major=10, max_q_len=bad, **_CFG)
