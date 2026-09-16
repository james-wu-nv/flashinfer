"""Selection contract of PagedAttention (experimental): device binding of a
``Resolution`` and the typed plan-time fallback within its candidate set.

Everything that needs no kernel runs without a GPU (explicit ``cc_major``
resolution, monkeypatched device properties); the controller-side checks
need one CUDA device.
"""

import types

import pytest
import torch

from flashinfer.experimental.paged_attention import Resolution
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
    attn = PagedAttention(torch.device(p["device"]), use_cuda_graph=True)
    with pytest.raises(ValueError, match="no pinned candidate"):
        attn.plan(make_metadata(p), backend=res, **_plan_kw(p))
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
    assert res.config[-4:-1] == (30.0, False, False)

    res = resolve_paged_attention(cc_major=9, custom_mask=True, **_CFG)
    assert res.excluded["fa3"] == "custom attention mask not supported"
    assert res.excluded["cudnn"] == "custom attention mask not supported"
    assert res.backends == ("fa2",)
    assert res.config[-4:-1] == (None, True, False)

    res = resolve_paged_attention(cc_major=10, sinks=True, **_CFG)
    assert res.excluded["cudnn"] == "attention sinks not supported"
    assert set(res.backends) == {"trtllm-gen", "cake", "fa2"}
    assert res.config[-4:-1] == (None, False, True)

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
    res = resolve_paged_attention(
        cc_major=10, **dict(_CFG, causal=False, window_left=16)
    )
    assert (
        res.excluded["trtllm-gen"]
        == "sliding window with non-causal attention not supported"
    )
    assert "cudnn" in res.excluded  # no window at all
    assert res.backends == ("fa2",)
