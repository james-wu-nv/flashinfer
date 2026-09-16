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
