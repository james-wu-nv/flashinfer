"""Host-only contract tests for the unified paged attention package.

Everything here runs with ``CUDA_VISIBLE_DEVICES=""``: the environment probes
are monkeypatched away, ``Transaction`` / ``GraphBuffers`` are exercised on
hand-built CPU buffers, and metadata validation goes through the host-mirror
validators.  This is the "host/static" CI layer of the test-coverage report
(§5.2): the contract is checked before any kernel exists, so a capability or
key regression fails on a CPU-only runner.

Known gaps recorded as strict xfails (they flip to XPASS when fixed):

- ledger M2 (WP-A): ``Transaction.__enter__`` does not roll back the copies
  made before a failing copy in the staging loop.
- WP-E (mla-alignment F6): ``Resolution.config`` does not carry the device /
  compute-capability identity it was resolved for.
"""

import dataclasses
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from flashinfer.experimental import paged_attention as pa
from flashinfer.experimental.paged_attention import (
    CAPABILITIES,
    HEURISTIC_ORDER,
    MIN_DENSE_PAGE_SIZE,
    GraphBuffers,
    GraphCapacity,
    PagedAttentionMetadata,
    _selection,
    heuristic_order,
    resolve_paged_attention,
)
from flashinfer.experimental.paged_attention._contracts import (
    LSE_MODES,
    _expect_lse_mode,
    _expect_page_size,
    _expect_window_left,
    resolve_config_key,
)
from flashinfer.experimental.paged_attention._controller import (
    PagedAttentionController,
)
from flashinfer.experimental.paged_attention._graph import Transaction
from flashinfer.experimental.paged_attention._planning import (
    Derived,
    FORM_BLOCK_TABLES,
    FORM_CUM_KV_SEQ_LENS,
    FORM_KV_PAGE_INDICES,
    FORM_KV_PAGE_INDPTR,
    FORM_Q_SEQ_LENS,
    validate_causal_envelope,
    validate_values,
)
from flashinfer.prefill import PagedAttention


@pytest.fixture
def no_probes(monkeypatch):
    """Selection without touching CUDA: every environment probe reports 'available'."""
    monkeypatch.setattr(
        _selection, "PROBES", {name: (lambda device: None) for name in CAPABILITIES}
    )


_BASE = dict(
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    q_dtype=torch.bfloat16,
    page_size=16,
    need_lse=True,
)


def _resolve(cc_major, **kw):
    return resolve_paged_attention(cc_major=cc_major, **{**_BASE, **kw})


# ---------------------------------------------------------------------------
# resolve_config_key: the drift detector between resolve() and plan()
# ---------------------------------------------------------------------------

KEY_KW = dict(
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    head_dim_vo=128,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    page_size=16,
    kv_layout="HND",
    causal=True,
    need_lse=True,
    window_left=-1,
    kv_input_form="block_tables",
    logits_soft_cap=None,
    custom_mask=False,
    sinks=False,
    max_q_len=None,
    cc_major=9,
    cc_minor=None,
    device_index=None,
)
KEY_DRIFT = dict(
    num_qo_heads=16,
    num_kv_heads=1,
    head_dim_qk=64,
    head_dim_vo=64,
    q_dtype=torch.float16,
    kv_dtype=torch.float8_e4m3fn,
    page_size=32,
    kv_layout="NHD",
    causal=False,
    need_lse=False,
    window_left=0,
    kv_input_form="page_indices",
    logits_soft_cap=30.0,
    custom_mask=True,
    sinks=True,
    max_q_len=4,
    cc_major=10,
    cc_minor=0,
    device_index=1,
)


def test_resolve_config_key_layout_is_pinned():
    """plan() compares this tuple against Resolution.config.  A layout change
    must be deliberate: update this pin together with the drift test below."""
    assert resolve_config_key(**KEY_KW) == (
        8,
        2,
        128,
        128,
        "torch.bfloat16",
        "torch.bfloat16",
        16,
        "HND",
        True,
        True,
        -1,
        "block_tables",
        None,  # logits_soft_cap
        False,  # custom mask
        False,  # sinks
        None,  # max_q_len hint (None = no hint: default order)
        (9, None, None),  # device binding: (cc_major, cc_minor, device_index)
    )


@pytest.mark.parametrize("field", sorted(KEY_KW))
def test_resolve_config_key_detects_single_field_drift(field):
    base = resolve_config_key(**KEY_KW)
    drifted = resolve_config_key(**{**KEY_KW, field: KEY_DRIFT[field]})
    assert drifted != base, field
    assert resolve_config_key(**KEY_KW) == base  # deterministic


def test_every_semantic_plan_kwarg_is_part_of_the_key():
    """A plan() kwarg (soft-cap, sinks, ...) that is not in the key would let
    a pinned Resolution silently accept a different configuration."""
    # plan() speaks lse_mode / use_sinks; the key stores need_lse / sinks
    renamed = {"lse_mode": "need_lse", "use_sinks": "sinks"}
    key_params = set(inspect.signature(resolve_config_key).parameters)
    for plan in (PagedAttentionController.plan, PagedAttention.plan):
        plan_params = set(inspect.signature(plan).parameters) - {
            "self",
            "metadata",
            "backend",
        }
        missing = {renamed.get(p, p) for p in plan_params} - key_params
        assert not missing, (
            f"{plan.__qualname__} kwargs missing from resolve_config_key: "
            f"{sorted(missing)}"
        )
    # the two metadata-derived facts the key must carry
    assert {"page_size", "kv_input_form"} <= key_params
    # resolve() accepts exactly the key fields plus its own routing arguments;
    # the device binding's minor/index are derived from ``device``, not passed
    resolve_params = set(inspect.signature(resolve_paged_attention).parameters)
    assert resolve_params - key_params <= {"device", "cc_major", "backend"}
    assert key_params - resolve_params <= {"cc_minor", "device_index"}


def test_resolution_is_frozen_and_config_matches_key(no_probes):
    res = _resolve(9)
    assert res.config == resolve_config_key(
        8,
        2,
        128,
        128,
        torch.bfloat16,
        torch.bfloat16,
        16,
        "HND",
        True,
        True,
        -1,
        "block_tables",
        None,
        False,
        False,
        None,  # max_q_len hint
        9,
        None,
        None,
    )
    assert res.max_q_len is None
    assert res.chosen == res.backends[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.backends = ()
    text = res.explain()
    assert "candidates" in text
    assert "max_q_len hint: none" in text
    assert all(name in text for name in res.excluded)


def test_resolution_config_carries_device_identity(no_probes):
    assert _resolve(9).config != _resolve(10).config


# ---------------------------------------------------------------------------
# PagedAttentionCapabilities.rejection_reason matrix
# ---------------------------------------------------------------------------


def _reason(name, **over):
    cap = CAPABILITIES[name]
    kw = dict(
        cc_major=min(cap.cc_majors),
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=16,
        kv_layout="HND",
        causal=True,
        need_lse=True,
        window_left=-1,
        kv_input_form="block_tables",
    )
    kw.update(over)
    return cap.rejection_reason(**kw)


def _capability_rows():
    rows = []
    for name, cap in CAPABILITIES.items():
        for cc in sorted(cap.cc_majors):
            rows.append((name, f"cc{cc}", dict(cc_major=cc), None))
        rows += [
            (name, "cc7", dict(cc_major=7), "compute capability"),
            # rejection order is pinned: compute capability is checked first
            (
                name,
                "cc7-before-dtype",
                dict(cc_major=7, q_dtype=torch.float32, kv_dtype=torch.float32),
                "compute capability",
            ),
            (
                name,
                "q-fp32",
                dict(q_dtype=torch.float32, kv_dtype=torch.float32),
                "unsupported q dtype",
            ),
            (name, "kv-f16-q-bf16", dict(kv_dtype=torch.float16), "dtype pair"),
            (name, "d32", dict(head_dim_qk=32, head_dim_vo=32), "head dims"),
            (name, "nhd", dict(kv_layout="NHD"), None),
            (name, "no-lse", dict(need_lse=False), None),
        ]
    # e5m2 KV: the generated fa2 kernels dequantize both fp8 formats
    # (measured on B200, see _capabilities.py); nobody else declares fp8 KV
    rows += [
        (name, "kv-e5m2", dict(kv_dtype=torch.float8_e5m2), "unsupported kv dtype")
        for name in CAPABILITIES
        if name != "fa2"
    ]
    rows += [
        ("fa2", "fp8-kv", dict(kv_dtype=torch.float8_e4m3fn), None),
        ("fa2", "kv-e5m2", dict(kv_dtype=torch.float8_e5m2), None),
        ("fa2", "d64", dict(head_dim_qk=64, head_dim_vo=64), None),
        ("fa2", "d256", dict(head_dim_qk=256, head_dim_vo=256), None),
        # 512: the Ampere+ large-head path, fa2 only (measured on B200)
        ("fa2", "d512", dict(head_dim_qk=512, head_dim_vo=512), None),
        # M21: the fa2 sink variant is wrong at head_dim 512 (measured); the
        # pair is excluded while plain D512 and sinks at D128 stay admitted
        (
            "fa2",
            "sinks-d512",
            dict(head_dim_qk=512, head_dim_vo=512, use_sinks=True),
            "attention sinks not supported at head dims (512, 512)",
        ),
        ("fa2", "sinks-d128", dict(use_sinks=True), None),
        ("fa3", "d512", dict(head_dim_qk=512, head_dim_vo=512), "head dims"),
        ("cudnn", "d512", dict(head_dim_qk=512, head_dim_vo=512), "head dims"),
        ("trtllm-gen", "d512", dict(head_dim_qk=512, head_dim_vo=512), "head dims"),
        ("cake", "d512", dict(head_dim_qk=512, head_dim_vo=512), "head dims"),
        ("fa2", "d512-256", dict(head_dim_qk=512, head_dim_vo=256), "head dims"),
        ("fa2", "d192-128", dict(head_dim_qk=192, head_dim_vo=128), "head dims"),
        ("fa2", "csr-page1", dict(kv_input_form="page_indices", page_size=1), None),
        ("fa2", "csr-page5", dict(kv_input_form="page_indices", page_size=5), None),
        ("fa2", "page1024", dict(page_size=1024), None),
        ("fa2", "noncausal", dict(causal=False), None),
        ("fa2", "window0", dict(window_left=0), None),
        ("fa3", "cc10", dict(cc_major=10), "compute capability"),
        ("fa3", "fp8-kv", dict(kv_dtype=torch.float8_e4m3fn), "unsupported kv dtype"),
        ("fa3", "d192-128", dict(head_dim_qk=192, head_dim_vo=128), "head dims"),
        ("fa3", "window0", dict(window_left=0), None),
        ("fa3", "noncausal", dict(causal=False), None),
        ("cudnn", "d192-128", dict(head_dim_qk=192, head_dim_vo=128), None),
        ("cudnn", "d64", dict(head_dim_qk=64, head_dim_vo=64), "head dims"),
        ("cudnn", "window0", dict(window_left=0), "sliding window"),
        ("cudnn", "noncausal", dict(causal=False), None),
        (
            "cudnn",
            "csr-page1",
            dict(kv_input_form="page_indices", page_size=1),
            "dense block table",
        ),
        (
            "cudnn",
            "csr-page4",
            dict(kv_input_form="page_indices", page_size=4),
            "dense block table",
        ),
        ("cudnn", "csr-page8", dict(kv_input_form="page_indices", page_size=8), None),
        ("cudnn", "fp8-kv", dict(kv_dtype=torch.float8_e4m3fn), "unsupported kv dtype"),
        ("trtllm-gen", "cc9", dict(cc_major=9), "compute capability"),
        ("trtllm-gen", "page8", dict(page_size=8), "unsupported page_size"),
        ("trtllm-gen", "page32", dict(page_size=32), None),
        ("trtllm-gen", "page64", dict(page_size=64), None),
        # 128 .. 1024: the shipped context kernels, measured on B200 for
        # trtllm-gen and for cake; nothing above 1024 ships
        ("trtllm-gen", "page128", dict(page_size=128), None),
        ("trtllm-gen", "page1024", dict(page_size=1024), None),
        ("trtllm-gen", "page2048", dict(page_size=2048), "unsupported page_size"),
        ("trtllm-gen", "page96", dict(page_size=96), "unsupported page_size"),
        ("cake", "page8", dict(page_size=8), "unsupported page_size"),
        ("cake", "page128", dict(page_size=128), None),
        ("cake", "page1024", dict(page_size=1024), None),
        ("cake", "page2048", dict(page_size=2048), "unsupported page_size"),
        ("trtllm-gen", "noncausal", dict(causal=False), None),  # measured on B200
        ("trtllm-gen", "window0", dict(window_left=0), None),
        (
            "trtllm-gen",
            "noncausal-window",
            dict(causal=False, window_left=16),
            "non-causal",
        ),
        ("trtllm-gen", "d64", dict(head_dim_qk=64, head_dim_vo=64), "head dims"),
        (
            "trtllm-gen",
            "fp8-kv",
            dict(kv_dtype=torch.float8_e4m3fn),
            "unsupported kv dtype",
        ),
        (
            "trtllm-gen",
            "csr-page16",
            dict(kv_input_form="page_indices", page_size=16),
            None,
        ),
        # feature axes: a backend lacking a requested feature is excluded with
        # a reason, never silently bypassed
        ("fa2", "softcap", dict(logits_soft_cap=30.0), None),
        ("fa3", "softcap", dict(logits_soft_cap=30.0), None),
        ("cudnn", "softcap", dict(logits_soft_cap=30.0), "logits soft cap"),
        ("trtllm-gen", "softcap", dict(logits_soft_cap=30.0), "logits soft cap"),
        ("fa2", "custom-mask", dict(use_custom_mask=True), None),
        ("fa3", "custom-mask", dict(use_custom_mask=True), "custom attention mask"),
        ("cudnn", "custom-mask", dict(use_custom_mask=True), "custom attention mask"),
        ("fa2", "sinks", dict(use_sinks=True), None),
        ("trtllm-gen", "sinks", dict(use_sinks=True), None),
        ("cake", "sinks", dict(use_sinks=True), None),
        ("cudnn", "sinks", dict(use_sinks=True), "attention sinks"),
    ]
    return rows


_CAP_ROWS = _capability_rows()


@pytest.mark.parametrize(
    "backend,label,over,expected",
    _CAP_ROWS,
    ids=[f"{r[0]}-{r[1]}" for r in _CAP_ROWS],
)
def test_capability_rejection_reason_matrix(backend, label, over, expected):
    reason = _reason(backend, **over)
    if expected is None:
        assert reason is None, f"{backend}/{label} unexpectedly excluded: {reason}"
    else:
        assert reason is not None, f"{backend}/{label} unexpectedly admitted"
        assert expected in reason, f"{backend}/{label}: {reason!r}"


# ---------------------------------------------------------------------------
# resolve_paged_attention(): ordering, explain, contract rejections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cc_major", sorted(HEURISTIC_ORDER))
def test_auto_orders_by_heuristic_and_explains_every_backend(no_probes, cc_major):
    res = _resolve(cc_major)
    # no hint: the default bucket of the table (today's order)
    expected = tuple(
        n for n in heuristic_order(cc_major) if _reason(n, cc_major=cc_major) is None
    )
    assert res.backends == expected
    assert set(res.backends) | set(res.excluded) == set(CAPABILITIES)
    assert not (set(res.backends) & set(res.excluded))
    for name, why in res.excluded.items():
        assert why == _reason(name, cc_major=cc_major)
    assert res.kv_layout == "HND"


def test_explicit_pin_evaluates_only_that_backend(no_probes):
    res = _resolve(10, backend="fa2")
    assert res.backends == ("fa2",)
    assert res.excluded == {}
    with pytest.raises(ValueError, match="unknown backend"):
        _resolve(10, backend="fa9")
    with pytest.raises(ValueError, match="compute capability"):
        _resolve(8, backend="trtllm-gen")


def test_no_runnable_backend_lists_every_reason(no_probes):
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        _resolve(10, head_dim_qk=32)
    assert all(name in str(ei.value) for name in CAPABILITIES)


def test_probe_failure_becomes_an_exclusion_reason(monkeypatch):
    probes = {name: (lambda device: None) for name in CAPABILITIES}
    probes["cudnn"] = lambda device: "cudnn-frontend python package not importable"
    monkeypatch.setattr(_selection, "PROBES", probes)
    res = _resolve(10)
    assert "cudnn" not in res.backends
    assert res.excluded["cudnn"] == "cudnn-frontend python package not importable"
    assert res.backends == ("trtllm-gen", "cake", "fa2")  # cudnn probed out


def test_probes_run_only_for_capability_admitted_backends(monkeypatch):
    calls = []

    def probe_for(name):
        def probe(device):
            calls.append(name)
            return None

        return probe

    monkeypatch.setattr(_selection, "PROBES", {n: probe_for(n) for n in CAPABILITIES})
    _resolve(8)  # fa3 / trtllm-gen are excluded statically on sm_8x
    assert set(calls) == {"fa2", "cudnn"}


@pytest.mark.parametrize(
    "kw,match",
    [
        (dict(num_qo_heads=7), "divisible"),
        (dict(num_qo_heads=0), "positive"),
        (dict(num_kv_heads=0), "positive"),
        (dict(kv_input_form="csr"), "kv_input_form"),
        (dict(window_left=-2), "window_left"),
        (dict(page_size=0), "page_size"),
        (dict(page_size=4), f"< {MIN_DENSE_PAGE_SIZE}"),
    ],
)
def test_resolve_rejects_contract_violations_before_capabilities(no_probes, kw, match):
    with pytest.raises(ValueError, match=match):
        _resolve(10, **kw)


def test_resolve_accepts_token_csr_below_the_dense_floor(no_probes):
    res = _resolve(10, page_size=4, kv_input_form="page_indices")
    assert res.backends == ("fa2",)
    assert "dense block table" in res.excluded["cudnn"]
    assert "unsupported page_size" in res.excluded["trtllm-gen"]


def test_heuristic_order_covers_declared_cc_majors():
    declared = set().union(*(cap.cc_majors for cap in CAPABILITIES.values()))
    assert declared <= set(HEURISTIC_ORDER), (
        f"cc majors without a preference order: {sorted(declared - set(HEURISTIC_ORDER))}"
    )
    for cc, buckets in HEURISTIC_ORDER.items():
        declared_here = {n for n, cap in CAPABILITIES.items() if cc in cap.cc_majors}
        bounds = [bound for bound, _ in buckets]
        # ascending max_q_len bounds, the default (None) last and only last
        assert bounds[-1] is None and None not in bounds[:-1], (cc, bounds)
        assert all(isinstance(b, int) and b >= 1 for b in bounds[:-1]), (cc, bounds)
        assert bounds[:-1] == sorted(bounds[:-1]) and len(set(bounds)) == len(bounds), (
            cc,
            bounds,
        )
        for bound, order in buckets:
            assert len(set(order)) == len(order), (cc, bound, order)
            # every bucket ranks the SAME backends: the hint changes the
            # order, never the candidate set
            assert set(order) == declared_here, (
                f"sm_{cc}x order {order} (max_q_len <= {bound}) != backends "
                f"declaring sm_{cc}x {sorted(declared_here)}"
            )


# ---------------------------------------------------------------------------
# the max_q_len hint: bucketed order, pinned in the key, shown by explain()
# ---------------------------------------------------------------------------


def _bucketed_ccs():
    return sorted(cc for cc, buckets in HEURISTIC_ORDER.items() if len(buckets) > 1)


def test_heuristic_order_lookup_walks_the_buckets():
    for cc, buckets in HEURISTIC_ORDER.items():
        default = buckets[-1][1]
        assert heuristic_order(cc) == default
        assert heuristic_order(cc, None) == default
        for i, (bound, order) in enumerate(buckets[:-1]):
            prev = buckets[i - 1][0] if i else 0
            assert heuristic_order(cc, prev + 1) == order, (cc, prev + 1)
            assert heuristic_order(cc, bound) == order, (cc, bound)
            assert heuristic_order(cc, bound + 1) != order or (
                buckets[i + 1][1] == order
            ), (cc, bound + 1)
        top = buckets[-2][0] if len(buckets) > 1 else 0
        assert heuristic_order(cc, top + 1) == default
        assert heuristic_order(cc, 1 << 20) == default
    assert heuristic_order(7) == ()
    assert heuristic_order(7, 1) == ()


@pytest.mark.parametrize("cc_major", _bucketed_ccs())
def test_max_q_len_hint_selects_the_bucket_order(no_probes, cc_major):
    """sm_100 (B200) measured: fa2 beats the trtllm-gen / cake context
    kernels at decode and speculative query lengths (5x at q=1), so a hint
    at or below the bucket bound puts fa2 first; above it, and with no hint,
    the order is today's."""
    bound = HEURISTIC_ORDER[cc_major][0][0]
    plain = _resolve(cc_major)
    small = _resolve(cc_major, max_q_len=1)
    edge = _resolve(cc_major, max_q_len=bound)
    above = _resolve(cc_major, max_q_len=bound + 1)

    def runnable(order):
        return tuple(n for n in order if _reason(n, cc_major=cc_major) is None)

    assert small.backends == runnable(heuristic_order(cc_major, 1))
    assert edge.backends == small.backends
    assert above.backends == plain.backends == runnable(heuristic_order(cc_major))
    assert small.backends != plain.backends
    # the hint changes the order only: same candidate set, same exclusions
    assert set(small.backends) == set(plain.backends)
    assert small.excluded == plain.excluded
    # pinned: distinct keys per hint, everything but the hint slot identical
    assert small.max_q_len == 1 and edge.max_q_len == bound and plain.max_q_len is None
    assert small.config != plain.config and small.config != edge.config
    assert small.config[:-2] == plain.config[:-2]
    assert small.config[-1] == plain.config[-1]
    assert small.config[-2] == 1
    # explain() shows the hint
    assert (
        "max_q_len hint: 1 (order for batches with at most 1 query" in small.explain()
    )
    assert "max_q_len hint: none (default order)" in plain.explain()


def test_sm100_small_q_bucket_puts_fa2_first(no_probes):
    """The measured decode regret (PLAN §4 last row; WP-G / WP-K): B=32, q=1,
    kv=4096 on B200 ran 455 us on trtllm-gen against 91 us on fa2."""
    assert _resolve(10, max_q_len=1).chosen == "fa2"
    assert _resolve(10).chosen == "trtllm-gen"
    assert _resolve(10, max_q_len=1 << 12).chosen == "trtllm-gen"


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "1"])
def test_max_q_len_hint_is_validated(no_probes, bad):
    with pytest.raises(ValueError, match="max_q_len must be None"):
        _resolve(10, max_q_len=bad)


def test_max_q_len_hint_with_an_explicit_backend_is_pinned_only(no_probes):
    res = _resolve(10, backend="cudnn", max_q_len=2)
    assert res.backends == ("cudnn",)
    assert res.max_q_len == 2 and res.config[-2] == 2


# ---------------------------------------------------------------------------
# the small loud-error helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", LSE_MODES)
def test_lse_modes_accepted(mode):
    _expect_lse_mode(mode)


@pytest.mark.parametrize("mode", ["base10", "BASE2", "", None, 2])
def test_lse_modes_rejected(mode):
    with pytest.raises(ValueError, match="lse_mode"):
        _expect_lse_mode(mode)


@pytest.mark.parametrize("window_left", [-1, 0, 1, 4096])
def test_window_left_accepted(window_left):
    _expect_window_left(window_left)


@pytest.mark.parametrize("window_left", [-2, -100, 1.5, "3", None])
def test_window_left_rejected(window_left):
    with pytest.raises(ValueError, match="window_left"):
        _expect_window_left(window_left)


@pytest.mark.parametrize(
    "page_size,form,ok",
    [
        (1, "page_indices", True),
        (5, "page_indices", True),
        (MIN_DENSE_PAGE_SIZE, "block_tables", True),
        (MIN_DENSE_PAGE_SIZE - 1, "block_tables", False),
        (1, "block_tables", False),
        (0, "page_indices", False),
        (-1, "block_tables", False),
        (1.5, "page_indices", False),
    ],
)
def test_page_size_floor(page_size, form, ok):
    if ok:
        _expect_page_size(page_size, form)
    else:
        with pytest.raises(ValueError, match="page_size"):
            _expect_page_size(page_size, form)


# ---------------------------------------------------------------------------
# validate_values: the host-mirror checks behind PagedAttentionMetadata
# ---------------------------------------------------------------------------


def _host_case(**over):
    case = dict(
        qo=[0, 4, 6, 9],
        kv=[10, 6, 9],
        page_size=4,
        width=3,  # dense capacity 3 x 4 = 12 tokens
        max_q_len=4,
        max_kv_len=10,
        csr=None,  # or the length of a flat page-id list
        causal=False,
    )
    case.update(over)
    return case


def _validate(case):
    qo = torch.tensor(case["qo"], dtype=torch.int32)
    kv = torch.tensor(case["kv"], dtype=torch.int32)
    bt = idx = None
    if case["csr"] is None:
        bt = torch.zeros(len(case["kv"]), case["width"], dtype=torch.int32)
    else:
        idx = torch.zeros(case["csr"], dtype=torch.int32)
    host = validate_values(
        qo,
        kv,
        bt,
        idx,
        case["page_size"],
        case["max_q_len"],
        case["max_kv_len"],
        qo,
        kv,
    )
    if case["causal"]:
        validate_causal_envelope(host)
    return host


_HOST_ROWS = [
    ("valid-dense", {}, None),
    ("valid-csr-exact", dict(csr=3 + 2 + 3), None),
    ("valid-csr-overallocated", dict(csr=300), None),
    ("csr-too-short", dict(csr=7), "kv_page_indices has 7 entries"),
    ("q-zero-padding-row", dict(qo=[0, 4, 4, 8]), None),  # legal (ledger M17)
    ("q-zero-and-kv-zero-row", dict(qo=[0, 4, 4, 8], kv=[10, 0, 9]), None),
    ("q-zero-row-causal", dict(qo=[0, 4, 4, 8], causal=True), None),
    ("indptr-decreasing", dict(qo=[0, 5, 4, 9]), "non-decreasing"),
    ("all-q-zero", dict(qo=[0, 0, 0, 0]), "at least one query token"),
    ("indptr-not-from-zero", dict(qo=[1, 4, 6, 9]), r"qo_indptr\[0\] must be 0"),
    ("max-q-underclaim", dict(max_q_len=2), r"max_q_len \(2\) is smaller"),
    ("kv-zero-padding-row", dict(kv=[10, 0, 9]), None),  # legal padding row
    ("kv-zero-padding-row-causal", dict(kv=[10, 0, 9], causal=True), None),
    ("kv-negative", dict(kv=[10, -1, 9]), "kv_seq_lens"),
    ("causal-q-gt-kv", dict(kv=[10, 1, 9], causal=True), "q_len_i <= kv_len_i"),
    ("noncausal-q-gt-kv-ok", dict(kv=[10, 1, 9], causal=False), None),
    ("max-kv-underclaim", dict(max_kv_len=9), r"max_kv_len \(9\) is smaller"),
    (
        "dense-over-capacity",
        dict(kv=[13, 6, 9], max_kv_len=13),
        "exceeds block_tables capacity",
    ),
]


@pytest.mark.parametrize("label,over,match", _HOST_ROWS, ids=[r[0] for r in _HOST_ROWS])
def test_validate_values_matrix(label, over, match):
    case = _host_case(**over)
    if match is None:
        _validate(case)
    else:
        with pytest.raises(ValueError, match=match):
            _validate(case)


def test_validate_values_rejects_mismatched_mirrors():
    qo = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
    kv = torch.tensor([10, 6, 9], dtype=torch.int32)
    bt = torch.zeros(3, 3, dtype=torch.int32)
    with pytest.raises(ValueError, match="qo_indptr_cpu must be a CPU mirror"):
        validate_values(qo, kv, bt, None, 4, 3, 10, qo[:-1], kv)
    with pytest.raises(ValueError, match="kv_seq_lens_cpu must be a CPU mirror"):
        validate_values(qo, kv, bt, None, 4, 3, 10, qo, kv[:-1])


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("qo_indptr", torch.int64),
        ("qo_indptr", torch.float32),
        ("kv_seq_lens", torch.int64),
        ("kv_seq_lens", torch.float32),
    ],
)
def test_validate_values_rejects_mirrors_of_the_wrong_dtype(name, dtype):
    """A float mirror used to surface as a numpy casting error and an int64
    one was truncated to int32 silently; both are contract violations."""
    qo = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
    kv = torch.tensor([10, 6, 9], dtype=torch.int32)
    bt = torch.zeros(3, 3, dtype=torch.int32)
    mirrors = dict(qo_indptr=qo, kv_seq_lens=kv)
    mirrors[name] = mirrors[name].to(dtype)
    with pytest.raises(
        ValueError, match=f"{name}_cpu must be int32 like {name}, got {dtype}"
    ):
        validate_values(
            qo, kv, bt, None, 4, 4, 10, mirrors["qo_indptr"], mirrors["kv_seq_lens"]
        )


def test_causal_envelope_names_the_offending_request():
    qo = torch.tensor([0, 2, 7, 9], dtype=torch.int32)
    bt = torch.zeros(3, 3, dtype=torch.int32)  # capacity 3 x 4 = 12

    def host(kv):
        kv = torch.tensor(kv, dtype=torch.int32)
        return validate_values(qo, kv, bt, None, 4, 5, int(kv.max()), qo, kv)

    validate_causal_envelope(host([2, 5, 2]))
    with pytest.raises(ValueError, match="request 1 has q_len 5 > kv_len 4"):
        validate_causal_envelope(host([2, 4, 2]))


def test_metadata_constructor_rejects_cpu_tensors_and_paging_form_errors():
    qo = torch.tensor([0, 2], dtype=torch.int32)
    kv = torch.tensor([4], dtype=torch.int32)
    bt = torch.zeros(1, 1, dtype=torch.int32)
    idx = torch.zeros(1, dtype=torch.int32)
    common = dict(page_size=16, max_q_len=2, max_kv_len=4)
    with pytest.raises(ValueError, match="CUDA tensor"):
        PagedAttentionMetadata.dense(qo, kv, bt, **common)
    with pytest.raises(ValueError, match="CUDA tensor"):
        PagedAttentionMetadata.csr(qo, kv, idx, **common)
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        PagedAttentionMetadata(
            qo_indptr=qo, kv_seq_lens=kv, block_tables=bt, kv_page_indices=idx, **common
        )
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        PagedAttentionMetadata(qo_indptr=qo, kv_seq_lens=kv, **common)


# ---------------------------------------------------------------------------
# controller / public class: rejections that need no device
# ---------------------------------------------------------------------------


def test_controller_rejects_calls_before_plan_without_cuda():
    ctl = PagedAttentionController(torch.device("cpu"))
    assert ctl.backend is None and ctl.resolution is None
    z = torch.zeros(1)
    with pytest.raises(ValueError, match="before plan"):
        ctl.run(z, (z, z))
    with pytest.raises(ValueError, match="before plan"):
        ctl.explain()
    with pytest.raises(ValueError, match="PagedAttentionMetadata"):
        ctl.plan(
            "not metadata",
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
        )


def test_public_class_validates_workspace_buffer_without_cuda():
    dev = torch.device("cpu")
    PagedAttention(dev, workspace_buffer=torch.empty(64, dtype=torch.uint8))
    PagedAttention(dev, workspace_buffer=torch.empty(64, dtype=torch.int8))
    with pytest.raises(ValueError, match="1-D"):
        PagedAttention(dev, workspace_buffer=torch.empty(4, 4, dtype=torch.uint8))
    with pytest.raises(ValueError, match="uint8"):
        PagedAttention(dev, workspace_buffer=torch.empty(16, dtype=torch.float32))
    with pytest.raises(ValueError, match="torch.Tensor"):
        PagedAttention(dev, workspace_buffer=1 << 20)
    with pytest.raises(ValueError, match="1-D"):
        PagedAttention(dev, workspace_buffer=torch.empty(64, dtype=torch.uint8)[::2])


# ---------------------------------------------------------------------------
# Transaction: snapshot / restore on fake buffers
# ---------------------------------------------------------------------------


def _staging(n=3, fail_at=None):
    dsts = [torch.full((2,), float(i + 1)) for i in range(n)]
    originals = [d.clone() for d in dsts]
    srcs = [torch.full((2,), 10.0 * (i + 1)) for i in range(n)]
    if fail_at is not None:
        srcs[fail_at] = torch.zeros(3)  # shape mismatch: copy_ raises RuntimeError
    return dsts, srcs, originals


def test_transaction_commit_publishes_every_copy():
    dsts, srcs, _ = _staging()
    with Transaction(list(zip(dsts, srcs, strict=True))) as tx:
        for d, s in zip(dsts, srcs, strict=True):
            assert torch.equal(d, s)  # staged on enter
        tx.commit()
    for d, s in zip(dsts, srcs, strict=True):
        assert torch.equal(d, s)


def test_transaction_body_failure_restores_every_destination():
    dsts, srcs, originals = _staging()
    with (
        pytest.raises(RuntimeError, match="backend plan failed"),
        Transaction(list(zip(dsts, srcs, strict=True))),
    ):
        raise RuntimeError("backend plan failed")
    for d, o in zip(dsts, originals, strict=True):
        assert torch.equal(d, o)


def test_transaction_without_commit_restores():
    dsts, srcs, originals = _staging()
    with Transaction(list(zip(dsts, srcs, strict=True))):
        pass
    for d, o in zip(dsts, originals, strict=True):
        assert torch.equal(d, o)


@pytest.mark.parametrize("fail_at", [0, 1, 2])
def test_transaction_failing_copy_in_enter_restores_earlier_destinations(fail_at):
    """Every copy position may fail (mla-alignment F2 acceptance); the
    destinations written before it must be back at their previous values."""
    dsts, srcs, originals = _staging(fail_at=fail_at)
    with pytest.raises(RuntimeError), Transaction(list(zip(dsts, srcs, strict=True))):
        pass
    for i, (d, o) in enumerate(zip(dsts, originals, strict=True)):
        assert torch.equal(d, o), f"destination {i} not restored: {d.tolist()}"


# ---------------------------------------------------------------------------
# GraphBuffers.preflight / targets with hand-built capacity (no CUDA storage)
# ---------------------------------------------------------------------------

CAP_DENSE = GraphCapacity(
    batch_size=4,
    kv_input_form="block_tables",
    page_size=16,
    max_q_len=32,
    max_kv_len=256,
    total_q_tokens=64,
    table_width=16,
    flat_capacity=64,
)
CAP_CSR = GraphCapacity(
    batch_size=4,
    kv_input_form="page_indices",
    page_size=1,
    max_q_len=32,
    max_kv_len=256,
    total_q_tokens=64,
    flat_capacity=700,  # the flat form derives its dense width (256) on demand
)


def _buffers(cap):
    gb = GraphBuffers.__new__(GraphBuffers)  # __init__ would allocate CUDA storage
    gb.capacity = cap
    gb._device = torch.device("cpu")
    gb.block_tables = None  # reserved on demand by reserve_dense_table()
    gb.rows = None  # bound by the first run()
    gb.stream = None  # bound by the first graph-mode plan()
    return gb


def _meta(cap, **over):
    m = dict(
        batch_size=cap.batch_size,
        kv_input_form=cap.kv_input_form,
        page_size=cap.page_size,
        max_q_len=cap.max_q_len,
        max_kv_len=cap.max_kv_len,
        total_q_tokens=cap.total_q_tokens,
        qo_indptr=torch.zeros(cap.batch_size + 1, dtype=torch.int32),
        kv_seq_lens=torch.zeros(cap.batch_size, dtype=torch.int32),
    )
    if cap.kv_input_form == "block_tables":
        m.update(
            block_tables=torch.zeros(
                cap.batch_size, cap.table_width, dtype=torch.int32
            ),
            kv_page_indices=None,
        )
    else:
        m.update(
            block_tables=None,
            kv_page_indices=torch.zeros(cap.flat_capacity, dtype=torch.int32),
        )
    m.update(over)
    return SimpleNamespace(**m)


_PREFLIGHT_REJECT = [
    ("batch", CAP_DENSE, dict(batch_size=3), "batch_size"),
    (
        "form",
        CAP_DENSE,
        dict(
            kv_input_form="page_indices",
            block_tables=None,
            kv_page_indices=torch.zeros(64, dtype=torch.int32),
        ),
        "kv_input_form",
    ),
    ("page", CAP_DENSE, dict(page_size=32), "page_size"),
    ("max_q-over", CAP_DENSE, dict(max_q_len=64), "max_q_len"),
    ("max_kv-over", CAP_DENSE, dict(max_kv_len=512), "max_kv_len"),
    ("total_q-over", CAP_DENSE, dict(total_q_tokens=80), "total_q_tokens"),
    (
        "width-wider",
        CAP_DENSE,
        dict(block_tables=torch.zeros(4, 17, dtype=torch.int32)),
        "block_tables width",
    ),
    (
        "width-narrower",
        CAP_DENSE,
        dict(block_tables=torch.zeros(4, 15, dtype=torch.int32)),
        "block_tables width",
    ),
    (
        "csr-over-capacity",
        CAP_CSR,
        dict(kv_page_indices=torch.zeros(701, dtype=torch.int32)),
        "kv_page_indices has",
    ),
]


@pytest.mark.parametrize(
    "label,cap,over,match", _PREFLIGHT_REJECT, ids=[r[0] for r in _PREFLIGHT_REJECT]
)
def test_preflight_rejects_what_a_captured_kernel_would_misread(
    label, cap, over, match
):
    with pytest.raises(ValueError, match=match) as ei:
        _buffers(cap).preflight(_meta(cap, **over))
    assert "CUDA graph re-plan" in str(ei.value)


_PREFLIGHT_ACCEPT = [
    ("dense-same", CAP_DENSE, {}),
    ("csr-same", CAP_CSR, {}),
    ("csr-shorter", CAP_CSR, dict(kv_page_indices=torch.zeros(300, dtype=torch.int32))),
    # the maxes and the total token count are upper bounds (capacity
    # substitution): a smaller live batch fits the captured bucket
    ("dense-max_q-under", CAP_DENSE, dict(max_q_len=16)),
    ("dense-max_kv-under", CAP_DENSE, dict(max_kv_len=128)),
    ("dense-total_q-under", CAP_DENSE, dict(total_q_tokens=48)),
]


@pytest.mark.parametrize(
    "label,cap,over", _PREFLIGHT_ACCEPT, ids=[r[0] for r in _PREFLIGHT_ACCEPT]
)
def test_preflight_accepts_a_batch_that_fits(label, cap, over):
    _buffers(cap).preflight(_meta(cap, **over))


def _cpu_reserved(cap):
    gb = _buffers(cap)
    b = cap.batch_size
    i32 = dict(dtype=torch.int32)
    gb.qo_indptr = torch.zeros(b + 1, **i32)
    gb.kv_seq_lens = torch.zeros(b, **i32)
    gb.block_tables = torch.zeros(b, cap.dense_table_width, **i32)
    gb.kv_page_indices = torch.zeros(cap.flat_capacity, **i32)
    gb.q_seq_lens = torch.zeros(b, **i32)
    gb.cum_kv_seq_lens = torch.zeros(b + 1, **i32)
    gb.kv_page_indptr = torch.zeros(b + 1, **i32)
    return gb


_DEVICE_FORMS = frozenset(
    {FORM_Q_SEQ_LENS, FORM_CUM_KV_SEQ_LENS, FORM_KV_PAGE_INDPTR, FORM_KV_PAGE_INDICES}
)


def _fresh(cap, n_flat, dense):
    b = cap.batch_size
    needs = _DEVICE_FORMS | ({FORM_BLOCK_TABLES} if dense else frozenset())
    return Derived(
        needs=needs,
        q_seq_lens=torch.ones(b, dtype=torch.int32),
        cum_kv_seq_lens=torch.arange(b + 1, dtype=torch.int32),
        kv_page_indptr=torch.arange(b + 1, dtype=torch.int32),
        kv_page_indices=torch.arange(n_flat, dtype=torch.int32),
        block_tables=torch.ones(b, cap.dense_table_width, dtype=torch.int32)
        if dense
        else None,
    )


def test_csr_staging_writes_only_the_live_prefix_of_the_reserved_flat_buffer():
    gb = _cpu_reserved(CAP_CSR)
    n = 300
    meta = _meta(CAP_CSR, kv_page_indices=torch.arange(1, n + 1, dtype=torch.int32))
    pairs = gb.targets(meta, _fresh(CAP_CSR, n, dense=False))
    for dst, src in pairs:
        assert dst.shape == src.shape, (dst.shape, src.shape)
    dsts = [d for d, _ in pairs]
    flat = [d for d in dsts if d.data_ptr() == gb.kv_page_indices.data_ptr()]
    assert len(flat) == 1 and flat[0].numel() == n  # a prefix view, not the buffer
    assert not any(d is gb.block_tables for d in dsts)  # no dense candidate
    with Transaction(pairs) as tx:
        tx.commit()
    assert torch.equal(
        gb.kv_page_indices[:n], torch.arange(1, n + 1, dtype=torch.int32)
    )
    assert not gb.kv_page_indices[n:].any()  # the reserved tail is untouched
    # a dense-needing candidate adds the derived table
    pairs = gb.targets(meta, _fresh(CAP_CSR, n, dense=True))
    assert any(d is gb.block_tables for d, _ in pairs)


def test_dense_staging_covers_the_whole_table_and_flat_buffer():
    gb = _cpu_reserved(CAP_DENSE)
    meta = _meta(CAP_DENSE, block_tables=torch.full((4, 16), 7, dtype=torch.int32))
    pairs = gb.targets(meta, _fresh(CAP_DENSE, 64, dense=True))
    dsts = [d for d, _ in pairs]
    assert any(d is gb.block_tables for d in dsts)
    flat = [d for d in dsts if d.data_ptr() == gb.kv_page_indices.data_ptr()]
    assert len(flat) == 1 and flat[0].numel() == CAP_DENSE.flat_capacity  # b*W exactly
    for dst, src in pairs:
        assert dst.shape == src.shape, (dst.shape, src.shape)


def test_derived_view_exposes_reserved_storage_by_identity():
    gb = _cpu_reserved(CAP_DENSE)
    view = gb.derived_view(
        needs=_DEVICE_FORMS, fresh=_fresh(CAP_DENSE, 64, dense=False)
    )
    assert view.block_tables is None
    assert view.kv_page_indices is gb.kv_page_indices
    assert view.q_seq_lens is gb.q_seq_lens
    assert view.cum_kv_seq_lens is gb.cum_kv_seq_lens
    assert view.kv_page_indptr is gb.kv_page_indptr
    dense_view = gb.derived_view(
        needs=_DEVICE_FORMS | {FORM_BLOCK_TABLES},
        fresh=_fresh(CAP_DENSE, 64, dense=True),
    )
    assert dense_view.block_tables is gb.block_tables
    # unrequested forms are None, never a stale reserved buffer
    narrow = gb.derived_view(
        needs={FORM_Q_SEQ_LENS}, fresh=_fresh(CAP_DENSE, 64, dense=False)
    )
    assert narrow.q_seq_lens is gb.q_seq_lens and narrow.kv_page_indices is None


# ---------------------------------------------------------------------------
# import hygiene / public surface
# ---------------------------------------------------------------------------


def test_importing_flashinfer_does_not_load_the_experimental_package():
    code = (
        "import sys, flashinfer, flashinfer.prefill as p\n"
        "assert p.PagedAttention and p.resolve_paged_attention\n"
        "print('EXPERIMENTAL_LOADED', "
        "'flashinfer.experimental.paged_attention' in sys.modules)\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "EXPERIMENTAL_LOADED False" in proc.stdout


def test_lazy_value_types_are_reachable_from_prefill():
    from flashinfer import _paged_attention as entry
    from flashinfer import prefill

    assert prefill.PagedAttentionMetadata is pa.PagedAttentionMetadata
    assert prefill.Resolution is pa.Resolution
    assert prefill.PagedAttentionCapabilities is pa.PagedAttentionCapabilities
    assert entry.BackendCapability is pa.PagedAttentionCapabilities  # pre-rename alias
    assert entry.CAPABILITIES is pa.CAPABILITIES
    assert {"PagedAttentionMetadata", "Resolution", "CAPABILITIES"} <= set(dir(entry))
    with pytest.raises(AttributeError):
        prefill.NoSuchPagedAttentionThing  # noqa: B018
    with pytest.raises(AttributeError):
        entry.NoSuchPagedAttentionThing  # noqa: B018


@pytest.mark.parametrize("cc_major", [8, 9, 10, 12])
@pytest.mark.parametrize(
    "feature,reason",
    [
        ({"logits_soft_cap": 30.0}, "logits soft cap not supported"),
        ({"custom_mask": True}, "custom attention mask not supported"),
    ],
)
def test_feature_rejection_reasons_follow_the_architecture_gate(
    no_probes, cc_major, feature, reason
):
    """Static: on every architecture a backend the CC admits but the feature
    excludes carries the feature reason; a backend the CC does not admit is
    excluded for the CC first (review R3 — the GPU tests must not assume the
    B200 candidate set)."""
    plain = _resolve(cc_major)
    res = _resolve(cc_major, **feature)
    assert res.backends, f"cc_major={cc_major}: the fa backends must remain"
    assert set(res.backends) <= {"fa2", "fa3"}
    for name in CAPABILITIES:
        if name in res.backends:
            continue
        why = res.excluded[name]
        if name in plain.backends:
            assert reason in why, (cc_major, name, why)
        else:
            assert "compute capability" in why or name in plain.excluded, (
                cc_major,
                name,
                why,
            )
