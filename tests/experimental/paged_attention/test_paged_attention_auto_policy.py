"""``backend="auto"`` of PagedAttention (experimental): the plan-time policy.

Three layers, mirroring ``tests/attention/test_mla_auto_backend.py``:

1. Policy (no GPU): ``_auto_policy`` orders every candidate from host facts;
   the SM100 table sits on both sides of each threshold and the ported cuDNN
   region is pinned against the legacy wrapper's predicate.
2. Planning (no GPU, fake backends): the controller walks the policy's order,
   moves on only after a typed rejection, and freezes the choice in graph mode.
3. Execution (GPU): auto against the fp32 oracle in each region, eager and
   under capture / update / replay.  Only the tests whose purpose is the
   policy assert ``attn.backend``.
"""

import dataclasses
import itertools
import math

import pytest
import torch

import flashinfer.prefill as legacy_prefill
from flashinfer.experimental.paged_attention import _auto_policy, _controller
from flashinfer.experimental.paged_attention import _selection
from flashinfer.experimental.paged_attention._auto_policy import (
    _AUTO_BACKEND_CANDIDATES,
    _PlanFacts,
    _in_sm100_cudnn_region,
    ordered_backends,
    ordered_sm100_backends,
    plan_order,
)
from flashinfer.experimental.paged_attention._backends import CAPABILITIES
from flashinfer.experimental.paged_attention._backends._capabilities import (
    _BackendPlanUnsupportedError,
)
from flashinfer.experimental.paged_attention._contracts import PagedAttentionMetadata
from flashinfer.experimental.paged_attention._controller import (
    PagedAttentionController,
)
from flashinfer.experimental.paged_attention._graph import GraphCapacity
from flashinfer.experimental.paged_attention._planning import Derived
from flashinfer.experimental.paged_attention._selection import (
    resolve_paged_attention as _resolve,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import LSE_TOL, OUT_TOL, make_metadata

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# --------------------------------------------------------------------------
# 1. policy
# --------------------------------------------------------------------------


def _indptr(q_lens):
    return torch.tensor([0, *itertools.accumulate(q_lens)], dtype=torch.int32)


def _facts(q_lens, kv_lens, *, hq=32, hk=8, host=True, graph=False, **over):
    """Facts of one batch; ``over`` replaces any field (declared bounds,
    features, ...).  ``host=False`` models device-only lengths."""
    values = dict(
        batch_size=len(q_lens),
        total_q_tokens=sum(q_lens),
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        num_qo_heads=hq,
        num_kv_heads=hk,
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
        use_cuda_graph=graph,
        qo_indptr=_indptr(q_lens) if host else None,
        kv_seq_lens=torch.tensor(kv_lens, dtype=torch.int32) if host else None,
    )
    values.update(over)
    return _PlanFacts(**values)


@pytest.fixture
def cudnn_qualified(monkeypatch):
    """The cuDNN version gate answers yes, wherever the test runs."""
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)


# (id, q_lens, kv_lens, facts overrides, first preference).  Heads are 32/8
# unless overridden.  Shapes marked [m] are measured cases of the GB200 fit
# (reports/prefill-auto-design-20261109, results/gb200-*-f2*); the others
# step one unit across a threshold from one.
_SM100_TABLE = [
    # ---- rows >= 64 at head_dim 128 (fill 1.0; KV outside the cuDNN region)
    ("rows64-graph [m]", [256], [131072], dict(graph=True), "trtllm-gen"),
    ("rows63-graph", [252], [131072], dict(graph=True), "fa2"),
    ("rows64-eager", [256], [131072], {}, "trtllm-gen"),
    ("rows63-eager", [252], [131072], {}, "fa2"),
    ("rows48 [m]", [256], [4096], dict(hq=24, graph=True), "fa2"),
    # ---- rows >= 96 below head_dim 128
    (
        "d64-rows96 [m]",
        [128] * 3,
        [16384] * 3,
        dict(head_dim_qk=64, head_dim_vo=64, graph=True),
        "trtllm-gen",
    ),
    (
        "d64-rows64 [m]",
        [128] * 2,
        [16384] * 2,
        dict(head_dim_qk=64, head_dim_vo=64, graph=True),
        "fa2",
    ),
    (
        "d64-rows95",
        [128, 128, 124],
        [16384] * 3,
        dict(head_dim_qk=64, head_dim_vo=64),
        "fa2",
    ),
    # ---- fill >= 0.375 in graph mode
    ("fill0.375-graph [m]", [48] * 8, [2048] * 8, dict(graph=True), "trtllm-gen"),
    ("fill0.367-graph", [47] * 8, [2048] * 8, dict(graph=True), "fa2"),
    ("fill0.25-graph [m]", [32] * 16, [2048] * 16, dict(graph=True), "fa2"),
    ("fill0.19-graph [m]", [24] * 16, [8192] * 16, dict(graph=True), "fa2"),
    # ---- fill >= 0.75 eager: the same half-filled tiles stay on fa2
    ("fill0.375-eager [m]", [48] * 16, [2048] * 16, {}, "fa2"),
    ("fill0.5-eager [m]", [64] * 16, [2048] * 16, {}, "fa2"),
    ("fill0.5-graph [m]", [64] * 16, [2048] * 16, dict(graph=True), "trtllm-gen"),
    ("fill0.75-eager [m]", [96] * 8, [2048] * 8, {}, "trtllm-gen"),
    ("fill0.74-eager", [95] * 8, [2048] * 8, {}, "fa2"),
    # ragged: the grid follows the longest request, the rows the total
    ("ragged-fill0.75", [128, 64], [1024, 1024], dict(hq=64), "trtllm-gen"),
    ("ragged-fill0.74", [128, 63], [1024, 1024], dict(hq=64), "fa2"),
    # decode / speculative lengths never fill a tile
    ("decode [m]", [2] * 256, [2048] * 256, dict(hq=64, graph=True), "fa2"),
    # ---- cuDNN region, prefix: 64 <= max_q <= 1024, 2048 <= max_kv <= 32768
    ("prefix-q128-kv16384 [m]", [128], [16384], {}, "cudnn"),
    ("prefix-q64 [m]", [64], [16384], {}, "cudnn"),
    ("prefix-q63", [63], [16384], {}, "fa2"),
    ("prefix-q1024 [m]", [1024], [32768], dict(hq=8, hk=1), "cudnn"),
    ("prefix-q1025", [1025], [32768], dict(hq=8, hk=1), "trtllm-gen"),
    ("prefix-kv2048 [m]", [128], [2048], {}, "cudnn"),
    ("prefix-kv2047 [m]", [128], [2047], {}, "fa2"),
    ("prefix-kv32768 [m]", [128] * 3, [32768] * 3, dict(hq=16, hk=4), "cudnn"),
    ("prefix-kv32784", [128] * 3, [32784] * 3, dict(hq=16, hk=4), "fa2"),
    ("prefix-kv=4q", [512], [2048], {}, "cudnn"),
    ("prefix-kv<4q", [513], [2048], {}, "trtllm-gen"),
    # exact work = heads * sum(q_i * kv_i) >= 8 Mi
    ("work-8Mi", [256], [2048], dict(hq=16, hk=4), "cudnn"),
    ("work-below-8Mi", [255], [2048], dict(hq=16, hk=4), "fa2"),
    (
        "work-exact-ragged-out",
        [64] * 4,
        [8192, 64, 64, 64],
        dict(hq=8, hk=2),
        "fa2",
    ),
    # ... the bound-based test when the lengths are not on the host
    (
        "work-bound-device-lengths",
        [64] * 4,
        [8192, 64, 64, 64],
        dict(hq=8, hk=2, host=False),
        "cudnn",
    ),
    ("batch4 [m]", [64] * 4, [8192] * 4, {}, "cudnn"),
    ("batch5", [64] * 5, [8192] * 5, {}, "fa2"),
    ("batch6-rows96 [m]", [128] * 6, [8192] * 6, dict(hq=16, hk=4), "trtllm-gen"),
    ("heads8", [1024], [4096], dict(hq=8, hk=2), "cudnn"),
    ("heads4", [2048], [8192], dict(hq=4, hk=1), "trtllm-gen"),
    ("heads64 [m]", [128], [8192], dict(hq=64, hk=8), "cudnn"),
    ("heads128", [128], [8192], dict(hq=128, hk=8), "trtllm-gen"),
    ("group16 [m]", [128], [8192], dict(hq=32, hk=2), "cudnn"),
    ("group32", [128], [8192], dict(hq=32, hk=1), "fa2"),
    ("group3", [256], [4096], dict(hq=24, hk=8), "fa2"),
    # ---- cuDNN region, full prefill: 1024 <= max_q == max_kv <= 8192
    ("full-1024 [m]", [1024], [1024], dict(hq=8, hk=1), "cudnn"),
    ("full-1023", [1023], [1023], dict(hq=8, hk=1), "fa2"),
    ("full-8192 [m]", [8192], [8192], dict(hq=64, hk=8), "cudnn"),
    ("full-8193", [8193], [8193], dict(hq=64, hk=8), "trtllm-gen"),
    ("full-ragged [m]", [4921], [4921], dict(hq=16, hk=4), "cudnn"),
    ("full-each-q=kv", [2048, 1024], [2048, 1024], {}, "cudnn"),
    ("full-mixed-prefix", [2048, 1024], [2048, 2048], {}, "trtllm-gen"),
    # equal maxima alone do not prove full prefill
    ("full-device-lengths", [2048], [2048], dict(host=False), "trtllm-gen"),
    # ---- the feature conditions around the legacy call
    ("window", [128], [16384], dict(window_left=127), "fa2"),
    ("window-rows256 [m]", [128] * 8, [8192] * 8, dict(window_left=1023), "trtllm-gen"),
    ("non-causal", [128], [16384], dict(causal=False), "fa2"),
    ("soft-cap", [128], [16384], dict(logits_soft_cap=30.0), "fa2"),
    ("custom-mask", [128], [16384], dict(custom_mask=True), "fa2"),
    ("sinks", [128], [16384], dict(use_sinks=True), "fa2"),
    ("fp8-kv", [128], [16384], dict(kv_dtype=torch.float8_e4m3fn), "fa2"),
    (
        "fp16",
        [128],
        [16384],
        dict(q_dtype=torch.float16, kv_dtype=torch.float16),
        "fa2",
    ),
    ("page32", [128], [16384], dict(page_size=32), "fa2"),
    ("nhd", [128], [16384], dict(kv_layout="NHD"), "fa2"),
    ("d64", [128], [16384], dict(head_dim_qk=64, head_dim_vo=64), "fa2"),
    ("d192-128", [128], [16384], dict(head_dim_qk=192), "fa2"),
    # ---- graph mode decides on the capacity's bounds
    (
        "graph-capacity-bounds",
        [4, 4],
        [64, 64],
        dict(graph=True, total_q_tokens=1024, max_q_len=512, max_kv_len=1024),
        "trtllm-gen",
    ),
    (
        "graph-capacity-region",
        [64],
        [16384],
        dict(graph=True, total_q_tokens=128, max_q_len=128, max_kv_len=16384),
        "cudnn",
    ),
    # full prefill in a bucket whose KV capacity is wider than the query
    # (GB200 graph: B1 32/8 q = kv = 4921, capacity 8192: cudnn 159, trtllm-gen 205)
    (
        "graph-full-wider-capacity [m]",
        [4921],
        [4921],
        dict(graph=True, max_kv_len=8192),
        "cudnn",
    ),
    (
        "graph-full-capacity-above-limit",
        [4921],
        [4921],
        dict(graph=True, max_kv_len=8208),
        "trtllm-gen",
    ),
    (
        "graph-wider-capacity-not-full",
        [4921],
        [4920],
        dict(graph=True, max_kv_len=8192),
        "trtllm-gen",
    ),
    ("eager-bounds-must-match", [4921], [4921], dict(max_kv_len=8192), "trtllm-gen"),
]


@pytest.mark.parametrize(
    "q_lens,kv_lens,over,first",
    [row[1:] for row in _SM100_TABLE],
    ids=[row[0] for row in _SM100_TABLE],
)
def test_sm100_first_preference(cudnn_qualified, q_lens, kv_lens, over, first):
    facts = _facts(q_lens, kv_lens, **over)
    order = ordered_sm100_backends(facts)
    assert order[0] == first
    assert sorted(order) == sorted(_AUTO_BACKEND_CANDIDATES)
    assert ordered_sm100_backends(facts) == order
    assert ordered_backends((10, 0), facts) == order


def test_sm100_fallback_order(cudnn_qualified):
    """Inside the region cuDNN leads and the measured pair follows in its own
    order; outside it cuDNN is the last resort."""
    region_rows_small = _facts([128], [16384])
    region_rows_large = _facts([256] * 2, [8192] * 2, hq=64, hk=8)
    assert ordered_sm100_backends(region_rows_small)[:3] == (
        "cudnn",
        "fa2",
        "trtllm-gen",
    )
    assert ordered_sm100_backends(region_rows_large)[:3] == (
        "cudnn",
        "trtllm-gen",
        "fa2",
    )
    for facts in (_facts([252], [131072]), _facts([256], [131072])):
        assert ordered_sm100_backends(facts)[-1] == "cudnn"


def test_sm100_region_needs_the_cudnn_version_gate(monkeypatch):
    facts = _facts([128], [16384])
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: False)
    assert ordered_sm100_backends(facts)[0] == "fa2"
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    assert ordered_sm100_backends(facts)[0] == "cudnn"


def test_sm100_cudnn_region_equals_the_legacy_predicate(monkeypatch):
    """The policy duplicates ``_blackwell_paged_auto_cudnn``; this pins the
    copy to the original over shapes, heads, dtypes and both length paths."""
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(legacy_prefill, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(legacy_prefill, "get_compute_capability", lambda dev: (10, 0))
    batches = [
        ([128], [2048]),
        ([64], [2048]),
        ([63], [2048]),
        ([128], [2047]),
        ([1024], [4096]),
        ([1025], [8192]),
        ([512], [2048]),
        ([513], [2048]),
        ([128], [32768]),
        ([128], [32784]),
        ([1024], [1024]),
        ([1023], [1023]),
        ([8192], [8192]),
        ([8193], [8193]),
        ([2048, 1024], [2048, 2048]),
        ([2048, 1024], [2048, 1024]),
        ([2048, 0], [2048, 0]),
        ([128, 100], [4096, 3000]),
        ([64, 1, 1, 1], [8192, 16, 16, 16]),
        ([64] * 4, [8192, 64, 64, 64]),
        ([256] * 4, [8192] * 4),
        ([256] * 5, [8192] * 5),
    ]
    heads = [(4, 4), (8, 8), (8, 1), (12, 4), (16, 4), (32, 8), (32, 2), (32, 1)]
    heads += [(64, 8), (128, 8)]
    bf16, fp16, fp8 = torch.bfloat16, torch.float16, torch.float8_e4m3fn
    admitted = 0
    total = 0
    for (q_lens, kv_lens), (hq, hk), head_dim, page, layout, (
        qd,
        kd,
    ), host, loose in itertools.product(
        batches,
        heads,
        (128, 64),
        (16, 32),
        ("HND", "NHD"),
        ((bf16, bf16), (fp16, fp16), (bf16, fp8)),
        (True, False),
        (False, True),
    ):
        max_q, max_kv = max(q_lens), max(kv_lens)
        if loose:  # declared bounds above the batch's own (a graph capacity)
            max_kv = 1 << (max_kv - 1).bit_length()
        qo_indptr = _indptr(q_lens)
        kv = torch.tensor(kv_lens, dtype=torch.int32)
        want = legacy_prefill._blackwell_paged_auto_cudnn(
            None,
            layout,
            qd,
            kd,
            qd,
            head_dim,
            head_dim,
            len(q_lens),
            hq,
            hk,
            max_q,
            max_kv,
            page,
            sum(q_lens),
            qo_indptr,
            kv if host else None,
        )
        facts = _facts(
            q_lens,
            kv_lens,
            hq=hq,
            hk=hk,
            host=host,
            max_kv_len=max_kv,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            page_size=page,
            kv_layout=layout,
            q_dtype=qd,
            kv_dtype=kd,
        )
        got = _in_sm100_cudnn_region(facts)
        assert got == want, (q_lens, kv_lens, hq, hk, head_dim, page, layout, kd, host)
        admitted += got
        total += 1
    assert 100 < admitted < total - 100  # the grid exercises both answers


def test_policy_rereads_host_lengths_on_each_call(cudnn_qualified):
    facts = _facts([256], [2048], hq=16, hk=4)  # exactly 8 Mi of work
    assert ordered_sm100_backends(facts)[0] == "cudnn"
    facts.kv_seq_lens[0] = 2047  # the caller mutates its mirror in place
    assert ordered_sm100_backends(facts)[0] == "fa2"
    facts.kv_seq_lens[0] = 2048
    assert ordered_sm100_backends(facts)[0] == "cudnn"


@pytest.mark.parametrize(
    "over,match",
    [
        (dict(qo_indptr=_indptr([64, 64])), "query spans"),
        (dict(kv_seq_lens=torch.tensor([8192, 8192], dtype=torch.int32)), "KV lengths"),
        (dict(kv_seq_lens=torch.tensor([20000], dtype=torch.int32)), "max_kv_len"),
        (dict(qo_indptr=torch.tensor([0, 2000], dtype=torch.int32)), "max_q_len"),
        (dict(qo_indptr=torch.tensor([128, 0], dtype=torch.int32)), "query lengths"),
        (dict(kv_seq_lens=torch.tensor([-1], dtype=torch.int32)), "KV lengths"),
    ],
)
def test_malformed_host_lengths_raise(cudnn_qualified, over, match):
    """Never a silent fall back to another order."""
    facts = _facts([128], [16384], **over)
    with pytest.raises(ValueError, match=match):
        ordered_sm100_backends(facts)


@pytest.mark.parametrize(
    "over",
    [
        dict(batch_size=0),
        dict(total_q_tokens=0),
        dict(max_q_len=0),
        dict(max_kv_len=-1),
        dict(num_kv_heads=0),
        dict(batch_size=True),
        dict(max_q_len=128.0),
    ],
)
def test_malformed_facts_raise(over):
    with pytest.raises(ValueError, match="auto policy"):
        _facts([128], [16384], **over)


def test_policy_never_reads_a_device_tensor(cudnn_qualified):
    """A tensor that is not on the host is treated as absent (the bound-based
    answer), not moved: a fake stands in for a CUDA tensor so this runs
    anywhere."""

    class _DeviceOnly:
        device = torch.device("cuda", 0)

        def __getattr__(self, name):
            raise AssertionError(f"device tensor touched: .{name}")

    facts = _facts(
        [64] * 4,
        [8192, 64, 64, 64],
        hq=8,
        hk=2,
        qo_indptr=_DeviceOnly(),
        kv_seq_lens=_DeviceOnly(),
    )
    assert facts.host_lengths() is None
    assert ordered_sm100_backends(facts)[0] == "cudnn"


@pytest.mark.parametrize(
    "cc,q_lens,over,order",
    [
        ((8, 0), [128], {}, ("fa2", "cudnn")),
        ((8, 6), [1], {}, ("fa2", "cudnn")),
        ((9, 0), [128], {}, ("fa3", "fa2", "cudnn")),
        ((9, 0), [65], {}, ("fa3", "fa2", "cudnn")),
        ((9, 0), [64], {}, ("fa2", "fa3", "cudnn")),
        ((9, 0), [64], dict(q_dtype=torch.float8_e4m3fn), ("fa3", "fa2", "cudnn")),
        ((10, 3), [128], {}, ("trtllm-gen", "cudnn", "fa2")),
        ((10, 3), [17], {}, ("trtllm-gen", "cudnn", "fa2")),
        ((10, 3), [16], {}, ("fa2", "trtllm-gen", "cudnn")),
        ((12, 0), [128], {}, ("fa2", "cudnn")),
        ((12, 1), [1], {}, ("fa2", "cudnn")),
    ],
)
def test_other_architectures_keep_their_order(cudnn_qualified, cc, q_lens, over, order):
    """Only exact SM100 has the measured rule; the same facts that pick cuDNN
    there leave every other architecture on its previous order."""
    facts = _facts(q_lens, [16384], **over)
    full = ordered_backends(cc, facts)
    assert full[: len(order)] == order
    assert sorted(full) == sorted(_AUTO_BACKEND_CANDIDATES)
    assert ordered_backends(cc, facts) == full


def test_default_order_matches_the_resolve_time_table(cudnn_qualified):
    """With a long query the plan-time order of the unmeasured architectures
    is the resolve-time default bucket."""
    facts = _facts([4096], [200704])
    for cc in ((8, 0), (9, 0), (10, 3), (12, 0)):
        static = _selection.heuristic_order(cc[0])
        assert plan_order(static, cc, facts) == static


def test_cake_is_not_an_auto_candidate(cudnn_qualified):
    assert "cake" not in _AUTO_BACKEND_CANDIDATES
    assert set(_AUTO_BACKEND_CANDIDATES) <= set(CAPABILITIES)
    for cc in ((8, 0), (9, 0), (10, 0), (10, 3), (11, 0), (12, 0)):
        for facts in (_facts([128], [16384]), _facts([1], [64])):
            assert "cake" not in ordered_backends(cc, facts)
    assert ordered_backends((11, 0), _facts([1], [64])) == _AUTO_BACKEND_CANDIDATES


def test_plan_order_never_changes_the_candidate_set(cudnn_qualified):
    facts = _facts([128], [16384])
    assert plan_order(("fa2", "cudnn"), (10, 0), facts) == ("cudnn", "fa2")
    assert plan_order(("trtllm-gen", "fa2"), (10, 0), facts) == ("fa2", "trtllm-gen")
    # a member the policy does not rank keeps its place after the ranked ones
    assert plan_order(("cake", "fa2", "cudnn"), (10, 0), facts) == (
        "cudnn",
        "fa2",
        "cake",
    )
    assert plan_order(("fa2",), (10, 0), facts) == ("fa2",)


def test_resolve_excludes_cake_from_auto_only(monkeypatch):
    monkeypatch.setattr(
        _selection, "PROBES", {name: (lambda device: None) for name in CAPABILITIES}
    )
    cfg = dict(
        cc_major=10,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
    )
    auto = _resolve(**cfg)
    assert set(auto.backends) == {"trtllm-gen", "cudnn", "fa2"}
    assert "not an auto candidate" in auto.excluded["cake"]
    assert "backend='cake'" in auto.explain()
    assert _resolve(backend="cake", **cfg).backends == ("cake",)


# --------------------------------------------------------------------------
# 2. planning (CPU, fake backends)
# --------------------------------------------------------------------------


class _HostMetadata(PagedAttentionMetadata):
    """Metadata on the host: the real class requires CUDA tensors and derives
    device forms; the controller's selection needs neither."""

    def __post_init__(self):
        pass

    @property
    def total_q_tokens(self):
        return int(self.qo_indptr_cpu[-1])

    def validate_causal_envelope(self):
        pass

    def derived(self, *, needs, max_kv_len=None):
        return Derived(needs=frozenset())


def _host_metadata(q_lens, kv_lens, *, width=None, max_q_len=None):
    qo_indptr = _indptr(q_lens)
    kv = torch.tensor(kv_lens, dtype=torch.int32)
    width = width or math.ceil(max(kv_lens) / 16)
    return _HostMetadata(
        qo_indptr=qo_indptr.clone(),
        kv_seq_lens=kv.clone(),
        page_size=16,
        max_q_len=max_q_len or max(q_lens),
        max_kv_len=max(kv_lens),
        block_tables=torch.zeros(len(q_lens), width, dtype=torch.int32),
        qo_indptr_cpu=qo_indptr,
        kv_seq_lens_cpu=kv,
    )


_HOST_KW = dict(
    num_qo_heads=16, num_kv_heads=4, head_dim_qk=128, q_dtype=torch.bfloat16
)
# One batch per SM100 region at 16/4 heads, with the full order it gets.
_CUDNN_BATCH = ([128], [4096])  # 8 Mi of work, 16 rows
_TRTLLM_BATCH = ([512, 512], [600, 1024])  # 128 rows, fill 1.0
_FA2_BATCH = ([16, 7], [300, 97])
_CUDNN_ORDER = ["cudnn", "fa2", "trtllm-gen"]
_TRTLLM_ORDER = ["trtllm-gen", "fa2", "cudnn"]
_FA2_ORDER = ["fa2", "trtllm-gen", "cudnn"]


class _Planners:
    """Fake backends behind the real controller, like MLA's ``_cpu_planners``."""

    def __init__(self):
        self.preflights = []
        self.plans = []
        self.runs = []
        self.reject = {}  # name -> exception raised by preflight()
        self.plan_error = {}  # name -> exception raised by plan()

    def make_backend(self, name, device, kv_layout, workspace, *, graph_capacity=None):
        planners = self

        class _Backend:
            def preflight(self, meta):
                planners.preflights.append(name)
                if name in planners.reject:
                    raise planners.reject[name]

            def plan(self, meta, derived):
                if name in planners.plan_error:
                    raise planners.plan_error[name]
                planners.plans.append(name)

            def run(self, q, k_cache, v_cache, **kwargs):
                planners.runs.append(name)
                return q, None

        return _Backend()

    def decline(self, *names):
        self.reject = {
            name: _BackendPlanUnsupportedError(f"{name} declines") for name in names
        }

    def controller(self, **kwargs):
        return PagedAttentionController(
            torch.device("cpu"),
            workspace_buffer=torch.empty(8, dtype=torch.uint8),
            **kwargs,
        )


def _cpu_planners(monkeypatch, cc=(10, 0)):
    planners = _Planners()
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(
        _selection, "PROBES", {name: (lambda device: None) for name in CAPABILITIES}
    )
    monkeypatch.setattr(
        _controller,
        "resolve_paged_attention",
        lambda *, device, **kwargs: _resolve(cc_major=cc[0], **kwargs),
    )
    monkeypatch.setattr(
        PagedAttentionController, "_device_binding", lambda self: (*cc, 0)
    )
    monkeypatch.setattr(_controller, "make_backend", planners.make_backend)
    monkeypatch.setattr(_controller, "derived_needs", lambda name: frozenset())
    monkeypatch.setattr(_controller, "_expect_not_capturing", lambda what: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")
    return planners


@pytest.fixture
def planners(monkeypatch):
    return _cpu_planners(monkeypatch)


@pytest.mark.parametrize(
    "batch,order",
    [
        (_CUDNN_BATCH, _CUDNN_ORDER),
        (_TRTLLM_BATCH, _TRTLLM_ORDER),
        (_FA2_BATCH, _FA2_ORDER),
    ],
    ids=["cudnn", "trtllm-gen", "fa2"],
)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_auto_resolves_at_plan(planners, batch, order, graph):
    attn = planners.controller(use_cuda_graph=graph)
    assert attn.backend is None and attn.selection_order == ()
    attn.plan(_host_metadata(*batch), **_HOST_KW)
    assert attn.backend == order[0]
    assert list(attn.selection_order) == order
    assert planners.preflights == planners.plans == [order[0]]
    assert attn.selection_trace == ((order[0], "preflight", "accepted"),)
    assert f"plan-time order: {order}" in attn.explain()


@pytest.mark.parametrize(
    "batch,order",
    [
        (_CUDNN_BATCH, _CUDNN_ORDER),
        (_TRTLLM_BATCH, _TRTLLM_ORDER),
        (_FA2_BATCH, _FA2_ORDER),
    ],
    ids=["cudnn", "trtllm-gen", "fa2"],
)
def test_auto_reaches_every_candidate_after_typed_rejections(planners, batch, order):
    for accepted in range(len(order)):
        planners.preflights.clear()
        planners.decline(*order[:accepted])
        attn = planners.controller()
        attn.plan(_host_metadata(*batch), **_HOST_KW)
        assert attn.backend == order[accepted]
        assert planners.preflights == order[: accepted + 1]
        trace = attn.selection_trace
        assert [name for name, _, _ in trace] == order[: accepted + 1]
        for name, phase, reason in trace[:-1]:
            assert (phase, reason) == ("preflight", f"{name} declines")
        text = attn.explain()
        for name in order[:accepted]:
            assert f"{name} [preflight]: {name} declines" in text


def test_all_rejected_message_is_deterministic(planners):
    planners.decline(*_TRTLLM_ORDER)
    messages = []
    for _ in range(2):
        attn = planners.controller()
        with pytest.raises(ValueError) as error:
            attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
        messages.append(str(error.value))
        assert attn.backend is None
    assert messages[0] == messages[1]
    assert messages[0] == (
        "no pinned candidate can plan this batch (candidates ['trtllm-gen', 'fa2', "
        "'cudnn']: trtllm-gen: trtllm-gen declines; fa2: fa2 declines; "
        "cudnn: cudnn declines)"
    )
    assert planners.plans == []


@pytest.mark.parametrize(
    "error", [RuntimeError("out of memory"), ValueError("bad input"), KeyError("jit")]
)
def test_fatal_preflight_errors_propagate(planners, error):
    planners.reject = {"trtllm-gen": error}
    attn = planners.controller()
    with pytest.raises(type(error)) as raised:
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert raised.value is error
    assert planners.preflights == ["trtllm-gen"]  # no fallback after a fatal error
    assert attn.backend is None


def test_fatal_plan_errors_propagate(planners):
    planners.plan_error = {"trtllm-gen": RuntimeError("compile failed")}
    attn = planners.controller()
    with pytest.raises(RuntimeError, match="compile failed"):
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert planners.preflights == ["trtllm-gen"] and planners.plans == []
    assert attn.backend is None


def test_explicit_backend_is_strict(planners):
    """A named backend is planned as named, whatever the policy prefers, and
    its typed rejection is an error rather than a fallback."""
    attn = planners.controller()
    attn.plan(_host_metadata(*_TRTLLM_BATCH), backend="fa2", **_HOST_KW)
    assert attn.backend == "fa2" and planners.preflights == ["fa2"]
    assert attn.selection_order == ("fa2",)
    planners.preflights.clear()
    planners.decline("fa2")
    with pytest.raises(ValueError, match="'fa2' cannot plan this batch: fa2 declines"):
        attn.plan(_host_metadata(*_TRTLLM_BATCH), backend="fa2", **_HOST_KW)
    assert planners.preflights == ["fa2"]


def test_cake_is_never_chosen_by_auto(planners):
    planners.decline(*_TRTLLM_ORDER)
    with pytest.raises(ValueError, match="no pinned candidate") as error:
        planners.controller().plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert "cake" not in planners.preflights and "cake" not in str(error.value)
    # ... and it still runs when named
    planners.decline()
    attn = planners.controller()
    attn.plan(_host_metadata(*_TRTLLM_BATCH), backend="cake", **_HOST_KW)
    assert attn.backend == "cake"


def test_eager_replan_reselects(planners):
    attn = planners.controller()
    for batch, order in (
        (_TRTLLM_BATCH, _TRTLLM_ORDER),
        (_FA2_BATCH, _FA2_ORDER),
        (_CUDNN_BATCH, _CUDNN_ORDER),
        (_TRTLLM_BATCH, _TRTLLM_ORDER),
    ):
        attn.plan(_host_metadata(*batch), **_HOST_KW)
        assert attn.backend == order[0]
        assert list(attn.selection_order) == order
    assert planners.plans == ["trtllm-gen", "fa2", "cudnn", "trtllm-gen"]


def test_graph_mode_keeps_the_first_backend(planners):
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert attn.backend == "trtllm-gen"
    # a batch of the same bucket that eager auto would give to fa2
    small = _host_metadata([4, 4], [64, 64], width=64)
    planners.preflights.clear()
    attn.plan(small, **_HOST_KW)
    assert attn.backend == "trtllm-gen"
    assert planners.preflights == ["trtllm-gen"]
    attn.update(small)
    assert attn.backend == "trtllm-gen" and attn.selection_order == ("trtllm-gen",)
    eager = planners.controller()
    eager.plan(small, **_HOST_KW)
    assert eager.backend == "fa2"


def test_graph_mode_typed_fallback_freezes_the_selection(planners):
    planners.decline("trtllm-gen")
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert attn.backend == "fa2"
    # the first choice would accept now; the capture keeps fa2
    planners.decline()
    planners.preflights.clear()
    attn.update(_host_metadata(*_TRTLLM_BATCH))
    assert attn.backend == "fa2" and planners.preflights == ["fa2"]
    # the frozen backend declining later is an error, not a switch
    planners.decline("fa2")
    with pytest.raises(ValueError, match="no pinned candidate"):
        attn.update(_host_metadata(*_TRTLLM_BATCH))
    assert attn.backend == "fa2"


def test_graph_mode_ranks_on_the_capacity(planners):
    """The kernels are planned with the capacity's bounds, so those decide —
    not the first batch that happens to arrive."""
    capacity = GraphCapacity(
        batch_size=2,
        total_q_tokens=1024,
        max_q_len=512,
        max_kv_len=1024,
        page_size=16,
        table_width=64,
    )
    attn = planners.controller(graph_capacity=capacity)
    attn.plan(_host_metadata([4, 4], [64, 64], width=64), **_HOST_KW)
    assert attn.backend == "trtllm-gen"
    assert list(attn.selection_order) == _TRTLLM_ORDER


def test_failed_replan_keeps_the_published_plan_runnable(planners):
    attn = planners.controller()
    md = _host_metadata(*_FA2_BATCH)
    attn.plan(md, **_HOST_KW)
    q = torch.zeros(md.total_q_tokens, 16, 128, dtype=torch.bfloat16)
    cache = torch.zeros(4, 4, 16, 128, dtype=torch.bfloat16)
    order_before, trace_before = attn.selection_order, attn.selection_trace

    planners.decline(*_TRTLLM_ORDER)
    with pytest.raises(ValueError, match="no pinned candidate"):
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    planners.decline()
    planners.plan_error = {"trtllm-gen": RuntimeError("compile failed")}
    with pytest.raises(RuntimeError, match="compile failed"):
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)

    assert attn.backend == "fa2"
    assert attn.selection_order == order_before
    assert attn.selection_trace == trace_before
    attn.run(q, (cache, cache))
    assert planners.runs == ["fa2"]


def test_policy_reads_the_metadata_mirrors_on_each_plan(planners):
    """One metadata object planned twice: an in-place change of its host
    mirror between the plans changes the selection."""
    attn = planners.controller()
    md = _host_metadata([256], [2048])  # 16 heads: exactly 8 Mi of work
    attn.plan(md, **_HOST_KW)
    assert attn.backend == "cudnn"
    md.kv_seq_lens_cpu[0] = 2000
    attn.plan(md, **_HOST_KW)
    assert attn.backend == "fa2"


def test_malformed_mirrors_fail_the_plan(planners):
    attn = planners.controller()
    md = _host_metadata([256], [2048])
    md.kv_seq_lens_cpu[0] = 4096  # above the declared max_kv_len
    with pytest.raises(ValueError, match="auto policy: KV lengths"):
        attn.plan(md, **_HOST_KW)
    assert planners.preflights == [] and attn.backend is None


def test_the_hint_no_longer_orders_the_plan(planners):
    """A hinted Resolution lists fa2 first, an unhinted one trtllm-gen; both
    plan the same batch on the backend the batch calls for, and the hint
    still bounds the batches."""
    cfg = dict(_HOST_KW, cc_major=10, page_size=16)
    hinted = _resolve(max_q_len=16, **cfg)
    plain = _resolve(**cfg)
    wide = _resolve(max_q_len=512, **cfg)
    assert hinted.chosen == "fa2" and plain.chosen == wide.chosen == "trtllm-gen"
    for resolution, batch, order in (
        (plain, _FA2_BATCH, _FA2_ORDER),
        (hinted, _FA2_BATCH, _FA2_ORDER),
        (plain, _TRTLLM_BATCH, _TRTLLM_ORDER),
        (wide, _TRTLLM_BATCH, _TRTLLM_ORDER),
        (wide, _CUDNN_BATCH, _CUDNN_ORDER),
    ):
        attn = planners.controller()
        attn.plan(_host_metadata(*batch), backend=resolution, **_HOST_KW)
        assert list(attn.selection_order) == order
        assert attn.backend == order[0]
    with pytest.raises(ValueError, match="exceeds the pinned Resolution's hint"):
        planners.controller().plan(
            _host_metadata(*_TRTLLM_BATCH), backend=hinted, **_HOST_KW
        )
    assert planners.preflights.count("trtllm-gen") == 2


def test_a_narrowed_resolution_is_ranked_within_its_set(planners):
    cfg = dict(_HOST_KW, cc_major=10, page_size=16)
    narrowed = dataclasses.replace(_resolve(**cfg), backends=("fa2", "cudnn"))
    attn = planners.controller()
    attn.plan(_host_metadata(*_TRTLLM_BATCH), backend=narrowed, **_HOST_KW)
    assert attn.selection_order == ("fa2", "cudnn")
    attn.plan(_host_metadata(*_CUDNN_BATCH), backend=narrowed, **_HOST_KW)
    assert attn.selection_order == ("cudnn", "fa2")


@pytest.mark.parametrize(
    "cc,q_lens,order",
    [
        ((9, 0), [128], ["fa3", "fa2", "cudnn"]),
        ((9, 0), [64], ["fa2", "fa3", "cudnn"]),
        ((8, 0), [128], ["fa2", "cudnn"]),
        ((12, 0), [128], ["fa2", "cudnn"]),
        ((10, 3), [128], ["trtllm-gen", "cudnn", "fa2"]),
        ((10, 3), [16], ["fa2", "trtllm-gen", "cudnn"]),
    ],
)
def test_non_sm100_keeps_its_order(monkeypatch, cc, q_lens, order):
    """The SM100 cuDNN-region batch on other architectures: the walk follows
    their previous order to the last candidate."""
    planners = _cpu_planners(monkeypatch, cc)
    planners.decline(*order[:-1])
    attn = planners.controller()
    attn.plan(_host_metadata(q_lens, [4096]), **_HOST_KW)
    assert planners.preflights == order
    assert attn.backend == order[-1]
    assert list(attn.selection_order) == order


# --------------------------------------------------------------------------
# 3. execution (GPU)
# --------------------------------------------------------------------------

_HQ, _HK, _HD, _PAGE = 16, 4, 128, 16
# name -> (capture batch, update batch, table width in pages).  The update
# changes the lengths, the query partition and (by seed) every page id while
# staying inside the first batch's capacity.  Each first batch is in its
# region both eager and in graph mode.
_REGIONS = {
    # prefix region, ragged: 16 * (128 * 4096 + 100 * 3000) = 13 Mi of work
    "cudnn": (([128, 100], [4096, 3000]), ([90, 120], [2000, 4000]), 256),
    # 112 rows, fill 0.875
    "trtllm-gen": (([512, 384], [600, 1024]), ([300, 500], [1024, 700]), 64),
    "fa2": (
        ([16, 7, 1, 30], [300, 97, 1000, 40]),
        ([10, 9, 2, 20], [100, 500, 900, 64]),
        63,
    ),
}


def _is_sm100():
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor) == (10, 0)


def _expected_backend(region):
    """The region's backend where the measured policy applies, else ``None``
    (other architectures only get the numerical check)."""
    if not _is_sm100():
        return None
    if region == "cudnn" and not _auto_policy._cudnn_supports_paged_auto():
        return None
    return region


def _problem(q_lens, kv_lens, width, seed):
    """A paged batch with scattered page ids over a pool sized for any batch
    of the same shape class, in the prototype suite's problem-dict format."""
    dev = "cuda:0"
    g = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=dev).manual_seed(seed)
    batch = len(q_lens)
    pool = batch * width + 8
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    block_tables = torch.zeros(batch, width, dtype=torch.int32)
    offset = 0
    for i, kv_len in enumerate(kv_lens):
        pages = (kv_len + _PAGE - 1) // _PAGE
        block_tables[i, :pages] = perm[offset : offset + pages]
        offset += pages
    qo_indptr = _indptr(q_lens)
    kv = torch.tensor(kv_lens, dtype=torch.int32)

    def randn(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16, device=dev, generator=g_dev)

    k_cache = randn(pool, _HK, _PAGE, _HD)
    v_cache = randn(pool, _HK, _PAGE, _HD)
    return dict(
        q=randn(sum(q_lens), _HQ, _HD),
        k_cache=k_cache,
        v_cache=v_cache,
        k_ref=k_cache,
        v_ref=v_cache,
        qo_indptr=qo_indptr.to(dev),
        qo_indptr_cpu=qo_indptr,
        kv_seq_lens=kv.to(dev),
        kv_seq_lens_cpu=kv,
        block_tables=block_tables.to(dev),
        page_size=_PAGE,
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        num_qo_heads=_HQ,
        num_kv_heads=_HK,
        head_dim_qk=_HD,
        head_dim_vo=_HD,
        dtype=torch.bfloat16,
        device=dev,
    )


def _region_problems(region):
    first, second, width = _REGIONS[region]
    return _problem(*first, width, seed=11), _problem(*second, width, seed=12)


def _plan_auto(attn, p, **kwargs):
    attn.plan(
        make_metadata(p, **kwargs),
        num_qo_heads=_HQ,
        num_kv_heads=_HK,
        head_dim_qk=_HD,
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
        backend="auto",
    )


def _assert_matches_oracle(p, out, lse, *, sm_scale=None, v_scale=1.0):
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        _PAGE,
        True,
        sm_scale=sm_scale,
    )
    rows = p["q"].shape[0]
    torch.testing.assert_close(out[:rows].float(), ref_out * v_scale, **OUT_TOL)
    torch.testing.assert_close(lse[:rows], ref_lse, **LSE_TOL)


def _decline(monkeypatch, name):
    """Make backend ``name`` decline every batch from its real preflight."""
    cls = _controller.make_backend.__globals__["_CLASSES"][name]
    real = cls.preflight

    def preflight(self, meta):
        if self.name == name:
            raise _BackendPlanUnsupportedError(f"{name} declines")
        return real(self, meta)

    monkeypatch.setattr(cls, "preflight", preflight)


def _capture(attn, q, k_cache, v_cache, out, lse):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # module load / cuDNN plan build stay out of the capture
            attn.run(q, (k_cache, v_cache), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn.run(q, (k_cache, v_cache), out=out, lse=lse)
    return graph


def _graph_buffers(p):
    q, k_cache, v_cache = p["q"].clone(), p["k_cache"].clone(), p["v_cache"].clone()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], _HQ, dtype=torch.float32, device=q.device)
    return q, k_cache, v_cache, out, lse


def _load(p, q, k_cache, v_cache):
    q[: p["q"].shape[0]].copy_(p["q"])
    k_cache.copy_(p["k_cache"])
    v_cache.copy_(p["v_cache"])


@needs_cuda
@pytest.mark.parametrize("region", list(_REGIONS))
def test_auto_eager_matches_the_oracle(region):
    p1, p2 = _region_problems(region)
    attn = PagedAttentionFacade()
    _plan_auto(attn, p1)
    expected = _expected_backend(region)
    if expected is not None:
        assert attn.backend == expected
    out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))
    _assert_matches_oracle(p1, out, lse)
    # eager re-plan: new lengths, query partition and page ids
    _plan_auto(attn, p2)
    out, lse = attn.run(p2["q"], (p2["k_cache"], p2["v_cache"]))
    _assert_matches_oracle(p2, out, lse)


@needs_cuda
def test_auto_eager_reselects_across_regions():
    """One eager instance over a batch of each region and back."""
    attn = PagedAttentionFacade()
    for region in ("fa2", "cudnn", "trtllm-gen", "fa2", "trtllm-gen", "cudnn"):
        p, _ = _region_problems(region)
        _plan_auto(attn, p)
        expected = _expected_backend(region)
        if expected is not None:
            assert attn.backend == expected
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        _assert_matches_oracle(p, out, lse)


@needs_cuda
@pytest.mark.parametrize("region", list(_REGIONS))
def test_auto_capture_update_replay(region):
    p1, p2 = _region_problems(region)
    attn = PagedAttentionFacade(use_cuda_graph=True)
    q, k_cache, v_cache, out, lse = _graph_buffers(p1)
    _plan_auto(attn, p1)
    expected = _expected_backend(region)
    if expected is not None:
        assert attn.backend == expected
    frozen = attn.backend
    graph = _capture(attn, q, k_cache, v_cache, out, lse)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(p1, out, lse)

    _load(p2, q, k_cache, v_cache)
    attn.update(make_metadata(p2))
    assert attn.backend == frozen
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(p2, out, lse)

    _load(p1, q, k_cache, v_cache)
    attn.update(make_metadata(p1))
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(p1, out, lse)


@needs_cuda
@pytest.mark.parametrize("region", ["cudnn", "trtllm-gen"])
def test_graph_typed_fallback_freezes_and_replays(monkeypatch, region):
    """The first choice declines the capture batch; the capture then keeps
    the fallback through updates that the first choice would accept."""
    p1, p2 = _region_problems(region)
    attn = PagedAttentionFacade(use_cuda_graph=True)
    _plan_auto(attn, p1)
    first = attn.backend
    order = attn._impl.selection_order
    if len(order) < 2:
        pytest.skip("needs two runnable candidates")

    attn = PagedAttentionFacade(use_cuda_graph=True)
    q, k_cache, v_cache, out, lse = _graph_buffers(p1)
    with monkeypatch.context() as patch:
        _decline(patch, first)
        _plan_auto(attn, p1)
    fallback = attn.backend
    assert fallback == order[1]
    assert attn._impl.selection_trace[0] == (first, "preflight", f"{first} declines")
    graph = _capture(attn, q, k_cache, v_cache, out, lse)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(p1, out, lse)

    _load(p2, q, k_cache, v_cache)
    attn.update(make_metadata(p2))
    assert attn.backend == fallback
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_oracle(p2, out, lse)


@needs_cuda
@pytest.mark.parametrize("region", list(_REGIONS))
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_auto_runtime_scales(region, graph):
    """``sm_scale`` / ``k_scale`` / ``v_scale`` arrive only at run(): whatever
    backend auto planned must apply them."""
    p, _ = _region_problems(region)
    sm_scale, k_scale, v_scale = 0.05, 1.5, 0.5
    attn = PagedAttentionFacade(use_cuda_graph=graph)
    _plan_auto(attn, p)
    out, lse = attn.run(
        p["q"],
        (p["k_cache"], p["v_cache"]),
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    _assert_matches_oracle(p, out, lse, sm_scale=sm_scale * k_scale, v_scale=v_scale)
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    _assert_matches_oracle(p, out, lse)


@needs_cuda
@pytest.mark.parametrize("region", list(_REGIONS))
def test_auto_plan_does_not_read_the_device_back(region):
    """Lengths given on the device only: the one documented readback is at
    metadata construction; the plan — policy included — stays sync-free and
    picks the same backend as with caller-owned mirrors."""
    p1, p2 = _region_problems(region)
    attn = PagedAttentionFacade()
    _plan_auto(attn, p1)  # warm: module load, cuDNN plan build
    with_mirrors = attn.backend
    attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))
    md1 = make_metadata(p1, with_mirrors=False)
    md2 = make_metadata(p2, with_mirrors=False)
    kwargs = dict(
        num_qo_heads=_HQ,
        num_kv_heads=_HK,
        head_dim_qk=_HD,
        q_dtype=torch.bfloat16,
        lse_mode="base2",
    )
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attn.plan(md2, **kwargs)
        attn.plan(md1, **kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert attn.backend == with_mirrors
    out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))
    _assert_matches_oracle(p1, out, lse)


def PagedAttentionFacade(**kwargs):
    from flashinfer.prefill import PagedAttention

    return PagedAttention(torch.device("cuda:0"), **kwargs)
