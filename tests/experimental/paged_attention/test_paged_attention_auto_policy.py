"""``backend="auto"`` of PagedAttention (experimental): the plan-time policy.

Three layers, mirroring ``tests/attention/test_mla_auto_backend.py``:

1. Policy (no GPU): ``_auto_policy`` orders every candidate from host facts;
   the SM100 table sits on both sides of each threshold and the copied cuDNN
   region is pinned against the legacy wrapper's predicate.  The parts of the
   legacy admission that are not copied (head dim 256 on SM100 and on SM90)
   are pinned as known differences.
2. Planning (no GPU, fake backends): the controller hands the policy the
   plan's facts, walks the order it returns, moves on only after a typed
   rejection, and freezes the choice in graph mode.
3. Execution (GPU): auto against the fp32 oracle, eager and under capture /
   re-plan / replay.  No test here asserts which backend the rule picks: the
   lifetime tests put a named candidate first (``_force_first``) and check
   that it ran, and the two tests that keep the real rule check numbers only.

Only the layer-1 tables (``_SM100_TABLE``, ``test_sm100_fallback_order``)
depend on the fitted trtllm-gen / fa2 thresholds of SM100.  Layer 2 ranks
that pair with a stand-in (``_layer2_sm100_rule``) and layer 3 injects the
order, so a re-fit touches the tables alone.
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
# unless overridden.  Shapes marked [m] are measured cases of the GB200 fit;
# the others step one unit across a threshold from one.  The rows of the
# cuDNN region's extension carry no mark: that part is copied from the legacy
# predicate, not fitted.  A pair is the first two preferences: trtllm-gen
# ahead of cuDNN is the eager small-kernel exception inside the region, so
# such a row still tells which side of the region's bound its batch is on.
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
    # ---- eager, fewer than 1024 keys per query row: fa2 between 148 rows and
    # rows * keys = 2^19 (graph mode has no such rule)
    ("short-kv-rows256 [m]", [256] * 4, [256] * 4, {}, "fa2"),
    (
        "short-kv-rows256-graph [m]",
        [256] * 4,
        [256] * 4,
        dict(graph=True),
        "trtllm-gen",
    ),
    # the lengths decide, not a bound declared above them; the bound stands
    # in only when the lengths are not on the host
    ("short-kv-loose-bound", [128] * 8, [256] * 8, dict(max_kv_len=8192), "fa2"),
    (
        "short-kv-loose-bound-device-lengths",
        [128] * 8,
        [256] * 8,
        dict(max_kv_len=8192, host=False),
        "trtllm-gen",
    ),
    ("short-kv-device-lengths", [128] * 8, [256] * 8, dict(host=False), "fa2"),
    ("short-kv-1023", [128] * 8, [1023] * 8, {}, "fa2"),
    ("short-kv-1024", [128] * 8, [1024] * 8, {}, "trtllm-gen"),
    # a sliding window bounds the keys of a row whatever the KV length
    ("short-kv-window127 [m]", [128] * 8, [8192] * 8, dict(window_left=127), "fa2"),
    (
        "short-kv-window127-graph [m]",
        [128] * 8,
        [8192] * 8,
        dict(window_left=127, graph=True),
        "trtllm-gen",
    ),
    ("short-kv-window1022", [128] * 8, [8192] * 8, dict(window_left=1022), "fa2"),
    # up to 148 rows fa2 splits the KV range and trtllm-gen keeps the batch
    ("short-kv-rows148 [m]", [592], [8192], dict(window_left=127), "trtllm-gen"),
    ("short-kv-rows149", [596], [8192], dict(window_left=127), "fa2"),
    ("short-kv-rows160 [m]", [128] * 5, [8192] * 5, dict(window_left=127), "fa2"),
    # rows * keys per row from 2^19 up
    ("short-kv-work-2^19 [m]", [128] * 32, [512] * 32, {}, "trtllm-gen"),
    (
        "short-kv-work-below-2^19",
        [128] * 32,
        [8192] * 32,
        dict(window_left=510),
        "fa2",
    ),
    ("short-kv-rows1536-kv256 [m]", [128] * 48, [256] * 48, {}, "fa2"),
    ("short-kv-rows2048-kv256 [m]", [256] * 32, [256] * 32, {}, "trtllm-gen"),
    # ---- cuDNN region, the bounded domain (at most four requests from 8 Mi
    # of work), prefix: 64 <= max_q <= 1024, 2048 <= max_kv <= 32768,
    # max_kv >= 4 * max_q
    ("prefix-q128-kv16384 [m]", [128], [16384], {}, "cudnn"),
    ("prefix-q64 [m]", [64], [16384], {}, "cudnn"),
    ("prefix-q63", [63], [16384], {}, "fa2"),
    ("prefix-q1024 [m]", [1024], [32768], dict(hq=8, hk=1), "cudnn"),
    # one more query token leaves the prefix test, and at 256 Mi the
    # extension admits the batch from its lengths: with those on the host only
    ("prefix-q1025", [1025], [32768], dict(hq=8, hk=1), "cudnn"),
    (
        "prefix-q1025-device-lengths",
        [1025],
        [32768],
        dict(hq=8, hk=1, host=False),
        "trtllm-gen",
    ),
    ("prefix-kv2048 [m]", [128], [2048], {}, ("trtllm-gen", "cudnn")),
    ("prefix-kv2047 [m]", [128], [2047], {}, "fa2"),
    ("prefix-kv32768 [m]", [128] * 3, [32768] * 3, dict(hq=16, hk=4), "cudnn"),
    ("prefix-kv32769", [128] * 3, [32769] * 3, dict(hq=16, hk=4), "fa2"),
    # max_kv >= 4 * max_q decides below 16 Mi (8 heads: 8.0 Mi); the
    # extension has no such ratio (32 heads: 32 Mi)
    ("prefix-kv=4q-8Mi", [512], [2048], dict(hq=8, hk=2), ("trtllm-gen", "cudnn")),
    ("prefix-kv<4q-8Mi", [513], [2048], dict(hq=8, hk=2), "fa2"),
    ("prefix-kv=4q-1", [513], [2051], dict(hq=8, hk=2), "fa2"),
    ("prefix-kv=4q", [512], [2048], {}, ("trtllm-gen", "cudnn")),
    ("prefix-kv<4q-32Mi", [513], [2048], {}, ("trtllm-gen", "cudnn")),
    # exact work = heads * sum(q_i * kv_i) >= 8 Mi
    ("work-8Mi", [256], [2048], dict(hq=16, hk=4), ("trtllm-gen", "cudnn")),
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
    # more than four requests of at most 128 query tokens are outside (above
    # 128 the extension below admits them)
    ("batch4 [m]", [64] * 4, [8192] * 4, {}, "cudnn"),
    ("batch5", [64] * 5, [8192] * 5, {}, "fa2"),
    ("batch6-rows96 [m]", [128] * 6, [8192] * 6, dict(hq=16, hk=4), "trtllm-gen"),
    ("heads8", [1024], [4096], dict(hq=8, hk=2), ("trtllm-gen", "cudnn")),
    # one shape on both sides of ``8 <= num_qo_heads`` (4/1 is the measured
    # graph case B1 q2048 kv2048)
    ("heads8-full", [2048], [2048], dict(hq=8, hk=2, graph=True), "cudnn"),
    ("heads4-full [m]", [2048], [2048], dict(hq=4, hk=1, graph=True), "trtllm-gen"),
    ("heads64 [m]", [128], [8192], dict(hq=64, hk=8), "cudnn"),
    ("heads128", [128], [8192], dict(hq=128, hk=8), "trtllm-gen"),
    ("group16 [m]", [128], [8192], dict(hq=32, hk=2), "cudnn"),
    ("group32", [128], [8192], dict(hq=32, hk=1), "fa2"),
    ("group3", [256], [4096], dict(hq=24, hk=8), "fa2"),
    # ---- cuDNN region, the bounded domain, full prefill:
    # 1024 <= max_q == max_kv <= 8192 with every q_i == kv_i
    ("full-1024 [m]", [1024], [1024], dict(hq=8, hk=1), ("trtllm-gen", "cudnn")),
    ("full-1023", [1023], [1023], dict(hq=8, hk=1), "fa2"),
    ("full-8192 [m]", [8192], [8192], dict(hq=64, hk=8), "cudnn"),
    ("full-ragged [m]", [4921], [4921], dict(hq=16, hk=4), "cudnn"),
    ("full-each-q=kv", [2048, 1024], [2048, 1024], {}, "cudnn"),
    # the q_i == kv_i proof decides below 16 Mi (8 heads: 9.5 and 10.9 Mi);
    # from 16 Mi a batch that fails it can enter the extension as a mixed
    # batch (32 heads: 192 Mi)
    (
        "full-each-q=kv-10Mi",
        [1100, 200],
        [1100, 200],
        dict(hq=8, hk=1),
        ("trtllm-gen", "cudnn"),
    ),
    ("full-mixed-prefix-11Mi", [1100, 200], [1100, 1100], dict(hq=8, hk=1), "fa2"),
    ("full-mixed-prefix", [2048, 1024], [2048, 2048], {}, "cudnn"),
    # equal maxima alone do not prove full prefill
    ("full-device-lengths", [2048], [2048], dict(host=False), "trtllm-gen"),
    # ---- cuDNN region, the extension: any number of requests from 16 Mi of
    # work, on the batch's own lengths.  Every q_i == kv_i: the longest query
    # in 1024..16384, the other requests of any length
    ("ext-full-8193", [8193], [8193], dict(hq=64, hk=8), "cudnn"),
    ("ext-full-16384", [16384], [16384], dict(hq=64, hk=8), "cudnn"),
    ("ext-full-16385", [16385], [16385], dict(hq=64, hk=8), "trtllm-gen"),
    ("ext-full-batch16", [1024] * 16, [1024] * 16, {}, ("trtllm-gen", "cudnn")),
    ("ext-full-batch16-q1023", [1023] * 16, [1023] * 16, {}, "trtllm-gen"),
    ("ext-full-one-short", [1] + [4096] * 4, [1] + [4096] * 4, {}, "cudnn"),
    # (the smallest query total of such a batch, 1023 + batch: 16.0 Mi at 16
    # heads)
    (
        "ext-full-total-1023+batch",
        [1024] + [1] * 8,
        [1024] + [1] * 8,
        dict(hq=16, hk=4),
        ("trtllm-gen", "cudnn"),
    ),
    # ... otherwise every q_i > 128, the longest query at most 8192 and the
    # longest KV in 2048..32768
    ("ext-batch5-q129", [129] * 5, [8192] * 5, {}, "cudnn"),
    ("ext-batch5-q128", [128] * 5, [8192] * 5, {}, "trtllm-gen"),
    ("ext-batch32", [256] * 32, [4096] * 32, {}, "cudnn"),
    ("ext-batch256", [256] * 256, [4096] * 256, {}, "cudnn"),
    (
        "ext-ragged",
        [300, 200, 500, 150, 1000, 129],
        [4096, 2048, 8192, 3000, 16384, 2048],
        {},
        "cudnn",
    ),
    # one request of 128 tokens keeps the batch out, whatever the others carry
    ("ext-all-q129", [129] * 5, [2048] * 5, dict(hq=16, hk=1), ("trtllm-gen", "cudnn")),
    ("ext-one-q128", [128] + [129] * 4, [2048] * 5, dict(hq=16, hk=1), "fa2"),
    ("ext-q8192", [8192], [16384], {}, "cudnn"),
    ("ext-q8193", [8193], [16384], {}, "trtllm-gen"),
    ("ext-kv2048", [256] * 5, [2048] * 5, {}, ("trtllm-gen", "cudnn")),
    ("ext-kv2047", [256] * 5, [2047] * 5, {}, "trtllm-gen"),
    ("ext-kv32768", [256] * 5, [32768] * 5, {}, "cudnn"),
    ("ext-kv32769", [256] * 5, [32769] * 5, {}, "trtllm-gen"),
    # exact work >= 16 Mi: 8 heads * 6 * 129 * 2710 = 16.003 Mi, 15.997 Mi at 2709
    ("ext-work-16Mi", [129] * 6, [2710] * 6, dict(hq=8, hk=2), ("trtllm-gen", "cudnn")),
    ("ext-work-below-16Mi", [129] * 6, [2709] * 6, dict(hq=8, hk=2), "fa2"),
    # the lengths decide, under any bounds declared above them ...
    (
        "ext-loose-bounds",
        [256] * 6,
        [4096] * 6,
        dict(max_q_len=2048, max_kv_len=131072),
        ("trtllm-gen", "cudnn"),
    ),
    # ... and only when they are on the host
    ("ext-device-lengths", [256] * 6, [4096] * 6, dict(host=False), "trtllm-gen"),
    # ---- inside the cuDNN region, eager: trtllm-gen ahead of cuDNN for a
    # small kernel.  At most 4096 keys per query row ...
    ("small-kv4096 [m]", [256] * 8, [4096] * 8, {}, ("trtllm-gen", "cudnn")),
    ("small-kv4097", [256] * 8, [4097] * 8, {}, "cudnn"),
    # ... at most 256 rows in the longest request (32 heads: 1024 tokens) ...
    ("small-request-rows256 [m]", [1024], [2048], {}, ("trtllm-gen", "cudnn")),
    ("small-request-rows257", [1025], [2048], {}, "cudnn"),
    ("small-request-rows512 [m]", [1024], [2048], dict(hq=64, hk=8), "cudnn"),
    # ... and rows * keys of at most 2^22 (16 requests: 4096 rows of 1024 keys)
    ("small-work-2^22 [m]", [1024] * 16, [1024] * 16, {}, ("trtllm-gen", "cudnn")),
    ("small-work-above-2^22", [1024] * 17, [1024] * 17, {}, "cudnn"),
    ("small-work-2^23 [m]", [256] * 32, [4096] * 32, {}, "cudnn"),
    # graph plans keep cuDNN first (GB200 graph: B2 32/8 q128 kv2048: cudnn
    # 31-32, trtllm-gen 33-34)
    ("small-graph [m]", [128] * 2, [2048] * 2, dict(graph=True), "cudnn"),
    # the batch's own lengths decide under looser bounds; device-only lengths
    # leave the bounds
    (
        "small-loose-bounds",
        [256] * 8,
        [4096] * 8,
        dict(max_q_len=2048, max_kv_len=8192),
        ("trtllm-gen", "cudnn"),
    ),
    (
        "small-device-lengths",
        [128] * 2,
        [2048] * 2,
        dict(host=False),
        ("trtllm-gen", "cudnn"),
    ),
    (
        "small-device-lengths-kv-bound",
        [128] * 2,
        [2048] * 2,
        dict(host=False, max_kv_len=8192),
        "cudnn",
    ),
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
    # ---- graph mode: the bounds are the capacity's, the per-request lengths
    # (exact work, q == kv, every test of the extension) the first batch's
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
    # the same capacity, first batches on both sides of 8 Mi of work
    # (32 * 16 * 16384 = 8 Mi)
    (
        "graph-capacity-region-first-batch-8Mi",
        [16],
        [16384],
        dict(graph=True, total_q_tokens=128, max_q_len=128, max_kv_len=16384),
        "cudnn",
    ),
    (
        "graph-capacity-region-first-batch-below-8Mi",
        [16],
        [16383],
        dict(graph=True, total_q_tokens=128, max_q_len=128, max_kv_len=16384),
        "fa2",
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
    # that batch carries 739 Mi: its lengths admit it under any capacity
    (
        "graph-full-any-capacity",
        [4921],
        [4921],
        dict(graph=True, max_kv_len=131072),
        "cudnn",
    ),
    # (a query above its KV: no part of the region admits the batch)
    (
        "graph-wider-capacity-not-full",
        [4921],
        [4920],
        dict(graph=True, max_kv_len=8192),
        "trtllm-gen",
    ),
    # below 16 Mi (8 heads, q = kv = 1100: 9.2 Mi) the capacity decides:
    # 1024 <= max_q <= max_kv <= 8192, with every q_i == kv_i on the host
    (
        "graph-full-wider-capacity-9Mi",
        [1100],
        [1100],
        dict(hq=8, hk=2, graph=True, max_kv_len=8192),
        "cudnn",
    ),
    (
        "graph-full-capacity-above-limit-9Mi",
        [1100],
        [1100],
        dict(hq=8, hk=2, graph=True, max_kv_len=8193),
        "trtllm-gen",
    ),
    (
        "graph-wider-capacity-not-full-9Mi",
        [1100],
        [1101],
        dict(hq=8, hk=2, graph=True, max_kv_len=8192),
        "trtllm-gen",
    ),
    # the lower limit of that capacity test (8 heads, q = kv = 1024: 8.0 Mi)
    (
        "graph-full-wider-capacity-q1024",
        [1024],
        [1024],
        dict(hq=8, hk=1, graph=True, max_kv_len=2048),
        "cudnn",
    ),
    # a prefix batch whose graph bounds also fit the widened full-prefill test
    # (GB200 graph: B2 64/8 q1024 kv4096: cudnn 196, trtllm-gen 209)
    (
        "graph-prefix-inside-full-bounds [m]",
        [1024] * 2,
        [4096] * 2,
        dict(hq=64, graph=True),
        "cudnn",
    ),
    # eager, the same two full prefills under a KV bound above the query: the
    # bound test needs max_q == max_kv, so below 16 Mi the batch is outside;
    # from 16 Mi its lengths admit it
    (
        "eager-bounds-must-match-9Mi",
        [1100],
        [1100],
        dict(hq=8, hk=2, max_kv_len=8192),
        "trtllm-gen",
    ),
    ("eager-full-loose-bound", [4921], [4921], dict(max_kv_len=8192), "cudnn"),
    # the extension under a capacity: the first batch's lengths alone, with
    # the capacity's bounds and total far above them
    (
        "graph-ext-loose-capacity",
        [256] * 6,
        [4096] * 6,
        dict(graph=True, total_q_tokens=12288, max_q_len=2048, max_kv_len=131072),
        "cudnn",
    ),
    (
        "graph-ext-loose-capacity-one-q128",
        [256] * 5 + [128],
        [4096] * 6,
        dict(graph=True, total_q_tokens=12288, max_q_len=2048, max_kv_len=131072),
        "trtllm-gen",
    ),
    # a padding request (q = kv = 0) keeps a first batch out of the extension
    # and of the full-prefill proof
    ("graph-full-batch2", [4096] * 2, [4096] * 2, dict(graph=True), "cudnn"),
    (
        "graph-full-batch2-padded",
        [4096, 4096, 0],
        [4096, 4096, 0],
        dict(graph=True),
        "trtllm-gen",
    ),
]


@pytest.mark.parametrize(
    "q_lens,kv_lens,over,first",
    [row[1:] for row in _SM100_TABLE],
    ids=[row[0] for row in _SM100_TABLE],
)
def test_sm100_first_preference(cudnn_qualified, q_lens, kv_lens, over, first):
    facts = _facts(q_lens, kv_lens, **over)
    order = ordered_sm100_backends(facts)
    if isinstance(first, tuple):
        assert order[: len(first)] == first
    else:
        assert order[0] == first
    # cuDNN is in front of fa2 inside its region only
    assert _in_sm100_cudnn_region(facts) == (order.index("cudnn") < order.index("fa2"))
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
    # the eager small-kernel exception puts trtllm-gen in front and keeps
    # cuDNN ahead of fa2
    region_small_kernel = _facts([256] * 2, [2048] * 2, hq=64, hk=8)
    assert ordered_sm100_backends(region_small_kernel)[:3] == (
        "trtllm-gen",
        "cudnn",
        "fa2",
    )
    for facts in (_facts([252], [131072]), _facts([256], [131072])):
        assert ordered_sm100_backends(facts)[-1] == "cudnn"


def test_sm100_region_needs_the_cudnn_version_gate(monkeypatch):
    facts = _facts([128], [16384])
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: False)
    assert ordered_sm100_backends(facts)[-1] == "cudnn"
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    assert ordered_sm100_backends(facts)[0] == "cudnn"


class _DeviceOnly:
    """Stands in for a CUDA tensor, so the device-only paths run anywhere:
    any use of it other than asking for its device fails the test."""

    device = torch.device("cuda", 0)

    def __getattr__(self, name):
        raise AssertionError(f"device tensor touched: .{name}")


class _HostUnread:
    """Stands in for a host mirror the policy must not read: any use of it
    other than asking for its device fails the test."""

    device = torch.device("cpu")

    def __getattr__(self, name):
        raise AssertionError(f"host lengths read: .{name}")


def test_sm100_cudnn_region_against_the_legacy_predicate(monkeypatch):
    """The policy duplicates the head-dim-128 part of
    ``_blackwell_paged_auto_cudnn``; this pins the copy to the original over
    batches, heads, the static conditions, bounds declared above the lengths
    and every length path (host, absent, device-only, one mirror on a
    device).

    Eager facts: the two are equal.  CUDA-graph facts carry a capacity; the
    original is handed the capacity as its declared maxima and, with host
    lengths, the batch's own query total (the legacy wrapper has no capacity
    and passes that).  The copy then admits everything the original admits,
    and more only where its full-prefill bound test is wider
    (``max_q <= max_kv`` for a capacity instead of ``max_q == max_kv``): at
    most four requests with every q equal to its KV on the host and
    8 Mi <= work < 16 Mi.  From 16 Mi the original admits such a batch from
    its lengths under any bounds."""
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(legacy_prefill, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(legacy_prefill, "get_compute_capability", lambda dev: (10, 0))
    mi = 1 << 20
    batches = [
        # ---- the bounded domain: at most four requests inside a bound test
        ([128], [2048]),
        ([64], [2048]),
        ([63], [2048]),
        ([128], [2047]),
        ([1024], [4096]),
        ([1025], [8192]),
        ([512], [2048]),
        ([513], [2048]),
        ([128], [32768]),
        ([128], [32769]),
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
        ([128, 257], [32768, 16385]),
        ([256] * 4, [8192] * 4),
        ([256] * 5, [8192] * 5),
        # full prefill whose length is not a power of two, as a bucket with a
        # looser KV bound sees it
        ([4921], [4921]),
        ([3000, 1500], [3000, 1500]),
        ([4921], [4920]),
        # ... with 8 Mi <= work < 16 Mi at 8 heads (9.2 Mi at 1100 tokens,
        # 15.997 Mi at 1448, 16.02 Mi at 1449): the band in which only the
        # wider graph-mode bound test admits it, for at most four requests
        ([1100], [1100]),
        ([1100], [1101]),
        ([1100, 200], [1100, 200]),
        ([1100, 200], [1100, 1100]),
        ([1448], [1448]),
        ([1449], [1449]),
        ([1100, 8, 8, 8], [1100, 8, 8, 8]),
        ([1100, 8, 8, 8, 8], [1100, 8, 8, 8, 8]),
        # ---- the extension: more than four requests, every q above 128
        ([64] * 5, [8192] * 5),
        ([128] * 5, [8192] * 5),
        ([129] * 5, [8192] * 5),
        ([128] * 6, [6145] * 6),
        ([129] * 6, [6145] * 6),
        ([256] * 6, [4096] * 6),
        ([256] * 16, [4096] * 16),
        ([256] * 32, [4096] * 32),
        # a short request next to long ones
        ([128, 129, 129, 129, 129], [2048] * 5),
        ([129] * 5, [2048] * 5),
        ([17, 17, 17, 17, 3072], [32768, 32768, 32768, 32768, 3104]),
        ([1, 4096, 4096, 4096, 4096], [1, 4096, 4096, 4096, 4096]),
        # one 1024-token request next to one-token ones: the smallest query
        # total of an admitted batch of nine, 1023 + batch
        ([1024] + [1] * 8, [1024] + [1] * 8),
        ([1023] + [1] * 8, [1023] + [1] * 8),
        # 16 Mi of work at 8 heads: 8 * 6 * 129 * 2710 = 16.003 Mi, 15.997 Mi
        # at 2709; the two-request batch carries 10.0 Mi
        ([129] * 6, [2709] * 6),
        ([129] * 6, [2710] * 6),
        ([1024, 129], [1024, 2048]),
        # the longest query of a batch that is not all full prefill
        ([8192], [16384]),
        ([8193], [16384]),
        ([8192, 300, 300, 300, 300], [16384, 4096, 4096, 4096, 4096]),
        ([8193, 300, 300, 300, 300], [16384, 4096, 4096, 4096, 4096]),
        # ... and its longest KV
        ([256] * 5, [2047] * 5),
        ([256] * 5, [2048] * 5),
        ([256] * 5, [32768] * 5),
        ([256] * 5, [32769] * 5),
        ([256, 512, 1024, 2048, 3072], [4096, 8192, 6144, 12288, 8192]),
        # every q equal to its KV: the longest query in 1024..16384
        ([1024] * 16, [1024] * 16),
        ([1023] * 16, [1023] * 16),
        ([1000] * 5, [1000] * 5),
        ([513] * 6, [513] * 6),
        ([16384], [16384]),
        ([16385], [16385]),
        ([16384] * 5, [16384] * 5),
        # a query above its KV, a request without tokens
        ([300, 300, 300, 300, 3000], [4096, 4096, 4096, 4096, 2000]),
        ([512, 0, 512, 512, 512], [4096] * 5),
        ([2048, 2048, 2048, 2048, 0], [2048, 2048, 2048, 2048, 0]),
        # (equal sums of Q and KV are not full prefill)
        ([2048, 1024], [1024, 2048]),
    ]
    heads = [(4, 4), (8, 8), (8, 1), (12, 4), (16, 4), (32, 8), (32, 2), (32, 1)]
    heads += [(64, 8), (128, 8), (12, 8)]
    heads += [(7, 7), (65, 65)]  # one query head outside 8..64 on either side
    bf16, fp16, fp8 = torch.bfloat16, torch.float16, torch.float8_e4m3fn
    qualified = ((128, 128), 16, "HND", (bf16, bf16))
    statics = list(
        itertools.product(
            ((128, 128), (64, 64)),
            (16, 32),
            ("HND", "NHD"),
            ((bf16, bf16), (fp16, fp16), (bf16, fp8)),
        )
    )
    # the QK and VO head dims differ, or the Q and KV dtypes, with one of each
    # at the qualified value (no capability row declares these pairs)
    unpaired = [
        ((128, 64), 16, "HND", (bf16, bf16)),
        ((128, 128), 16, "HND", (fp16, bf16)),
    ]
    # (static conditions, length paths, declared bounds, CUDA graph, a total
    # above the batch's)
    grids = [
        # every static condition on and off, lengths on the host or absent,
        # the batch's own bounds or a looser KV bound
        (statics, ("host", None), ("exact", "kv"), (False, True), (False,)),
        (unpaired, ("host", None), ("exact",), (False, True), (False,)),
        # the qualified configuration: device-only lengths, a looser query
        # bound, and a declared total above the batch's as a capacity has it
        (
            [qualified],
            ("host", None, "device", "kv-device", "qo-device"),
            ("exact", "kv", "q", "q+kv", "q=kv"),
            (False, True),
            (False, True),
        ),
    ]
    admitted = widened = extended = relaxed = total = 0
    for (q_lens, kv_lens), (hq, hk), grid in itertools.product(batches, heads, grids):
        qo_indptr = _indptr(q_lens)
        kv = torch.tensor(kv_lens, dtype=torch.int32)
        tile = (max(q_lens) // 128 + 1) * 128  # a query bound above the batch's
        pow2 = 1 << (max(kv_lens) - 1).bit_length()
        bounds = {
            "exact": (max(q_lens), max(kv_lens)),
            "kv": (max(q_lens), pow2),
            "q": (tile, max(kv_lens)),
            "q+kv": (tile, 8 * pow2),  # a table sized for a longer context
            "q=kv": (max(tile, pow2),) * 2,  # equal bounds above the lengths
        }
        work = hq * sum(q * k for q, k in zip(q_lens, kv_lens, strict=True))
        for static, host, loose, graph, spare in itertools.product(*grid):
            (head_dim_qk, head_dim_vo), page, layout, (qd, kd) = static
            max_q, max_kv = bounds[loose]
            total_q = sum(q_lens) + (max_q if spare else 0)
            on_host = host == "host"
            # A mirror that is not on the host is absent or a device tensor.
            # The original's device-only state is a host qo_indptr with
            # device KV lengths; the copy takes either mirror on a device
            # ("kv-device", "qo-device") as both ("device").
            away = None if host is None else _DeviceOnly()
            want = legacy_prefill._blackwell_paged_auto_cudnn(
                None,
                layout,
                qd,
                kd,
                qd,
                head_dim_qk,
                head_dim_vo,
                len(q_lens),
                hq,
                hk,
                max_q,
                max_kv,
                page,
                # The original requires the query lengths to sum to this
                # total; graph facts carry the capacity's total and drop that
                # clause.  Without host lengths the total is only a bound.
                sum(q_lens) if graph and on_host else total_q,
                qo_indptr,
                kv if on_host else away,
            )
            facts = _facts(
                q_lens,
                kv_lens,
                hq=hq,
                hk=hk,
                host=False,
                graph=graph,
                total_q_tokens=total_q,
                max_q_len=max_q,
                max_kv_len=max_kv,
                head_dim_qk=head_dim_qk,
                head_dim_vo=head_dim_vo,
                page_size=page,
                kv_layout=layout,
                q_dtype=qd,
                kv_dtype=kd,
                qo_indptr=qo_indptr if host in ("host", "kv-device") else away,
                kv_seq_lens=kv if host in ("host", "qo-device") else away,
            )
            got = _in_sm100_cudnn_region(facts)
            case = (q_lens, kv_lens, hq, hk, static, host, loose, graph, spare)
            if got != want:
                assert got and not want, case
                assert graph and on_host and q_lens == kv_lens, case
                assert len(q_lens) <= 4 and max(q_lens) >= 1024, case
                assert 1024 <= max_q < max_kv <= 8192, case
                assert 8 * mi <= work < 16 * mi, case
                widened += 1
            if got and len(q_lens) > 4:  # only the extension admits these
                extended += 1
                relaxed += graph and spare
            admitted += got
            total += 1
    assert 100 < admitted < total - 100  # the grid exercises both answers,
    assert extended > 100  # ... the extension,
    assert relaxed > 0  # ... under a capacity's total above the batch's,
    assert widened > 0  # ... and the graph-only band


@pytest.fixture
def cudnn_gate(monkeypatch):
    """The cuDNN version gate answers yes and counts how often it is asked."""
    asked = []
    monkeypatch.setattr(
        _auto_policy, "_cudnn_supports_paged_auto", lambda: not asked.append(1)
    )
    return asked


@pytest.fixture
def cudnn_gate_forbidden(monkeypatch):
    """Asking the cuDNN version gate (an import of the optional frontend)
    fails the test."""

    def asked():
        pytest.fail("the cuDNN version gate was asked")

    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", asked)


# Facts that leave the SM100 cuDNN region on a host value alone, one
# condition each; the base batch [128] over [16384] at 32/8 heads is inside.
_OUTSIDE_WITHOUT_THE_GATE = [
    dict(window_left=127),
    dict(causal=False),
    dict(logits_soft_cap=30.0),
    dict(custom_mask=True),
    dict(use_sinks=True),
    dict(kv_dtype=torch.float8_e4m3fn),
    dict(q_dtype=torch.float16, kv_dtype=torch.float16),
    dict(page_size=32),
    dict(kv_layout="NHD"),
    dict(head_dim_qk=64, head_dim_vo=64),
    dict(head_dim_qk=192),
    dict(hq=4, hk=1),
    dict(hq=128, hk=8),
    dict(hq=32, hk=1),
    dict(max_q_len=2048, total_q_tokens=2048, host=False),  # neither bound test
    dict(max_kv_len=32769, host=False),
    # more than four requests: only host lengths above 128 tokens could admit
    dict(batch_size=5, host=False),
    # the bound-based work is below 8 Mi
    dict(max_kv_len=2048, host=False, hq=16, hk=4),
]


@pytest.mark.parametrize("over", _OUTSIDE_WITHOUT_THE_GATE, ids=str)
def test_sm100_region_asks_the_cudnn_gate_last(cudnn_gate_forbidden, over):
    """The gate imports the optional cuDNN frontend; a plan that a host value
    already keeps out of the region never asks it."""
    facts = _facts([128], [16384], **over)
    assert not _in_sm100_cudnn_region(facts)
    assert ordered_sm100_backends(facts)[0] != "cudnn"


def test_sm100_region_asks_the_cudnn_gate_inside(cudnn_gate):
    facts = _facts([128], [16384])
    assert ordered_sm100_backends(facts)[0] == "cudnn"
    assert len(cudnn_gate) == 1
    # host lengths that fail the exact test are known only after the gate
    assert not _in_sm100_cudnn_region(_facts([64] * 4, [8192, 64, 64, 64], hq=8, hk=2))
    assert len(cudnn_gate) == 2


def test_sm100_region_reads_no_lengths_for_short_queries(
    monkeypatch, cudnn_gate_forbidden
):
    """The extension needs a query above 128 tokens.  Under a ``max_q_len``
    of at most 128 a plan outside the bounded domain is refused on its
    bounds: a decode or speculative plan neither reads its host lengths nor
    asks the gate for the extension."""

    unread = dict(qo_indptr=_HostUnread(), kv_seq_lens=_HostUnread())
    # (the query total of the last two has room for a 1024-token request next
    # to non-empty ones, so max_q_len alone keeps their lengths unread)
    for q_len, batch in ((1, 8), (16, 8), (128, 5), (128, 9), (16, 128)):
        facts = _facts([q_len] * batch, [8192] * batch, **unread)
        assert not _in_sm100_cudnn_region(facts)
    # control: one more query token, and the answer needs the lengths
    monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
    with pytest.raises(AssertionError, match="host lengths read"):
        _in_sm100_cudnn_region(_facts([129] * 5, [8192] * 5, **unread))


@pytest.mark.parametrize(
    "q_lens,reads",
    [
        ([1] * 64, False),  # decode
        ([16] * 64, False),  # speculative
        # 128 tokens per request, and one more token in the total
        ([128] * 5, False),
        ([129] + [128] * 4, True),
        # nine requests: a total of 1023 + 9 has room for a 1024-token one
        ([1023] + [1] * 8, False),
        ([1024] + [1] * 8, True),
    ],
    ids=["decode", "speculative", "q128", "q128+1", "one-q1023", "one-q1024"],
)
def test_sm100_region_reads_no_lengths_under_a_small_query_total(
    monkeypatch, q_lens, reads
):
    """A plan can declare a ``max_q_len`` far above its queries (here 2048).
    The extension needs more than 128 tokens for every request, or a
    1024-token request next to non-empty ones; a query total that allows
    neither refuses the plan before its host lengths are read or the gate is
    asked, so a decode batch does not pay for the extension under any
    declared bound."""

    facts = _facts(
        q_lens,
        [8192] * len(q_lens),
        max_q_len=2048,
        qo_indptr=_HostUnread(),
        kv_seq_lens=_HostUnread(),
    )
    if reads:
        monkeypatch.setattr(_auto_policy, "_cudnn_supports_paged_auto", lambda: True)
        with pytest.raises(AssertionError, match="host lengths read"):
            _in_sm100_cudnn_region(facts)
    else:
        monkeypatch.setattr(
            _auto_policy,
            "_cudnn_supports_paged_auto",
            lambda: pytest.fail("the cuDNN version gate was asked"),
        )
        assert not _in_sm100_cudnn_region(facts)


def test_policy_rereads_host_lengths_on_each_call(cudnn_qualified):
    facts = _facts([128], [8192], hq=8, hk=1)  # exactly 8 Mi of work
    assert ordered_sm100_backends(facts)[0] == "cudnn"
    facts.kv_seq_lens[0] = 8191  # the caller mutates its mirror in place
    assert ordered_sm100_backends(facts)[-1] == "cudnn"
    facts.kv_seq_lens[0] = 8192
    assert ordered_sm100_backends(facts)[0] == "cudnn"


@pytest.mark.parametrize(
    "over,match",
    [
        (dict(qo_indptr=_indptr([64, 64])), "query spans"),
        (dict(kv_seq_lens=torch.tensor([8192, 8192], dtype=torch.int32)), "KV lengths"),
        (dict(kv_seq_lens=torch.tensor([20000], dtype=torch.int32)), "max_kv_len"),
        (dict(qo_indptr=torch.tensor([0, 2000], dtype=torch.int32)), "max_q_len"),
        (dict(qo_indptr=torch.tensor([128, 0], dtype=torch.int32)), "query lengths"),
        (dict(total_q_tokens=100), "total_q_tokens 100"),
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
def test_other_architectures_keep_their_order(
    cudnn_gate_forbidden, cc, q_lens, over, order
):
    """Only exact SM100 has the measured rule; the same facts that pick cuDNN
    there leave every other architecture on its fixed order, without
    asking the cuDNN version gate."""
    facts = _facts(q_lens, [16384], **over)
    full = ordered_backends(cc, facts)
    assert full[: len(order)] == order
    assert sorted(full) == sorted(_AUTO_BACKEND_CANDIDATES)
    assert ordered_backends(cc, facts) == full


def test_default_order_matches_the_resolve_time_table(cudnn_gate_forbidden):
    """With a long query the plan-time order of the unmeasured architectures
    is the resolve-time default bucket; the other SM10x parts match the
    table in every hinted bucket too, on both sides of each bound (the bound
    is written once in ``_selection.HEURISTIC_ORDER`` and once in the
    policy).  SM90 is left out of the hinted walk: its plan-time fa2-first
    rule at ``max_q_len <= 64`` is the legacy wrapper's and has no
    resolve-time bucket."""
    facts = _facts([4096], [200704])
    for cc in ((8, 0), (9, 0), (10, 3), (12, 0)):
        static = _selection.heuristic_order(cc[0])
        assert plan_order(static, cc, facts) == static
    hinted = _selection.HEURISTIC_ORDER[10][:-1]
    assert hinted  # the walk below is not empty
    for bound, _ in hinted:
        for q_len in (1, bound, bound + 1):
            static = _selection.heuristic_order(10, q_len)
            assert plan_order(static, (10, 3), _facts([q_len], [200704])) == static
    for major in (8, 9, 12):  # one bucket each: nothing else to walk
        assert len(_selection.HEURISTIC_ORDER[major]) == 1


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
    full = ordered_sm100_backends(facts)
    for subset in (("trtllm-gen", "fa2"), ("fa2", "trtllm-gen"), ("fa2", "fa3")):
        assert plan_order(subset, (10, 0), facts) == tuple(
            name for name in full if name in subset
        )
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


# Known differences to the legacy wrapper's ``auto``: parts of its cuDNN
# admission that are not copied.  Each test says what closing it changes.


def _resolve_head_dim_256(monkeypatch, cc_major, q_dtype=torch.bfloat16):
    monkeypatch.setattr(
        _selection, "PROBES", {name: (lambda device: None) for name in CAPABILITIES}
    )
    return _resolve(
        cc_major=cc_major,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim_qk=256,
        q_dtype=q_dtype,
        page_size=16,
    )


def test_sm100_head_dim_256_is_not_copied(monkeypatch, cudnn_qualified):
    """The legacy SM100 admission has a second domain at head dim 256
    (``_blackwell_paged_gqa_auto_cudnn``: GQA prefix batches of at most four
    requests from 16 Mi of work) behind a gate of its own, which needs a
    newer cudnn-frontend than the head-dim-128 gate.  Here cuDNN is not a
    candidate at head dims (256, 256) and the region is False there, so with
    that gate open the legacy wrapper plans cuDNN where this API plans fa2.

    To close it: declare (256, 256) for cuDNN in the capability table and
    copy that predicate with its gate into the region.  This test then goes,
    and head dim 256 joins the grid of
    ``test_sm100_cudnn_region_against_the_legacy_predicate``."""
    monkeypatch.setattr(legacy_prefill, "get_compute_capability", lambda dev: (10, 0))
    resolution = _resolve_head_dim_256(monkeypatch, 10)
    assert resolution.backends == ("fa2",)
    assert "(256, 256)" in resolution.excluded["cudnn"]
    facts = _facts([128], [16384], head_dim_qk=256, head_dim_vo=256)  # 64 Mi
    bf16 = torch.bfloat16
    for gate in (False, True):
        monkeypatch.setattr(
            legacy_prefill,
            "_cudnn_supports_paged_gqa_auto",
            lambda head_dim, gate=gate: gate,
        )
        admitted = legacy_prefill._blackwell_paged_auto_cudnn(
            None,
            "HND",
            bf16,
            bf16,
            bf16,
            256,
            256,
            1,
            32,
            8,
            128,
            16384,
            16,
            128,
            facts.qo_indptr,
            facts.kv_seq_lens,
        )
        assert admitted is gate
        assert not _in_sm100_cudnn_region(facts)
    assert ordered_sm100_backends(facts)[0] == "fa2"
    assert plan_order(resolution.backends, (10, 0), facts) == ("fa2",)


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float16])
def test_sm90_head_dim_256_is_not_copied(monkeypatch, cudnn_gate_forbidden, q_dtype):
    """The legacy wrapper's ``auto`` admits cuDNN at head dim 256 on one
    Hopper part (``_hopper_paged_auto_cudnn``: H100 80GB HBM3, bf16 or fp16,
    a longest query above 64 tokens, from 16 Mi of work).  Here cuDNN is not
    a candidate at head dims (256, 256) on any architecture and the SM90
    order reads no device fact: fa3 leads such a batch, fa2 one whose
    longest query has at most 64 tokens.

    To close it: declare (256, 256) for cuDNN, give the policy the device
    facts that predicate reads and a region for SM90 pinned against it as
    the SM100 one is.  The candidate set and the orders below then change."""
    resolution = _resolve_head_dim_256(monkeypatch, 9, q_dtype)
    assert resolution.backends == ("fa3", "fa2")
    assert "(256, 256)" in resolution.excluded["cudnn"]
    for q_len, order in ((65, ("fa3", "fa2")), (64, ("fa2", "fa3"))):
        facts = _facts(
            [q_len],
            [16384],
            head_dim_qk=256,
            head_dim_vo=256,
            q_dtype=q_dtype,
            kv_dtype=q_dtype,
        )
        assert plan_order(resolution.backends, (9, 0), facts) == order


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


def _host_metadata(q_lens, kv_lens, *, width=None, max_q_len=None, page_size=16):
    qo_indptr = _indptr(q_lens)
    kv = torch.tensor(kv_lens, dtype=torch.int32)
    width = width or math.ceil(max(kv_lens) / page_size)
    return _HostMetadata(
        qo_indptr=qo_indptr.clone(),
        kv_seq_lens=kv.clone(),
        page_size=page_size,
        max_q_len=max_q_len or max(q_lens),
        max_kv_len=max(kv_lens),
        block_tables=torch.zeros(len(q_lens), width, dtype=torch.int32),
        qo_indptr_cpu=qo_indptr,
        kv_seq_lens_cpu=kv,
    )


_HOST_KW = dict(
    num_qo_heads=16, num_kv_heads=4, head_dim_qk=128, q_dtype=torch.bfloat16
)
# One batch per order at 16/4 heads, with the order it gets from the real
# cuDNN region and the stand-in rule below.
_CUDNN_BATCH = ([128], [4096])  # 8 Mi of work
_TRTLLM_BATCH = ([512, 512], [600, 1024])
_FA2_BATCH = ([16, 7], [300, 97])
_CUDNN_ORDER = ["cudnn", "fa2", "trtllm-gen"]
_TRTLLM_ORDER = ["trtllm-gen", "fa2", "cudnn"]
_FA2_ORDER = ["fa2", "trtllm-gen", "cudnn"]


def _layer2_sm100_rule(facts):
    """Stand-in for ``ordered_sm100_backends`` in this layer.  The tests here
    are about the controller (what it hands the policy, the walk, the
    freeze), not about where the measured trtllm-gen / fa2 thresholds sit, so
    that pair is ranked on one fact with a fixed bound.  The cuDNN region is
    the real one."""
    pair = ("trtllm-gen", "fa2") if facts.max_q_len >= 256 else ("fa2", "trtllm-gen")
    if _in_sm100_cudnn_region(facts):
        return _auto_policy._prefer("cudnn", *pair)
    return _auto_policy._prefer(*pair)


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
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", _layer2_sm100_rule)
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


@pytest.fixture
def policy_calls(monkeypatch):
    """Record what the controller hands the policy: ``(candidates, compute
    capability, facts)`` per call, the real ``plan_order`` still answering."""
    calls = []

    def spy(candidates, compute_capability, facts):
        calls.append((candidates, compute_capability, facts))
        return plan_order(candidates, compute_capability, facts)

    monkeypatch.setattr(_controller, "plan_order", spy)
    return calls


_SPY_BATCH = ([300, 7], [4921, 100])  # 308 pages: the inferred KV capacity is 4928


@pytest.mark.parametrize(
    "mode,bounds",
    [
        # (total_q_tokens, max_q_len, max_kv_len) the facts must carry
        ("eager", (307, 300, 4921)),
        ("graph", (307, 300, 4928)),  # GraphCapacity.from_metadata: whole pages
        ("capacity", (1024, 512, 8192)),
    ],
)
def test_plan_mode_and_bounds_reach_the_policy(planners, policy_calls, mode, bounds):
    """What the policy is asked, not what it answers (no threshold is
    involved): the bounds the kernels are planned with, the mode, and the
    batch's own host mirrors."""
    kwargs = dict(use_cuda_graph=True) if mode == "graph" else {}
    if mode == "capacity":
        kwargs = dict(
            graph_capacity=GraphCapacity(
                batch_size=2,
                total_q_tokens=1024,
                max_q_len=512,
                max_kv_len=8192,
                page_size=16,
                table_width=512,
            )
        )
    attn = planners.controller(**kwargs)
    md = _host_metadata(*_SPY_BATCH, width=512 if mode == "capacity" else None)
    attn.plan(md, **_HOST_KW)
    ((candidates, cc, facts),) = policy_calls
    assert candidates == attn.resolution.backends and cc == (10, 0)
    assert (facts.total_q_tokens, facts.max_q_len, facts.max_kv_len) == bounds
    assert facts.use_cuda_graph is (mode != "eager")
    assert facts.batch_size == 2 and facts.page_size == 16
    assert facts.qo_indptr is md.qo_indptr_cpu
    assert facts.kv_seq_lens is md.kv_seq_lens_cpu
    assert facts.host_lengths() == tuple(list(lens) for lens in _SPY_BATCH)
    assert attn.selection_order == plan_order(candidates, cc, facts)


_DEFAULT_FACTS = dict(
    num_qo_heads=16,
    num_kv_heads=4,
    head_dim_qk=128,
    head_dim_vo=128,
    kv_layout="HND",
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    causal=True,
    window_left=-1,
    logits_soft_cap=None,
    custom_mask=False,
    use_sinks=False,
)


def test_metadata_page_size_reaches_the_policy(planners, policy_calls):
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata(*_SPY_BATCH, page_size=32), **_HOST_KW)
    ((_, _, facts),) = policy_calls
    assert facts.page_size == 32
    assert facts.max_kv_len == 4928  # 154 pages of 32


_FP8 = torch.float8_e4m3fn
# case -> (plan() keywords, the facts that differ from _DEFAULT_FACTS)
_PLAN_ARGUMENTS = {
    "defaults": ({}, {}),
    "non-causal": (dict(causal=False), dict(causal=False)),
    "window": (dict(window_left=127), dict(window_left=127)),
    "nhd": (dict(kv_layout="NHD"), dict(kv_layout="NHD")),
    "sinks": (dict(use_sinks=True), dict(use_sinks=True)),
    "fp16": (
        dict(q_dtype=torch.float16),
        dict(q_dtype=torch.float16, kv_dtype=torch.float16),
    ),
    "d64": (dict(head_dim_qk=64), dict(head_dim_qk=64, head_dim_vo=64)),
    "heads32/8": (
        dict(num_qo_heads=32, num_kv_heads=8),
        dict(num_qo_heads=32, num_kv_heads=8),
    ),
    "heads8/8": (
        dict(num_qo_heads=8, num_kv_heads=8),
        dict(num_qo_heads=8, num_kv_heads=8),
    ),
    "soft-cap": (dict(logits_soft_cap=30.0), dict(logits_soft_cap=30.0)),
    # normalized before the policy sees them: an int cap is a float, 0 is off
    "soft-cap-int": (dict(logits_soft_cap=30), dict(logits_soft_cap=30.0)),
    "soft-cap-zero": (dict(logits_soft_cap=0), dict(logits_soft_cap=None)),
    "custom-mask": (
        dict(custom_mask=torch.ones(300 * 4921 + 7 * 100, dtype=torch.bool)),
        dict(custom_mask=True),
    ),
    "fp8-kv": (dict(kv_dtype=_FP8), dict(kv_dtype=_FP8)),
    "d192/128": (dict(head_dim_qk=192, head_dim_vo=128), dict(head_dim_qk=192)),
}


@pytest.mark.parametrize("case", list(_PLAN_ARGUMENTS))
def test_plan_arguments_reach_the_policy(planners, monkeypatch, policy_calls, case):
    """Every plan() argument the policy can rank on arrives in the facts as
    the caller gave it (or as plan() defaults it), and nothing else changes.
    What the policy answers is not looked at."""
    plan_kw, changed = _PLAN_ARGUMENTS[case]
    resolve = _controller.resolve_paged_attention

    def at_least_two(**kwargs):
        # The capability table leaves one candidate for a soft cap, a custom
        # mask, FP8 KV and head dim 192/128, and one candidate is planned
        # without asking the policy.  A second (fake) candidate stands in for
        # what a later capability or a caller's own Resolution would add.
        resolution = resolve(**kwargs)
        if len(resolution.backends) > 1:
            return resolution
        (only,) = resolution.backends
        other = "cudnn" if only == "fa2" else "fa2"
        return dataclasses.replace(resolution, backends=(only, other))

    monkeypatch.setattr(_controller, "resolve_paged_attention", at_least_two)
    attn = planners.controller()
    attn.plan(_host_metadata(*_SPY_BATCH), **dict(_HOST_KW, **plan_kw))
    ((candidates, cc, facts),) = policy_calls
    assert candidates == attn.resolution.backends and cc == (10, 0)
    want = dict(_DEFAULT_FACTS, **changed)
    got = {name: getattr(facts, name) for name in want}
    assert got == want
    # 30 == 30.0 and 1 == True: the types are part of what arrives
    assert {k: type(v) for k, v in got.items()} == {k: type(v) for k, v in want.items()}


@pytest.mark.parametrize(
    "batch",
    [_CUDNN_BATCH, _TRTLLM_BATCH, _FA2_BATCH],
    ids=["long-kv", "long-q", "small"],
)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_the_real_rule_is_walked_in_its_own_order(
    planners, monkeypatch, policy_calls, batch, graph
):
    """The one place in this layer where the real SM100 rule, measured
    thresholds included, sits behind the real controller: whatever it
    answers for the facts it was handed is the order that is published and
    walked.  The answer itself is not looked at."""
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", ordered_sm100_backends)
    planners.decline(*CAPABILITIES)
    attn = planners.controller(use_cuda_graph=graph)
    with pytest.raises(ValueError, match="no pinned candidate"):
        attn.plan(_host_metadata(*batch), **_HOST_KW)
    ((candidates, cc, facts),) = policy_calls
    order = list(plan_order(candidates, cc, facts))
    assert sorted(order) == sorted(candidates)
    assert planners.preflights == order
    planners.decline()
    attn.plan(_host_metadata(*batch), **_HOST_KW)
    assert list(attn.selection_order) == order and attn.backend == order[0]


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
    # the same on a later candidate: the first declines (typed), the second
    # fails, and the walk stops there instead of trying the third
    planners.preflights.clear()
    planners.reject = {
        "trtllm-gen": _BackendPlanUnsupportedError("trtllm-gen declines"),
        "fa2": error,
    }
    with pytest.raises(type(error)) as raised:
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert raised.value is error
    assert planners.preflights == ["trtllm-gen", "fa2"]
    assert attn.backend is None and attn.selection_trace == ()


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


def _forbid_ranking(monkeypatch):
    """Any entry into the policy fails the test from here on."""

    def forbidden(*args, **kwargs):
        pytest.fail("the policy was asked to rank the candidates")

    monkeypatch.setattr(_controller, "plan_order", forbidden)
    monkeypatch.setattr(_auto_policy, "plan_order", forbidden)
    monkeypatch.setattr(_auto_policy, "ordered_backends", forbidden)


def test_graph_mode_keeps_the_first_backend(planners, monkeypatch):
    """A graph-mode instance ranks once: a re-plan does not ask the policy
    and reports the frozen backend alone."""
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert attn.backend == "trtllm-gen"
    assert list(attn.selection_order) == _TRTLLM_ORDER
    # a batch of the same bucket that eager auto would give to fa2
    small = _host_metadata([4, 4], [64, 64], width=64)
    planners.preflights.clear()
    with monkeypatch.context() as patch:
        _forbid_ranking(patch)
        attn.plan(small, **_HOST_KW)
        assert attn.backend == "trtllm-gen"
        assert attn.selection_order == ("trtllm-gen",)
        assert planners.preflights == ["trtllm-gen"]
    eager = planners.controller()  # control: the same batch does rank eagerly
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
    attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert attn.backend == "fa2" and planners.preflights == ["fa2"]
    # the frozen backend declining later is an error, not a switch, and it
    # names the frozen backend only
    planners.decline("fa2")
    with pytest.raises(ValueError, match="no pinned candidate") as by_plan:
        attn.plan(_host_metadata(*_TRTLLM_BATCH), **_HOST_KW)
    assert "candidates ['fa2']" in str(by_plan.value)
    assert attn.backend == "fa2"


def test_graph_mode_ranks_on_the_capacity(planners):
    """The kernels are planned with the capacity's bounds, so the rows / fill
    rule reads those, not the first batch that happens to arrive.  (The cuDNN
    region reads the first batch's lengths as well: next two tests.)"""
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


def test_graph_mode_cudnn_admission_reads_the_first_batch(planners):
    """One capacity, two first batches on either side of 8 Mi of work (16
    heads * 256 * 2048): the bounds come from the capacity, the exact work
    and the q == kv proof from the first batch's host lengths, as in the
    legacy wrapper.  Two instances of one bucket can therefore freeze
    different backends; later batches are not re-ranked."""
    capacity = GraphCapacity(
        batch_size=1,
        total_q_tokens=256,
        max_q_len=256,
        max_kv_len=4096,
        page_size=16,
        table_width=256,
    )
    frozen = []
    for batch in (([256], [2048]), ([255], [2048])):
        attn = planners.controller(graph_capacity=capacity)
        attn.plan(_host_metadata(*batch, width=256), **_HOST_KW)
        frozen.append(attn.backend)
        # the other first batch arrives later: the capture keeps its backend
        attn.plan(_host_metadata([256], [2048], width=256), **_HOST_KW)
        attn.plan(_host_metadata([255], [2048], width=256), **_HOST_KW)
        assert attn.backend == frozen[-1]
    assert frozen[0] == "cudnn" and frozen[1] != "cudnn"


def test_graph_mode_cudnn_extension_reads_the_first_batch(planners):
    """The extension of the region reads the first planned batch alone.  Six
    requests of 256 tokens over 4096 keys (96 Mi of work at 16 heads) freeze
    cuDNN under a capacity far above them: its query and KV bounds pass
    neither bound test, and its total is above the batch's.  A first batch
    with one request of 128 tokens does not; later batches are not
    re-ranked."""
    capacity = GraphCapacity(
        batch_size=6,
        total_q_tokens=6 * 2048,
        max_q_len=2048,
        max_kv_len=65536,
        page_size=16,
        table_width=4096,
    )
    batches = [[256] * 6, [256] * 5 + [128]]
    frozen = []
    for q_lens in batches:
        attn = planners.controller(graph_capacity=capacity)
        attn.plan(_host_metadata(q_lens, [4096] * 6, width=4096), **_HOST_KW)
        frozen.append(attn.backend)
        for later in batches:
            attn.plan(_host_metadata(later, [4096] * 6, width=4096), **_HOST_KW)
        assert attn.backend == frozen[-1]
    assert frozen[0] == "cudnn" and frozen[1] != "cudnn"


def test_graph_mode_full_prefill_under_the_inferred_capacity(planners):
    """Full prefill with a length that is not a page multiple: the inferred
    capacity's KV bound is the table's (69 pages = 1104 > 1100).  At 8 heads
    the batch carries 9.2 Mi of work, below the 16 Mi from which its lengths
    would admit it under any bound, so only the widened graph-mode bound test
    admits it; a request that is not full prefill is not admitted."""
    heads = dict(_HOST_KW, num_qo_heads=8, num_kv_heads=2)
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata([1100], [1100]), **heads)
    assert attn.backend == "cudnn"
    attn = planners.controller(use_cuda_graph=True)
    attn.plan(_host_metadata([1100], [1101]), **heads)
    assert attn.backend != "cudnn"


def test_failed_replan_keeps_the_published_plan_runnable(planners, monkeypatch):
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
    walked = list(planners.preflights), list(planners.plans)
    _forbid_ranking(monkeypatch)  # run() neither ranks nor plans
    attn.run(q, (cache, cache))
    assert planners.runs == ["fa2"]
    assert (planners.preflights, planners.plans) == walked
    assert attn.selection_order == order_before


def test_policy_reads_the_metadata_mirrors_on_each_plan(planners):
    """One metadata object planned twice: an in-place change of its host
    mirror between the plans changes the selection."""
    attn = planners.controller()
    md = _host_metadata([256], [2048])  # 16 heads: exactly 8 Mi of work
    attn.plan(md, **_HOST_KW)
    assert attn.backend == "cudnn"
    md.kv_seq_lens_cpu[0] = 2000
    attn.plan(md, **_HOST_KW)
    assert attn.backend != "cudnn"


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
    their fixed order to the last candidate."""
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
# shape -> (first batch, second batch, table width in pages).  The second
# batch changes the lengths, the query partition and (by seed) every page id
# while staying inside the first batch's capacity, so it serves as a graph
# re-plan as well as an eager one.  When written, the SM100 rule gave
# "long-kv" to cuDNN, "long-q" to trtllm-gen and "short-q" to fa2; no test
# below depends on that.
_SHAPES = {
    # ragged prefix: 16 * (128 * 4096 + 100 * 3000) = 13 Mi of work
    "long-kv": (([128, 100], [4096, 3000]), ([90, 120], [2000, 4000]), 256),
    "long-q": (([512, 384], [600, 1024]), ([300, 500], [1024, 700]), 64),
    "short-q": (
        ([16, 7, 1, 30], [300, 97, 1000, 40]),
        ([10, 9, 2, 20], [100, 500, 900, 64]),
        63,
    ),
}
# The shape a candidate is given when a test puts it first.
_SHAPE_OF = {
    "cudnn": "long-kv",
    "trtllm-gen": "long-q",
    "fa3": "long-q",
    "fa2": "short-q",
}


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


def _problems(shape):
    first, second, width = _SHAPES[shape]
    return _problem(*first, width, seed=11), _problem(*second, width, seed=12)


def _force_first(monkeypatch, *lead):
    """Inject the policy's order, as the MLA suite does: the ``lead``
    candidates first, in that order, then the rest.  Lifetime and numerical
    tests must not depend on what the measured rule ranks first for a shape
    (docs/development/backend_notes.md).  Skips where a ``lead`` candidate
    cannot run."""
    runnable = _resolve(
        device=torch.device("cuda:0"),
        num_qo_heads=_HQ,
        num_kv_heads=_HK,
        head_dim_qk=_HD,
        q_dtype=torch.bfloat16,
        page_size=_PAGE,
    ).backends
    for name in lead:
        if name not in runnable:
            pytest.skip(f"{name} is not an auto candidate here ({runnable})")
    order = lead + tuple(b for b in _AUTO_BACKEND_CANDIDATES if b not in lead)
    monkeypatch.setattr(_auto_policy, "ordered_backends", lambda cc, facts: order)
    return runnable


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


def _replay(graph, out, lse):
    """Replay over poisoned outputs: a row the replay does not write fails
    the oracle instead of keeping an earlier launch's result (the warm-up
    runs of ``_capture`` leave the capture batch's correct answer behind)."""
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()


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
@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_auto_matches_the_oracle(shape, graph):
    """The real rule with the real preflights, whatever it picks here: the
    numbers only, over a plan and a re-plan with new lengths, query
    partition and page ids."""
    p1, p2 = _problems(shape)
    attn = PagedAttentionFacade(use_cuda_graph=graph)
    # (a graph-mode instance cannot grow back past the rows its last run() saw)
    for p in (p1, p2) if graph else (p1, p2, p1):
        _plan_auto(attn, p)
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        _assert_matches_oracle(p, out, lse)


@needs_cuda
@pytest.mark.parametrize("lead", ["fa2", "cudnn"])
def test_auto_eager_switches_providers(monkeypatch, lead):
    """One eager instance over every runnable candidate and back, ``lead``
    planned first on the fresh instance.  Each return to a candidate plans
    its shape's other batch (new lengths, query partition and page ids), so
    state left from its earlier visit cannot stand in for the new plan."""
    names = [lead, *(n for n in _force_first(monkeypatch, lead) if n != lead)]
    if len(names) < 2:
        pytest.skip("needs two runnable candidates")
    visits = [(name, 0) for name in names]
    visits += [(name, 1) for name in (names[0], *reversed(names[1:]))]
    attn = PagedAttentionFacade()
    for name, which in visits:
        p = _problems(_SHAPE_OF[name])[which]
        _force_first(monkeypatch, name)
        _plan_auto(attn, p)
        assert attn.backend == name  # the injected order, not the rule's
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        _assert_matches_oracle(p, out, lse)


@needs_cuda
@pytest.mark.parametrize("first", _AUTO_BACKEND_CANDIDATES)
def test_auto_capture_replan_replay(monkeypatch, first):
    p1, p2 = _problems(_SHAPE_OF[first])
    _force_first(monkeypatch, first)
    attn = PagedAttentionFacade(use_cuda_graph=True)
    q, k_cache, v_cache, out, lse = _graph_buffers(p1)
    _plan_auto(attn, p1)
    assert attn.backend == first  # the injected order, not the rule's
    _forbid_ranking(monkeypatch)  # run(), capture, re-plan and replay never rank
    graph = _capture(attn, q, k_cache, v_cache, out, lse)
    _replay(graph, out, lse)
    _assert_matches_oracle(p1, out, lse)

    _load(p2, q, k_cache, v_cache)
    _plan_auto(attn, p2)
    assert attn.backend == first
    _replay(graph, out, lse)
    _assert_matches_oracle(p2, out, lse)

    # back to the larger batch: the rows past p2's total must be written again
    _load(p1, q, k_cache, v_cache)
    _plan_auto(attn, p1)
    _replay(graph, out, lse)
    _assert_matches_oracle(p1, out, lse)


@needs_cuda
@pytest.mark.parametrize(
    "first,fallback",
    [
        ("cudnn", "fa2"),
        ("fa2", "cudnn"),  # a fallback outside cuDNN's auto admission
        ("trtllm-gen", "cudnn"),
        ("fa2", "trtllm-gen"),
        ("fa3", "fa2"),
    ],
)
def test_graph_typed_fallback_freezes_and_replays(monkeypatch, first, fallback):
    """The first choice declines the capture batch; the capture then keeps
    the fallback through a re-plan that the first choice would accept."""
    p1, p2 = _problems(_SHAPE_OF[first])
    _force_first(monkeypatch, first, fallback)
    attn = PagedAttentionFacade(use_cuda_graph=True)
    q, k_cache, v_cache, out, lse = _graph_buffers(p1)
    with monkeypatch.context() as patch:
        _decline(patch, first)
        _plan_auto(attn, p1)
    assert attn.backend == fallback
    assert attn._impl.selection_trace == (
        (first, "preflight", f"{first} declines"),
        (fallback, "preflight", "accepted"),
    )
    _forbid_ranking(monkeypatch)
    graph = _capture(attn, q, k_cache, v_cache, out, lse)
    _replay(graph, out, lse)
    _assert_matches_oracle(p1, out, lse)

    _load(p2, q, k_cache, v_cache)
    _plan_auto(attn, p2)
    assert attn.backend == fallback
    _replay(graph, out, lse)
    _assert_matches_oracle(p2, out, lse)


@needs_cuda
@pytest.mark.parametrize("first", _AUTO_BACKEND_CANDIDATES)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_auto_runtime_scales(monkeypatch, first, graph):
    """``sm_scale`` / ``k_scale`` / ``v_scale`` arrive only at run(): whichever
    candidate auto planned must apply them."""
    p, _ = _problems(_SHAPE_OF[first])
    _force_first(monkeypatch, first)
    sm_scale, k_scale, v_scale = 0.05, 1.5, 0.5
    attn = PagedAttentionFacade(use_cuda_graph=graph)
    _plan_auto(attn, p)
    assert attn.backend == first  # the injected order, not the rule's
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
@pytest.mark.parametrize(
    "first,shape",
    [(None, shape) for shape in _SHAPES] + list(_SHAPE_OF.items()),
    ids=[f"rule-{shape}" for shape in _SHAPES] + [f"first-{b}" for b in _SHAPE_OF],
)
def test_auto_plan_does_not_read_the_device_back(monkeypatch, first, shape):
    """Lengths given on the device only: the one documented readback is at
    metadata construction, which fills the metadata's own host mirrors; the
    plan — policy included — reads those, so it neither synchronizes nor
    waits for queued GPU work.  (The policy therefore sees the same facts as
    with caller-owned mirrors and the choice is the same by construction;
    its answer for facts without host lengths is covered at policy level, in
    the rule table.)  Under the real rule for each shape, and with each
    candidate put first."""
    p1, p2 = _problems(shape)
    if first is not None:
        _force_first(monkeypatch, first)
    attn = PagedAttentionFacade()
    for p in (p2, p1):  # warm both shapes: module load, cuDNN plan build
        _plan_auto(attn, p)
        attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        if first is not None:
            assert attn.backend == first  # control: the injected order was walked
    with_mirrors = attn.backend
    md1 = make_metadata(p1, with_mirrors=False)
    md2 = make_metadata(p2, with_mirrors=False)
    kwargs = dict(
        num_qo_heads=_HQ,
        num_kv_heads=_HK,
        head_dim_qk=_HD,
        q_dtype=torch.bfloat16,
        lse_mode="base2",
    )
    for md in (md2, md1):
        # One busy window per plan.  The drain first: fa2 / fa3 wait for their
        # own previous schedule upload by design (fa_backend.plan; pinned by
        # test_replan_waits_for_the_previous_schedule_upload).
        torch.cuda.synchronize()
        torch.cuda._sleep(1 << 28)  # a plan() that waits returns only after this
        busy = torch.cuda.Event()
        busy.record()
        torch.cuda.set_sync_debug_mode("error")
        try:
            attn.plan(md, **kwargs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert not busy.query(), "auto plan() waited for queued GPU work"
        assert md.qo_indptr_cpu is not None and md.kv_seq_lens_cpu is not None
    assert attn.backend == with_mirrors  # same facts, same choice
    out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))
    _assert_matches_oracle(p1, out, lse)


def PagedAttentionFacade(**kwargs):
    from flashinfer.prefill import PagedAttention

    return PagedAttention(torch.device("cuda:0"), **kwargs)
