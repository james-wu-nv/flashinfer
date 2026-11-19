"""Automatic paged-prefill selection: architecture dispatch and ranking.

The policy only orders the auto candidates from plan-time request facts. It
never decides support: the static capability table and probes
(``_selection.py``) own the candidate set, each backend's ``preflight()`` owns
batch eligibility (typed ``_BackendPlanUnsupportedError``), and the controller
walks the order returned here. Mirrors ``flashinfer/mla/_batch_mla/_auto_policy.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ._contracts import _expect

# Default fallback order: generated FA, then the Blackwell context kernel,
# then cuDNN.  Architecture policies move their measured preferences first.
# ``cake`` is deliberately absent on every architecture: it runs only when
# pinned by name.
_AUTO_BACKEND_CANDIDATES = ("fa2", "fa3", "trtllm-gen", "cudnn")

_FP8 = (torch.float8_e4m3fn, torch.float8_e5m2)


def _prefer(*backends: str) -> Tuple[str, ...]:
    return backends + tuple(b for b in _AUTO_BACKEND_CANDIDATES if b not in backends)


def _cudnn_supports_paged_auto() -> bool:
    # Deferred: the cuDNN frontend is optional and only the SM100 region asks.
    from ...cudnn.prefill import _cudnn_supports_paged_auto as supported

    return supported()


@dataclass(frozen=True)
class _PlanFacts:
    """What the policy may read about one plan(): host values only.

    ``total_q_tokens`` / ``max_q_len`` / ``max_kv_len`` are the bounds the
    kernels are planned with: the batch's declared values in eager mode, the
    graph capacity's in CUDA-graph mode.  ``qo_indptr`` / ``kv_seq_lens`` are
    the batch's host mirrors when it has them; they are read on every call
    (callers may mutate them in place) and a device tensor is treated as
    absent, never read back.
    """

    batch_size: int
    total_q_tokens: int
    max_q_len: int
    max_kv_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    kv_layout: str
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    causal: bool
    window_left: int
    logits_soft_cap: Optional[float]
    custom_mask: bool
    use_sinks: bool
    use_cuda_graph: bool
    qo_indptr: Optional[torch.Tensor] = None
    kv_seq_lens: Optional[torch.Tensor] = None

    def __post_init__(self):
        for name in (
            "batch_size",
            "total_q_tokens",
            "max_q_len",
            "num_qo_heads",
            "num_kv_heads",
            "page_size",
        ):
            value = getattr(self, name)
            _expect(
                isinstance(value, int) and not isinstance(value, bool) and value >= 1,
                f"auto policy: {name} must be a positive host int, got {value!r}",
            )
        _expect(
            isinstance(self.max_kv_len, int) and self.max_kv_len >= 0,
            f"auto policy: max_kv_len must be a host int >= 0, got {self.max_kv_len!r}",
        )

    def host_lengths(self) -> Optional[Tuple[List[int], List[int]]]:
        """Per-request ``(q_lens, kv_lens)`` from the host mirrors, or ``None``.

        Malformed mirrors are an error, not a reason to rank differently.
        """
        qo, kv = self.qo_indptr, self.kv_seq_lens
        if (
            qo is None
            or kv is None
            or qo.device.type != "cpu"
            or kv.device.type != "cpu"
        ):
            return None
        bounds, kv_lens = qo.flatten().tolist(), kv.flatten().tolist()
        q_lens = [end - start for start, end in zip(bounds, bounds[1:], strict=False)]
        # Plans read their lengths through here, so a message (it formats both
        # lists) is built only when its check fails.
        if not len(q_lens) == len(kv_lens) == self.batch_size:
            raise ValueError(
                f"auto policy: {len(q_lens)} query spans and {len(kv_lens)} KV "
                f"lengths for a batch of {self.batch_size}"
            )
        if not (
            min(q_lens) >= 0
            and max(q_lens) <= self.max_q_len
            and sum(q_lens) <= self.total_q_tokens
        ):
            raise ValueError(
                f"auto policy: query lengths {q_lens} do not fit max_q_len "
                f"{self.max_q_len} / total_q_tokens {self.total_q_tokens}"
            )
        if not (min(kv_lens) >= 0 and max(kv_lens) <= self.max_kv_len):
            raise ValueError(
                f"auto policy: KV lengths {kv_lens} do not fit max_kv_len "
                f"{self.max_kv_len}"
            )
        return q_lens, kv_lens


def ordered_sm80_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA2 on SM8x, with cuDNN as fallback."""
    return _prefer("fa2", "cudnn")


def ordered_sm90_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA3 on SM90, except FA2 for short non-FP8 queries.

    FA3 uses a fixed query tile; the legacy wrapper measured FA2 faster on H20
    when the longest query has at most 64 tokens.  The legacy wrapper's cuDNN
    admission on H100 (head dim 256, where cuDNN is not a candidate of this
    API) is not copied.
    """
    if facts.max_q_len <= 64 and facts.q_dtype not in _FP8:
        return _prefer("fa2", "fa3", "cudnn")
    return _prefer("fa3", "fa2", "cudnn")


def ordered_sm10x_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Blackwell parts other than SM100: the order measured on B200 (SM100),
    not measured on these parts.

    The context kernel reads a request's whole KV for a handful of query
    tokens, so FA2's split-KV schedule leads at decode / speculative lengths
    (B200: 5.6x at q=1, 2.5x at q=16, crossover by q=256; see
    ``_selection.HEURISTIC_ORDER``).
    """
    if facts.max_q_len <= 16:
        return _prefer("fa2", "trtllm-gen", "cudnn")
    return _prefer("trtllm-gen", "cudnn", "fa2")


def ordered_sm12x_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA2 on SM12x, with cuDNN as fallback."""
    return _prefer("fa2", "cudnn")


def _in_sm100_cudnn_region(facts: _PlanFacts) -> bool:
    """Whether a plan is in the SM100 cuDNN region, the legacy head-dim-128 domain.

    Duplicates the head-dim-128 part of
    ``flashinfer.prefill._blackwell_paged_auto_cudnn`` and the feature
    conditions around its call in
    ``BatchPrefillWithPagedKVCacheWrapper.plan()`` that have a counterpart
    here.  On eager facts the two are equal at head dim 128, and a test pins
    them against each other.  (The legacy predicate also tests that
    ``qo_indptr`` starts at 0; the metadata guarantees it, so the copy has
    no such clause.)  The legacy predicate also admits a head-dim-256 GQA
    domain; that part is not copied (cuDNN is not a candidate of this API at
    head dim 256), so head dim 256 is outside the region.

    Every admission needs a causal plan without window, soft cap, custom mask
    or sinks; HND bf16 Q and KV at head dim 128 and page size 16; 8 to 64
    query heads in groups of 1, 2, 4, 8 or 16; and the cuDNN version gate,
    which is the one the legacy wrapper asks (a FROST runtime with the paged
    packed-split selector on cuDNN 9.24 or newer).  With Mi for 2^20 and
    ``work`` for ``num_qo_heads * sum(q_i * kv_i)`` over the requests, two
    parts admit:

    * The bounded domain, which is the original one: at most four requests
      whose bounds pass the prefix test (``64 <= max_q <= 1024``,
      ``2048 <= max_kv <= 32768``, ``max_kv >= 4 * max_q``) or the
      full-prefill test (``1024 <= max_q == max_kv <= 8192``), with
      ``work >= 8 Mi``.  Equal bounds do not prove full prefill, so that test
      also needs every ``q_i == kv_i > 0`` and a longest query of at least
      1024 tokens.
    * The extension, for any number of requests: ``work >= 16 Mi`` and every
      ``0 < q_i <= kv_i``; then either every ``q_i == kv_i`` with the longest
      query in 1024..16384, or every ``q_i > 128`` with the longest query at
      most 8192 and the longest KV in 2048..32768.  A request of at most 128
      query tokens keeps a batch that is not all full prefill out of the
      extension, whatever work the other requests carry.

    The two bound tests and a cheap work bound
    (``total_q_tokens * num_qo_heads * max_kv_len >= 8 Mi``, tested before
    the gate) read the declared bounds ``max_q_len`` / ``max_kv_len``.  The
    work, the ``q_i == kv_i`` proof and every test of the extension read the
    per-request host lengths; "longest" there is the batch's own, so a batch
    that qualifies for the extension is admitted under any bounds declared
    above its lengths.  Two host values rule the extension out before a
    length is read: a ``max_q_len`` of at most 128, and a ``total_q_tokens``
    that allows neither more than 128 tokens for every request nor a
    1024-token request next to non-empty ones
    (``total_q_tokens <= 128 * batch_size`` and
    ``total_q_tokens < 1023 + batch_size``).  A plan outside the bounded
    domain is then refused on them, so a decode batch does not read its
    lengths under any ``max_q_len``.

    Without host lengths only the prefix test can admit, for at most four
    requests and with the cheap bound as its only work test; full prefill
    and the extension are never assumed from the bounds.

    CUDA-graph facts mix two sources, and the answer follows both:

    * ``max_q_len`` / ``max_kv_len`` / ``total_q_tokens`` are the graph
      capacity's.  The bound tests and the cheap work bound read those.  A
      capacity cannot show ``max_q == max_kv`` for a full prefill whose
      length is not a page multiple, so the full-prefill bound test is
      ``1024 <= max_q <= max_kv <= 8192`` there.
    * ``qo_indptr`` / ``kv_seq_lens`` are the lengths of the batch being
      planned.  A graph-mode instance ranks until a plan succeeds, so they
      are the first planned batch's, and that batch alone decides the work,
      the full-prefill proof and the extension.  The capacity does not limit
      the extension: a first batch that qualifies freezes cuDNN, planned
      with the capacity's bounds, however far those are above it.  Two
      instances with one capacity can freeze different backends, and later
      batches are not checked against the region.  The legacy predicate
      also requires the query lengths to sum to its declared total; a
      capacity's total is an upper bound, so graph facts drop that clause.

    Given the same bounds and lengths, graph facts admit everything the
    legacy predicate admits (the same test checks it).  Apart from the
    dropped total clause they admit more only through the wider full-prefill
    bound test: at most four requests of full prefill with
    ``8 Mi <= work < 16 Mi`` under a capacity whose ``max_kv`` is above its
    ``max_q``.  From 16 Mi the extension admits such a batch under any
    bounds.
    """
    if not (
        facts.causal
        and facts.window_left == -1
        and facts.logits_soft_cap is None
        and not facts.custom_mask
        and not facts.use_sinks
    ):
        return False
    max_q, max_kv = facts.max_q_len, facts.max_kv_len
    hq, hk = facts.num_qo_heads, facts.num_kv_heads
    prefix = 64 <= max_q <= 1024 and 2048 <= max_kv <= 32768 and max_kv >= 4 * max_q
    full = 1024 <= max_q <= 8192 and max_q == max_kv
    if facts.use_cuda_graph and not full:
        # A graph bucket declares its capacity, and a table is at least a page
        # wider than a query that is not a multiple of it, so equal bounds
        # cannot identify full prefill there.  Accept a KV bound up to the
        # same limit; the host lengths below still have to prove q_i == kv_i.
        full = 1024 <= max_q <= max_kv <= 8192
    # The original domain: at most four requests inside one of the bound tests.
    bounded = facts.batch_size <= 4 and (prefix or full)
    # The extension reads per-request lengths only, so it needs them on the
    # host.  It also needs every query above 128 tokens, or a longest one of
    # 1024 with no empty request beside it.  Two host values rule both out
    # before a length is read: a max_q_len of at most 128, and a query total
    # of at most 128 per request that is also below 1023 + batch (a graph
    # capacity's total is an upper bound of the batch's, so this holds there
    # too).  A decode batch therefore never reads its lengths for the
    # extension, whatever max_q_len it is planned with.
    qo, kv = facts.qo_indptr, facts.kv_seq_lens
    total_q, batch = facts.total_q_tokens, facts.batch_size
    can_extend = (
        max_q > 128
        and (total_q > 128 * batch or total_q >= 1023 + batch)
        and qo is not None
        and kv is not None
        and qo.device.type == "cpu"
        and kv.device.type == "cpu"
    )
    if not (bounded or can_extend):
        return False
    if not (
        facts.kv_layout == "HND"
        and facts.q_dtype == facts.kv_dtype == torch.bfloat16
        and facts.head_dim_qk == facts.head_dim_vo == 128
        and 8 <= hq <= 64
        and hq % hk == 0
        and hq // hk in (1, 2, 4, 8, 16)
        and facts.page_size == 16
        # Cheap upper-bound rejection before querying optional runtime support.
        and facts.total_q_tokens * hq * max_kv >= 8 * 1024 * 1024
        and _cudnn_supports_paged_auto()
    ):
        return False
    # Pair each query with its own KV length when the lengths are on the host;
    # with device-only lengths keep the bound-based answer of the prefix test.
    lengths = facts.host_lengths()
    if lengths is None:
        return prefix
    q_lens, kv_lens = lengths
    work = hq * sum(q * kv for q, kv in zip(q_lens, kv_lens, strict=True))
    each_full = q_lens == kv_lens
    if bounded:
        # Maxima alone cannot distinguish full prefill from a mixed-prefix
        # batch.  (A graph bucket can satisfy both bound tests; the prefix
        # one needs no proof.)
        proven_full = full and each_full and min(q_lens) > 0 and max(q_lens) >= 1024
        if (prefix or proven_full) and work >= 8 * 1024 * 1024:
            return True
    if not can_extend:
        return False
    # The extension: any batch size, the batch's own lengths, twice the work.
    if not (
        work >= 16 * 1024 * 1024
        and all(0 < q <= kv for q, kv in zip(q_lens, kv_lens, strict=True))
        # The legacy predicate requires the lengths to sum to its declared
        # total.  A graph capacity's total is an upper bound, which
        # host_lengths() has checked.
        and (facts.use_cuda_graph or sum(q_lens) == facts.total_q_tokens)
    ):
        return False
    if each_full:
        return 1024 <= max(q_lens) <= 16384
    # A short request needs its own qualification even when a longer one
    # carries the work.
    return min(q_lens) > 128 and max(q_lens) <= 8192 and 2048 <= max(kv_lens) <= 32768


# SM100 rule.  The thresholds are fitted on GB200 (cuDNN 9.27, cudnn-frontend
# 1.32) and measured there against the legacy wrapper's auto over 1322 cases
# (661 shapes, eager and CUDA graph; 2026-11-18, with the legacy wrapper's
# widened cuDNN admission).  A case passes when no paired measurement block
# is more than 1% slower than the legacy time and is a regression when every
# block is more than 1% slower:
#
#   shapes           mode   geomean speedup  pass  inconclusive  regression
#   219, main list   graph  1.49x             163            47           9
#                    eager  1.41x             205            14           0
#   661, all lists   graph  1.39x             494           151          16
#                    eager  1.20x             575            85           1
#
# The graph regressions run cuDNN under both APIs (at most 6.8% and 21 us
# longer) and are an effect of the measurement: the legacy wrapper compared
# with itself is flagged as often (about 3% of such shapes per run, measured
# 2026-11-10), and over 24 processes per shape the unified time was 0.975 to
# 1.017 of the legacy time (mean 1.002).  The eager one (head dim 64, B3
# q128 kv2048 64/8, fa2 under the legacy wrapper and trtllm-gen here) takes
# 10% longer in this run and 4% less in the one before.  The main list holds
# engine benchmark and CI configurations, the shapes the legacy wrapper's
# first cuDNN region was admitted on and length-regime probes.  Of the other
# 442 shapes, 212 are placed on the thresholds below, 176 inside the widened
# cuDNN admission (geomean 1.07x eager, 1.05x graph: cuDNN under both APIs
# except the eager small kernels below) and 54 at head dim 256 (fa2 under
# both, 1.00x), so the main list is the representative one.
#
#   rows = num_qo_heads * total_q / 128              query rows, in 128-row tiles
#   ntg  = batch * num_qo_heads * ceil(max_q / 128)  the context kernel's CTA grid
#   fill = rows / ntg                                how full those tiles are
#
# run() medians in us of each backend named through this API, from the
# measurement the thresholds were fitted on (2026-11-10); bf16, causal, page
# 16, GQA 32/8 unless noted:
#
#   mode   shape                         rows  fill   fa2  trtllm-gen  cudnn  first
#   graph  B1 q128  kv16384                32  1.00   115         193     52  cudnn (region)
#   graph  B1 q1024 kv32768 8/1            64  1.00   431         367    133  cudnn (region)
#   graph  B8 q48   kv2048                 96  0.375   89          63     78  trtllm-gen
#   graph  B6 q128  kv2048  16/4           96  1.00    73          36     46  trtllm-gen
#   graph  B1 q2048 kv2048  4/1            64  1.00    53          33     35  trtllm-gen
#   graph  B8 q32   kv2048                 64  0.25    56          61     77  fa2
#   graph  B16 q24  kv8192                 96  0.19   242         432    526  fa2
#   graph  B4 q32   kv8192                 32  0.25    72         112    174  fa2
#   graph  B2 q128  kv16384 d64            64  1.00   166         181      -  fa2
#   graph  B3 q128  kv16384 d64            96  1.00   229         189      -  trtllm-gen
#   eager  B8 q96   kv2048                192  0.75   130         106    150  trtllm-gen
#   eager  B16 q64  kv2048                256  0.50   132         165    185  fa2
#   eager  B16 q48  kv2048                192  0.375  132         168    185  fa2
#
# The thresholds are coarse crossovers, not exact optima: at rows 48 / fill
# 0.375 (B4 q48) trtllm-gen already leads in graph mode, 35 against 51 us at
# kv 2048 and 115 against 128 us at kv 8192.
_SM100_MIN_ROWS = 64
_SM100_MIN_ROWS_SMALL_HEAD_DIM = 96
# Between fill 0.375 and 0.75 the context kernel is only a few percent ahead;
# eager, its host cost on a multi-request batch is larger than that gain.
_SM100_MIN_FILL_GRAPH = 0.375
_SM100_MIN_FILL_EAGER = 0.75
# Eager only, for a batch the test above gives to trtllm-gen: when a query row
# attends few keys (kv_eff: the longest KV length of the batch, or
# window_left + 1 if that is smaller) the kernels take tens of microseconds
# and the launch decides.
# Measured on GB200 with fa2 launched directly (2026-11-10, 431 eager cases),
# trtllm-gen time / fa2 time by rows and kv_eff:
#
#   rows       kv_eff 128   256         512         1024
#   128..148   0.65-0.75    0.64-0.74   0.61-0.75   0.64-0.80
#   160..256   1.27-1.46    1.23-1.54   1.10-1.16   0.75-0.96
#   512        1.12-1.27    1.01-1.31   0.80-1.09   0.61-0.84
#   768        1.23         1.06-1.12   0.94-0.95
#   1024       1.02-1.22    0.90-1.06   0.73-0.94   0.56-0.78
#   1536       1.11         0.97-1.07   0.87-0.88
#   2048                    0.81-0.87   0.73        0.54-0.67
#
# - Up to 148 rows fa2 takes 71-88 us against 38-63 us at 160-256 rows: its
#   planner splits the KV range while the CTA grid fits the 148 SMs of a B200
#   and pays a merge kernel that short KV does not earn back.  trtllm-gen
#   keeps those batches.
# - From 1024 keys per row, or rows * kv_eff of 2^19, trtllm-gen leads again.
#   Between 2^18 and 2^19 the two are within about 10% either way; fa2 is the
#   legacy wrapper's choice there and keeps the batch.
# - In CUDA-graph mode a replay has no launch cost and trtllm-gen leads in all
#   117 such cases (0.25-0.73), so the exception is eager only.
_SM100_EAGER_SHORT_KV = 1024
_SM100_EAGER_SHORT_KV_SPLIT_ROWS = 148
_SM100_EAGER_SHORT_KV_MIN_WORK = 1 << 19
# Eager only, inside the cuDNN region: a small kernel goes to trtllm-gen ahead
# of cuDNN.  Measured on GB200 (2026-11-17, cuDNN 9.27, each backend named
# through this API, the 291 eager shapes of the comparison lists that fall in
# the region), trtllm-gen run() time / cuDNN run() time, range and geometric
# mean, by kv_eff, rows * kv_eff and the longest request's rows
# (num_qo_heads * its query length / 128):
#
#   kv_eff       rows * kv_eff   request rows <= 256   257..512         > 512
#   <= 4096      <= 2^22         0.70-1.00  0.86 (83)  0.79-1.10  0.96  0.98-1.15  1.04
#   <= 4096      2^22..2^23      0.92-1.01  0.97  (8)  0.88-1.07  0.99  1.02
#   <= 4096      > 2^23          0.96-1.11  1.04  (3)  0.94-1.09  1.02  0.99-1.21  1.09
#   4097..8192   <= 2^22         0.91-1.87  1.31 (22)  0.86-1.25  1.05  1.04
#   > 8192       <= 2^22         1.01-2.62  1.71 (22)
#
# - Only the first cell is a consistent gain: none of its 83 shapes is more
#   than 1% slower on trtllm-gen, and against the legacy wrapper's auto (cuDNN
#   on all of them) run() takes 0.83 of the time.  plan() is cheaper as well
#   (about 0.05 ms against 0.3-0.6 ms), which no other cell has to offer
#   either: one plan() and 32 run() calls take 0.84 of the legacy time there.
# - With longer KV the context kernel loses to cuDNN's split, up to 2.6x at
#   16384 keys, and with a longer request or more work the two are within
#   about 10% either way; cuDNN, the legacy wrapper's choice, keeps those.
# - In CUDA-graph mode the same cell is 0.70-1.48 (mean 0.95, 17 of 83 more
#   than 1% slower, the shapes of at most 64 rows), so graph plans keep cuDNN
#   first in the whole region.
_SM100_REGION_EAGER_MAX_KV = 4096
_SM100_REGION_EAGER_MAX_REQUEST_ROWS = 256
_SM100_REGION_EAGER_MAX_WORK = 1 << 22


def _attended_kv(facts: _PlanFacts) -> int:
    """Most keys one query row of the batch attends (``kv_eff`` above).

    The longest KV length of the batch, capped by the sliding window.  The
    length comes from the host mirror when the batch has one: ``max_kv_len``
    is a bound a caller may declare well above the lengths (a bucket, the
    table's capacity), and it stands in only for device-only lengths.
    """
    longest = facts.max_kv_len
    lengths = facts.host_lengths()
    if lengths is not None:
        longest = max(lengths[1])
    if facts.window_left >= 0:
        longest = min(longest, facts.window_left + 1)
    return longest


def ordered_sm100_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Order every candidate for SM100 from the plan facts, without deciding support.

    1. cuDNN first inside ``_in_sm100_cudnn_region``, the legacy wrapper's
       head-dim-128 admission, copied and not fitted here: at most four
       requests inside its prefix / full-prefill bounds from 8 Mi of work
       (``num_qo_heads * sum(q_i * kv_i)``), and from 16 Mi a batch of any
       size whose own lengths qualify.  The cuDNN version gate is one of its
       conditions.  One exception, fitted here and eager only: trtllm-gen
       goes ahead of cuDNN for a small kernel, which is at most 4096 keys per
       query row (``_attended_kv``), at most 256 rows in the longest request
       (``num_qo_heads * its query length / 128``) and ``rows * keys`` of at
       most 2^22.
    2. Otherwise trtllm-gen ahead of fa2 when ``rows`` reaches 64 (96 below
       head dim 128) at a ``fill`` of 0.375 in CUDA-graph mode, 0.75 eager.
       Eager and above 148 rows it also needs 1024 keys per query row
       (``_attended_kv``) or ``rows * keys`` of 2^19.
    3. Otherwise fa2 ahead of trtllm-gen.

    fa3 and, outside its region, cuDNN follow as the last resorts.
    """
    rows = facts.num_qo_heads * facts.total_q_tokens / 128
    ntg = facts.batch_size * facts.num_qo_heads * math.ceil(facts.max_q_len / 128)
    min_rows = (
        _SM100_MIN_ROWS if facts.head_dim_qk >= 128 else _SM100_MIN_ROWS_SMALL_HEAD_DIM
    )
    min_fill = _SM100_MIN_FILL_GRAPH if facts.use_cuda_graph else _SM100_MIN_FILL_EAGER
    trtllm_gen_first = rows >= min_rows and rows / ntg >= min_fill
    if (
        trtllm_gen_first
        and not facts.use_cuda_graph
        and rows > _SM100_EAGER_SHORT_KV_SPLIT_ROWS
    ):
        kv_eff = _attended_kv(facts)
        trtllm_gen_first = (
            kv_eff >= _SM100_EAGER_SHORT_KV
            or rows * kv_eff >= _SM100_EAGER_SHORT_KV_MIN_WORK
        )
    if trtllm_gen_first:
        order = _prefer("trtllm-gen", "fa2")
    else:
        order = _prefer("fa2", "trtllm-gen")
    # cuDNN is qualified only inside the legacy region; elsewhere it stays the
    # last resort behind both measured choices.
    if not _in_sm100_cudnn_region(facts):
        return order
    if not facts.use_cuda_graph:
        # The longest request's query: the batch's own when its lengths are on
        # the host, as for kv_eff; max_q_len is a bound that may be declared
        # above it.
        longest_q = facts.max_q_len
        lengths = facts.host_lengths()
        if lengths is not None:
            longest_q = max(lengths[0])
        kv_eff = _attended_kv(facts)
        if (
            kv_eff <= _SM100_REGION_EAGER_MAX_KV
            and facts.num_qo_heads * longest_q
            <= 128 * _SM100_REGION_EAGER_MAX_REQUEST_ROWS
            and rows * kv_eff <= _SM100_REGION_EAGER_MAX_WORK
        ):
            return ("trtllm-gen", "cudnn") + tuple(
                b for b in order if b not in ("trtllm-gen", "cudnn")
            )
    return ("cudnn",) + tuple(b for b in order if b != "cudnn")


def ordered_backends(
    compute_capability: Tuple[int, int], facts: _PlanFacts
) -> Tuple[str, ...]:
    """Every auto candidate in preference order for one plan().

    Only exact SM100 ``(10, 0)`` has a rule fitted to measurements
    (``ordered_sm100_backends``).  The other architectures are dispatched by
    compute-capability major to fixed orders: SM90 takes the legacy wrapper's
    short-query rule, the other SM10x parts the resolve-time order
    (``_selection.HEURISTIC_ORDER``) evaluated on the plan's ``max_q_len``
    in place of the hint, SM8x and SM12x fa2 then cuDNN whatever the shape.
    An unknown major gets the default order.
    """
    major, _minor = compute_capability
    if compute_capability == (10, 0):
        return ordered_sm100_backends(facts)
    if major == 10:
        return ordered_sm10x_backends(facts)
    if major == 9:
        return ordered_sm90_backends(facts)
    if major == 8:
        return ordered_sm80_backends(facts)
    if major == 12:
        return ordered_sm12x_backends(facts)
    return _AUTO_BACKEND_CANDIDATES


def plan_order(
    candidates: Tuple[str, ...],
    compute_capability: Tuple[int, int],
    facts: _PlanFacts,
) -> Tuple[str, ...]:
    """``candidates`` (a Resolution's pinned set) in the policy's order.

    The controller asks on every eager plan() and, in CUDA-graph mode, on
    each plan() until one succeeds and freezes the backend; it does not ask
    for a single candidate.  The set is never changed: a member the policy
    does not rank (a caller's own Resolution may carry one, ``cake`` for
    instance) keeps its place after the ranked ones.
    """
    order = ordered_backends(compute_capability, facts)
    return tuple(b for b in order if b in candidates) + tuple(
        b for b in candidates if b not in order
    )


__all__ = [
    "_AUTO_BACKEND_CANDIDATES",
    "_PlanFacts",
    "ordered_backends",
    "ordered_sm80_backends",
    "ordered_sm90_backends",
    "ordered_sm100_backends",
    "ordered_sm10x_backends",
    "ordered_sm12x_backends",
    "plan_order",
]
