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
        _expect(
            len(q_lens) == len(kv_lens) == self.batch_size,
            f"auto policy: {len(q_lens)} query spans and {len(kv_lens)} KV lengths "
            f"for a batch of {self.batch_size}",
        )
        _expect(
            min(q_lens) >= 0
            and max(q_lens) <= self.max_q_len
            and sum(q_lens) <= self.total_q_tokens,
            f"auto policy: query lengths {q_lens} do not fit max_q_len "
            f"{self.max_q_len} / total_q_tokens {self.total_q_tokens}",
        )
        _expect(
            min(kv_lens) >= 0 and max(kv_lens) <= self.max_kv_len,
            f"auto policy: KV lengths {kv_lens} do not fit max_kv_len {self.max_kv_len}",
        )
        return q_lens, kv_lens


def ordered_sm80_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA2 on SM80, with cuDNN as fallback."""
    return _prefer("fa2", "cudnn")


def ordered_sm90_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA3 on SM90, except FA2 for short non-FP8 queries.

    FA3 uses a fixed query tile; the legacy wrapper measured FA2 faster on H20
    when the longest query has at most 64 tokens.
    """
    if facts.max_q_len <= 64 and facts.q_dtype not in _FP8:
        return _prefer("fa2", "fa3", "cudnn")
    return _prefer("fa3", "fa2", "cudnn")


def ordered_sm10x_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Blackwell parts other than SM100: the unmeasured B200 order.

    The context kernel reads a request's whole KV for a handful of query
    tokens, so FA2's split-KV schedule leads at decode / speculative lengths
    (B200: 5.6x at q=1, 2.5x at q=16, crossover by q=256; see
    ``_selection.HEURISTIC_ORDER``).
    """
    if facts.max_q_len <= 16:
        return _prefer("fa2", "trtllm-gen", "cudnn")
    return _prefer("trtllm-gen", "cudnn", "fa2")


def ordered_sm12x_backends(facts: _PlanFacts) -> Tuple[str, ...]:
    """Prefer FA2 on SM120, with cuDNN as fallback."""
    return _prefer("fa2", "cudnn")


def _in_sm100_cudnn_region(facts: _PlanFacts) -> bool:
    """Whether cuDNN leads on SM100: the legacy wrapper's qualified domain.

    Duplicates ``flashinfer.prefill._blackwell_paged_auto_cudnn`` and the
    feature conditions around its call in
    ``BatchPrefillWithPagedKVCacheWrapper.plan()`` that have a counterpart
    here.  On eager facts the two are equal, and a test pins them against
    each other.

    CUDA-graph facts mix two sources, and the answer follows both:

    * ``max_q_len`` / ``max_kv_len`` / ``total_q_tokens`` are the graph
      capacity's.  The prefix and full-prefill bound tests and the cheap work
      bound read those.  A capacity cannot show ``max_q == max_kv`` for a
      full prefill whose length is not a page multiple, so the full-prefill
      bound test is ``1024 <= max_q <= max_kv <= 8192`` there.  That is wider
      than the legacy predicate, which is given the batch's own maxima; the
      same test checks that graph facts admit everything the legacy predicate
      admits and differ only by this widening.
    * ``qo_indptr`` / ``kv_seq_lens`` are the lengths of the batch being
      planned.  The exact work (``>= 8 Mi``) and the ``q_i == kv_i`` proof of
      full prefill read those.  A graph-mode instance ranks once, on its
      first plan, so they are the first batch's: two instances with one
      capacity can freeze different backends, and later batches are not
      checked against the region.

    Without host lengths only the prefix bounds can admit; full prefill is
    never assumed from the bounds.
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
    if not (prefix or full):
        return False
    if not (
        facts.kv_layout == "HND"
        and facts.q_dtype == facts.kv_dtype == torch.bfloat16
        and facts.head_dim_qk == facts.head_dim_vo == 128
        and 1 <= facts.batch_size <= 4
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
    # with device-only lengths keep the bound-based answer.
    lengths = facts.host_lengths()
    if lengths is None:
        return prefix
    q_lens, kv_lens = lengths
    # Maxima alone cannot distinguish full prefill from a mixed-prefix batch.
    # (A graph bucket can satisfy both bound tests; the prefix one needs no proof.)
    if not prefix and not (
        max(q_lens) >= 1024
        and all(q == kv and q > 0 for q, kv in zip(q_lens, kv_lens, strict=True))
    ):
        return False
    work = hq * sum(q * kv for q, kv in zip(q_lens, kv_lens, strict=True))
    return work >= 8 * 1024 * 1024


# SM100 rule, fitted on GB200 (cuDNN 9.27, cudnn-frontend 1.32) over 594
# cases (297 shapes, eager and CUDA graph): against the legacy wrapper's auto
# the geomean speedup is 1.56x in graph mode (one case more than 5% slower,
# worst 0.92) and 1.37x eager.
#
#   rows = num_qo_heads * total_q / 128              query rows, in 128-row tiles
#   ntg  = batch * num_qo_heads * ceil(max_q / 128)  the context kernel's CTA grid
#   fill = rows / ntg                                how full those tiles are
#
# run() medians in us; bf16, causal, page 16, GQA 32/8 unless noted:
#
#   mode   shape                         rows  fill   fa2  trtllm-gen  cudnn  first
#   graph  B1 q128  kv16384                32  1.00   146         225     99  cudnn (region)
#   graph  B1 q1024 kv32768 8/1            64  1.00   465         403    187  cudnn (region)
#   graph  B8 q48   kv2048                 96  0.375   88          61     76  trtllm-gen
#   graph  B6 q128  kv2048  16/4           96  1.00    74          36     47  trtllm-gen
#   graph  B1 q2048 kv2048  4/1            64  1.00    54          34     36  trtllm-gen
#   graph  B8 q32   kv2048                 64  0.25    55          60     76  fa2
#   graph  B16 q24  kv8192                 96  0.19   249         421    538  fa2
#   graph  B4 q32   kv8192                 32  0.25    72         107    173  fa2
#   graph  B2 q128  kv16384 d64            64  1.00   166         187      -  fa2
#   graph  B3 q128  kv16384 d64            96  1.00   230         190      -  trtllm-gen
#   eager  B8 q96   kv2048                192  0.75   151         103    153  trtllm-gen
#   eager  B16 q64  kv2048                256  0.50   142         165    182  fa2
#   eager  B16 q48  kv2048                192  0.375  144         163    185  fa2
#
# (the two region rows are the eager measurements; graph mode agrees.)  The
# thresholds are coarse crossovers, not exact optima: at rows 48 / fill 0.375
# trtllm-gen already leads by a few microseconds in graph mode.
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
    """Order every candidate for SM100 from the plan facts, without deciding support."""
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
    if _in_sm100_cudnn_region(facts):
        return ("cudnn",) + tuple(b for b in order if b != "cudnn")
    return order


def ordered_backends(
    compute_capability: Tuple[int, int], facts: _PlanFacts
) -> Tuple[str, ...]:
    """Every auto candidate in preference order for one plan().

    Only exact SM100 has a measured policy; the other architectures keep the
    order they had before plan-time selection.
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

    The set is never changed: a member the policy does not rank (a caller's
    own Resolution may carry one) keeps its place after the ranked ones.
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
