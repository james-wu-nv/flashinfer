"""Legacy -> unified: tests/attention/test_modular_fmha_prefill_paged.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and every row skips on the ``is_sm100a_supported`` gate, so the
modular cute-dsl paged kernel never executed in PR CI (reports/unified-
prefill-round4-20260918/ci-status.md).  On B200 (nvidia-cutlass-dsl 4.8) the
legacy file runs: 44 passed in 85 s (round-4 legacy run).

The modular cute-dsl prefill kernel (``flashinfer.cute_dsl.attention``, the
``cute-dsl`` backend of the legacy wrappers) is not a unified backend.  What
this file converts is the WORKLOAD of every legacy test: the legacy fixtures
(``build_paged`` / ``_build_wrapper_problem`` imported from the legacy module,
seeds 42 / 0, the same scrambled page pools, NaN pool fills and null blocks)
run in the flat CSR form on every unified backend that resolves (fa2,
trtllm-gen, cake, cuDNN -- excluded ones are recorded with the resolve
reason) and on ``auto`` (the serving backend is recorded).  The legacy
assertion is BITWISE equality with the ragged modular kernel on the same
logical problem; that is a backend-private property (``native-only``), so
each converted row asserts instead that the paged unified output is finite
(no NaN leaked from an unreferenced page or a null block), matches the fp32
attention over the RAGGED K/V (the legacy reference's logical problem) and
matches the fp32 oracle over the page pool, both at the suite budget.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- bitwise paged == ragged (every ``run_paged_vs_ragged`` row and the wrapper
  bitwise rows): native-only property of one kernel; tolerance-checked here.
- non-causal + sliding window (``test_paged_windowed[False-16-255]``,
  ``test_paged_null_block_window_clamp[False-16-127]``): no unified backend
  declares the combination (M20: the fa2 windowed KV range is trimmed as if
  causal; trtllm-gen / cake ship no such kernel; cuDNN has no window), so
  those rows skip with every resolve reason.  The modular kernel supports it
  natively -- a support-surface gap of the unified API.
- head_dim 64 (``test_paged_d64_windowed_null_block``): fa2 only.
- page_size 8: trtllm-gen / cake declare 16..1024; fa2 and cuDNN (dense
  derivation at page >= 8) run it.
- fp8 q (``test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper``): EXPECT_FP8_Q;
  the bf16-q / fp8-KV half of the fixture runs on fa2 (declared) against the
  oracle on the dequantized pool.
- mixed K / V dtypes (``test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged``,
  bf16 K with an fp8 V): the unified plan has ONE ``kv_dtype`` and ``run()``
  rejects a V of another dtype (EXPECT_MIXED_KV_DTYPE).
- attention variants (``test_paged_wrapper_variants_bitwise_vs_ragged``):
  ``sink`` maps to ``use_sinks`` / ``run(sinks=)``; ``sigmoid`` and ``alibi``
  (a logits transform and a position-dependent score mod) have no unified
  spelling (EXPECT_ATTENTION_VARIANTS).
- wrapper rejections (``test_paged_wrapper_rejections``): contract differences
  recorded, not gaps -- fa2 accepts page_size 48 (any page size) and a bf16 q
  with an fp8 KV (declared); a zero-length KV request is a legal padding row
  (design doc "Zero-row contract"); ``kv_cache_sf`` is not a ``run()`` kwarg
  (EXPECT_NVFP4_KV); K / V dtype mismatches at ``run()`` are rejected.
- ``FLASHINFER_VALIDATE_INPUTS`` NaN scan
  (``test_paged_wrapper_validate_inputs_nan_scan``): a cute-dsl wrapper
  feature; the unified API has no input scan (native-only); the row records
  that the env var has no effect and that fa2 masks the poisoned tail.
"""

import inspect
import math

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_ATTENTION_VARIANTS,
    EXPECT_FP8_Q,
    EXPECT_MIXED_KV_DTYPE,
    EXPECT_NVFP4_KV,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    csr_metadata_from_legacy,
    fp8_q_rejected,
    gated,
    oracle,
    run_on_backends,
)
from .paged_attention_reference import reference_paged_prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_modular_fmha_prefill_paged.py"


LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_bitwise_vs_ragged",
        ["test_paged_bitwise_vs_ragged"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; grid causal x page 8/16/64/128 x the two shape sets; page 8 "
        "excludes trtllm-gen / cake (declared 16..1024)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_identity_table",
        ["test_paged_identity_table"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; identity table (pure indirection)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_d64_windowed_null_block",
        ["test_paged_d64_windowed_null_block"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; D64 x window 127 x NaN null blocks below the window: fa2 is the "
        "only backend declaring D64; the null-block clamp holds (finite)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_nan_pool_tail_clamp",
        ["test_paged_nan_pool_tail_clamp"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; unreferenced pool pages NaN: no backend reads past a request's page "
        "count (finite on fa2, trtllm-gen, cake, cuDNN)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_windowed",
        ["test_paged_windowed"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; window 255 / 511; the non-causal + window row "
        "([1024]-[1024]-False-16-255) has no unified backend (M20) and skips with "
        "every resolve reason -- the modular kernel supports it natively",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_null_block_window_clamp",
        ["test_paged_null_block_window_clamp"],
        "partial",
        "same legacy fixture (build_paged: seed 42 tensors, scrambled pool with 5 "
        "slack pages, NHD, bf16, H8:2, D128) in the flat CSR form on every resolving "
        "unified backend and auto (recorded); the legacy BITWISE paged == ragged "
        "modular-kernel assertion is native-only -- asserted here: finite output, "
        "fp32 attention over the ragged K/V and the oracle over the pool at OUT_TOL / "
        "LSE_TOL; reclaimed pages below the window point at a NaN null page: the "
        "window clamp holds on every resolving backend (finite); the non-causal row "
        "([300]-[1500]-False-16-127) skips (M20)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper"],
        "partial",
        "same wrapper fixture (_build_wrapper_problem seed 42: [300, 1291] / [300, "
        "1547], page 16, stacked NHD cache as K = cache[:, 0] / V = cache[:, 1] "
        "views, causal, window 127) on every resolving backend and auto vs the ragged "
        "fp32 reference and the oracle; the bitwise ragged-wrapper comparison and the "
        "_cute_dsl_use_fmha routing assertion are native-only",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_hnd_bitwise_vs_nhd",
        ["test_paged_wrapper_hnd_bitwise_vs_nhd"],
        "partial",
        "same fixture in NHD and HND (cache.transpose(2, 3).contiguous()) on every "
        "resolving backend; HND vs NHD at 1e-2 (bitwise is native-only) and both vs "
        "the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_lse",
        ["test_paged_wrapper_lse"],
        "partial",
        "same fixture; lse_mode 'none' and 'base2' plans on every resolving backend: "
        "the outputs agree at 1e-3 (bitwise is native-only), the LSE is finite, "
        "(total_q, 8), fp32 and matches the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper"],
        "unsupported-by-design",
        "uniform fp8 q / K / V has no unified spelling (fp8 q: EXPECT_FP8_Q, asserted "
        "for page 8/16, window 511); the bf16-q / e4m3-KV half of the fixture runs on "
        "fa2 (declared) vs the oracle on the dequantized pool",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged",
        ["test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged"],
        "unsupported-by-design",
        "bf16 K with an fp8 V: the unified plan declares ONE kv_dtype and run() "
        "rejects a V of another dtype (EXPECT_MIXED_KV_DTYPE, asserted on fa2 with "
        "the legacy fixture, window 127)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_variants_bitwise_vs_ragged",
        ["test_paged_wrapper_variants_bitwise_vs_ragged"],
        "partial",
        "sink variant: the legacy linspace(-1, 1) sinks through use_sinks / "
        "run(sinks=) on every resolving backend vs the sink-aware oracle; sigmoid "
        "(logits transform) and alibi (position score-mod) have no unified spelling "
        "(EXPECT_ATTENTION_VARIANTS: plan() has no variant argument, asserted)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_rejections",
        ["test_paged_wrapper_rejections"],
        "partial",
        "unified analogs of the five legacy rejections: page_size 48 is accepted by "
        "fa2 (any page size) and rejected by trtllm-gen / cake (declared list); bf16 "
        "q + fp8 KV is a declared fa2 pair (contract difference); an fp8 K or fp32 V "
        "at run() is rejected against the planned kv_dtype; kv_cache_sf is not a "
        "run() kwarg (EXPECT_NVFP4_KV); a zero-length KV request is a legal padding "
        "row (contract difference: the other rows stay correct)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_validate_inputs_nan_scan",
        ["test_paged_wrapper_validate_inputs_nan_scan"],
        "native-only",
        "FLASHINFER_VALIDATE_INPUTS=1 is a cute-dsl wrapper NaN scan of the "
        "referenced pages; the unified API has no input scan: the row records that "
        "the env var has no effect on run() and that fa2 masks the poisoned in-page "
        "tail (EXPECT_INPAGE_TAIL_IGNORED documents the other backends)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


# ---------------------------------------------------------------------------
# the legacy fixtures (imported from the legacy module: they need the cute-dsl
# package the legacy file gates on; a missing package skips the row)
# ---------------------------------------------------------------------------


def _legacy():
    import tests.attention.test_modular_fmha_prefill_paged as legacy

    return legacy


def _cumsum32(lens):
    t = torch.zeros(len(lens) + 1, dtype=torch.int32)
    t[1:] = torch.cumsum(torch.tensor(lens), 0)
    return t


def _ragged_reference(q, k, v, qo_cpu, kv_cpu, *, causal, window_left, sinks=None):
    """fp32 attention over the RAGGED K/V (the legacy reference's logical
    problem): one page per token, identity page ids."""
    total_k = k.shape[0]
    return reference_paged_prefill(
        q,
        k.view(total_k, 1, k.shape[1], k.shape[2]),
        v.view(total_k, 1, v.shape[1], v.shape[2]),
        qo_cpu,
        kv_cpu,
        None,
        1,
        causal,
        window_left=window_left,
        kv_layout="NHD",
        kv_page_indices=torch.arange(total_k, dtype=torch.int32, device=q.device),
        sinks=sinks,
    )


def _unified_paged_vs_ragged(
    seq_lens_q,
    seq_lens_k,
    causal,
    page_size,
    record_property,
    scramble=True,
    pool_fill=777.0,
    window_left=-1,
    null_below_window=False,
    h_q=8,
    h_k=2,
    d=128,
    dt=torch.bfloat16,
):
    """``run_paged_vs_ragged`` through the unified API: the same tensors and
    pool, the flat CSR form, every resolving backend."""
    legacy = _legacy()
    qo_indptr = _cumsum32(seq_lens_q)
    kv_indptr = _cumsum32(seq_lens_k)
    s_q_all, s_k_all = int(qo_indptr[-1]), int(kv_indptr[-1])
    torch.manual_seed(42)
    q = torch.randn(s_q_all, h_q, d, dtype=dt, device=DEVICE)
    k = torch.randn(s_k_all, h_k, d, dtype=dt, device=DEVICE)
    v = torch.randn(s_k_all, h_k, d, dtype=dt, device=DEVICE)
    k_pool, v_pool, page_table, _page_indptr = legacy.build_paged(
        k,
        v,
        kv_indptr,
        page_size,
        scramble,
        pool_fill,
        null_window_left=(window_left if null_below_window else None),
        qo_indptr=qo_indptr,
    )
    kv_lens = torch.tensor(seq_lens_k, dtype=torch.int32)
    md = PagedAttentionMetadata.csr(
        qo_indptr.to(DEVICE),
        kv_lens.to(DEVICE),
        page_table.to(torch.int32),
        page_size=page_size,
        max_q_len=max(seq_lens_q),
        max_kv_len=max(seq_lens_k),
        qo_indptr_cpu=qo_indptr,
        kv_seq_lens_cpu=kv_lens,
    )
    results = run_on_backends(
        md,
        q,
        (k_pool, v_pool),
        num_qo_heads=h_q,
        num_kv_heads=h_k,
        head_dim_qk=d,
        q_dtype=dt,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        record_property=record_property,
    )
    ref, ref_lse = _ragged_reference(
        q, k, v, qo_indptr, kv_lens, causal=causal, window_left=window_left
    )
    # the oracle gathers only referenced pages / positions; NaN fills in
    # unreferenced pages and null blocks are outside every visible set, and
    # nan_to_num keeps the finite pool values unchanged
    o_out, o_lse = oracle(
        md,
        q,
        k_pool.nan_to_num(nan=0.0),
        v_pool.nan_to_num(nan=0.0),
        causal=causal,
        kv_layout="NHD",
        window_left=window_left,
    )
    for _name, _served, out, lse in results:
        assert not out.float().isnan().any().item(), "NaN leaked into paged output"
        torch.testing.assert_close(out.float(), ref, **OUT_TOL)
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    return results


SHAPES = [
    ([512, 512], [512, 512]),  # uniform, page-aligned
    ([7, 300, 1291], [23, 300, 1547]),  # varlen, partial last pages, s_k > s_q
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [8, 16, 64, 128])
@pytest.mark.parametrize("shapes", SHAPES)
def test_paged_bitwise_vs_ragged(causal, page_size, shapes, record_property):
    sq, sk = shapes
    _unified_paged_vs_ragged(sq, sk, causal, page_size, record_property)


def test_paged_identity_table(record_property):
    """Identity table (pages in pool order) -- isolates pure indirection."""
    _unified_paged_vs_ragged(
        [512, 512], [512, 512], True, 16, record_property, scramble=False
    )


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_d64_windowed_null_block(page_size, record_property):
    """head_dim 64 x window clamp x NaN null blocks (fa2 is the only backend
    declaring D64)."""
    results = _unified_paged_vs_ragged(
        [1291],
        [1547],
        True,
        page_size,
        record_property,
        pool_fill=float("nan"),
        window_left=127,
        null_below_window=True,
        d=64,
    )
    assert all(served == "fa2" for _n, served, _o, _l in results)


@pytest.mark.parametrize(
    "causal,page_size",
    [(True, 8), (True, 16), (False, 64)],
)
def test_paged_nan_pool_tail_clamp(causal, page_size, record_property):
    """Unreferenced pool pages NaN: no backend may read past a request's
    page count."""
    _unified_paged_vs_ragged(
        [7, 300, 1291],
        [23, 300, 1547],
        causal,
        page_size,
        record_property,
        pool_fill=float("nan"),
    )


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([1024], [1024], False, 16, 255),
        ([1024], [1024], True, 8, 255),
        ([1291], [1547], True, 64, 511),
    ],
)
def test_paged_windowed(sq, sk, causal, page_size, window_left, record_property):
    """Windowed paged (all pages live); the non-causal + window row has no
    unified backend (M20) and skips with the resolve reasons."""
    _unified_paged_vs_ragged(
        sq, sk, causal, page_size, record_property, window_left=window_left
    )


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([300], [1500], False, 16, 127),
        ([1291], [1547], True, 64, 511),
        ([7, 300, 1291], [23, 300, 1547], True, 16, 127),
        ([7, 300, 1291], [23, 300, 1547], True, 8, 127),
    ],
)
def test_paged_null_block_window_clamp(
    sq, sk, causal, page_size, window_left, record_property
):
    """Null-block contract: reclaimed out-of-window table slots point at a
    NaN null page; no backend may read them."""
    _unified_paged_vs_ragged(
        sq,
        sk,
        causal,
        page_size,
        record_property,
        pool_fill=float("nan"),
        window_left=window_left,
        null_below_window=True,
    )


# ---------------------------------------------------------------------------
#  Wrapper-level rows (the legacy BatchPrefillWithPagedKVCacheWrapper cute-dsl
#  fixture _build_wrapper_problem)
# ---------------------------------------------------------------------------


def _wrapper_problem(seq_lens_q, seq_lens_k, page_size, **kw):
    legacy = _legacy()
    (q, k_rag, v_rag, cache, qo, kv_tok, kv_pg, ids, lpl) = (
        legacy._build_wrapper_problem(seq_lens_q, seq_lens_k, page_size, **kw)
    )
    md = csr_metadata_from_legacy(qo, kv_pg, ids, lpl, page_size)
    return dict(
        q=q,
        k_rag=k_rag,
        v_rag=v_rag,
        cache=cache,  # (pool, 2, page_size, h_k, d) NHD
        qo=qo,
        kv_tok=kv_tok,
        kv_pg=kv_pg,
        ids=ids,
        lpl=lpl,
        md=md,
        page_size=page_size,
        h_q=kw.get("h_q", 8),
        h_k=kw.get("h_k", 2),
        d=kw.get("d", 128),
        dt=kw.get("dt", torch.bfloat16),
    )


def _wrapper_common(w, **overrides):
    common = dict(
        num_qo_heads=w["h_q"],
        num_kv_heads=w["h_k"],
        head_dim_qk=w["d"],
        q_dtype=w["dt"],
        kv_layout="NHD",
        causal=True,
    )
    common.update(overrides)
    return common


def _check_wrapper_results(
    w, results, *, causal=True, window_left=-1, k=None, v=None, sinks=None
):
    k = w["cache"][:, 0] if k is None else k
    v = w["cache"][:, 1] if v is None else v
    ref, _ = _ragged_reference(
        w["q"],
        w["k_rag"],
        w["v_rag"],
        w["qo"].cpu(),
        w["md"].kv_seq_lens_cpu,
        causal=causal,
        window_left=window_left,
        sinks=sinks,
    )
    o_out, o_lse = oracle(
        w["md"],
        w["q"],
        k,
        v,
        causal=causal,
        kv_layout="NHD",
        window_left=window_left,
        sinks=sinks,
    )
    for _name, _served, out, lse in results:
        assert torch.isfinite(out.float()).all()
        torch.testing.assert_close(out.float(), ref, **OUT_TOL)
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        if lse is not None:
            torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    return o_out, o_lse


def test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper(record_property):
    """The legacy windowed plan (causal, window 127) on every resolving
    backend; the bitwise ragged-wrapper comparison is native-only."""
    w = _wrapper_problem([300, 1291], [300, 1547], 16)
    results = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        window_left=127,
        record_property=record_property,
        **_wrapper_common(w),
    )
    _check_wrapper_results(w, results, window_left=127)


def test_paged_wrapper_hnd_bitwise_vs_nhd(record_property):
    w = _wrapper_problem([300, 1291], [300, 1547], 16)
    nhd = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        record_property=record_property,
        **_wrapper_common(w),
    )
    _check_wrapper_results(w, nhd)
    cache_hnd = w["cache"].transpose(2, 3).contiguous()  # (pool, 2, h_k, page, d)
    hnd = run_on_backends(
        w["md"],
        w["q"],
        (cache_hnd[:, 0], cache_hnd[:, 1]),
        include_auto=False,
        **_wrapper_common(w, kv_layout="HND"),
    )
    o_out, o_lse = oracle(
        w["md"], w["q"], cache_hnd[:, 0], cache_hnd[:, 1], causal=True, kv_layout="HND"
    )
    by_name = {name: out for name, _s, out, _l in nhd}
    for name, _served, out, lse in hnd:
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
        if name in by_name:  # HND vs NHD on the same backend (bitwise is native-only)
            torch.testing.assert_close(
                out.float(), by_name[name].float(), rtol=1e-2, atol=1e-2
            )


def test_paged_wrapper_lse(record_property):
    w = _wrapper_problem([300, 1291], [300, 1547], 16)
    with_lse = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        record_property=record_property,
        **_wrapper_common(w),
    )
    _check_wrapper_results(w, with_lse)
    without = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        lse_mode="none",
        include_auto=False,
        **_wrapper_common(w),
    )
    by_name = {name: out for name, _s, out, _l in with_lse}
    for name, _served, out, lse in without:
        assert lse is None
        if name in by_name:
            torch.testing.assert_close(
                out.float(), by_name[name].float(), rtol=1e-3, atol=1e-3
            )
    for _name, _served, _out, lse in with_lse:
        assert lse.shape == (w["q"].shape[0], 8) and lse.dtype == torch.float32
        assert torch.isfinite(lse).all().item()


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper(page_size, record_property):
    """Uniform fp8 q / K / V: fp8 q is rejected (EXPECT_FP8_Q); the bf16-q /
    fp8-KV half of the same fixture runs on fa2 against the oracle on the
    dequantized pool."""
    dt = torch.float8_e4m3fn
    res = fp8_q_rejected(
        backend="fa2",
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=page_size,
        kv_layout="NHD",
        causal=True,
        window_left=511,
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: run the legacy fp8 q here")
    assert not EXPECT_FP8_Q
    w = _wrapper_problem([300, 1291], [300, 1547], page_size, dt=torch.bfloat16)
    cache8 = w["cache"].to(dt)
    results = run_on_backends(
        w["md"],
        w["q"],
        (cache8[:, 0], cache8[:, 1]),
        kv_dtype=dt,
        window_left=511,
        backends=("fa2",),
        record_property=record_property,
        **_wrapper_common(w),
    )
    deq = cache8.to(torch.bfloat16)
    o_out, o_lse = oracle(
        w["md"],
        w["q"],
        deq[:, 0],
        deq[:, 1],
        causal=True,
        kv_layout="NHD",
        window_left=511,
    )
    for _name, _served, out, lse in results:
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged(page_size):
    """bf16 Q/K with an fp8 V cache: one planned kv_dtype, so run() rejects
    the V (EXPECT_MIXED_KV_DTYPE)."""
    w = _wrapper_problem([300, 1291], [300, 1547], page_size)
    res = resolve_paged_attention(
        device=torch.device(DEVICE),
        page_size=page_size,
        need_lse=False,
        window_left=127,
        kv_input_form="page_indices",
        backend="fa2",
        **_wrapper_common(w),
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(w["md"], window_left=127, backend=res, **_wrapper_common(w))
    out = gated(
        EXPECT_MIXED_KV_DTYPE,
        lambda: attn.run(
            w["q"], (w["cache"][:, 0], w["cache"][:, 1].to(torch.float8_e4m3fn))
        ),
        match="v_cache dtype",
    )
    if out is not None:
        pytest.fail("EXPECT_MIXED_KV_DTYPE flipped: assert the mixed-V result here")


@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("variant_name", ["sigmoid", "alibi", "sink"])
def test_paged_wrapper_variants_bitwise_vs_ragged(
    variant_name, page_size, record_property
):
    """sink -> use_sinks / run(sinks=); sigmoid / alibi have no unified
    spelling (EXPECT_ATTENTION_VARIANTS)."""
    plan_params = inspect.signature(PagedAttention.plan).parameters
    assert ("variant" in plan_params) == EXPECT_ATTENTION_VARIANTS
    if variant_name != "sink":
        if EXPECT_ATTENTION_VARIANTS:
            pytest.fail(
                f"EXPECT_ATTENTION_VARIANTS flipped: port the {variant_name} row"
            )
        return
    w = _wrapper_problem([300, 1291], [300, 1547], page_size)
    sinks = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32, device=DEVICE)
    results = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        sinks=sinks,
        record_property=record_property,
        **_wrapper_common(w),
    )
    _check_wrapper_results(w, results, sinks=sinks)


def test_paged_wrapper_rejections():
    """The unified analogs of the legacy wrapper rejections (contract
    differences recorded in the LEGACY_MAP note)."""
    w = _wrapper_problem([300], [300], 16)
    common = _wrapper_common(w)
    dev = torch.device(DEVICE)
    # page_size 48: fa2 runs any page size; trtllm-gen / cake declare 16..1024
    resolve_paged_attention(
        device=dev, page_size=48, kv_input_form="page_indices", backend="fa2", **common
    )
    for backend in ("trtllm-gen", "cake"):
        with pytest.raises(ValueError, match="unsupported page_size 48"):
            resolve_paged_attention(device=dev, page_size=48, backend=backend, **common)
    # bf16 q with an fp8 KV is a declared fa2 pair (the legacy plan rejects it)
    res = resolve_paged_attention(
        device=dev,
        page_size=16,
        kv_input_form="page_indices",
        kv_dtype=torch.float8_e4m3fn,
        backend="fa2",
        **common,
    )
    assert res.backends == ("fa2",)
    # K of another dtype at run (plan bf16): rejected against the planned kv_dtype
    res = resolve_paged_attention(
        device=dev, page_size=16, kv_input_form="page_indices", backend="fa2", **common
    )
    attn = PagedAttention(dev)
    attn.plan(w["md"], backend=res, **common)
    k_c, v_c = w["cache"][:, 0].to(torch.float8_e4m3fn), w["cache"][:, 1]
    with pytest.raises(ValueError, match="k_cache dtype"):
        attn.run(w["q"], (k_c, v_c))
    # unsupported V dtype at run
    with pytest.raises(ValueError, match="v_cache dtype"):
        attn.run(w["q"], (w["cache"][:, 0], w["cache"][:, 1].to(torch.float32)))
    # kv_cache_sf: not a run() kwarg (EXPECT_NVFP4_KV)
    run_params = inspect.signature(PagedAttention.run).parameters
    assert ("kv_cache_sf" in run_params) == EXPECT_NVFP4_KV
    if not EXPECT_NVFP4_KV:
        with pytest.raises(TypeError, match="kv_cache_sf"):
            attn.run(
                w["q"],
                (w["cache"][:, 0], w["cache"][:, 1]),
                kv_cache_sf=torch.zeros(1, device=DEVICE),
            )
    # zero-length KV item: a legal padding row in the unified contract (the
    # legacy wrapper rejects it); the live row stays correct
    qo_e = torch.tensor([0, 300, 316], dtype=torch.int32)
    kv_pg_e = torch.cat([w["kv_pg"], w["kv_pg"][-1:]])
    lpl_e = torch.cat([w["lpl"], torch.zeros_like(w["lpl"][:1])])
    q_e = torch.cat([w["q"], torch.randn(16, 8, 128, dtype=w["dt"], device=DEVICE)])
    md_e = csr_metadata_from_legacy(qo_e, kv_pg_e, w["ids"], lpl_e, 16)
    assert md_e.kv_seq_lens_cpu.tolist() == [300, 0]
    attn_e = PagedAttention(dev)
    attn_e.plan(
        md_e,
        backend="fa2",
        causal=False,
        **{k: v for k, v in common.items() if k != "causal"},
    )
    out, _ = attn_e.run(q_e, (w["cache"][:, 0], w["cache"][:, 1]))
    ref, _ = _ragged_reference(
        w["q"],
        w["k_rag"],
        w["v_rag"],
        w["qo"].cpu(),
        w["md"].kv_seq_lens_cpu,
        causal=False,
        window_left=-1,
    )
    torch.testing.assert_close(out[:300].float(), ref, **OUT_TOL)


def test_paged_wrapper_validate_inputs_nan_scan(monkeypatch, record_property):
    """FLASHINFER_VALIDATE_INPUTS is a cute-dsl wrapper NaN scan; the unified
    API has no input scan (native-only): the env var has no effect and fa2
    masks the poisoned in-page tail past last_page_len."""
    w = _wrapper_problem([300], [300], 16)
    # poison the tail of the last referenced page (past last_page_len)
    w["cache"][int(w["ids"][-1]), 1, -1] = float("nan")
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    results = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        backends=("fa2",),
        include_auto=False,
        record_property=record_property,
        **_wrapper_common(w),
    )
    for _name, _served, out, _lse in results:
        assert out.shape == w["q"].shape
        assert torch.isfinite(out.float()).all()  # fa2 masks the tail
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "0")
    results0 = run_on_backends(
        w["md"],
        w["q"],
        (w["cache"][:, 0], w["cache"][:, 1]),
        backends=("fa2",),
        include_auto=False,
        **_wrapper_common(w),
    )
    torch.testing.assert_close(results0[0][2], results[0][2], rtol=0, atol=0)
    assert math.isfinite(float(results0[0][2].float().abs().max()))
