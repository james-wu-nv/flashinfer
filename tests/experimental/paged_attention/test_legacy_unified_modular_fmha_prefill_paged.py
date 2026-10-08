"""Legacy -> unified: tests/attention/test_modular_fmha_prefill_paged.py

The legacy file tests the paged mode of the modular cute-dsl prefill kernel
(``flashinfer.cute_dsl.attention``).  That kernel is not a unified backend, so
every converted test runs the paged half on ``PagedAttention`` pinned to fa2
AS A SUBSTITUTE: fa2 is the one unified backend that serves the whole legacy
geometry (page sizes 8 / 48, head_dim 64, NHD and HND).  The fixtures are the
legacy ones (``build_paged`` / ``_build_wrapper_problem``, same seeds, same
scrambled pools, NaN fills and null blocks), so the node ids equal the legacy
ids.  The legacy reference is still the ragged modular kernel on the same
logical problem; the legacy BITWISE paged == ragged assertion only holds
within one kernel, so the substitute is compared at a tolerance, and against
the fp32 paged-attention oracle (output and LSE).

Not expressible on the unified API (asserted as rejections): non-causal +
sliding window on fa2 (M20), an fp8 q, a V cache of another dtype than K, and
the sigmoid / ALiBi attention variants.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import is_sm100a_supported

from .paged_attention_reference import reference_paged_prefill

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

from flashinfer.cute_dsl.attention import (  # noqa: E402
    ALiBiAttention,
    AttentionWithSink,
    BatchPrefillCuteDSLWrapper,
    SigmoidAttention,
)
from tests.attention.test_modular_fmha_prefill_paged import (  # noqa: E402
    _build_wrapper_problem,
    build_paged,
)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_modular_fmha_prefill_paged.py"
LEGACY_MAP = [
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_bitwise_vs_ragged",
        ["test_paged_bitwise_vs_ragged"],
        "partial",
        "fa2 substitutes for the modular cute-dsl kernel; same fixture and grid; "
        "legacy ragged modular reference at OUT_TOL (bitwise only holds within one "
        "kernel) plus the fp32 oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_identity_table",
        ["test_paged_identity_table"],
        "partial",
        "fa2 substitute; identity page table; ragged modular reference at OUT_TOL "
        "plus the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_d64_windowed_null_block",
        ["test_paged_d64_windowed_null_block"],
        "partial",
        "fa2 substitute; D64, window 127, NaN null blocks below the window: output "
        "finite, ragged modular reference at OUT_TOL plus the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_nan_pool_tail_clamp",
        ["test_paged_nan_pool_tail_clamp"],
        "partial",
        "fa2 substitute; NaN unreferenced pool pages: output finite, ragged modular "
        "reference at OUT_TOL plus the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_windowed",
        ["test_paged_windowed"],
        "partial",
        "fa2 substitute; causal windowed rows vs the ragged modular reference and the "
        "oracle; the non-causal + window row asserts the fa2 rejection (M20)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_null_block_window_clamp",
        ["test_paged_null_block_window_clamp"],
        "partial",
        "fa2 substitute; NaN null blocks below the window: output finite, ragged "
        "modular reference plus the oracle; the non-causal row asserts the fa2 "
        "rejection (M20)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper"],
        "partial",
        "fa2 substitute; causal window 127 vs the ragged modular wrapper at OUT_TOL "
        "(bitwise only within one kernel) plus the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_hnd_bitwise_vs_nhd",
        ["test_paged_wrapper_hnd_bitwise_vs_nhd"],
        "partial",
        "fa2 substitute; HND bitwise equal to NHD (the legacy assertion) plus the "
        "oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_lse",
        ["test_paged_wrapper_lse"],
        "partial",
        "fa2 substitute; LSE is a plan-time choice: lse_mode none vs base2 outputs "
        "bitwise equal, LSE finite with the legacy shape, plus the oracle",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper"],
        "unsupported-by-design",
        "fp8 q has no unified backend: plan() rejects it",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged",
        ["test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged"],
        "unsupported-by-design",
        "the plan has one kv_dtype: run() rejects an fp8 V with a bf16 K",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_variants_bitwise_vs_ragged",
        ["test_paged_wrapper_variants_bitwise_vs_ragged"],
        "partial",
        "fa2 substitute; sink runs through use_sinks / run(sinks=) vs the ragged "
        "modular AttentionWithSink reference and the oracle; sigmoid and alibi have "
        "no unified spelling (plan() rejects variant=)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_rejections",
        ["test_paged_wrapper_rejections"],
        "partial",
        "fa2 substitute; an fp8 K / fp32 V at run() and kv_cache_sf are rejected; "
        "page_size 48, bf16 q + fp8 KV and a zero-length KV request are accepted "
        "(contract differences; the zero-length batch's live row matches the oracle)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_validate_inputs_nan_scan",
        ["test_paged_wrapper_validate_inputs_nan_scan"],
        "native-only",
        "FLASHINFER_VALIDATE_INPUTS is a cute-dsl wrapper NaN scan; PagedAttention "
        "has none: with the env var set, fa2 masks the poisoned in-page tail and "
        "matches the oracle",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

# the legacy wrapper fixture's model shape
GEOMETRY = dict(num_qo_heads=8, num_kv_heads=2, head_dim_qk=128, q_dtype=torch.bfloat16)


def _skip_unless_sm100():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")


def _csr_metadata(seq_lens_q, seq_lens_k, page_ids, page_size):
    qo_indptr = torch.zeros(len(seq_lens_q) + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_q), 0)
    kv_lens = torch.tensor(seq_lens_k, dtype=torch.int32)
    return PagedAttentionMetadata.csr(
        qo_indptr.cuda(),
        kv_lens.cuda(),
        page_ids.to(torch.int32),
        page_size=page_size,
        max_q_len=max(seq_lens_q),
        max_kv_len=max(seq_lens_k),
        qo_indptr_cpu=qo_indptr,
        kv_seq_lens_cpu=kv_lens,
    )


def _modular_ragged(q, k, v, qo_indptr, kv_indptr, causal, window_left, variant=None):
    """The legacy reference: the ragged modular cute-dsl kernel."""
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    w = BatchPrefillCuteDSLWrapper(ws)
    w.plan(
        qo_indptr.cuda(),
        kv_indptr.cuda(),
        num_qo_heads=q.shape[1],
        num_kv_heads=k.shape[1],
        head_dim_qk=q.shape[2],
        causal=causal,
        window_left=window_left,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        variant=variant,
    )
    return w.run(q, k, v)


def _assert_matches_oracle(
    out,
    lse,
    md,
    q,
    k_cache,
    v_cache,
    *,
    causal,
    window_left=-1,
    kv_layout="NHD",
    sinks=None,
):
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        None,
        md.page_size,
        causal,
        window_left=window_left,
        kv_layout=kv_layout,
        kv_page_indices=md.kv_page_indices,
        sinks=sinks,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _run_paged_vs_ragged(
    seq_lens_q,
    seq_lens_k,
    causal,
    page_size,
    scramble=True,
    pool_fill=777.0,
    window_left=-1,
    null_below_window=False,
    h_q=8,
    h_k=2,
    d=128,
    dt=torch.bfloat16,
):
    """The legacy ``run_paged_vs_ragged`` with the paged half on fa2."""
    device = "cuda"
    B = len(seq_lens_q)
    qo_indptr = torch.zeros(B + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_q), 0)
    kv_indptr = torch.zeros(B + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens_k), 0)
    s_q_all, s_k_all = int(qo_indptr[-1]), int(kv_indptr[-1])

    torch.manual_seed(42)
    q = torch.randn(s_q_all, h_q, d, dtype=dt, device=device)
    k = torch.randn(s_k_all, h_k, d, dtype=dt, device=device)
    v = torch.randn(s_k_all, h_k, d, dtype=dt, device=device)

    k_pool, v_pool, page_table, _ = build_paged(
        k,
        v,
        kv_indptr,
        page_size,
        scramble,
        pool_fill,
        null_window_left=(window_left if null_below_window else None),
        qo_indptr=qo_indptr,
    )
    md = _csr_metadata(seq_lens_q, seq_lens_k, page_table, page_size)

    attn = PagedAttention(torch.device(device))
    attn.plan(
        md,
        num_qo_heads=h_q,
        num_kv_heads=h_k,
        head_dim_qk=d,
        q_dtype=dt,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (k_pool, v_pool))
    assert not out.float().isnan().any().item(), "NaN leaked into paged output"

    ref = _modular_ragged(q, k, v, qo_indptr, kv_indptr, causal, window_left)
    torch.testing.assert_close(out, ref, **OUT_TOL)

    # NaN only fills pages outside every visible set (unreferenced pages, null
    # blocks below the window); zero them so the oracle's 0-weight gather of a
    # whole page stays finite
    _assert_matches_oracle(
        out,
        lse,
        md,
        q,
        k_pool.nan_to_num(nan=0.0),
        v_pool.nan_to_num(nan=0.0),
        causal=causal,
        window_left=window_left,
    )


SHAPES = [
    ([512, 512], [512, 512]),  # uniform, page-aligned
    ([7, 300, 1291], [23, 300, 1547]),  # varlen, partial last pages, s_k > s_q
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [8, 16, 64, 128])
@pytest.mark.parametrize("shapes", SHAPES)
def test_paged_bitwise_vs_ragged(causal, page_size, shapes):
    _skip_unless_sm100()
    sq, sk = shapes
    _run_paged_vs_ragged(sq, sk, causal, page_size)


def test_paged_identity_table():
    """Identity table (pages in pool order) -- isolates pure indirection."""
    _skip_unless_sm100()
    _run_paged_vs_ragged([512, 512], [512, 512], True, 16, scramble=False)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_d64_windowed_null_block(page_size):
    """head_dim 64 x window clamp x NaN null blocks."""
    _skip_unless_sm100()
    _run_paged_vs_ragged(
        [1291],
        [1547],
        True,
        page_size,
        pool_fill=float("nan"),
        window_left=127,
        null_below_window=True,
        d=64,
    )


@pytest.mark.parametrize(
    "causal,page_size",
    [(True, 8), (True, 16), (False, 64)],
)
def test_paged_nan_pool_tail_clamp(causal, page_size):
    """Unreferenced pool pages NaN: they must stay out of the output."""
    _skip_unless_sm100()
    _run_paged_vs_ragged(
        [7, 300, 1291], [23, 300, 1547], causal, page_size, pool_fill=float("nan")
    )


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([1024], [1024], False, 16, 255),
        ([1024], [1024], True, 8, 255),
        ([1291], [1547], True, 64, 511),
    ],
)
def test_paged_windowed(sq, sk, causal, page_size, window_left):
    """Windowed paged vs windowed ragged (all pages live)."""
    _skip_unless_sm100()
    if not causal:
        # fa2 trims the windowed KV range as if causal (M20): declined at plan
        with pytest.raises(ValueError, match="non-causal attention not supported"):
            _run_paged_vs_ragged(sq, sk, causal, page_size, window_left=window_left)
        return
    _run_paged_vs_ragged(sq, sk, causal, page_size, window_left=window_left)


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([300], [1500], False, 16, 127),
        ([1291], [1547], True, 64, 511),
        ([7, 300, 1291], [23, 300, 1547], True, 16, 127),
        ([7, 300, 1291], [23, 300, 1547], True, 8, 127),
    ],
)
def test_paged_null_block_window_clamp(sq, sk, causal, page_size, window_left):
    """Null-block contract: reclaimed out-of-window table slots point at a
    NaN null page that must never be read."""
    _skip_unless_sm100()
    kwargs = dict(
        pool_fill=float("nan"), window_left=window_left, null_below_window=True
    )
    if not causal:
        # fa2 trims the windowed KV range as if causal (M20): declined at plan
        with pytest.raises(ValueError, match="non-causal attention not supported"):
            _run_paged_vs_ragged(sq, sk, causal, page_size, **kwargs)
        return
    _run_paged_vs_ragged(sq, sk, causal, page_size, **kwargs)


# ---------------------------------------------------------------------------
#  Wrapper-level rows (the legacy _build_wrapper_problem fixture; its stacked
#  NHD cache (pool, 2, page, h_k, d) is passed as the K / V views)
# ---------------------------------------------------------------------------


def test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper():
    _skip_unless_sm100()
    (q, k_rag, v_rag, cache, qo, kv_tok, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, 16)
    attn = PagedAttention(q.device)
    attn.plan(
        md,
        **GEOMETRY,
        kv_layout="NHD",
        causal=True,
        window_left=127,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out_paged, lse = attn.run(q, (cache[:, 0], cache[:, 1]))

    out_rag = _modular_ragged(q, k_rag, v_rag, qo, kv_tok, True, 127)
    torch.testing.assert_close(out_paged, out_rag, **OUT_TOL)
    _assert_matches_oracle(
        out_paged, lse, md, q, cache[:, 0], cache[:, 1], causal=True, window_left=127
    )


def test_paged_wrapper_hnd_bitwise_vs_nhd():
    _skip_unless_sm100()
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, 16)
    plan_kw = dict(**GEOMETRY, causal=True, lse_mode="base2", backend="fa2")

    w_nhd = PagedAttention(q.device)
    w_nhd.plan(md, kv_layout="NHD", **plan_kw)
    assert w_nhd.backend == "fa2"
    out_nhd, lse = w_nhd.run(q, (cache[:, 0], cache[:, 1]))

    w_hnd = PagedAttention(q.device)
    w_hnd.plan(md, kv_layout="HND", **plan_kw)
    assert w_hnd.backend == "fa2"
    cache_hnd = cache.transpose(2, 3).contiguous()
    out_hnd, _ = w_hnd.run(q, (cache_hnd[:, 0], cache_hnd[:, 1]))
    assert torch.equal(out_hnd.view(torch.int16), out_nhd.view(torch.int16))

    _assert_matches_oracle(out_nhd, lse, md, q, cache[:, 0], cache[:, 1], causal=True)


def test_paged_wrapper_lse():
    _skip_unless_sm100()
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], 16
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, 16)
    plan_kw = dict(**GEOMETRY, kv_layout="NHD", causal=True, backend="fa2")

    # the LSE is a plan-time choice here (a run-time flag in the legacy wrapper)
    w = PagedAttention(q.device)
    w.plan(md, lse_mode="none", **plan_kw)
    assert w.backend == "fa2"
    out, _ = w.run(q, (cache[:, 0], cache[:, 1]))

    w_lse = PagedAttention(q.device)
    w_lse.plan(md, lse_mode="base2", **plan_kw)
    assert w_lse.backend == "fa2"
    out_lse, lse = w_lse.run(q, (cache[:, 0], cache[:, 1]))

    assert torch.equal(out_lse.view(torch.int16), out.view(torch.int16))
    assert lse.shape == (q.shape[0], 8) and torch.isfinite(lse).all().item()
    _assert_matches_oracle(out_lse, lse, md, q, cache[:, 0], cache[:, 1], causal=True)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper(page_size):
    """Uniform fp8 q / K / V: no unified backend takes an fp8 q."""
    _skip_unless_sm100()
    dt = torch.float8_e4m3fn
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size, dt=torch.bfloat16
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, page_size)
    with pytest.raises(ValueError, match="unsupported q dtype"):
        PagedAttention(q.device).plan(
            md,
            **{**GEOMETRY, "q_dtype": dt},
            kv_dtype=dt,
            kv_layout="NHD",
            causal=True,
            window_left=511,
            backend="fa2",
        )


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged(page_size):
    """bf16 Q/K with an fp8 V cache: the plan has one kv_dtype."""
    _skip_unless_sm100()
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, page_size)
    attn = PagedAttention(q.device)
    attn.plan(
        md, **GEOMETRY, kv_layout="NHD", causal=True, window_left=127, backend="fa2"
    )
    assert attn.backend == "fa2"
    with pytest.raises(ValueError, match="v_cache dtype"):
        attn.run(q, (cache[:, 0], cache[:, 1].to(torch.float8_e4m3fn)))


@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("variant_name", ["sigmoid", "alibi", "sink"])
def test_paged_wrapper_variants_bitwise_vs_ragged(variant_name, page_size):
    """sink maps to use_sinks / run(sinks=); sigmoid (a logits transform) and
    ALiBi (a position score-mod) have no unified spelling."""
    _skip_unless_sm100()
    (q, k_rag, v_rag, cache, qo, kv_tok, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300, 1291], [300, 1547], page_size
    )
    md = _csr_metadata([300, 1291], [300, 1547], ids, page_size)
    h_q = 8
    plan_kw = dict(**GEOMETRY, kv_layout="NHD", causal=True, backend="fa2")

    if variant_name == "sigmoid":
        variant = SigmoidAttention(scale=1.0 / 128**0.5, bias=-2.0)
    elif variant_name == "alibi":
        variant = ALiBiAttention(
            torch.linspace(0.1, 0.9, h_q, dtype=torch.float32, device="cuda")
        )
    if variant_name != "sink":
        with pytest.raises(TypeError, match="variant"):
            PagedAttention(q.device).plan(md, variant=variant, **plan_kw)
        return

    sinks = torch.linspace(-1.0, 1.0, h_q, dtype=torch.float32, device="cuda")
    attn = PagedAttention(q.device)
    attn.plan(md, lse_mode="base2", use_sinks=True, **plan_kw)
    assert attn.backend == "fa2"
    out_paged, lse = attn.run(q, (cache[:, 0], cache[:, 1]), sinks=sinks)

    out_rag = _modular_ragged(
        q, k_rag, v_rag, qo, kv_tok, True, -1, variant=AttentionWithSink(sinks)
    )
    torch.testing.assert_close(out_paged, out_rag, **OUT_TOL)
    _assert_matches_oracle(
        out_paged, lse, md, q, cache[:, 0], cache[:, 1], causal=True, sinks=sinks
    )


def test_paged_wrapper_rejections():
    _skip_unless_sm100()
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300], [300], 16
    )
    md = _csr_metadata([300], [300], ids, 16)
    plan_kw = dict(**GEOMETRY, kv_layout="NHD", causal=True, backend="fa2")

    # page size 48: fa2 serves any page size (the legacy wrapper rejects it)
    md48 = _csr_metadata([300], [300], ids[:7], 48)
    assert PagedAttention(q.device).plan(md48, **plan_kw).backend == "fa2"
    # bf16 q with an fp8 KV: a declared fa2 pair (the legacy wrapper rejects it)
    attn = PagedAttention(q.device).plan(md, kv_dtype=torch.float8_e4m3fn, **plan_kw)
    assert attn.backend == "fa2"

    attn = PagedAttention(q.device).plan(md, lse_mode="base2", **plan_kw)
    assert attn.backend == "fa2"
    # K of another dtype than planned
    with pytest.raises(ValueError, match="k_cache dtype"):
        attn.run(q, (cache[:, 0].to(torch.float8_e4m3fn), cache[:, 1]))
    # unsupported V dtype
    with pytest.raises(ValueError, match="v_cache dtype"):
        attn.run(q, (cache[:, 0], cache[:, 1].to(torch.float32)))
    # NVFP4 block scales: no run() kwarg
    with pytest.raises(TypeError, match="kv_cache_sf"):
        attn.run(
            q,
            (cache[:, 0], cache[:, 1]),
            kv_cache_sf=torch.zeros(1, device="cuda"),
        )

    # zero-length KV request: a legal padding row (the legacy wrapper rejects
    # it); the live request is unaffected
    md_e = _csr_metadata([300, 16], [300, 0], ids, 16)
    q_e = torch.cat([q, torch.randn(16, 8, 128, dtype=q.dtype, device=q.device)])
    out_e, lse_e = attn.plan(md_e, lse_mode="base2", **plan_kw).run(
        q_e, (cache[:, 0], cache[:, 1])
    )
    _assert_matches_oracle(
        out_e[:300], lse_e[:300], md, q, cache[:, 0], cache[:, 1], causal=True
    )


def test_paged_wrapper_validate_inputs_nan_scan(monkeypatch):
    """FLASHINFER_VALIDATE_INPUTS is a cute-dsl wrapper NaN scan; PagedAttention
    has none, and fa2 never reads the poisoned in-page tail."""
    _skip_unless_sm100()
    (q, _, _, cache, _qo, _, _kv_pg, ids, _lpl) = _build_wrapper_problem(
        [300], [300], 16
    )
    md = _csr_metadata([300], [300], ids, 16)
    attn = PagedAttention(q.device)
    attn.plan(
        md, **GEOMETRY, kv_layout="NHD", causal=True, lse_mode="base2", backend="fa2"
    )
    assert attn.backend == "fa2"
    # poison the tail of the last referenced page (past last_page_len)
    cache[int(ids[-1]), 1, -1] = float("nan")
    monkeypatch.setenv("FLASHINFER_VALIDATE_INPUTS", "1")
    out, lse = attn.run(q, (cache[:, 0], cache[:, 1]))
    assert torch.isfinite(out.float()).all().item()
    _assert_matches_oracle(
        out,
        lse,
        md,
        q,
        cache[:, 0],
        cache[:, 1].nan_to_num(nan=0.0),
        causal=True,
    )
