"""Legacy -> unified: tests/attention/test_trtllm_gen_attention_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and every row skips on the ``compute_capability == 10`` gate, so
its kernels never executed in PR CI (reports/unified-prefill-round4-20260918/
ci-status.md).  The B200 runs of this file and of the legacy file are the
first complete records.

The legacy fixture (``_test_trtllm_batch_prefill`` and the helpers of
``tests/attention/test_trtllm_gen_attention_decode.py`` under seed 0) is
re-created per row and its stacked ``(pages, 2, ...)`` pool is handed over as
the ``K = kv[:, 0]`` / ``V = kv[:, 1]`` views (no copy), in the dense form
with the legacy page table.  Every numeric row asserts the legacy reference
(the fa2 paged wrapper, or ``sink_attention_unified`` for the sink rows) at
the legacy budget (1e-2 out, 1e-3 LSE) AND the fp32 oracle.  The backend is
pinned (``trtllm-gen``; ``attn.backend`` asserted) and the same workload is
also planned with ``backend="auto"`` -- the backend that served it is recorded
as the junit property ``auto_backend``.  The legacy parametrize axes and
values are kept so node ids compare one to one; the axes that have no unified
meaning are inert here (``window_left=-1`` only, ``enable_pdl=None``,
``max_q_len`` / ``max_kv_len`` single-valued).  The batch-128/256 shapes are
``slow`` (``FI_PARITY_SLOW=1``); the default subset runs the batch-4 shapes,
which cover every other axis value.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_trtllm_batch_prefill: the fp8 / nvfp4 dtype triples have no unified
  spelling (fp8 q with q_scale: EXPECT_FP8_Q; an output dtype independent of
  q: EXPECT_OUTPUT_DTYPE; packed nvfp4 KV + block scale factors:
  EXPECT_NVFP4_KV) -- those rows assert the clear rejection.  ``skips_softmax``
  (skip_softmax_threshold_scale_factor, an approximate softmax) has no plan /
  run knob (EXPECT_SKIP_SOFTMAX).  ``uses_shared_paged_kv_idx=False`` (the
  interleaved 2p / 2p+1 layout with a [B, 2, M] table) is two page-id
  mappings, the metadata takes one (EXPECT_INDEPENDENT_KV_TABLES).  head_dim
  256 is capability-excluded on trtllm-gen (``_capabilities.py`` declares
  (128, 128) only, per the capability-honesty rule; production trtllm-gen
  ships 256 / 512): the row asserts the rejection (EXPECT_TRTLLM_HEAD_DIM_256)
  and runs the same fixture on fa2, which declares 256.
- test_trtllm_batch_prefill_lse_contract: the legacy ``(return_lse=False,
  provide_lse=True)`` cell fills the caller buffer and returns the output
  alone; the unified contract rejects ``lse=`` under ``lse_mode='none'``
  before any launch (asserted) and the same buffer is the output of a base2
  plan.  The workspace guard-region check is a native-only contract.
- test_trtllm_batch_prefill_cubin_variants: fp8 QKV with the sm_107-only
  spcompress cubins; the legacy function skips itself on B200 (SM107 only);
  fp8 q / output dtype as above, cubin selection is a backend-suite contract.
- test_trtllm_batch_prefill_head_dim_512: D512 is capability-excluded on
  trtllm-gen (EXPECT_TRTLLM_HEAD_DIM_512; production ships it); fa2 declares
  (512, 512) since WP-T, so the bf16 / fp16 rows run the legacy fixture on fa2
  against the legacy SDPA reference and the oracle; fp8 triples assert
  EXPECT_FP8_Q.
- the five ``test_trtllm_gen_prefill*`` functions call
  ``trtllm_ragged_attention_deepseek`` (ragged MLA prefill): out-of-scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_NVFP4_KV,
    EXPECT_OUTPUT_DTYPE,
    EXPECT_SKIP_SOFTMAX,
    EXPECT_TRTLLM_HEAD_DIM_256,
    EXPECT_TRTLLM_HEAD_DIM_512,
    EXPECT_TRTLLM_LARGE_PAGES,
    LSE_TOL,
    OUT_TOL,
    assert_legacy_close,
    check_legacy_map,
    check_legacy_map_complete,
    fp8_q_rejected,
    gated,
    independent_tables_rejected,
    legacy_fa2_paged_reference,
    legacy_sink_reference,
    legacy_trtllm_problem,
    oracle,
    output_dtype_knob_present,
    plan_legacy_problem,
    reference_long,
    skip_softmax_knob_present,
    slow_case,
    trtllm_resolve_kwargs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_trtllm_gen_attention_prefill.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill",
        ["test_trtllm_batch_prefill"],
        "partial",
        "same grid and node ids (HND/NHD x 10 shapes x 7 dtype triples x sink x "
        "D128/256 x strided q x skip-softmax x shared/independent tables x causal; "
        "batch 128/256 shapes slow); bf16/fp16 D128 rows run the legacy fixture (seed "
        "0) on pinned trtllm-gen and on auto (recorded) vs the legacy fa2 / sink "
        "reference at 1e-2 (+ LSE 1e-3) and the oracle; D256 asserts the trtllm-gen "
        "rejection (EXPECT_TRTLLM_HEAD_DIM_256) and runs on fa2; fp8 / nvfp4 triples, "
        "skip-softmax and independent K/V tables assert their rejections "
        "(EXPECT_FP8_Q + EXPECT_OUTPUT_DTYPE + EXPECT_NVFP4_KV, EXPECT_SKIP_SOFTMAX, "
        "EXPECT_INDEPENDENT_KV_TABLES); window_left / enable_pdl / max_q_len / "
        "max_kv_len are single-valued inert axes",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_lse_contract",
        ["test_trtllm_batch_prefill_lse_contract"],
        "partial",
        "same fixture (HND, B2, page16, H4:2, fp16, q<=64, kv<=128, D128); three of "
        "the legacy (return_lse, provide_lse) cells map to lse_mode base2 / a caller "
        "lse buffer (returned tensor IS the buffer, finite, fp32, vs the fa2 LSE at "
        "1e-3 and the oracle); the (False, True) cell is a contract difference: "
        "unified rejects lse= with lse_mode='none' instead of filling it; the "
        "workspace guard-region check is native-only",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_bs1",
        ["test_trtllm_batch_prefill_bs1"],
        "partial",
        "same grid (HND/NHD, B1, page16, H64:8, bf16, q = kv = 8192, D128/256, "
        "skip-softmax, shared/independent tables); the kernel rows need "
        "FI_PARITY_SLOW=1: D128 on trtllm-gen (and auto) vs the legacy fa2 reference "
        "(1e-2) and the chunked oracle, D256 asserts the trtllm-gen rejection and "
        "runs on fa2; skip-softmax / independent tables assert their rejections",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_cubin_variants",
        ["test_trtllm_batch_prefill_cubin_variants"],
        "unsupported-by-design",
        "fp8 QKV with bf16/fp16/fp8 output and the sm_107-only spcompress cubins; the "
        "legacy function skips on B200 itself; unified has no fp8 q / output dtype "
        "(EXPECT_FP8_Q, EXPECT_OUTPUT_DTYPE): every id asserts the resolve rejection; "
        "cubin selection is a backend-suite contract",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_dynamic_page_size_gqa",
        ["test_trtllm_batch_prefill_dynamic_page_size_gqa"],
        "partial",
        "pages 128/256/512/1024 are declared on trtllm-gen and cake since WP-T "
        "(EXPECT_TRTLLM_LARGE_PAGES): the legacy fixture (B4, H10:2, bf16, q<=257, "
        "kv<=1024, causal) runs on trtllm-gen, cake and auto vs the legacy fa2 "
        "reference and the oracle; uses_shared_paged_kv_idx=False asserts "
        "EXPECT_INDEPENDENT_KV_TABLES",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_head_dim_512",
        ["test_trtllm_batch_prefill_head_dim_512"],
        "partial",
        "same grid (layouts x 3 shapes x 4 dtype triples x q 1/255/511 x kv 511/2047 "
        "x skip-softmax x shared/independent; batch 128 slow); D512 is "
        "capability-excluded on trtllm-gen (EXPECT_TRTLLM_HEAD_DIM_512, asserted) and "
        "declared on fa2 (WP-T), so the bf16/fp16 rows run the legacy fixture on fa2 "
        "and auto vs the legacy SDPA reference (1e-2) and the oracle; fp8 triples "
        "assert EXPECT_FP8_Q; skip-softmax / independent tables assert their "
        "rejections",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill",
        [],
        "out-of-scope",
        "trtllm_ragged_attention_deepseek (ragged MLA prefill), not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_use_fp16_softmax",
        [],
        "out-of-scope",
        "ragged MLA prefill (SM107-only fp16-softmax cubin variant), not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_fp8",
        [],
        "out-of-scope",
        "ragged MLA prefill with fp8 inputs on the cute-dsl backend, not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_bs1",
        [],
        "out-of-scope",
        "ragged MLA prefill, not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_glm5",
        [],
        "out-of-scope",
        "ragged MLA prefill with the GLM-5 MHA dimensions, not paged",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


# ---------------------------------------------------------------------------
# the legacy grids (values as in the legacy file; big batches under `slow`)
# ---------------------------------------------------------------------------

TRTLLM_BATCH_PREFILL_SHAPES = [
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    pytest.param(128, 16, 2, 5, marks=pytest.mark.slow),
    pytest.param(128, 32, 4, 1, marks=pytest.mark.slow),
    pytest.param(128, 64, 2, 8, marks=pytest.mark.slow),
    pytest.param(256, 16, 4, 8, marks=pytest.mark.slow),
    pytest.param(256, 32, 2, 8, marks=pytest.mark.slow),
    pytest.param(256, 64, 4, 1, marks=pytest.mark.slow),
    pytest.param(256, 64, 4, 5, marks=pytest.mark.slow),
]

TRTLLM_BATCH_PREFILL_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
]


def _quantized_triple_rejected(
    q_dtype,
    kv_dtype,
    o_dtype,
    *,
    head_dim,
    page_size,
    kv_layout,
    causal,
    window_left=-1,
) -> bool:
    """The legacy fp8 / nvfp4 dtype triples: fp8 q is rejected at resolve
    (EXPECT_FP8_Q); an output dtype independent of q and packed nvfp4 KV with
    block scale factors have no spelling at all (EXPECT_OUTPUT_DTYPE,
    EXPECT_NVFP4_KV).  Returns True when the row stops here."""
    import inspect

    from flashinfer.prefill import PagedAttention

    assert output_dtype_knob_present() == EXPECT_OUTPUT_DTYPE
    run_params = inspect.signature(PagedAttention.run).parameters
    assert ("kv_cache_sf" in run_params) == EXPECT_NVFP4_KV
    res = fp8_q_rejected(
        backend="trtllm-gen",
        head_dim=head_dim,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        window_left=window_left,
    )
    if res is not None:
        pytest.fail(
            f"EXPECT_FP8_Q flipped: port the legacy ({q_dtype}, {kv_dtype}, "
            f"{o_dtype}) fixture with q_scale / o_scale here"
        )
    return True


def _skip_softmax_rejected() -> bool:
    """skip_softmax_threshold_scale_factor (approximate softmax) is a
    trtllm-gen launch knob with no unified spelling (EXPECT_SKIP_SOFTMAX).
    Returns True when the row stops here."""
    assert skip_softmax_knob_present() == EXPECT_SKIP_SOFTMAX
    return not EXPECT_SKIP_SOFTMAX


def _independent_tables_row(p) -> bool:
    """Returns True when the uses_shared_paged_kv_idx=False row stops here."""
    if independent_tables_rejected(p) is None:
        return True
    pytest.fail("EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here")


def _assert_vs_legacy_and_oracle(
    p, md, out, lse, *, causal, enable_sink, kv_layout, check_lse=True
):
    if enable_sink:
        ref = legacy_sink_reference(p, causal=causal)
        assert_legacy_close(out, ref)
    else:
        ref, lse_ref = legacy_fa2_paged_reference(p, causal=causal)
        assert_legacy_close(out, ref)
        if check_lse and lse is not None:
            torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=causal,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    if lse is not None:
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def _run_auto_and_record(
    p, record_property, *, causal, enable_sink, q_input, k, v, kv_layout
):
    """The same workload under backend='auto': record the serving backend and
    hold it to the oracle."""
    attn, md = plan_legacy_problem(p, "auto", causal=causal, use_sinks=enable_sink)
    record_property("auto_backend", attn.backend)
    out, lse = attn.run(
        q_input,
        (k, v),
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
    )
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=causal,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    TRTLLM_BATCH_PREFILL_SHAPES,
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    TRTLLM_BATCH_PREFILL_DTYPES,
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_q_len", [511])
@pytest.mark.parametrize("max_kv_len", [2047])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_batch_prefill(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
    record_property,
):
    from tests.attention.test_trtllm_gen_attention_decode import (
        flip_coin,
        make_query_non_contiguous,
    )

    # legacy skips (kept so the node ids agree case by case)
    if not causal and window_left >= 0:
        pytest.skip("Non-causal paged trtllm-gen tests only cover dense attention")
    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )
    if kv_dtype == "nvfp4":
        if q_dtype != "fp8":
            pytest.skip("NVFP4 KV cache requires FP8 query")
        if o_dtype != "fp8":
            pytest.skip("NVFP4 KV cache only supports FP8 output")

    if q_dtype == "fp8" and _quantized_triple_rejected(
        q_dtype,
        kv_dtype,
        o_dtype,
        head_dim=head_dim,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
    ):
        return
    if skips_softmax and _skip_softmax_rejected():
        return
    p = legacy_trtllm_problem(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        head_dim,
    )
    if not uses_shared_paged_kv_idx and _independent_tables_row(p):
        return
    backend = "trtllm-gen"
    if head_dim == 256:
        res = gated(
            EXPECT_TRTLLM_HEAD_DIM_256,
            lambda: resolve_paged_attention(
                **trtllm_resolve_kwargs(p, causal=causal, backend="trtllm-gen")
            ),
            match="unsupported head dims \\(256, 256\\)",
        )
        backend = "trtllm-gen" if res is not None else "fa2"
    attn, md = plan_legacy_problem(p, backend, causal=causal, use_sinks=enable_sink)
    q_input = (
        make_query_non_contiguous(p["q"], p["num_qo_heads"], p["head_dim"])
        if non_contiguous_query
        else p["q"].contiguous()
    )
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]  # views of the stacked pool
    assert k.data_ptr() == p["kv_cache"].data_ptr()
    # the legacy coin decides whether the caller supplies the output buffer
    out_buf = None
    if flip_coin(batch_size, page_size, num_kv_heads, head_grp_size, o_dtype):
        out_buf = torch.empty(
            p["q"].shape[0],
            p["num_qo_heads"],
            p["head_dim"],
            dtype=p["dtype"],
            device=DEVICE,
        )
    out, lse = attn.run(
        q_input,
        (k, v),
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
        out=out_buf,
    )
    if out_buf is not None:
        assert out.data_ptr() == out_buf.data_ptr()
    assert lse.dtype == torch.float32 and lse.shape == (
        p["q"].shape[0],
        p["num_qo_heads"],
    )
    assert torch.isfinite(lse).all()
    _assert_vs_legacy_and_oracle(
        p, md, out, lse, causal=causal, enable_sink=enable_sink, kv_layout=kv_layout
    )
    _run_auto_and_record(
        p,
        record_property,
        causal=causal,
        enable_sink=enable_sink,
        q_input=q_input,
        k=k,
        v=v,
        kv_layout=kv_layout,
    )


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_lse_contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("provide_lse", [False, True])
def test_trtllm_batch_prefill_lse_contract(return_lse, provide_lse):
    p = legacy_trtllm_problem("HND", 2, 16, 2, 2, "fp16", 64, 128, 128)
    ref, lse_ref = legacy_fa2_paged_reference(p, causal=True)
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    provided = (
        torch.full(
            (p["q"].shape[0], p["num_qo_heads"]),
            float("nan"),
            device=DEVICE,
            dtype=torch.float32,
        )
        if provide_lse
        else None
    )
    if not return_lse and not provide_lse:
        attn, md = plan_legacy_problem(p, "trtllm-gen", causal=True, lse_mode="none")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
        assert lse is None
    elif not return_lse and provide_lse:
        # contract difference: legacy fills the provided buffer and returns
        # only the output; unified rejects lse= under lse_mode='none' before
        # any launch, and the same buffer is the output of a base2 plan
        attn, md = plan_legacy_problem(p, "trtllm-gen", causal=True, lse_mode="none")
        with pytest.raises(ValueError, match="lse_mode='none'"):
            attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        assert torch.isnan(provided).all()  # untouched
        attn, md = plan_legacy_problem(p, "trtllm-gen", causal=True, lse_mode="base2")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        assert lse.data_ptr() == provided.data_ptr()
    else:
        attn, md = plan_legacy_problem(p, "trtllm-gen", causal=True, lse_mode="base2")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        if provide_lse:
            assert lse.data_ptr() == provided.data_ptr()
    if lse is not None:
        assert lse.dtype == torch.float32
        assert lse.shape == (p["q"].shape[0], p["num_qo_heads"])
        assert torch.isfinite(lse).all(), (
            "trtllm-gen context kernel produced non-finite LSE"
        )
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    assert_legacy_close(out, ref)
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=True,
        kv_layout="HND",
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    if lse is not None:
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_bs1: q = kv = 8192
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (1, 16, 8, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [8192])
@pytest.mark.parametrize("max_kv_len", [8192])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_bs1(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
    record_property,
):
    if skips_softmax and _skip_softmax_rejected():
        return
    slow_case("B1, q = kv = 8192, 64 query heads")
    p = legacy_trtllm_problem(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        head_dim,
    )
    if not uses_shared_paged_kv_idx and _independent_tables_row(p):
        return
    backend = "trtllm-gen"
    if head_dim == 256:
        res = gated(
            EXPECT_TRTLLM_HEAD_DIM_256,
            lambda: resolve_paged_attention(
                **trtllm_resolve_kwargs(p, causal=True, backend="trtllm-gen")
            ),
            match="unsupported head dims \\(256, 256\\)",
        )
        backend = "trtllm-gen" if res is not None else "fa2"
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    attn, md = plan_legacy_problem(p, backend, causal=True)
    out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
    ref, lse_ref = legacy_fa2_paged_reference(p, causal=True)
    assert_legacy_close(out, ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    o_out, o_lse = reference_long(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=True,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    attn_auto, _ = plan_legacy_problem(p, "auto", causal=True)
    record_property("auto_backend", attn_auto.backend)
    out_a, lse_a = attn_auto.run(p["q"], (k, v), sm_scale=p["sm_scale"])
    torch.testing.assert_close(out_a.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse_a, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_cubin_variants (fp8 QKV; spcompress is sm_107-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
    ],
)
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
    ],
)
@pytest.mark.parametrize(
    "head_dim,window_left",
    [
        (128, -1),
        (256, -1),
        (128, 127),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False, True])
@pytest.mark.parametrize("max_q_len", [511, 3023])
@pytest.mark.parametrize("max_kv_len", [2047, 8192])
def test_trtllm_batch_prefill_cubin_variants(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
):
    """The legacy row is SM107-only (spcompress cubins) and skips on B200; the
    unified analog is the fp8-q / output-dtype rejection for its dtype
    triple, head dim and window."""
    _quantized_triple_rejected(
        q_dtype,
        kv_dtype,
        o_dtype,
        head_dim=head_dim,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=True,
        window_left=window_left,
    )


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_dynamic_page_size_gqa: pages 128..1024
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_dynamic_page_size_gqa(
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    record_property,
) -> None:
    p = legacy_trtllm_problem("HND", 4, page_size, 2, 5, "bf16", 257, 1024, 128)
    if not uses_shared_paged_kv_idx and _independent_tables_row(p):
        return
    backends = []
    for backend in ("trtllm-gen", "cake"):
        res = gated(
            EXPECT_TRTLLM_LARGE_PAGES,
            lambda backend=backend: resolve_paged_attention(
                **trtllm_resolve_kwargs(p, causal=True, backend=backend)
            ),
            match=f"unsupported page_size {page_size}",
        )
        if res is not None:
            backends.append(backend)
    backends.append("fa2")  # runs any page size
    ref, lse_ref = legacy_fa2_paged_reference(p, causal=True)
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    o_out = o_lse = None
    for backend in backends + ["auto"]:
        attn, md = plan_legacy_problem(p, backend, causal=True)
        if backend == "auto":
            record_property("auto_backend", attn.backend)
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
        assert_legacy_close(out, ref)
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
        if o_out is None:
            o_out, o_lse = oracle(
                md,
                p["q"],
                p["ref_kv_cache"][:, 0],
                p["ref_kv_cache"][:, 1],
                causal=True,
                kv_layout="HND",
                sm_scale=p["sm_scale"],
            )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_head_dim_512
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
        (4, 32, 4, 5),
        pytest.param(128, 16, 2, 8, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [1, 255, 511])
@pytest.mark.parametrize("max_kv_len", [511, 2047])
@pytest.mark.parametrize("head_dim", [512])
@pytest.mark.parametrize("non_contiguous_query", [False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_head_dim_512(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
    record_property,
):
    """D512 (Gemma-style full attention): trtllm-gen is capability-excluded
    (EXPECT_TRTLLM_HEAD_DIM_512, asserted); fa2 declares (512, 512), so the
    bf16 / fp16 rows run the legacy fixture on fa2 (and auto) against the
    legacy reference for D512 -- torch SDPA on the reference pool -- and the
    oracle."""
    from tests.attention.test_trtllm_gen_attention_decode import sdpa_paged_reference

    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )
    if q_dtype == "fp8" and _quantized_triple_rejected(
        q_dtype,
        kv_dtype,
        o_dtype,
        head_dim=head_dim,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=True,
    ):
        return
    if skips_softmax and _skip_softmax_rejected():
        return
    p = legacy_trtllm_problem(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        head_dim,
    )
    if not uses_shared_paged_kv_idx and _independent_tables_row(p):
        return
    res = gated(
        EXPECT_TRTLLM_HEAD_DIM_512,
        lambda: resolve_paged_attention(
            **trtllm_resolve_kwargs(p, causal=True, backend="trtllm-gen")
        ),
        match="unsupported head dims \\(512, 512\\)",
    )
    backend = "trtllm-gen" if res is not None else "fa2"
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    ref = sdpa_paged_reference(
        p["ref_q"],
        p["ref_kv_cache"],
        p["q_lens"],
        p["seq_lens"],
        p["page_table"],
        page_size,
        p["num_qo_heads"],
        p["num_kv_heads"],
        head_dim,
        kv_layout,
        window_left,
    )
    o_out = o_lse = None
    for name in (backend, "auto"):
        attn, md = plan_legacy_problem(p, name, causal=True)
        if name == "auto":
            record_property("auto_backend", attn.backend)
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
        assert_legacy_close(out, ref)
        if o_out is None:
            o_out, o_lse = oracle(
                md,
                p["q"],
                p["ref_kv_cache"][:, 0],
                p["ref_kv_cache"][:, 1],
                causal=True,
                kv_layout=kv_layout,
                sm_scale=p["sm_scale"],
            )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
