"""Legacy -> unified parity, group B: fp8 KV paged prefill.

Each test re-runs the legacy test's own fixture (same grid, seed, value
scale, quantization and scales) through ``PagedAttention`` and asserts

1. the legacy assertion: the legacy reference kernel at the legacy budget;
2. adapter equivalence: the fp32 oracle on the DEQUANTIZED pool (03 §2.5);
3. calibration inheritance: the fp32 oracle on the UNQUANTIZED pool at the
   legacy calibration budget (03 §2.5, the second equivalence).

The legacy paged fp8 wrappers plan without ``causal``, i.e. non-causal;
the unified plans below say so explicitly.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_parity_b_helpers import (
    DEVICE,
    EXPECT_FA2_E5M2_KV,
    EXPECT_FA2_HEAD_DIM_512,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    csr_metadata_from_legacy,
    gated,
    oracle,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_MAP = [
    # (legacy nodeid or function, unified test function(s) in this file, status, note)
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale",
        ["test_fp8_calibration_scale"],
        "partial",
        "same grid (B12/17, q1/7/53, kv54/97, page1/8/16, H4/32:4, D64-512, HND/NHD, "
        "e4m3/e5m2), seed 42, 0.05 pool, amax/256 scales; the combined fp8 pool as "
        "K=kv[:,0]/V=kv[:,1] views, CSR form, non-causal as legacy; asserted vs the "
        "legacy fa2 fp16 kernel at the legacy budget (1e-2, 2e-1), vs the oracle on the "
        "unquantized pool at the same budget, and vs the oracle on the dequantized pool "
        "(OUT/LSE_TOL); e5m2 and D512 ids assert the resolve rejection until "
        "EXPECT_FA2_E5M2_KV / EXPECT_FA2_HEAD_DIM_512 (WP-T)",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_decode_with_prefill_with_paged_kv_cache",
        ["test_fp8_decode_with_prefill"],
        "partial",
        "same grid (B12/17, kv54/97, page1/8/16, H4/32:4, D128/256, HND/NHD, e4m3/e5m2), "
        "seed 42, 0.1 pool cast to fp8, q_len 1, no scales; unified fa2 vs the legacy "
        "fa2 decode wrapper at the legacy budget (1e-2) and vs the dequantized oracle; "
        "e5m2 ids assert the rejection until EXPECT_FA2_E5M2_KV",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_paged_fp8_h512_long_q_keeps_cta32",
        ["test_fp8_h512_long_q_paged"],
        "unsupported-by-design",
        "D512 fp8 KV is capability-excluded on fa2 (EXPECT_FA2_HEAD_DIM_512, WP-T); the "
        "CTA_TILE_Q == 32 plan_info assertion is a private scheduler contract and stays "
        "native-only; the unified test asserts the rejection and, once flipped, the "
        "fixture's finite output plus the dequantized oracle",
    ),
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_chunked_prefill"],
        "unsupported-by-design",
        "legacy fixture imported as is (q17, KV10003, H16:2, D512, page16, NHD, sm_scale 1, "
        "k=v_scale .02) and the legacy torch oracle + budget; D512 on fa2 is "
        "capability-excluded (EXPECT_FA2_HEAD_DIM_512, WP-T)",
    ),
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_tensor_core_decode"],
        "unsupported-by-design",
        "same fixture with q_len 1 (the legacy tensor-core decode wrapper maps to the "
        "unified prefill path); D512 on fa2 is capability-excluded "
        "(EXPECT_FA2_HEAD_DIM_512, WP-T)",
    ),
]

FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
FP8_IDS = ["e4m3", "e5m2"]

_workspace = None


def _legacy_workspace():
    global _workspace
    if _workspace is None:
        _workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    return _workspace


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())


def _fa2_fp8_gate(dtype, head_dim):
    """(flag, match) for the fa2 fp8-KV resolve of a legacy id."""
    if dtype == torch.float8_e5m2 and not EXPECT_FA2_E5M2_KV:
        return False, "unsupported kv dtype"
    if head_dim == 512 and not EXPECT_FA2_HEAD_DIM_512:
        return False, "unsupported head dims \\(512, 512\\)"
    return True, ""


def _resolve_fa2(
    *, num_qo_heads, num_kv_heads, head_dim, kv_dtype, page_size, kv_layout, causal
):
    return resolve_paged_attention(
        device=torch.device(DEVICE),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=kv_dtype,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=True,
        kv_input_form="page_indices",
        backend="fa2",
    )


# ---------------------------------------------------------------------------
# test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len", [1, 7, 53])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=FP8_IDS)
def test_fp8_calibration_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    flag, match = _fa2_fp8_gate(dtype, head_dim)
    res = gated(
        flag,
        lambda: _resolve_fa2(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=dtype,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=False,
        ),
        match=match,
    )
    if res is None:
        return

    # ---- the legacy fixture, verbatim ----
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.05
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.05
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = _legacy_workspace()
    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_f16.run(q, kv_data)  # the legacy reference
    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256
    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    # ---- unified: the combined fp8 pool as two views, legacy CSR mapped ----
    k_view, v_view = kv_data_fp8[:, 0], kv_data_fp8[:, 1]
    assert k_view.data_ptr() == kv_data_fp8.data_ptr()  # no copy
    md = csr_metadata_from_legacy(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=dtype,
        kv_layout=kv_layout,
        causal=False,  # the legacy plan() default
        lse_mode="base2",
        backend=res,
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (k_view, v_view), k_scale=k_scale, v_scale=v_scale)

    # 1. the legacy assertion (fp16 kernel vs fp8 kernel, legacy budget)
    torch.testing.assert_close(o_fp16, out, atol=1e-2, rtol=2e-1)
    # 3. calibration inheritance: oracle on the UNQUANTIZED pool, legacy budget
    ref_out_f16, _ = oracle(
        md, q, k_data[:, 0], v_data[:, 0], causal=False, kv_layout=kv_layout
    )
    torch.testing.assert_close(out.float(), ref_out_f16, atol=1e-2, rtol=2e-1)
    # 2. adapter equivalence: oracle on the DEQUANTIZED pool with the same scales
    ref_out, ref_lse = oracle(
        md,
        q,
        k_fp8[:, 0].float() * k_scale,
        v_fp8[:, 0].float() * v_scale,
        causal=False,
        kv_layout=kv_layout,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    assert lse.shape == (q.shape[0], num_qo_heads) and lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_batch_decode_with_prefill_with_paged_kv_cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", FP8_DTYPES, ids=FP8_IDS)
def test_fp8_decode_with_prefill(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
):
    flag, match = _fa2_fp8_gate(dtype, head_dim)
    res = gated(
        flag,
        lambda: _resolve_fa2(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=dtype,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=False,
        ),
        match=match,
    )
    if res is None:
        return

    # ---- the legacy fixture, verbatim ----
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=torch.float16).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.1
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.1
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    ).to(dtype)
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int()
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    # the legacy reference: the fa2 decode wrapper on the same fp8 pool
    workspace_buffer = _legacy_workspace()
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    decode_wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=dtype,
    )
    o_decode_fp8 = decode_wrapper.run(q, kv_data)

    # ---- unified: q_len 1 through the prefill path, pool views, CSR ----
    md = csr_metadata_from_legacy(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=dtype,
        kv_layout=kv_layout,
        causal=False,
        lse_mode="base2",
        backend=res,
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (kv_data[:, 0], kv_data[:, 1]))  # no scales, as legacy

    torch.testing.assert_close(o_decode_fp8, out, atol=1e-2, rtol=1e-2)  # legacy
    ref_out, ref_lse = oracle(
        md,
        q,
        kv_data[:, 0].float(),
        kv_data[:, 1].float(),
        causal=False,
        kv_layout=kv_layout,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_paged_fp8_h512_long_q_keeps_cta32
# ---------------------------------------------------------------------------


def test_fp8_h512_long_q_paged():
    """D512 fp8 KV, q = kv = 128, batch 2, page 16, NHD (the legacy fixture).

    The legacy assertion has two parts: ``_plan_info[CTA_TILE_Q] == 32`` (a
    scheduler contract of the native wrapper, native-only) and a finite
    output.  Here: the fa2 capability rejection for (512, 512) until
    EXPECT_FA2_HEAD_DIM_512, then the finite output and the dequantized
    oracle (stronger than the legacy finite check)."""
    head_dim, batch_size, page_size = 512, 2, 16
    qo_len = kv_len = 128
    num_qo_heads = num_kv_heads = 4
    res = gated(
        EXPECT_FA2_HEAD_DIM_512,
        lambda: _resolve_fa2(
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=torch.float8_e4m3fn,
            page_size=page_size,
            kv_layout="NHD",
            causal=False,
        ),
        match="unsupported head dims \\(512, 512\\)",
    )
    if res is None:
        return
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data_fp8 = (
        torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        )
        .to(0)
        .to(torch.float8_e4m3fn)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    md = csr_metadata_from_legacy(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float8_e4m3fn,
        kv_layout="NHD",
        causal=False,
        lse_mode="base2",
        backend=res,
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (kv_data_fp8[:, 0], kv_data_fp8[:, 1]))
    assert torch.isfinite(out).all()  # the legacy assertion
    ref_out, ref_lse = oracle(
        md,
        q,
        kv_data_fp8[:, 0].float(),
        kv_data_fp8[:, 1].float(),
        causal=False,
        kv_layout="NHD",
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_gemma4_fa2_accuracy.py (the legacy fixture and oracle imported as is)
# ---------------------------------------------------------------------------


def _gemma4_unified(q_len):
    from tests.attention import test_gemma4_fa2_accuracy as legacy

    res = gated(
        EXPECT_FA2_HEAD_DIM_512,
        lambda: resolve_paged_attention(
            device=legacy.DEVICE,
            num_qo_heads=legacy.NUM_QO_HEADS,
            num_kv_heads=legacy.NUM_KV_HEADS,
            head_dim_qk=legacy.HEAD_DIM,
            q_dtype=legacy.Q_DTYPE,
            kv_dtype=legacy.KV_DTYPE,
            page_size=legacy.PAGE_SIZE,
            kv_layout="NHD",
            causal=True,
            need_lse=True,
            kv_input_form="page_indices",
            backend="fa2",
        ),
        match="unsupported head dims \\(512, 512\\)",
    )
    if res is None:
        return
    q, k_cache, v_cache, num_pages = legacy._make_inputs(q_len)
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len = legacy._paged_metadata(
        q_len, num_pages
    )
    md = csr_metadata_from_legacy(
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len, legacy.PAGE_SIZE
    )
    assert int(md.kv_seq_lens_cpu[0]) == legacy.KV_LEN
    attn = PagedAttention(legacy.DEVICE)
    attn.plan(
        md,
        num_qo_heads=legacy.NUM_QO_HEADS,
        num_kv_heads=legacy.NUM_KV_HEADS,
        head_dim_qk=legacy.HEAD_DIM,
        q_dtype=legacy.Q_DTYPE,
        kv_dtype=legacy.KV_DTYPE,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        backend=res,
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(
        q,
        (k_cache, v_cache),
        sm_scale=legacy.SM_SCALE,
        k_scale=legacy.K_SCALE,
        v_scale=legacy.V_SCALE,
    )
    ref_out, ref_lse = legacy._reference(q, k_cache, v_cache)
    legacy._assert_matches_reference(out, lse, ref_out, ref_lse)  # legacy budget
    o_out, o_lse = oracle(
        md,
        q,
        k_cache.float() * legacy.K_SCALE,
        v_cache.float() * legacy.V_SCALE,
        causal=True,
        kv_layout="NHD",
        sm_scale=legacy.SM_SCALE,
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def test_gemma4_fp8_kv_head_dim_512_chunked_prefill():
    _gemma4_unified(17)


def test_gemma4_fp8_kv_head_dim_512_tensor_core_decode():
    _gemma4_unified(1)
