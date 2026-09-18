"""Legacy -> unified: tests/attention/test_fp8_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only (not in the A10G fixed
shards); the unified file is in no default lane (tests/experimental is
excluded by norecursedirs).

Each paged row keeps the legacy grid, seed, value scale, quantization and
scales (the combined ``(pages, 2, ...)`` fp8 pool is handed over as the
``K = kv[:, 0]`` / ``V = kv[:, 1]`` views, no copy; the legacy CSR maps
losslessly to ``PagedAttentionMetadata.csr``), pins ``backend="fa2"`` and
asserts

1. the legacy assertion: the legacy reference kernel at the legacy budget;
2. adapter equivalence: the fp32 oracle on the DEQUANTIZED pool (OUT/LSE_TOL);
3. calibration inheritance (the calibration test): the fp32 oracle on the
   UNQUANTIZED fp16 pool at the legacy calibration budget (1e-2 / 2e-1).

The legacy paged fp8 wrappers plan without ``causal``, i.e. non-causal; the
unified plans say so explicitly.  e5m2 KV and head_dim 512 on fa2 are
declared since WP-T (EXPECT_FA2_E5M2_KV / EXPECT_FA2_HEAD_DIM_512 True); the
gates stay so a flipped declaration turns the rows back into rejection
assertions instead of failures.  The default run is the full legacy grid
(1152 + 96 + 1 ids, about 20 s warm on B200), so no ``slow`` subset.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_paged_fp8_h512_long_q_keeps_cta32: the ``_plan_info[CTA_TILE_Q] == 32``
  half of the legacy assertion is a private scheduler contract of the native
  wrapper (native-only); the unified row keeps the finite-output half and adds
  the dequantized oracle.
- test_batch_prefill_with_ragged_kv_cache_fp8, test_ragged_fp8_h512_long_q_keeps_cta32,
  test_ragged_fp8_calibration_scales: ragged KV (BatchPrefillWithRaggedKVCacheWrapper),
  not paged prefill -> out-of-scope.
- test_single_prefill_fp8_h512_long_q: single_prefill_with_kv_cache -> out-of-scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FA2_E5M2_KV,
    EXPECT_FA2_HEAD_DIM_512,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    csr_metadata_from_legacy,
    gated,
    oracle,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_fp8_prefill.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale",
        ["test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale"],
        "equivalent",
        "same grid (B12/17, q1/7/53, kv54/97, page1/8/16, H4/32:4, D64-512, HND/NHD, "
        "e4m3/e5m2), seed 42, 0.05 pool, amax/256 scales, the legacy "
        "skip_if_head_dim_unsupported gate; the combined fp8 pool as K=kv[:,0]/V=kv[:,1] "
        "views, CSR form, non-causal as legacy, pinned fa2; asserted vs the legacy fa2 "
        "fp16 kernel at the legacy budget (1e-2, 2e-1), vs the oracle on the unquantized "
        "pool at the same budget, and vs the oracle on the dequantized pool (OUT/LSE_TOL); "
        "e5m2 and D512 run positively since WP-T (the EXPECT_* gates would turn them back "
        "into rejection assertions)",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_ragged_kv_cache_fp8",
        [],
        "out-of-scope",
        "ragged KV (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_decode_with_prefill_with_paged_kv_cache",
        ["test_batch_decode_with_prefill_with_paged_kv_cache"],
        "equivalent",
        "same grid (B12/17, kv54/97, page1/8/16, H4/32:4, D128/256, HND/NHD, e4m3/e5m2), "
        "seed 42, 0.1 pool cast to fp8, q_len 1, no scales; the unified fa2 prefill path "
        "stands in for the legacy prefill wrapper and is asserted vs the legacy fa2 decode "
        "wrapper at the legacy budget (1e-2, 1e-2) and vs the dequantized oracle",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_ragged_fp8_h512_long_q_keeps_cta32",
        [],
        "out-of-scope",
        "ragged KV wrapper (and a private plan_info scheduler contract), not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_paged_fp8_h512_long_q_keeps_cta32",
        ["test_paged_fp8_h512_long_q_keeps_cta32"],
        "partial",
        "same fixture (B2, q=kv=128, page16, H4:4, D512, NHD, e4m3 KV, no scales) on "
        "pinned fa2 with the legacy finite-output assertion plus the dequantized oracle "
        "(OUT/LSE_TOL); the _plan_info[CTA_TILE_Q] == 32 half of the legacy assertion is "
        "a private scheduler contract of the native wrapper and stays native-only",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_single_prefill_fp8_h512_long_q",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_ragged_fp8_calibration_scales",
        [],
        "out-of-scope",
        "ragged KV wrapper with k/v scales, not paged prefill",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


_workspace = None


def _legacy_workspace():
    global _workspace
    if _workspace is None:
        _workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    return _workspace


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
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
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
    from tests.attention.test_fp8_prefill import skip_if_head_dim_unsupported

    skip_if_head_dim_unsupported(head_dim)  # the legacy gate (SM80+ for D512)
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
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_decode_with_prefill_with_paged_kv_cache(
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


def test_paged_fp8_h512_long_q_keeps_cta32():
    """D512 fp8 KV, q = kv = 128, batch 2, page 16, NHD (the legacy fixture).

    The legacy assertion has two parts: ``_plan_info[CTA_TILE_Q] == 32`` (a
    scheduler contract of the native wrapper, native-only) and a finite
    output.  Here: the fa2 capability gate for (512, 512)
    (EXPECT_FA2_HEAD_DIM_512), then the finite output and the dequantized
    oracle (stronger than the legacy finite check)."""
    from tests.attention.test_fp8_prefill import skip_if_head_dim_unsupported

    head_dim, batch_size, page_size = 512, 2, 16
    skip_if_head_dim_unsupported(head_dim)  # the legacy gate
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
