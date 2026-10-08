"""Legacy -> unified: tests/attention/test_cudnn_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only, and only if the runner
image ships cudnn-frontend (not declared in the repo); the unified file is in
no default lane (tests/experimental is excluded by norecursedirs).

``test_cudnn_prefill`` rebuilds the legacy fixture exactly (seed 1, the
``as_strided`` combined pool with HND shape and NHD-ordered strides handed
over as the same K/V views without a copy, the legacy block table) and runs
it on pinned ``backend="cudnn"`` in the dense form; the assertion is the
legacy one (the fa2 wrapper on the combined pool at 3e-3 / 1e-2) plus the fp32
oracle, and the legacy ``return_lse`` axis (inert in the legacy body) becomes
the ``lse_mode`` axis with the LSE checked against the oracle.  The default
run is the full legacy grid (432 ids, of which 144 inherit the legacy
``s_qo > s_kv`` skip; about 45 s warm on B200).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_cudnn_prefill_fp8: fp8 (e4m3) QUERY with q_scale / k_scale / v_scale
  tensors and an o_data_type independent of q.  The unified envelope admits
  fp16 / bf16 q only (fp8 is a KV-only axis of run(): per-tensor host-float
  k_scale / v_scale), so resolve() rejects q_dtype float8 for cudnn with the
  capability reason (_capabilities.py: q_dtypes = {fp16, bf16}; design doc
  docs/design_docs/paged_attention_unified_lifecycle.md).  Needs a
  quantization descriptor (q dtype, q_scale, o_dtype) -- EXPECT_FP8_Q +
  EXPECT_OUTPUT_DTYPE.  The legacy function itself xfails on Blackwell before
  running and skips elsewhere, so there is no legacy pass to inherit.
- test_cudnn_prefill: the legacy is_cuda_graph_compatible=[True] axis is a
  no-op in both bodies and is kept for node-id parity only.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FP8_Q,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    gated,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_cudnn_prefill.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill",
        ["test_cudnn_prefill"],
        "equivalent",
        "same grid (B1/4, s_qo 8/17/700, s_kv 8/32/1066, page 8/16/64, Hkv 1/4, H4, "
        "causal, return_lse, is_cuda_graph_compatible), seed 1, the legacy as_strided "
        "combined pool handed over as the same K/V views (no copy), dense form with the "
        "legacy block table, pinned cudnn; asserted vs the legacy fa2 reference at the "
        "legacy budget (3e-3, 1e-2) and vs the oracle; return_lse (unused by the legacy "
        "body) maps to lse_mode base2/none and the LSE is checked against the oracle; "
        "the legacy s_qo > s_kv skip is inherited",
    ),
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill_fp8",
        ["test_cudnn_prefill_fp8"],
        "unsupported-by-design",
        "fp8 Q with q_scale / o_data_type has no unified spelling (EXPECT_FP8_Q, "
        "EXPECT_OUTPUT_DTYPE: q_dtypes of every backend are fp16/bf16, fp8 is a KV-only "
        "axis); each legacy id asserts the cudnn resolve rejection on the same grid "
        "(the legacy skips inherited); the legacy function itself xfails on Blackwell "
        "before running, so there is no legacy pass to inherit",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


_legacy_ws = None


def _legacy_workspace(device):
    global _legacy_ws
    if _legacy_ws is None:
        _legacy_ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return _legacy_ws


# ---------------------------------------------------------------------------
# test_cudnn_prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")  # the legacy skip

    # ---- the legacy fixture, verbatim ----
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]
    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size, dim=0
            ),
        ]
    ).int()
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )
    scale = float(1.0 / (head_dim**0.5))
    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # ---- the legacy reference: the fa2 wrapper on the combined pool ----
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _legacy_workspace(device), "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )
    output_ref = wrapper.run(q, kv_cache)

    # ---- unified cuDNN: dense form, the legacy K/V views (HND shape,
    # NHD-ordered strides: the graph is stride-driven), legacy table ----
    assert k_cache.data_ptr() == kv_cache.data_ptr()  # views, no copy
    md = dense_metadata(qo_indptr, actual_seq_lens_kv.view(-1), block_tables, page_size)
    lse_mode = "base2" if return_lse else "none"
    attn = plan_pinned(
        "cudnn",
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=causal,
        lse_mode=lse_mode,
    )
    output, lse = attn.run(q, (k_cache, v_cache), sm_scale=scale)

    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)  # legacy
    ref_out, ref_lse = oracle(
        md, q, k_cache, v_cache, causal=causal, kv_layout="HND", sm_scale=scale
    )
    torch.testing.assert_close(output.float(), ref_out, **OUT_TOL)
    if return_lse:
        assert lse.shape == (q.shape[0], num_qo_heads) and lse.dtype == torch.float32
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    else:
        assert lse is None


# ---------------------------------------------------------------------------
# test_cudnn_prefill_fp8 (legacy: xfail on Blackwell; fp8 Q has no unified spelling)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill_fp8(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    """The legacy fixture quantizes q, K and V to e4m3 with per-tensor scale
    tensors and asks for a bf16 output.  Unified: the cudnn resolve rejects
    the fp8 q dtype (capability reason); flips with EXPECT_FP8_Q once a
    quantization descriptor (q dtype, q_scale, o_dtype) is part of the
    contract, at which point the legacy fixture is to be ported here."""
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")  # the legacy skip
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=page_size,
            kv_layout="HND",
            causal=causal,
            need_lse=return_lse,
            backend="cudnn",
        ),
        match="unsupported q dtype",
    )
    if res is None:
        return
    pytest.fail(
        "EXPECT_FP8_Q flipped: port the legacy q_scale / k_scale / v_scale / "
        "o_data_type fixture (tests/attention/test_cudnn_prefill.py::"
        "test_cudnn_prefill_fp8) here"
    )
