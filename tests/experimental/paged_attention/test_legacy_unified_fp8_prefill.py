"""Legacy -> unified: tests/attention/test_fp8_prefill.py

The two paged-prefill tests run the legacy fixture through
``PagedAttention`` pinned to fa2, the backend the legacy wrappers name.  Same
grid, same seed, same tensors (the combined ``(pages, 2, ...)`` fp8 pool is
passed as the ``kv[:, 0]`` / ``kv[:, 1]`` views; the legacy CSR becomes
``PagedAttentionMetadata.csr``), so the node ids equal the legacy ids.  The
legacy prefill wrapper plans without ``causal``, i.e. non-causal; the unified
plans say so.  Each case checks the legacy assertion at the legacy tolerance,
and the output and LSE against the fp32 paged-attention oracle on the
dequantized pool.

``test_paged_fp8_h512_long_q_keeps_cta32`` also asserts the native wrapper's
private ``_plan_info`` tile choice, which has no unified counterpart.  The
decode, ragged and single-prefill tests are out of scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_fp8_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale",
        ["test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale"],
        "equivalent",
        "same grid, seed and tensors on fa2 (non-causal as legacy); legacy fp16-kernel "
        "reference at the legacy tolerance plus the fp32 oracle on the dequantized pool "
        "(output and LSE)",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_ragged_kv_cache_fp8",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_decode_with_prefill_with_paged_kv_cache",
        [],
        "out-of-scope",
        "decode test (q_len 1 against the decode wrapper); out of the prefill scope for now",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_ragged_fp8_h512_long_q_keeps_cta32",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_paged_fp8_h512_long_q_keeps_cta32",
        ["test_paged_fp8_h512_long_q_keeps_cta32"],
        "partial",
        "same fixture on fa2 with the legacy finite-output assertion plus the fp32 "
        "oracle; the legacy _plan_info[CTA_TILE_Q] == 32 check is a private scheduler "
        "detail of the native wrapper",
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
        "ragged KV wrapper, not paged prefill",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _uniform_csr(batch_size, qo_len, kv_len, kv_indices, page_size):
    """The legacy uniform batch (every request qo_len / kv_len, arange page
    ids) as unified CSR metadata."""
    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * qo_len
    kv_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)
    return PagedAttentionMetadata.csr(
        qo_indptr_cpu.to(0),
        kv_lens_cpu.to(0),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _oracle(md, q, k, v, kv_layout):
    """fp32 oracle, non-causal (the legacy plan default)."""
    return reference_paged_prefill(
        q,
        k,
        v,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        None,
        md.page_size,
        False,
        kv_layout=kv_layout,
        kv_page_indices=md.kv_page_indices,
    )


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

    skip_if_head_dim_unsupported(head_dim)

    # the legacy fixture, verbatim
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

    # the legacy reference: the fa2 wrapper on the fp16 pool
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
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
    o_fp16 = wrapper_f16.run(q, kv_data)
    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256
    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    md = _uniform_csr(batch_size, qo_len, kv_len, kv_indices, page_size)
    attn = PagedAttention(q.device)
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
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(
        q, (kv_data_fp8[:, 0], kv_data_fp8[:, 1]), k_scale=k_scale, v_scale=v_scale
    )

    # legacy assertion at the legacy tolerance
    torch.testing.assert_close(o_fp16, out, atol=1e-2, rtol=2e-1)

    # fp32 oracle on the dequantized pool: output and base-2 LSE
    ref_out, ref_lse = _oracle(
        md,
        q,
        k_fp8[:, 0].float() * k_scale,
        v_fp8[:, 0].float() * v_scale,
        kv_layout,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_paged_fp8_h512_long_q_keeps_cta32():
    from tests.attention.test_fp8_prefill import skip_if_head_dim_unsupported

    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)

    # the legacy fixture, verbatim
    torch.manual_seed(42)
    batch_size = 2
    qo_len = kv_len = 128
    page_size = 16
    num_qo_heads = num_kv_heads = 4
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
    kv_indices = torch.arange(0, total_num_pages).to(0).int()

    md = _uniform_csr(batch_size, qo_len, kv_len, kv_indices, page_size)
    attn = PagedAttention(q.device)
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
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (kv_data_fp8[:, 0], kv_data_fp8[:, 1]))

    # legacy assertion (the _plan_info tile check has no unified counterpart)
    assert torch.isfinite(out).all()

    # fp32 oracle: output and base-2 LSE
    ref_out, ref_lse = _oracle(
        md, q, kv_data_fp8[:, 0].float(), kv_data_fp8[:, 1].float(), "NHD"
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
