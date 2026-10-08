"""Legacy -> unified: tests/attention/test_non_contiguous_prefill.py

``test_batch_paged_prefill_packed_input`` runs the legacy fixture through
``PagedAttention`` pinned to fa2, the backend the legacy wrapper picks on this
GPU: q is the head slice of a fused QKV projection (token stride
``(Hq + 2 Hkv) * D``) over two fp16 NHD pools at page 1 / 5 (the flat page-id
form).  Same grid and tensors, so the node ids equal the legacy ids.  Each case
checks the legacy comparison (slice vs ``q.contiguous()``) at the legacy
tolerance, and both outputs and LSEs against the fp32 paged-attention oracle.

The single-prefill and ragged legacy tests are out of scope: ``PagedAttention``
is the paged-prefill API.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_non_contiguous_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_non_contiguous_prefill.py::test_single_prefill_packed_input",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache on a packed QKV view, not paged prefill",
    ),
    (
        "tests/attention/test_non_contiguous_prefill.py::test_batch_ragged_prefill_packed_input",
        [],
        "out-of-scope",
        "ragged KV (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill",
    ),
    (
        "tests/attention/test_non_contiguous_prefill.py::test_batch_paged_prefill_packed_input",
        ["test_batch_paged_prefill_packed_input"],
        "equivalent",
        "same grid and tensors on fa2 (CSR form, page 1 / 5); slice vs "
        "q.contiguous() at the legacy tolerance plus the fp32 oracle on both "
        "(output and LSE)",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [1, 19, 99])
@pytest.mark.parametrize("page_size", [1, 5])
@pytest.mark.parametrize("seq_len", [1, 7, 127, 257])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
def test_batch_paged_prefill_packed_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")

    # the legacy fixture, verbatim (same RNG order: k, v, qkv)
    nnz = batch_size * seq_len
    num_pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * num_pages_per_req
    k_cache = torch.randn(
        size=(num_pages, page_size, num_kv_heads, head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)

    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * seq_len
    kv_lens_cpu = torch.full((batch_size,), seq_len, dtype=torch.int32)
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda:0")
    md = PagedAttentionMetadata.csr(
        qo_indptr_cpu.to("cuda:0"),
        kv_lens_cpu.to("cuda:0"),
        kv_page_indices,
        page_size=page_size,
        max_q_len=seq_len,
        max_kv_len=seq_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    attn = PagedAttention(torch.device("cuda:0"))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout="NHD",
        causal=causal,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"

    qkv_packed = torch.randn(
        size=(nnz, (num_qo_heads + 2 * num_kv_heads) * head_dim),
        dtype=torch.float16,
        device="cuda:0",
    )
    qkv_split_idx = (
        num_qo_heads * head_dim,
        num_kv_heads * head_dim,
        num_kv_heads * head_dim,
    )
    q, _, _ = qkv_packed.split(qkv_split_idx, dim=-1)
    q = q.view(-1, num_qo_heads, head_dim)
    out_packed, lse_packed = attn.run(q, paged_kv_cache)
    out_contig, lse_contig = attn.run(q.contiguous(), paged_kv_cache)

    # legacy assertion at the legacy tolerance
    torch.testing.assert_close(out_packed, out_contig, rtol=1e-3, atol=2e-3)

    # fp32 oracle: output and base-2 LSE, for both q layouts
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        kv_lens_cpu,
        None,
        page_size,
        causal,
        kv_layout="NHD",
        kv_page_indices=kv_page_indices,
    )
    for out, lse in ((out_packed, lse_packed), (out_contig, lse_contig)):
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
