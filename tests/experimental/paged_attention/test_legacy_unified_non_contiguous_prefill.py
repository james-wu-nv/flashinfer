"""Legacy -> unified: tests/attention/test_non_contiguous_prefill.py

``test_batch_paged_prefill_packed_input`` runs the legacy fixture through
``PagedAttention`` on fa2, fa3, cudnn and trtllm-gen: q is the head slice of a
fused QKV projection (token stride ``(Hq + 2 Hkv) * D``) over two fp16 NHD
pools at page 1 / 5 (the flat page-id form).  Node ids are the legacy ids with
a ``-<backend>`` suffix; a backend that cannot run a case on this GPU is
skipped with the resolver's reason.  The legacy wrapper's "auto" runs fa2
(fa3 on SM90a), so that row runs the legacy kernel.

Like the legacy run, the plan requests no LSE.  Each case keeps the legacy
comparison (slice vs ``q.contiguous()``) at the legacy tolerance; that check
only compares two runs, so both outputs are also checked against the fp32
oracle.

The single-prefill and ragged legacy tests are out of scope: ``PagedAttention``
is the paged-prefill API.
"""

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

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
        "same grid on fa2 / fa3 / cudnn / trtllm-gen (CSR form, page 1 / 5), "
        "no LSE like the legacy run; slice vs q.contiguous() at the legacy "
        "tolerance plus the fp32 oracle on both outputs",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS)
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
    backend,
):
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

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
    workspace_buffer = torch.empty(
        (256 * 1024 * 1024,), dtype=torch.uint8, device="cuda:0"
    )
    qo_indptr = torch.tensor(
        [i * seq_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    kv_seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda:0")
    paged_kv_indices = torch.tensor(
        list(range(num_pages)), dtype=torch.int32, device="cuda:0"
    )
    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        kv_seq_lens,
        paged_kv_indices,
        page_size=page_size,
        max_q_len=seq_len,
        max_kv_len=seq_len,
    )
    attn = PagedAttention(workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout="NHD",
        causal=causal,
        backend=backend,
    )
    assert attn.backend == backend

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
    # pretend that we have already appended k/v to paged_kv table
    q = q.view(-1, num_qo_heads, head_dim)
    o_packed, _ = attn.run(q, paged_kv_cache)
    o_contiguous, _ = attn.run(q.contiguous(), paged_kv_cache)
    torch.testing.assert_close(o_packed, o_contiguous, rtol=1e-3, atol=2e-3)

    # fp32 oracle, for both q layouts
    ref_out, _ = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr.cpu(),
        kv_seq_lens.cpu(),
        None,
        page_size,
        causal,
        kv_layout="NHD",
        kv_page_indices=paged_kv_indices,
    )
    for o in (o_packed, o_contiguous):
        torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
