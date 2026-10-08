"""Legacy -> unified: tests/attention/test_sliding_window.py

``test_batch_paged_prefill_sliding_window`` runs the legacy fixture (two fp16
NHD pools, uniform lengths, ``kv_indices = arange``, causal sliding window)
through ``PagedAttention`` pinned to fa2.  The legacy grid's own ``backend``
axis (fa2 / auto) is kept: the legacy wrapper's "auto" runs fa2 on this GPU,
so both rows pin fa2.  Same grid and tensors, so the node ids equal the legacy
ids.  Each case checks the legacy reference (``single_prefill_with_kv_cache``
per request, fa2) at the legacy tolerance, and the output and LSE against the
fp32 paged-attention oracle.

The decode, single-prefill and ragged legacy tests are out of scope:
``PagedAttention`` is the paged-prefill API.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import get_compute_capability

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_sliding_window.py"
LEGACY_MAP = [
    (
        "tests/attention/test_sliding_window.py::test_single_decode_sliding_window",
        [],
        "out-of-scope",
        "single_decode_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_decode_sliding_window",
        [],
        "out-of-scope",
        "decode entry (BatchDecodeWithPagedKVCacheWrapper); PagedAttention has no "
        "decode route",
    ),
    (
        "tests/attention/test_sliding_window.py::test_single_decode_prefill_sliding_window_match",
        [],
        "out-of-scope",
        "single decode vs single prefill, not paged prefill",
    ),
    (
        "tests/attention/test_sliding_window.py::test_single_prefill_sliding_window",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_paged_prefill_sliding_window",
        ["test_batch_paged_prefill_sliding_window"],
        "equivalent",
        "same grid and tensors; legacy backend axis kept (fa2 / auto, both run "
        "fa2 here); legacy per-request single_prefill reference at the legacy "
        "tolerance plus the fp32 oracle (output and LSE)",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_ragged_prefill_sliding_window",
        [],
        "out-of-scope",
        "ragged KV (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 397, 1177])
@pytest.mark.parametrize("qo_len", [1, 37, 47])
@pytest.mark.parametrize("window_left", [13, 33, 111])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_batch_paged_prefill_sliding_window(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    backend,
):
    if head_dim > 256 and get_compute_capability(torch.device("cuda:0"))[0] < 8:
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads < num_kv_heads is not supported")

    # the legacy fixture, verbatim (same RNG order: q, k, v)
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    k_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    v_data = torch.randn(
        total_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )

    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * qo_len
    kv_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)
    kv_indices = torch.arange(total_num_pages, dtype=torch.int32, device="cuda:0")
    md = PagedAttentionMetadata.csr(
        qo_indptr_cpu.to("cuda:0"),
        kv_lens_cpu.to("cuda:0"),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    attn = PagedAttention(torch.device("cuda:0"))
    # the legacy wrapper's "auto" runs fa2 on this GPU
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout="NHD",
        causal=True,
        window_left=window_left,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, lse = attn.run(q, (k_data, v_data))

    # legacy reference at the legacy tolerance
    for i in range(batch_size):
        pages = slice(i * num_pages_per_seq, (i + 1) * num_pages_per_seq)
        ki = k_data[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        vi = v_data[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        qi = q[i * qo_len : (i + 1) * qo_len]
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, window_left=window_left, causal=True, backend="fa2"
        )
        o_i = o[i * qo_len : (i + 1) * qo_len]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)

    # fp32 oracle: output and base-2 LSE
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_data,
        v_data,
        qo_indptr_cpu,
        kv_lens_cpu,
        None,
        page_size,
        True,
        window_left=window_left,
        kv_layout="NHD",
        kv_page_indices=kv_indices,
    )
    torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
