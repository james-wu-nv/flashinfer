"""Legacy -> unified: tests/attention/test_sliding_window.py

``test_batch_paged_prefill_sliding_window`` runs the legacy fixture (fp16 NHD
pools, uniform lengths, ``kv_indices = arange``, causal sliding window) through
``PagedAttention``.  The legacy grid's own ``backend`` axis (fa2 / auto) becomes
the unified backends fa2, fa3, cudnn and trtllm-gen: the legacy wrapper's
"auto" runs fa2 (fa3 on SM90a at head_dim <= 256), so the ``-fa2`` ids equal the
legacy ids.  A backend that cannot run a case on this GPU is skipped with the
resolver's reason.

Like the legacy run, the plan requests no LSE, and each case checks the legacy
reference (``single_prefill_with_kv_cache`` per request, fa2) at the legacy
tolerance.

The decode, single-prefill and ragged legacy tests are out of scope:
``PagedAttention`` is the paged-prefill API.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

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
        "same grid; the legacy backend axis (fa2 / auto) becomes fa2 / fa3 / "
        "cudnn / trtllm-gen; no LSE like the legacy run; legacy per-request "
        "single_prefill reference at the legacy tolerance",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_ragged_prefill_sliding_window",
        [],
        "out-of-scope",
        "ragged KV (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


@pytest.mark.parametrize("batch_size", [12, 17, 30])
@pytest.mark.parametrize("kv_len", [54, 397, 1177])
@pytest.mark.parametrize("qo_len", [1, 37, 47])
@pytest.mark.parametrize("window_left", [13, 33, 111])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", BACKENDS)
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
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads < num_kv_heads is not supported")
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout="NHD",
            causal=True,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
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
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    metadata = PagedAttentionMetadata.csr(
        q_indptr,
        torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda:0"),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(torch.device("cuda:0"), workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout="NHD",
        causal=True,
        window_left=window_left,
        backend=backend,
    )
    assert attn.backend == backend
    o, _ = attn.run(
        q,
        (k_data, v_data),
    )

    for i in range(batch_size):
        qi = q[q_indptr[i] : q_indptr[i + 1]]
        ki = torch.cat(
            [
                k_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                k_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        vi = torch.cat(
            [
                v_data[kv_indptr[i] : kv_indptr[i + 1] - 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                v_data[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :],
            ],
            dim=0,
        )
        o_ref_i = flashinfer.single_prefill_with_kv_cache(
            qi, ki, vi, window_left=window_left, causal=True, backend="fa2"
        )
        o_i = o[q_indptr[i] : q_indptr[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-3, atol=1e-3)
