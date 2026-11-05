"""Legacy -> unified: tests/attention/test_hopper.py

``test_batch_paged_prefill`` checks the Hopper fa3 paged-prefill kernel
against the fa2 wrapper.  Here the fa3 side runs through ``PagedAttention``
(``backend="fa3"``); the reference is the legacy one, the fa2 wrapper on the
same 256 MiB workspace, at the legacy tolerance (output and LSE).  Same grid,
same seed, same tensors (the legacy page ids become
``PagedAttentionMetadata.csr``), so the node ids equal the legacy ids.  Like
the legacy, every case skips without SM90a, so all of them skip on B200; a
GPU where fa3 cannot run the case is skipped with the resolver's reason.

The two multi-item-scoring tests pass ``prefix_len_ptr`` /
``token_pos_in_items_ptr`` / ``token_pos_in_items_len`` / ``max_item_len_ptr``
to ``plan``.  ``PagedAttention.plan`` has no such inputs, so they are expected
failures (after the legacy SM90a skip).  The single-prefill and ragged tests
are out of scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from flashinfer.utils import is_sm90a_supported

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_hopper.py"
LEGACY_MAP = [
    (
        "tests/attention/test_hopper.py::test_single_prefill",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_ragged_prefill",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_hopper.py::test_deepseek_prefill",
        [],
        "out-of-scope",
        "ragged KV wrapper (head_dim 192 / 128), not paged prefill",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_paged_prefill",
        ["test_batch_paged_prefill"],
        "equivalent",
        "same grid, seed and tensors; fa3 through PagedAttention vs the legacy fa2 "
        "wrapper reference on the legacy 256 MiB workspace, output and LSE at the "
        "legacy 1e-3; SM90a only, as legacy",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3"],
        "unsupported-by-design",
        "xfail after the legacy SM90a skip: plan() has no multi-item scoring inputs "
        "(API limit)",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2"],
        "unsupported-by-design",
        "two-request variant of the row above; same xfail",
    ),
]

MULTI_ITEM_XFAIL = (
    "API limit: PagedAttention.plan() has no multi-item scoring inputs "
    "(prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, "
    "max_item_len_ptr)"
)


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [11, 12, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_qo_heads", [1, 4, 8])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_paged_prefill(
    batch_size,
    seq_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    causal,
    head_dim,
    logits_soft_cap,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.half,
            page_size=page_size,
            kv_layout="NHD",
            causal=causal,
            need_lse=True,
            logits_soft_cap=logits_soft_cap or None,
            kv_input_form="page_indices",
            backend="fa3",
        )
    except ValueError as e:
        pytest.skip(f"fa3 not runnable here: {e}")
    torch.random.manual_seed(42)
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=torch.half, device="cuda"
    )
    num_pages_per_request = (seq_len + page_size - 1) // page_size
    k = torch.randn(
        batch_size * num_pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )
    v = torch.randn(
        batch_size * num_pages_per_request,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.half,
        device="cuda",
    )

    workspace_buffer = torch.empty(
        256 * 1024 * 1024, dtype=torch.uint8, device="cuda:0"
    )

    # the legacy reference: the fa2 wrapper
    wrapper_sm80 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, backend="fa2"
    )

    last_page_len = seq_len - (num_pages_per_request - 1) * page_size
    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(
        0, batch_size * num_pages_per_request + 1, num_pages_per_request
    ).int()
    kv_indices = torch.arange(0, batch_size * num_pages_per_request).int()
    last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32)

    wrapper_sm80.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )
    o_sm80, lse_sm80 = wrapper_sm80.run_return_lse(q, (k, v))

    # fa3 through PagedAttention, on the same workspace
    metadata = PagedAttentionMetadata.csr(
        qo_indptr.to(0),
        torch.full((batch_size,), seq_len, dtype=torch.int32).to(0),  # kv_seq_lens
        kv_indices.to(0),
        page_size=page_size,
        max_q_len=seq_len,
        max_kv_len=seq_len,
    )
    attn = PagedAttention(torch.device("cuda:0"), workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.half,
        kv_layout="NHD",
        causal=causal,
        lse_mode="base2",
        logits_soft_cap=logits_soft_cap or None,
        backend="fa3",
    )
    assert attn.backend == "fa3"
    o_sm90, lse_sm90 = attn.run(q, (k, v))

    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
        (97, 81, 16, list(range(80)) + [0], 97, [79]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    logits_soft_cap,
    return_lse,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(MULTI_ITEM_XFAIL)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (
            54,
            37,
            [17, 17],
            list(range(17))
            + list(range(19))
            + [0]
            + [0] * 63
            + list(range(15))
            + list(range(21))
            + [0],
            100,
            [18, 20],
        ),
        (
            97,
            81,
            [16, 16],
            list(range(80)) + [0] * 17 + list(range(76)) + [0] * 5,
            97,
            [79, 75],
        ),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    logits_soft_cap,
    return_lse,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(MULTI_ITEM_XFAIL)
