"""Legacy -> unified: tests/attention/test_shared_prefix_kernels.py

``test_batch_attention_with_shared_prefix_paged_kv_cache`` builds the legacy
fixture (fp16 NHD pool filled with ``append_paged_kv_cache``: the shared pages
first, then each request's unique pages).  Its subject,
``MultiLevelCascadeAttentionWrapper``, is two paged-prefill levels merged
with ``merge_state_in_place``; the unified test computes exactly that with
``PagedAttention``, once per backend (fa2, fa3, cudnn and trtllm-gen; node ids
are the legacy ids with a ``-<backend>`` suffix, a backend that cannot run the
case on this GPU is skipped with the resolver's reason):

- top level: every query as ONE request over the shared pages, non-causal;
- bottom level: each request over its unique pages, ``causal`` as legacy;
- the bottom level runs first and the top level is merged into it with
  ``flashinfer.merge_state_in_place`` on the base-2 LSEs, as
  ``MultiLevelCascadeAttentionWrapper.run`` does; both plans share the legacy
  32 MiB workspace.

The result is checked against the legacy reference, the deprecated
``Batch{Decode,Prefill}WithSharedPrefixPagedKVCacheWrapper``, at the legacy
tolerance.  The legacy wrapper's ``auto`` picks fa2 (fa3 on SM90a), so that row
runs the legacy kernels.

``test_merge_state_in_place_with_mask`` tests the merge operator, not paged
attention: out of scope.
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
LEGACY_SOURCE = "tests/attention/test_shared_prefix_kernels.py"
LEGACY_MAP = [
    (
        "tests/attention/test_shared_prefix_kernels.py::test_batch_attention_with_shared_prefix_paged_kv_cache",
        ["test_batch_attention_with_shared_prefix_paged_kv_cache"],
        "equivalent",
        "same grid and pool; the MultiLevel two-level cascade computed with "
        "PagedAttention on fa2 / fa3 / cudnn / trtllm-gen (top level: all queries "
        "over the shared pages; bottom: unique pages; merge_state_in_place) vs the "
        "legacy SharedPrefix wrappers at the legacy tolerance",
    ),
    (
        "tests/attention/test_shared_prefix_kernels.py::test_merge_state_in_place_with_mask",
        [],
        "out-of-scope",
        "merge_state_in_place operator, not paged attention",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


def ceil_div(a, b):
    return (a + b - 1) // b


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("stage", ["decode", "append"])
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("unique_kv_len", [37, 17])
@pytest.mark.parametrize("shared_kv_len", [128, 512, 2048])
@pytest.mark.parametrize("num_heads", [8, 16])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("page_size", [1, 16])
def test_batch_attention_with_shared_prefix_paged_kv_cache(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
    backend,
):
    if stage == "decode" and causal:
        pytest.skip("Causal attention is not required in decode stage")
    assert shared_kv_len % page_size == 0
    kv_layout = "NHD"
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=True,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    if stage == "append":
        q = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    else:
        q = torch.randn(batch_size, num_heads, head_dim).to(0).half()
        q_indptr = torch.arange(0, batch_size + 1).to(0).int()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(0).half()
    k_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()
    v_unique = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(0).half()

    kv_data = (
        torch.zeros(
            ceil_div(shared_kv_len, page_size)
            + batch_size * ceil_div(unique_kv_len, page_size),
            2,
            page_size,
            num_heads,
            head_dim,
        )
        .to(0)
        .half()
    )
    shared_kv_indices = torch.arange(0, ceil_div(shared_kv_len, page_size)).to(0).int()
    shared_append_indptr = torch.arange(0, 2).to(0).int() * shared_kv_len
    shared_kv_indptr = torch.arange(0, 2).to(0).int() * ceil_div(
        shared_kv_len, page_size
    )
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            k_shared.shape[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )
    unique_kv_indices = torch.arange(
        0, batch_size * ceil_div(unique_kv_len, page_size)
    ).to(0).int() + ceil_div(shared_kv_len, page_size)
    unique_append_indptr = torch.arange(0, batch_size + 1).to(0).int() * unique_kv_len
    unique_kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * ceil_div(
        unique_kv_len, page_size
    )
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            k_unique.shape[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )

    # MultiLevelCascadeAttentionWrapper with PagedAttention: two levels
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    kv_cache = (kv_data[:, 0], kv_data[:, 1])
    plan_kw = dict(
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        lse_mode="base2",
        backend=backend,
    )

    # top level: every query as one request over the shared pages (non-causal)
    qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32).to(0)
    metadata_top = PagedAttentionMetadata.csr(
        qo_indptr_top,
        torch.tensor([shared_kv_len], dtype=torch.int32).to(0),
        shared_kv_indices,
        page_size=page_size,
        max_q_len=q.shape[0],
        max_kv_len=shared_kv_len,
    )
    attn_top = PagedAttention(workspace)
    attn_top.plan(metadata_top, **plan_kw, causal=False)
    assert attn_top.backend == backend

    # bottom level: each request over its unique pages
    if stage == "decode":
        qo_indptr_bottom = torch.arange(0, batch_size + 1, dtype=torch.int32).to(0)
    else:
        qo_indptr_bottom = (
            torch.arange(0, batch_size + 1, dtype=torch.int32).to(0) * unique_kv_len
        )
    metadata_bottom = PagedAttentionMetadata.csr(
        qo_indptr_bottom,
        torch.full((batch_size,), unique_kv_len, dtype=torch.int32).to(0),
        unique_kv_indices,
        page_size=page_size,
        max_q_len=q.shape[0] // batch_size,
        max_kv_len=unique_kv_len,
    )
    attn_bottom = PagedAttention(workspace)
    attn_bottom.plan(metadata_bottom, **plan_kw, causal=causal)
    assert attn_bottom.backend == backend

    # run as MultiLevelCascadeAttentionWrapper.run does: the bottom level,
    # then the top level merged into it
    o_cascade, lse_cascade = attn_bottom.run(q, kv_cache)
    o_top, lse_top = attn_top.run(q, kv_cache)
    flashinfer.merge_state_in_place(o_cascade, lse_cascade, o_top, lse_top)

    if stage == "decode":
        shared_prefix_decode_wrapper = (
            flashinfer.BatchDecodeWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )
        shared_prefix_decode_wrapper.begin_forward(
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_decode_wrapper.forward(
            q, k_shared, v_shared, kv_data
        )
    else:
        shared_prefix_prefill_wrapper = (
            flashinfer.BatchPrefillWithSharedPrefixPagedKVCacheWrapper(
                torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
            )
        )
        shared_prefix_prefill_wrapper.begin_forward(
            q_indptr,
            unique_kv_indptr,
            unique_kv_indices,
            unique_last_page_len,
            num_heads,
            num_heads,
            head_dim,
            page_size,
        )
        o_two_level = shared_prefix_prefill_wrapper.forward(
            q, k_shared, v_shared, kv_data, causal=causal
        )

    torch.testing.assert_close(o_cascade, o_two_level, rtol=1e-3, atol=1e-3)
