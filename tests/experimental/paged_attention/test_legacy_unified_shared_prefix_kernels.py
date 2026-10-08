"""Legacy -> unified: tests/attention/test_shared_prefix_kernels.py

``test_batch_attention_with_shared_prefix_paged_kv_cache`` builds the legacy
fixture (fp16 NHD pool filled with ``append_paged_kv_cache``: the shared pages
first, then each request's unique pages) and runs it through
``PagedAttention`` pinned to fa2, the backend the legacy cascade wrapper's
prefill levels use on this GPU.  Same grid and tensors, so the node ids equal
the legacy ids.  Cascade attention is not a ``PagedAttention`` feature, so each
case runs it two ways: one plan whose page lists are ``shared ++ unique`` per
request, and the two-level composition (one run over the shared pages, one
over the unique pages, merged with ``flashinfer.merge_state`` on the base-2
LSE).  Both are checked against the legacy reference
(``MultiLevelCascadeAttentionWrapper``) at the legacy tolerance and against the
fp32 paged-attention oracle (output and LSE).

``test_merge_state_in_place_with_mask`` tests the merge operator, not paged
attention: out of scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_shared_prefix_kernels.py"
LEGACY_MAP = [
    (
        "tests/attention/test_shared_prefix_kernels.py::test_batch_attention_with_shared_prefix_paged_kv_cache",
        ["test_batch_attention_with_shared_prefix_paged_kv_cache"],
        "partial",
        "same grid and pool on fa2; one plan over shared ++ unique pages and the "
        "two-level composition (two runs + merge_state), both vs "
        "MultiLevelCascadeAttentionWrapper at the legacy tolerance plus the fp32 "
        "oracle; the cascade operator itself is not a PagedAttention feature",
    ),
    (
        "tests/attention/test_shared_prefix_kernels.py::test_merge_state_in_place_with_mask",
        [],
        "out-of-scope",
        "merge_state_in_place operator, not paged attention",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def ceil_div(a, b):
    return (a + b - 1) // b


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
):
    if stage == "decode" and causal:
        pytest.skip("Causal attention is not required in decode stage")
    assert shared_kv_len % page_size == 0

    # the legacy fixture, verbatim (same RNG order: q, k/v_shared, k/v_unique)
    kv_layout = "NHD"
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

    # legacy reference: the two-level cascade wrapper
    multi_level_wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0), kv_layout
    )
    qo_indptr_top = torch.tensor([0, q.shape[0]], dtype=torch.int32).to(0)
    multi_level_wrapper.plan(
        [qo_indptr_top, q_indptr],
        [shared_kv_indptr, unique_kv_indptr],
        [shared_kv_indices, unique_kv_indices],
        [shared_last_page_len, unique_last_page_len],
        num_heads,
        num_heads,
        head_dim,
        page_size,
        causal=causal,
    )
    o_multi_level = multi_level_wrapper.run(q, kv_data)

    k_cache, v_cache = kv_data[:, 0], kv_data[:, 1]
    qo_indptr_cpu = q_indptr.cpu()
    q_len = int(qo_indptr_cpu[1])

    def plan_and_run(kv_len, page_ids, causal):
        """One fa2 plan over ``page_ids`` (per-request lists) and its run."""
        kv_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)
        md = PagedAttentionMetadata.csr(
            q_indptr,
            kv_lens_cpu.to(0),
            torch.cat(page_ids),
            page_size=page_size,
            max_q_len=q_len,
            max_kv_len=kv_len,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_lens_cpu,
        )
        attn = PagedAttention(torch.device("cuda:0"))
        attn.plan(
            md,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_layout=kv_layout,
            causal=causal,
            lse_mode="base2",
            backend="fa2",
        )
        assert attn.backend == "fa2"
        return md, *attn.run(q, (k_cache, v_cache))

    shared_ids = [shared_kv_indices] * batch_size
    unique_ids = list(unique_kv_indices.chunk(batch_size))

    # (a) one plan: every request reads the shared pages, then its own
    md, o, lse = plan_and_run(
        shared_kv_len + unique_kv_len,
        [torch.cat(ids) for ids in zip(shared_ids, unique_ids, strict=True)],
        causal,
    )
    torch.testing.assert_close(o, o_multi_level, rtol=1e-3, atol=1e-3)

    # (b) two levels: shared pages (non-causal) and unique pages, merged
    _, o_shared, lse_shared = plan_and_run(shared_kv_len, shared_ids, False)
    _, o_unique, lse_unique = plan_and_run(unique_kv_len, unique_ids, causal)
    o_merged, lse_merged = flashinfer.merge_state(
        o_shared, lse_shared, o_unique, lse_unique
    )
    torch.testing.assert_close(o_merged, o_multi_level, rtol=1e-3, atol=1e-3)

    # fp32 oracle on the one-plan batch: output and base-2 LSE, for both
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        None,
        page_size,
        causal,
        kv_layout=kv_layout,
        kv_page_indices=md.kv_page_indices,
    )
    for out, lse_ in ((o, lse), (o_merged, lse_merged)):
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse_, ref_lse, **LSE_TOL)
