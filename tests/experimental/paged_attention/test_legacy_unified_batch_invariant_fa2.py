"""Legacy -> unified: tests/attention/test_batch_invariant_fa2.py

``test_batch_prefill_tensor_cores`` checks that fa2 is batch invariant under
the planner controls ``fixed_split_size`` / ``disable_split_kv``: the first
``invariant_bs`` requests of a batch must give bitwise the same output and LSE
when they are planned on their own.  ``PagedAttention.plan()`` takes both
controls, so the test runs on the legacy grid (the node ids equal the legacy
ids) with the legacy tensors: fp16, one combined (K, V) pool, pages in order,
every request the same length, non-causal (the legacy ``plan()`` default).
As in the legacy test one instance plans the full batch and then its prefix.
Each case also checks the full batch against the fp32 oracle, which the
legacy test does not.

The control is declared by fa2 alone, so only fa2 runs; a GPU where it cannot
is skipped with the resolver's reason.  The ``ROPE_LLAMA`` rows are expected
failures: ``PagedAttention`` has no fused positional encoding.

Every request of the grid has the same query length, so the batch and its
prefix plan the same query tile; the bitwise claim does not extend to batches
whose average query length puts them on different tiles
(``test_paged_attention_split_kv.py``).

The legacy decode test (``BatchDecodeWithPagedKVCacheWrapper``) is out of
scope: ``PagedAttention`` is the paged-prefill API and has no decode route.
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
LEGACY_SOURCE = "tests/attention/test_batch_invariant_fa2.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_decode_tensor_cores",
        [],
        "out-of-scope",
        "decode entry (BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores); "
        "PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_prefill_tensor_cores",
        ["test_batch_prefill_tensor_cores"],
        "partial",
        "same grid and tensors on fa2 with plan(fixed_split_size=) / "
        "plan(disable_split_kv=True): the legacy bitwise comparison of a batch "
        "with its prefix, plus the full batch against the fp32 oracle; "
        "ROPE_LLAMA rows xfail (no fused RoPE)",
    ),
]

DEVICE = torch.device("cuda:0")
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="batch invariance tests require CUDA"
)


@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("invariant_bs", [2])
@pytest.mark.parametrize("kv_len", [4096, 5000])
@pytest.mark.parametrize("qo_len", [128, 256])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    qo_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    if pos_encoding_mode != "NONE":
        pytest.xfail("API limit: PagedAttention has no fused RoPE (pos_encoding_mode)")
    num_qo_heads = num_kv_heads * group_size
    # the legacy call: fixed_split_size unless the split is disabled
    control = (
        dict(disable_split_kv=True)
        if disable_split_kv
        else dict(fixed_split_size=fixed_split_size)
    )
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=False,
            need_lse=True,
            kv_input_form="page_indices",
            backend="fa2",
            **control,
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = (
        (total_num_pages, 2, num_kv_heads, page_size, head_dim)
        if kv_layout == "HND"
        else (total_num_pages, 2, page_size, num_kv_heads, head_dim)
    )
    kv_data = torch.randn(*kv_shape, device=DEVICE, dtype=torch.float16) / 10
    kv_cache = (kv_data[:, 0], kv_data[:, 1])  # PagedAttention takes the (K, V) pair
    kv_indices = torch.arange(total_num_pages, device=DEVICE, dtype=torch.int32)

    attn = PagedAttention(DEVICE)

    def run(num_requests):
        qo_indptr_cpu = torch.arange(num_requests + 1, dtype=torch.int32) * qo_len
        kv_seq_lens_cpu = torch.full((num_requests,), kv_len, dtype=torch.int32)
        metadata = PagedAttentionMetadata.csr(
            qo_indptr_cpu.to(DEVICE),
            kv_seq_lens_cpu.to(DEVICE),
            kv_indices[: num_requests * num_pages_per_seq],
            page_size=page_size,
            max_q_len=qo_len,
            max_kv_len=kv_len,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )
        attn.plan(
            metadata,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_layout=kv_layout,
            causal=False,
            lse_mode="base2",
            backend="fa2",
            **control,
        )
        out, lse = attn.run(q[: num_requests * qo_len], kv_cache)
        return out, lse, qo_indptr_cpu, kv_seq_lens_cpu

    o, lse, qo_indptr_cpu, kv_seq_lens_cpu = run(batch_size)
    o_ref, lse_ref = reference_paged_prefill(
        q,
        kv_cache[0],
        kv_cache[1],
        qo_indptr_cpu,
        kv_seq_lens_cpu,
        None,
        page_size,
        False,
        kv_layout=kv_layout,
        kv_page_indices=kv_indices,
    )
    torch.testing.assert_close(o.float(), o_ref, **OUT_TOL)
    torch.testing.assert_close(lse, lse_ref, **LSE_TOL)

    # Test invariant batch size
    o_invariant, lse_invariant, _, _ = run(invariant_bs)
    assert torch.equal(o[: invariant_bs * qo_len], o_invariant)
    assert torch.equal(lse[: invariant_bs * qo_len], lse_invariant)
