"""Legacy -> unified: tests/attention/test_hopper.py

``test_batch_paged_prefill`` runs the legacy fixture through two
``PagedAttention`` instances pinned to fa3 and fa2, the two backends the
legacy test compares.  Same grid, same seed, same tensors (the legacy CSR
becomes ``PagedAttentionMetadata.csr``), so the node ids equal the legacy
ids.  Each case checks the legacy fa2-vs-fa3 assertion at the legacy
tolerance, and both outputs and LSEs against the fp32 paged-attention oracle.
Like the legacy, every case skips without SM90a (so all of them skip on
B200).

The two multi-item-scoring tests pass ``prefix_len_ptr`` /
``token_pos_in_items_ptr`` / ``token_pos_in_items_len`` / ``max_item_len_ptr``
to ``plan``; ``PagedAttention.plan`` has no multi-item mask, and the tests
assert it rejects those keywords.  The single-prefill and ragged tests are out
of scope.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import is_sm90a_supported

from .paged_attention_reference import reference_paged_prefill

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
        "same grid, seed and tensors on fa3 and fa2; legacy fa2-vs-fa3 assertion at the "
        "legacy tolerance plus the fp32 oracle on both (output and LSE); SM90a only, as "
        "legacy",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3"],
        "unsupported-by-design",
        "plan() has no multi-item scoring mask: the prefix_len_ptr / token_pos_in_items "
        "keywords are rejected (TypeError); SM90a only, as legacy",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2"],
        "unsupported-by-design",
        "two-request variant of the row above; same rejection",
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
        qo_indptr_cpu.to(kv_indices.device),
        kv_lens_cpu.to(kv_indices.device),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _chunked_oracle(q, k, v, batch_size, seq_len, page_size, causal, cap):
    """fp32 oracle per request and per 512-query chunk, so the score matrix of
    the 32767-token rows fits.  Rows [a, b) of a causal request see the first
    b keys: exactly the envelope of a b - a query request over b keys."""
    pages_per_req = (seq_len + page_size - 1) // page_size
    outs, lses = [], []
    for i in range(batch_size):
        page_ids = torch.arange(
            i * pages_per_req, (i + 1) * pages_per_req, dtype=torch.int32
        )
        for a in range(0, seq_len, 512):
            b = min(seq_len, a + 512)
            o, lse = reference_paged_prefill(
                q[i * seq_len + a : i * seq_len + b],
                k,
                v,
                torch.tensor([0, b - a], dtype=torch.int32),
                torch.tensor([b if causal else seq_len], dtype=torch.int32),
                None,
                page_size,
                causal,
                kv_layout="NHD",
                kv_page_indices=page_ids.to(q.device),
                logits_soft_cap=cap,
            )
            outs.append(o)
            lses.append(lse)
    return torch.cat(outs), torch.cat(lses)


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

    # the legacy fixture, verbatim
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
    kv_indices = torch.arange(0, batch_size * num_pages_per_request).int().cuda()

    md = _uniform_csr(batch_size, seq_len, seq_len, kv_indices, page_size)
    cap = logits_soft_cap if logits_soft_cap > 0 else None
    results = {}
    for backend in ("fa2", "fa3"):
        attn = PagedAttention(q.device)
        attn.plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.half,
            kv_layout="NHD",
            causal=causal,
            lse_mode="base2",
            logits_soft_cap=cap,
            backend=backend,
        )
        assert attn.backend == backend
        results[backend] = attn.run(q, (k, v))
    o_sm80, lse_sm80 = results["fa2"]
    o_sm90, lse_sm90 = results["fa3"]

    # legacy assertion at the legacy tolerance: fa2 and fa3 agree
    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)

    # fp32 oracle on both backends: output and base-2 LSE
    ref_out, ref_lse = _chunked_oracle(
        q, k, v, batch_size, seq_len, page_size, causal, cap
    )
    for out, lse in results.values():
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _assert_multi_item_rejected(
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
    """The legacy multi-item fixture (unseeded, as legacy); plan() with the
    multi-item keywords raises TypeError on the first of them."""
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")

    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indices = torch.arange(0, total_num_pages).int().to(0)
    md = _uniform_csr(batch_size, qo_len, kv_len, kv_indices, page_size)

    attn = PagedAttention(q.device)
    with pytest.raises(TypeError, match="prefix_len_ptr"):
        attn.plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.half,
            kv_layout=kv_layout,
            causal=causal,
            lse_mode="base2" if return_lse else "none",
            logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
            backend="fa3",
            prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
            token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
            .to(dtype=torch.uint16)
            .to(0),
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=torch.tensor(max_item_len_ptr)
            .to(dtype=torch.uint16)
            .to(0),
        )


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
    _assert_multi_item_rejected(
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
    )


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
    _assert_multi_item_rejected(
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
    )
