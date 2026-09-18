"""Legacy -> unified parity, group B: the Hopper (fa3) paged entries.

``tests/attention/test_hopper.py::test_batch_paged_prefill`` compares the
fa3 and fa2 wrappers on the same paged fixture.  Here the same fixture runs
through ``PagedAttention(backend="fa3")`` and ``PagedAttention(backend="fa2")``:
the legacy fa2-vs-fa3 assertion at the legacy budget (1e-3), and both
against the fp32 oracle (soft cap included).  On B200 every fa3 row skips
with the resolve reason (fa3 needs SM90a); the file is a collect-check
there.  The test names contain ``fa3`` so ``-k "not fa3"`` deselects them on
the shared Blackwell container.

H100 commands (repo root, the H100 lane):

    python -m pytest -q -ra -o faulthandler_timeout=300 \
        tests/experimental/test_paged_attention_legacy_parity_b_hopper.py
    FI_PARITY_SLOW=1 python -m pytest -q -ra -o faulthandler_timeout=300 \
        tests/experimental/test_paged_attention_legacy_parity_b_hopper.py
    python -m pytest -q -ra -o faulthandler_timeout=300 \
        tests/experimental/test_paged_attention_legacy_parity_b_sinks.py -k fa3

Default subset: seq_len 11 / 12 / 99 / 1763; ``FI_PARITY_SLOW=1`` adds the
legacy 9999 and 32767 rows (up to 16 x 32767 tokens).
"""

import inspect

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .legacy_parity_b_helpers import (
    DEVICE,
    EXPECT_MULTI_ITEM_SCORING,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    csr_metadata_from_legacy,
    oracle,
    reference_long,
    resolve_or_skip,
    slow_case,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_MAP = [
    # (legacy nodeid or function, unified test function(s) in this file, status, note)
    (
        "tests/attention/test_hopper.py::test_batch_paged_prefill",
        ["test_hopper_batch_paged_prefill_fa3"],
        "partial",
        "same grid (B1/4/8/16, seq 11/12/99/1763/9999/32767, page 1/16, H (1,4,8) x (1,4,8) "
        "divisible, causal, D64/128/256, cap 0/30), seed 42, fp16, NHD separate K/V pools, "
        "CSR form; unified fa3 vs unified fa2 at the legacy 1e-3 budget and both vs the "
        "oracle (chunked for the long rows); B200: every row skips with the fa3 resolve "
        "reason (collect-check only), pending the H100 run; seq >= 9999 needs FI_PARITY_SLOW=1",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3",
        ["test_hopper_multi_item_scoring_fa3_unsupported"],
        "unsupported-by-design",
        "prefix_len_ptr / token_pos_in_items_ptr / token_pos_in_items_len / max_item_len_ptr "
        "have no unified spelling (EXPECT_MULTI_ITEM_SCORING); fa3 also declares no custom "
        "mask, so the visible set cannot be spelled as a bool mask on that backend either; "
        "an adapter needs the multi-item mask as a plan axis (or a mask builder plus fa3 "
        "custom-mask support) with the legacy cap 30 / return_lse combinations",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2",
        ["test_hopper_multi_item_scoring_fa3_unsupported"],
        "unsupported-by-design",
        "the two-request variant of the row above (per-request prefix lengths and item "
        "position tables); same gap",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())


def _resolve(
    backend, *, num_qo_heads, num_kv_heads, head_dim, causal, logits_soft_cap, page_size
):
    return resolve_or_skip(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        page_size=page_size,
        kv_layout="NHD",
        causal=causal,
        need_lse=True,
        logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
        kv_input_form="page_indices",
    )


HEAD_PAIRS = [
    (q, kv) for q in (1, 4, 8) for kv in (1, 4, 8) if q % kv == 0
]  # the legacy grid minus its "not divisible" skips


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [11, 12, 99, 1763, 9999, 32767])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", HEAD_PAIRS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_hopper_batch_paged_prefill_fa3(
    batch_size,
    seq_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    causal,
    head_dim,
    logits_soft_cap,
):
    res_fa3 = _resolve(
        "fa3",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        page_size=page_size,
    )
    res_fa2 = _resolve(
        "fa2",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        page_size=page_size,
    )
    if seq_len >= 9999:
        slow_case(f"batch {batch_size} x seq {seq_len}")

    # ---- the legacy fixture, verbatim ----
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
    last_page_len = seq_len - (num_pages_per_request - 1) * page_size
    qo_indptr = torch.arange(0, batch_size * seq_len + 1, seq_len).int()
    kv_indptr = torch.arange(
        0, batch_size * num_pages_per_request + 1, num_pages_per_request
    ).int()
    kv_indices = torch.arange(0, batch_size * num_pages_per_request).int()
    last_page_len = torch.full((batch_size,), last_page_len, dtype=torch.int32)

    md = csr_metadata_from_legacy(
        qo_indptr, kv_indptr, kv_indices, last_page_len, page_size
    )
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout="NHD",
        causal=causal,
        lse_mode="base2",
        logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
    )
    outs = {}
    for name, res in (("fa2", res_fa2), ("fa3", res_fa3)):
        attn = PagedAttention(torch.device(DEVICE))
        attn.plan(md, backend=res, **plan_kw)
        assert attn.backend == name
        outs[name] = attn.run(q, (k, v))
    (o_sm80, lse_sm80), (o_sm90, lse_sm90) = outs["fa2"], outs["fa3"]
    # the legacy assertion: fa2 and fa3 agree
    torch.testing.assert_close(lse_sm80, lse_sm90, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_sm80, o_sm90, rtol=1e-3, atol=1e-3)
    # both against the oracle (soft cap applied to the scaled scores); the
    # long rows use the query-chunked oracle
    cap = logits_soft_cap if logits_soft_cap > 0 else None
    if seq_len >= 9999:
        ref_out, ref_lse = reference_long(
            md, q, k, v, causal=causal, kv_layout="NHD", logits_soft_cap=cap
        )
    else:
        ref_out, ref_lse = oracle(
            md, q, k, v, causal=causal, kv_layout="NHD", logits_soft_cap=cap
        )
    for o, lse in outs.values():
        torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_hopper_multi_item_scoring_fa3_unsupported():
    """The multi-item scoring plan arguments have no unified spelling; the
    row flips when EXPECT_MULTI_ITEM_SCORING lands (a plan axis carrying the
    per-request prefix length and per-token item positions, or a mask
    builder plus fa3 custom-mask support)."""
    params = inspect.signature(PagedAttention.plan).parameters
    present = all(
        name in params
        for name in (
            "prefix_len_ptr",
            "token_pos_in_items_ptr",
            "token_pos_in_items_len",
            "max_item_len_ptr",
        )
    )
    assert present == EXPECT_MULTI_ITEM_SCORING
    if present:
        pytest.fail(
            "EXPECT_MULTI_ITEM_SCORING flipped: port the two legacy fixtures here"
        )
