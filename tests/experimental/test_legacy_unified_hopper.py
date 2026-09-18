"""Legacy -> unified: tests/attention/test_hopper.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only (the one lane that
executes fa3); the unified file is in no default lane (tests/experimental is
excluded by norecursedirs).

``test_batch_paged_prefill`` compares the fa3 and fa2 wrappers on one paged
fixture; here the same fixture (seed 42, fp16, NHD separate K/V pools, the
legacy CSR mapped losslessly) runs through ``PagedAttention(backend="fa3")``
and ``PagedAttention(backend="fa2")``: the legacy fa2-vs-fa3 assertion at the
legacy 1e-3 budget, and both against the fp32 oracle (soft cap included; the
query-chunked oracle for the 9999 / 32767 rows).  fa3 needs SM90a, so on
B200 every row is collected and skips with the resolve reason ("fa3:
unsupported compute capability sm_10x"); the legacy rows skip there too
("SM90A is not supported").  The legacy 9999 / 32767 rows (up to 16 x 32767
tokens) carry the ``slow`` marker: ``FI_PARITY_SLOW=1`` runs them; the
default subset (seq 11 / 12 / 99 / 1763) covers every other axis value.

Note on ``-k "not fa3"`` (the shared Blackwell container's deselection):
the two multi-item legacy names contain ``fa3`` and are deselected by it,
although their rejection assertions run on any GPU; run this file without
``-k`` for the by-file junit (every fa3 row skips within seconds on B200).

H100 commands (repo root, the H100 lane):

    python -m pytest -q -ra -o faulthandler_timeout=300 \\
        tests/experimental/test_legacy_unified_hopper.py
    FI_PARITY_SLOW=1 python -m pytest -q -ra -o faulthandler_timeout=300 \\
        tests/experimental/test_legacy_unified_hopper.py

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3 (+ _bsz2):
  the plan arguments prefix_len_ptr / token_pos_in_items_ptr /
  token_pos_in_items_len / max_item_len_ptr (multi-item scoring mask) have
  no unified spelling: PagedAttention.plan() rejects them as unexpected
  keywords (asserted, EXPECT_MULTI_ITEM_SCORING).  The unified mask axes are
  causal / window_left / custom_mask (_capabilities.py), and fa3 declares
  supports_custom_mask=False (the SM90 kernels reject MaskMode.CUSTOM), so
  the visible set cannot be spelled as a bool mask on fa3 either; an adapter
  needs the multi-item mask as a plan axis (or a mask builder plus fa3
  custom-mask support).  The positive branch (unified vs the legacy fa2
  wrapper with the multi-item arguments) is written and runs once the flag
  flips.
- test_single_prefill: single_prefill_with_kv_cache -> out-of-scope.
- test_batch_ragged_prefill, test_deepseek_prefill: ragged wrappers ->
  out-of-scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_MULTI_ITEM_SCORING,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    csr_metadata_from_legacy,
    gated,
    oracle,
    plan_pinned,
    reference_long,
    resolve_or_skip,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_hopper.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
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
        "partial",
        "same grid (B1/4/8/16, seq 11/12/99/1763/9999/32767, page 1/16, Hq 1/4/8, Hkv "
        "1/4/8 with the legacy 'not divisible' skip, causal, D64/128/256, cap 0/30), seed "
        "42, fp16, NHD separate K/V pools, CSR form; unified fa3 vs unified fa2 at the "
        "legacy 1e-3 budget and both vs the oracle (query-chunked for the long rows); on "
        "B200 every row skips with the fa3 resolve reason (collect-check only), pending "
        "the H100 run; the 9999 / 32767 rows are slow-marked (FI_PARITY_SLOW=1)",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3"],
        "unsupported-by-design",
        "same grid (2 legacy cases, page 1/5/16, H4/32:4, D128, causal, NHD, cap 0/30, "
        "return_lse); prefix_len_ptr / token_pos_in_items_ptr / token_pos_in_items_len / "
        "max_item_len_ptr have no unified spelling: plan() rejects them as unexpected "
        "keywords (EXPECT_MULTI_ITEM_SCORING); fa3 also declares no custom mask, so the "
        "visible set cannot be a bool mask there either; the positive branch (unified fa2 "
        "and, where runnable, fa3 vs the legacy fa2 wrapper at 1e-3) runs once flipped",
    ),
    (
        "tests/attention/test_hopper.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring_fa3_bsz2"],
        "unsupported-by-design",
        "the two-request variant (per-request prefix lengths and item position tables) "
        "of the row above; same gap (EXPECT_MULTI_ITEM_SCORING), same positive branch",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


# ---------------------------------------------------------------------------
# test_batch_paged_prefill
# ---------------------------------------------------------------------------

SLOW = pytest.mark.slow


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
@pytest.mark.parametrize(
    "seq_len",
    [11, 12, 99, 1763, pytest.param(9999, marks=SLOW), pytest.param(32767, marks=SLOW)],
)
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
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be divisible by num_kv_heads")  # legacy skip
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
    # fa3 first: on non-SM90 hardware the row skips with the resolve reason
    # (the legacy "SM90A is not supported" skip)
    attn_fa3 = plan_pinned("fa3", md, **plan_kw)
    attn_fa2 = plan_pinned("fa2", md, **plan_kw)
    o_sm90, lse_sm90 = attn_fa3.run(q, (k, v))
    o_sm80, lse_sm80 = attn_fa2.run(q, (k, v))
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
    for o, lse in ((o_sm80, lse_sm80), (o_sm90, lse_sm90)):
        torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# multi-item scoring (fa3): no unified spelling for the item-position tables
# ---------------------------------------------------------------------------


def _multi_item_fixture(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
):
    """The legacy fixture (unseeded, as legacy): combined (pages, 2, ...) pool,
    uniform request lengths, arange page ids."""
    q = torch.randn(batch_size * qo_len, num_qo_heads, head_dim).to(0).half()
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(total_num_pages, 2, num_kv_heads, page_size, head_dim).to(0).half()
        if kv_layout == "HND"
        else torch.randn(total_num_pages, 2, page_size, num_kv_heads, head_dim)
        .to(0)
        .half()
    )
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    return dict(
        q=q,
        kv_data=kv_data,
        q_indptr=q_indptr_cpu,
        kv_indptr=kv_indptr_cpu,
        kv_indices=kv_indices_cpu,
        kv_last_page_len=kv_last_page_len_cpu,
    )


def _multi_item_unified(
    *,
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
    f = _multi_item_fixture(
        batch_size,
        kv_len,
        qo_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        kv_layout,
    )
    md = csr_metadata_from_legacy(
        f["q_indptr"], f["kv_indptr"], f["kv_indices"], f["kv_last_page_len"], page_size
    )
    multi_item = dict(
        prefix_len_ptr=torch.tensor(prefix_len_ptr).to(dtype=torch.uint32).to(0),
        token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
        .to(dtype=torch.uint16)
        .to(0),
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=torch.tensor(max_item_len_ptr).to(dtype=torch.uint16).to(0),
    )
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        lse_mode="base2" if return_lse else "none",
        logits_soft_cap=logits_soft_cap if logits_soft_cap > 0 else None,
    )
    res_fa2 = resolve_or_skip(
        "fa2",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=return_lse,
        logits_soft_cap=plan_kw["logits_soft_cap"],
        kv_input_form="page_indices",
    )
    attn = PagedAttention(torch.device(DEVICE))
    # the clear rejection: plan() has no multi-item keywords
    planned = gated(
        EXPECT_MULTI_ITEM_SCORING,
        lambda: attn.plan(md, backend=res_fa2, **plan_kw, **multi_item),
        match="prefix_len_ptr",
        exc=TypeError,
    )
    if planned is None and not EXPECT_MULTI_ITEM_SCORING:
        return

    # ---- positive branch (EXPECT_MULTI_ITEM_SCORING): the legacy fa2 wrapper
    # with the multi-item arguments is the reference (the legacy test compares
    # fa2 and fa3 with them); unified fa2 and, where runnable, fa3 ----
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper_fa2 = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fa2.plan(
        f["q_indptr"].to(0),
        f["kv_indptr"].to(0),
        f["kv_indices"].to(0),
        f["kv_last_page_len"].to(0),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        **multi_item,
    )
    o_ref, lse_ref = wrapper_fa2.run_return_lse(f["q"], f["kv_data"])
    k_view, v_view = f["kv_data"][:, 0], f["kv_data"][:, 1]
    outs = [attn.run(f["q"], (k_view, v_view))]
    try:
        attn_fa3 = PagedAttention(torch.device(DEVICE))
        attn_fa3.plan(md, backend="fa3", **plan_kw, **multi_item)
        outs.append(attn_fa3.run(f["q"], (k_view, v_view)))
    except ValueError:
        pass  # fa3 not runnable here (SM90a only)
    for out, lse in outs:
        torch.testing.assert_close(o_ref, out, rtol=1e-3, atol=1e-3)
        if return_lse:
            torch.testing.assert_close(lse_ref, lse, rtol=1e-3, atol=1e-3)


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
    _multi_item_unified(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        prefix_len_ptr=prefix_len_ptr,
        token_pos_in_items_ptr=token_pos_in_items_ptr,
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=max_item_len_ptr,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        causal=causal,
        kv_layout=kv_layout,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
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
    _multi_item_unified(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        prefix_len_ptr=prefix_len_ptr,
        token_pos_in_items_ptr=token_pos_in_items_ptr,
        token_pos_in_items_len=token_pos_in_items_len,
        max_item_len_ptr=max_item_len_ptr,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        causal=causal,
        kv_layout=kv_layout,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
