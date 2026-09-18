"""Legacy -> unified parity, group B: the cuDNN paged-prefill entries.

Every legacy fixture is rebuilt as the legacy test built it (same seeds,
same ``as_strided`` combined pool, same page tables) and run through
``PagedAttention(backend="cudnn")``; the assertions are the legacy ones (the
legacy fa2 reference at the legacy budget) plus the fp32 oracle.

The cuDNN graph-cache regressions are process-global facts of the native
``cudnn_batch_prefill_with_kv_cache`` cache; the unified counterparts drive
the same cache through the facade with the legacy fixtures, so a stale
replay would surface as a wrong answer against the oracle here too.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FP8_Q,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    dense_metadata,
    gated,
    oracle,
    resolve_or_skip,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_MAP = [
    # (legacy nodeid or function, unified test function(s) in this file, status, note)
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill",
        ["test_cudnn_prefill"],
        "equivalent",
        "same grid (B1/4, s_qo 8/17/700, s_kv 8/32/1066, page 8/16/64, Hkv 1/4, H4, "
        "causal, return_lse), seed 1, the legacy as_strided combined pool handed over as "
        "the same K/V views (no copy), dense form with the legacy block table; asserted "
        "vs the legacy fa2 reference at the legacy budget (3e-3, 1e-2) and vs the oracle; "
        "return_lse (unused by the legacy body) maps to lse_mode base2/none and the LSE is "
        "checked against the oracle; is_cuda_graph_compatible=[True] is a no-op axis",
    ),
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill_fp8",
        ["test_cudnn_prefill_fp8_q_unsupported"],
        "unsupported-by-design",
        "fp8 Q with q_scale / o_data_type has no unified spelling (EXPECT_FP8_Q, "
        "EXPECT_OUTPUT_DTYPE); the legacy function itself xfails on Blackwell before "
        "running (legacy-known-xfail), so there is no legacy pass to inherit",
    ),
    (
        "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py::test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy",
        ["test_cudnn_paged_prefill_cu_seqlens"],
        "partial",
        "the legacy fixture (_make_paged_inputs) is imported and run through the facade; "
        "unified fixes token-unit offsets, so the legacy direct-vs-element-offset gate "
        "comparison (a monkeypatched private version gate) stays native-only; asserted vs "
        "the native cuDNN call at the legacy budget (1e-2) and vs the oracle",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_scale_in_graph_cache_key",
        ["test_cudnn_scale_in_graph_cache_key"],
        "partial",
        "the legacy ragged fixture (B2, q32, kv48, H4, D128, bf16, scale s then 3s) laid "
        "into a page-16 pool with an identity table (the legacy call is ragged: no paged "
        "spelling exists for it natively); one unified plan, two runs with the two scales, "
        "each vs the legacy torch reference (2e-2) and the oracle",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_block_table_strides_in_graph_cache_key",
        ["test_cudnn_block_table_strides_in_graph_cache_key"],
        "equivalent",
        "same fixture (B3, H4, D128, page16, kv120, q32, width+3 table, column view vs "
        "contiguous, both orders), same 2e-2 budget vs the legacy reference, plus the oracle",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_table_width_in_graph_cache_key",
        ["test_cudnn_table_width_in_graph_cache_key"],
        "partial",
        "the legacy width-64 / width-67 problems are imported and both correct through "
        "the facade, plus the width-64 problem on a capacity-67 table (unified takes the "
        "width-exact view, so the legacy 'reject a wider table' contract does not apply); "
        "the private _sdpa_prefill_key_fn distinctness assertion stays native-only",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_failed_build_leaves_no_cache_entry",
        ["test_cudnn_failed_plan_leaves_previous_plan_runnable"],
        "partial",
        "the legacy over-wide table no longer fails a cuDNN build through the facade (it "
        "is viewed to the exact width); the unified analog rejects a table whose page "
        "dimension is not unit-stride, twice with the same message, and the previous "
        "plan stays runnable and correct (transactional publication)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())


_legacy_ws = None


def _legacy_workspace(device):
    global _legacy_ws
    if _legacy_ws is None:
        _legacy_ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    return _legacy_ws


def _plan_cudnn(
    md, *, num_qo_heads, num_kv_heads, head_dim, causal, lse_mode, kv_layout="HND"
):
    res = resolve_or_skip(
        "cudnn",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        page_size=md.page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=lse_mode != "none",
        kv_input_form="block_tables",
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout=kv_layout,
        causal=causal,
        lse_mode=lse_mode,
        backend=res,
    )
    assert attn.backend == "cudnn"
    return attn


# ---------------------------------------------------------------------------
# test_cudnn_prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
def test_cudnn_prefill(
    batch_size, s_qo, s_kv, page_size, num_kv_heads, num_qo_heads, causal, return_lse
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")  # the legacy skip

    # ---- the legacy fixture, verbatim ----
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]
    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size, dim=0
            ),
        ]
    ).int()
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )
    scale = float(1.0 / (head_dim**0.5))
    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # ---- the legacy reference: the fa2 wrapper on the combined pool ----
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _legacy_workspace(device), "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )
    output_ref = wrapper.run(q, kv_cache)

    # ---- unified cuDNN: dense form, the legacy K/V views (HND shape,
    # NHD-ordered strides: the graph is stride-driven), legacy table ----
    assert k_cache.data_ptr() == kv_cache.data_ptr()  # views, no copy
    md = dense_metadata(qo_indptr, actual_seq_lens_kv.view(-1), block_tables, page_size)
    lse_mode = "base2" if return_lse else "none"
    attn = _plan_cudnn(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        lse_mode=lse_mode,
    )
    output, lse = attn.run(q, (k_cache, v_cache), sm_scale=scale)

    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)  # legacy
    ref_out, ref_lse = oracle(
        md, q, k_cache, v_cache, causal=causal, kv_layout="HND", sm_scale=scale
    )
    torch.testing.assert_close(output.float(), ref_out, **OUT_TOL)
    if return_lse:
        assert lse.shape == (q.shape[0], num_qo_heads) and lse.dtype == torch.float32
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    else:
        assert lse is None


# ---------------------------------------------------------------------------
# test_cudnn_prefill_fp8 (legacy: xfail on Blackwell; fp8 Q has no unified spelling)
# ---------------------------------------------------------------------------


def test_cudnn_prefill_fp8_q_unsupported():
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=128,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=16,
            kv_layout="HND",
            causal=True,
            need_lse=False,
            backend="cudnn",
        ),
        match="unsupported q dtype",
    )
    if res is None:
        return
    pytest.fail(
        "EXPECT_FP8_Q flipped: port the legacy q_scale / o_data_type fixture "
        "(tests/attention/test_cudnn_prefill.py::test_cudnn_prefill_fp8) here"
    )


# ---------------------------------------------------------------------------
# test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(64, 512), (17, 200)])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_paged_prefill_cu_seqlens(
    batch_size, s_qo, s_kv, page_size, num_kv_heads, causal
):
    from tests.attention.test_cudnn_prefill_paged_cu_seqlens import _make_paged_inputs

    device = "cuda:0"
    num_qo_heads, head_dim = 8, 128
    inp = _make_paged_inputs(
        batch_size, s_qo, s_kv, page_size, num_qo_heads, num_kv_heads, head_dim, device
    )
    scale = float(head_dim**-0.5)

    # the legacy reference: the native call with the default (production) gate
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    out_native = cudnn_batch_prefill_with_kv_cache(
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        scale,
        ws,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=inp["actual_seq_lens_q"],
        actual_seq_lens_kv=inp["actual_seq_lens_kv"],
        block_tables=inp["block_tables"],
        causal=causal,
        return_lse=False,
        batch_offsets_q=inp["qo_indptr"],
        batch_offsets_units="tokens",
    )[0]

    md = dense_metadata(
        inp["qo_indptr"],
        inp["actual_seq_lens_kv"].view(-1),
        inp["block_tables"],
        page_size,
    )
    attn = _plan_cudnn(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        lse_mode="none",  # legacy return_lse=False
    )
    out, lse = attn.run(inp["q"], (inp["k_cache"], inp["v_cache"]), sm_scale=scale)
    assert lse is None
    torch.testing.assert_close(out, out_native, atol=1e-2, rtol=1e-2)  # legacy budget
    ref_out, _ = oracle(
        md,
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        causal=causal,
        kv_layout="HND",
        sm_scale=scale,
    )
    torch.testing.assert_close(out.float(), ref_out, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# test_cudnn_graph_cache_key.py — prefill entries
# ---------------------------------------------------------------------------


def test_cudnn_scale_in_graph_cache_key():
    """Same shapes twice, different scales, ONE plan: a stale-scale graph
    replay (the legacy bug) shows up against the legacy torch reference and
    the oracle at the second scale."""
    from tests.attention.test_cudnn_graph_cache_key import _reference

    device = "cuda:0"
    torch.manual_seed(0)
    batch_size, num_qo_heads, num_kv_heads, head_dim = 2, 4, 4, 128
    q_lens = torch.tensor([32, 32], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([48, 48], dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    q = torch.randn(
        int(q_lens.sum()), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        int(kv_lens.sum()), num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    # the legacy call is ragged; the paged spelling: page 16, three pages per
    # request, identity table over the same storage (no copy)
    page_size = 16
    pages = int(kv_lens.sum()) // page_size
    k_cache = k.view(pages, page_size, num_kv_heads, head_dim)  # NHD
    v_cache = v.view(pages, page_size, num_kv_heads, head_dim)
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).view(
        batch_size, -1
    )
    md = dense_metadata(qo_indptr, kv_lens, block_tables, page_size)
    attn = _plan_cudnn(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=True,
        lse_mode="base2",
        kv_layout="NHD",
    )
    for scale_mult in (1.0, 3.0):
        scale = scale_mult / math.sqrt(head_dim)
        out, lse = attn.run(q, (k_cache, v_cache), sm_scale=scale)
        ref = _reference(q, k, v, q_lens, kv_lens, scale, causal=True)
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, s=scale: f"scale={s}: stale-scale graph replay?\n{m}",
        )
        o_out, o_lse = oracle(
            md, q, k_cache, v_cache, causal=True, kv_layout="NHD", sm_scale=scale
        )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def _legacy_table_stride_fixture(device):
    torch.manual_seed(0)
    batch_size, num_heads, head_dim, page_size = 3, 4, 128, 16
    kv_len, q_len = 120, 32
    width = (kv_len + page_size - 1) // page_size
    pool_pages = batch_size * width + 8
    q_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    q = torch.randn(
        batch_size * q_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_cache = torch.randn(
        pool_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    perm = torch.randperm(pool_pages, dtype=torch.int32, device=device)
    wide = torch.zeros(batch_size, width + 3, dtype=torch.int32, device=device)
    wide[:, :width] = perm[: batch_size * width].view(batch_size, width)
    view = wide[:, :width]
    packed = view.contiguous()
    assert view.stride(0) == width + 3 and packed.stride(0) == width

    def reference():
        outs = []
        for i in range(batch_size):
            pages = packed[i].to(torch.int64)
            k_i = (
                k_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(num_heads, -1, head_dim)[:, :kv_len]
                .float()
            )
            v_i = (
                v_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(num_heads, -1, head_dim)[:, :kv_len]
                .float()
            )
            q_i = q[i * q_len : (i + 1) * q_len].float()
            scores = torch.einsum("qhd,hkd->hqk", q_i, k_i) / math.sqrt(head_dim)
            qpos = torch.arange(q_len, device=device).unsqueeze(1)
            kpos = torch.arange(kv_len, device=device).unsqueeze(0)
            allowed = kpos <= (kv_len - q_len) + qpos
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            outs.append(torch.einsum("hqk,hkd->qhd", torch.softmax(scores, -1), v_i))
        return torch.cat(outs)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        qo_indptr=qo_indptr,
        kv_lens=kv_lens,
        view=view,
        packed=packed,
        page_size=page_size,
        num_heads=num_heads,
        head_dim=head_dim,
        reference=reference,
    )


@pytest.mark.parametrize("order", ["view_then_contiguous", "contiguous_then_view"])
def test_cudnn_block_table_strides_in_graph_cache_key(order):
    """The legacy fixture: a column view of a wider table (row stride width+3)
    and its contiguous copy, same values, in both orders, one process.  A
    graph cached without the table strides in its key replays the wrong
    pages (71.7% wrong elements in the legacy report)."""
    device = "cuda:0"
    f = _legacy_table_stride_fixture(device)
    ref = f["reference"]()
    tables = (
        (f["view"], f["packed"])
        if order == "view_then_contiguous"
        else (f["packed"], f["view"])
    )
    attn = None
    for table in tables:
        strides = tuple(table.stride())
        md = dense_metadata(f["qo_indptr"], f["kv_lens"], table, f["page_size"])
        attn = _plan_cudnn(
            md,
            num_qo_heads=f["num_heads"],
            num_kv_heads=f["num_heads"],
            head_dim=f["head_dim"],
            causal=True,
            lse_mode="base2",
        )
        out, lse = attn.run(f["q"], (f["k_cache"], f["v_cache"]))
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, st=strides: (
                f"block_tables strides {st}: stale-stride graph replay?\n{m}"
            ),
        )
        o_out, o_lse = oracle(
            md, f["q"], f["k_cache"], f["v_cache"], causal=True, kv_layout="HND"
        )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def test_cudnn_table_width_in_graph_cache_key():
    """The legacy width-64 and width-67 problems (imported), both correct in
    one process, then the width-64 problem on a capacity-67 table.  Through
    the facade the wider table is viewed to the exact width (cuDNN's
    finalize demands width == ceil(max_kv / page)), so the legacy 'reject the
    wider table' contract has no unified counterpart; its distinct-key
    assertion (a private key function) stays native-only."""
    from tests.attention.test_cudnn_graph_cache_key import _paged_problem

    device = "cuda:0"
    torch.manual_seed(0)
    p64 = _paged_problem(device, batch_size=2, width=64)
    p67 = _paged_problem(device, batch_size=2, width=67)
    wide = torch.zeros(2, 67, dtype=torch.int32, device=device)
    wide[:, :64] = p64["block_tables"]
    cases = [
        (p64, p64["block_tables"], "width-64 exact"),
        (p67, p67["block_tables"], "width-67 exact"),
        (p64, wide, "width-64 problem on a capacity-67 table (view path)"),
        (p64, p64["block_tables"], "width-64 exact again"),
    ]
    for p, table, what in cases:
        md = dense_metadata(p["qo_indptr"], p["kv_lens"], table, 16)
        attn = _plan_cudnn(
            md,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=p["head_dim"],
            causal=True,
            lse_mode="base2",
        )
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        torch.testing.assert_close(
            out.float(),
            p["reference"](),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, w=what: f"{w}: stale-width graph replay?\n{m}",
        )
        o_out, o_lse = oracle(
            md, p["q"], p["k_cache"], p["v_cache"], causal=True, kv_layout="HND"
        )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def test_cudnn_failed_plan_leaves_previous_plan_runnable():
    """Unified analog of the legacy 'failed build leaves no cache entry': a
    plan the cuDNN backend rejects (page dimension not unit-stride) fails
    twice with the same message and the previous plan stays runnable and
    correct; a valid re-plan afterwards is correct too.  The legacy trigger
    (an over-wide table -> CUDNN_STATUS_BAD_PARAM at finalize) cannot be
    reached through the facade: the table is viewed to the exact width."""
    from tests.attention.test_cudnn_graph_cache_key import _paged_problem

    device = "cuda:0"
    torch.manual_seed(1)
    p = _paged_problem(device, batch_size=3, width=8)
    ref = p["reference"]()
    md = dense_metadata(p["qo_indptr"], p["kv_lens"], p["block_tables"], 16)
    attn = _plan_cudnn(
        md,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=p["head_dim"],
        causal=True,
        lse_mode="base2",
    )
    out, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # a (3, 8) table whose page dimension has stride 3: same values, a layout
    # the cuDNN backend refuses at plan (it would bake the strides in)
    transposed = p["block_tables"].t().contiguous().t()
    assert transposed.shape == p["block_tables"].shape and transposed.stride(1) == 3
    assert torch.equal(transposed, p["block_tables"])
    bad_md = dense_metadata(p["qo_indptr"], p["kv_lens"], transposed, 16)
    messages = []
    for _ in range(2):  # the second attempt must fail the same way
        with pytest.raises(ValueError, match="unit-stride along the page") as info:
            attn.plan(
                bad_md,
                num_qo_heads=4,
                num_kv_heads=4,
                head_dim_qk=p["head_dim"],
                q_dtype=torch.bfloat16,
                causal=True,
                lse_mode="base2",
                backend="cudnn",
            )
        messages.append(str(info.value))
    assert messages[0] == messages[1]
    # the previous plan is still the published one and still correct
    assert attn.backend == "cudnn"
    out, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    # and a valid re-plan is correct
    attn.plan(
        md,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=p["head_dim"],
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
        backend="cudnn",
    )
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    o_out, o_lse = oracle(
        md, p["q"], p["k_cache"], p["v_cache"], causal=True, kv_layout="HND"
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
