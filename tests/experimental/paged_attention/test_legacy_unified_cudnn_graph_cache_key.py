"""Legacy -> unified: tests/attention/test_cudnn_graph_cache_key.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only, and only if the runner
image ships cudnn-frontend; the unified file is in no default lane
(tests/experimental is excluded by norecursedirs).

The cuDNN graph-cache regressions are process-global facts of the native
``cudnn_batch_prefill_with_kv_cache`` cache; the unified counterparts drive
the same cache through the facade with the legacy fixtures (imported where
the legacy module exposes them), so a stale replay surfaces as a wrong answer
against the legacy torch reference and the fp32 oracle here too.  Every
prefill row pins ``backend="cudnn"`` in the dense form.  Default run = every
prefill function (7 ids).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_cudnn_prefill_scale_in_graph_cache_key: the legacy call is RAGGED
  (token-unit kv_indptr, no table); the fixture is laid into a page-16 pool
  with an identity table over the same storage (view, no copy) so the paged
  facade can address it; one unified plan, two runs with the two scales.
- test_cudnn_prefill_table_width_in_graph_cache_key: the legacy 'reject a
  wider table for the width-64 max_kv' contract has no unified counterpart:
  the facade views the table to the exact width ceil(max_kv / page) before
  binding it, so the capacity-67 table is CORRECT for the width-64 problem
  here (asserted); the private _sdpa_prefill_key_fn distinctness assertion
  stays native-only.
- test_cudnn_prefill_failed_build_leaves_no_cache_entry: the legacy trigger
  (over-wide table -> CUDNN_STATUS_BAD_PARAM at finalize) cannot be reached
  through the facade for the same reason; the unified analog is the plan-time
  rejection of a table whose page dimension is not unit-stride (twice, same
  message) with the previous plan left runnable and correct (transactional
  publication) and a valid re-plan correct afterwards.
- the four test_cudnn_decode_* functions: cudnn_batch_decode_with_kv_cache
  (decode-only entry) -> out-of-scope.
"""

import math

import pytest
import torch

from .legacy_unified_helpers import (
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_cudnn_graph_cache_key.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_scale_in_graph_cache_key",
        ["test_cudnn_prefill_scale_in_graph_cache_key"],
        "partial",
        "the legacy ragged fixture (B2, q32, kv48, H4, D128, bf16, seed 0, scale s then "
        "3s) laid into a page-16 pool with an identity table over the same storage (view, "
        "no copy: the legacy call is ragged and has no paged spelling natively); one "
        "unified cudnn plan, two runs with the two scales, each vs the legacy torch "
        "reference (_reference, 2e-2) and the oracle (out + LSE)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_scale_in_graph_cache_key",
        [],
        "out-of-scope",
        "cudnn_batch_decode_with_kv_cache (decode-only entry)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_block_table_strides_in_graph_cache_key",
        ["test_cudnn_prefill_block_table_strides_in_graph_cache_key"],
        "equivalent",
        "same fixture (B3, H4, D128, page16, kv120, q32, seed 0, width+3 capacity table: "
        "column view vs contiguous copy, both orders), pinned cudnn, the same 2e-2 budget "
        "vs the legacy reference, plus the oracle (out + LSE)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_table_width_in_graph_cache_key",
        ["test_cudnn_prefill_table_width_in_graph_cache_key"],
        "partial",
        "the legacy width-64 / width-67 problems (imported _paged_problem, seed 0) are "
        "both correct through the facade in one process, plus the width-64 problem on a "
        "capacity-67 table (the facade takes the width-exact view, so the legacy 'reject "
        "a wider table' contract does not apply and the case is asserted correct instead) "
        "and the width-64 problem again; the private _sdpa_prefill_key_fn distinctness "
        "assertion stays native-only",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_failed_build_leaves_no_cache_entry",
        ["test_cudnn_prefill_failed_build_leaves_no_cache_entry"],
        "partial",
        "the legacy over-wide table no longer fails a cuDNN build through the facade (it "
        "is viewed to the exact width); the unified analog rejects at plan a table whose "
        "page dimension is not unit-stride, twice with the same message, and the previous "
        "plan stays runnable and correct (transactional publication); a valid re-plan is "
        "correct afterwards (legacy reference 2e-2 + oracle); same fixture (seed 1, B3, "
        "width 8)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_block_table_strides_in_graph_cache_key",
        [],
        "out-of-scope",
        "cudnn_batch_decode_with_kv_cache (decode-only entry)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_table_width_in_graph_cache_key",
        [],
        "out-of-scope",
        "cudnn_batch_decode_with_kv_cache (decode-only entry)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_failed_build_leaves_no_cache_entry",
        [],
        "out-of-scope",
        "cudnn_batch_decode_with_kv_cache (decode-only entry)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def _plan_cudnn(md, *, num_qo_heads, num_kv_heads, head_dim, kv_layout="HND"):
    return plan_pinned(
        "cudnn",
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout=kv_layout,
        causal=True,
        lse_mode="base2",
    )


def test_cudnn_prefill_scale_in_graph_cache_key():
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
    assert k_cache.data_ptr() == k.data_ptr()
    block_tables = torch.arange(pages, dtype=torch.int32, device=device).view(
        batch_size, -1
    )
    md = dense_metadata(qo_indptr, kv_lens, block_tables, page_size)
    attn = _plan_cudnn(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
    """The legacy fixture of test_cudnn_prefill_block_table_strides_in_graph_cache_key
    (inline in the legacy test, so rebuilt here verbatim)."""
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
def test_cudnn_prefill_block_table_strides_in_graph_cache_key(order):
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
    for table in tables:
        strides = tuple(table.stride())
        md = dense_metadata(f["qo_indptr"], f["kv_lens"], table, f["page_size"])
        attn = _plan_cudnn(
            md,
            num_qo_heads=f["num_heads"],
            num_kv_heads=f["num_heads"],
            head_dim=f["head_dim"],
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


def test_cudnn_prefill_table_width_in_graph_cache_key():
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
        attn = _plan_cudnn(md, num_qo_heads=4, num_kv_heads=4, head_dim=p["head_dim"])
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


def test_cudnn_prefill_failed_build_leaves_no_cache_entry():
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
    attn = _plan_cudnn(md, num_qo_heads=4, num_kv_heads=4, head_dim=p["head_dim"])
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
    attn = plan_pinned(
        "cudnn",
        md,
        attn=attn,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=p["head_dim"],
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
    )
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    o_out, o_lse = oracle(
        md, p["q"], p["k_cache"], p["v_cache"], causal=True, kv_layout="HND"
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
