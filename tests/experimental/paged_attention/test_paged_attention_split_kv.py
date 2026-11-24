"""Split-KV control of PagedAttention (experimental): ``fixed_split_size`` and
``disable_split_kv``, the arguments of the fa2 planner that fix how a
request's KV range is split across kernel blocks.

What is pinned here, on fa2 (the one backend that declares the control):

- results against the fp32 oracle, and that the planner did what was asked
  (no split / the given chunk size), across layouts, head dims, page sizes,
  an fp8 KV cache and the feature axes;
- batch invariance: a request's output and LSE are bitwise the same in a
  batch and in a prefix of it, whenever the planner gives both the same query
  tile -- and the documented limit, as a strict expected failure: the tile
  follows the batch's average query length, which neither argument fixes;
- the contract: validation, capability exclusion, Resolution pinning, CUDA
  graphs (``disable_split_kv`` frozen by the first plan, ``fixed_split_size``
  rejected) and the scratch workspace (the per-batch need equals the
  planner's own count, and ``workspace_requirements()`` covers it).
"""

import re

import pytest
import torch

from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    GraphCapacity,
    PagedAttention,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_coverage import build_problem
from .test_paged_attention_prototype import LSE_TOL, OUT_TOL, make_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda:0")
CONTROLS = [
    dict(disable_split_kv=True),
    dict(fixed_split_size=8),
    dict(fixed_split_size=64),
]
CONTROL_IDS = ["no-split", "split-8", "split-64"]
_BASE = dict(
    num_qo_heads=8, num_kv_heads=2, head_dim_qk=128, page_size=16, dtype=torch.bfloat16
)


@pytest.fixture(autouse=True)
def _fa2_or_skip():
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
            disable_split_kv=True,
            backend="fa2",
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")


def _problem(q_lens, kv_lens, seed=0, **over):
    return build_problem(list(q_lens), list(kv_lens), seed=seed, **{**_BASE, **over})


def _view(p, q_lens, kv_lens):
    """The first ``len(q_lens)`` requests of ``p`` with these lengths, as a
    batch of its own on the same Q rows, KV pool and pages (a request keeps
    its inputs while the lengths before it and its own are unchanged)."""
    n, page = len(q_lens), p["page_size"]
    q_lens = torch.tensor(q_lens, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)
    assert bool((kv_lens <= p["kv_seq_lens_cpu"][:n]).all())
    qo = torch.cat(
        [torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0).to(torch.int32)]
    )
    assert int(qo[-1]) <= p["q"].shape[0]
    pages = (kv_lens + page - 1) // page
    tables = p["block_tables"][:n]
    return dict(
        p,
        q=p["q"][: int(qo[-1])],
        qo_indptr=qo.to(DEVICE),
        qo_indptr_cpu=qo,
        kv_seq_lens=kv_lens.to(DEVICE),
        kv_seq_lens_cpu=kv_lens,
        block_tables=tables,
        kv_page_indices=torch.cat([tables[i, : int(pages[i])] for i in range(n)]),
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
    )


def _prefix(p, n):
    q_lens = p["qo_indptr_cpu"].diff().tolist()[:n]
    return _view(p, q_lens, p["kv_seq_lens_cpu"].tolist()[:n])


def _plan_kw(p, **extra):
    kw = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        kv_layout=p.get("kv_layout", "HND"),
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    kw.update(extra)
    return kw


def _run(p, *, sinks=None, attn=None, **plan_extra):
    attn = attn or PagedAttention(DEVICE)
    attn.plan(
        make_metadata(p), **_plan_kw(p, use_sinks=sinks is not None, **plan_extra)
    )
    run_kw = dict(k_scale=p.get("k_scale"), v_scale=p.get("v_scale"))
    if sinks is not None:
        run_kw["sinks"] = sinks
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), **run_kw)
    return attn, out, lse


def _reference(p, *, causal=True, window_left=-1, **kw):
    return reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        causal,
        window_left=window_left,
        kv_layout=p.get("kv_layout", "HND"),
        **kw,
    )


def _assert_matches(out, lse, ref):
    torch.testing.assert_close(out.float(), ref[0], **OUT_TOL)
    torch.testing.assert_close(lse, ref[1], **LSE_TOL)


def _plan_info(attn):
    """(work items, query tile, split_kv) of the fa2 plan behind ``attn``:
    PrefillPlanInfo::ToVector (include/flashinfer/attention/scheduler.cuh)."""
    info = attn._impl._active._active._plan_info
    return int(info[0]), int(info[3]), bool(info[14])


def _expected_items(p, tile, split, window_left=-1):
    """The planner's work items for a fixed split of ``split`` pages: one per
    (request, query tile, KV chunk)."""
    gqa = p["num_qo_heads"] // p["num_kv_heads"]
    page = p["page_size"]
    items, any_split = 0, False
    for q_len, kv_len in zip(
        p["qo_indptr_cpu"].diff().tolist(), p["kv_seq_lens_cpu"].tolist(), strict=True
    ):
        pages = -(-kv_len // page)
        if window_left >= 0:
            pages = min(-(-(window_left + tile) // page), pages)
        chunks = -(-max(pages, 1) // split)
        any_split |= chunks > 1
        items += -(-q_len * gqa // tile) * chunks
    return items, any_split


def _assert_planned_as_asked(attn, p, control, window_left=-1):
    items, tile, split_kv = _plan_info(attn)
    if control.get("disable_split_kv"):
        assert not split_kv
    else:
        want = _expected_items(p, tile, control["fixed_split_size"], window_left)
        assert (items, split_kv) == want


# name: (q_lens, kv_lens, requests in the prefix, problem overrides, plan overrides)
CASES = {
    "uniform": ((128,) * 4, (4096,) * 4, 2, dict(num_qo_heads=16, num_kv_heads=4), {}),
    "mixed-kv": (
        (64,) * 3,
        (4096, 300, 9000),
        1,
        dict(num_qo_heads=16, num_kv_heads=4),
        {},
    ),
    "mixed-q": (
        (40, 700, 3),
        (3000, 700, 8000),
        2,
        dict(num_qo_heads=32, num_kv_heads=8),
        {},
    ),
    "nhd-fp16": (
        (100, 90, 110),
        (2000, 500, 3000),
        2,
        dict(kv_layout="NHD", dtype=torch.float16),
        {},
    ),
    "csr-page1": (
        (64,) * 3,
        (600, 200, 1000),
        1,
        dict(page_size=1, input_form="page_indices"),
        {},
    ),
    "head-dim-64": ((100,) * 3, (2000, 500, 3000), 2, dict(head_dim_qk=64), {}),
    "head-dim-256": ((100,) * 3, (2000, 500, 3000), 2, dict(head_dim_qk=256), {}),
    "fp8-kv": (
        (100,) * 3,
        (2000, 500, 3000),
        2,
        dict(kv_dtype=torch.float8_e4m3fn),
        {},
    ),
    "noncausal": ((100,) * 3, (2000, 500, 3000), 2, {}, dict(causal=False)),
    "window": ((100,) * 3, (2000, 500, 3000), 2, {}, dict(window_left=255)),
    "soft-cap": ((100,) * 3, (2000, 500, 3000), 2, {}, dict(logits_soft_cap=30.0)),
}


@pytest.mark.parametrize("control", CONTROLS, ids=CONTROL_IDS)
@pytest.mark.parametrize("case", sorted(CASES))
def test_split_kv_control_oracle_and_batch_invariance(case, control):
    """Each control against the oracle, planned as asked, and bitwise equal
    for the requests a batch shares with a prefix of it."""
    q_lens, kv_lens, n_sub, problem_kw, plan_kw = CASES[case]
    full = _problem(q_lens, kv_lens, seed=11, **problem_kw)
    sub = _prefix(full, n_sub)
    ref_kw = {
        k: plan_kw[k]
        for k in ("causal", "window_left", "logits_soft_cap")
        if k in plan_kw
    }
    window_left = plan_kw.get("window_left", -1)

    attn_full, out_full, lse_full = _run(full, **plan_kw, **control)
    _assert_matches(out_full, lse_full, _reference(full, **ref_kw))
    _assert_planned_as_asked(attn_full, full, control, window_left)

    attn_sub, out_sub, lse_sub = _run(sub, **plan_kw, **control)
    _assert_matches(out_sub, lse_sub, _reference(sub, **ref_kw))
    _assert_planned_as_asked(attn_sub, sub, control, window_left)

    # the precondition of the bitwise claim: both batches got the same tile
    assert _plan_info(attn_full)[1] == _plan_info(attn_sub)[1]
    rows = sub["q"].shape[0]
    assert torch.equal(out_full[:rows], out_sub)
    assert torch.equal(lse_full[:rows], lse_sub)


@pytest.mark.parametrize("control", CONTROLS, ids=CONTROL_IDS)
def test_split_kv_control_with_attention_sinks(control):
    """The AttentionSink kernel variant takes the same planner arguments."""
    full = _problem((100,) * 3, (2000, 500, 3000), seed=13)
    sub = _prefix(full, 2)
    g = torch.Generator().manual_seed(13)
    sinks = (torch.rand(full["num_qo_heads"], generator=g) * 5).to(DEVICE)

    attn_full, out_full, lse_full = _run(full, sinks=sinks, **control)
    _assert_matches(out_full, lse_full, _reference(full, sinks=sinks))
    _assert_planned_as_asked(attn_full, full, control)
    attn_sub, out_sub, lse_sub = _run(sub, sinks=sinks, **control)
    _assert_matches(out_sub, lse_sub, _reference(sub, sinks=sinks))

    assert _plan_info(attn_full)[1] == _plan_info(attn_sub)[1]
    rows = sub["q"].shape[0]
    assert torch.equal(out_full[:rows], out_sub)
    assert torch.equal(lse_full[:rows], lse_sub)


@pytest.mark.parametrize("control", CONTROLS, ids=CONTROL_IDS)
def test_split_kv_control_with_a_custom_mask(control):
    full = _problem((100,) * 3, (2000, 500, 3000), seed=17)
    sub = _prefix(full, 2)
    q_lens = full["qo_indptr_cpu"].diff().to(torch.int64)
    kv_lens = full["kv_seq_lens_cpu"].to(torch.int64)
    g = torch.Generator().manual_seed(17)
    mask = (torch.rand(int((q_lens * kv_lens).sum()), generator=g) < 0.7).to(DEVICE)
    sub_mask = mask[: int((q_lens[:2] * kv_lens[:2]).sum())]

    attn_full, out_full, lse_full = _run(full, custom_mask=mask, **control)
    _assert_matches(out_full, lse_full, _reference(full, custom_mask=mask))
    _assert_planned_as_asked(attn_full, full, control)
    attn_sub, out_sub, lse_sub = _run(sub, custom_mask=sub_mask, **control)
    _assert_matches(out_sub, lse_sub, _reference(sub, custom_mask=sub_mask))

    assert _plan_info(attn_full)[1] == _plan_info(attn_sub)[1]
    rows = sub["q"].shape[0]
    assert torch.equal(out_full[:rows], out_sub)
    assert torch.equal(lse_full[:rows], lse_sub)


@pytest.mark.parametrize("control", CONTROLS, ids=CONTROL_IDS)
def test_split_kv_control_does_not_fix_the_query_tile(control):
    """The limit of the control, as an expected failure.  Request 0 (8 query
    tokens, 32 packed rows) plans tile 64 alone and tile 128 next to a
    200-token request; both runs match the oracle, and the last bits differ.
    Bitwise-equal outputs here mean the limit is gone: this test then fails
    and asks for the docs (plan() docstring, package README) to be updated."""
    full = _problem((8, 200), (2048, 2048), seed=19)
    sub = _prefix(full, 1)
    attn_full, out_full, lse_full = _run(full, **control)
    attn_sub, out_sub, lse_sub = _run(sub, **control)
    _assert_matches(out_full, lse_full, _reference(full))
    _assert_matches(out_sub, lse_sub, _reference(sub))
    if _plan_info(attn_full)[1] == _plan_info(attn_sub)[1]:
        pytest.skip("the planner gave both batches the same tile on this device")
    if torch.equal(out_full[:8], out_sub) and torch.equal(lse_full[:8], lse_sub):
        pytest.fail(
            "request 0 is bitwise the same under two query tiles: the "
            "documented limit of the split-KV control no longer holds"
        )
    pytest.xfail(
        "the fa2 planner derives the query tile from the batch's average "
        "packed query length (FA2DetermineCtaTileQ), which neither split-KV "
        "control fixes: a request's last bits differ between two batches "
        "planned with different tiles"
    )


# --------------------------------------------------------------------------
# contract: validation, capability exclusion, pinning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad,match",
    [
        (dict(fixed_split_size=0), "fixed_split_size must be None"),
        (dict(fixed_split_size=-1), "fixed_split_size must be None"),
        (dict(fixed_split_size=True), "fixed_split_size must be None"),
        (dict(fixed_split_size=16.0), "fixed_split_size must be None"),
        (dict(disable_split_kv=1), "disable_split_kv must be a bool"),
        (dict(fixed_split_size=16, disable_split_kv=True), "the two contradict"),
    ],
)
def test_plan_validates_the_split_kv_control(bad, match):
    """Rejected before anything is planned; the previous plan stays runnable."""
    p = _problem((100,) * 2, (2000, 500), seed=23)
    attn, out, _ = _run(p, disable_split_kv=True)
    with pytest.raises(ValueError, match=match):
        attn.plan(make_metadata(p), **_plan_kw(p, **bad))
    again, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    assert torch.equal(again, out)


@pytest.mark.parametrize("control", CONTROLS[:2], ids=CONTROL_IDS[:2])
def test_split_kv_control_excludes_the_other_backends(control):
    """`auto` lands on fa2 and names the reason for every backend it dropped;
    naming one of them raises with the same reason."""
    p = _problem((128,) * 4, (4096,) * 4, seed=29, num_qo_heads=32, num_kv_heads=8)
    resolve_kw = dict(
        device=DEVICE,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
        need_lse=True,
    )
    plain = resolve_paged_attention(**resolve_kw)
    res = resolve_paged_attention(**resolve_kw, **control)
    assert res.backends == ("fa2",)
    for name in plain.backends:
        if name != "fa2":
            assert "split-KV control" in res.excluded[name]
            with pytest.raises(ValueError, match="split-KV control"):
                PagedAttention(DEVICE).plan(
                    make_metadata(p), **_plan_kw(p, backend=name, **control)
                )

    attn, out, lse = _run(p, backend="auto", **control)
    assert attn.backend == "fa2"
    assert "split-KV control" in attn.explain() or plain.backends == ("fa2",)
    _assert_matches(out, lse, _reference(p))


def test_split_kv_control_is_part_of_the_pinned_resolution():
    p = _problem((100,) * 2, (2000, 500), seed=31)
    resolve_kw = dict(
        device=DEVICE,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
        need_lse=True,
    )
    plain = resolve_paged_attention(**resolve_kw)
    pinned = resolve_paged_attention(**resolve_kw, fixed_split_size=8)
    attn = PagedAttention(DEVICE)
    # a Resolution resolved without the control does not cover a plan with it,
    # nor the other way round, nor another split size
    for res, control in (
        (plain, dict(fixed_split_size=8)),
        (plain, dict(disable_split_kv=True)),
        (pinned, {}),
        (pinned, dict(fixed_split_size=16)),
    ):
        with pytest.raises(ValueError, match="do not match the pinned Resolution"):
            attn.plan(make_metadata(p), **_plan_kw(p, backend=res, **control))
    _, out, lse = _run(p, attn=attn, backend=pinned, fixed_split_size=8)
    assert attn.backend == "fa2"
    _assert_matches(out, lse, _reference(p))


# --------------------------------------------------------------------------
# CUDA graphs
# --------------------------------------------------------------------------


def _capture(attn, q, k, v, out, lse):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    return g


def test_disable_split_kv_under_cuda_graph():
    """A graph instance plans every batch with its capacity's tile, so with
    ``disable_split_kv`` request 0 replays bitwise the same next to different
    requests; the first plan freezes the argument."""
    a = _problem((64, 100, 30), (2000, 500, 3000), seed=37)
    b = _view(a, (64, 80, 50), (2000, 300, 1000))
    attn = PagedAttention(DEVICE, use_cuda_graph=True)
    q, k, v = a["q"], a["k_cache"], a["v_cache"]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], a["num_qo_heads"], dtype=torch.float32, device=DEVICE)

    attn.plan(make_metadata(a), **_plan_kw(a, disable_split_kv=True))
    assert not _plan_info(attn)[2]
    g = _capture(attn, q, k, v, out, lse)
    g.replay()
    torch.cuda.synchronize()
    _assert_matches(out, lse, _reference(a))
    first = out[:64].clone(), lse[:64].clone()

    attn.plan(make_metadata(b), **_plan_kw(b, disable_split_kv=True))
    g.replay()
    torch.cuda.synchronize()
    _assert_matches(out, lse, _reference(b))
    assert torch.equal(out[:64], first[0]) and torch.equal(lse[:64], first[1])

    # frozen with the first plan; the rejected re-plan writes nothing
    with pytest.raises(ValueError, match="disable_split_kv changed"):
        attn.plan(make_metadata(a), **_plan_kw(a))
    g.replay()
    torch.cuda.synchronize()
    _assert_matches(out, lse, _reference(b))


def test_fixed_split_size_is_rejected_under_cuda_graph():
    """Whether a fixed split splits a batch depends on its KV lengths, and a
    captured graph launches what the first plan decided: refused, on a first
    plan (which leaves no capacity behind) and on a re-plan."""
    p = _problem((64, 100, 30), (2000, 500, 3000), seed=41)
    attn = PagedAttention(DEVICE, use_cuda_graph=True)
    with pytest.raises(ValueError, match="fixed_split_size=8 with use_cuda_graph=True"):
        attn.plan(make_metadata(p), **_plan_kw(p, fixed_split_size=8))
    assert attn.backend is None
    _, out, lse = _run(p, attn=attn, disable_split_kv=True)
    _assert_matches(out, lse, _reference(p))
    with pytest.raises(ValueError, match="fixed_split_size=8 with use_cuda_graph=True"):
        attn.plan(make_metadata(p), **_plan_kw(p, fixed_split_size=8))
    again, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    assert torch.equal(again, out)


# --------------------------------------------------------------------------
# scratch workspace
# --------------------------------------------------------------------------


def _planner_float_bytes(p, *, window_left=-1, **control):
    """What the fa2 planner allocates from the float workspace for ``p`` under
    ``control``: the legacy wrapper's ``workspace_size()``, the planner in
    counting mode."""
    page = p["page_size"]
    kv = p["kv_seq_lens_cpu"]
    pages = (kv + page - 1) // page
    indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), pages.cumsum(0).to(torch.int32)]
    )
    last = torch.where(pages > 0, kv - (pages - 1) * page, torch.tensor(page))
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(16, dtype=torch.uint8, device=DEVICE), "HND", backend="fa2"
    )
    float_bytes, _ = wrapper.workspace_size(
        p["qo_indptr_cpu"],
        indptr,
        torch.arange(int(indptr[-1]), dtype=torch.int32),
        last.to(torch.int32),
        p["num_qo_heads"],
        p["num_kv_heads"],
        p["head_dim_qk"],
        page,
        head_dim_vo=p["head_dim_vo"],
        causal=True,
        window_left=window_left,
        q_data_type=p["dtype"],
        kv_data_type=p.get("kv_dtype"),
        seq_lens=kv,
        **control,
    )
    return int(float_bytes)


def _need_via_plan(p, *, window_left=-1, **control):
    """The per-batch need of the plan-time check, read from the ValueError a
    16-byte buffer provokes; 0 when the plan goes through on 16 bytes."""
    attn = PagedAttention(
        DEVICE, workspace_buffer=torch.empty(16, dtype=torch.uint8, device=DEVICE)
    )
    try:
        attn.plan(make_metadata(p), **_plan_kw(p, window_left=window_left, **control))
    except ValueError as exc:
        m = re.search(r"needs (\d+) bytes of scratch workspace", str(exc))
        assert m, f"unexpected ValueError: {exc}"
        assert attn.backend is None
        return int(m.group(1))
    return 0


def _requirements(p, *, window_left=-1, use_cuda_graph=False, **control):
    return PagedAttention.workspace_requirements(
        GraphCapacity.from_metadata(make_metadata(p)),
        device=DEVICE,
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        window_left=window_left,
        use_cuda_graph=use_cuda_graph,
        backend="fa2",
        **control,
    )


WORKSPACE_SHAPES = {
    "uniform": ((128,) * 3, (2048,) * 3, {}),
    "ragged": ((40, 700, 3, 1), (3000, 700, 8000, 16), {}),
    "short-kv": ((64, 64), (100, 120), {}),
    "page1": ((64,) * 2, (300, 90), dict(page_size=1, input_form="page_indices")),
    "head-dim-256": ((100,) * 3, (2000, 500, 3000), dict(head_dim_qk=256)),
    "gqa-1": ((300, 8), (4096, 4096), dict(num_qo_heads=4, num_kv_heads=4)),
}


@pytest.mark.parametrize("window_left", [-1, 255])
@pytest.mark.parametrize("split", [1, 8, 64, 1000])
@pytest.mark.parametrize("shape", sorted(WORKSPACE_SHAPES))
def test_fixed_split_workspace_need_is_the_planners(shape, split, window_left):
    """The plan-time check declines a ``fixed_split_size`` batch for exactly
    the bytes the planner would allocate, and workspace_requirements() covers
    them (no kernel runs)."""
    q_lens, kv_lens, over = WORKSPACE_SHAPES[shape]
    p = _problem(q_lens, kv_lens, seed=43, **over)
    control = dict(fixed_split_size=split)
    planner = _planner_float_bytes(p, window_left=window_left, **control)
    assert _need_via_plan(p, window_left=window_left, **control) == planner
    assert _requirements(p, window_left=window_left, **control) >= planner


def test_split_kv_control_and_workspace_requirements():
    p = _problem((128,) * 3, (2048,) * 3, seed=47)
    default = _requirements(p)
    fixed = _requirements(p, fixed_split_size=1)
    need = _planner_float_bytes(p, fixed_split_size=1)
    assert default < need <= fixed

    # a buffer of the bound, and one of exactly the planner's bytes, plan and run
    for nbytes in (fixed, need):
        attn = PagedAttention(
            DEVICE,
            workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=DEVICE),
        )
        _, out, lse = _run(p, attn=attn, fixed_split_size=1)
        _assert_matches(out, lse, _reference(p))
    # the bound computed without the argument does not cover the plan
    small = PagedAttention(
        DEVICE, workspace_buffer=torch.empty(default, dtype=torch.uint8, device=DEVICE)
    )
    with pytest.raises(ValueError, match=rf"needs {need} bytes of scratch workspace"):
        small.plan(make_metadata(p), **_plan_kw(p, fixed_split_size=1))

    # no split, no scratch: in both modes the bound is 0 and 16 bytes plan and run
    assert _requirements(p, disable_split_kv=True) == 0
    assert _requirements(p, disable_split_kv=True, use_cuda_graph=True) == 0
    assert _planner_float_bytes(p, disable_split_kv=True) == 0
    tiny = PagedAttention(
        DEVICE, workspace_buffer=torch.empty(16, dtype=torch.uint8, device=DEVICE)
    )
    _, out, lse = _run(p, attn=tiny, disable_split_kv=True)
    _assert_matches(out, lse, _reference(p))

    # there is no graph-mode bound for a fixed split (plan() rejects it there)
    with pytest.raises(ValueError, match="use_cuda_graph=False"):
        _requirements(p, fixed_split_size=1, use_cuda_graph=True)
