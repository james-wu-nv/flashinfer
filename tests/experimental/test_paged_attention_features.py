"""Feature axes of PagedAttention (experimental) against the oracle:
logits soft cap, custom masks and attention sinks, plus their plan-time
contract (pinning, validation, capability exclusion).

fa3 cases skip where the GPU is not SM90a; cuDNN never runs a feature (it is
capability-excluded); trtllm-gen runs sinks only.
"""

import math

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_SHAPE = dict(
    batch_size=4,
    max_q=48,
    max_kv=320,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)


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
    )
    kw.update(extra)
    return kw


def _resolve_kw(p, **extra):
    kw = dict(
        device=torch.device(p["device"]),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        page_size=p["page_size"],
        kv_layout=p.get("kv_layout", "HND"),
        need_lse=True,
    )
    kw.update(extra)
    return kw


def _run(p, backend, *, sinks=None, sm_scale=None, **plan_extra):
    attn = PagedAttention(torch.device(p["device"]))
    attn.plan(make_metadata(p), backend=backend, **_plan_kw(p, **plan_extra))
    run_kw = dict(sm_scale=sm_scale, k_scale=p.get("k_scale"), v_scale=p.get("v_scale"))
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


def _assert_matches(out, lse, ref_out, ref_lse):
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    assert lse.shape == ref_lse.shape and lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _skip_unless_runnable(p, backend, **features):
    try:
        return resolve_paged_attention(backend=backend, **_resolve_kw(p, **features))
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")


# --------------------------------------------------------------------------
# logits soft cap
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("logits_soft_cap", [30.0, 50.0])
@pytest.mark.parametrize(
    "causal,window_left", [(True, -1), (True, 16), (False, -1), (True, 0)]
)
def test_logits_soft_cap(backend, logits_soft_cap, causal, window_left):
    p = make_problem(seed=int(logits_soft_cap) + window_left + 7 * causal, **_SHAPE)
    _skip_unless_runnable(
        p,
        backend,
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
    )
    # scores are pushed toward the cap so the tanh matters at these values
    p["q"] = p["q"] * 4
    _, out, lse = _run(
        p,
        backend,
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
    )
    ref_out, ref_lse = _reference(
        p, causal=causal, window_left=window_left, logits_soft_cap=logits_soft_cap
    )
    _assert_matches(out, lse, ref_out, ref_lse)
    # and the cap really changed the answer relative to the uncapped oracle
    # (window_left=0 is a one-key softmax: the cap cannot show there)
    if window_left != 0:
        plain_out, _ = _reference(p, causal=causal, window_left=window_left)
        assert not torch.allclose(ref_out, plain_out, **OUT_TOL)


def test_logits_soft_cap_fp8_kv_fa2():
    p = make_problem(seed=91, **dict(_SHAPE, kv_dtype=torch.float8_e4m3fn))
    _skip_unless_runnable(p, "fa2", logits_soft_cap=30.0)
    p["q"] = p["q"] * 4
    _, out, lse = _run(p, "fa2", logits_soft_cap=30.0)
    ref_out, ref_lse = _reference(p, logits_soft_cap=30.0)
    _assert_matches(out, lse, ref_out, ref_lse)


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_logits_soft_cap_sees_the_k_scaled_logits(backend):
    """k_scale on a bf16 cache folds into the softmax scale BEFORE the soft
    cap, as K * k_scale would (the legacy wrapper's order): cap 30, k 0.5 /
    v 2.0 against the oracle on the scaled K / V."""
    p = make_problem(seed=95, **_SHAPE)
    _skip_unless_runnable(p, backend, logits_soft_cap=30.0)
    p["q"] = p["q"] * 4
    p["k_scale"], p["v_scale"] = 0.5, 2.0
    _, out, lse = _run(p, backend, logits_soft_cap=30.0)
    s = 1.0 / math.sqrt(p["head_dim_qk"])
    scaled = dict(p, v_ref=p["v_ref"].float() * 2.0)
    ref_out, ref_lse = _reference(scaled, sm_scale=s * 0.5, logits_soft_cap=30.0)
    _assert_matches(out, lse, ref_out, ref_lse)
    # the order matters at these magnitudes: capping the unscaled logits
    # gives another result
    other_out, _ = _reference(scaled, logits_soft_cap=30.0)
    assert not torch.allclose(ref_out, other_out, **OUT_TOL)


def test_logits_soft_cap_excludes_backends_at_plan():
    p = make_problem(seed=93, **_SHAPE)
    res = _resolve_or_skip(p, "auto")
    attn = PagedAttention(torch.device(p["device"]))
    # a Resolution resolved without the feature cannot be used with it (pinning)
    with pytest.raises(ValueError, match="pinned Resolution"):
        attn.plan(make_metadata(p), backend=res, **_plan_kw(p, logits_soft_cap=30.0))
    # auto: only the fa backends remain, with reasons for the others
    attn.plan(make_metadata(p), backend="auto", **_plan_kw(p, logits_soft_cap=30.0))
    assert attn.backend in ("fa2", "fa3")
    text = attn.explain()
    # only a backend this architecture admits is excluded FOR the feature; on
    # other architectures the capability gate speaks first (review R3)
    admitted = set(res.backends)
    for backend in ("cudnn", "trtllm-gen"):
        if backend in admitted:
            assert f"{backend}: logits soft cap not supported" in text
        with pytest.raises(
            ValueError,
            match="logits soft cap not supported"
            if backend in admitted
            else "logits soft cap not supported|unsupported compute capability",
        ):
            attn.plan(
                make_metadata(p), backend=backend, **_plan_kw(p, logits_soft_cap=30.0)
            )


# --------------------------------------------------------------------------
# custom masks
# --------------------------------------------------------------------------


def _tree_problem(seed, *, prefix_lens, depth=2, branch=2):
    """A speculative-decoding tree per request: a root, ``branch`` children,
    ``branch**2`` grandchildren ... appended after a committed prefix.  Every
    tree token attends the whole prefix and its own ancestors (including
    itself) and NOTHING else - in particular two tokens at the same depth
    never attend each other although the causal envelope would allow it."""
    nodes = sum(branch**d for d in range(depth + 1))
    parent = [-1]
    for n in range(1, nodes):
        parent.append((n - 1) // branch)
    b = len(prefix_lens)
    q_lens = torch.full((b,), nodes, dtype=torch.int32)
    kv_lens = torch.tensor(prefix_lens, dtype=torch.int32) + nodes
    p = make_problem(
        seed=seed,
        batch_size=b,
        max_q=nodes,
        max_kv=int(kv_lens.max()),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    # rebuild the lengths deterministically (make_problem randomizes them)
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages = (kv_lens + p["page_size"] - 1) // p["page_size"]
    width = int(pages.max())
    pool = p["k_cache"].shape[0]
    assert int(pages.sum()) <= pool
    perm = torch.randperm(pool, generator=torch.Generator().manual_seed(seed))
    bt = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages[i])
        bt[i, :n] = perm[off : off + n].to(torch.int32)
        off += n
    dev = torch.device(p["device"])
    total_q = int(qo_indptr_cpu[-1])
    p.update(
        q=torch.randn(total_q, 8, 128, dtype=torch.bfloat16, device=dev),
        qo_indptr=qo_indptr_cpu.to(dev),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(dev),
        kv_seq_lens_cpu=kv_lens,
        block_tables=bt.to(dev),
        max_q_len=nodes,
        max_kv_len=int(kv_lens.max()),
    )
    masks = []
    for i in range(b):
        prefix, lkv = int(prefix_lens[i]), int(kv_lens[i])
        m = torch.zeros(nodes, lkv, dtype=torch.bool)
        m[:, :prefix] = True
        for n in range(nodes):
            a = n
            while a >= 0:
                m[n, prefix + a] = True
                a = parent[a]
        masks.append(m.reshape(-1))
    p["custom_mask"] = torch.cat(masks).to(dev)
    p["tree"] = dict(nodes=nodes, parent=parent)
    return p


@pytest.mark.parametrize("backend", ["fa2"])
def test_custom_mask_tree(backend):
    p = _tree_problem(97, prefix_lens=[40, 3, 77])
    _skip_unless_runnable(p, backend, custom_mask=True)
    _, out, lse = _run(p, backend, custom_mask=p["custom_mask"])
    ref_out, ref_lse = _reference(p, custom_mask=p["custom_mask"])
    _assert_matches(out, lse, ref_out, ref_lse)
    # the mask really excluded same-depth siblings: differs from plain causal
    causal_out, _ = _reference(p)
    assert not torch.allclose(ref_out, causal_out, **OUT_TOL)
    # and a sibling-only perturbation of K changes nothing for its sibling:
    # query = second grandchild (node 4), sibling = first grandchild (node 3)
    nodes = p["tree"]["nodes"]
    q_row = 4  # request 0 starts at token 0
    kv_pos = int(p["kv_seq_lens_cpu"][0]) - nodes + 3
    page = int(p["block_tables"][0, kv_pos // p["page_size"]])
    slot = kv_pos % p["page_size"]
    k2 = p["k_cache"].clone()
    v2 = p["v_cache"].clone()
    k2[page, :, slot, :] = 5.0
    v2[page, :, slot, :] = 5.0
    attn = PagedAttention(torch.device(p["device"]))
    attn.plan(
        make_metadata(p), backend=backend, **_plan_kw(p, custom_mask=p["custom_mask"])
    )
    out2, _ = attn.run(p["q"], (k2, v2))
    torch.testing.assert_close(out2[q_row], out[q_row], atol=0, rtol=0)
    assert not torch.allclose(
        out2[3].float(), out[3].float(), **OUT_TOL
    )  # node 3 sees itself


def test_custom_mask_is_anded_with_the_envelope():
    """An all-True mask cannot widen causal / window: equals the plain plan."""
    p = make_problem(seed=101, **_SHAPE)
    _skip_unless_runnable(p, "fa2", custom_mask=True, window_left=16)
    q_lens = p["qo_indptr_cpu"].diff().to(torch.int64)
    numel = int((q_lens * p["kv_seq_lens_cpu"].to(torch.int64)).sum())
    ones = torch.ones(numel, dtype=torch.bool, device=p["device"])
    for window_left in (-1, 16):
        _, out_m, lse_m = _run(p, "fa2", custom_mask=ones, window_left=window_left)
        ref_out, ref_lse = _reference(p, window_left=window_left)
        _assert_matches(out_m, lse_m, ref_out, ref_lse)
    # non-causal + mask: only the mask applies
    tri = _reference(p, causal=False, custom_mask=ones)
    _, out_n, lse_n = _run(p, "fa2", custom_mask=ones, causal=False)
    _assert_matches(out_n, lse_n, *tri)


def test_custom_mask_plan_is_sync_free():
    """A custom-mask plan on fa2 issues no blocking copy and no host sync:
    the envelope is computed from the device metadata and the wrapper packs
    the mask from host-computed indptrs (ledger M6; the plan() contract)."""
    p = make_problem(seed=107, **_SHAPE)
    _skip_unless_runnable(p, "fa2", custom_mask=True, window_left=16)
    dev = torch.device(p["device"])
    q_lens = p["qo_indptr_cpu"].diff().to(torch.int64)
    numel = int((q_lens * p["kv_seq_lens_cpu"].to(torch.int64)).sum())
    g = torch.Generator(device=dev).manual_seed(107)
    mask = torch.rand(numel, device=dev, generator=g) < 0.7
    attn = PagedAttention(dev)
    md = make_metadata(p)
    kw = _plan_kw(p, custom_mask=mask, window_left=16)
    attn.plan(md, backend="fa2", **kw)  # warm-up: JIT modules, allocator
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        attn.plan(md, backend="fa2", **kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    ref_out, ref_lse = _reference(p, window_left=16, custom_mask=mask)
    _assert_matches(out, lse, ref_out, ref_lse)


def test_custom_mask_contract():
    p = make_problem(seed=103, **_SHAPE)
    _skip_unless_runnable(p, "fa2", custom_mask=True)
    q_lens = p["qo_indptr_cpu"].diff().to(torch.int64)
    numel = int((q_lens * p["kv_seq_lens_cpu"].to(torch.int64)).sum())
    dev = torch.device(p["device"])
    good = torch.ones(numel, dtype=torch.bool, device=dev)
    attn = PagedAttention(dev)
    md = make_metadata(p)
    with pytest.raises(ValueError, match="sum\\(q_len_i \\* kv_len_i\\)"):
        attn.plan(md, backend="fa2", **_plan_kw(p, custom_mask=good[:-1]))
    with pytest.raises(ValueError, match="bool tensor"):
        attn.plan(md, backend="fa2", **_plan_kw(p, custom_mask=good.to(torch.uint8)))
    with pytest.raises(ValueError, match="bound to"):
        attn.plan(md, backend="fa2", **_plan_kw(p, custom_mask=good.cpu()))
    with pytest.raises(ValueError, match="1-D"):
        attn.plan(md, backend="fa2", **_plan_kw(p, custom_mask=good.view(1, -1)))
    # backends without custom-mask support are excluded with the reason; a
    # backend this architecture does not admit is excluded for that first
    admitted = set(_resolve_or_skip(p, "auto").backends)
    for backend in ("cudnn", "trtllm-gen", "fa3"):
        with pytest.raises(
            ValueError,
            match="custom attention mask not supported"
            if backend in admitted
            else "custom attention mask not supported|unsupported compute capability",
        ):
            attn.plan(md, backend=backend, **_plan_kw(p, custom_mask=good))
    # graph mode: named follow-up, not a fallback
    g = PagedAttention(dev, use_cuda_graph=True)
    with pytest.raises(ValueError, match="follow-up"):
        g.plan(md, backend="fa2", **_plan_kw(p, custom_mask=good))
    assert g.backend is None


# --------------------------------------------------------------------------
# attention sinks
# --------------------------------------------------------------------------


def _sinks(p, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(p["num_qo_heads"], generator=g) * 5).to(
        device=p["device"], dtype=torch.float32
    )


@pytest.mark.parametrize("backend", ["fa2", "fa3", "trtllm-gen", "cake"])
@pytest.mark.parametrize("causal,window_left", [(True, -1), (True, 16), (False, -1)])
@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
def test_attention_sinks(backend, causal, window_left, lse_mode):
    p = make_problem(seed=107 + window_left + 3 * causal, **_SHAPE)
    _skip_unless_runnable(
        p, backend, causal=causal, window_left=window_left, sinks=True
    )
    sinks = _sinks(p)
    _, out, lse = _run(
        p,
        backend,
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        use_sinks=True,
        sinks=sinks,
    )
    ref_out, ref_lse = _reference(
        p,
        causal=causal,
        window_left=window_left,
        sinks=sinks,
        lse_base="e" if lse_mode == "basee" else "2",
    )
    _assert_matches(out, lse, ref_out, ref_lse)  # LSE includes the sink
    plain_out, plain_lse = _reference(
        p,
        causal=causal,
        window_left=window_left,
        lse_base="e" if lse_mode == "basee" else "2",
    )
    assert not torch.allclose(ref_out, plain_out, **OUT_TOL)
    assert not torch.allclose(ref_lse, plain_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", ["fa2", "trtllm-gen", "cake"])
def test_attention_sinks_are_a_per_run_value(backend):
    """One plan, two run() calls with different sinks (per-layer values)."""
    p = make_problem(seed=113, **_SHAPE)
    _skip_unless_runnable(p, backend, sinks=True)
    attn = PagedAttention(torch.device(p["device"]))
    attn.plan(make_metadata(p), backend=backend, **_plan_kw(p, use_sinks=True))
    for seed in (1, 2):
        sinks = _sinks(p, seed)
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), sinks=sinks)
        _assert_matches(out, lse, *_reference(p, sinks=sinks))


@pytest.mark.parametrize("backend", ["fa2", "fa3", "trtllm-gen", "cake"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_sinks_with_kv_scales(backend, dtype):
    """k_scale / v_scale on a float cache together with sinks: the sink
    logit competes with the k-scaled logits (fa folds k_scale into the sink
    variant's sm_scale argument, whose run() has no k_scale; trtllm-gen and
    cake take bmm1 / bmm2), and v_scale multiplies the sink-weighted output."""
    p = make_problem(seed=117, **dict(_SHAPE, dtype=dtype))
    _skip_unless_runnable(p, backend, sinks=True)
    sinks = _sinks(p, seed=5)
    p["k_scale"], p["v_scale"] = 0.5, 2.0
    _, out, lse = _run(p, backend, use_sinks=True, sinks=sinks)
    s = 1.0 / math.sqrt(p["head_dim_qk"])
    scaled = dict(p, v_ref=p["v_ref"].float() * 2.0)
    ref_out, ref_lse = _reference(scaled, sm_scale=s * 0.5, sinks=sinks)
    _assert_matches(out, lse, ref_out, ref_lse)
    other_out, other_lse = _reference(scaled, sinks=sinks)  # sink vs unscaled logits
    assert not torch.allclose(ref_out, other_out, **OUT_TOL)
    assert not torch.allclose(ref_lse, other_lse, **LSE_TOL)


def test_sinks_contract():
    p = make_problem(seed=127, **_SHAPE)
    _skip_unless_runnable(p, "fa2", sinks=True)
    dev = torch.device(p["device"])
    sinks = _sinks(p)
    kv = (p["k_cache"], p["v_cache"])
    # cuDNN is excluded at resolve, with the reason
    with pytest.raises(ValueError, match="attention sinks not supported"):
        resolve_paged_attention(backend="cudnn", **_resolve_kw(p, sinks=True))
    res = resolve_paged_attention(backend="auto", **_resolve_kw(p, sinks=True))
    assert "cudnn" not in res.backends
    assert res.excluded["cudnn"] == "attention sinks not supported"
    attn = PagedAttention(dev)
    md = make_metadata(p)
    # sinks without the plan flag: loud (the default kernels would drop them)
    attn.plan(md, backend="fa2", **_plan_kw(p))
    with pytest.raises(ValueError, match="use_sinks=True"):
        attn.run(p["q"], kv, sinks=sinks)
    # the plan flag without sinks: loud
    attn.plan(md, backend="fa2", **_plan_kw(p, use_sinks=True))
    with pytest.raises(ValueError, match="no sinks= tensor"):
        attn.run(p["q"], kv)
    for bad in (
        sinks.to(torch.bfloat16),
        sinks[:-1],
        sinks.cpu(),
        sinks.view(1, -1),
        torch.zeros(2 * p["num_qo_heads"], device=dev)[::2],
    ):
        with pytest.raises(ValueError, match="sinks must be a contiguous fp32"):
            attn.run(p["q"], kv, sinks=bad)
    # a Resolution resolved without sinks cannot be used with them (pinning)
    plain = resolve_paged_attention(backend="auto", **_resolve_kw(p))
    with pytest.raises(ValueError, match="pinned Resolution"):
        attn.plan(md, backend=plain, **_plan_kw(p, use_sinks=True))


def test_sinks_with_soft_cap_is_a_typed_rejection_on_fa():
    """The AttentionSink variant has no soft-cap hook: fa declines the batch
    with the typed signal, `auto` reports every reason, an explicit backend
    raises."""
    p = make_problem(seed=131, **_SHAPE)
    _skip_unless_runnable(p, "fa2", sinks=True)
    attn = PagedAttention(torch.device(p["device"]))
    md = make_metadata(p)
    kw = _plan_kw(p, use_sinks=True, logits_soft_cap=30.0)
    with pytest.raises(ValueError, match="no pinned candidate.*no logits soft cap"):
        attn.plan(md, backend="auto", **kw)
    with pytest.raises(
        ValueError, match="'fa2' cannot plan this batch.*no logits soft cap"
    ):
        attn.plan(md, backend="fa2", **kw)
    assert attn.backend is None
    # sinks + custom mask: not verified -> also typed
    numel = int(
        (
            p["qo_indptr_cpu"].diff().to(torch.int64)
            * p["kv_seq_lens_cpu"].to(torch.int64)
        ).sum()
    )
    ones = torch.ones(numel, dtype=torch.bool, device=p["device"])
    with pytest.raises(ValueError, match="not verified"):
        attn.plan(md, backend="fa2", **_plan_kw(p, use_sinks=True, custom_mask=ones))


def test_sink_kernels_are_specialized_per_head_dim():
    """Two sink plans in one process, head_dim 64 then 128 (same dtype,
    window and backend): the second must run its own JIT module.  Before the
    fix the sink wrapper's module URI omitted the head dims, so the D=128
    plan reused the D=64 module and left half of its output unwritten
    (review CR02)."""
    outs = []
    for seed, d in ((131, 64), (132, 128)):
        p = make_problem(seed=seed, **dict(_SHAPE, head_dim_qk=d))
        _skip_unless_runnable(p, "fa2", sinks=True)
        sinks = _sinks(p, seed)
        attn = PagedAttention(torch.device(p["device"]))
        attn.plan(make_metadata(p), backend="fa2", **_plan_kw(p, use_sinks=True))
        out = torch.full(
            (p["q"].shape[0], p["num_qo_heads"], d),
            float("nan"),
            dtype=p["dtype"],
            device=p["device"],
        )
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), sinks=sinks, out=out)
        torch.cuda.synchronize()
        assert torch.isfinite(out.float()).all(), f"D={d}: unwritten output rows"
        _assert_matches(out, lse, *_reference(p, sinks=sinks))
        outs.append(attn._impl._active._active)
    assert outs[0] is not outs[1]


@pytest.mark.xfail(
    strict=True,
    reason="ledger M20: the fa2 kernel trims the windowed KV range as if causal, so "
    "non-causal + sliding window is wrong for requests longer than 128 query "
    "tokens with a history (B200: q 256 / kv 768 / window 128 -> rows 0..127 off "
    "by up to 0.4).  Flip fa2's supports_window_noncausal back to True when this "
    "passes.",
)
def test_fa2_kernel_noncausal_sliding_window_defect():
    """Runs the fa2 kernel through the legacy wrapper (the unified capability
    table declares the combination unsupported, see M20) on the failing shape
    and compares with the oracle.  Strict xfail: a kernel fix turns it into an
    XPASS and asks for the capability flip."""
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    dev = torch.device("cuda:0")
    hq, hk, d, page = 32, 8, 128, 16
    q_len, kv_len, window = 256, 768, 128
    g = torch.Generator(device=dev).manual_seed(2020)
    q = torch.randn(q_len, hq, d, dtype=torch.float16, device=dev, generator=g)
    pages = kv_len // page
    k = torch.randn(pages, hk, page, d, dtype=torch.float16, device=dev, generator=g)
    v = torch.randn(pages, hk, page, d, dtype=torch.float16, device=dev, generator=g)
    qo = torch.tensor([0, q_len], dtype=torch.int32)
    kv_indptr = torch.tensor([0, pages], dtype=torch.int32)
    ids = torch.arange(pages, dtype=torch.int32, device=dev)
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
    w.plan(
        qo,
        kv_indptr,
        ids,
        torch.tensor([page], dtype=torch.int32),
        hq,
        hk,
        d,
        page,
        causal=False,
        window_left=window,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    out = w.run(q, (k, v))
    torch.cuda.synchronize()
    ref_out, _ = reference_paged_prefill(
        q,
        k,
        v,
        qo,
        torch.tensor([kv_len], dtype=torch.int32),
        None,
        page,
        False,
        window_left=window,
        kv_page_indices=ids,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)


@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("head_dim", [128, 256])
def test_attention_sinks_with_fp8_kv(kv_dtype, head_dim):
    """fa2 attention sinks over an fp8 KV cache (M22): the sink variant used to
    be declined at plan time as "not verified"; measured on B200 it matches
    the sink-aware oracle on the dequantized cache, with k_scale folded into
    the launch scale and v_scale applied to the output."""
    p = make_problem(
        seed=140 + head_dim, **dict(_SHAPE, head_dim_qk=head_dim), kv_dtype=kv_dtype
    )
    _skip_unless_runnable(p, "fa2", sinks=True)
    sinks = _sinks(p, 3)
    attn, out, lse = _run(p, "fa2", sinks=sinks, use_sinks=True)
    assert attn.backend == "fa2"
    _assert_matches(out, lse, *_reference(p, sinks=sinks))
