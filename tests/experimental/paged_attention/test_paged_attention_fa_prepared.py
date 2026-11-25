"""Direct-launch path of the unified fa2 / fa3 backend (``backend`` pinned).

``plan()`` plans through the legacy ``BatchPrefillWithPagedKVCacheWrapper``
and stages the positional argument list that wrapper's ``run()`` would hand
to its module's ``paged_run``; ``run()`` fills the entries only a call knows
and launches the module itself.  The wrapper's own ``run()`` remains the path
of every plan the staging declines (the AttentionSink variant among them).
Both paths launch the same kernel with the same arguments, so this is a
host-path change only.  Pinned here, against the fp32 paged-attention oracle
and without timing thresholds:

1. Drift guard: one planned wrapper, sent once through the staged list and
   once through its own ``run()``, hands ``paged_run`` the same arguments --
   across masks, sliding window, soft cap, custom mask, LSE modes, fp8 KV,
   run-time scales, layouts, eager and CUDA-graph mode -- and an omitted
   ``out`` is allocated zero-filled where the wrapper's is.  Host only: the
   arguments are recorded, not launched.  An upstream change to the legacy
   argument list fails here.
2. Direct launch and wrapper ``run()`` give bitwise equal output and LSE, and
   match the oracle, for a single request and a ragged batch with a padding
   request, with caller buffers, without, and with one of the two.
3. Plain plans take the direct launch, sink plans the wrapper's ``run()``,
   and one instance alternates between them.
4. Eager re-plans of one instance: other lengths and page ids, another
   dtype (another module), custom mask to plain; a rejected re-plan leaves
   the previous plan runnable through the path it was staged for.
5. CUDA-graph mode: capture on buffers larger than the batch, re-plan
   with other lengths and page ids, replay; no device allocation in a
   ``run()`` with caller buffers.
6. ``run()`` does not synchronize.
7. Staging declines every wrapper state for which the legacy ``run()``
   leaves its plain path, and accounts for every attribute that ``run()``
   reads.

fa3 needs SM90a: its cases skip elsewhere.  The backend's private state is
read through ``attn._impl._active``, as the neighbouring backend tests do.
"""

import inspect
import re

import pytest
import torch

from flashinfer.experimental.paged_attention._backends import fa_backend
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    GraphCapacity,
    PagedAttention,
    resolve_paged_attention,
)
from flashinfer.utils import _get_cache_alibi_slopes_buf

from .paged_attention_reference import reference_paged_prefill, scratch_workspace
from .test_paged_attention_coverage import (
    assert_canaries,
    build_problem,
    canary_buffers,
)
from .test_paged_attention_prototype import LSE_TOL, OUT_TOL, make_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda:0"
FA_BACKENDS = ["fa2", "fa3"]
PAGE = 16
FP8 = torch.float8_e4m3fn
# positions, in the staged ``paged_run`` argument list, of what run() fills
_CALL_ARGS = (
    fa_backend._ARG_Q,
    fa_backend._ARG_K_CACHE,
    fa_backend._ARG_V_CACHE,
    fa_backend._ARG_OUT,
    fa_backend._ARG_LSE,
    fa_backend._ARG_SM_SCALE,
)

# A one-token request, a full-prefill request with q == kv, two prefixes.
_RAGGED = ([5, 17, 1, 9], [40, 17, 70, 33])
# The same with a padding request (kv_len 0, one query token) in the middle.
_RAGGED_PADDING = ([5, 1, 17, 1, 9], [40, 0, 17, 70, 33])
_BATCHES = {"single": ([24], [200]), "ragged_padding": _RAGGED_PADDING}

# One feature combination the fa backends serve per name: build_problem()
# overrides, plan() keywords, run() keywords.  ``custom_mask=True`` stands for
# a mask built for the problem.
_CASES = {
    "causal": ({}, {}, {}),
    "noncausal": ({}, dict(causal=False), {}),
    "window": ({}, dict(window_left=16), {}),
    "soft_cap": ({}, dict(logits_soft_cap=30.0), {}),
    "custom_mask": ({}, dict(custom_mask=True), {}),
    "lse_none": ({}, dict(lse_mode="none"), {}),
    "lse_basee": ({}, dict(lse_mode="basee"), {}),
    "fp8_kv": (dict(kv_dtype=FP8), {}, {}),
    "kv_scales": ({}, {}, dict(k_scale=0.5, v_scale=2.0)),
    "nhd": (dict(kv_layout="NHD"), {}, {}),
    "fp16_csr": (dict(dtype=torch.float16, input_form="page_indices"), {}, {}),
    "combined": (
        dict(kv_layout="NHD", kv_dtype=FP8),
        dict(window_left=16, logits_soft_cap=30.0, lse_mode="basee"),
        dict(sm_scale=0.11),
    ),
}


def _problem(q_lens, kv_lens, *, seed, pool=None, **overrides):
    """``build_problem`` at 8 / 2 heads, head dim 128, page 16, bf16 unless
    overridden.  ``pool`` fixes the number of pages in the K/V pool, so
    batches with different lengths share the K/V shapes of a capture."""
    pages = sum(-(-int(kv) // PAGE) for kv in kv_lens)
    kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=PAGE,
        dtype=torch.bfloat16,
        pool_slack=8 if pool is None else pool - pages,
        device=DEVICE,
    )
    kw.update(overrides)
    return build_problem(q_lens, kv_lens, seed=seed, **kw)


def _diagonal_safe_mask(p, seed):
    """Random flattened per-request masks that keep every query's own
    position, so no row of a live request is fully masked."""
    g = torch.Generator().manual_seed(seed)
    blocks = [torch.zeros(0, dtype=torch.bool)]
    for lq, lkv in zip(
        p["qo_indptr_cpu"].diff().tolist(), p["kv_seq_lens_cpu"].tolist(), strict=True
    ):
        if lkv == 0:
            continue
        block = torch.rand(lq, lkv, generator=g) < 0.6
        rows = torch.arange(lq)
        block[rows, lkv - lq + rows] = True
        blocks.append(block.flatten())
    return torch.cat(blocks).to(DEVICE)


def _case(name, q_lens, kv_lens, *, seed, **overrides):
    """Problem, plan() keywords and run() keywords of ``_CASES[name]``."""
    problem_kw, plan_kw, run_kw = _CASES[name]
    p = _problem(q_lens, kv_lens, seed=seed, **{**overrides, **problem_kw})
    plan_kw, run_kw = dict(plan_kw), dict(run_kw)
    if plan_kw.get("logits_soft_cap"):
        p["q"] = p["q"] * 4  # push the scores towards the cap
    if plan_kw.get("custom_mask"):
        plan_kw["custom_mask"] = _diagonal_safe_mask(p, seed)
    # an fp8 pool brings its dequantization scales
    run_kw.setdefault("k_scale", p["k_scale"])
    run_kw.setdefault("v_scale", p["v_scale"])
    return p, plan_kw, run_kw


def _runnable(p, backend, *, causal=True, window_left=-1, lse_mode="base2", **features):
    """None where the pinned backend resolves for these plan() keywords on
    this device, else the reason it does not."""
    try:
        resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            head_dim_vo=p["head_dim_vo"],
            q_dtype=p["dtype"],
            kv_dtype=p["kv_dtype"],
            page_size=p["page_size"],
            kv_layout=p["kv_layout"],
            causal=causal,
            need_lse=lse_mode != "none",
            window_left=window_left,
            kv_input_form=p["input_form"],
            logits_soft_cap=features.get("logits_soft_cap"),
            custom_mask=features.get("custom_mask") is not None,
            sinks=features.get("use_sinks", False),
            backend=backend,
        )
    except ValueError as e:
        return str(e)
    return None


def _plan(p, backend, *, attn=None, graph_capacity=None, **plan_kw):
    """Resolve-or-skip (fa3 needs SM90a and takes no custom mask), then plan
    ``p`` on the pinned backend; returns (attn, backend object)."""
    reason = _runnable(p, backend, **plan_kw)
    if reason is not None:
        pytest.skip(f"{backend} not runnable here: {reason}")
    if attn is None:
        attn = PagedAttention(
            scratch_workspace(torch.device(DEVICE)), graph_capacity=graph_capacity
        )
    plan_kw.setdefault("lse_mode", "base2")
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p["kv_dtype"],
        kv_layout=p["kv_layout"],
        backend=backend,
        **plan_kw,
    )
    assert attn.backend == backend
    return attn, attn._impl._active


def _oracle(p, plan_kw=None, run_kw=None):
    """fp32 oracle for the plan() and run() keywords of one call.  A float
    pool's ``k_scale`` is a factor of the softmax scale and its ``v_scale``
    of V (the run() contract); an fp8 pool's scales are already in the
    dequantized ``k_ref`` / ``v_ref``."""
    plan_kw, run_kw = plan_kw or {}, run_kw or {}
    sm_scale = run_kw.get("sm_scale") or p["head_dim_qk"] ** -0.5
    v_ref = p["v_ref"]
    if p["kv_dtype"] == p["dtype"]:
        sm_scale *= run_kw.get("k_scale") or 1.0
        v_ref = v_ref.float() * (run_kw.get("v_scale") or 1.0)
    csr = p["input_form"] == "page_indices"
    return reference_paged_prefill(
        p["q"],
        p["k_ref"],
        v_ref,
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        None if csr else p["block_tables"],
        p["page_size"],
        plan_kw.get("causal", True),
        sm_scale=sm_scale,
        window_left=plan_kw.get("window_left", -1),
        kv_layout=p["kv_layout"],
        kv_page_indices=p["kv_page_indices"] if csr else None,
        lse_base="e" if plan_kw.get("lse_mode") == "basee" else "2",
        logits_soft_cap=plan_kw.get("logits_soft_cap"),
        custom_mask=plan_kw.get("custom_mask"),
        sinks=run_kw.get("sinks"),
    )


def _live_rows(p):
    """Mask over the batch's query rows, False for the rows of padding
    requests (kv_len 0): the contract leaves those unspecified, so a kernel
    may not write them and no comparison may include them."""
    return torch.repeat_interleave(
        p["kv_seq_lens_cpu"] > 0, p["qo_indptr_cpu"].diff().long()
    ).to(DEVICE)


def _assert_oracle(out, lse, p, plan_kw=None, run_kw=None):
    """Oracle on the batch's live rows (the batch is the first
    ``total_q_tokens`` rows of capture buffers)."""
    ref_out, ref_lse = _oracle(p, plan_kw, run_kw)
    n = p["q"].shape[0]
    live = _live_rows(p)
    torch.testing.assert_close(out[:n][live].float(), ref_out[live], **OUT_TOL)
    if (plan_kw or {}).get("lse_mode", "base2") == "none":
        assert lse is None, "lse_mode='none' must return (out, None)"
    else:
        torch.testing.assert_close(lse[:n][live], ref_lse[live], **LSE_TOL)


def _buffers(p, *, rows=None, lse=True):
    """q / out / lse with ``rows`` rows (default: the batch's): the batch in
    the first rows of q, NaN in out and lse."""
    n, h = p["q"].shape[0], p["num_qo_heads"]
    rows = n if rows is None else rows
    q = torch.zeros(rows, h, p["head_dim_qk"], dtype=p["dtype"], device=DEVICE)
    q[:n] = p["q"]
    out = torch.full(
        (rows, h, p["head_dim_vo"]), float("nan"), dtype=p["dtype"], device=DEVICE
    )
    lse_buf = torch.full((rows, h), float("nan"), device=DEVICE) if lse else None
    return q, out, lse_buf


def _capacity(p, *, rows, max_q_len=None):
    """Explicit graph capacity for batches shaped like ``p`` with ``rows``
    query rows (more than a batch has: the launch then gets a row prefix)."""
    b, width = p["block_tables"].shape
    csr = p["input_form"] == "page_indices"
    return GraphCapacity(
        batch_size=b,
        total_q_tokens=rows,
        max_q_len=max_q_len or p["max_q_len"],
        max_kv_len=width * PAGE,
        page_size=PAGE,
        kv_input_form=p["input_form"],
        table_width=None if csr else width,
        flat_capacity=b * width if csr else None,
    )


def _spy_paged_run(monkeypatch, module, *, launch=True):
    """The (args, kwargs) of every call of ``module.paged_run``; with
    ``launch=False`` the calls are recorded and not made.  Assert on the
    length of the list, never on the list: its repr is every tensor of every
    launch, the workspace included."""
    real = module.paged_run
    launches = []

    def spy(*args, **kwargs):
        launches.append((args, kwargs))
        if launch:
            real(*args, **kwargs)

    monkeypatch.setattr(module, "paged_run", spy)
    return launches


def _spy_wrapper_run(monkeypatch, wrapper):
    """One entry per call of this wrapper's own ``run()``."""
    real = wrapper.run
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(wrapper, "run", spy)
    return calls


def _decline_staging(monkeypatch):
    """Every plan made while this holds runs through the wrapper's ``run()``."""
    monkeypatch.setattr(fa_backend._FaBackend, "_stage_paged_run", lambda *args: None)


def _poison_empty(monkeypatch):
    """``torch.empty`` hands out NaN instead of whatever the allocator holds,
    so a buffer allocated with it cannot pass for a zero-filled one."""
    real = torch.empty

    def empty(*args, **kwargs):
        t = real(*args, **kwargs)
        return t.fill_(float("nan")) if t.is_floating_point() else t

    monkeypatch.setattr(torch, "empty", empty)


def _describe(argument):
    """A ``paged_run`` argument in a failure message: a tensor by its shape,
    dtype and address, never by its values."""
    if torch.is_tensor(argument):
        return (
            f"tensor(shape={tuple(argument.shape)}, dtype={argument.dtype}, "
            f"data_ptr={argument.data_ptr():#x})"
        )
    return repr(argument)


def _same_argument(direct, legacy, *, fresh=False):
    """One ``paged_run`` argument of the two paths: the same object; for a
    tensor otherwise the same view (a row prefix is sliced per call) or, where
    each path allocates its own (``fresh``), the same shape, dtype and device;
    for anything else the same type and value."""
    if direct is legacy:
        return not fresh or direct is None
    if torch.is_tensor(direct) and torch.is_tensor(legacy):
        return (
            direct.shape == legacy.shape
            and direct.stride() == legacy.stride()
            and direct.dtype == legacy.dtype
            and direct.device == legacy.device
            and (fresh or direct.data_ptr() == legacy.data_ptr())
        )
    if torch.is_tensor(direct) or torch.is_tensor(legacy):
        return False
    return type(direct) is type(legacy) and direct == legacy


def _argument_drift(direct, legacy, fresh):
    """Where the staged ``paged_run`` arguments differ from the ones the
    wrapper's ``run()`` passes, one line per difference; ``fresh`` are the
    positions each path allocates for itself."""
    if len(direct) != len(legacy):
        return [
            f"paged_run gets {len(direct)} staged arguments and {len(legacy)} "
            "from the wrapper's run(): the legacy argument list changed"
        ]
    return [
        f"paged_run argument {i}: staged {_describe(d)}, the wrapper's run() "
        f"passes {_describe(l)}"
        for i, (d, l) in enumerate(zip(direct, legacy, strict=True))
        if not _same_argument(d, l, fresh=i in fresh)
    ]


# ---------------------------------------------------------------------------
# 1: drift guard
# ---------------------------------------------------------------------------

# what the drift guard found, for the fixture below
_DRIFT = []


@pytest.fixture(autouse=True)
def _no_launch_of_a_drifted_list(request):
    """The drift guard runs first and launches nothing.  Once it has found the
    staged list out of step with the wrapper's ``run()``, a real launch of
    that list can fault on the device and take the process, and with it the
    guard's report: the rest of the file, which launches, is skipped."""
    if _DRIFT and request.function is not test_staged_arguments_match_the_wrapper_run:
        pytest.skip(f"the staged paged_run arguments drifted: {_DRIFT[0]}")


def _drift_params():
    for case, (_, plan_kw, _) in _CASES.items():
        for graph in (False, True):
            if graph and plan_kw.get("custom_mask"):
                continue  # the fa backends take custom masks on eager instances only
            yield pytest.param(
                case, graph, id=f"{case}-{'graph' if graph else 'eager'}"
            )


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("buffers", ["caller", "alloc"])
@pytest.mark.parametrize("case,graph", list(_drift_params()))
def test_staged_arguments_match_the_wrapper_run(
    monkeypatch, backend, case, graph, buffers
):
    """The staged list couples the backend to the argument order of the
    legacy wrapper's ``run()``.  Send ONE planned wrapper through the staged
    list and through its own ``run()`` with the same tensors and compare what
    reaches the module's ``paged_run``: the same length, no keywords, and
    argument by argument the same object, view or value.  The calls are
    recorded, not launched: a shifted tensor or pointer argument would fault
    on the device before the comparison could name it (the results of real
    launches are compared in ``test_direct_launch_equals_the_wrapper_run``).
    In graph mode the capture buffers are larger than the batch, so q / out /
    lse are row prefixes; with ``buffers="alloc"`` each path allocates out /
    lse, and the direct path's out is zero-filled where the wrapper's is."""
    p, plan_kw, run_kw = _case(case, *_RAGGED, seed=11)
    n = p["q"].shape[0]
    rows = n + 7 if graph else n
    attn, backend_obj = _plan(
        p, backend, graph_capacity=_capacity(p, rows=rows) if graph else None, **plan_kw
    )
    wrapper = backend_obj._active
    assert backend_obj._staged is not None, "a plain plan was not staged"
    q, out, lse = _buffers(p, rows=rows, lse=plan_kw.get("lse_mode") != "none")
    if buffers == "alloc":
        out = lse = None
        _poison_empty(monkeypatch)
    kv = (p["k_cache"], p["v_cache"])
    launches = _spy_paged_run(monkeypatch, wrapper._cached_module, launch=False)
    wrapper_runs = _spy_wrapper_run(monkeypatch, wrapper)

    attn.run(q, kv, out=out, lse=lse, **run_kw)
    assert (len(launches), len(wrapper_runs)) == (1, 0)
    # run() filled a copy: the staged list holds none of the call's tensors
    assert all(backend_obj._staged[1][i] is None for i in _CALL_ARGS)
    monkeypatch.setattr(backend_obj, "_staged", None)  # same plan, the wrapper's run()
    attn.run(q, kv, out=out, lse=lse, **run_kw)
    assert (len(launches), len(wrapper_runs)) == (2, 1)

    (direct, direct_kw), (legacy, legacy_kw) = launches
    assert not direct_kw and not legacy_kw
    allocated = (fa_backend._ARG_OUT, fa_backend._ARG_LSE) if buffers == "alloc" else ()
    drift = _argument_drift(direct, legacy, allocated)
    _DRIFT.extend(drift)
    assert not drift, "\n".join(drift)
    if buffers == "alloc":
        # Nothing was launched, so each path's out is as that path allocated
        # it: zeros from the wrapper's run() for an omitted out (rows no
        # kernel writes then read 0), NaN from the poisoned torch.empty.
        zero_filled = [
            not args[fa_backend._ARG_OUT].count_nonzero().item()
            for args in (direct, legacy)
        ]
        only = "the wrapper's run()" if zero_filled[1] else "the direct launch"
        assert zero_filled[0] == zero_filled[1], (
            f"only {only} zero-fills an omitted out"
        )
    # the slots are the call's tensors, where the wrapper's run() puts them
    assert _same_argument(legacy[fa_backend._ARG_Q], q[:n])
    assert legacy[fa_backend._ARG_K_CACHE] is kv[0]
    assert legacy[fa_backend._ARG_V_CACHE] is kv[1]
    if buffers == "caller":
        assert _same_argument(legacy[fa_backend._ARG_OUT], out[:n])
    assert isinstance(legacy[fa_backend._ARG_SM_SCALE], float)


# ---------------------------------------------------------------------------
# 2: direct launch == wrapper run(), bitwise, and both match the oracle
# ---------------------------------------------------------------------------


def _bitwise_params():
    for case, (_, plan_kw, _) in _CASES.items():
        buffers = ["alloc", "caller"]
        if plan_kw.get("lse_mode") != "none":
            # one of out= / lse= passed, the other allocated by the backend
            buffers += ["out_only", "lse_only"]
        for batch in _BATCHES:
            for b in buffers:
                yield pytest.param(case, batch, b, id=f"{case}-{batch}-{b}")


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("case,batch,buffers", list(_bitwise_params()))
def test_direct_launch_equals_the_wrapper_run(
    monkeypatch, backend, case, batch, buffers
):
    """Two instances plan the same batch, one staged for the direct launch and
    one with staging declined (the wrapper's ``run()``): output and LSE are
    bitwise equal on the live rows and match the oracle.  A caller's buffer
    is the tensor returned, and the memory around it is untouched."""
    p, plan_kw, run_kw = _case(case, *_BATCHES[batch], seed=21)
    need_lse = plan_kw.get("lse_mode") != "none"
    direct, direct_backend = _plan(p, backend, **plan_kw)
    with monkeypatch.context() as declined:
        _decline_staging(declined)
        fallback, fallback_backend = _plan(p, backend, **plan_kw)
    assert direct_backend._staged is not None and fallback_backend._staged is None
    launches = _spy_paged_run(monkeypatch, direct_backend._active._cached_module)
    direct_wrapper_runs = _spy_wrapper_run(monkeypatch, direct_backend._active)
    fallback_wrapper_runs = _spy_wrapper_run(monkeypatch, fallback_backend._active)

    kv = (p["k_cache"], p["v_cache"])
    pad = 64
    results = []
    for attn in (direct, fallback):
        out_big, out_view, lse_big, lse_view = canary_buffers(p, pad)
        out_buf = out_view if buffers in ("caller", "out_only") else None
        lse_buf = lse_view if need_lse and buffers in ("caller", "lse_only") else None
        out, lse = attn.run(p["q"], kv, out=out_buf, lse=lse_buf, **run_kw)
        assert out_buf is None or out is out_buf
        assert lse_buf is None or lse is lse_buf
        assert_canaries(out_big, out_view, pad)
        assert_canaries(lse_big, lse_view, pad)
        results.append((out, lse))
    assert (len(launches), len(direct_wrapper_runs), len(fallback_wrapper_runs)) == (
        2,
        0,
        1,
    )

    (out_direct, lse_direct), (out_fallback, lse_fallback) = results
    live = _live_rows(p)
    assert out_direct.dtype == p["dtype"]
    assert torch.equal(out_direct[live], out_fallback[live])
    if need_lse:
        assert torch.equal(lse_direct[live], lse_fallback[live])
    _assert_oracle(out_direct, lse_direct, p, plan_kw, run_kw)
    _assert_oracle(out_fallback, lse_fallback, p, plan_kw, run_kw)


# ---------------------------------------------------------------------------
# 3: which path a plan takes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", FA_BACKENDS)
def test_plain_plans_launch_directly_and_sink_plans_through_the_wrapper(
    monkeypatch, backend
):
    """plain -> sink -> plain on one instance: the plain plans launch the
    module without the wrapper's ``run()``, the sink plan runs the variant
    wrapper's ``run()``, and neither leaves state that breaks the other."""
    p1 = _problem(*_RAGGED, seed=31)
    p2 = _problem([3, 30, 8], [64, 30, 19], seed=32)
    sinks = torch.linspace(1.0, 5.0, p1["num_qo_heads"], device=DEVICE)

    attn, backend_obj = _plan(p1, backend)
    plain = backend_obj._active
    assert backend_obj._staged is not None
    launches = _spy_paged_run(monkeypatch, plain._cached_module)
    plain_runs = _spy_wrapper_run(monkeypatch, plain)
    for _ in range(2):
        out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]))
        _assert_oracle(out, lse, p1)
    assert (len(launches), len(plain_runs)) == (2, 0)

    _plan(p1, backend, attn=attn, use_sinks=True)
    sink = backend_obj._active
    assert sink is not plain and backend_obj._staged is None
    sink_runs = _spy_wrapper_run(monkeypatch, sink)
    out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]), sinks=sinks)
    _assert_oracle(out, lse, p1, run_kw=dict(sinks=sinks))
    plain_out, _ = _oracle(p1)
    assert not torch.allclose(out.float(), plain_out, **OUT_TOL)  # the sink counted
    assert (len(sink_runs), len(launches), len(plain_runs)) == (1, 2, 0)

    _plan(p2, backend, attn=attn)
    assert backend_obj._active is plain and backend_obj._staged is not None
    out, lse = attn.run(p2["q"], (p2["k_cache"], p2["v_cache"]))
    _assert_oracle(out, lse, p2)
    assert (len(sink_runs), len(launches), len(plain_runs)) == (1, 3, 0)


# ---------------------------------------------------------------------------
# 4: eager re-plans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", FA_BACKENDS)
def test_eager_replans_restage_the_launch(monkeypatch, backend):
    """One instance, re-planned with other lengths and page ids, with another
    q / KV dtype (the wrapper swaps its module) and from a custom mask back to
    a plain plan: every run launches the module of its own plan directly and
    matches that plan's oracle."""
    attn, backend_obj = _plan(_problem(*_RAGGED, seed=41), backend)
    wrapper_runs = _spy_wrapper_run(monkeypatch, backend_obj._active)

    def replan_and_check(p, **plan_kw):
        _plan(p, backend, attn=attn, **plan_kw)
        staged = backend_obj._staged
        assert staged is not None
        launches = _spy_paged_run(monkeypatch, staged[0])
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        assert (len(launches), len(wrapper_runs)) == (1, 0)
        _assert_oracle(out, lse, p, plan_kw)
        return staged, out

    first, _ = replan_and_check(_problem(*_RAGGED, seed=41))
    # other lengths, page ids and batch size: a new list for the same module
    other, _ = replan_and_check(_problem([3, 30, 8], [64, 30, 19], seed=42))
    assert other[1] is not first[1] and other[0] is first[0]
    # bf16 -> fp16: another module, taken from the wrapper after its plan
    half, _ = replan_and_check(_problem(*_RAGGED, seed=43, dtype=torch.float16))
    assert half[0] is not first[0]
    # back to bf16 -- with a custom mask where the backend takes one -- then
    # the same batch plain: the mask does not outlive its plan
    p = _problem(*_RAGGED, seed=44)
    mask = _diagonal_safe_mask(p, 44)
    masked = None
    if _runnable(p, backend, custom_mask=mask) is None:
        _, masked = replan_and_check(p, custom_mask=mask)
    back, plain = replan_and_check(p)
    assert back[0] is first[0]
    assert masked is None or not torch.equal(masked, plain)


def _rejected_replan_params():
    # The module's planner failing after the wrapper has already taken the
    # new batch's index buffers, dtypes and module is survived by the direct
    # launch only (its staged list holds the previous plan's own module and
    # tensors); the wrapper's run() then reads the half-planned state, as it
    # always did.
    # "slopes" is the lookup of the staged launch's ALiBi slopes, the one step
    # of staging that allocates.  plan() makes it before the wrapper's plan:
    # after that plan has committed the new schedule, a failure would leave
    # the previous plan's list launching against it.
    for previous in ("direct", "wrapper_run", "sink"):
        for fails_in in ("slopes", "wrapper", "planner"):
            if (previous, fails_in) != ("wrapper_run", "planner"):
                yield pytest.param(previous, fails_in, id=f"{previous}-{fails_in}")


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("previous,fails_in", list(_rejected_replan_params()))
def test_rejected_replan_keeps_the_previous_plan_runnable(
    monkeypatch, backend, previous, fails_in
):
    """A re-plan to another batch and dtype that raises -- in the ALiBi slopes
    lookup of the staging, at the start of the wrapper's plan, or in the
    module's planner after the wrapper took the new batch's index buffers,
    dtypes and module -- publishes nothing: the previous plan still runs
    through the path it was staged for and matches its oracle, and a later
    plan succeeds."""
    p1 = _problem(*_RAGGED, seed=45)
    p2 = _problem([3, 30, 8], [64, 30, 19], seed=46, dtype=torch.float16)
    _, donor = _plan(p2, backend)
    planner = donor._active._cached_module  # the module a plain plan of p2 loads
    plan_kw, run_kw = {}, {}
    if previous == "sink":
        plan_kw["use_sinks"] = True
        run_kw["sinks"] = torch.linspace(-1.0, 1.0, p1["num_qo_heads"], device=DEVICE)
    with monkeypatch.context() as declined:
        if previous == "wrapper_run":
            _decline_staging(declined)
        attn, backend_obj = _plan(p1, backend, **plan_kw)
    staged, active = backend_obj._staged, backend_obj._active
    assert (staged is not None) == (previous == "direct")
    assert active._cached_module is not planner
    launches = _spy_paged_run(monkeypatch, active._cached_module)
    wrapper_runs = _spy_wrapper_run(monkeypatch, active)

    def boom(*args, **kwargs):
        raise RuntimeError("injected plan failure")

    with monkeypatch.context() as failing:
        if fails_in == "slopes":
            failing.setattr("flashinfer.utils._get_cache_alibi_slopes_buf", boom)
        elif fails_in == "wrapper":
            failing.setattr(backend_obj._wrapper, "plan", boom)
        else:
            failing.setattr(planner, "plan", boom)
        with pytest.raises(RuntimeError, match="injected plan failure"):
            _plan(p2, backend, attn=attn)
    assert backend_obj._staged is staged and backend_obj._active is active

    out, lse = attn.run(p1["q"], (p1["k_cache"], p1["v_cache"]), **run_kw)
    _assert_oracle(out, lse, p1, plan_kw, run_kw)
    assert (len(launches), len(wrapper_runs)) == (1, 0 if previous == "direct" else 1)

    _plan(p2, backend, attn=attn)
    out, lse = attn.run(p2["q"], (p2["k_cache"], p2["v_cache"]))
    _assert_oracle(out, lse, p2)


# ---------------------------------------------------------------------------
# 5: CUDA-graph mode
# ---------------------------------------------------------------------------


def _load(buffers, p):
    """Put batch ``p`` into the capture buffers; NaN where a replay writes."""
    q, k, v, out, lse = buffers
    q[: p["q"].shape[0]] = p["q"]
    k.copy_(p["k_cache"])
    v.copy_(p["v_cache"])
    out.fill_(float("nan"))
    if lse is not None:
        lse.fill_(float("nan"))


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("path", ["direct", "wrapper_run"])
@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
def test_graph_capture_replan_replay(monkeypatch, backend, path, lse_mode):
    """Capture one run on buffers of 48 rows for a 32-row batch (the launch
    gets the row prefix), replay; re-plan -- without a synchronization --
    with other query partitions, KV lengths and page ids (29 rows), replay.
    Run-time scales are captured with the launch: ``k_scale`` in the softmax
    scale, ``v_scale`` and the base-e fold over the whole buffers."""
    rows, pool, width = 48, 24, 6
    p1 = _problem(*_RAGGED, seed=51, pool=pool, table_width=width)
    p2 = _problem(
        [9, 2, 12, 6], [64, 30, 20, 80], seed=52, pool=pool, table_width=width
    )
    plan_kw, run_kw = dict(lse_mode=lse_mode), dict(k_scale=0.5, v_scale=2.0)
    if path == "wrapper_run":
        _decline_staging(monkeypatch)
    attn, backend_obj = _plan(
        p1, backend, graph_capacity=_capacity(p1, rows=rows, max_q_len=32), **plan_kw
    )
    assert (backend_obj._staged is not None) == (path == "direct")
    wrapper_runs = _spy_wrapper_run(monkeypatch, backend_obj._active)

    q, out, lse = _buffers(p1, rows=rows, lse=lse_mode != "none")
    buffers = (q, p1["k_cache"].clone(), p1["v_cache"].clone(), out, lse)
    kv = buffers[1:3]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):  # module load outside the capture
            attn.run(q, kv, out=out, lse=lse, **run_kw)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn.run(q, kv, out=out, lse=lse, **run_kw)
    assert bool(wrapper_runs) == (path == "wrapper_run")

    for p in (p1, p2):
        _load(buffers, p)
        if p is p2:
            torch.cuda.set_sync_debug_mode("error")
            try:
                _plan(p, backend, attn=attn, **plan_kw)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            assert (backend_obj._staged is not None) == (path == "direct")
        graph.replay()
        torch.cuda.synchronize()
        _assert_oracle(out, lse, p, plan_kw, run_kw)


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
def test_run_with_caller_buffers_makes_no_device_allocation(backend, graph, lse_mode):
    """With out= / lse= supplied, a warm direct launch allocates nothing on
    the device -- eager, and in graph mode where the launch gets row-prefix
    views of larger buffers; the scales and the base-e fold work in place."""
    p = _problem(*_RAGGED, seed=55)
    rows = p["q"].shape[0] + (8 if graph else 0)
    plan_kw, run_kw = dict(lse_mode=lse_mode), dict(k_scale=0.5, v_scale=2.0)
    attn, backend_obj = _plan(
        p, backend, graph_capacity=_capacity(p, rows=rows) if graph else None, **plan_kw
    )
    assert backend_obj._staged is not None
    q, out, lse = _buffers(p, rows=rows, lse=lse_mode != "none")
    kv = (p["k_cache"], p["v_cache"])
    dev = torch.device(DEVICE)
    attn.run(q, kv, out=out, lse=lse, **run_kw)  # warm: module load
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    got_out, got_lse = attn.run(q, kv, out=out, lse=lse, **run_kw)
    torch.cuda.synchronize(dev)
    allocations = torch.cuda.memory_stats(dev)["allocation.all.allocated"] - before
    assert allocations == 0, f"run() made {allocations} device allocation(s)"
    assert got_out is out and got_lse is lse
    _assert_oracle(out, lse, p, plan_kw, run_kw)


# ---------------------------------------------------------------------------
# 6: run() does not synchronize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("lse_mode", ["none", "basee"])
def test_run_does_not_synchronize(backend, lse_mode):
    """The direct launch issues no blocking call -- allocating its own
    buffers or writing the caller's, behind a busy stream."""
    p = _problem(*_RAGGED, seed=61)
    plan_kw, run_kw = dict(lse_mode=lse_mode), dict(k_scale=0.5, v_scale=2.0)
    attn, backend_obj = _plan(p, backend, **plan_kw)
    assert backend_obj._staged is not None
    _, out_buf, lse_buf = _buffers(p, lse=lse_mode != "none")
    kv = (p["k_cache"], p["v_cache"])
    attn.run(p["q"], kv, **run_kw)  # warm: module load
    torch.cuda.synchronize()
    busy = torch.randn(2048, 2048, device=DEVICE)
    for _ in range(8):
        busy = busy @ busy.T / 2048.0  # queued ahead of the launches
    torch.cuda.set_sync_debug_mode("error")
    try:
        allocated = attn.run(p["q"], kv, **run_kw)
        caller = attn.run(p["q"], kv, out=out_buf, lse=lse_buf, **run_kw)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for out, lse in (allocated, caller):
        _assert_oracle(out, lse, p, plan_kw, run_kw)


# ---------------------------------------------------------------------------
# 7: when staging declines
# ---------------------------------------------------------------------------

# The wrapper states for which the legacy run() leaves its plain path before
# or instead of the paged_run launch the staged list models: the attribute
# and a value that puts the wrapper there.
_DECLINED_STATES = {
    "batch_invariant": ("_batch_invariant", object()),
    "backend_cudnn": ("_backend", "cudnn"),
    "backend_trtllm_gen": ("_backend", "trtllm-gen"),
    "backend_cute_dsl": ("_backend", "cute-dsl"),
    "backend_cute_dsl_prims": ("_backend", "cute-dsl-prims"),
    "jit_module": ("_jit_module", object()),
    "nvfp4_kv": ("_cached_kv_data_type", torch.uint8),
    "token_head_sf": ("_use_token_head_sf", True),
    "eagle_verify_fp8kv_sm90": ("_eagle_verify_fp8kv_sm90_state", object()),
    "multi_item_scoring": ("_prefix_len_ptr", torch.zeros(4, dtype=torch.int32)),
    "fp8_q": ("_cached_q_data_type", FP8),
    "no_plan_info": ("_plan_info", None),
    "no_module": ("_cached_module", None),
    "subclass": ("__class__", None),
}


@pytest.mark.parametrize("backend", FA_BACKENDS)
@pytest.mark.parametrize("state", list(_DECLINED_STATES))
def test_staging_declines_what_the_wrapper_run_does_not_launch_plainly(
    monkeypatch, backend, state
):
    """Each state alone turns a staged plain plan into a declined one, so
    run() keeps calling the wrapper's own ``run()`` there (``subclass``: any
    class with its own ``run()``, as the AttentionSink variant is)."""
    attn, backend_obj = _plan(_problem(*_RAGGED, seed=71), backend)
    wrapper, meta = backend_obj._active, attn._impl._meta
    slopes = _get_cache_alibi_slopes_buf(meta.num_qo_heads, wrapper.device)
    assert backend_obj._stage_paged_run(wrapper, meta, slopes) is not None
    attribute, value = _DECLINED_STATES[state]
    if state == "subclass":
        value = type("Variant", (type(wrapper),), {})
    monkeypatch.setattr(wrapper, attribute, value)
    assert backend_obj._stage_paged_run(wrapper, meta, slopes) is None


# What the legacy run() reads from the wrapper and the staging neither stages
# nor declines on, and why it need not.
_NOT_STAGED = {
    # launches of the other backends, declined by ``_backend``
    "_cudnn_lse_unservable",
    "_cudnn_plan",
    "_cudnn_prepared",
    "_cudnn_prepared_cache",
    "_seq_lens_q",
    "_seq_lens_kv",
    "_cute_dsl_wrapper",
    "_prims_backend",
    # the argument tail of a custom JIT module, declined by ``_jit_module``
    "_jit_additional_tensor_names",
    "_jit_additional_scalar_names",
    # checks of per-(token, head) scaled KV, declined by ``_use_token_head_sf``
    "_head_dim_qk",
    "_head_dim_vo",
    # the q row count: the controller checks q against the plan
    "_qo_indptr_last",
    # the plan-time softmax scale: the backend's run() passes the call's
    "_sm_scale",
}


def _private_attributes_read(function, owner):
    """The ``_name`` attributes the source of ``function`` reads from the
    variable ``owner``, directly or through ``getattr``."""
    source = inspect.getsource(inspect.unwrap(function))
    return set(re.findall(rf"\b{owner}\.(_[a-z]\w*)", source)) | set(
        re.findall(rf'getattr\(\s*{owner},\s*"(_[a-z]\w*)"', source)
    )


def test_staging_accounts_for_every_attribute_the_wrapper_run_reads():
    """The decline list names the legacy ``run()``'s ways out of its plain
    path by attribute, and the drift guard sees only what reaches
    ``paged_run``: a dispatch keyed on plan-time state that upstream adds
    ahead of the launch would be visible to neither.  So every private
    attribute that ``run()`` reads is one the staging reads (to stage it or to
    decline on it) or one listed in ``_NOT_STAGED`` with its reason; a new one
    fails here until it is audited, and staged, declined on (with its case in
    ``_DECLINED_STATES``) or listed."""
    legacy = _private_attributes_read(BatchPrefillWithPagedKVCacheWrapper.run, "self")
    staged = _private_attributes_read(fa_backend._FaBackend._stage_paged_run, "w")
    # the sources were found and parsed: both name what every launch needs
    assert {"_plan_info", "_cached_module", "_batch_invariant"} <= legacy & staged
    unaccounted = sorted(legacy - staged - _NOT_STAGED)
    assert not unaccounted, (
        f"BatchPrefillWithPagedKVCacheWrapper.run() reads {unaccounted}, which "
        "_FaBackend._stage_paged_run neither stages nor declines on"
    )
