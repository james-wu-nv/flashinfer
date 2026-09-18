"""Coverage tests TC01-TC12 for the unified paged attention API (experimental).

Companion to ``test_paged_attention_prototype.py`` (conformance matrix),
``test_paged_attention_fuzzer.py`` (reject-or-correct) and
``test_paged_attention_cuda_graph.py`` (re-plan protocol).  Each test here
closes one gap named in the test-coverage report (T01-T12) and its review:

- TC01 LSE modes x caller buffers (``none`` returns None, buffers are the
  caller's, neighbouring memory untouched)
- TC02 one plan, many layer scales (also fp8 KV scale pairs; invalid scales)
- TC03 page / window edges, dense and CSR bitwise identical
- TC04 real mixed lengths in ONE call (vLLM and SGLang fixtures, skewed batch)
- TC05 strided inputs: supported-and-correct or rejected, never wrong
- TC06 shared prefix pages, NaN canaries in unreachable pages, remapping
- TC07 CUDA-graph replay across features (CSR page 1, NHD, SWA, fp16, fp8 KV,
  basee / none)
- TC08 capability required rows: a manifest the backend's own resolve() may
  not filter — a capability regression FAILS, only hardware / dependency
  unavailability skips
- TC09 contract rejections: field-level ValueError before any launch, and
  legal inputs are not rejected
- TC10 CSR / dense equivalence per backend (including fa3)
- TC11 several instances on the shared workspace, each against its own oracle
- TC12 the legacy empty-split-chunk fixture (q2 / kv129 / page16)

The fp32 oracle in ``paged_attention_reference.py`` is the only reference;
no backend is ever compared against another backend as truth.

Known library gaps are recorded as ``xfail(strict=True)`` with the ledger id
and the work package that fixes them, so they flip to XPASS when fixed; none
remain at the integrated head.
"""

import math

import pytest
import torch

from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    quantize_kv,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda:0"
BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "cake", "auto"]
EXPLICIT_BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "cake"]
FP8 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
CANARY = -7.0  # exactly representable in bf16 / fp16 / fp32


# ---------------------------------------------------------------------------
# problem builder: explicit per-request lengths, seeded on the CUDA RNG too
# ---------------------------------------------------------------------------


def build_problem(
    q_lens,
    kv_lens,
    *,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo=None,
    page_size,
    dtype,
    seed,
    kv_layout="HND",
    input_form="block_tables",
    kv_dtype=None,
    pool_slack=8,
    table_width=None,
    value_scale=1.0,
    device=DEVICE,
):
    """A paged-prefill problem with explicit lengths, scattered page ids and
    Q/K/V drawn from generators seeded by ``seed`` (bitwise reproducible).

    Returns the same dict layout as ``test_paged_attention_prototype.make_problem``
    so the shared helpers (``make_metadata``, ``_resolve_or_skip``) apply.
    ``kv_dtype=float8_e4m3fn`` quantizes K/V per tensor and keeps the
    dequantized values as ``k_ref`` / ``v_ref`` for the oracle.
    """
    head_dim_vo = head_dim_vo or head_dim_qk
    q_lens = torch.as_tensor(q_lens, dtype=torch.int32)
    kv_lens = torch.as_tensor(kv_lens, dtype=torch.int32)
    assert q_lens.shape == kv_lens.shape and q_lens.dim() == 1
    b = int(q_lens.shape[0])
    g = torch.Generator().manual_seed(seed)
    g_cuda = torch.Generator(device=device).manual_seed(seed)

    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages_per_seq = (kv_lens + page_size - 1) // page_size
    width = int(pages_per_seq.max()) if b else 0
    if table_width is not None:
        assert table_width >= width
        width = table_width
    pool_pages = int(pages_per_seq.sum()) + pool_slack
    perm = torch.randperm(pool_pages, generator=g, dtype=torch.int32)
    block_tables_cpu = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages_per_seq[i])
        block_tables_cpu[i, :n] = perm[off : off + n]
        off += n
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages_per_seq[i])] for i in range(b)]
        + [torch.zeros(0, dtype=torch.int32)]
    ).to(torch.int32)

    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(
        total_q, num_qo_heads, head_dim_qk, dtype=dtype, device=device, generator=g_cuda
    )
    if kv_layout == "HND":
        k_shape = (pool_pages, num_kv_heads, page_size, head_dim_qk)
        v_shape = (pool_pages, num_kv_heads, page_size, head_dim_vo)
    else:
        k_shape = (pool_pages, page_size, num_kv_heads, head_dim_qk)
        v_shape = (pool_pages, page_size, num_kv_heads, head_dim_vo)
    k_cache = torch.randn(*k_shape, dtype=dtype, device=device, generator=g_cuda)
    v_cache = torch.randn(*v_shape, dtype=dtype, device=device, generator=g_cuda)
    if value_scale != 1.0:
        q = (q * value_scale).to(dtype)
        k_cache = (k_cache * value_scale).to(dtype)
        v_cache = (v_cache * value_scale).to(dtype)
    k_ref, v_ref, k_scale, v_scale = k_cache, v_cache, None, None
    if kv_dtype is not None and kv_dtype != dtype:
        k_cache, v_cache, k_ref, v_ref, k_scale, v_scale = quantize_kv(
            k_cache, v_cache, kv_dtype
        )

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_ref=k_ref,
        v_ref=v_ref,
        kv_dtype=kv_dtype if kv_dtype is not None else dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(device),
        kv_seq_lens_cpu=kv_lens,
        block_tables=block_tables_cpu.to(device),
        kv_page_indices=kv_page_indices_cpu.to(device),
        kv_layout=kv_layout,
        input_form=input_form,
        page_size=page_size,
        max_q_len=int(q_lens.max()) if b else 1,
        max_kv_len=int(kv_lens.max()) if b else 1,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=dtype,
        device=device,
    )


def reference(p, *, causal=True, window_left=-1, sm_scale=None, lse_base="2"):
    """fp32 oracle on the problem dict, honouring its paging form."""
    csr = p.get("input_form") == "page_indices"
    return reference_paged_prefill(
        p["q"].contiguous(),
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        None if csr else p["block_tables"],
        p["page_size"],
        causal,
        sm_scale=sm_scale,
        window_left=window_left,
        kv_layout=p.get("kv_layout", "HND"),
        kv_page_indices=p["kv_page_indices"] if csr else None,
        lse_base=lse_base,
    )


def plan(attn, p, backend, *, causal=True, window_left=-1, lse_mode="base2"):
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        kv_layout=p.get("kv_layout", "HND"),
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        backend=backend,
    )
    return attn


def run(attn, p, **kw):
    kw.setdefault("k_scale", p.get("k_scale"))
    kw.setdefault("v_scale", p.get("v_scale"))
    return attn.run(p["q"], (p["k_cache"], p["v_cache"]), **kw)


def assert_matches(
    out, lse, p, *, causal=True, window_left=-1, sm_scale=None, lse_mode
):
    ref_out, ref_lse = reference(
        p,
        causal=causal,
        window_left=window_left,
        sm_scale=sm_scale,
        lse_base="e" if lse_mode == "basee" else "2",
    )
    assert torch.isfinite(out).all(), "non-finite output"
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    if lse_mode == "none":
        assert lse is None, "lse_mode='none' must return (out, None)"
    else:
        assert lse.shape == (p["q"].shape[0], p["num_qo_heads"])
        assert lse.dtype == torch.float32
        assert torch.isfinite(lse).all(), "non-finite LSE"
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    return ref_out, ref_lse


def plan_and_check(p, backend, **kw):
    """resolve-or-skip, plan, run, oracle.  Returns (attn, out, lse)."""
    causal = kw.get("causal", True)
    window_left = kw.get("window_left", -1)
    lse_mode = kw.get("lse_mode", "base2")
    _resolve_or_skip(
        p, backend, causal=causal, need_lse=lse_mode != "none", window_left=window_left
    )
    attn = plan(
        PagedAttention(torch.device(p["device"])),
        p,
        backend,
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
    )
    out, lse = run(attn, p)
    assert_matches(
        out, lse, p, causal=causal, window_left=window_left, lse_mode=lse_mode
    )
    return attn, out, lse


def launch_spy(attn):
    """Count backend launches so a rejection can be proven to happen BEFORE
    any kernel is dispatched."""
    active = attn._impl._active
    calls = []
    real = active.run

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    active.run = spy
    return calls


def canary_buffers(p, pad=64):
    """Caller out/lse buffers carved from bigger canary-filled allocations."""
    total, h = p["q"].shape[0], p["num_qo_heads"]
    dev = p["q"].device
    out_big = torch.full(
        (total + 2 * pad, h, p["head_dim_vo"]), CANARY, dtype=p["dtype"], device=dev
    )
    lse_big = torch.full((total + 2 * pad, h), CANARY, dtype=torch.float32, device=dev)
    return out_big, out_big[pad : pad + total], lse_big, lse_big[pad : pad + total]


def assert_canaries(big, inner, pad):
    total = inner.shape[0]
    assert torch.all(big[:pad] == CANARY), "memory BEFORE the caller buffer was written"
    assert torch.all(big[pad + total :] == CANARY), (
        "memory AFTER the caller buffer was written"
    )


# ---------------------------------------------------------------------------
# TC01 — LSE modes and caller buffers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("buffers", ["alloc", "caller"])
@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_tc01_lse_modes_and_caller_buffers(backend, lse_mode, buffers, kv):
    """Same problem under every LSE mode, with and without caller buffers.

    Contract: ``none`` returns ``(out, None)``; base2 / basee match the oracle
    in that base; returned tensors ARE the caller's buffers; the memory around
    a caller buffer is untouched; ``lse=`` with ``lse_mode='none'`` is
    rejected before launch and leaves the plan runnable."""
    p = build_problem(
        [5, 1, 17],
        [40, 9, 33],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_dtype=FP8 if kv == "fp8" else None,
        seed=101,
    )
    _resolve_or_skip(p, backend, need_lse=lse_mode != "none")
    attn = plan(PagedAttention(torch.device(DEVICE)), p, backend, lse_mode=lse_mode)
    pad = 64
    out_big, out_buf, lse_big, lse_buf = canary_buffers(p, pad)
    if buffers == "caller":
        out, lse = run(
            attn, p, out=out_buf, lse=lse_buf if lse_mode != "none" else None
        )
        assert out.data_ptr() == out_buf.data_ptr(), "out is not the caller's buffer"
        assert_canaries(out_big, out_buf, pad)
        if lse_mode != "none":
            assert lse.data_ptr() == lse_buf.data_ptr(), (
                "lse is not the caller's buffer"
            )
            assert_canaries(lse_big, lse_buf, pad)
        else:
            assert torch.all(lse_big == CANARY)  # never handed over, never written
            with pytest.raises(ValueError, match="lse_mode='none'"):
                run(attn, p, out=out_buf, lse=lse_buf)
            assert torch.all(lse_big == CANARY)
            out, lse = run(attn, p, out=out_buf)  # the plan is still runnable
    else:
        out, lse = run(attn, p)
        assert out.data_ptr() not in (out_buf.data_ptr(),)
    assert_matches(out, lse, p, lse_mode=lse_mode)
    if lse_mode == "base2":
        # base-2 and natural-log are one fold apart: cross-check the contract
        _, ref_e = reference(p, lse_base="e")
        torch.testing.assert_close(lse * math.log(2.0), ref_e, **LSE_TOL)


# ---------------------------------------------------------------------------
# TC02 — same plan, per-layer scales
# ---------------------------------------------------------------------------

_TC02_SHAPE = dict(num_qo_heads=8, num_kv_heads=2, head_dim_qk=128, page_size=16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc02_same_plan_layer_scales(backend):
    """ONE plan; run with sm_scale = s, 3s, s.  Each run matches the oracle at
    its own scale and the third returns to the first result (the existing
    ``test_paged_attention_sm_scale_replan`` re-plans every time and so
    cannot see a stale per-plan scale)."""
    p = build_problem(
        [7, 20, 1], [64, 30, 100], dtype=torch.bfloat16, seed=202, **_TC02_SHAPE
    )
    _resolve_or_skip(p, backend)
    attn = plan(PagedAttention(torch.device(DEVICE)), p, backend)
    s = 1.0 / math.sqrt(p["head_dim_qk"])
    results = []
    for scale in (s, 3.0 * s, s):
        out, lse = run(attn, p, sm_scale=scale)
        assert_matches(out, lse, p, sm_scale=scale, lse_mode="base2")
        results.append((out.clone(), lse.clone()))
    assert not torch.allclose(results[0][0], results[1][0], **OUT_TOL)  # 3s differs
    torch.testing.assert_close(results[2][0], results[0][0], atol=4e-3, rtol=2e-2)
    torch.testing.assert_close(results[2][1], results[0][1], atol=4e-3, rtol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc02_same_plan_fp8_kv_scale_pairs(backend):
    """fa2 fp8 KV: one plan, two different positive (k_scale, v_scale) pairs;
    the oracle dequantizes with the scales of each call."""
    p = build_problem(
        [7, 20, 1],
        [64, 30, 100],
        dtype=torch.bfloat16,
        kv_dtype=FP8,
        seed=203,
        **_TC02_SHAPE,
    )
    _resolve_or_skip(p, backend)
    attn = plan(PagedAttention(torch.device(DEVICE)), p, backend)
    base_k, base_v = p["k_scale"], p["v_scale"]
    for ks, vs in ((base_k, base_v), (2.0 * base_k, 0.5 * base_v), (base_k, base_v)):
        out, lse = run(attn, p, k_scale=ks, v_scale=vs)
        q = dict(p, k_ref=p["k_cache"].float() * ks, v_ref=p["v_cache"].float() * vs)
        assert_matches(out, lse, q, lse_mode="base2")


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc02_invalid_scales_rejected_before_launch(backend):
    p = build_problem([4, 9], [30, 40], dtype=torch.bfloat16, seed=204, **_TC02_SHAPE)
    _resolve_or_skip(p, backend)
    attn = plan(PagedAttention(torch.device(DEVICE)), p, backend)
    calls = launch_spy(attn)
    for bad in (float("nan"), float("inf"), -float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="sm_scale"):
            run(attn, p, sm_scale=bad)
    for nm in ("k_scale", "v_scale"):
        with pytest.raises(ValueError, match="fp8 KV caches only"):
            run(attn, p, **{nm: 1.0})  # bf16 KV plan: scales are meaningless
    assert calls == [], "a rejected call reached the backend"
    out, lse = run(attn, p)  # legal call still works
    assert calls == [1]
    assert_matches(out, lse, p, lse_mode="base2")

    p8 = build_problem(
        [4, 9], [30, 40], dtype=torch.bfloat16, kv_dtype=FP8, seed=205, **_TC02_SHAPE
    )
    if "fa2" not in resolve_paged_attention(
        device=torch.device(DEVICE),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        kv_dtype=FP8,
        page_size=16,
        need_lse=True,
        backend="auto",
    ).backends or backend not in ("fa2", "auto"):
        return
    attn8 = plan(PagedAttention(torch.device(DEVICE)), p8, backend)
    calls8 = launch_spy(attn8)
    for nm in ("k_scale", "v_scale"):
        for bad in (float("nan"), float("inf"), 0.0, -0.5):
            with pytest.raises(ValueError, match=nm):
                run(attn8, p8, **{nm: bad})
    assert calls8 == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc02_graph_bakes_the_captured_sm_scale(backend):
    """Documentary: a captured graph keeps the scalar it was captured with.
    A later eager run with another scale does not change what replay
    computes — changing sm_scale under a graph needs a recapture."""
    p = build_problem([6, 12], [40, 70], dtype=torch.bfloat16, seed=206, **_TC02_SHAPE)
    _resolve_or_skip(p, backend)
    dev = torch.device(DEVICE)
    attn = plan(PagedAttention(dev, use_cuda_graph=True), p, backend)
    out = torch.empty(p["q"].shape[0], 8, 128, dtype=torch.bfloat16, device=dev)
    lse = torch.empty(p["q"].shape[0], 8, dtype=torch.float32, device=dev)
    s = 1.0 / math.sqrt(128)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run(attn, p, out=out, lse=lse, sm_scale=s)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run(attn, p, out=out, lse=lse, sm_scale=s)
    g.replay()
    torch.cuda.synchronize()
    assert_matches(out, lse, p, sm_scale=s, lse_mode="base2")
    run(attn, p, out=out, lse=lse, sm_scale=3.0 * s)  # eager, other scale
    torch.cuda.synchronize()
    assert_matches(out, lse, p, sm_scale=3.0 * s, lse_mode="base2")
    g.replay()  # the graph still computes the captured scale
    torch.cuda.synchronize()
    assert_matches(out, lse, p, sm_scale=s, lse_mode="base2")


# ---------------------------------------------------------------------------
# TC09 — contract rejections (field-level ValueError before launch)
# ---------------------------------------------------------------------------

# The zero-row contract (ledger M11): a request with kv_len == 0 is a legal
# padding row whose output is finite and whose neighbours are unaffected (see
# test_tc09_zero_kv_row).  Kept as a switch so the pre-contract behaviour can
# still be exercised against an older branch.
ZERO_KV_ROWS_ARE_LEGAL = True


def _legal(seed=900, **over):
    kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    kw.update(over)
    return build_problem([4, 8, 2], [20, 40, 9], seed=seed, **kw)


def _with_indptr(p, ip_cpu):
    return dict(p, qo_indptr_cpu=ip_cpu, qo_indptr=ip_cpu.to(p["device"]))


def _with_kv_lens(p, kv_cpu):
    return dict(p, kv_seq_lens_cpu=kv_cpu, kv_seq_lens=kv_cpu.to(p["device"]))


def _md_rows():
    dev = torch.device(DEVICE)
    i32 = dict(dtype=torch.int32, device=dev)
    rows = [
        (
            "batch0",
            lambda p: dict(
                p,
                qo_indptr=torch.zeros(1, **i32),
                qo_indptr_cpu=torch.zeros(1, dtype=torch.int32),
                kv_seq_lens=torch.zeros(0, **i32),
                kv_seq_lens_cpu=torch.zeros(0, dtype=torch.int32),
                block_tables=torch.zeros(0, 3, **i32),
            ),
            "batch size must be >= 1",
        ),
        (
            "indptr_decreasing",
            lambda p: _with_indptr(p, torch.tensor([0, 5, 4, 14], dtype=torch.int32)),
            "non-decreasing",
        ),
        (
            "all_q_len0",
            lambda p: dict(
                _with_indptr(p, torch.zeros(4, dtype=torch.int32)),
                q=p["q"][:0],
            ),
            "at least one query token",
        ),
        (
            "indptr_not_from_zero",
            lambda p: _with_indptr(p, p["qo_indptr_cpu"] + 2),
            r"qo_indptr\[0\] must be 0",
        ),
        (
            "indptr_int64",
            lambda p: dict(p, qo_indptr=p["qo_indptr"].to(torch.int64)),
            "must be int32",
        ),
        (
            "indptr_on_cpu",
            lambda p: dict(p, qo_indptr=p["qo_indptr_cpu"]),
            "must be on CUDA device",
        ),
        (
            "kv_lens_2d",
            lambda p: dict(p, kv_seq_lens=p["kv_seq_lens"].unsqueeze(0)),
            "must be 1-D",
        ),
        (
            "kv_lens_wrong_length",
            lambda p: dict(p, kv_seq_lens=p["kv_seq_lens"][:-1]),
            r"qo_indptr must have shape \(batch_size\+1,\)",
        ),
        (
            "block_tables_1d",
            lambda p: dict(p, block_tables=p["block_tables"].flatten()),
            "must be 2-D",
        ),
        (
            "block_tables_wrong_rows",
            lambda p: dict(p, block_tables=p["block_tables"][:-1]),
            "block_tables must have shape",
        ),
        (
            "block_tables_int64",
            lambda p: dict(p, block_tables=p["block_tables"].to(torch.int64)),
            "must be int32",
        ),
        ("both_paging_forms", lambda p: dict(p, input_form="both"), "EXACTLY ONE"),
        (
            "csr_too_short",
            lambda p: dict(
                p, input_form="page_indices", kv_page_indices=p["kv_page_indices"][:-1]
            ),
            "kv_page_indices has",
        ),
        ("max_q_len0", lambda p: dict(p, max_q_len=0), "must be a positive host int"),
        (
            "max_q_len_underclaim",
            lambda p: dict(p, max_q_len=p["max_q_len"] - 1),
            "is smaller than the actual longest query",
        ),
        (
            "max_kv_len_underclaim",
            lambda p: dict(p, max_kv_len=p["max_kv_len"] - 1),
            "is smaller than the actual longest KV",
        ),
        (
            "max_kv_len_float",
            lambda p: dict(p, max_kv_len=float(p["max_kv_len"])),
            "must be a positive host int",
        ),
        ("dense_page_size4", lambda p: dict(p, page_size=4), "< 8"),
        (
            "kv_lens_exceed_table",
            lambda p: dict(
                _with_kv_lens(p, torch.tensor([20, 3 * 16 + 1, 9], dtype=torch.int32)),
                max_kv_len=3 * 16 + 1,
            ),
            "exceeds block_tables capacity",
        ),
    ]
    return rows


_MD_ROWS = _md_rows()


@pytest.mark.parametrize("label,mutate,match", _MD_ROWS, ids=[r[0] for r in _MD_ROWS])
def test_tc09_metadata_rejections(label, mutate, match):
    """Malformed canonical metadata is rejected at construction with a
    field-level ValueError (no plan, no launch)."""
    p = mutate(_legal())
    with pytest.raises(ValueError, match=match):
        make_metadata(p)


def test_tc09_zero_kv_row():
    """kv_len == 0 rows are legal padding rows (ledger M11): finite output,
    neighbours unaffected."""
    p = _legal(seed=901)
    kv0 = p["kv_seq_lens_cpu"].clone()
    kv0[1] = 0
    p0 = _with_kv_lens(p, kv0)
    if not ZERO_KV_ROWS_ARE_LEGAL:
        with pytest.raises(ValueError, match="outside the v1 envelope"):
            make_metadata(p0)
        return
    # zero-row contract: q rows of request 1 are padding; rows 0 and 2 must
    # equal the same problem without request 1
    md = make_metadata(p0)
    attn = PagedAttention(torch.device(DEVICE)).plan(
        md,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        causal=False,
        lse_mode="base2",
    )
    out, lse = run(attn, p0)
    keep = torch.cat([torch.arange(0, 4), torch.arange(12, 14)]).to(DEVICE)
    assert torch.isfinite(out[keep]).all()  # padding rows may be left unwritten
    q2 = dict(
        p,
        q=p["q"][keep],
        qo_indptr_cpu=torch.tensor([0, 4, 6], dtype=torch.int32),
        qo_indptr=torch.tensor([0, 4, 6], dtype=torch.int32, device=DEVICE),
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"][[0, 2]],
        kv_seq_lens=p["kv_seq_lens"][[0, 2]],
        block_tables=p["block_tables"][[0, 2]],
        max_q_len=4,
    )
    ref_out, ref_lse = reference(q2, causal=False)
    torch.testing.assert_close(out[keep].float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse[keep], ref_lse, **LSE_TOL)


def _drop_query_rows(p, request, *, kv_zero):
    """Request ``request`` of a build_problem() batch becomes a q_len 0 row
    (its query tokens are removed from ``q``; ``kv_zero`` also zeroes its KV
    length, vLLM's padded tail row).  Returns the problem and the compacted
    problem without that request, the oracle for the remaining rows."""
    qo = p["qo_indptr_cpu"]
    s, e = int(qo[request]), int(qo[request + 1])
    keep = torch.cat([torch.arange(0, s), torch.arange(e, int(qo[-1]))])
    q_lens = qo.diff().clone()
    q_lens[request] = 0
    new_qo = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    kv = p["kv_seq_lens_cpu"].clone()
    if kv_zero:
        kv[request] = 0
    p0 = dict(
        _with_kv_lens(_with_indptr(p, new_qo), kv),
        q=p["q"][keep.to(p["device"])],
        max_q_len=int(q_lens.max()),
    )
    others = [i for i in range(qo.shape[0] - 1) if i != request]
    kept_lens = q_lens[others]
    compact = dict(
        p,
        q=p0["q"],
        qo_indptr_cpu=torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(kept_lens, 0, dtype=torch.int32),
            ]
        ),
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"][others],
        block_tables=p["block_tables"][others],
        max_q_len=int(kept_lens.max()),
    )
    compact["qo_indptr"] = compact["qo_indptr_cpu"].to(p["device"])
    compact["kv_seq_lens"] = compact["kv_seq_lens_cpu"].to(p["device"])
    return p0, compact


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kv_zero", [False, True], ids=["kv_live", "kv_zero"])
def test_tc09_zero_q_row(backend, kv_zero):
    """q_len == 0 rows are legal (ledger M17): request 1 owns no query token
    (``kv_zero``: and no KV either, vLLM's padded query_start_loc tail).  The
    remaining rows equal the same batch without request 1.  Measured
    natively on B200 for fa2, cuDNN, trtllm-gen and cake before the contract
    was relaxed; cake declines the kv_len 0 batch (M19) and auto moves on."""
    p0, compact = _drop_query_rows(_legal(seed=903), 1, kv_zero=kv_zero)
    assert int(p0["qo_indptr_cpu"][1]) == int(p0["qo_indptr_cpu"][2])
    _resolve_or_skip(p0, backend)
    md = make_metadata(p0)
    kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
    )
    if backend == "cake" and kv_zero:
        with pytest.raises(ValueError, match="kv_len == 0"):
            PagedAttention(torch.device(DEVICE)).plan(md, backend=backend, **kw)
        return
    attn = PagedAttention(torch.device(DEVICE)).plan(md, backend=backend, **kw)
    out, lse = run(attn, p0)
    ref_out, ref_lse = reference(compact, causal=True)
    assert out.shape[0] == ref_out.shape[0] == int(p0["qo_indptr_cpu"][-1])
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_tc09_zero_kv_row_cake_declines():
    """ledger M19: the cake kernel hangs on a kv_len == 0 request, so cake
    declines padding rows with the typed signal — auto never lands on it and
    an explicit pin gets a ValueError instead of a hung device."""
    p = _legal(seed=902)
    kv0 = p["kv_seq_lens_cpu"].clone()
    kv0[1] = 0
    p0 = _with_kv_lens(p, kv0)
    md = make_metadata(p0)
    kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
    )
    _resolve_or_skip(p0, "cake")
    with pytest.raises(ValueError, match="kv_len == 0"):
        PagedAttention(torch.device(DEVICE)).plan(md, backend="cake", **kw)
    attn = PagedAttention(torch.device(DEVICE)).plan(md, backend="auto", **kw)
    assert attn.backend != "cake"
    out, _ = run(attn, p0)
    # The padding row's output is not asserted finite here: on SM100 auto
    # lands on trtllm-gen, which leaves a kv_len 0 row's output rows
    # unwritten (measured on B200 with a NaN-poisoned allocation; fa2 and
    # cuDNN write zeros), so that assertion held by allocator luck only.
    # The live rows are pinned by test_tc09_zero_kv_row.
    assert out.shape[0] == int(p0["qo_indptr_cpu"][-1])


_PLAN_ROWS = [
    ("unknown_backend", dict(backend="fa9"), "unknown backend"),
    ("unknown_lse_mode", dict(lse_mode="base10"), "lse_mode must be one of"),
    ("window_left_minus2", dict(window_left=-2), "window_left must be >= -1"),
    ("kv_layout_bad", dict(kv_layout="DNH"), "kv_layout must be"),
    ("heads_not_divisible", dict(num_qo_heads=7), "divisible"),
    ("head_dim_vo_unsupported", dict(head_dim_vo=96), "no runnable backend"),
]


@pytest.mark.parametrize("label,over,match", _PLAN_ROWS, ids=[r[0] for r in _PLAN_ROWS])
def test_tc09_plan_rejections(label, over, match):
    p = _legal(seed=902)
    kw = dict(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        causal=True,
        lse_mode="base2",
        backend="auto",
    )
    kw.update(over)
    attn = PagedAttention(torch.device(DEVICE))
    with pytest.raises(ValueError, match=match):
        attn.plan(make_metadata(p), **kw)
    assert attn.backend is None  # nothing was published
    with pytest.raises(ValueError, match="before plan"):
        run(attn, p)


def test_tc09_causal_envelope_and_metadata_device():
    p = _legal(seed=903)
    kv = p["kv_seq_lens_cpu"].clone()
    kv[1] = 7  # q_len 8 > kv_len 7
    attn = PagedAttention(torch.device(DEVICE))
    with pytest.raises(ValueError, match="q_len_i <= kv_len_i"):
        plan(attn, _with_kv_lens(p, kv), "auto", causal=True)
    # the same metadata is legal non-causal
    plan(attn, _with_kv_lens(p, kv), "fa2", causal=False)
    out, lse = run(attn, _with_kv_lens(p, kv))
    assert_matches(out, lse, _with_kv_lens(p, kv), causal=False, lse_mode="base2")
    with pytest.raises(ValueError, match="PagedAttentionMetadata"):
        attn.plan(
            {"not": "metadata"},
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc09_run_rejections_before_launch(backend):
    """run()-time contract: every malformed tensor argument is a field-level
    ValueError raised BEFORE the backend is entered (launch spy), and the
    legal call afterwards is correct — legal inputs are never rejected."""
    p = _legal(seed=904)
    _resolve_or_skip(p, backend)
    dev = torch.device(DEVICE)
    fresh = PagedAttention(dev)
    with pytest.raises(ValueError, match="before plan"):
        run(fresh, p)
    attn = plan(PagedAttention(dev), p, backend)
    calls = launch_spy(attn)
    q, k, v = p["q"], p["k_cache"], p["v_cache"]
    total, h, d = q.shape
    rows = [
        ("q_wrong_dtype", dict(q=q.to(torch.float16)), "q dtype"),
        ("q_wrong_tokens", dict(q=q[:-1]), "tokens but qo_indptr sums"),
        ("q_2d", dict(q=q.reshape(total, h * d)), "q must be packed"),
        ("q_wrong_heads", dict(q=q[:, :4]), "q shape"),
        ("kv_not_pair", dict(kv=(k,)), r"must be a \(k_cache, v_cache\) pair"),
        ("k_transposed", dict(kv=(k.permute(0, 2, 1, 3), v)), "look transposed"),
        ("k_3d", dict(kv=(k[0], v)), "must be 4-D paged"),
        ("v_wrong_dtype", dict(kv=(k, v.to(torch.float16))), "v_cache dtype"),
        ("v_wrong_head_dim", dict(kv=(k, v[..., :64])), "v_cache shape"),
        (
            "out_wrong_shape",
            dict(out=torch.empty(total, h, d // 2, dtype=q.dtype, device=dev)),
            "out must be contiguous",
        ),
        (
            "out_noncontiguous",
            dict(
                out=torch.empty(h, total, d, dtype=q.dtype, device=dev).transpose(0, 1)
            ),
            "out must be contiguous",
        ),
        (
            "out_wrong_dtype",
            dict(out=torch.empty(total, h, d, dtype=torch.float32, device=dev)),
            "out must match q dtype",
        ),
        (
            "lse_wrong_dtype",
            dict(lse=torch.empty(total, h, dtype=torch.float16, device=dev)),
            "lse must be contiguous fp32",
        ),
        (
            "lse_wrong_shape",
            dict(lse=torch.empty(total, h + 1, dtype=torch.float32, device=dev)),
            "lse must be contiguous fp32",
        ),
        ("sm_scale_zero", dict(sm_scale=0.0), "sm_scale"),
        ("k_scale_on_bf16_kv", dict(k_scale=1.0), "fp8 KV caches only"),
    ]
    for label, over, match in rows:
        kv = over.pop("kv", (k, v))
        qq = over.pop("q", q)
        with pytest.raises(ValueError, match=match):
            attn.run(qq, kv, **over)
        assert calls == [], f"{label}: rejected call reached the backend"
    out, lse = run(attn, p)
    assert calls == [1]
    assert_matches(out, lse, p, lse_mode="base2")


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc09_legal_inputs_are_not_rejected(backend):
    """Positive control for the rejection matrix: engine-shaped legal inputs
    (uniform decode rows next to a prefill row, an over-allocated sglang-style
    CSR tail) run and match the oracle."""
    p = build_problem(
        [1, 1, 1, 9],
        [17, 33, 64, 40],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        seed=905,
    )
    plan_and_check(p, backend)
    tail = torch.full((256,), 0, dtype=torch.int32, device=DEVICE)  # sglang +256
    p_csr = dict(
        p,
        input_form="page_indices",
        kv_page_indices=torch.cat([p["kv_page_indices"], tail]),
    )
    plan_and_check(p_csr, backend)


_WIDE_TABLE_BACKENDS = [
    "fa2",
    "fa3",
    "cudnn",
    "cake",
    "trtllm-gen",
    "auto",
]


@pytest.mark.parametrize("backend", _WIDE_TABLE_BACKENDS)
def test_tc09_wide_dense_table_is_legal(backend):
    """A persistent engine block table is wider than the batch needs
    (vLLM: max_model_len / page_size columns).  That is a legal input."""
    p = build_problem(
        [1, 1, 1, 9],
        [17, 33, 64, 40],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        seed=906,
        table_width=8,  # ceil(64 / 16) = 4 columns are live
    )
    plan_and_check(p, backend)


# ---------------------------------------------------------------------------
# TC10 — CSR / dense equivalence, one item per backend (fa3 included)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", EXPLICIT_BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tc10_csr_dense_equivalence(backend, dtype):
    """Dense and flat-indices forms of the SAME tensors are bitwise identical
    per backend, and each matches the oracle (the derivation is exact)."""
    p = build_problem(
        [12, 1, 30, 7],
        [50, 20, 100, 64],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=dtype,
        seed=1001,
    )
    p_csr = dict(p, input_form="page_indices")
    _resolve_or_skip(p, backend)
    _resolve_or_skip(p_csr, backend)
    _, out_a, lse_a = plan_and_check(p, backend)
    _, out_b, lse_b = plan_and_check(p_csr, backend)
    assert torch.equal(out_a, out_b), "dense vs CSR outputs differ"
    assert torch.equal(lse_a, lse_b), "dense vs CSR LSE differ"


# ---------------------------------------------------------------------------
# TC12 — the legacy empty-split-chunk fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_tc12_empty_split_chunk_fixture(backend, dtype):
    """q2 / kv129 / page16 (legacy ``test_batch_prefill_kernels`` fixture that
    produced an empty split chunk) with small-magnitude inputs: finite and
    correct output and LSE."""
    p = build_problem(
        [2],
        [129],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=dtype,
        seed=1201,
        value_scale=0.05,
    )
    plan_and_check(p, backend)
    plan_and_check(p, backend, lse_mode="basee")


# ---------------------------------------------------------------------------
# required-row helpers: a declared row may only be skipped by an environment
# probe, never by the backend's own capability table
# ---------------------------------------------------------------------------

_ENVIRONMENT_REASONS = ("requires SM90a", "not importable", "CUDA >=")


def _cc_major():
    return torch.cuda.get_device_capability(torch.device(DEVICE))[0]


def resolve_required(p, backend, *, causal=True, need_lse=True, window_left=-1):
    """Resolve a row the backend DECLARES.  A capability-table rejection of a
    declared row is a regression and FAILS; only environment probes skip."""
    from flashinfer.experimental.paged_attention import CAPABILITIES

    form = "page_indices" if p.get("input_form") == "page_indices" else "block_tables"
    common = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        page_size=p["page_size"],
        kv_layout=p.get("kv_layout", "HND"),
        causal=causal,
        need_lse=need_lse,
        window_left=window_left,
        kv_input_form=form,
    )
    cap = CAPABILITIES.get(backend)
    if cap is not None:
        declared = cap.rejection_reason(
            cc_major=_cc_major(),
            q_dtype=common["q_dtype"],
            kv_dtype=common["kv_dtype"] or common["q_dtype"],
            head_dim_qk=common["head_dim_qk"],
            head_dim_vo=common["head_dim_vo"],
            page_size=common["page_size"],
            kv_layout=common["kv_layout"],
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=form,
        )
        if declared is not None:
            pytest.skip(f"{backend} does not declare this row: {declared}")
    try:
        return resolve_paged_attention(
            device=torch.device(DEVICE), backend=backend, **common
        )
    except ValueError as e:
        if any(word in str(e) for word in _ENVIRONMENT_REASONS):
            pytest.skip(f"environment: {e}")
        pytest.fail(f"declared row rejected by resolve() [{backend}]: {e}")


def plan_and_check_required(p, backend, **kw):
    causal = kw.get("causal", True)
    window_left = kw.get("window_left", -1)
    lse_mode = kw.get("lse_mode", "base2")
    resolve_required(
        p, backend, causal=causal, need_lse=lse_mode != "none", window_left=window_left
    )
    attn = plan(
        PagedAttention(torch.device(p["device"])),
        p,
        backend,
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
    )
    out, lse = run(attn, p)
    assert_matches(
        out, lse, p, causal=causal, window_left=window_left, lse_mode=lse_mode
    )
    return attn, out, lse


# ---------------------------------------------------------------------------
# TC03 — page and window edges
# ---------------------------------------------------------------------------

_TC03_PAGES = [("csr", 1), ("csr", 5), ("dense", 8), ("dense", 16), ("dense", 32)]
_TC03_WINDOWS = ["none", "0", "1", "P-1", "P", "P+1"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "form,page", _TC03_PAGES, ids=[f"{f}{p}" for f, p in _TC03_PAGES]
)
@pytest.mark.parametrize("window", _TC03_WINDOWS)
def test_tc03_page_and_window_edges(backend, form, page, window):
    """One batch holding every (q_len in {1,2}) x (kv_len in {P-1, P, P+1})
    request, under window_left in {-1, 0, 1, P-1, P, P+1}.  Last-page
    normalisation, bottom-right causal alignment and the window's left/right
    edges all land in one call.  For page sizes >= 8 the dense and CSR forms
    of the same tensors must be bitwise identical.  Every row a backend
    declares is REQUIRED (fa2 page 5 etc.): resolve() failing is a bug."""
    window_left = {
        "none": -1,
        "0": 0,
        "1": 1,
        "P-1": page - 1,
        "P": page,
        "P+1": page + 1,
    }[window]
    kv_cands = [k for k in (page - 1, page, page + 1) if k >= 1]
    reqs = [(q, kv) for kv in kv_cands for q in (1, 2) if q <= kv]
    p = build_problem(
        [q for q, _ in reqs],
        [kv for _, kv in reqs],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=page,
        dtype=torch.bfloat16,
        input_form="page_indices" if form == "csr" else "block_tables",
        seed=300 + page,
    )
    _, out, lse = plan_and_check_required(p, backend, window_left=window_left)
    if form == "dense":
        p_csr = dict(p, input_form="page_indices")
        _, out_b, lse_b = plan_and_check_required(
            p_csr, backend, window_left=window_left
        )
        assert torch.equal(out, out_b), "dense vs CSR outputs differ"
        assert torch.equal(lse, lse_b), "dense vs CSR LSE differ"


def test_tc03_graph_mode_csr_page1_reserves_no_dense_table():
    p = build_problem(
        [3, 5],
        [300, 512],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=1,
        dtype=torch.bfloat16,
        input_form="page_indices",
        seed=333,
    )
    resolve_required(p, "fa2")
    attn = plan(PagedAttention(torch.device(DEVICE), use_cuda_graph=True), p, "fa2")
    gb = attn._impl._graph  # white-box on purpose: the reservation is the bug
    dense = getattr(gb, "block_tables", None)
    assert dense is None or dense.numel() == 0, (
        f"reserved dense table of shape {tuple(dense.shape)} for a CSR-only plan"
    )


# ---------------------------------------------------------------------------
# TC04 — real mixed lengths in ONE call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("heads", [(32, 8), (6, 1)], ids=["h32-8", "h6-1"])
@pytest.mark.parametrize("page", [16, 32])
def test_tc04_vllm_prefill_fixture_whole_batch(backend, heads, page):
    """vLLM ``tests/kernels/attention/test_flashinfer.py::
    test_flashinfer_prefill_with_paged_kv``: (q, kv) = [(1, 1328), (5, 18),
    (129, 463)], heads 32:8 and 6:1 (GQA ratio 6), D128, page 16/32, bf16,
    NHD combined pool — all three rows in ONE call.  The vLLM test never
    passes ``causal=True`` to the wrapper while its reference applies a
    bottom-right causal mask; here causal is explicit.  The 32768-page pool
    of the original is not reproduced (address-range coverage is a separate,
    heavy case)."""
    p = build_problem(
        [1, 5, 129],
        [1328, 18, 463],
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim_qk=128,
        page_size=page,
        dtype=torch.bfloat16,
        kv_layout="NHD",
        seed=400 + page,
        pool_slack=16,
    )
    plan_and_check(p, backend)


# name, page_size, prefix_lens, extend_lens, (num_heads, num_kv_heads)
_SGLANG_DENSE_CASES = [
    ("mha_extend_page_size_1", 1, (2, 4), (3, 1), (4, 4)),
    ("mha_extend_zero_prefix_exact_page", 16, (0,), (16,), (4, 4)),
    ("mha_extend_zero_prefix_input_page_edges", 16, (0, 0, 0), (15, 16, 17), (4, 4)),
    ("mha_extend_prefix_exact_page", 16, (16,), (2,), (4, 4)),
    ("mha_extend_total_exact_page", 16, (8,), (8,), (4, 4)),
    ("mha_extend_cross_page_boundary", 16, (15,), (2,), (4, 4)),
    ("mha_extend_ragged_page_boundary", 16, (0, 8, 16), (15, 8, 1), (4, 4)),
    ("mha_extend_page32_cross_boundary", 32, (31,), (2,), (4, 4)),
    ("mha_decode_page_boundary", 16, (14, 15, 16), (1, 1, 1), (4, 4)),
    ("mha_decode_bsz1_nonzero_prefix", 16, (7,), (1,), (4, 4)),
    ("gqa_decode_page_boundary", 16, (14, 15, 16), (1, 1, 1), (4, 2)),
    ("mqa_extend_total_exact_page", 16, (8,), (8,), (4, 1)),
    ("gqa_extend_ragged_page_boundary", 16, (0, 8, 16), (15, 8, 1), (4, 2)),
    ("mqa_extend_ragged_page_boundary", 16, (0, 8, 16), (15, 8, 1), (4, 1)),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(
    "name,page,prefix,extend,heads",
    _SGLANG_DENSE_CASES,
    ids=[c[0] for c in _SGLANG_DENSE_CASES],
)
def test_tc04_sglang_prefix_extend_cases(
    backend, head_dim, name, page, prefix, extend, heads
):
    """SGLang ``attention_unittest/attention_methods/dense_attention.py``
    input-layout cases: kv = prefix + extend, q = extend, token-CSR page ids
    (sglang form); the same tensors through the dense form must be bitwise
    identical where the page size allows it.  SGLang runs these at head_dim
    64; 128 is added so the dense-needing backends participate."""
    kv_lens = [p_ + e for p_, e in zip(prefix, extend, strict=True)]
    p = build_problem(
        list(extend),
        kv_lens,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim_qk=head_dim,
        page_size=page,
        dtype=torch.float16,
        kv_layout="NHD",
        input_form="page_indices",
        seed=410 + page + head_dim,
    )
    _, out, lse = plan_and_check(p, backend)
    if page >= 8:
        _, out_b, lse_b = plan_and_check(dict(p, input_form="block_tables"), backend)
        assert torch.equal(out, out_b) and torch.equal(lse, lse_b)


# name, prefix_lens, extend_lens, sliding_window_size
_SGLANG_SWA_CASES = [
    ("swa_extend_no_prefix_window_edges", (0, 0, 0), (3, 4, 5), 4),
    ("swa_extend_prefix_window_edges", (3, 4, 5), (2, 2, 2), 4),
    ("swa_extend_no_prefix_above_window_long", (0, 0, 0), (6, 8, 12), 4),
    ("swa_decode_within_window", (1, 2, 3), (1, 1, 1), 4),
    ("swa_decode_above_window", (7, 8, 9), (1, 1, 1), 4),
    ("dflash_verify_swa_window_edges", (1, 4, 9), (3, 3, 3), 4),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("convention", ["W", "W-1"])
@pytest.mark.parametrize(
    "name,prefix,extend,window",
    _SGLANG_SWA_CASES,
    ids=[c[0] for c in _SGLANG_SWA_CASES],
)
def test_tc04_sglang_swa_cases(backend, convention, name, prefix, extend, window):
    """SGLang SWA cases (window 4: below / at / above the window).  Engines
    disagree on whether the window counts the current token (vLLM passes
    ``sliding_window - 1``, SGLang's FlashInfer decode metadata keeps
    ``window + 1`` keys), so both conversions are run; the oracle defines
    ``window_left`` as a distance (``window_left + 1`` visible positions)."""
    window_left = window if convention == "W" else window - 1
    kv_lens = [p_ + e for p_, e in zip(prefix, extend, strict=True)]
    p = build_problem(
        list(extend),
        kv_lens,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.float16,
        kv_layout="NHD",
        input_form="page_indices",
        seed=430 + window_left,
    )
    _, out, lse = plan_and_check(p, backend, window_left=window_left)
    _, out_b, lse_b = plan_and_check(
        dict(p, input_form="block_tables"), backend, window_left=window_left
    )
    assert torch.equal(out, out_b) and torch.equal(lse, lse_b)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", ["dense", "csr"])
def test_tc04_skewed_batch_100_decode_rows_plus_8_prefill_rows(backend, form):
    """100 x q_len 1 next to 8 x q_len 17 in one call (the mixed decode /
    prefill step a unified API must serve without splitting)."""
    g = torch.Generator().manual_seed(440)
    q_lens = [1] * 100 + [17] * 8
    kv_lens = [int(torch.randint(q, 301, (1,), generator=g)) for q in q_lens]
    p = build_problem(
        q_lens,
        kv_lens,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        input_form="page_indices" if form == "csr" else "block_tables",
        seed=441,
        pool_slack=32,
    )
    plan_and_check(p, backend)


# ---------------------------------------------------------------------------
# TC05 — strided inputs: supported-and-correct or rejected, never wrong
# ---------------------------------------------------------------------------


def _tc05_params():
    rows = []
    for backend in EXPLICIT_BACKENDS:
        for case in ("fused_qkv_slice", "storage_offset", "head_stride_gap"):
            rows.append(pytest.param(backend, case, id=f"{backend}-{case}"))
        rows.append(
            pytest.param(
                backend,
                "kv_independent_strides",
                id=f"{backend}-kv_independent_strides",
            )
        )
        rows.append(
            pytest.param(
                backend,
                "q_last_dim_stride2",
                id=f"{backend}-q_last_dim_stride2",
            )
        )
        rows.append(
            pytest.param(
                backend,
                "page_table_narrow_view",
                id=f"{backend}-page_table_narrow_view",
            )
        )
        rows.append(
            pytest.param(
                backend,
                "k_cache_on_cpu",
                id=f"{backend}-k_cache_on_cpu",
            )
        )
    # backend="auto" must route a narrow table view past trtllm-gen (which
    # declines it with the typed signal) to a backend that walks the table by
    # its strides, and the result must be correct
    rows.append(
        pytest.param("auto", "page_table_narrow_view", id="auto-page_table_narrow_view")
    )
    return rows


def _blocking_spy(attn):
    """Replace the backend's run() so that reaching it is itself the failure
    (an input the controller must reject is never allowed to launch)."""
    active = attn._impl._active

    def block(*a, **k):
        raise AssertionError("input reached the backend instead of being rejected")

    active.run = block


@pytest.mark.parametrize("backend,case", _tc05_params())
def test_tc05_strided_inputs(backend, case):
    """Strided views engines actually produce.  fa2/fa3 declare strided Q
    support, so fused-QKV slices and head-stride gaps MUST run and be correct
    there; other backends may reject (ValueError) but must never return wrong
    numbers.  Storage offsets are legal everywhere.  Inner-dim stride 2, a
    narrow page-table view, independent K/V pool strides and a K cache on the
    wrong device must be rejected-or-correct, at plan or at run."""
    p = build_problem(
        [6, 1, 19],
        [40, 9, 70],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        seed=500,
    )
    _resolve_or_skip(p, backend)
    dev = torch.device(DEVICE)
    total, h, d = p["q"].shape
    g = torch.Generator(device=dev).manual_seed(501)
    required = False  # must-be-correct (vs. reject-or-correct)
    q, k, v, bt = p["q"], p["k_cache"], p["v_cache"], p["block_tables"]
    if case == "fused_qkv_slice":
        fused = torch.randn(total, 3 * h, d, dtype=q.dtype, device=dev, generator=g)
        fused[:, :h] = q
        q = fused[:, :h]
        assert not q.is_contiguous() and q.stride(-1) == 1
        required = backend in ("fa2", "fa3")
    elif case == "storage_offset":
        big = torch.randn(total + 16, h, d, dtype=q.dtype, device=dev, generator=g)
        big[8 : 8 + total] = q
        q = big[8 : 8 + total]
        assert q.is_contiguous() and q.storage_offset() != 0
        required = True
    elif case == "head_stride_gap":
        big = torch.randn(total, 2 * h, d, dtype=q.dtype, device=dev, generator=g)
        big[:, ::2] = q
        q = big[:, ::2]
        assert q.stride(1) == 2 * d and q.stride(-1) == 1
        required = backend in ("fa2", "fa3")
    elif case == "q_last_dim_stride2":
        big = torch.randn(total, h, 2 * d, dtype=q.dtype, device=dev, generator=g)
        big[:, :, ::2] = q
        q = big[:, :, ::2]
        assert q.stride(-1) == 2
    elif case == "kv_independent_strides":
        pool, hk, ps, _ = k.shape
        big_v = torch.randn(pool, hk, 2 * ps, d, dtype=v.dtype, device=dev, generator=g)
        big_v[:, :, :ps] = v
        v = big_v[:, :, :ps]
        assert v.stride() != k.stride() and v.stride(-1) == 1
        # fa2 and cudnn walk each pool by its own strides; the SM90 (fa3)
        # binding takes one stride set for K and V, so fa3 rejects (R4)
        required = backend == "fa2"
    elif case == "page_table_narrow_view":
        b, w = bt.shape
        wide = torch.zeros(b, 2 * w, dtype=torch.int32, device=dev)
        wide[:, :w] = bt
        bt = wide[:, :w]
        assert bt.stride(0) == 2 * w
        required = backend == "auto"
    elif case == "k_cache_on_cpu":
        k = k.cpu()
    else:  # pragma: no cover
        raise AssertionError(case)

    p2 = dict(p, q=q, k_cache=k, v_cache=v, k_ref=k.to(dev), v_ref=v, block_tables=bt)
    try:
        attn = plan(PagedAttention(dev), p2, backend)
    except ValueError as e:  # rejected at plan time (before any state moved)
        assert not required, f"{backend} rejected a supported strided input: {e}"
        assert str(e)
        return
    if case == "k_cache_on_cpu":
        _blocking_spy(attn)
        with pytest.raises(ValueError):
            attn.run(q, (k, v))
        return
    try:
        out, lse = attn.run(q, (k, v))
    except ValueError as e:
        assert not required, f"{backend} rejected a supported strided input: {e}"
        assert str(e)
        return
    torch.cuda.synchronize()
    assert_matches(out, lse, p2, lse_mode="base2")


# ---------------------------------------------------------------------------
# TC06 — shared pages and NaN canaries
# ---------------------------------------------------------------------------


def _rebuild_csr(p, bt):
    pages = (p["kv_seq_lens_cpu"] + p["page_size"] - 1) // p["page_size"]
    bt_cpu = bt.cpu()
    flat = torch.cat(
        [bt_cpu[i, : int(pages[i])] for i in range(bt.shape[0])]
        + [torch.zeros(0, dtype=torch.int32)]
    )
    return dict(p, block_tables=bt, kv_page_indices=flat.to(torch.int32).to(bt.device))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("form", ["dense", "csr"])
def test_tc06_shared_prefix_pages_and_nan_canaries(backend, form):
    """Two requests share their first two physical pages (a radix-cache
    prefix), next to a 3-token and a 200-token request.  Every pool page no
    live row references is filled with NaN, the dense table's dead columns
    and the CSR over-allocated tail point at NaN pages: outputs must be finite
    and match the oracle.  Then the long request is remapped to fresh pages
    and its old pages are poisoned; after a re-plan nothing may read them."""
    page = 16
    p = build_problem(
        [9, 4, 1, 33],
        [70, 45, 3, 200],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=page,
        dtype=torch.bfloat16,
        input_form="page_indices" if form == "csr" else "block_tables",
        seed=600,
        pool_slack=24,
    )
    _resolve_or_skip(p, backend)
    dev = torch.device(DEVICE)
    bt = p["block_tables"].clone()
    bt[1, :2] = bt[0, :2]  # request 1 shares request 0's two prefix pages
    pages = (p["kv_seq_lens_cpu"] + page - 1) // page

    def poison_unreferenced(p, bt):
        live = set()
        for i in range(bt.shape[0]):
            live |= set(bt[i, : int(pages[i])].tolist())
        dead = [i for i in range(p["k_cache"].shape[0]) if i not in live]
        assert len(dead) >= 14
        k, v = p["k_cache"].clone(), p["v_cache"].clone()
        k[dead] = float("nan")
        v[dead] = float("nan")
        bt = bt.clone()
        for i in range(bt.shape[0]):
            bt[i, int(pages[i]) :] = dead[0]  # dead dense columns -> a NaN page
        q = _rebuild_csr(dict(p, k_cache=k, v_cache=v, k_ref=k, v_ref=v), bt)
        if form == "csr":
            tail = torch.full((64,), dead[0], dtype=torch.int32, device=dev)
            q["kv_page_indices"] = torch.cat([q["kv_page_indices"], tail])
        return q, dead

    p1, dead = poison_unreferenced(p, bt)
    attn = plan(PagedAttention(dev), p1, backend)
    out, lse = run(attn, p1)
    assert_matches(out, lse, p1, lse_mode="base2")

    # remap request 3 (13 pages) onto fresh pages, then poison its old pages
    n3 = int(pages[3])
    old = bt[3, :n3].clone()
    new = torch.tensor(dead[1 : 1 + n3], dtype=torch.int32, device=dev)
    k2, v2 = p1["k_cache"].clone(), p1["v_cache"].clone()
    k2[new.long()] = p["k_cache"][old.long()]
    v2[new.long()] = p["v_cache"][old.long()]
    k2[old.long()] = float("nan")
    v2[old.long()] = float("nan")
    bt2 = p1["block_tables"].clone()
    bt2[3, :n3] = new
    p2 = _rebuild_csr(dict(p1, k_cache=k2, v_cache=v2, k_ref=k2, v_ref=v2), bt2)
    if form == "csr":
        tail = torch.full((64,), dead[0], dtype=torch.int32, device=dev)
        p2["kv_page_indices"] = torch.cat([p2["kv_page_indices"], tail])
    plan(attn, p2, backend)
    out2, lse2 = run(attn, p2)
    assert_matches(out2, lse2, p2, lse_mode="base2")
    # rows 0..2 are untouched by the remap: bitwise identical to round one
    n_keep = int(p["qo_indptr_cpu"][3])
    assert torch.equal(out[:n_keep], out2[:n_keep])
    assert torch.equal(lse[:n_keep], lse2[:n_keep])


# ---------------------------------------------------------------------------
# TC07 — CUDA-graph replay across features
# ---------------------------------------------------------------------------

_TC07_FEATURES = {
    "csr_page1": dict(page_size=1, input_form="page_indices"),
    "nhd": dict(kv_layout="NHD"),
    "swa16": dict(window_left=16),
    "fp16": dict(dtype=torch.float16),
    "fp8_kv": dict(kv_dtype=FP8),
    "basee": dict(lse_mode="basee"),
    "lse_none": dict(lse_mode="none"),
}


def sibling_problem(p, seed):
    """Same capture shapes (batch, total q tokens, per-request page budget,
    host maxes), different request order, page permutation and contents."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device(p["device"])
    b = p["kv_seq_lens_cpu"].shape[0]
    perm_req = torch.randperm(b, generator=g)
    q_lens = p["qo_indptr_cpu"].diff()[perm_req]
    kv_lens = p["kv_seq_lens_cpu"][perm_req].clone()
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    page = p["page_size"]
    pages = (kv_lens + page - 1) // page
    width = p["block_tables"].shape[1]
    pool = p["k_cache"].shape[0]
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    bt = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages[i])
        bt[i, :n] = perm[off : off + n]
        off += n
    flat = torch.cat([bt[i, : int(pages[i])] for i in range(b)]).to(torch.int32)
    g_cuda = torch.Generator(device=dev).manual_seed(seed)
    q = torch.randn(p["q"].shape, dtype=p["dtype"], device=dev, generator=g_cuda)
    k = torch.randn(p["k_ref"].shape, dtype=p["dtype"], device=dev, generator=g_cuda)
    v = torch.randn(p["v_ref"].shape, dtype=p["dtype"], device=dev, generator=g_cuda)
    k_ref, v_ref = k, v
    if p["k_scale"] is not None:  # fp8 KV: same per-tensor scales as the capture
        k = (k.float() / p["k_scale"]).clamp_(-448.0, 448.0).to(p["k_cache"].dtype)
        v = (v.float() / p["v_scale"]).clamp_(-448.0, 448.0).to(p["v_cache"].dtype)
        k_ref, v_ref = k.float() * p["k_scale"], v.float() * p["v_scale"]
    return dict(
        p,
        q=q,
        k_cache=k,
        v_cache=v,
        k_ref=k_ref,
        v_ref=v_ref,
        qo_indptr=qo_indptr_cpu.to(dev),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(dev),
        kv_seq_lens_cpu=kv_lens,
        block_tables=bt.to(dev),
        kv_page_indices=flat.to(dev),
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("feature", list(_TC07_FEATURES))
def test_tc07_graph_cross_feature_replay(backend, feature):
    """Under the fixed-capacity contract: plan -> capture -> re-plan a sibling
    batch -> replay -> re-plan the original -> replay, for CSR page 1, NHD,
    SWA, fp16, fa2 fp8 KV, basee and none.  Each round's out (and LSE) is
    checked against the oracle and the caller's static buffers are the ones
    written."""
    f = _TC07_FEATURES[feature]
    lse_mode = f.get("lse_mode", "base2")
    window_left = f.get("window_left", -1)
    p1 = build_problem(
        [8, 1, 20, 3],
        [64, 30, 100, 48],
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=f.get("page_size", 16),
        dtype=f.get("dtype", torch.bfloat16),
        kv_layout=f.get("kv_layout", "HND"),
        input_form=f.get("input_form", "block_tables"),
        kv_dtype=f.get("kv_dtype"),
        seed=700,
    )
    _resolve_or_skip(p1, backend, need_lse=lse_mode != "none", window_left=window_left)
    dev = torch.device(DEVICE)
    attn = PagedAttention(dev, use_cuda_graph=True)
    plan(attn, p1, backend, window_left=window_left, lse_mode=lse_mode)
    q = p1["q"].clone()
    k = p1["k_cache"].clone()
    v = p1["v_cache"].clone()
    out = torch.empty(q.shape[0], 8, 128, dtype=p1["dtype"], device=dev)
    lse = torch.empty(q.shape[0], 8, dtype=torch.float32, device=dev)
    kw = dict(out=out, k_scale=p1["k_scale"], v_scale=p1["v_scale"])
    if lse_mode != "none":
        kw["lse"] = lse

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            attn.run(q, (k, v), **kw)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        r_out, r_lse = attn.run(q, (k, v), **kw)
    assert r_out.data_ptr() == out.data_ptr()
    if lse_mode != "none":
        assert r_lse.data_ptr() == lse.data_ptr()
    else:
        assert r_lse is None

    p2 = sibling_problem(p1, seed=701)
    for round_, p in enumerate((p1, p2, p1)):
        if round_ > 0:
            q.copy_(p["q"])
            k.copy_(p["k_cache"])
            v.copy_(p["v_cache"])
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                plan(attn, p, backend, window_left=window_left, lse_mode=lse_mode)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        g.replay()
        torch.cuda.synchronize()
        assert_matches(
            out,
            lse if lse_mode != "none" else None,
            p,
            window_left=window_left,
            lse_mode=lse_mode,
        )


# ---------------------------------------------------------------------------
# TC08 — capability required rows (manifest, not filtered by resolve)
# ---------------------------------------------------------------------------

_ALL_FA2_ARCHES = frozenset({8, 9, 10, 12})
_ALL_CUDNN_ARCHES = frozenset({8, 9, 10, 12})


def _row(backend, arch, hq, hk, dqk, dvo, dtype, form, page, **over):
    r = dict(
        backend=backend,
        arch=frozenset(arch),
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=dqk,
        head_dim_vo=dvo,
        dtype=dtype,
        kv_dtype=None,
        kv_layout="HND",
        form=form,
        page_size=page,
        causal=True,
        window_left=-1,
        lse_mode="base2",
        note="",
    )
    r.update(over)
    return r


BF16, F16 = torch.bfloat16, torch.float16

# Rows that MUST run on the named architectures.  Hardware / dependency
# unavailability may skip; a capability-table rejection FAILS.
REQUIRED_ROWS = [
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 64, 64, BF16, "dense", 8),
    _row("fa2", _ALL_FA2_ARCHES, 8, 8, 128, 128, F16, "csr", 5, kv_layout="NHD"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 1, 256, 256, BF16, "dense", 16),
    # head_dim 512 (Gemma-4 full attention): measured on B200 (2026-09-17)
    _row("fa2", _ALL_FA2_ARCHES, 16, 2, 512, 512, BF16, "dense", 16),
    _row(
        "fa2",
        _ALL_FA2_ARCHES,
        16,
        2,
        512,
        512,
        BF16,
        "csr",
        16,
        kv_dtype=FP8,
        kv_layout="NHD",
        note="Gemma-4 fp8 KV layout",
    ),
    _row("fa2", _ALL_FA2_ARCHES, 10, 2, 128, 128, BF16, "csr", 5, note="GQA ratio 5"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "csr", 1, kv_layout="NHD"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 1, 128, 128, F16, "dense", 8, note="MQA Hkv=1"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, kv_dtype=FP8),
    # e5m2 KV: measured on B200 (2026-09-17), see _capabilities.py
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, kv_dtype=E5M2),
    _row(
        "fa2",
        _ALL_FA2_ARCHES,
        8,
        2,
        128,
        128,
        F16,
        "csr",
        1,
        kv_dtype=E5M2,
        kv_layout="NHD",
    ),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, window_left=32),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, causal=False),
    _row("fa2", _ALL_FA2_ARCHES, 4, 4, 64, 64, F16, "csr", 1, kv_layout="NHD"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, lse_mode="none"),
    _row("fa2", _ALL_FA2_ARCHES, 8, 2, 128, 128, F16, "dense", 32, lse_mode="basee"),
    _row("fa3", {9}, 8, 2, 128, 128, BF16, "dense", 16),
    _row("fa3", {9}, 8, 2, 64, 64, F16, "csr", 1, kv_layout="NHD"),
    _row("fa3", {9}, 8, 1, 256, 256, BF16, "dense", 16),
    _row("fa3", {9}, 10, 2, 128, 128, BF16, "csr", 5, note="GQA ratio 5"),
    _row("cudnn", _ALL_CUDNN_ARCHES, 8, 2, 128, 128, BF16, "dense", 16),
    _row("cudnn", _ALL_CUDNN_ARCHES, 8, 2, 192, 128, F16, "dense", 16),
    _row(
        "cudnn", _ALL_CUDNN_ARCHES, 8, 2, 128, 128, BF16, "dense", 32, kv_layout="NHD"
    ),
    _row(
        "cudnn", _ALL_CUDNN_ARCHES, 8, 2, 128, 128, F16, "csr", 8, note="derived dense"
    ),
    _row("cudnn", _ALL_CUDNN_ARCHES, 10, 2, 128, 128, BF16, "dense", 16, note="GQA 5"),
    _row("cudnn", _ALL_CUDNN_ARCHES, 8, 1, 128, 128, BF16, "dense", 16, note="MQA"),
    _row("cudnn", _ALL_CUDNN_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, causal=False),
    _row(
        "cudnn", _ALL_CUDNN_ARCHES, 8, 2, 128, 128, BF16, "dense", 16, lse_mode="basee"
    ),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, BF16, "dense", 16),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, F16, "dense", 32, kv_layout="NHD"),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, BF16, "dense", 64),
    _row(
        "trtllm-gen",
        {10},
        8,
        1,
        128,
        128,
        BF16,
        "dense",
        16,
        note="MQA Hkv=1 (issue 2232)",
    ),
    _row("trtllm-gen", {10}, 8, 1, 128, 128, F16, "dense", 32, note="MQA Hkv=1 fp16"),
    _row("trtllm-gen", {10}, 10, 2, 128, 128, BF16, "dense", 16, note="GQA ratio 5"),
    _row("trtllm-gen", {10}, 64, 8, 128, 128, BF16, "dense", 16, note="GQA ratio 8"),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, BF16, "csr", 16, note="derived dense"),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, BF16, "dense", 16, window_left=127),
    _row("trtllm-gen", {10}, 8, 2, 128, 128, BF16, "dense", 16, lse_mode="none"),
]


def _row_id(r):
    bits = [
        r["backend"],
        f"h{r['num_qo_heads']}-{r['num_kv_heads']}",
        f"d{r['head_dim_qk']}-{r['head_dim_vo']}",
        str(r["dtype"]).replace("torch.", ""),
        r["kv_layout"],
        f"{r['form']}{r['page_size']}",
    ]
    if r["kv_dtype"] is not None:
        bits.append(str(r["kv_dtype"]).replace("torch.float8_", "") + "kv")
    if not r["causal"]:
        bits.append("noncausal")
    if r["window_left"] >= 0:
        bits.append(f"w{r['window_left']}")
    if r["lse_mode"] != "base2":
        bits.append(r["lse_mode"])
    return "-".join(bits)


@pytest.mark.parametrize("row", REQUIRED_ROWS, ids=[_row_id(r) for r in REQUIRED_ROWS])
def test_tc08_capability_required_rows(row):
    """The manifest of rows the current GPU family MUST run.  The backend's
    own resolve() is not allowed to filter them: a ValueError from a
    capability rule is a regression and fails; only environment probes
    (missing cudnn-frontend, fa3 on non-Hopper) skip."""
    cc = _cc_major()
    if cc not in row["arch"]:
        pytest.skip(f"row not required on sm_{cc}x")
    page = row["page_size"]
    p = build_problem(
        [5, 1, 12, 2],
        [max(3 * page + 1, 5), 9, max(2 * page, 12), 2],
        num_qo_heads=row["num_qo_heads"],
        num_kv_heads=row["num_kv_heads"],
        head_dim_qk=row["head_dim_qk"],
        head_dim_vo=row["head_dim_vo"],
        page_size=page,
        dtype=row["dtype"],
        kv_layout=row["kv_layout"],
        input_form="page_indices" if row["form"] == "csr" else "block_tables",
        kv_dtype=row["kv_dtype"],
        seed=800 + page,
    )
    need_lse = row["lse_mode"] != "none"
    try:
        res = resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=row["num_qo_heads"],
            num_kv_heads=row["num_kv_heads"],
            head_dim_qk=row["head_dim_qk"],
            head_dim_vo=row["head_dim_vo"],
            q_dtype=row["dtype"],
            kv_dtype=row["kv_dtype"],
            page_size=page,
            kv_layout=row["kv_layout"],
            causal=row["causal"],
            need_lse=need_lse,
            window_left=row["window_left"],
            kv_input_form=p["input_form"],
            backend=row["backend"],
        )
    except ValueError as e:
        if any(word in str(e) for word in _ENVIRONMENT_REASONS):
            pytest.skip(f"environment: {e}")
        pytest.fail(f"capability regression: required row rejected by resolve(): {e}")
    assert res.backends == (row["backend"],)
    attn = plan(
        PagedAttention(torch.device(DEVICE)),
        p,
        res,
        causal=row["causal"],
        window_left=row["window_left"],
        lse_mode=row["lse_mode"],
    )
    assert attn.backend == row["backend"]
    out, lse = run(attn, p)
    assert_matches(
        out,
        lse,
        p,
        causal=row["causal"],
        window_left=row["window_left"],
        lse_mode=row["lse_mode"],
    )


# ---------------------------------------------------------------------------
# TC11 — several instances on the shared workspace, each against its oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_tc11_multi_instance_shared_workspace_oracle(backend):
    """Two graph-bucket instances and one eager instance, planned on different
    problems and run alternately on the shared per-device scratch workspace:
    every result matches its OWN oracle, so sharing the workspace leaks no
    plan state between instances (``test_paged_attention_workspace.py``
    checks the memory, this checks the numbers)."""
    dev = torch.device(DEVICE)
    common = dict(num_qo_heads=8, num_kv_heads=2, head_dim_qk=128, page_size=16)
    probs = [
        build_problem(
            [8, 1, 20, 3], [64, 30, 100, 48], dtype=torch.bfloat16, seed=1100, **common
        ),
        build_problem(
            [1] * 6,
            [17, 200, 33, 64, 9, 120],
            dtype=torch.bfloat16,
            seed=1101,
            **common,
        ),
        build_problem([33, 5], [300, 7], dtype=torch.bfloat16, seed=1102, **common),
    ]
    for p in probs:
        _resolve_or_skip(p, backend)
    insts = [
        PagedAttention(dev, use_cuda_graph=True),
        PagedAttention(dev, use_cuda_graph=True),
        PagedAttention(dev),
    ]
    for attn, p in zip(insts, probs, strict=True):
        plan(attn, p, backend)
    refs = [reference(p) for p in probs]
    for _ in range(2):
        for attn, p, (ref_out, ref_lse) in zip(insts, probs, refs, strict=True):
            out, lse = run(attn, p)
            torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
            torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    # and they really do share one scratch buffer (the point of the exercise)
    assert len({a._impl._workspace.data_ptr() for a in insts}) == 1
