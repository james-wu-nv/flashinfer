"""Query and page-table stride contract for the unified paged-prefill API.

Two layers, one oracle:

1. Native probe (facts).  For each backend's *native* entry point (the legacy
   ``BatchPrefillWithPagedKVCacheWrapper``, ``cudnn_batch_prefill_with_kv_cache``,
   ``trtllm_batch_context_with_kv_cache``) run the SAME query values through
   different storage layouts and classify the outcome against the fp32 oracle:
   ``pass`` (correct), ``wrong`` (accepted, silently wrong numbers) or
   ``reject`` (the native raised).  ``NATIVE_Q_OUTCOME`` is the measured table
   (B200 / SM100, see reports/unified-prefill-implementation-20260916/wp-b.md);
   the controller's query ABI checks are derived from it.  A ``wrong`` entry
   that starts passing means a binding changed — relax the matching check.

2. Unified contract.  The controller and backends must reject exactly what the
   natives misread, and accept (correctly) what they handle: no ``.contiguous()``
   copy hides a layout the kernel cannot address.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import make_problem

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

# fa3 is not in the native probe: no SM90 in the verification pool (the
# capability-honesty rule forbids encoding an unmeasured outcome).
NATIVE_BACKENDS = ["fa2", "cudnn", "trtllm-gen"]


def _problem(seed=101, **overrides):
    kw = dict(
        batch_size=3,
        max_q=32,
        max_kv=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    kw.update(overrides)
    return make_problem(seed=seed, **kw)


def _skip_unless_runnable(p, backend):
    try:
        resolve_paged_attention(
            device=torch.device(p["device"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            q_dtype=p["dtype"],
            page_size=p["page_size"],
            causal=True,
            need_lse=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"backend {backend} not runnable here: {e}")


def _oracle(p, q=None, **kw):
    return reference_paged_prefill(
        p["q"] if q is None else q,
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
        **kw,
    )


# ---------------------------------------------------------------------------
# Query storage layouts.  Each builder returns a (T, H, D) view holding the
# SAME values as p["q"], so the oracle answer is unchanged.
# ---------------------------------------------------------------------------


def q_contiguous(p):
    return p["q"]


def q_fused_qkv_head_slice(p):
    """q = qkv[:, :Hq] of a fused (T, Hq+2Hkv, D) projection buffer:
    inner stride 1, head stride D, token stride (Hq+2Hkv)*D."""
    q = p["q"]
    t, h, d = q.shape
    fused = torch.randn(t, h + 2 * p["num_kv_heads"], d, dtype=q.dtype, device=q.device)
    fused[:, :h] = q
    return fused[:, :h]


def q_fused_qkv_mid_slice(p):
    """q in the MIDDLE of a fused buffer: strided AND a non-zero storage offset."""
    q = p["q"]
    t, h, d = q.shape
    hk = p["num_kv_heads"]
    fused = torch.randn(t, h + 2 * hk, d, dtype=q.dtype, device=q.device)
    fused[:, hk : hk + h] = q
    return fused[:, hk : hk + h]


def q_inner_stride_2(p):
    """buf[..., ::2] of a (T, H, 2D) buffer: head_dim interleaved (stride 2)."""
    q = p["q"]
    t, h, d = q.shape
    buf = torch.randn(t, h, 2 * d, dtype=q.dtype, device=q.device)
    buf[..., ::2] = q
    return buf[..., ::2]


def q_head_stride_padded(p):
    """buf[..., :D] of a (T, H, D+8) buffer: inner stride 1, head stride D+8."""
    q = p["q"]
    t, h, d = q.shape
    buf = torch.randn(t, h, d + 8, dtype=q.dtype, device=q.device)
    buf[..., :d] = q
    return buf[..., :d]


def q_storage_offset(p):
    """buf[1:] of a (T+1, H, D) buffer: contiguous, storage_offset = H*D."""
    q = p["q"]
    buf = torch.randn(q.shape[0] + 1, *q.shape[1:], dtype=q.dtype, device=q.device)
    buf[1:] = q
    return buf[1:]


Q_LAYOUTS = {
    "contiguous": q_contiguous,
    "fused_qkv_head_slice": q_fused_qkv_head_slice,
    "fused_qkv_mid_slice": q_fused_qkv_mid_slice,
    "inner_stride_2": q_inner_stride_2,
    "head_stride_padded": q_head_stride_padded,
    "storage_offset": q_storage_offset,
}

# Measured native outcomes (B200 / SM100, bf16, GQA 8/2, D=128, page 16).
#   fa2 and trtllm-gen pass token and head strides to the kernel and assume a
#   dense head_dim: any unit-inner-stride view is correct, inner stride 2 is
#   silently wrong (max abs error ~1.5 / ~1.3 against ~0.0025).
#   cuDNN builds its graph from q.stride() but scales the token-unit ragged
#   offsets by Hq*D: only packed THD (any storage offset) is correct; every
#   other layout is silently wrong except inner stride 2, which it rejects.
NATIVE_Q_OUTCOME = {
    "fa2": {
        "contiguous": "pass",
        "fused_qkv_head_slice": "pass",
        "fused_qkv_mid_slice": "pass",
        "inner_stride_2": "wrong",
        "head_stride_padded": "pass",
        "storage_offset": "pass",
    },
    "cudnn": {
        "contiguous": "pass",
        "fused_qkv_head_slice": "wrong",
        "fused_qkv_mid_slice": "wrong",
        "inner_stride_2": "reject",
        "head_stride_padded": "wrong",
        "storage_offset": "pass",
    },
    "trtllm-gen": {
        "contiguous": "pass",
        "fused_qkv_head_slice": "pass",
        "fused_qkv_mid_slice": "pass",
        "inner_stride_2": "wrong",
        "head_stride_padded": "pass",
        "storage_offset": "pass",
    },
}


@pytest.fixture(scope="module")
def workspace():
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")


def _native_runner(backend, p, workspace):
    """The backend's native call, closed over the problem's metadata, as a
    ``q -> (out, packed_lse)`` function (mirrors what the backend modules do)."""
    dev = p["q"].device
    page = p["page_size"]
    hq, hk, d = p["num_qo_heads"], p["num_kv_heads"], p["head_dim_qk"]
    b = int(p["kv_seq_lens_cpu"].shape[0])
    sm_scale = 1.0 / d**0.5
    if backend == "fa2":
        from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

        w = BatchPrefillWithPagedKVCacheWrapper(workspace, "HND", backend="fa2")
        kv_lens = p["kv_seq_lens_cpu"].to(torch.int32)
        pages = (kv_lens + page - 1) // page
        kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(pages, 0, dtype=torch.int32),
            ]
        )
        last = ((kv_lens - 1) % page + 1).to(torch.int32)
        w.plan(
            p["qo_indptr_cpu"].to(torch.int32),
            kv_indptr,
            p["kv_page_indices"],
            last,
            hq,
            hk,
            d,
            page,
            causal=True,
            q_data_type=p["dtype"],
            kv_data_type=p["dtype"],
        )
        return lambda q: w.run(q, (p["k_cache"], p["v_cache"]), return_lse=True)
    if backend == "cudnn":
        from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache

        tok = torch.arange(p["q"].shape[0], device=dev)
        bid = torch.searchsorted(p["qo_indptr"][1:].long(), tok, right=True)
        pos = tok - p["qo_indptr"].long()[bid]

        def run(q):
            out, lse = cudnn_batch_prefill_with_kv_cache(
                q,
                p["k_cache"],
                p["v_cache"],
                sm_scale,
                workspace.view(torch.int8),
                max_token_per_sequence=p["max_q_len"],
                max_sequence_kv=p["max_kv_len"],
                actual_seq_lens_q=p["qo_indptr"].diff().view(b, 1, 1, 1),
                actual_seq_lens_kv=p["kv_seq_lens"].view(b, 1, 1, 1),
                block_tables=p["block_tables"],
                causal=True,
                return_lse=True,
                lse_base="2",
                batch_offsets_q=p["qo_indptr"],
                batch_offsets_units="tokens",
            )
            return out, lse[bid, pos, :]  # padded (b, max_q, h) -> packed

        return run
    if backend == "trtllm-gen":
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        cum_kv = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=dev),
                torch.cumsum(p["kv_seq_lens"], 0, dtype=torch.int32),
            ]
        )
        return lambda q: trtllm_batch_context_with_kv_cache(
            q,
            (p["k_cache"], p["v_cache"]),
            workspace,
            p["block_tables"],
            p["kv_seq_lens"],
            p["max_q_len"],
            p["max_kv_len"],
            sm_scale,
            1.0,
            b,
            p["qo_indptr"],
            cum_kv,
            window_left=-1,
            kv_layout="HND",
            causal=True,
            return_lse=True,
        )
    raise AssertionError(backend)


@pytest.mark.parametrize("layout", list(Q_LAYOUTS))
@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_native_query_stride_probe(backend, layout, workspace):
    """Pin the measured native behaviour per query layout (see NATIVE_Q_OUTCOME).

    ``wrong`` asserts the native REALLY returns wrong numbers: if this starts
    failing, the binding learned the layout and the unified contract can be
    relaxed for that backend."""
    p = _problem()
    _skip_unless_runnable(p, backend)
    q = Q_LAYOUTS[layout](p)
    assert torch.equal(q, p["q"])  # same values, different storage
    run = _native_runner(backend, p, workspace)
    expected = NATIVE_Q_OUTCOME[backend][layout]
    if expected == "reject":
        # cuDNN's graph builder: "stride for the last dimension ... should be 1"
        with pytest.raises(Exception, match="stride"):
            out, _ = run(q)
            torch.cuda.synchronize()
        return
    out, lse = run(q)
    torch.cuda.synchronize()
    ref_out, ref_lse = _oracle(p)
    out_ok = torch.allclose(out.float(), ref_out, **OUT_TOL)
    lse_ok = torch.allclose(lse.float(), ref_lse, **LSE_TOL)
    if expected == "pass":
        assert out_ok and lse_ok, (
            f"{backend} native misread q layout {layout!r} strides {tuple(q.stride())}"
        )
    else:
        assert not (out_ok and lse_ok), (
            f"{backend} native now handles q layout {layout!r} (strides "
            f"{tuple(q.stride())}) correctly — update NATIVE_Q_OUTCOME and relax "
            "the matching controller/capability check"
        )


# ---------------------------------------------------------------------------
# Unified contract: what the controller/backends must reject or handle.
# ---------------------------------------------------------------------------

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


def _plan(p, backend, **plan_kw):
    from flashinfer.prefill import PagedAttention

    from .test_paged_attention_prototype import make_metadata

    _skip_unless_runnable(p, backend)
    kw = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        kv_layout=p.get("kv_layout", "HND"),
        lse_mode="base2",
        backend=backend,
    )
    kw.update(plan_kw)
    attn = PagedAttention(torch.device(p["device"]))
    attn.plan(make_metadata(p), **kw)
    return attn


def _check_unified(attn, p, q):
    out, lse = attn.run(q, (p["k_cache"], p["v_cache"]))
    ref_out, ref_lse = _oracle(p)
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    return out, lse


@pytest.mark.parametrize("backend", BACKENDS)
def test_run_rejects_non_unit_inner_stride(backend):
    """q.stride(-1) != 1 is a silent misread on fa2/trtllm-gen (native probe)
    and a graph-build error on cuDNN: the controller rejects it for EVERY
    backend before any launch, as a ValueError naming the constraint."""
    p = _problem()
    attn = _plan(p, backend)
    with pytest.raises(ValueError, match=r"stride\(-1\) == 1"):
        attn.run(q_inner_stride_2(p), (p["k_cache"], p["v_cache"]))
    # the same values in an addressable layout still run
    _check_unified(attn, p, p["q"])


@pytest.mark.parametrize("which", ["q", "k_cache", "v_cache", "out", "lse"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_run_rejects_foreign_device(backend, which):
    """Every run() tensor must live on the instance's device: backends take raw
    pointers, so a CPU (or other-GPU) tensor is a fault or a misread."""
    p = _problem()
    attn = _plan(p, backend)
    q, k, v = p["q"], p["k_cache"], p["v_cache"]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    tensors = dict(q=q, k_cache=k, v_cache=v, out=out, lse=lse)
    tensors[which] = tensors[which].cpu()
    with pytest.raises(ValueError, match=f"{which} lives on cpu"):
        attn.run(
            tensors["q"],
            (tensors["k_cache"], tensors["v_cache"]),
            out=tensors["out"],
            lse=tensors["lse"],
        )


@pytest.mark.parametrize(
    "layout",
    [
        "fused_qkv_head_slice",
        "fused_qkv_mid_slice",
        "head_stride_padded",
        "storage_offset",
    ],
)
@pytest.mark.parametrize("backend", BACKENDS)
def test_unified_query_layouts(backend, layout):
    """Unit-inner-stride views: backends whose capability says any such view
    is addressable must match the oracle on it (no hidden copy); a backend
    that needs packed q must reject, not misread (cuDNN, fused-QKV slices)."""
    from flashinfer.experimental.paged_attention import CAPABILITIES

    p = _problem()
    attn = _plan(p, backend)
    q = Q_LAYOUTS[layout](p)
    assert q.stride(-1) == 1 and torch.equal(q, p["q"])
    if CAPABILITIES[backend].requires_contiguous_q and not q.is_contiguous():
        with pytest.raises(ValueError, match="requires packed q"):
            attn.run(q, (p["k_cache"], p["v_cache"]))
        return
    _check_unified(attn, p, q)


def test_trtllm_fused_qkv_engine_shape():
    """The vLLM-style shape the sibling probe measured (32/8 heads, fused QKV):
    trtllm-gen through the unified API on the strided slice, no copy."""
    p = _problem(
        seed=7, batch_size=4, max_q=64, max_kv=512, num_qo_heads=32, num_kv_heads=8
    )
    attn = _plan(p, "trtllm-gen")
    q = q_fused_qkv_head_slice(p)
    assert tuple(q.stride()) == (48 * 128, 128, 1)
    _check_unified(attn, p, q)


# ---------------------------------------------------------------------------
# Page-table strides (dense backends) and flat page-id storage (FA).
# ---------------------------------------------------------------------------


def _with_capacity_table(p, extra_cols=3):
    """Engine-style table: wider than ceil(max_kv/page) by ``extra_cols``,
    the extra columns holding valid (unused) pool page ids."""
    p = dict(p)
    bt = p["block_tables"]
    b, w = bt.shape
    pool = p["k_cache"].shape[0]
    wide = torch.zeros(b, w + extra_cols, dtype=bt.dtype, device=bt.device)
    wide[:, :w] = bt
    wide[:, w:] = torch.arange(extra_cols, device=bt.device, dtype=bt.dtype) % pool
    p["block_tables"] = wide
    return p, w


DENSE_BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_capacity_width_block_table(backend):
    """A contiguous table wider than ceil(max_kv_len/page) with the ACTUAL max
    (vLLM hands over its capacity-width table) runs everywhere: cuDNN takes
    the width-exact view internally (it demands width == ceil(max_kv/page)),
    trtllm-gen and the FA path ignore the extra columns."""
    p, _ = _with_capacity_table(_problem(seed=103))
    attn = _plan(p, backend)
    _check_unified(attn, p, p["q"])


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_block_table_row_stride_view(backend, batch_size):
    """block_tables[:, :w] of a wider table (row stride > width): trtllm-gen
    walks the table as a packed array and must REJECT at plan (the sibling
    probe measured 31.9% wrong elements); cuDNN's stride-driven graph and the
    FA derivation address it correctly.  With one row the view IS contiguous
    (a size-1 dim has no effective stride) and every backend accepts it."""
    p, w = _with_capacity_table(_problem(seed=104, batch_size=batch_size))
    view = p["block_tables"][:, :w]
    assert view.stride(0) == w + 3 and view.is_contiguous() == (batch_size == 1)
    p["block_tables"] = view
    if backend == "trtllm-gen" and not view.is_contiguous():
        _skip_unless_runnable(p, backend)
        with pytest.raises(ValueError, match=r"contiguous.*block_tables"):
            _plan(p, backend)
        # the hinted fix is accepted and correct
        p["block_tables"] = view.contiguous()
        _check_unified(_plan(p, backend), p, p["q"])
        return
    attn = _plan(p, backend)
    _check_unified(attn, p, p["q"])


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_csr_noncontiguous_page_indices(backend):
    """A strided view as the flat kv_page_indices: the FA kernels walk the
    list as a raw pointer and must reject at plan; dense-needing backends
    derive their table by a gather and stay correct."""
    from flashinfer.experimental.paged_attention import CAPABILITIES

    p = dict(_problem(seed=105), input_form="page_indices")
    live = p["kv_page_indices"]
    inter = torch.stack([live, torch.full_like(live, -7)], dim=1).flatten()
    p["kv_page_indices"] = inter[::2]
    assert p["kv_page_indices"].stride(0) == 2 and torch.equal(
        p["kv_page_indices"], live
    )
    if not CAPABILITIES[backend].needs_dense:
        _skip_unless_runnable(p, backend)
        with pytest.raises(ValueError, match="contiguous kv_page_indices"):
            _plan(p, backend)
        return
    attn = _plan(p, backend)
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        None,
        p["page_size"],
        True,
        kv_page_indices=live,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", DENSE_BACKENDS)
def test_capacity_width_block_table_graph_mode(backend):
    """Graph mode with an engine-style capacity-width table: the backends see
    the reserved (b, capacity) buffer, cuDNN's width-exact view of it is as
    pointer-stable as the buffer, and a re-plan into the same storage keeps
    the captured graph correct."""
    from flashinfer.prefill import PagedAttention

    from .test_paged_attention_prototype import make_metadata

    p, _ = _with_capacity_table(_problem(seed=106))
    _skip_unless_runnable(p, backend)
    dev = torch.device(p["device"])
    attn = PagedAttention(dev, use_cuda_graph=True)
    plan_kw = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        lse_mode="base2",
        backend=backend,
    )
    attn.plan(make_metadata(p), **plan_kw)
    q, k, v = p["q"].clone(), p["k_cache"].clone(), p["v_cache"].clone()
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=dev)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            attn.run(q, (k, v), out=out, lse=lse)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        attn.run(q, (k, v), out=out, lse=lse)
    ref_out, ref_lse = _oracle(p)
    for _ in range(2):  # replay, re-plan into the reserved storage, replay
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
        attn.plan(make_metadata(p), **plan_kw)


@pytest.mark.parametrize("order", ["view_then_contiguous", "contiguous_then_view"])
def test_cudnn_graph_cache_keys_block_table_strides(order):
    """Regression: the cuDNN prefill graph cache keyed on q/k/v strides but not
    on the page table's, while the graph bakes the table's strides in.  A
    graph built for block_tables[:, :w] of a wider table (row stride w+3)
    replayed on a contiguous (b, w) table of the same shape read the wrong
    pages (71.7% wrong elements) — in one process, in either order."""
    p, w = _with_capacity_table(_problem(seed=107))
    view = p["block_tables"][:, :w]
    packed = view.contiguous()
    first, second = (
        (view, packed) if order == "view_then_contiguous" else (packed, view)
    )
    for table in (first, second):
        q = dict(p, block_tables=table)
        _check_unified(_plan(q, "cudnn"), q, q["q"])


def test_fa3_rejects_independent_kv_strides_before_launch():
    """The SM90 binding asserts equal K/V page and token strides; the fa3
    backend refuses such a pool with a ValueError before any launch (R4).
    Exercised on the backend object directly so the check is verified on
    hardware without an SM90 device."""
    from flashinfer.experimental.paged_attention._backends.fa_backend import _FaBackend

    be = object.__new__(_FaBackend)
    be.name = "fa3"
    be._lse_mode = "none"
    be._total_q_tokens = 4
    be._use_sinks = False
    k = torch.zeros(3, 2, 16, 128, dtype=torch.bfloat16, device="cuda:0")
    big_v = torch.zeros(3, 2, 32, 128, dtype=torch.bfloat16, device="cuda:0")
    v = big_v[:, :, :16]
    q = torch.zeros(4, 8, 128, dtype=torch.bfloat16, device="cuda:0")
    with pytest.raises(ValueError, match="fa3 requires k_cache and v_cache"):
        be.run(q, k, v, sm_scale=1.0)
    be.name = "fa2"
    be._active = None  # fa2 would proceed to the wrapper: not reached here
    with pytest.raises(AttributeError):
        be.run(q, k, v, sm_scale=1.0)
