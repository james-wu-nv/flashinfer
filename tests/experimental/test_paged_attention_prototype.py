"""Conformance matrix for the unified paged-prefill prototype.

One backend-parametrized test, one reference oracle, one output contract:
- outputs match the fp32 paged-attention oracle
- LSE is ALWAYS base-2, packed (total_q_tokens, num_qo_heads), fp32 —
  including for cuDNN, whose native natural-log padded stats are normalized
  in its adapter.  This is the first test in the repo that pins the cuDNN
  LSE base against an independent reference.

This file doubles as executable documentation of the API (proposal P0).
"""

import zlib

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "cake", "auto"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

FP8_KV_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def quantize_kv(k_cache, v_cache, kv_dtype):
    """Per-tensor fp8 quantization of a float K/V pool for the tests: scale =
    amax / the format's largest finite value (448 for e4m3fn, 57344 for
    e5m2), so both formats use their full range.  Returns the quantized
    caches, the dequantized ``k_ref`` / ``v_ref`` the oracle reads, and the
    two scales ``run()`` takes."""
    qmax = torch.finfo(kv_dtype).max
    k_scale = float(k_cache.abs().amax().item()) / qmax
    v_scale = float(v_cache.abs().amax().item()) / qmax
    k_q = (k_cache.float() / k_scale).to(kv_dtype)
    v_q = (v_cache.float() / v_scale).to(kv_dtype)
    return k_q, v_q, k_q.float() * k_scale, v_q.float() * v_scale, k_scale, v_scale


def make_problem(
    seed,
    *,
    batch_size,
    max_q,
    max_kv,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo=None,
    page_size,
    dtype,
    device="cuda:0",
    uniform_q1=False,
    kv_layout="HND",
    input_form="block_tables",
    kv_dtype=None,
):
    """Random valid paged-prefill problem with scattered (non-identity) page ids.

    Metadata comes from a host generator and Q/K/V from a device generator,
    both seeded by ``seed``, so a repro line reproduces the tensors bitwise
    (the fuzzer prints ``seed=`` on every failure).

    An fp8 ``kv_dtype`` (``float8_e4m3fn`` / ``float8_e5m2``) quantizes K/V
    per-tensor (scale = amax / the format's largest finite value) and keeps
    the dequantized values as ``k_ref``/``v_ref`` for the oracle, so the
    kernel is judged on its math, not on the quantization error.
    """
    head_dim_vo = head_dim_vo or head_dim_qk
    g = torch.Generator().manual_seed(seed)
    g_dev = torch.Generator(device=device).manual_seed(seed)
    if uniform_q1:
        q_lens = torch.ones(batch_size, dtype=torch.int32)
    else:
        q_lens = torch.randint(
            1, max_q + 1, (batch_size,), generator=g, dtype=torch.int32
        )
    kv_extra = torch.randint(
        0, max_kv - 1, (batch_size,), generator=g, dtype=torch.int32
    )
    kv_lens = torch.minimum(q_lens + kv_extra, torch.tensor(max_kv, dtype=torch.int32))

    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages_per_seq = (kv_lens + page_size - 1) // page_size
    width = int(pages_per_seq.max())
    pool_pages = int(pages_per_seq.sum()) + 8  # slack: unused pool pages
    perm = torch.randperm(pool_pages, generator=g, dtype=torch.int32)
    block_tables_cpu = torch.zeros(batch_size, width, dtype=torch.int32)
    off = 0
    for i in range(batch_size):
        n = int(pages_per_seq[i])
        block_tables_cpu[i, :n] = perm[off : off + n]
        off += n

    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(
        total_q, num_qo_heads, head_dim_qk, dtype=dtype, device=device, generator=g_dev
    )
    if kv_layout == "HND":
        k_shape = (pool_pages, num_kv_heads, page_size, head_dim_qk)
        v_shape = (pool_pages, num_kv_heads, page_size, head_dim_vo)
    else:  # NHD
        k_shape = (pool_pages, page_size, num_kv_heads, head_dim_qk)
        v_shape = (pool_pages, page_size, num_kv_heads, head_dim_vo)
    k_cache = torch.randn(*k_shape, dtype=dtype, device=device, generator=g_dev)
    v_cache = torch.randn(*v_shape, dtype=dtype, device=device, generator=g_dev)
    k_ref, v_ref, k_scale, v_scale = k_cache, v_cache, None, None
    if kv_dtype is not None and kv_dtype != dtype:
        k_cache, v_cache, k_ref, v_ref, k_scale, v_scale = quantize_kv(
            k_cache, v_cache, kv_dtype
        )

    # flat CSR page-id list: request-ordered concatenation of each row's
    # live prefix (same info as the dense table)
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages_per_seq[i])] for i in range(batch_size)]
    ).to(torch.int32)

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
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        dtype=dtype,
        device=device,
    )


def make_metadata(p, *, with_mirrors=True):
    """Problem dict -> PagedAttentionMetadata.  ``input_form="both"`` calls the
    raw constructor with both paging forms so the fuzzer can check that this
    is rejected structurally."""
    form = p.get("input_form", "block_tables")
    common = dict(
        page_size=p["page_size"],
        max_q_len=p["max_q_len"],
        max_kv_len=p["max_kv_len"],
        qo_indptr_cpu=p["qo_indptr_cpu"] if with_mirrors else None,
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"] if with_mirrors else None,
    )
    if form == "block_tables":
        return PagedAttentionMetadata.dense(
            p["qo_indptr"], p["kv_seq_lens"], p["block_tables"], **common
        )
    if form == "page_indices":
        return PagedAttentionMetadata.csr(
            p["qo_indptr"], p["kv_seq_lens"], p["kv_page_indices"], **common
        )
    return PagedAttentionMetadata(
        qo_indptr=p["qo_indptr"],
        kv_seq_lens=p["kv_seq_lens"],
        block_tables=p["block_tables"],
        kv_page_indices=p["kv_page_indices"],
        **common,
    )


def run_unified(
    p,
    backend,
    *,
    causal=True,
    lse_mode="base2",
    with_mirrors=True,
    sm_scale=None,
    window_left=-1,
    logits_soft_cap=None,
    custom_mask=None,
    sinks=None,
):
    """Plan and run one batch; the feature axes (``logits_soft_cap``, a
    flattened ``custom_mask``, per-head ``sinks``) are forwarded to plan()
    (``use_sinks`` follows ``sinks is not None``) and run()."""
    attn = PagedAttention(torch.device(p["device"]))
    md = make_metadata(p, with_mirrors=with_mirrors)
    attn.plan(
        md,
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
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        use_sinks=sinks is not None,
        backend=backend,
    )
    run_kwargs = {}
    if sinks is not None:
        run_kwargs["sinks"] = sinks
    out, lse = attn.run(
        p["q"],
        (p["k_cache"], p["v_cache"]),
        out=p.get("_out_override"),
        lse=p.get("_lse_override"),
        sm_scale=sm_scale,
        k_scale=p.get("k_scale"),
        v_scale=p.get("v_scale"),
        **run_kwargs,
    )
    return attn, out, lse


def _resolve_or_skip(p, backend, *, causal=True, need_lse=True, window_left=-1):
    try:
        return resolve_paged_attention(
            device=torch.device(p["device"]),
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
            kv_input_form=(
                "page_indices"
                if p.get("input_form") == "page_indices"
                else "block_tables"
            ),
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"backend {backend} not runnable here: {e}")


def check(p, backend, *, causal=True, window_left=-1):
    _resolve_or_skip(p, backend, causal=causal, window_left=window_left)
    _, out, lse = run_unified(p, backend, causal=causal, window_left=window_left)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"] if p.get("input_form") != "page_indices" else None,
        p["page_size"],
        causal,
        window_left=window_left,
        kv_layout=p.get("kv_layout", "HND"),
        kv_page_indices=p.get("kv_page_indices"),
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    # Output contract: LSE base-2, packed (tokens, h), fp32 — for everyone.
    assert lse.shape == (p["q"].shape[0], p["num_qo_heads"])
    assert lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "batch_size,max_q,max_kv,heads,page_size",
    [
        (4, 64, 512, (8, 8), 16),  # MHA
        (4, 64, 512, (8, 2), 16),  # GQA
        (3, 48, 300, (8, 1), 32),  # MQA, ragged, page 32
        (1, 128, 128, (4, 4), 64),  # single request
    ],
)
def test_paged_attention_conformance(
    backend, batch_size, max_q, max_kv, heads, page_size
):
    p = make_problem(
        seed=zlib.crc32(
            repr((backend, batch_size, max_q, max_kv, heads, page_size)).encode()
        ),
        batch_size=batch_size,
        max_q=max_q,
        max_kv=max_kv,
        num_qo_heads=heads[0],
        num_kv_heads=heads[1],
        head_dim_qk=128,
        page_size=page_size,
        dtype=torch.bfloat16,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paged_attention_decode_shape(backend):
    """Uniform q_len=1 through the same API — decode is a special case of the
    unified contract, not a different world (proposal / PD-统一 argument)."""
    p = make_problem(
        seed=7,
        batch_size=8,
        max_q=1,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        uniform_q1=True,
    )
    check(p, backend)


def test_paged_attention_noncausal_fa2():
    p = make_problem(
        seed=11,
        batch_size=4,
        max_q=32,
        max_kv=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, "fa2", causal=False)


def test_paged_attention_no_mirrors_documented_sync():
    """Without host mirrors the facade does one documented D2H and results
    are identical — the sync is a perf note, never a semantics change."""
    p = make_problem(
        seed=13,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    _resolve_or_skip(p, "fa2")
    _, out_a, lse_a = run_unified(p, "fa2", with_mirrors=True)
    _, out_b, lse_b = run_unified(p, "fa2", with_mirrors=False)
    assert torch.equal(out_a, out_b)
    assert torch.equal(lse_a, lse_b)


def test_resolve_is_static_and_explains():
    """resolve_paged_attention needs no tensors — callable at engine init —
    and reports per-backend exclusion reasons (the anti-rot 'explain')."""
    res = resolve_paged_attention(
        cc_major=9,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
        causal=True,
        need_lse=True,
    )
    assert res.chosen == res.backends[0]
    assert "trtllm-gen" in res.excluded  # sm_90 excluded, with a reason
    assert "sm_9" in res.excluded["trtllm-gen"]
    # explicit pin of an impossible backend raises with the reason
    with pytest.raises(ValueError, match="compute capability"):
        resolve_paged_attention(
            cc_major=8,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
            backend="trtllm-gen",
        )
    # GQA violation is a contract error, not a backend error
    with pytest.raises(ValueError, match="divisible"):
        resolve_paged_attention(
            cc_major=9,
            num_qo_heads=7,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
        )


@pytest.mark.parametrize("backend", ["cudnn", "fa2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attention_headdim_192_128(backend, dtype):
    """(192,128) head dims — capability-honesty: declared rows are tested."""
    if backend == "fa2":
        pytest.skip("fa2 (192,128) not declared in the prototype capability set")
    p = make_problem(
        seed=17,
        batch_size=3,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=192,
        head_dim_vo=128,
        page_size=16,
        dtype=dtype,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen", "cake"])
def test_paged_attention_noncausal(backend):
    p = make_problem(
        seed=11,
        batch_size=4,
        max_q=32,
        max_kv=128,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend, causal=False)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kv_dtype", FP8_KV_DTYPES, ids=["e4m3", "e5m2"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attention_fp8_kv(backend, kv_dtype, dtype):
    """fp8 (e4m3 or e5m2) KV cache with a bf16 / fp16 q and per-tensor
    k_scale/v_scale at run(); the oracle sees the dequantized values, so this
    checks the kernel's in-kernel dequant + scale plumbing, not the
    quantization error (e5m2 measured on B200, see the capability table)."""
    p = make_problem(
        seed=31,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=dtype,
        kv_dtype=kv_dtype,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paged_attention_lse_basee(backend):
    """lse_mode="basee" returns natural-log LSE from every backend (cuDNN
    natively, FA/trtllm-gen via one fold), matching the reference."""
    p = make_problem(
        seed=23,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    _resolve_or_skip(p, backend)
    _, out, lse = run_unified(p, backend, lse_mode="basee")
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
        lse_base="e",
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paged_attention_sm_scale_replan(backend):
    """One plan, two run() calls with different sm_scale must each be correct
    (sm_scale is a per-layer run-time value).  Also the regression for the
    cuDNN graph-cache stale-scale replay (the cache key omitted attn_scale;
    found by this prototype's fuzzer, fixed in flashinfer/cudnn/prefill.py)."""
    p = make_problem(
        seed=19,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    _resolve_or_skip(p, backend)
    for scale_mult in (1.0, 3.0):
        sm_scale = scale_mult / (128**0.5)
        _, out, lse = run_unified(p, backend, sm_scale=sm_scale)
        ref_out, ref_lse = reference_paged_prefill(
            p["q"],
            p["k_ref"],
            p["v_ref"],
            p["qo_indptr_cpu"],
            p["kv_seq_lens_cpu"],
            p["block_tables"],
            p["page_size"],
            True,
            sm_scale=sm_scale,
        )
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_resolution_pinning():
    """plan(backend=Resolution) enforces the init-time pinned config."""
    p = make_problem(
        seed=23,
        batch_size=2,
        max_q=16,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    res = _resolve_or_skip(p, "auto")
    attn = PagedAttention(torch.device(p["device"]))
    md = make_metadata(p)
    attn.plan(
        md,
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend=res,
    )
    assert attn.backend in res.backends
    # drifted config (different heads) must be rejected, not silently re-resolved
    with pytest.raises(ValueError, match="pinned Resolution"):
        attn.plan(
            md,
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_qo_heads"],  # MHA instead of GQA
            head_dim_qk=p["head_dim_qk"],
            q_dtype=p["dtype"],
            causal=True,
            lse_mode="base2",
            backend=res,
        )


def test_envelope_rejections():
    """Negative KV lengths and causal q_len>kv_len on a live row are outside
    the v1 envelope and must be rejected loudly (backends disagree on the LSE
    of fully-masked rows: fa2 finite sentinel vs cudnn -inf).  kv_len 0 is a
    padding row, covered by test_zero_length_kv_rows_are_padding."""
    p = make_problem(
        seed=29,
        batch_size=3,
        max_q=8,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    kvn = p["kv_seq_lens_cpu"].clone()
    kvn[1] = -1
    with pytest.raises(ValueError, match="kv_seq_lens must be >= 0"):
        run_unified(
            {**p, "kv_seq_lens_cpu": kvn, "kv_seq_lens": kvn.to(p["device"])}, "fa2"
        )
    # causal q>kv: force q_len 8 > kv_len 4 on request 0
    kvq = p["kv_seq_lens_cpu"].clone()
    q_lens = p["qo_indptr_cpu"].diff()
    kvq[0] = max(1, int(q_lens[0]) - 1)
    with pytest.raises(ValueError, match="q_len_i <= kv_len_i"):
        run_unified(
            {**p, "kv_seq_lens_cpu": kvq, "kv_seq_lens": kvq.to(p["device"])}, "fa2"
        )


def _padded_problem(
    seed,
    *,
    input_form,
    style,
    page_size=16,
    device="cuda:0",
    kv_lens=(37, 0, 64, 0, 9),
    pool=None,
):
    """A 5-request batch (q lens 5,1,7,1,3) whose kv_len-0 rows (by default
    rows 1 and 3) are engine padding rows.

    ``style="vllm"``: kv_len 0, one query token, table row filled with the
    null block id 0 (vLLM's NULL_BLOCK_ID) - and pool page 0 is poisoned with
    NaN so any read of it shows.  For the CSR form the padding row owns zero
    pages.  ``style="sglang"``: the fill value is 1, i.e. an ordinary live row
    of length 1 on page 0 (kept finite here).  ``pool`` fixes the KV pool
    size (graph tests swap pools of equal size).  Returns the problem dict,
    the boolean mask of live query tokens and the padding-row mask.
    """
    g = torch.Generator().manual_seed(seed)
    hq, hk, d = 8, 2, 128
    q_lens = torch.tensor([5, 1, 7, 1, 3], dtype=torch.int32)
    kv_lens = torch.tensor(list(kv_lens), dtype=torch.int32)
    padding = kv_lens == 0
    if style == "sglang":
        kv_lens = kv_lens.clone()
        kv_lens[padding] = 1
    b = q_lens.shape[0]
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages = (kv_lens + page_size - 1) // page_size
    width = int(pages.max())
    pool = int(pages.sum()) + 6 if pool is None else pool
    assert pool > int(pages.sum())
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    perm = perm[perm != 0]  # keep page 0 as the null block
    table = torch.zeros(b, width, dtype=torch.int32)
    flat = []
    off = 0
    for i in range(b):
        n = int(pages[i])
        if style == "sglang" and bool(padding[i]):
            table[i, :1] = 0
            flat.append(torch.zeros(1, dtype=torch.int32))
            continue
        table[i, :n] = perm[off : off + n]
        flat.append(perm[off : off + n])
        off += n
    kv_page_indices_cpu = torch.cat(flat) if flat else torch.zeros(0, dtype=torch.int32)
    total = int(qo_indptr_cpu[-1])
    q = torch.randn(total, hq, d, dtype=torch.bfloat16, device=device)
    k = torch.randn(pool, hk, page_size, d, dtype=torch.bfloat16, device=device)
    v = torch.randn_like(k)
    if style == "vllm":
        k[0] = float("nan")
        v[0] = float("nan")
    live_tokens = torch.ones(total, dtype=torch.bool)
    for i in range(b):
        if bool(padding[i]):
            live_tokens[int(qo_indptr_cpu[i]) : int(qo_indptr_cpu[i + 1])] = False
    p = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        k_ref=k,
        v_ref=v,
        kv_dtype=torch.bfloat16,
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(device),
        kv_seq_lens_cpu=kv_lens,
        block_tables=table.to(device),
        kv_page_indices=kv_page_indices_cpu.to(device),
        input_form=input_form,
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=d,
        head_dim_vo=d,
        dtype=torch.bfloat16,
        device=device,
    )
    return p, live_tokens, padding


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen"])
@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
@pytest.mark.parametrize("style", ["vllm", "sglang"])
def test_zero_length_kv_rows_are_padding(backend, input_form, style):
    """kv_len 0 rows are legal padding rows (ledger M11): the call succeeds,
    reads no page of those rows (page 0 is NaN-poisoned in the vLLM style),
    and every live row matches the oracle.  Their output and LSE are
    unspecified (trtllm-gen leaves them unwritten) and not compared.  The
    sglang style (fill 1) is an ordinary batch and must match the oracle on
    every row."""
    p, live, padding = _padded_problem(seed=61, input_form=input_form, style=style)
    _resolve_or_skip(p, backend)
    _, out, lse = run_unified(p, backend)
    assert torch.isfinite(out.float()[live.to(out.device)]).all()
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"] if input_form == "block_tables" else None,
        p["page_size"],
        True,
        kv_page_indices=p["kv_page_indices"],
    )
    if style == "sglang":
        live = torch.ones_like(live)
    live_dev = live.to(out.device)
    torch.testing.assert_close(out.float()[live_dev], ref_out[live_dev], **OUT_TOL)
    torch.testing.assert_close(lse[live_dev], ref_lse[live_dev], **LSE_TOL)
    assert int(padding.sum()) == 2


def test_padding_row_last_page_len_convention():
    """A padding row's derived last-page length is page_size (so the legacy
    wrapper's own get_seq_lens formula also yields 0 for it), its page indptr
    is flat, and the fa2 planner ignores the value: feeding the wrapper 0,
    page_size or an arbitrary 7 for that row gives identical live rows and a
    finite padding row."""
    from flashinfer.experimental.paged_attention import derived_needs
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    p, live, padding = _padded_problem(seed=67, input_form="page_indices", style="vllm")
    _resolve_or_skip(p, "fa2")
    page = p["page_size"]
    d = make_metadata(p).derived(needs=derived_needs("fa2"))
    last = d.kv_last_page_len_host
    indptr = d.kv_page_indptr_host
    for i in range(padding.shape[0]):
        if bool(padding[i]):
            assert int(last[i]) == page
            assert int(indptr[i]) == int(indptr[i + 1])
        else:
            assert 1 <= int(last[i]) <= page
    assert torch.equal(
        (indptr[1:] - indptr[:-1] - 1) * page + last, p["kv_seq_lens_cpu"]
    )  # get_seq_lens() of the legacy wrapper reproduces the lengths, 0 included

    dev = torch.device(p["device"])
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
    outs = []
    for value in (0, page, 7):
        last_v = last.clone()
        last_v[padding] = value
        w = BatchPrefillWithPagedKVCacheWrapper(ws, "HND", backend="fa2")
        w.plan(
            d.qo_indptr_host,
            indptr,
            d.kv_page_indices,
            last_v,
            p["num_qo_heads"],
            p["num_kv_heads"],
            p["head_dim_qk"],
            page,
            causal=True,
            q_data_type=p["dtype"],
            kv_data_type=p["dtype"],
            seq_lens=p["kv_seq_lens_cpu"],
        )
        out = w.run(p["q"], (p["k_cache"], p["v_cache"]))
        assert torch.isfinite(out.float()).all()
        outs.append(out)
    live_dev = live.to(dev)
    for out in outs[1:]:
        assert torch.equal(out[live_dev], outs[0][live_dev])


def test_derived_dense_width_override():
    """derived(max_kv_len=...) sizes the dense table derived from flat page
    ids at the given width (graph mode passes the capture capacity so the
    table matches the reserved storage); the batch's own value is the floor,
    and each width is cached separately."""
    from flashinfer.experimental.paged_attention import derived_needs

    p = make_problem(
        seed=69,
        batch_size=3,
        max_q=8,
        max_kv=100,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        input_form="page_indices",
    )
    md = make_metadata(p)
    needs = derived_needs("trtllm-gen")
    own = md.derived(needs=needs)
    wide = md.derived(needs=needs, max_kv_len=p["max_kv_len"] + 50)
    assert own.block_tables.shape[1] == (p["max_kv_len"] + 15) // 16
    assert wide.block_tables.shape[1] == (p["max_kv_len"] + 50 + 15) // 16
    assert md.derived(needs=needs, max_kv_len=p["max_kv_len"] + 50) is wide
    pages = (p["kv_seq_lens_cpu"] + 15) // 16
    for i in range(3):
        n = int(pages[i])
        assert torch.equal(wide.block_tables[i, :n], own.block_tables[i, :n])
    with pytest.raises(ValueError, match="narrower"):
        md.derived(needs=needs, max_kv_len=p["max_kv_len"] - 1)


def test_derive_is_sync_free():
    """The derivation layer must not synchronize (proposal P1 acceptance:
    with mirrors, plan() is zero-D2H).  Guards against masked-select /
    repeat_interleave style data-dependent-size ops sneaking back in."""
    p = make_problem(
        seed=31,
        batch_size=6,
        max_q=16,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    md_dense = make_metadata(p)
    md_csr = make_metadata(dict(p, input_form="page_indices"))
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        d = md_dense.derived(
            needs={"kv_page_indices", "kv_page_indptr", "cum_kv_seq_lens"}
        )
        # reverse direction: flat indices -> dense, also zero-sync
        d2 = md_csr.derived(needs={"block_tables", "q_seq_lens"})
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # correctness of the scatter-compaction vs a host-side reference
    pages = (p["kv_seq_lens_cpu"] + p["page_size"] - 1) // p["page_size"]
    expected = torch.cat(
        [
            p["block_tables"].cpu()[i, : int(pages[i])]
            for i in range(p["kv_seq_lens_cpu"].shape[0])
        ]
    )
    total_pages = int(pages.sum())
    assert torch.equal(d.kv_page_indices.cpu()[:total_pages], expected)
    # CSR->dense round trip: live prefix of each derived dense row matches
    dense = d2.block_tables.cpu()
    for i in range(p["kv_seq_lens_cpu"].shape[0]):
        n = int(pages[i])
        assert torch.equal(dense[i, :n], p["block_tables"].cpu()[i, :n])


def test_derived_forms_are_need_based():
    """Only the forms a backend declares are derived (ledger M7): unrequested
    fields are None and reading one through require() fails loudly, while
    the same object caches one derivation per need set."""
    from flashinfer.experimental.paged_attention import derived_needs

    p = make_problem(
        seed=33,
        batch_size=3,
        max_q=8,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    md = make_metadata(p)
    trt = md.derived(needs=derived_needs("trtllm-gen"))
    assert trt.cum_kv_seq_lens is not None and trt.block_tables is not None
    assert trt.q_seq_lens is None and trt.kv_page_indices is None
    with pytest.raises(AssertionError, match="not requested"):
        trt.require("kv_page_indices")
    assert md.derived(needs=derived_needs("trtllm-gen")) is trt  # cached
    fa = md.derived(needs=derived_needs("fa2"))
    assert fa is not trt and fa.kv_page_indices is not None
    assert fa.cum_kv_seq_lens is None
    with pytest.raises(ValueError, match="unknown derived form"):
        md.derived(needs={"nonsense"})


@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
def test_host_derivation_matches_device_reference(input_form):
    """Every derived form comes from the host mirrors through one pinned
    staging upload; the values must equal what the device tensors imply, the
    host arrays must be pinned int32, and derivation must not sync."""
    from flashinfer.experimental.paged_attention import DERIVED_FORMS

    p = make_problem(
        seed=35,
        batch_size=5,
        max_q=16,
        max_kv=200,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        input_form=input_form,
    )
    md = make_metadata(p)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        d = md.derived(needs=DERIVED_FORMS)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    page = p["page_size"]
    kv = p["kv_seq_lens"]
    zero = torch.zeros(1, dtype=torch.int32, device=kv.device)
    pages = (kv + page - 1) // page
    assert torch.equal(d.q_seq_lens, p["qo_indptr"].diff())
    assert torch.equal(
        d.cum_kv_seq_lens, torch.cat([zero, torch.cumsum(kv, 0, dtype=torch.int32)])
    )
    assert torch.equal(
        d.kv_page_indptr, torch.cat([zero, torch.cumsum(pages, 0, dtype=torch.int32)])
    )
    for name, want in (
        ("qo_indptr_host", p["qo_indptr_cpu"]),
        ("kv_seq_lens_host", p["kv_seq_lens_cpu"]),
        ("kv_page_indptr_host", d.kv_page_indptr.cpu()),
        ("kv_last_page_len_host", ((p["kv_seq_lens_cpu"] - 1) % page + 1)),
    ):
        got = getattr(d, name)
        assert got.dtype == torch.int32 and got.is_pinned(), name
        assert torch.equal(got, want.to(torch.int32)), name
    # cross-form conversions still agree with the caller's tables
    live = torch.cat(
        [p["block_tables"][i, : int(pages[i])] for i in range(kv.shape[0])]
    )
    assert torch.equal(d.kv_page_indices[: live.shape[0]], live)
    for i in range(kv.shape[0]):
        n = int(pages[i])
        assert torch.equal(d.block_tables[i, :n], p["block_tables"][i, :n])


def _gpu_activity(fn):
    """Names of the GPU-side events one call enqueues (kernels + memcpys)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen"])
@pytest.mark.parametrize("input_form", ["block_tables", "page_indices"])
def test_warm_plan_uploads_from_pinned_memory_only(backend, input_form):
    """A warm plan() must never upload from pageable host memory: such a
    non_blocking copy is a blocking staging copy on the host (ledger M6).
    Every host array the generated-FA wrapper uploads is a pinned view of the
    derivation staging, and the dense backends' one upload is that staging."""
    p = make_problem(
        seed=37,
        batch_size=4,
        max_q=16,
        max_kv=200,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        input_form=input_form,
    )
    res = _resolve_or_skip(p, backend)
    attn = PagedAttention(torch.device(p["device"]))
    spec = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend=res,
    )
    for _ in range(2):  # warm: JIT load, first-plan allocations
        attn.plan(make_metadata(p), **spec)
    events = _gpu_activity(lambda: attn.plan(make_metadata(p), **spec))
    pageable = [e for e in events if "Pageable" in e]
    assert not pageable, f"pageable memcpy in a warm plan: {pageable}"
    assert not any("DtoH" in e for e in events), events  # zero-sync plan


@pytest.mark.parametrize("max_q", [32, 1])
def test_cudnn_writes_packed_lse_directly(max_q):
    """cuDNN writes the packed (tokens, h) LSE into the caller's buffer (ledger
    M12): through batch_offsets_stats for max_q > 1, and as the padded
    (b, 1, h) layout that IS the packed layout at max_q == 1 (where cuDNN
    9.25 writes no stats with a ragged offset) - no gather kernel, no copy,
    and the same numbers as the oracle.  On a cuDNN without ragged stats the
    backend falls back to the gather path and this test only checks the
    numbers."""
    p = make_problem(
        seed=39,
        batch_size=4,
        max_q=max_q,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        uniform_q1=max_q == 1,
    )
    _resolve_or_skip(p, "cudnn")
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
        lse_base="e",
    )
    lse_buf = torch.full(
        (p["q"].shape[0], p["num_qo_heads"]),
        float("nan"),
        dtype=torch.float32,
        device=p["device"],
    )
    attn, out, lse = run_unified(
        {**p, "_lse_override": lse_buf}, "cudnn", lse_mode="basee"
    )
    assert lse is lse_buf  # written in place, whichever path
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    if not attn._impl._active.lse_written_packed:
        pytest.skip("this cuDNN takes the gather fallback (probe failed)")
    # the direct path: run() with a natural-log LSE enqueues only cuDNN's own
    # work (its SDPA kernels and, at max_q == 1, its stats memset) - no torch
    # gather kernel and no copy
    kv = (p["k_cache"], p["v_cache"])
    with_lse = _gpu_activity(lambda: attn.run(p["q"], kv, lse=lse_buf))
    assert with_lse and all("cudnn" in e for e in with_lse), with_lse


def _cudnn_probe_module_or_skip():
    from flashinfer.cudnn.prefill import _cudnn_supports_direct_seqlens
    from flashinfer.experimental.paged_attention._backends import cudnn_backend

    if not _cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True):
        pytest.skip("this cuDNN / frontend has no direct paged seq-lens path")
    return cudnn_backend


def test_cudnn_packed_lse_probe_caches_only_the_backends_answer(monkeypatch):
    """The one-time probe caches False only when cuDNN declines the graph
    (cudnnGraphNotSupportedError).  Any other failure propagates uncached: it
    used to be swallowed and cached as False for the process lifetime, so one
    transient error routed every later LSE plan to the gather fallback."""
    import cudnn

    cb = _cudnn_probe_module_or_skip()
    dev = torch.device("cuda")
    ws = torch.empty(16, dtype=torch.int8, device=dev)  # never reached by the fakes
    monkeypatch.setattr(cb, "_PACKED_LSE_SUPPORTED", {})

    def transient(device, workspace):
        raise RuntimeError("transient")

    monkeypatch.setattr(cb, "_probe_packed_lse", transient)
    with pytest.raises(RuntimeError, match="transient"):
        cb._packed_lse_supported(dev, ws)
    assert dev not in cb._PACKED_LSE_SUPPORTED

    def declined(device, workspace):
        raise cudnn.cudnnGraphNotSupportedError("ragged stats not supported")

    monkeypatch.setattr(cb, "_probe_packed_lse", declined)
    assert cb._packed_lse_supported(dev, ws) is False
    assert cb._PACKED_LSE_SUPPORTED[dev] is False
    # once per device: a later success is not consulted
    monkeypatch.setattr(cb, "_probe_packed_lse", lambda device, workspace: None)
    assert cb._packed_lse_supported(dev, ws) is False
    monkeypatch.setattr(cb, "_PACKED_LSE_SUPPORTED", {})
    assert cb._packed_lse_supported(dev, ws) is True


def test_cudnn_packed_lse_probe_is_sync_free():
    """The probe's toy inputs reach the device through one pinned
    asynchronous upload: no blocking copy, no sync, so the first LSE plan on
    a device is as zero-sync as the later ones (and a sync-debug trip cannot
    poison the cached answer)."""
    cb = _cudnn_probe_module_or_skip()
    dev = torch.device("cuda")
    ws = torch.empty(128 << 20, dtype=torch.int8, device=dev)
    cb._probe_packed_lse(dev, ws)  # warm: graph build, allocator
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        cb._probe_packed_lse(dev, ws)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("window_left", [0, 16, 127])
def test_paged_attention_sliding_window(backend, window_left):
    """window_left plumbed through every windowed backend; cudnn is
    capability-excluded (skip via resolve)."""
    p = make_problem(
        seed=37,
        batch_size=4,
        max_q=48,
        max_kv=384,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
    )
    check(p, backend, window_left=window_left)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paged_attention_nhd_layout(backend):
    p = make_problem(
        seed=41,
        batch_size=4,
        max_q=32,
        max_kv=256,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_layout="NHD",
    )
    check(p, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("page_size", [1, 16])
def test_paged_attention_csr_page_indices(backend, page_size):
    """The flat kv_page_indices form (sglang-style); page_size=1 token-CSR is
    in-envelope here, with dense-needing backends capability-excluded."""
    p = make_problem(
        seed=43,
        batch_size=4,
        max_q=32,
        max_kv=192,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=page_size,
        dtype=torch.bfloat16,
        input_form="page_indices",
    )
    check(p, backend)


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen", "cake"])
def test_paged_attention_fp16(backend):
    p = make_problem(
        seed=53,
        batch_size=3,
        max_q=32,
        max_kv=192,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.float16,
    )
    check(p, backend)


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attention_wide_head_dims(backend, head_dim, dtype):
    p = make_problem(
        seed=59,
        batch_size=3,
        max_q=24,
        max_kv=160,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        page_size=16,
        dtype=dtype,
    )
    check(p, backend)


# NOTE: fa2/fa3 paged (192,128) is NOT declared: the paged kernel requires
# k_page_stride == v_page_stride ("K and V must have same page stride for
# sparse attention", batch_prefill_sm90.cu:235), which separately-allocated
# K(D=192)/V(D=128) pools violate.  It is reachable only with a
# stride-matched allocation contract — a capability axis with an allocation
# precondition, out of prototype scope.  cudnn handles (192,128) fine.
