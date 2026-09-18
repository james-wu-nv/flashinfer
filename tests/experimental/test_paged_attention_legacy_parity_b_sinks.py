"""Legacy -> unified parity, group B: attention sinks on paged KV.

The legacy paged sink entries (``BatchAttentionWithAttentionSinkWrapper``
in ``tests/attention/test_attention_sink.py``) use page_size 1 -- one page
per token -- with contiguous and fragmented page ids; here that is the CSR
form (``PagedAttentionMetadata.csr``), the legacy ``(tokens, H, D)`` K/V
viewed as ``(pages, 1, H, D)`` NHD pools without a copy, the legacy sink
values and ``sm_scale``.  Every case is asserted against the legacy
``sink_attention_unified`` reference at the legacy budget (fp16 1e-3, bf16
1e-2) and against the sink-aware fp32 oracle (output and LSE; the LSE
includes the sink).

The Blackwell entries (``test_attention_sink_blackwell.py``) run the
legacy trtllm-gen fixture on the pinned ``trtllm-gen`` and ``cake``
backends with the dense form.

fa3 rows skip on B200 with the resolve reason (fa3 needs SM90a); the H100
command is in the report.
"""

import math
import random

import pytest
import torch

from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_parity_b_helpers import (
    DEVICE,
    EXPECT_SINK_JIT_URI_HAS_HEAD_DIM,
    EXPECT_TRTLLM_HEAD_DIM_64,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    csr_metadata_page1,
    dense_metadata,
    gated,
    oracle,
    resolve_or_skip,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_MAP = [
    # (legacy nodeid or function, unified test function(s) in this file, status, note)
    (
        "tests/attention/test_attention_sink.py::test_attention_sink",
        ["test_attention_sink"],
        "partial",
        "the paged half of the legacy test: same grid (fp16/bf16, B1/4/16, seq 1/4/16/128, "
        "H32:8/32, window -1/128, causal, fa2/fa3), seed 42, page-1 CSR with the legacy "
        "contiguous and fragmented (random.seed(42 + pages)) page ids, K/V as (pages, 1, H, D) "
        "views; vs sink_attention_unified at the legacy budget and the sink-aware oracle "
        "(out + LSE); fa3 rows skip on B200 (pending H100); the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_incremental_generation",
        ["test_attention_sink_incremental_generation"],
        "partial",
        "same grid (B1/4/16, initial 32/128, steps 1/2/4, H32:8/32, window, causal, dtype, "
        "fa2/fa3) and RNG order; ONE PagedAttention re-planned per step as the KV grows by "
        "one token with the same sink values, contiguous and fragmented page-1 CSR; vs the "
        "incremental reference at the legacy budget and the oracle; fa3 pending H100",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_chunk_prefill",
        ["test_attention_sink_chunk_prefill"],
        "partial",
        "same grid (B1/4/16, chunk 128/256, history 256/512, H32:8/32, window, causal, "
        "dtype, fa2/fa3), same skip (chunk >= history); the legacy imperative xfail for "
        "non-causal + window is a non-strict xfail marker here so the unified result is "
        "recorded (B200: 32 XPASS, 4 XFAIL: the fa2 window mask is wrong for the query rows "
        "more than window_left before the sequence end when q_len > window_left, sinks or "
        "not; see wp-s-b.md open issues); page-1 CSR contiguous + fragmented; vs the chunk "
        "reference and the oracle; fa3 pending H100",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_varlen",
        ["test_attention_sink_varlen"],
        "partial",
        "the five legacy indptr configurations x dtype x H32:8/32 x window x causal x "
        "fa2/fa3, page-1 CSR contiguous + fragmented; vs the varlen reference and the "
        "oracle; fa3 pending H100",
    ),
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_context_attention_sink",
        ["test_blackwell_context_attention_sink"],
        "partial",
        "same grid (fp16/bf16, B1/4/16, page32, seq 32/128/1024, H32:8/32, D64/128), seed 0, "
        "sm_scale 1.0, dense form, pinned trtllm-gen and (added) cake; vs the prefill sink "
        "reference at the legacy budget (fp16 2e-3/1e-3, bf16 1e-2) and the oracle; D64 is "
        "capability-excluded on both (EXPECT_TRTLLM_HEAD_DIM_64) and runs on fa2 instead, "
        "where it reproduces CR02 (sink JIT URI without head_dim): skipped with the reason "
        "until EXPECT_SINK_JIT_URI_HAS_HEAD_DIM; passes in its own process",
    ),
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_decode_attention_sink",
        ["test_blackwell_decode_attention_sink"],
        "partial",
        "the decode entry (trtllm_batch_decode_with_kv_cache, q_len 1) on the unified "
        "context path: same fixture with one query token per request, sm_scale 1.0, vs the "
        "incremental sink reference at the legacy budget (fp16 1e-3, bf16 1e-2) and the "
        "oracle; D64 as above (CR02 skip)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())


HEAD_DIM = 128


def _legacy_tol(dtype):
    return (
        dict(rtol=1e-3, atol=1e-3)
        if dtype == torch.float16
        else dict(rtol=1e-2, atol=1e-2)
    )


def _pages_view(t):
    """(tokens, H, D) -> (tokens, 1, H, D): one NHD page per token, no copy."""
    v = t.view(t.shape[0], 1, t.shape[1], t.shape[2])
    assert v.data_ptr() == t.data_ptr()
    return v


def _fragmented(total_pages, seed, k_tokens, v_tokens, dtype, num_kv_heads):
    """The legacy fragmented page allocation: a pool twice the size, half the
    ids occupied, the tokens copied into the free ids in order."""
    random.seed(seed)
    all_pages = list(range(0, total_pages * 2))
    occupied_pages = set(
        random.sample(all_pages, min(total_pages, len(all_pages) // 2))
    )
    available_pages = [p for p in all_pages if p not in occupied_pages]
    kv_indices_fragmented = torch.tensor(
        available_pages[:total_pages], dtype=torch.int32, device=DEVICE
    )
    k_paged_frag = torch.randn(
        total_pages * 2, 1, num_kv_heads, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    v_paged_frag = torch.randn(
        total_pages * 2, 1, num_kv_heads, HEAD_DIM, dtype=dtype, device=DEVICE
    )
    for i, page_idx in enumerate(kv_indices_fragmented):
        k_paged_frag[page_idx, 0] = k_tokens[i]
        v_paged_frag[page_idx, 0] = v_tokens[i]
    return kv_indices_fragmented, k_paged_frag, v_paged_frag


def _resolve_sinks(
    backend,
    *,
    num_qo_heads,
    num_kv_heads,
    dtype,
    causal,
    window_left,
    page_size=1,
    kv_layout="NHD",
    head_dim=HEAD_DIM,
):
    return resolve_or_skip(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=True,
        window_left=window_left,
        kv_input_form="page_indices" if page_size == 1 else "block_tables",
        sinks=True,
    )


def _run_sinks(
    attn,
    res,
    md,
    q,
    k,
    v,
    *,
    num_qo_heads,
    num_kv_heads,
    dtype,
    causal,
    window_left,
    sink,
    sm_scale,
    backend,
    kv_layout="NHD",
    head_dim=HEAD_DIM,
):
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout=kv_layout,
        causal=causal,
        window_left=window_left,
        lse_mode="base2",
        use_sinks=True,
        backend=res,
    )
    assert attn.backend == backend
    return attn.run(q, (k, v), sm_scale=sm_scale, sinks=sink)


def _check(
    out,
    lse,
    o_ref,
    md,
    q,
    k,
    v,
    *,
    dtype,
    causal,
    window_left,
    sink,
    sm_scale,
    kv_layout="NHD",
    num_qo_heads=None,  # the shared ``common`` dict carries the plan's heads too
    num_kv_heads=None,
):
    torch.testing.assert_close(out, o_ref, **_legacy_tol(dtype))  # legacy assertion
    ref_out, ref_lse = oracle(
        md,
        q,
        k,
        v,
        causal=causal,
        kv_layout=kv_layout,
        sm_scale=sm_scale,
        window_left=window_left,
        sinks=sink,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    assert lse.shape == (q.shape[0], sink.shape[0]) and lse.dtype == torch.float32
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)  # the LSE includes the sink


# ---------------------------------------------------------------------------
# test_attention_sink (paged half)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 4, 16, 128])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink(
    dtype, batch_size, seq_len, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    res = _resolve_sinks(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
    )
    # ---- the legacy fixture (RNG order as legacy: seed, then q, k, v, sink) ----
    torch.manual_seed(42)
    head_dim = HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=dtype, device=DEVICE
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        batch_size=batch_size,
        mode="prefill",
    )

    common = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
        sink=sink,
        sm_scale=sm_scale,
    )
    # contiguous page ids: one page per token
    kv_indices_host = torch.arange(0, batch_size * seq_len, dtype=torch.int32)
    md = csr_metadata_page1(qo_indptr_host, kv_indptr_host, kv_indices_host)
    attn = PagedAttention(torch.device(DEVICE))
    out, lse = _run_sinks(
        attn, res, md, q, _pages_view(k), _pages_view(v), backend=backend, **common
    )
    _check(out, lse, o_ref, md, q, _pages_view(k), _pages_view(v), **common)

    # fragmented page ids (the legacy production scenario)
    total_pages = batch_size * seq_len
    if total_pages > 1:
        idx, k_frag, v_frag = _fragmented(
            total_pages, 42 + total_pages, k, v, dtype, num_kv_heads
        )
        md_f = csr_metadata_page1(qo_indptr_host, kv_indptr_host, idx)
        out, lse = _run_sinks(
            PagedAttention(torch.device(DEVICE)),
            res,
            md_f,
            q,
            k_frag,
            v_frag,
            backend=backend,
            **common,
        )
        _check(out, lse, o_ref, md_f, q, k_frag, v_frag, **common)


# ---------------------------------------------------------------------------
# test_attention_sink_incremental_generation (paged half)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("initial_seq_len", [32, 128])
@pytest.mark.parametrize("num_generation_steps", [1, 2, 4])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_incremental_generation(
    dtype,
    batch_size,
    initial_seq_len,
    num_generation_steps,
    num_qo_heads,
    num_kv_heads,
    window_left,
    causal,
    backend,
):
    """q_len 1, the KV grows by one token per step, the sink values stay:
    one PagedAttention instance re-planned per step."""
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    res = _resolve_sinks(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
    )
    torch.manual_seed(42)
    head_dim = HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    k_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    v_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    common = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
        sink=sink,
        sm_scale=sm_scale,
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn_frag = PagedAttention(torch.device(DEVICE))
    k_accumulated = v_accumulated = None
    for step in range(num_generation_steps):
        current_kv_len = initial_seq_len + step
        q_new = torch.randn(
            batch_size, num_qo_heads, head_dim, dtype=dtype, device=DEVICE
        )
        k_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
        )
        v_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
        )
        if step == 0:
            k_cache_current, v_cache_current = k_cache, v_cache
        else:
            k_cache_current = torch.cat([k_cache, k_accumulated], dim=1)
            v_cache_current = torch.cat([v_cache, v_accumulated], dim=1)
        o_ref = sink_attention_unified(
            q_new,
            k_cache_current,
            v_cache_current,
            sink,
            window_left,
            causal,
            sm_scale,
            mode="incremental",
        )
        qo_indptr_host = torch.arange(0, batch_size + 1, dtype=torch.int32)
        kv_indptr_host = torch.arange(
            0, batch_size * current_kv_len + 1, current_kv_len, dtype=torch.int32
        )
        q_flashinfer = q_new.view(batch_size, num_qo_heads, head_dim)
        k_flashinfer = k_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )
        v_flashinfer = v_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )

        kv_indices_host = torch.arange(
            0, batch_size * current_kv_len, dtype=torch.int32
        )
        md = csr_metadata_page1(qo_indptr_host, kv_indptr_host, kv_indices_host)
        k_p, v_p = _pages_view(k_flashinfer), _pages_view(v_flashinfer)
        out, lse = _run_sinks(
            attn, res, md, q_flashinfer, k_p, v_p, backend=backend, **common
        )
        _check(out, lse, o_ref, md, q_flashinfer, k_p, v_p, **common)

        total_pages = batch_size * current_kv_len
        if total_pages > 1:
            idx, k_frag, v_frag = _fragmented(
                total_pages,
                42 + step + current_kv_len,
                k_flashinfer,
                v_flashinfer,
                dtype,
                num_kv_heads,
            )
            md_f = csr_metadata_page1(qo_indptr_host, kv_indptr_host, idx)
            out, lse = _run_sinks(
                attn_frag,
                res,
                md_f,
                q_flashinfer,
                k_frag,
                v_frag,
                backend=backend,
                **common,
            )
            _check(out, lse, o_ref, md_f, q_flashinfer, k_frag, v_frag, **common)

        if step == 0:
            k_accumulated, v_accumulated = k_new, v_new
        else:
            k_accumulated = torch.cat([k_accumulated, k_new], dim=1)
            v_accumulated = torch.cat([v_accumulated, v_new], dim=1)


# ---------------------------------------------------------------------------
# test_attention_sink_chunk_prefill (paged half)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("historical_len", [256, 512])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_chunk_prefill(
    request,
    dtype,
    batch_size,
    chunk_size,
    historical_len,
    num_qo_heads,
    num_kv_heads,
    window_left,
    causal,
    backend,
):
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    if not causal and window_left >= 0:
        # the legacy test xfails this cell imperatively (never runs it); here
        # it runs and the outcome is recorded (XPASS if the unified path is fine)
        request.node.add_marker(
            pytest.mark.xfail(
                strict=False,
                reason="legacy xfail: attention sink with sliding window and non-causal",
            )
        )
    if chunk_size >= historical_len:
        pytest.skip(
            "chunk_size should be smaller than historical_len for meaningful chunk prefill test"
        )
    res = _resolve_sinks(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
    )
    torch.manual_seed(42)
    head_dim = HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    total_kv_len = historical_len + chunk_size
    q_chunk = torch.randn(
        batch_size * chunk_size, num_qo_heads, head_dim, dtype=dtype, device=DEVICE
    )
    k_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    v_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    o_ref = sink_attention_unified(
        q_chunk,
        k_all,
        v_all,
        sink,
        window_left,
        causal,
        sm_scale,
        batch_size=batch_size,
        mode="chunk",
    )
    common = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
        sink=sink,
        sm_scale=sm_scale,
    )
    qo_indptr_host = torch.arange(
        0, batch_size * chunk_size + 1, chunk_size, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * total_kv_len + 1, total_kv_len, dtype=torch.int32
    )
    kv_indices_host = torch.arange(0, batch_size * total_kv_len, dtype=torch.int32)
    md = csr_metadata_page1(qo_indptr_host, kv_indptr_host, kv_indices_host)
    k_p, v_p = _pages_view(k_all), _pages_view(v_all)
    out, lse = _run_sinks(
        PagedAttention(torch.device(DEVICE)),
        res,
        md,
        q_chunk,
        k_p,
        v_p,
        backend=backend,
        **common,
    )
    _check(out, lse, o_ref, md, q_chunk, k_p, v_p, **common)

    total_pages = batch_size * total_kv_len
    idx, k_frag, v_frag = _fragmented(
        total_pages, 42 + batch_size + total_kv_len, k_all, v_all, dtype, num_kv_heads
    )
    md_f = csr_metadata_page1(qo_indptr_host, kv_indptr_host, idx)
    out, lse = _run_sinks(
        PagedAttention(torch.device(DEVICE)),
        res,
        md_f,
        q_chunk,
        k_frag,
        v_frag,
        backend=backend,
        **common,
    )
    _check(out, lse, o_ref, md_f, q_chunk, k_frag, v_frag, **common)


# ---------------------------------------------------------------------------
# test_attention_sink_varlen (paged half)
# ---------------------------------------------------------------------------

VARLEN_CONFIGS = [
    (
        [0, 32, 64, 128, 256],
        [0, 128, 256, 512, 1024],
        "4 requests: prefill-like scenarios",
    ),
    ([0, 1, 2, 3, 4], [0, 128, 256, 384, 512], "4 requests: incremental generation"),
    ([0, 50, 150, 200], [0, 200, 600, 800], "3 requests: mixed lengths"),
    (
        [0, 100, 200, 400, 600, 1000],
        [0, 300, 600, 1200, 1800, 3000],
        "5 requests: large sequences",
    ),
    ([0, 16, 32, 96, 128], [0, 64, 128, 384, 512], "4 requests: chunk prefill-like"),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "indptr_config", VARLEN_CONFIGS, ids=[c[2] for c in VARLEN_CONFIGS]
)
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_varlen(
    dtype, indptr_config, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    qo_indptr, kv_indptr, _description = indptr_config
    batch_size = len(qo_indptr) - 1
    total_qo_len, total_kv_len = qo_indptr[-1], kv_indptr[-1]
    if causal:
        for i in range(batch_size):
            if qo_indptr[i + 1] - qo_indptr[i] > kv_indptr[i + 1] - kv_indptr[i]:
                pytest.skip(
                    "qo_len > kv_len not supported for causal attention in varlen mode"
                )
    res = _resolve_sinks(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
    )
    torch.manual_seed(42)
    head_dim = HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    q = torch.randn(total_qo_len, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device=DEVICE)
    kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=DEVICE)
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        mode="varlen",
        qo_indptr=qo_indptr_tensor,
        kv_indptr=kv_indptr_tensor,
    )
    assert o_ref.shape == (total_qo_len, num_qo_heads, head_dim)
    common = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=causal,
        window_left=window_left,
        sink=sink,
        sm_scale=sm_scale,
    )
    kv_indices_host = torch.arange(0, total_kv_len, dtype=torch.int32)
    md = csr_metadata_page1(qo_indptr_tensor, kv_indptr_tensor, kv_indices_host)
    k_p, v_p = _pages_view(k), _pages_view(v)
    out, lse = _run_sinks(
        PagedAttention(torch.device(DEVICE)),
        res,
        md,
        q,
        k_p,
        v_p,
        backend=backend,
        **common,
    )
    _check(out, lse, o_ref, md, q, k_p, v_p, **common)

    idx, k_frag, v_frag = _fragmented(
        total_kv_len, 42 + batch_size + total_kv_len, k, v, dtype, num_kv_heads
    )
    md_f = csr_metadata_page1(qo_indptr_tensor, kv_indptr_tensor, idx)
    out, lse = _run_sinks(
        PagedAttention(torch.device(DEVICE)),
        res,
        md_f,
        q,
        k_frag,
        v_frag,
        backend=backend,
        **common,
    )
    _check(out, lse, o_ref, md_f, q, k_frag, v_frag, **common)


# ---------------------------------------------------------------------------
# test_attention_sink_blackwell.py (trtllm-gen fixture, sm_scale 1.0)
# ---------------------------------------------------------------------------


def _blackwell_fixture(
    dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim, q_len
):
    """The legacy Blackwell sink fixture: full pages, arange block table,
    separate HND K/V pools, ``sink = rand(H) * 5``; ``q_len`` is ``seq_len``
    for the context entry and 1 for the decode entry (RNG order as legacy)."""
    torch.manual_seed(0)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=DEVICE)
    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int32, device=DEVICE
    ).reshape(batch_size, max_num_blocks_per_seq)
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        batch_size * q_len, num_qo_heads, head_dim, dtype=dtype, device=DEVICE
    )
    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=DEVICE
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=DEVICE
    )
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * q_len
    md = dense_metadata(qo_indptr, seq_lens.cpu(), block_tables, page_size)
    return dict(
        q=q, k_cache=k_cache, v_cache=v_cache, sink=sink, md=md, seq_lens=seq_lens
    )


def _blackwell_backend(
    backend, *, num_qo_heads, num_kv_heads, dtype, head_dim, page_size
):
    """The pinned backend for a Blackwell row; D64 is capability-excluded on
    trtllm-gen / cake (EXPECT_TRTLLM_HEAD_DIM_64) and the row runs on fa2."""
    if head_dim == 64:
        res = gated(
            EXPECT_TRTLLM_HEAD_DIM_64,
            lambda: resolve_paged_attention(
                device=torch.device(DEVICE),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_dtype=dtype,
                page_size=page_size,
                kv_layout="HND",
                causal=True,
                need_lse=True,
                sinks=True,
                backend=backend,
            ),
            match="unsupported head dims \\(64, 64\\)",
        )
        if res is not None:
            return backend, res
        backend = "fa2"
    res = _resolve_sinks(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=True,
        window_left=-1,
        page_size=page_size,
        kv_layout="HND",
        head_dim=head_dim,
    )
    return backend, res


def _mark_cr02(request, backend, head_dim):
    """CR02 (WP-P): the sink wrapper's JIT URI has no head_dim, so a process
    that built the D128 sink module first serves it for D64 too.  Measured
    here (B200): the D64 fixture then returns NaN AND leaves a sticky illegal
    memory access that aborts the pytest process while the failure is being
    reported; the reverse order returns wrong values.  Alone (own process) the
    D64 rows pass.  Skipped, not xfailed, so the file survives; flips with
    EXPECT_SINK_JIT_URI_HAS_HEAD_DIM."""
    if (
        backend == "fa2"
        and head_dim != HEAD_DIM
        and not EXPECT_SINK_JIT_URI_HAS_HEAD_DIM
    ):
        pytest.skip(
            "CR02: BatchAttentionWithAttentionSinkWrapper JIT URI lacks head_dim; a D64 "
            "sink module after the D128 rows of this file reuses the D128 module (NaN + "
            "sticky illegal memory access); run the D64 rows in their own process"
        )


def _blackwell_tol(dtype, *, context):
    if dtype == torch.float16:
        return dict(atol=2e-3, rtol=1e-3) if context else dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("backend", ["trtllm-gen", "cake"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_context_attention_sink(
    request,
    backend,
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    import einops
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    backend, res = _blackwell_backend(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
        page_size=page_size,
    )
    _mark_cr02(request, backend, head_dim)
    f = _blackwell_fixture(
        dtype,
        batch_size,
        page_size,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    )
    attn = PagedAttention(torch.device(DEVICE))
    out, lse = _run_sinks(
        attn,
        res,
        f["md"],
        f["q"],
        f["k_cache"],
        f["v_cache"],
        backend=backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=True,
        window_left=-1,
        sink=f["sink"],
        sm_scale=1.0,
        kv_layout="HND",
        head_dim=head_dim,
    )
    k = einops.rearrange(f["k_cache"], "num_pages h p d -> (num_pages p) h d")
    v = einops.rearrange(f["v_cache"], "num_pages h p d -> (num_pages p) h d")
    o_ref = sink_attention_unified(
        f["q"], k, v, f["sink"], -1, True, 1.0, mode="prefill", batch_size=batch_size
    )
    torch.testing.assert_close(
        o_ref, out, **_blackwell_tol(dtype, context=True)
    )  # legacy
    ref_out, ref_lse = oracle(
        f["md"],
        f["q"],
        f["k_cache"],
        f["v_cache"],
        causal=True,
        kv_layout="HND",
        sm_scale=1.0,
        sinks=f["sink"],
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", ["trtllm-gen", "cake"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_decode_attention_sink(
    request,
    backend,
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    import einops
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    backend, res = _blackwell_backend(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
        page_size=page_size,
    )
    _mark_cr02(request, backend, head_dim)
    f = _blackwell_fixture(
        dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim, 1
    )
    attn = PagedAttention(torch.device(DEVICE))
    out, lse = _run_sinks(
        attn,
        res,
        f["md"],
        f["q"],
        f["k_cache"],
        f["v_cache"],
        backend=backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        causal=True,
        window_left=-1,
        sink=f["sink"],
        sm_scale=1.0,
        kv_layout="HND",
        head_dim=head_dim,
    )
    max_num_blocks_per_seq = f["md"].block_tables.shape[1]
    k = einops.rearrange(
        f["k_cache"], "(b n) h p d -> b (n p) h d", n=max_num_blocks_per_seq
    )
    v = einops.rearrange(
        f["v_cache"], "(b n) h p d -> b (n p) h d", n=max_num_blocks_per_seq
    )
    o_ref = sink_attention_unified(
        f["q"], k, v, f["sink"], -1, False, 1.0, mode="incremental"
    )
    torch.testing.assert_close(
        o_ref, out, **_blackwell_tol(dtype, context=False)
    )  # legacy
    ref_out, ref_lse = oracle(
        f["md"],
        f["q"],
        f["k_cache"],
        f["v_cache"],
        causal=True,
        kv_layout="HND",
        sm_scale=1.0,
        sinks=f["sink"],
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
