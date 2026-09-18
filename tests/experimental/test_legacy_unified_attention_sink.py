"""Legacy -> unified: tests/attention/test_attention_sink.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only (fa3 rows SM90a-gated;
the fa2 rows are not in the A10G fixed shards); the unified file is in no
default lane (tests/experimental is excluded by norecursedirs).

Each legacy function runs a ragged half (BatchPrefillWithRaggedKVCacheWrapper
+ the AttentionSink JIT variant) and a paged half
(BatchAttentionWithAttentionSinkWrapper, page_size 1, contiguous and
fragmented page ids).  The paged half is what converts: the legacy fixture
(seed 42, RNG order q, k, v, sink; ``sink = rand(H) * 5``; the legacy
fragmented allocation ``random.seed(42 + ...)``) with page_size 1 is the CSR
form (``PagedAttentionMetadata.csr``), the ``(tokens, H, D)`` K/V viewed as
``(pages, 1, H, D)`` NHD pools without a copy, pinned ``backend`` (fa2 / fa3
as the legacy axis), ``use_sinks=True`` at plan and the legacy sink tensor at
run.  Assertions: the legacy ``sink_attention_unified`` reference at the
legacy budget (fp16 1e-3, bf16 1e-2) and the sink-aware fp32 oracle (output
and LSE; the LSE includes the sink).  Default run = the full legacy grids
(3552 ids incl. fa3; ``-k "not fa3"`` on the shared Blackwell container).

fa3 rows: collected; on B200 they skip with the resolve reason ("fa3:
unsupported compute capability sm_10x").  H100 command (repo root):

    python -m pytest -q -ra -o faulthandler_timeout=300 \\
        tests/experimental/test_legacy_unified_attention_sink.py -k fa3

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- all four functions: the ragged half (BatchPrefillWithRaggedKVCacheWrapper
  with the AttentionSink JIT args) is not paged prefill and is not converted;
  the paged half is converted in full.
- non-causal + sliding window cells (causal=False, window_left=128) of all
  four functions: fa2 and fa3 declare supports_window_noncausal=False
  (_capabilities.py, M20: the windowed KV range of prefill.cuh is trimmed as
  if causal, wrong once q_len > window_left), so the unified rows SKIP with
  the resolve reason.  The legacy runs and passes these cells whenever
  q_len <= window_left (test_attention_sink seq <= 128, the incremental
  q_len 1 rows, the varlen configurations) and xfails them imperatively in
  test_attention_sink_chunk_prefill ("attention sink with sliding window and
  non-causal"): the declaration over-excludes the q_len <= window_left
  class the legacy covers (support-surface gap; a shape-aware declaration
  would re-admit it).
"""

import math
import random

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    csr_metadata_page1,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_attention_sink.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_attention_sink.py::test_attention_sink",
        ["test_attention_sink"],
        "partial",
        "the paged half of the legacy test: same grid (fp16/bf16, B1/4/16, seq 1/4/16/128, "
        "H32:8/32, window -1/128, causal, fa2/fa3), seed 42, page-1 CSR with the legacy "
        "contiguous and fragmented (random.seed(42 + pages)) page ids, K/V as "
        "(pages, 1, H, D) views; vs sink_attention_unified at the legacy budget and the "
        "sink-aware oracle (out + LSE); fa3 rows skip on B200 (pending H100); the "
        "non-causal + window cells skip with the resolve reason (fa2/fa3 declare "
        "supports_window_noncausal=False) where the legacy passes (q_len <= window_left); "
        "the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_incremental_generation",
        ["test_attention_sink_incremental_generation"],
        "partial",
        "same grid (B1/4/16, initial 32/128, steps 1/2/4, H32:8/32, window, causal, dtype, "
        "fa2/fa3) and RNG order; ONE PagedAttention re-planned per step as the KV grows by "
        "one token with the same sink values, contiguous and fragmented page-1 CSR; vs the "
        "incremental reference at the legacy budget and the oracle; fa3 pending H100; "
        "non-causal + window cells skip (declaration) where the legacy passes (q_len 1); "
        "the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_chunk_prefill",
        ["test_attention_sink_chunk_prefill"],
        "partial",
        "same grid (B1/4/16, chunk 128/256, history 256/512, H32:8/32, window, causal, "
        "dtype, fa2/fa3), same skip (chunk >= history); the legacy imperative xfail for "
        "non-causal + window is, through unified, the capability exclusion "
        "supports_window_noncausal=False (M20: the fa2 window mask is wrong for query rows "
        "more than window_left before the sequence end when q_len > window_left, sinks or "
        "not), so those cells skip with the resolve reason; page-1 CSR contiguous + "
        "fragmented; vs the chunk reference and the oracle; fa3 pending H100; the ragged "
        "half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_varlen",
        ["test_attention_sink_varlen"],
        "partial",
        "the five legacy indptr configurations x dtype x H32:8/32 x window x causal x "
        "fa2/fa3 with the legacy causal qo_len > kv_len skip, page-1 CSR contiguous + "
        "fragmented; vs the varlen reference and the oracle; fa3 pending H100; non-causal "
        "+ window cells skip (declaration) where the legacy passes; the ragged half is not "
        "paged",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


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


def _run_sinks(
    backend,
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
    attn=None,
):
    """Pinned-backend sink plan (skips with the resolve reason where the
    capability table excludes the row) and run with the legacy sink tensor."""
    attn = plan_pinned(
        backend,
        md,
        attn=attn,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=HEAD_DIM,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        lse_mode="base2",
        use_sinks=True,
    )
    out, lse = attn.run(q, (k, v), sm_scale=sm_scale, sinks=sink)
    return attn, out, lse


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
        kv_layout="NHD",
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
    _, out, lse = _run_sinks(backend, md, q, _pages_view(k), _pages_view(v), **common)
    _check(out, lse, o_ref, md, q, _pages_view(k), _pages_view(v), **common)

    # fragmented page ids (the legacy production scenario)
    total_pages = batch_size * seq_len
    if total_pages > 1:
        idx, k_frag, v_frag = _fragmented(
            total_pages, 42 + total_pages, k, v, dtype, num_kv_heads
        )
        md_f = csr_metadata_page1(qo_indptr_host, kv_indptr_host, idx)
        _, out, lse = _run_sinks(backend, md_f, q, k_frag, v_frag, **common)
        _check(out, lse, o_ref, md_f, q, k_frag, v_frag, **common)


# ---------------------------------------------------------------------------
# test_attention_sink_incremental_generation (paged half)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
        attn, out, lse = _run_sinks(
            backend, md, q_flashinfer, k_p, v_p, attn=attn, **common
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
            attn_frag, out, lse = _run_sinks(
                backend, md_f, q_flashinfer, k_frag, v_frag, attn=attn_frag, **common
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("historical_len", [256, 512])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_sink_chunk_prefill(
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
    """Chunk prefill: q_len != kv_len and q_len > 1.  The legacy test xfails
    the non-causal + window cells imperatively; through unified those cells
    are the capability exclusion supports_window_noncausal=False and skip at
    resolve with that reason."""
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    if chunk_size >= historical_len:
        pytest.skip(
            "chunk_size should be smaller than historical_len for meaningful chunk prefill test"
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
    _, out, lse = _run_sinks(backend, md, q_chunk, k_p, v_p, **common)
    _check(out, lse, o_ref, md, q_chunk, k_p, v_p, **common)

    total_pages = batch_size * total_kv_len
    idx, k_frag, v_frag = _fragmented(
        total_pages, 42 + batch_size + total_kv_len, k_all, v_all, dtype, num_kv_heads
    )
    md_f = csr_metadata_page1(qo_indptr_host, kv_indptr_host, idx)
    _, out, lse = _run_sinks(backend, md_f, q_chunk, k_frag, v_frag, **common)
    _check(out, lse, o_ref, md_f, q_chunk, k_frag, v_frag, **common)


# ---------------------------------------------------------------------------
# test_attention_sink_varlen (paged half)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "indptr_config",
    [
        # (qo_indptr, kv_indptr, description)
        (
            [0, 32, 64, 128, 256],
            [0, 128, 256, 512, 1024],
            "4 requests: prefill-like scenarios",
        ),
        (
            [0, 1, 2, 3, 4],
            [0, 128, 256, 384, 512],
            "4 requests: incremental generation",
        ),
        ([0, 50, 150, 200], [0, 200, 600, 800], "3 requests: mixed lengths"),
        (
            [0, 100, 200, 400, 600, 1000],
            [0, 300, 600, 1200, 1800, 3000],
            "5 requests: large sequences",
        ),
        (
            [0, 16, 32, 96, 128],
            [0, 64, 128, 384, 512],
            "4 requests: chunk prefill-like",
        ),
    ],
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

    qo_indptr, kv_indptr, description = indptr_config
    if len(qo_indptr) != len(kv_indptr):
        pytest.skip(
            f"qo_indptr and kv_indptr must have same batch size for {description}"
        )
    batch_size = len(qo_indptr) - 1
    total_qo_len, total_kv_len = qo_indptr[-1], kv_indptr[-1]
    head_dim = HEAD_DIM
    sm_scale = 1.0 / math.sqrt(head_dim)
    if causal:
        for i in range(batch_size):
            if qo_indptr[i + 1] - qo_indptr[i] > kv_indptr[i + 1] - kv_indptr[i]:
                pytest.skip(
                    "qo_len > kv_len not supported for causal attention in varlen mode"
                )
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
    _, out, lse = _run_sinks(backend, md, q, k_p, v_p, **common)
    _check(out, lse, o_ref, md, q, k_p, v_p, **common)

    idx, k_frag, v_frag = _fragmented(
        total_kv_len, 42 + batch_size + total_kv_len, k, v, dtype, num_kv_heads
    )
    md_f = csr_metadata_page1(qo_indptr_tensor, kv_indptr_tensor, idx)
    _, out, lse = _run_sinks(backend, md_f, q, k_frag, v_frag, **common)
    _check(out, lse, o_ref, md_f, q, k_frag, v_frag, **common)
