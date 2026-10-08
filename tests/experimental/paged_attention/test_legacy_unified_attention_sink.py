"""Legacy -> unified: tests/attention/test_attention_sink.py

Each legacy test runs a ragged half (``BatchPrefillWithRaggedKVCacheWrapper``
with the AttentionSink JIT variant) and a paged half
(``BatchAttentionWithAttentionSinkWrapper``, page_size 1, contiguous and then
fragmented page ids).  The paged half converts: the legacy fixture runs
through ``PagedAttention`` pinned to the legacy ``backend`` axis (fa2 / fa3;
fa3 skips without SM90a, as legacy), with the ``(tokens, H, D)`` K/V viewed as
``(tokens, 1, H, D)`` NHD pages and the page ids as
``PagedAttentionMetadata.csr``.  Same grid, same seed, same RNG order, so the
node ids equal the legacy ids.  Each case checks the legacy reference
(``sink_attention_unified``) at the legacy tolerance, and the output and LSE
against the fp32 paged-attention oracle (the LSE includes the sink).

Non-causal + sliding window (``causal=False, window_left=128``): fa2 and fa3
decline the combination (their windowed KV range is trimmed as if causal), so
those cases assert the plan's ``ValueError``.  The legacy passes them where
q_len <= window_left and xfails them in the chunk-prefill test.
"""

import math
import random

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import is_sm90a_supported

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_attention_sink.py"
LEGACY_MAP = [
    (
        "tests/attention/test_attention_sink.py::test_attention_sink",
        ["test_attention_sink"],
        "partial",
        "paged half only (contiguous + fragmented page ids, page_size 1) on fa2/fa3; "
        "legacy reference at the legacy tolerance plus the fp32 oracle (output and LSE); "
        "non-causal + window is declined by fa2/fa3 (asserted) where the legacy passes; "
        "the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_incremental_generation",
        ["test_attention_sink_incremental_generation"],
        "partial",
        "paged half only: one PagedAttention re-planned per generation step, contiguous + "
        "fragmented page ids; legacy reference plus the fp32 oracle; non-causal + window "
        "declined (asserted) where the legacy passes; the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_chunk_prefill",
        ["test_attention_sink_chunk_prefill"],
        "partial",
        "paged half only, contiguous + fragmented page ids; legacy reference plus the fp32 "
        "oracle; the legacy xfail for non-causal + window is a clean plan rejection here "
        "(asserted); the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_varlen",
        ["test_attention_sink_varlen"],
        "partial",
        "paged half only, the five legacy indptr configurations, contiguous + fragmented "
        "page ids; legacy reference plus the fp32 oracle; non-causal + window declined "
        "(asserted) where the legacy passes; the ragged half is not paged",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)
NONCAUSAL_WINDOW = "sliding window with non-causal attention not supported"


def _legacy_tol(dtype):
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-3)
    return dict(rtol=1e-2, atol=1e-2)


def _page1_csr(qo_indptr, kv_indptr, kv_indices):
    """The legacy page_size 1 paging (one page per token, token-unit
    kv_indptr) as unified CSR metadata."""
    qo_indptr_cpu = qo_indptr.cpu().int()
    kv_indptr_cpu = kv_indptr.cpu().int()
    kv_lens_cpu = kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]
    return PagedAttentionMetadata.csr(
        qo_indptr_cpu.cuda(),
        kv_lens_cpu.cuda(),
        kv_indices.int().cuda(),
        page_size=1,
        max_q_len=int((qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).max()),
        max_kv_len=int(kv_lens_cpu.max()),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _fragmented(seed, k, v):
    """The legacy fragmented allocation, verbatim: a pool of twice the pages,
    half of the ids occupied at random, the tokens copied into the free ids."""
    total_pages, num_kv_heads, head_dim = k.shape
    random.seed(seed)
    all_pages = list(range(0, total_pages * 2))
    occupied_pages = set(
        random.sample(all_pages, min(total_pages, len(all_pages) // 2))
    )
    available_pages = [p for p in all_pages if p not in occupied_pages]
    kv_indices_fragmented = torch.tensor(
        available_pages[:total_pages], dtype=torch.int32, device=k.device
    )
    k_paged_frag = torch.randn(
        total_pages * 2, 1, num_kv_heads, head_dim, dtype=k.dtype, device=k.device
    )
    v_paged_frag = torch.randn(
        total_pages * 2, 1, num_kv_heads, head_dim, dtype=k.dtype, device=k.device
    )
    for i, page_idx in enumerate(kv_indices_fragmented):
        k_paged_frag[page_idx, 0] = k[i]
        v_paged_frag[page_idx, 0] = v[i]
    return kv_indices_fragmented, k_paged_frag, v_paged_frag


def _run_and_check(
    backend, md, q, k_pages, v_pages, o_ref, sink, sm_scale, causal, window_left
):
    """Plan ``backend`` with sinks, run, then the legacy assertion and the fp32
    oracle.  ``k_pages`` / ``v_pages``: ``(pages, 1, H_kv, D)`` NHD."""
    attn = PagedAttention(q.device)
    attn.plan(
        md,
        num_qo_heads=q.shape[1],
        num_kv_heads=k_pages.shape[2],
        head_dim_qk=q.shape[2],
        q_dtype=q.dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        lse_mode="base2",
        use_sinks=True,
        backend=backend,
    )
    assert attn.backend == backend
    out, lse = attn.run(q, (k_pages, v_pages), sm_scale=sm_scale, sinks=sink)

    # legacy reference at the legacy tolerance
    torch.testing.assert_close(out, o_ref, **_legacy_tol(q.dtype))

    # fp32 oracle: output and base-2 LSE (the LSE includes the sink)
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_pages,
        v_pages,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        None,
        1,
        causal,
        sm_scale=sm_scale,
        window_left=window_left,
        kv_layout="NHD",
        kv_page_indices=md.kv_page_indices,
        sinks=sink,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _as_pages(t):
    """(tokens, H, D) -> (tokens, 1, H, D): one NHD page per token, a view."""
    return t.unsqueeze(1)


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

    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")

    # the legacy fixture (RNG order: q, k, v, sink, then the fragmented pools)
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    q = torch.randn(
        batch_size * seq_len, num_qo_heads, head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        batch_size * seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
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
    check = dict(sink=sink, sm_scale=sm_scale, causal=causal, window_left=window_left)

    # contiguous page ids
    kv_indices_host = torch.arange(0, batch_size * seq_len, dtype=torch.int32)
    md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices_host)
    if not causal and window_left >= 0:
        with pytest.raises(ValueError, match=NONCAUSAL_WINDOW):
            _run_and_check(backend, md, q, _as_pages(k), _as_pages(v), o_ref, **check)
        return
    _run_and_check(backend, md, q, _as_pages(k), _as_pages(v), o_ref, **check)

    # fragmented page ids
    total_pages = batch_size * seq_len
    if total_pages > 1:
        kv_indices, k_frag, v_frag = _fragmented(42 + total_pages, k, v)
        md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices)
        _run_and_check(backend, md, q, k_frag, v_frag, o_ref, **check)


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
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")

    # the legacy fixture (RNG order: k_cache, v_cache, sink; per step q_new,
    # k_new, v_new, then the fragmented pools)
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    k_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
    check = dict(sink=sink, sm_scale=sm_scale, causal=causal, window_left=window_left)

    k_accumulated = v_accumulated = None
    for step in range(num_generation_steps):
        current_kv_len = initial_seq_len + step
        q_new = torch.randn(
            batch_size, num_qo_heads, head_dim, dtype=dtype, device=device
        )
        k_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        if step == 0:
            k_cache_current = k_cache
            v_cache_current = v_cache
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

        # one query token per request; the cache flattened to one page per token
        qo_indptr_host = torch.arange(0, batch_size + 1, dtype=torch.int32)
        kv_indptr_host = torch.arange(
            0, batch_size * current_kv_len + 1, current_kv_len, dtype=torch.int32
        )
        k_flat = k_cache_current.reshape(-1, num_kv_heads, head_dim)
        v_flat = v_cache_current.reshape(-1, num_kv_heads, head_dim)
        kv_indices_host = torch.arange(
            0, batch_size * current_kv_len, dtype=torch.int32
        )
        md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices_host)
        if not causal and window_left >= 0:
            with pytest.raises(ValueError, match=NONCAUSAL_WINDOW):
                _run_and_check(
                    backend,
                    md,
                    q_new,
                    _as_pages(k_flat),
                    _as_pages(v_flat),
                    o_ref,
                    **check,
                )
            return
        _run_and_check(
            backend, md, q_new, _as_pages(k_flat), _as_pages(v_flat), o_ref, **check
        )

        total_pages = batch_size * current_kv_len
        if total_pages > 1:
            kv_indices, k_frag, v_frag = _fragmented(
                42 + step + current_kv_len, k_flat, v_flat
            )
            md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices)
            _run_and_check(backend, md, q_new, k_frag, v_frag, o_ref, **check)

        if step == 0:
            k_accumulated = k_new
            v_accumulated = v_new
        else:
            k_accumulated = torch.cat([k_accumulated, k_new], dim=1)
            v_accumulated = torch.cat([v_accumulated, v_new], dim=1)


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
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if chunk_size >= historical_len:
        pytest.skip(
            "chunk_size should be smaller than historical_len for meaningful chunk prefill test"
        )

    # the legacy fixture (RNG order: q, k, v, sink, then the fragmented pools)
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    total_kv_len = historical_len + chunk_size
    q_chunk = torch.randn(
        batch_size * chunk_size, num_qo_heads, head_dim, dtype=dtype, device=device
    )
    k_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
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
    check = dict(sink=sink, sm_scale=sm_scale, causal=causal, window_left=window_left)

    # contiguous page ids
    qo_indptr_host = torch.arange(
        0, batch_size * chunk_size + 1, chunk_size, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * total_kv_len + 1, total_kv_len, dtype=torch.int32
    )
    kv_indices_host = torch.arange(0, batch_size * total_kv_len, dtype=torch.int32)
    md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices_host)
    if not causal and window_left >= 0:
        with pytest.raises(ValueError, match=NONCAUSAL_WINDOW):
            _run_and_check(
                backend, md, q_chunk, _as_pages(k_all), _as_pages(v_all), o_ref, **check
            )
        return
    _run_and_check(
        backend, md, q_chunk, _as_pages(k_all), _as_pages(v_all), o_ref, **check
    )

    # fragmented page ids
    kv_indices, k_frag, v_frag = _fragmented(
        42 + batch_size + total_kv_len, k_all, v_all
    )
    md = _page1_csr(qo_indptr_host, kv_indptr_host, kv_indices)
    _run_and_check(backend, md, q_chunk, k_frag, v_frag, o_ref, **check)


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

    device = torch.device("cuda:0")
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    qo_indptr, kv_indptr, description = indptr_config
    if len(qo_indptr) != len(kv_indptr):
        pytest.skip(
            f"qo_indptr and kv_indptr must have same batch size for {description}"
        )
    batch_size = len(qo_indptr) - 1
    total_qo_len = qo_indptr[-1]
    total_kv_len = kv_indptr[-1]
    if causal:
        for i in range(batch_size):
            if qo_indptr[i + 1] - qo_indptr[i] > kv_indptr[i + 1] - kv_indptr[i]:
                pytest.skip(
                    "qo_len > kv_len not supported for causal attention in varlen mode"
                )

    # the legacy fixture (RNG order: q, k, v, sink, then the fragmented pools)
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    q = torch.randn(total_qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
    kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=device)
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5
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
    check = dict(sink=sink, sm_scale=sm_scale, causal=causal, window_left=window_left)

    # contiguous page ids
    kv_indices_host = torch.arange(0, total_kv_len, dtype=torch.int32, device=device)
    md = _page1_csr(qo_indptr_tensor, kv_indptr_tensor, kv_indices_host)
    if not causal and window_left >= 0:
        with pytest.raises(ValueError, match=NONCAUSAL_WINDOW):
            _run_and_check(backend, md, q, _as_pages(k), _as_pages(v), o_ref, **check)
        return
    _run_and_check(backend, md, q, _as_pages(k), _as_pages(v), o_ref, **check)

    # fragmented page ids
    kv_indices, k_frag, v_frag = _fragmented(42 + batch_size + total_kv_len, k, v)
    md = _page1_csr(qo_indptr_tensor, kv_indptr_tensor, kv_indices)
    _run_and_check(backend, md, q, k_frag, v_frag, o_ref, **check)
