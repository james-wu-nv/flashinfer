"""Legacy -> unified: tests/attention/test_attention_sink.py

Each legacy test runs a ragged half (``BatchPrefillWithRaggedKVCacheWrapper``
with the AttentionSink JIT variant) and a paged half
(``BatchAttentionWithAttentionSinkWrapper``, page_size 1, contiguous and then
fragmented page ids).  The paged half converts: the legacy fixture runs
through ``PagedAttention`` with the ``(tokens, H, D)`` K/V viewed as
``(tokens, 1, H, D)`` NHD pages and the page ids as
``PagedAttentionMetadata.csr``.  The legacy ``backend`` axis (fa2 / fa3)
becomes the unified backends fa2, fa3, cudnn and trtllm-gen, so the ``fa2`` /
``fa3`` ids equal the legacy ids; a backend that cannot run a case on this GPU
is skipped with the resolver's reason (this replaces the legacy FA3 gate, and
covers fa2 / fa3 declining non-causal + sliding window).

Like the legacy paged runs, the plans request no LSE and use the legacy
128 MiB workspace; each run is checked against the legacy reference
(``sink_attention_unified``) at the legacy tolerance.  The sink-inclusive LSE
is checked against the fp32 oracle by
``test_paged_attention_features.py::test_attention_sinks``.
"""

import math
import random

import pytest
import torch
from tests.test_helpers.sink_attention_reference import sink_attention_unified

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_attention_sink.py"
LEGACY_MAP = [
    (
        "tests/attention/test_attention_sink.py::test_attention_sink",
        ["test_attention_sink"],
        "partial",
        "paged half only (contiguous + fragmented page ids, page_size 1) on fa2 / fa3 / "
        "cudnn / trtllm-gen, no LSE like the legacy run; legacy reference at the legacy "
        "tolerance; non-causal + window is skipped (fa2 / fa3 decline it) where the "
        "legacy passes; the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_incremental_generation",
        ["test_attention_sink_incremental_generation"],
        "partial",
        "paged half only: one PagedAttention per generation step and page-id layout, "
        "contiguous + fragmented page ids, on fa2 / fa3 / cudnn / trtllm-gen; legacy "
        "reference; non-causal + window skipped where the legacy passes; the ragged half "
        "is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_chunk_prefill",
        ["test_attention_sink_chunk_prefill"],
        "partial",
        "paged half only, contiguous + fragmented page ids, on fa2 / fa3 / cudnn / "
        "trtllm-gen; legacy reference; the legacy xfail for non-causal + window is kept; "
        "the ragged half is not paged",
    ),
    (
        "tests/attention/test_attention_sink.py::test_attention_sink_varlen",
        ["test_attention_sink_varlen"],
        "partial",
        "paged half only, the five legacy indptr configurations, contiguous + fragmented "
        "page ids, on fa2 / fa3 / cudnn / trtllm-gen; legacy reference; non-causal + "
        "window skipped where the legacy passes; the ragged half is not paged",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len", [1, 4, 16, 128])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
def test_attention_sink(
    dtype, batch_size, seq_len, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    head_dim = 128
    try:
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=1,
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            sinks=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    qo_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int32
    )

    q = torch.randn(
        batch_size * seq_len,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    k = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )
    v = torch.randn(
        batch_size * seq_len,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
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

    # paged KV, page_size 1: one NHD page per token
    kv_indices_host = torch.arange(
        0,
        batch_size * seq_len,
        dtype=torch.int32,
    )
    metadata = PagedAttentionMetadata.csr(
        qo_indptr_host.to(device),
        kv_indptr_host.diff().to(device),  # kv_seq_lens
        kv_indices_host.to(device),
        page_size=1,
        max_q_len=seq_len,
        max_kv_len=seq_len,
    )
    attn = PagedAttention(device, workspace_buffer=float_workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        use_sinks=True,
        backend=backend,
    )
    assert attn.backend == backend
    o_paged, _ = attn.run(
        q, (k.unsqueeze(1), v.unsqueeze(1)), sm_scale=sm_scale, sinks=sink
    )
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices (production scenario)
    total_pages = batch_size * seq_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create a fragmented page allocation pattern
        random.seed(42 + total_pages)  # Deterministic but varied seed
        all_pages = list(range(0, total_pages * 2))  # Larger page pool
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create new paged KV cache with larger capacity
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k[i]
            v_paged_frag[page_idx, 0] = v[i]

        # Test with fragmented indices
        metadata_frag = PagedAttentionMetadata.csr(
            qo_indptr_host.to(device),
            kv_indptr_host.diff().to(device),  # kv_seq_lens
            kv_indices_fragmented,
            page_size=1,
            max_q_len=seq_len,
            max_kv_len=seq_len,
        )
        attn_frag = PagedAttention(device, workspace_buffer=float_workspace_buffer)
        attn_frag.plan(
            metadata_frag,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_layout="NHD",
            causal=causal,
            window_left=window_left,
            use_sinks=True,
            backend=backend,
        )
        o_paged_frag, _ = attn_frag.run(
            q, (k_paged_frag, v_paged_frag), sm_scale=sm_scale, sinks=sink
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("initial_seq_len", [32, 128])
@pytest.mark.parametrize("num_generation_steps", [1, 2, 4])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("window_left", [-1, 128])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("backend", BACKENDS)
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
    """
    Test incremental generation scenario: q_len=1, kv_len grows gradually
    Simulate the token-by-token generation process in real large model inference
    """
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    head_dim = 128
    try:
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=1,
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            sinks=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    sm_scale = 1.0 / math.sqrt(head_dim)

    torch.manual_seed(42)

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # Initialize KV cache - simulate state after prefill phase
    k_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        batch_size, initial_seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    k_accumulated = v_accumulated = None
    # Simulate incremental generation process
    for step in range(num_generation_steps):
        current_kv_len = initial_seq_len + step

        # Current generated new token (q_len=1)
        q_new = torch.randn(
            batch_size, num_qo_heads, head_dim, dtype=dtype, device=device
        )

        # K,V for newly generated token
        k_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_new = torch.randn(
            batch_size, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Update KV cache
        if step == 0:
            k_cache_current = k_cache
            v_cache_current = v_cache
        else:
            k_cache_current = torch.cat([k_cache, k_accumulated], dim=1)
            v_cache_current = torch.cat([v_cache, v_accumulated], dim=1)

        # Calculate reference result
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

        # Set correct indptr: q_len=1 for each batch, kv_len=current_kv_len for each batch
        qo_indptr_host = torch.arange(
            0, batch_size + 1, dtype=torch.int32
        )  # [0, 1, 2, ..., batch_size]
        kv_indptr_host = torch.arange(
            0, batch_size * current_kv_len + 1, current_kv_len, dtype=torch.int32
        )

        # Convert to format expected by flashinfer [total_q_len, num_heads, head_dim]
        q_flashinfer = q_new.view(
            batch_size, num_qo_heads, head_dim
        )  # [batch_size, num_heads, head_dim]
        k_flashinfer = k_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )
        v_flashinfer = v_cache_current.view(
            batch_size * current_kv_len, num_kv_heads, head_dim
        )

        # paged KV, page_size 1: one NHD page per token
        kv_indices_host = torch.arange(
            0,
            batch_size * current_kv_len,
            dtype=torch.int32,
        )
        metadata = PagedAttentionMetadata.csr(
            qo_indptr_host.to(device),
            kv_indptr_host.diff().to(device),  # kv_seq_lens
            kv_indices_host.to(device),
            page_size=1,
            max_q_len=1,
            max_kv_len=current_kv_len,
        )
        attn = PagedAttention(device, workspace_buffer=float_workspace_buffer)
        attn.plan(
            metadata,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_layout="NHD",
            causal=causal,
            window_left=window_left,
            use_sinks=True,
            backend=backend,
        )
        assert attn.backend == backend
        o_paged, _ = attn.run(
            q_flashinfer,
            (k_flashinfer.unsqueeze(1), v_flashinfer.unsqueeze(1)),
            sm_scale=sm_scale,
            sinks=sink,
        )
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

        # Test with non-contiguous KV indices for incremental generation
        total_pages = batch_size * current_kv_len
        if total_pages > 1:  # Only test fragmentation when we have multiple pages
            # Create fragmented page allocation pattern
            random.seed(42 + step + current_kv_len)  # Vary seed with step and length
            all_pages = list(range(0, total_pages * 2))
            occupied_pages = set(
                random.sample(all_pages, min(total_pages, len(all_pages) // 2))
            )
            available_pages = [p for p in all_pages if p not in occupied_pages]

            # Allocate non-contiguous pages
            kv_indices_fragmented = torch.tensor(
                available_pages[:total_pages], dtype=torch.int32, device=device
            )

            # Create fragmented paged KV cache
            k_paged_frag = torch.randn(
                total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
            )
            v_paged_frag = torch.randn(
                total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
            )

            # Copy K,V data to fragmented pages
            for i, page_idx in enumerate(kv_indices_fragmented):
                k_paged_frag[page_idx, 0] = k_flashinfer[i]
                v_paged_frag[page_idx, 0] = v_flashinfer[i]

            # Test with fragmented indices
            metadata_frag = PagedAttentionMetadata.csr(
                qo_indptr_host.to(device),
                kv_indptr_host.diff().to(device),  # kv_seq_lens
                kv_indices_fragmented,
                page_size=1,
                max_q_len=1,
                max_kv_len=current_kv_len,
            )
            attn_frag = PagedAttention(device, workspace_buffer=float_workspace_buffer)
            attn_frag.plan(
                metadata_frag,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_dtype=dtype,
                kv_layout="NHD",
                causal=causal,
                window_left=window_left,
                use_sinks=True,
                backend=backend,
            )
            o_paged_frag, _ = attn_frag.run(
                q_flashinfer,
                (k_paged_frag, v_paged_frag),
                sm_scale=sm_scale,
                sinks=sink,
            )

            # Verify fragmented result matches reference
            if dtype == torch.float16:
                torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
            else:
                torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)

        # Accumulate new K,V for next step
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
@pytest.mark.parametrize("backend", BACKENDS)
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
    """
    Test chunk prefill scenario: q_len != kv_len and q_len > 1
    Simulate chunk-based processing of long sequences where current chunk
    attends to all historical tokens plus current chunk tokens
    """
    if not causal and window_left >= 0:
        # xfail for non-causal + sliding window case
        pytest.xfail(
            "NOTE(Zihao): attention sink with sliding window and non-causal will fail after https://github.com/flashinfer-ai/flashinfer/pull/1661, temporarily xfail the test."
        )
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    head_dim = 128
    try:
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=1,
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            sinks=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    # Skip invalid combinations
    if chunk_size >= historical_len:
        pytest.skip(
            "chunk_size should be smaller than historical_len for meaningful chunk prefill test"
        )

    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    total_kv_len = historical_len + chunk_size

    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    # Create input tensors for chunk prefill scenario
    # q represents current chunk: [batch_size * chunk_size, num_heads, head_dim]
    q_chunk = torch.randn(
        batch_size * chunk_size, num_qo_heads, head_dim, dtype=dtype, device=device
    )

    # k, v represent all tokens (historical + current chunk)
    k_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_all = torch.randn(
        batch_size * total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    # Calculate reference result using chunk prefill mode
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

    # Set up indices for chunk prefill
    qo_indptr_host = torch.arange(
        0, batch_size * chunk_size + 1, chunk_size, dtype=torch.int32
    )
    kv_indptr_host = torch.arange(
        0, batch_size * total_kv_len + 1, total_kv_len, dtype=torch.int32
    )

    # paged KV, page_size 1: one NHD page per token
    kv_indices_host = torch.arange(
        0,
        batch_size * total_kv_len,
        dtype=torch.int32,
    )
    metadata = PagedAttentionMetadata.csr(
        qo_indptr_host.to(device),
        kv_indptr_host.diff().to(device),  # kv_seq_lens
        kv_indices_host.to(device),
        page_size=1,
        max_q_len=chunk_size,
        max_kv_len=total_kv_len,
    )
    attn = PagedAttention(device, workspace_buffer=float_workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        use_sinks=True,
        backend=backend,
    )
    assert attn.backend == backend
    o_paged, _ = attn.run(
        q_chunk, (k_all.unsqueeze(1), v_all.unsqueeze(1)), sm_scale=sm_scale, sinks=sink
    )
    if dtype == torch.float16:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_paged, o_ref, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices for chunk prefill
    total_pages = batch_size * total_kv_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create fragmented page allocation pattern
        random.seed(
            42 + batch_size + total_kv_len
        )  # Vary seed with batch and total length
        all_pages = list(range(0, total_pages * 2))
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create fragmented paged KV cache
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k_all[i]
            v_paged_frag[page_idx, 0] = v_all[i]

        # Test with fragmented indices
        metadata_frag = PagedAttentionMetadata.csr(
            qo_indptr_host.to(device),
            kv_indptr_host.diff().to(device),  # kv_seq_lens
            kv_indices_fragmented,
            page_size=1,
            max_q_len=chunk_size,
            max_kv_len=total_kv_len,
        )
        attn_frag = PagedAttention(device, workspace_buffer=float_workspace_buffer)
        attn_frag.plan(
            metadata_frag,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_layout="NHD",
            causal=causal,
            window_left=window_left,
            use_sinks=True,
            backend=backend,
        )
        o_paged_frag, _ = attn_frag.run(
            q_chunk, (k_paged_frag, v_paged_frag), sm_scale=sm_scale, sinks=sink
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_paged_frag, o_ref, rtol=1e-2, atol=1e-2)


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
@pytest.mark.parametrize("backend", BACKENDS)
def test_attention_sink_varlen(
    dtype, indptr_config, num_qo_heads, num_kv_heads, window_left, causal, backend
):
    """
    Test variable length sequences within a batch.
    Each request in the batch can have different query and key/value lengths.
    """
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    head_dim = 128
    try:
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=1,
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            sinks=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    # Unpack the indptr configuration
    qo_indptr, kv_indptr, description = indptr_config

    # Validate that qo_indptr and kv_indptr have same batch size
    if len(qo_indptr) != len(kv_indptr):
        pytest.skip(
            f"qo_indptr and kv_indptr must have same batch size for {description}"
        )

    batch_size = len(qo_indptr) - 1
    total_qo_len = qo_indptr[-1]
    total_kv_len = kv_indptr[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)

    # Check if any request has qo_len > kv_len for causal case
    if causal:
        for i in range(batch_size):
            qo_len_i = qo_indptr[i + 1] - qo_indptr[i]
            kv_len_i = kv_indptr[i + 1] - kv_indptr[i]
            if qo_len_i > kv_len_i:
                pytest.skip(
                    "qo_len > kv_len not supported for causal attention in varlen mode"
                )

    # Create input tensors
    q = torch.randn(total_qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    qo_indptr_tensor = torch.tensor(qo_indptr, dtype=torch.int32, device=device)
    kv_indptr_tensor = torch.tensor(kv_indptr, dtype=torch.int32, device=device)

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    # Test the variable length reference implementation
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

    # Verify output shape
    assert o_ref.shape == (
        total_qo_len,
        num_qo_heads,
        head_dim,
    ), f"Expected shape ({total_qo_len}, {num_qo_heads}, {head_dim}), got {o_ref.shape}"

    # Create workspace buffer
    float_workspace_buffer = torch.empty(
        128 * 1024 * 1024, dtype=torch.uint8, device=device
    )

    max_q_len = max(qo_indptr[i + 1] - qo_indptr[i] for i in range(batch_size))
    max_kv_len = max(kv_indptr[i + 1] - kv_indptr[i] for i in range(batch_size))

    # paged KV, page_size 1: one NHD page per token
    kv_indices_host = torch.arange(0, total_kv_len, dtype=torch.int32, device=device)
    metadata = PagedAttentionMetadata.csr(
        qo_indptr_tensor,
        kv_indptr_tensor.diff(),  # kv_seq_lens
        kv_indices_host,
        page_size=1,
        max_q_len=max_q_len,
        max_kv_len=max_kv_len,
    )
    attn = PagedAttention(device, workspace_buffer=float_workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        use_sinks=True,
        backend=backend,
    )
    assert attn.backend == backend
    o_paged, _ = attn.run(
        q, (k.unsqueeze(1), v.unsqueeze(1)), sm_scale=sm_scale, sinks=sink
    )
    if dtype == torch.float16:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-3, atol=1e-3)
    else:
        torch.testing.assert_close(o_ref, o_paged, rtol=1e-2, atol=1e-2)

    # Test with non-contiguous KV indices for variable length sequences
    total_pages = total_kv_len
    if total_pages > 1:  # Only test fragmentation when we have multiple pages
        # Create fragmented page allocation pattern
        random.seed(
            42 + batch_size + total_kv_len
        )  # Vary seed with batch and total length
        all_pages = list(range(0, total_pages * 2))
        occupied_pages = set(
            random.sample(all_pages, min(total_pages, len(all_pages) // 2))
        )
        available_pages = [p for p in all_pages if p not in occupied_pages]

        # Allocate non-contiguous pages
        kv_indices_fragmented = torch.tensor(
            available_pages[:total_pages], dtype=torch.int32, device=device
        )

        # Create fragmented paged KV cache
        k_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        v_paged_frag = torch.randn(
            total_pages * 2, 1, num_kv_heads, head_dim, dtype=dtype, device=device
        )

        # Copy K,V data to fragmented pages
        for i, page_idx in enumerate(kv_indices_fragmented):
            k_paged_frag[page_idx, 0] = k[i]
            v_paged_frag[page_idx, 0] = v[i]

        # Test with fragmented indices
        metadata_frag = PagedAttentionMetadata.csr(
            qo_indptr_tensor,
            kv_indptr_tensor.diff(),  # kv_seq_lens
            kv_indices_fragmented,
            page_size=1,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
        )
        attn_frag = PagedAttention(device, workspace_buffer=float_workspace_buffer)
        attn_frag.plan(
            metadata_frag,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_layout="NHD",
            causal=causal,
            window_left=window_left,
            use_sinks=True,
            backend=backend,
        )
        o_paged_frag, _ = attn_frag.run(
            q, (k_paged_frag, v_paged_frag), sm_scale=sm_scale, sinks=sink
        )

        # Verify fragmented result matches reference
        if dtype == torch.float16:
            torch.testing.assert_close(o_ref, o_paged_frag, rtol=1e-3, atol=1e-3)
        else:
            torch.testing.assert_close(o_ref, o_paged_frag, rtol=1e-2, atol=1e-2)
