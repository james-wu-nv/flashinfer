"""Legacy -> unified: tests/attention/test_attention_sink_blackwell.py

``test_blackwell_trtllm_gen_context_attention_sink`` runs the legacy fixture
(same grid, same seed, same tensors) through ``PagedAttention`` with
attention sinks, once per backend: fa2, fa3, cudnn and trtllm-gen.  A backend
that cannot run the fixture on this GPU is skipped with the resolver's
reason.  The legacy test calls ``trtllm_batch_context_with_kv_cache``, so the
trtllm-gen row runs the legacy kernel.

Like the legacy call, the plan requests no LSE (``lse_mode="none"``).  Each
case checks the legacy reference (``sink_attention_unified``) at the legacy
tolerance.  The sink-inclusive LSE is checked against the fp32 oracle by
``test_paged_attention_features.py::test_attention_sinks``.

The legacy decode test (``trtllm_batch_decode_with_kv_cache``) is out of scope:
``PagedAttention`` is the paged-prefill API and has no decode route.
"""

import einops
import pytest
import torch
from tests.attention.test_helpers.sink_attention_reference import sink_attention_unified

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_attention_sink_blackwell.py"
LEGACY_MAP = [
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_decode_attention_sink",
        [],
        "out-of-scope",
        "decode entry (trtllm_batch_decode_with_kv_cache); PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_context_attention_sink",
        ["test_blackwell_trtllm_gen_context_attention_sink"],
        "equivalent",
        "same grid, seed and tensors on fa2 / fa3 / cudnn / trtllm-gen with no "
        "LSE, like the legacy call; the legacy reference at the legacy tolerance",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_context_attention_sink(
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    backend,
):
    try:
        resolve_paged_attention(
            device=torch.device("cuda"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=page_size,
            kv_layout="HND",
            causal=True,
            need_lse=False,
            sinks=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    seed = 0
    torch.manual_seed(seed)
    device = "cuda:0"

    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()

    # Generate unique block IDs for all sequences
    block_tables = torch.arange(
        (batch_size * max_num_blocks_per_seq), dtype=torch.int32, device=device
    ).reshape(batch_size, max_num_blocks_per_seq)

    # Create separate K and V caches
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        num_tokens,
        num_qo_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )

    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    q_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )

    metadata = PagedAttentionMetadata.dense(
        q_indptr,
        seq_lens,
        block_tables,
        page_size=page_size,
        max_q_len=seq_len,
        max_kv_len=seq_len,
    )
    attn = PagedAttention(workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="HND",
        causal=True,
        window_left=-1,
        lse_mode="none",
        use_sinks=True,
        backend=backend,
    )
    assert attn.backend == backend
    output, _ = attn.run(q, (k_cache, v_cache), sm_scale=1.0, sinks=sink)

    k = einops.rearrange(
        k_cache,
        "num_pages h p d -> (num_pages p) h d",
    )
    v = einops.rearrange(
        v_cache,
        "num_pages h p d -> (num_pages p) h d",
    )

    o_ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        -1,
        True,
        1.0,
        mode="prefill",
        batch_size=batch_size,
    )

    if dtype == torch.float16:
        atol, rtol = 2e-3, 1e-3
    elif dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    torch.testing.assert_close(o_ref, output, atol=atol, rtol=rtol)
