"""Legacy -> unified: tests/attention/test_attention_sink_blackwell.py

``test_blackwell_trtllm_gen_context_attention_sink`` runs the legacy fixture
through ``PagedAttention`` pinned to trtllm-gen, the backend the legacy test
calls (``trtllm_batch_context_with_kv_cache``).  Same grid, same seed, same
tensors, so the node ids equal the legacy ids.  Each case checks the legacy
reference (``sink_attention_unified``) at the legacy tolerance, and the output
and LSE against the fp32 paged-attention oracle.

The legacy decode test (``trtllm_batch_decode_with_kv_cache``) is out of scope:
``PagedAttention`` is the paged-prefill API and has no decode route.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import get_compute_capability

from .paged_attention_reference import reference_paged_prefill

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
        "same grid, seed and tensors on trtllm-gen; legacy reference at the legacy "
        "tolerance plus the fp32 oracle (output and LSE)",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _legacy_tol(dtype):
    if dtype == torch.float16:
        return dict(atol=2e-3, rtol=1e-3)
    return dict(atol=1e-2, rtol=1e-2)


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
):
    import einops
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    device = torch.device("cuda:0")
    if get_compute_capability(device)[0] != 10:
        pytest.skip("trtllm-gen runs on SM100 / SM103 only")

    # the legacy fixture, verbatim (same RNG order: q, k, v, sink)
    torch.manual_seed(0)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    pages_per_seq = (seq_len + page_size - 1) // page_size
    block_tables = torch.arange(
        batch_size * pages_per_seq, dtype=torch.int32, device=device
    ).reshape(batch_size, pages_per_seq)
    num_tokens = seq_len * batch_size
    num_pages = (num_tokens + page_size - 1) // page_size
    q = torch.randn(num_tokens, num_qo_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.randn(
        num_pages, num_kv_heads, page_size, head_dim, dtype=dtype, device=device
    )
    sink = torch.rand(num_qo_heads, device=device, dtype=torch.float32) * 5

    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * seq_len
    kv_lens_cpu = seq_lens.cpu()
    md = PagedAttentionMetadata.dense(
        qo_indptr_cpu.to(device),
        seq_lens,
        block_tables,
        page_size=page_size,
        max_q_len=seq_len,
        max_kv_len=seq_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )

    attn = PagedAttention(device)
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
        use_sinks=True,
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"
    out, lse = attn.run(q, (k_cache, v_cache), sm_scale=1.0, sinks=sink)

    # legacy reference at the legacy tolerance
    k = einops.rearrange(k_cache, "num_pages h p d -> (num_pages p) h d")
    v = einops.rearrange(v_cache, "num_pages h p d -> (num_pages p) h d")
    o_ref = sink_attention_unified(
        q, k, v, sink, -1, True, 1.0, mode="prefill", batch_size=batch_size
    )
    torch.testing.assert_close(o_ref, out, **_legacy_tol(dtype))

    # fp32 oracle: output and base-2 LSE (the LSE includes the sink)
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        kv_lens_cpu,
        block_tables,
        page_size,
        True,
        sm_scale=1.0,
        kv_layout="HND",
        sinks=sink,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
