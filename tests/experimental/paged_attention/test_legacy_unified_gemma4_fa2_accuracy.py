"""Legacy -> unified: tests/attention/test_gemma4_fa2_accuracy.py

``test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch`` runs the
legacy fixture (imported from the legacy module: seed 42, Gemma-4
full-attention shape H16:2, head_dim 512, e4m3 KV cache with
``k_scale = v_scale = 0.02``, bf16 q / 16, ``sm_scale = 1``, KV 10003 tokens,
page 16, NHD) through ``PagedAttention`` on fa2, fa3, cudnn and trtllm-gen.
A backend that cannot run it on this GPU is skipped with the resolver's
reason.  The legacy test pins fa2.  Like the legacy run it is causal, returns
a base-2 LSE and uses a 512 MiB workspace, and it checks the legacy fp32
torch reference (output and LSE) at the legacy budget.

The legacy tensor-core decode test (``BatchDecodeWithPagedKVCacheWrapper``)
is out of scope: ``PagedAttention`` is the paged-prefill API and has no
decode route.
"""

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_gemma4_fa2_accuracy.py"
LEGACY_MAP = [
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch"],
        "equivalent",
        "legacy fixture (q17, KV10003, H16:2, D512, page16, NHD, e4m3 KV, scales .02) on "
        "fa2 / fa3 / cudnn / trtllm-gen; legacy fp32 torch reference (output and LSE) "
        "at the legacy budget",
    ),
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch",
        [],
        "out-of-scope",
        "decode entry (BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores=True); "
        "PagedAttention has no decode route",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch(backend):
    from tests.attention import test_gemma4_fa2_accuracy as legacy

    try:
        resolve_paged_attention(
            device=legacy.DEVICE,
            num_qo_heads=legacy.NUM_QO_HEADS,
            num_kv_heads=legacy.NUM_KV_HEADS,
            head_dim_qk=legacy.HEAD_DIM,
            q_dtype=legacy.Q_DTYPE,
            kv_dtype=legacy.KV_DTYPE,
            page_size=legacy.PAGE_SIZE,
            kv_layout="NHD",
            causal=True,
            need_lse=True,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    q_len = 17
    q, k_cache, v_cache, num_pages = legacy._make_inputs(q_len)

    # the legacy CSR: one request on pages 0 .. num_pages-1
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device=legacy.DEVICE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=legacy.DEVICE)
    kv_seq_lens = torch.tensor([legacy.KV_LEN], dtype=torch.int32, device=legacy.DEVICE)
    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        kv_seq_lens,
        kv_indices,
        page_size=legacy.PAGE_SIZE,
        max_q_len=q_len,
        max_kv_len=legacy.KV_LEN,
    )

    workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=legacy.DEVICE)
    attn = PagedAttention(workspace)
    attn.plan(
        metadata,
        num_qo_heads=legacy.NUM_QO_HEADS,
        num_kv_heads=legacy.NUM_KV_HEADS,
        head_dim_qk=legacy.HEAD_DIM,
        q_dtype=legacy.Q_DTYPE,
        kv_dtype=legacy.KV_DTYPE,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        backend=backend,
    )
    assert attn.backend == backend
    out, lse = attn.run(
        q,
        (k_cache, v_cache),
        sm_scale=legacy.SM_SCALE,
        k_scale=legacy.K_SCALE,
        v_scale=legacy.V_SCALE,
    )

    ref_out, ref_lse = legacy._reference(q, k_cache, v_cache)
    legacy._assert_matches_reference(out, lse, ref_out, ref_lse)
