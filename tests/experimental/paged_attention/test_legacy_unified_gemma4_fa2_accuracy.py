"""Legacy -> unified: tests/attention/test_gemma4_fa2_accuracy.py

``test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch`` runs the
legacy fixture (imported from the legacy module: seed 42, Gemma-4
full-attention shape H16:2, head_dim 512, e4m3 KV cache with
``k_scale = v_scale = 0.02``, bf16 q / 16, ``sm_scale = 1``, KV 10003 tokens,
page 16, NHD) through ``PagedAttention`` pinned to fa2, the backend the legacy
prefill wrapper names.  It checks the legacy torch reference at the legacy
budget, and the output and LSE against the fp32 paged-attention oracle on the
dequantized cache.

The legacy tensor-core decode test (``BatchDecodeWithPagedKVCacheWrapper``)
is out of scope: ``PagedAttention`` is the paged-prefill API and has no
decode route.
"""

import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_gemma4_fa2_accuracy.py"
LEGACY_MAP = [
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch"],
        "equivalent",
        "legacy fixture (q17, KV10003, H16:2, D512, page16, NHD, e4m3 KV, scales .02) on "
        "fa2; legacy torch reference at the legacy budget plus the fp32 oracle on the "
        "dequantized cache (output and LSE)",
    ),
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch",
        [],
        "out-of-scope",
        "decode entry (BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores=True); "
        "PagedAttention has no decode route",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch():
    from tests.attention import test_gemma4_fa2_accuracy as legacy

    legacy._require_ampere_or_newer()
    q_len = 17
    q, k_cache, v_cache, num_pages = legacy._make_inputs(q_len)
    device = q.device

    # the legacy CSR (one request, arange page ids) as unified CSR metadata
    qo_indptr_cpu = torch.tensor([0, q_len], dtype=torch.int32)
    kv_lens_cpu = torch.tensor([legacy.KV_LEN], dtype=torch.int32)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    md = PagedAttentionMetadata.csr(
        qo_indptr_cpu.to(device),
        kv_lens_cpu.to(device),
        kv_indices,
        page_size=legacy.PAGE_SIZE,
        max_q_len=q_len,
        max_kv_len=legacy.KV_LEN,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )

    attn = PagedAttention(device)
    attn.plan(
        md,
        num_qo_heads=legacy.NUM_QO_HEADS,
        num_kv_heads=legacy.NUM_KV_HEADS,
        head_dim_qk=legacy.HEAD_DIM,
        q_dtype=legacy.Q_DTYPE,
        kv_dtype=legacy.KV_DTYPE,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(
        q,
        (k_cache, v_cache),
        sm_scale=legacy.SM_SCALE,
        k_scale=legacy.K_SCALE,
        v_scale=legacy.V_SCALE,
    )

    # legacy reference at the legacy budget (out 2e-2; LSE rtol 1e-3, atol 2e-2)
    ref_out, ref_lse = legacy._reference(q, k_cache, v_cache)
    legacy._assert_matches_reference(out, lse, ref_out, ref_lse)

    # fp32 oracle on the dequantized cache: output and base-2 LSE
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache.float() * legacy.K_SCALE,
        v_cache.float() * legacy.V_SCALE,
        qo_indptr_cpu,
        kv_lens_cpu,
        None,
        legacy.PAGE_SIZE,
        True,
        sm_scale=legacy.SM_SCALE,
        kv_layout="NHD",
        kv_page_indices=kv_indices,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
