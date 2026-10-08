"""Legacy -> unified: tests/attention/test_gemma4_fa2_accuracy.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture, imported as is (seed 42, Gemma-4 full-attention
shape: 16 query heads, 2 KV heads, head_dim 512, fp8 e4m3 KV cache with
``k_scale = v_scale = 0.02``, bf16 q / 16, ``sm_scale = 1``, KV 10003 tokens,
page 16, NHD), through flashinfer.prefill.PagedAttention on fa2 and asserts
the legacy torch fp32 oracle at the legacy budget (out rtol/atol 2e-2, LSE
rtol 1e-3 / atol 2e-2) and the suite's fp32 oracle on the dequantized cache.
The added ``backend`` axis records which backends declare the shape.

CI: legacy file only in the H100 1/5 sampling lane (SM80+ gate).  The
unified file is not collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- (512, 512) with an fp8 KV cache is declared on fa2 since WP-T
  (EXPECT_FA2_HEAD_DIM_512), so both rows run the positive branch; cuDNN,
  trtllm-gen and cake skip (fp8 KV and D512 undeclared for them).
- ``test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch``: the
  legacy subject is the decode wrapper's tensor-core path (``use_tensor_cores
  =True`` runs the prefill kernel at q_len 1); the unified row is the same
  fixture as a q_len 1 prefill batch, so it is ``partial`` (same numerics,
  different entry point).
"""

import pytest
import torch

from tests.attention import test_gemma4_fa2_accuracy as legacy

from .legacy_unified_helpers import (
    EXPECT_FA2_HEAD_DIM_512,
    LegacyBatch,
    assert_every_backend_excluded,
    assert_oracle,
    backend_rows,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    run_batch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_gemma4_fa2_accuracy.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch"],
        "equivalent",
        "legacy fixture imported as is (q17, KV10003, H16:2, D512, page16, NHD, e4m3 KV, "
        "sm_scale 1, k=v_scale .02) and the legacy torch oracle + budget on fa2, plus "
        "the suite oracle on the dequantized cache; D512 fp8 on fa2 is declared since "
        "WP-T (EXPECT_FA2_HEAD_DIM_512); other backends skip (undeclared).",
    ),
    (
        "tests/attention/test_gemma4_fa2_accuracy.py::test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch",
        ["test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch"],
        "partial",
        "same fixture with q_len 1; the legacy subject is the tensor-core decode wrapper "
        "(the prefill kernel at q_len 1), the unified row the same batch as a q_len 1 "
        "prefill on fa2 at the legacy budget + oracle.",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


def _gemma4_unified(q_len, backend):
    legacy._require_ampere_or_newer()
    if not EXPECT_FA2_HEAD_DIM_512:
        assert_every_backend_excluded(
            "unsupported head dims (512, 512)",
            also=("unsupported kv dtype",),
            num_qo_heads=legacy.NUM_QO_HEADS,
            num_kv_heads=legacy.NUM_KV_HEADS,
            head_dim_qk=legacy.HEAD_DIM,
            q_dtype=legacy.Q_DTYPE,
            kv_dtype=legacy.KV_DTYPE,
            page_size=legacy.PAGE_SIZE,
            kv_layout="NHD",
            causal=True,
            need_lse=True,
        )
        return
    q, k_cache, v_cache, num_pages = legacy._make_inputs(q_len)
    qo_indptr, kv_indptr, kv_indices, kv_last_page_len = legacy._paged_metadata(
        q_len, num_pages
    )
    lb = LegacyBatch(
        q=q,
        k=k_cache,
        v=v_cache,
        q_indptr_cpu=qo_indptr.cpu(),
        kv_indptr_cpu=kv_indptr.cpu(),
        kv_indices_cpu=kv_indices.cpu(),
        last_page_len_cpu=kv_last_page_len.cpu(),
        page_size=legacy.PAGE_SIZE,
        kv_layout="NHD",
    )
    md = lb.metadata()
    assert int(md.kv_seq_lens_cpu[0]) == legacy.KV_LEN
    _, out, lse = run_batch(
        lb,
        md,
        backend,
        causal=True,
        sm_scale=legacy.SM_SCALE,
        k_scale=legacy.K_SCALE,
        v_scale=legacy.V_SCALE,
    )
    ref_out, ref_lse = legacy._reference(q, k_cache, v_cache)
    legacy._assert_matches_reference(out, lse, ref_out, ref_lse)  # legacy budget
    assert_oracle(
        lb,
        out,
        lse,
        causal=True,
        sm_scale=legacy.SM_SCALE,
        k=k_cache.float() * legacy.K_SCALE,
        v=v_cache.float() * legacy.V_SCALE,
    )


@pytest.mark.parametrize("backend", backend_rows())
def test_gemma4_fp8_kv_head_dim_512_chunked_prefill_matches_torch(backend):
    _gemma4_unified(17, backend)


@pytest.mark.parametrize("backend", backend_rows())
def test_gemma4_fp8_kv_head_dim_512_tensor_core_decode_matches_torch(backend):
    _gemma4_unified(1, backend)
