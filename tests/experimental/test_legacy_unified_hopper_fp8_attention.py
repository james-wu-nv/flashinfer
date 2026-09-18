"""Legacy -> unified: tests/attention/test_hopper_fp8_attention.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only (SM90a-gated); the
unified file is in no default lane (tests/experimental is excluded by
norecursedirs).

The three paged legacy functions quantize the QUERY (and K/V) to fp8 (e4m3 /
e5m2) with per-head or per-tensor SCALE TENSORS, ask the SM90 fp8 kernel for
a fp16 output and check a MSE budget against the fp16 fa3 wrapper.  None of
that has a unified spelling:

- fp8 q: every backend declares q_dtypes = {fp16, bf16}
  (flashinfer/experimental/paged_attention/_backends/_capabilities.py); fp8 is
  a KV-only axis, dequantized by the per-tensor host-float k_scale / v_scale
  of run().  resolve() rejects q_dtype float8 for every backend with
  "unsupported q dtype" (asserted here with backend="auto", so the message
  lists every candidate; on B200 fa3 is additionally excluded by compute
  capability).  Needs a quantization descriptor (q dtype, q_scale, o_dtype)
  -- EXPECT_FP8_Q + EXPECT_OUTPUT_DTYPE (design doc
  docs/design_docs/paged_attention_unified_lifecycle.md, "Quantization").
- per-head / device-tensor scales: run() takes host floats so the call
  stays sync-free (EXPECT_DEVICE_SCALES); per-head scales have no spelling.

Each unsupported row keeps the legacy grid and asserts the rejection on the
legacy fixture's static configuration; the positive branch cannot be written
until the descriptor exists (pytest.fail with the porting instruction).

H100 command (repo root): the rows assert the same rejection there (fa3
excluded by q dtype instead of compute capability):

    python -m pytest -q -ra -o faulthandler_timeout=300 \\
        tests/experimental/test_legacy_unified_hopper_fp8_attention.py

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_batch_prefill_paged, test_batch_prefill_paged_gqa,
  test_batch_prefill_paged_scale_types: fp8 q + scale tensors + o_dtype, see
  above (unsupported-by-design).
- test_single_prefill, test_block_sparse_attention (single_prefill /
  BlockSparseAttentionWrapper), test_batch_prefill_ragged (ragged),
  test_batch_decode_paged (BatchDecodeWithPagedKVCacheWrapper): out-of-scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_DEVICE_SCALES,
    EXPECT_FP8_Q,
    check_legacy_map,
    check_legacy_map_complete,
    gated,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_hopper_fp8_attention.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_hopper_fp8_attention.py::test_single_prefill",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_block_sparse_attention",
        [],
        "out-of-scope",
        "BlockSparseAttentionWrapper (BSR mask), not paged prefill",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_ragged",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged",
        ["test_batch_prefill_paged"],
        "unsupported-by-design",
        "same grid (B2/4, H8/32, D64/128/256, causal, e4m3/e5m2): fp8 QUERY with "
        "per-head q/k/v scale tensors and a fp16 output on the SM90 fp8 kernel; the "
        "unified envelope admits fp16/bf16 q only (fp8 is a KV-only axis with host-float "
        "per-tensor scales), so resolve() rejects q_dtype float8 for every backend "
        "(EXPECT_FP8_Q, EXPECT_OUTPUT_DTYPE); the legacy MSE < 1.0 assertion has no "
        "unified counterpart until a quantization descriptor exists",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged_gqa",
        ["test_batch_prefill_paged_gqa"],
        "unsupported-by-design",
        "same grid (B2, H32:8 / 16:4 / 8:2, D128, causal, e4m3): the fp8-q kernel with "
        "GQA head ratios; same gap (EXPECT_FP8_Q)",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_decode_paged",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper (decode-only entry)",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged_scale_types",
        ["test_batch_prefill_paged_scale_types"],
        "unsupported-by-design",
        "same grid (per_head / per_tensor, e4m3): per-head vs per-tensor fp8 scale "
        "TENSORS for q, k and v; unified run() takes host-float per-tensor k_scale / "
        "v_scale only and no q_scale (EXPECT_FP8_Q, EXPECT_DEVICE_SCALES)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def _assert_fp8_q_rejected(
    *, num_qo_heads, num_kv_heads, head_dim, causal, dtype, flag=EXPECT_FP8_Q
):
    """The legacy fixture's static configuration with a fp8 q: no unified
    backend admits it (backend="auto" lists every candidate's reason)."""
    res = gated(
        flag,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_dtype=dtype,
            page_size=16,  # the legacy page size
            kv_layout="NHD",
            causal=causal,
            need_lse=False,
            kv_input_form="page_indices",
            backend="auto",
        ),
        match="unsupported q dtype",
    )
    if res is None:
        return
    pytest.fail(
        "EXPECT_FP8_Q flipped: port the legacy per_head_symmetric_quant fixture "
        "(tests/attention/test_hopper_fp8_attention.py) here with the quantization "
        "descriptor (q_scale / o_dtype)"
    )


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_paged(batch_size, num_heads, head_dim, causal, dtype):
    _assert_fp8_q_rejected(
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
    )


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_gqa(
    batch_size, num_qo_heads, num_kv_heads, head_dim, causal, dtype
):
    _assert_fp8_q_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        dtype=dtype,
    )


@pytest.mark.parametrize("scale_type", ["per_head", "per_tensor"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_scale_types(scale_type, dtype):
    """Per-head ([H]) and per-tensor ([1]) fp8 scale tensors: beyond the fp8 q
    gap, run() has no tensor-valued scales (EXPECT_DEVICE_SCALES) and no
    per-head scale at all; both flags must flip for this row."""
    _assert_fp8_q_rejected(
        num_qo_heads=8,
        num_kv_heads=8,
        head_dim=128,
        causal=True,
        dtype=dtype,
        flag=EXPECT_FP8_Q and EXPECT_DEVICE_SCALES,
    )
