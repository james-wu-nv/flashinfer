"""Legacy -> unified: tests/attention/test_hopper_fp8_attention.py

The three paged legacy functions run the SM90 fp8 fa3 kernel: an fp8 (e4m3 /
e5m2) QUERY and K/V with per-head or per-tensor scale TENSORS and an fp16
output, checked by an MSE budget against the fp16 fa3 wrapper.
``PagedAttention`` has no spelling for that: every backend admits fp16 / bf16
q only (fp8 is a KV-only axis, dequantized by host-float per-tensor
``k_scale`` / ``v_scale``), there is no ``q_scale``, no per-head scale and no
output dtype.  So each legacy id pins fa3, the backend the legacy calls, on
the legacy static configuration (page 16, NHD, page-index form) and asserts
the clean "unsupported q dtype" rejection.  The legacy SM90a gate is kept, so
on other parts these rows skip like the legacy.

Single prefill, block-sparse, ragged and decode functions are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention
from flashinfer.utils import is_sm90a_supported

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_hopper_fp8_attention.py"
LEGACY_MAP = [
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
        "same grid and ids; fp8 q with per-head scale tensors and an fp16 output on "
        "fa3 has no unified spelling: pinned fa3 rejects q_dtype float8 (SM90a-gated "
        "like the legacy)",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged_gqa",
        ["test_batch_prefill_paged_gqa"],
        "unsupported-by-design",
        "same grid and ids (GQA 32:8 / 16:4 / 8:2); same fp8 q gap, pinned fa3 "
        "rejects q_dtype float8 (SM90a-gated like the legacy)",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_decode_paged",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper (decode entry); PagedAttention has no "
        "decode route",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged_scale_types",
        ["test_batch_prefill_paged_scale_types"],
        "unsupported-by-design",
        "same grid and ids; per-head / per-tensor fp8 scale tensors for q, k and v: "
        "run() takes host-float per-tensor k_scale / v_scale and no q_scale; pinned "
        "fa3 rejects q_dtype float8 (SM90a-gated like the legacy)",
    ),
]


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_paged(batch_size, num_heads, head_dim, causal, dtype):
    device = torch.device("cuda:0")
    if not is_sm90a_supported(device):
        pytest.skip("SM90A is not supported")

    with pytest.raises(ValueError, match="fa3: unsupported q dtype"):
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_dtype=dtype,
            page_size=16,
            kv_layout="NHD",
            causal=causal,
            kv_input_form="page_indices",
            backend="fa3",
        )


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_gqa(
    batch_size, num_qo_heads, num_kv_heads, head_dim, causal, dtype
):
    device = torch.device("cuda:0")
    if not is_sm90a_supported(device):
        pytest.skip("SM90A is not supported")

    with pytest.raises(ValueError, match="fa3: unsupported q dtype"):
        resolve_paged_attention(
            device=device,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_dtype=dtype,
            page_size=16,
            kv_layout="NHD",
            causal=causal,
            kv_input_form="page_indices",
            backend="fa3",
        )


@pytest.mark.parametrize("scale_type", ["per_head", "per_tensor"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_scale_types(scale_type, dtype):
    device = torch.device("cuda:0")
    if not is_sm90a_supported(device):
        pytest.skip("SM90A is not supported")

    # legacy: B2, H8:8, D128, page 16, causal; q / k / v scale tensors of
    # shape [H] (per_head) or [1] (per_tensor) -- neither has a unified spelling
    with pytest.raises(ValueError, match="fa3: unsupported q dtype"):
        resolve_paged_attention(
            device=device,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim_qk=128,
            q_dtype=dtype,
            kv_dtype=dtype,
            page_size=16,
            kv_layout="NHD",
            causal=True,
            kv_input_form="page_indices",
            backend="fa3",
        )
