"""Legacy -> unified: tests/attention/test_hopper_fp8_attention.py

The three paged legacy functions run the SM90 fp8 fa3 kernel: an fp8 (e4m3 /
e5m2) QUERY and K/V with per-head or per-tensor scale TENSORS and an fp16
output, checked by an MSE budget against the fp16 fa3 wrapper.
``PagedAttention`` cannot express that: fa3 declares fp16 / bf16 q only, the
output is written in q's dtype, ``run()`` takes no ``q_scale`` and only
host-float per-tensor ``k_scale`` / ``v_scale``.  So every legacy id is an
expected failure naming that API limit, after the legacy SM90a skip (so on
other parts these rows skip like the legacy).

Single prefill, block-sparse, ragged and decode functions are out of scope.
"""

import pytest
import torch

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
        "same grid and ids; xfail after the legacy SM90a skip: fp8 q with per-head "
        "scale tensors and an fp16 output has no PagedAttention spelling (API limit)",
    ),
    (
        "tests/attention/test_hopper_fp8_attention.py::test_batch_prefill_paged_gqa",
        ["test_batch_prefill_paged_gqa"],
        "unsupported-by-design",
        "same grid and ids (GQA 32:8 / 16:4 / 8:2); same xfail as the row above",
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
        "same grid and ids; per-head / per-tensor fp8 scale tensors for q, k and v; "
        "same xfail as the rows above",
    ),
]


FP8_Q_XFAIL = (
    "API limit: PagedAttention has no fp8-q path: fa3 declares fp16 / bf16 q only, "
    "run() writes the output in q's dtype (the legacy writes fp16 from fp8 q) and "
    "takes no q_scale and no per-head scale tensors (host-float per-tensor "
    "k_scale / v_scale only)"
)


@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_paged(batch_size, num_heads, head_dim, causal, dtype):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(FP8_Q_XFAIL)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 4), (8, 2)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_gqa(
    batch_size, num_qo_heads, num_kv_heads, head_dim, causal, dtype
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(FP8_Q_XFAIL)


@pytest.mark.parametrize("scale_type", ["per_head", "per_tensor"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_batch_prefill_paged_scale_types(scale_type, dtype):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(FP8_Q_XFAIL)
