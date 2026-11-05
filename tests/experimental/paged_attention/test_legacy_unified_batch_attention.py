"""Legacy -> unified: tests/attention/test_batch_attention.py

The legacy tests check ``flashinfer.BatchAttention``, the holistic persistent
kernel (``PersistentKernelTemplate``, its own JIT module and planner),
against the fa2 ``BatchPrefillWithPagedKVCacheWrapper`` ("old scheduler").
No ``PagedAttention`` backend runs that kernel: the unified backends are fa2,
fa3, cudnn, trtllm-gen and cake.  Running fa2 in its place would compare the
legacy reference kernel with itself.  So every case is an expected failure
naming the missing backend; the NVFP4 cases also need an NVFP4 KV cache, and
the causal "real workload" config (q_len > kv_len, fully masked rows) is
outside the causal contract too.  The grids are the legacy grids, so the node
ids equal the legacy ids.
"""

import numpy as np
import pytest
import torch

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_attention.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_with_noncontiguous_q",
        ["test_batch_attention_with_noncontiguous_q"],
        "unsupported-by-design",
        "xfail: no PagedAttention backend runs the BatchAttention holistic kernel "
        "(API limit)",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_correctness",
        ["test_batch_attention_correctness"],
        "unsupported-by-design",
        "same grid and ids; xfail: no PagedAttention backend runs the BatchAttention "
        "holistic kernel (API limit); the causal q_len > kv_len config is also "
        "outside the causal contract",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_nvfp4",
        ["test_batch_attention_nvfp4"],
        "unsupported-by-design",
        "same grid and ids; xfail: no BatchAttention backend and no NVFP4 KV cache "
        "(API limits)",
    ),
]

NO_BACKEND_XFAIL = (
    "API limit: no PagedAttention backend runs flashinfer.BatchAttention's "
    "holistic persistent kernel, the legacy subject (the unified backends are "
    "fa2, fa3, cudnn, trtllm-gen and cake)"
)


def _build_seq_len_configs():
    """Verbatim from the legacy file (seeds numpy and torch with 42)."""
    np.random.seed(42)
    torch.manual_seed(42)

    seq_len_configs = [
        [(146, 146)],
        [(67, 67)],
        [(8190, 7939)],
        [(2048, 1)] * 77,  # decode-only
        [(4099, 129)] * 2,  # prefill-only
        [(600, 1)] * 132 * 2 + [(5000, 3)] * 128,
        [(1024, 1)] * 100 + [(8192, 17)] * 8,  # speculative decode
        [(766, 2)] * 99 + [(1024, 512)] * 1,  # chunked prefill
        [(2, 235)] + [(1, 13353)],  # real workload
    ]

    bsz, stride, sparsity = 256, 16, 0.05
    full_kv_len = np.random.randint(1000, 11000, size=bsz)
    seq_len = []
    for i in range(bsz):
        if i % stride == 0:
            kv_len, qo_len = full_kv_len[i], stride + 1
        else:
            kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
        seq_len.append((kv_len, qo_len))
    seq_len_configs.append(seq_len)

    return seq_len_configs


def test_batch_attention_with_noncontiguous_q():
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize("seq_len_pairs", _build_seq_len_configs())
@pytest.mark.parametrize("page_block_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 7, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("v_scale", [2.0, None])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
def test_batch_attention_correctness(
    seq_len_pairs,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    v_scale,
    causal,
    layout,
    test_dtype,
    logits_soft_cap,
):
    reason = NO_BACKEND_XFAIL
    if causal and any(qo_len > kv_len for kv_len, qo_len in seq_len_pairs):
        reason += (
            "; causal rows with q_len > kv_len (fully masked) are outside the "
            "causal contract too"
        )
    pytest.xfail(reason)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_batch_attention_nvfp4(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    pytest.xfail(
        NO_BACKEND_XFAIL + "; no NVFP4 KV cache (packed FP4 KV and its kv_cache_sf "
        "scale factors) either"
    )
