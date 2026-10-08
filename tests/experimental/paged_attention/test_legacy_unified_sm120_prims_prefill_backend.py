"""Legacy -> unified: tests/attention/test_sm120_prims_prefill_backend.py

The legacy subject is the ``cute-dsl-prims`` backend of the paged / ragged
prefill wrappers (SM120 only, nvidia-cutlass-dsl >= 4.7.0).  Every paged
legacy case is an fp8 e4m3 q / K / V problem at head_dim 32 with an output
dtype of its own: the unified API has no fp8 q (nor head_dim 32, nor an output
dtype), so the converted cases assert the plan() rejection of the legacy
configuration and stop.  The legacy gate is mirrored, so the rows skip where
the legacy skips.

The ragged test is out of scope; the CUDA-graph compile-before-capture, the
backend-private option rejections (``max_sequence_kv``, sinks) and the
compile-cache / PDL accounting are contracts of the cute-dsl-prims backend
(native-only).
"""

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_sm120_prims_prefill_backend.py"
LEGACY_MAP = [
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_ragged_public_wrapper",
        [],
        "out-of-scope",
        "BatchPrefillWithRaggedKVCacheWrapper: ragged KV",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_paged_public_wrapper",
        ["test_paged_public_wrapper"],
        "unsupported-by-design",
        "fp8 q / K / V at head_dim 32 with a bf16 output: asserts the fp8 q rejection; "
        "legacy SM120 gate mirrored",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_reads_updated_caller_block_table",
        ["test_cuda_graph_reads_updated_caller_block_table"],
        "unsupported-by-design",
        "fp8 q / K / V at head_dim 32 with an fp16 output: asserts the fp8 q rejection; "
        "legacy SM120 gate mirrored",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_rejects_uncompiled_specialization",
        [],
        "native-only",
        "'compiled before CUDA Graph capture' artifact contract of cute-dsl-prims",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_fail_fast_for_unsupported_options",
        [],
        "native-only",
        "max_sequence_kv and the sinks NotImplementedError are cute-dsl-prims options",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_accepts_combined_hnd_cache_and_pdl",
        [],
        "native-only",
        "compile-cache hit accounting across enable_pdl values of cute-dsl-prims",
    ),
]


def _has_required_cutlass_dsl() -> bool:
    try:
        installed_version = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return False
    return Version(installed_version) >= Version("4.7.0")


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0)
    or not _has_required_cutlass_dsl(),
    reason="requires SM120 and nvidia-cutlass-dsl>=4.7.0",
)


@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_paged_public_wrapper(page_size):
    """The legacy batch (q lens 3 / 2, kv lens 5 / 4, H4:2, D32, HND, causal)
    in fp8 e4m3: plan() rejects the fp8 q on every backend."""
    device = torch.device("cuda:0")
    qo = torch.tensor([0, 3, 5], dtype=torch.int32)
    kv_lens = torch.tensor([5, 4], dtype=torch.int32)
    page_indices = torch.tensor([3, 1, 4], dtype=torch.int32)
    md = PagedAttentionMetadata.csr(
        qo.to(device),
        kv_lens.to(device),
        page_indices.to(device),
        page_size=page_size,
        max_q_len=3,
        max_kv_len=5,
        qo_indptr_cpu=qo,
        kv_seq_lens_cpu=kv_lens,
    )
    with pytest.raises(ValueError, match="unsupported q dtype torch.float8_e4m3fn"):
        PagedAttention(device).plan(
            md,
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim_qk=32,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            kv_layout="HND",
            causal=True,
            lse_mode="base2",
        )


def test_cuda_graph_reads_updated_caller_block_table():
    """The legacy graph batch (one q token over one full page 16, H2:1, D32,
    HND) in fp8 e4m3: plan() rejects the fp8 q on every backend."""
    device = torch.device("cuda:0")
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    kv_lens = torch.tensor([16], dtype=torch.int32)
    md = PagedAttentionMetadata.dense(
        indptr.to(device),
        kv_lens.to(device),
        torch.tensor([[0]], dtype=torch.int32, device=device),
        page_size=16,
        max_q_len=1,
        max_kv_len=16,
        qo_indptr_cpu=indptr,
        kv_seq_lens_cpu=kv_lens,
    )
    with pytest.raises(ValueError, match="unsupported q dtype torch.float8_e4m3fn"):
        PagedAttention(device, use_cuda_graph=True).plan(
            md,
            num_qo_heads=2,
            num_kv_heads=1,
            head_dim_qk=32,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            kv_layout="HND",
            causal=False,
        )
