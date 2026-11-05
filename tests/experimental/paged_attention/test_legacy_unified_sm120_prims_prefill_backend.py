"""Legacy -> unified: tests/attention/test_sm120_prims_prefill_backend.py

The legacy subject is the ``cute-dsl-prims`` backend of the paged prefill
wrapper (the SM120 CuTe-DSL FP8 FMHA; SM120 only, nvidia-cutlass-dsl >=
4.7.0).  Every paged legacy case is an fp8 e4m3 q / K / V problem at head_dim
32 with an output dtype of its own.  No ``PagedAttention`` backend runs that
kernel, fp8 q and head_dim 32 are not declared on any backend, and the output
is written in q's dtype, so every case is an expected failure naming those
API limits, plus the backend-specific contract a case checks (compiling
before CUDA-graph capture, the option rejections, PDL).  The legacy gate is
mirrored, so on B200 every case skips like the legacy.

The ragged test (and the ragged half of the option-rejection test) is out of
scope.
"""

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch
from packaging.version import Version

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
        "xfail after the legacy SM120 gate: no cute-dsl-prims backend, no fp8 q, no "
        "head_dim 32, no output dtype (API limits)",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_reads_updated_caller_block_table",
        ["test_cuda_graph_reads_updated_caller_block_table"],
        "unsupported-by-design",
        "same xfail as the row above",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_rejects_uncompiled_specialization",
        ["test_cuda_graph_rejects_uncompiled_specialization"],
        "unsupported-by-design",
        "xfail after the legacy SM120 gate: the compile-before-capture contract of "
        "the cute-dsl-prims backend, which PagedAttention does not have",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_fail_fast_for_unsupported_options",
        ["test_prims_fail_fast_for_unsupported_options"],
        "unsupported-by-design",
        "xfail after the legacy SM120 gate: the max_sequence_kv / sinks rejections of "
        "the cute-dsl-prims backend, which PagedAttention does not have",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_accepts_combined_hnd_cache_and_pdl",
        ["test_prims_accepts_combined_hnd_cache_and_pdl"],
        "unsupported-by-design",
        "xfail after the legacy SM120 gate: no cute-dsl-prims backend and no "
        "enable_pdl (API limits)",
    ),
]

NO_BACKEND_XFAIL = (
    "API limit: no PagedAttention backend runs the cute-dsl-prims SM120 FP8 FMHA, "
    "the legacy subject; fp8 q and head dims (32, 32) are not declared on any "
    "backend, and run() writes the output in q's dtype (the legacy writes fp16 / "
    "bf16 from fp8 q)"
)


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
    pytest.xfail(NO_BACKEND_XFAIL)


def test_cuda_graph_reads_updated_caller_block_table():
    pytest.xfail(NO_BACKEND_XFAIL)


def test_cuda_graph_rejects_uncompiled_specialization():
    pytest.xfail(
        NO_BACKEND_XFAIL + "; the case checks that backend's compile-before-capture "
        "contract"
    )


def test_prims_fail_fast_for_unsupported_options():
    pytest.xfail(
        NO_BACKEND_XFAIL + "; the case checks that backend's max_sequence_kv and "
        "sinks rejections"
    )


def test_prims_accepts_combined_hnd_cache_and_pdl():
    pytest.xfail(NO_BACKEND_XFAIL + "; no enable_pdl either")
