"""Legacy -> unified: tests/attention/test_sm120_prims_prefill_backend.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and every row skips on the SM120 + nvidia-cutlass-dsl>=4.7.0
gate, so the ``cute-dsl-prims`` backend never executed in PR CI (reports/
unified-prefill-round4-20260918/ci-status.md).  On B200 the legacy file skips
entirely (12 skipped).

The ``cute-dsl-prims`` backend of the legacy paged wrapper (SM120 only) is
not a unified backend, and every paged legacy row is an fp8 e4m3 q / K / V
problem at head_dim 32 with a bf16 / fp16 output -- three axes the unified
API does not express (fp8 q: EXPECT_FP8_Q; head_dim 32: EXPECT_HEAD_DIM_32;
an output dtype independent of q: EXPECT_OUTPUT_DTYPE), so no numeric
workload of this file can run through ``PagedAttention`` today: each paged
row asserts the clear rejections for its exact shape, and the lifecycle /
compile-cache / PDL contracts are ``native-only``.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_paged_public_wrapper: fp8 q, D32, bf16 output, ``workspace_size()``
  == (0, 0) (a backend-private fact; the unified analog is
  ``PagedAttention.workspace_requirements``) and ``enable_pdl`` (no unified
  knob) -- rejections asserted, nothing runs.
- test_cuda_graph_reads_updated_caller_block_table: the legacy graph re-reads
  a caller-owned table mutated in place; the unified graph mode re-plans via
  ``update(metadata)`` (converted on a runnable shape in
  test_legacy_unified_attention_ts_context.py); this row's fp8 / D32 fixture
  asserts the rejections.
- test_cuda_graph_rejects_uncompiled_specialization: a "compile before
  capture" contract of the cute-dsl backend; unified refuses plan() under
  capture altogether (native-only here, converted in the TensorSpeed file).
- test_prims_fail_fast_for_unsupported_options: ``max_sequence_kv`` and the
  sinks rejection are backend-private options; the unified analog of "a
  feature the backend lacks" is the resolve-time capability exclusion
  (asserted for sinks on cuDNN).
- test_prims_accepts_combined_hnd_cache_and_pdl: compile-cache hit accounting
  and PDL are backend-private (native-only).
- test_ragged_public_wrapper: ragged, out-of-scope.
"""

import inspect

import pytest
import torch

from flashinfer.experimental.paged_attention import CAPABILITIES
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_HEAD_DIM_32,
    EXPECT_OUTPUT_DTYPE,
    check_legacy_map,
    check_legacy_map_complete,
    fp8_q_rejected,
    gated,
    output_dtype_knob_present,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_sm120_prims_prefill_backend.py"

LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_ragged_public_wrapper",
        [],
        "out-of-scope",
        "BatchPrefillWithRaggedKVCacheWrapper(backend='cute-dsl-prims'): ragged",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_paged_public_wrapper",
        ["test_paged_public_wrapper"],
        "unsupported-by-design",
        "fp8 q / K / V at D32 with a bf16 output (H4:2, page 16/32/64/128, causal): "
        "fp8 q (EXPECT_FP8_Q), head_dim 32 (EXPECT_HEAD_DIM_32) and the output dtype "
        "(EXPECT_OUTPUT_DTYPE) each assert their rejection per page size; the "
        "workspace_size() == (0, 0) fact and enable_pdl are backend-private",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_reads_updated_caller_block_table",
        ["test_cuda_graph_reads_updated_caller_block_table"],
        "unsupported-by-design",
        "same fp8 / D32 fixture: rejections asserted; the graph-replay semantics "
        "differ by design (legacy re-reads the caller's table in place, unified "
        "re-plans through update(metadata), converted on a runnable shape in "
        "test_legacy_unified_attention_ts_context.py)",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_cuda_graph_rejects_uncompiled_specialization",
        ["test_sm120_prims_is_not_a_unified_backend"],
        "native-only",
        "'compiled before CUDA Graph capture' is a cute-dsl-prims artifact contract; "
        "unified plan() refuses to run under capture at all (converted in the "
        "TensorSpeed file)",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_fail_fast_for_unsupported_options",
        ["test_prims_fail_fast_for_unsupported_options"],
        "native-only",
        "max_sequence_kv and the sinks NotImplementedError are backend-private "
        "options of cute-dsl-prims; the unified analog -- a capability exclusion "
        "named at resolve time -- is asserted for sinks on cuDNN",
    ),
    (
        "tests/attention/test_sm120_prims_prefill_backend.py::test_prims_accepts_combined_hnd_cache_and_pdl",
        ["test_sm120_prims_is_not_a_unified_backend"],
        "native-only",
        "compile-cache hit accounting across enable_pdl values is a cute-dsl-prims "
        "contract; unified has no PDL knob (a backend launch detail)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def test_sm120_prims_is_not_a_unified_backend():
    """The anchor of the native-only rows: no cute-dsl-prims backend, no
    compile-cache / PDL surface on PagedAttention."""
    assert "cute-dsl-prims" not in CAPABILITIES
    params = set(inspect.signature(PagedAttention.plan).parameters) | set(
        inspect.signature(PagedAttention.run).parameters
    )
    assert not any("pdl" in p or "cache_info" in p for p in params)
    assert not any(
        hasattr(PagedAttention, attr) for attr in ("cache_info", "cache_clear")
    )


def _assert_legacy_regime_rejected(*, page_size: int) -> None:
    """The whole legacy regime -- fp8 q / K / V, head_dim 32, an output dtype
    of its own -- for the legacy shape (H4:2 / H2:1, HND, causal)."""
    assert output_dtype_knob_present() == EXPECT_OUTPUT_DTYPE
    res = fp8_q_rejected(
        backend="fa2",
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,  # fp8 q is rejected before the head dim is looked at
        page_size=page_size,
        kv_layout="HND",
        causal=True,
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: run the legacy fp8 fixture here")
    res = gated(
        EXPECT_HEAD_DIM_32,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim_qk=32,
            q_dtype=torch.bfloat16,
            page_size=page_size,
            kv_layout="HND",
            causal=True,
            need_lse=True,
        ),
        match="unsupported head dims \\(32, 32\\)",
    )
    if res is not None:
        pytest.fail("EXPECT_HEAD_DIM_32 flipped: run the D32 bf16 twin here")


@pytest.mark.parametrize("page_size", [16, 32, 64, 128])
def test_paged_public_wrapper(page_size):
    _assert_legacy_regime_rejected(page_size=page_size)
    # the unified workspace contract that replaces workspace_size() == (0, 0)
    assert callable(getattr(PagedAttention, "workspace_requirements", None))


def test_cuda_graph_reads_updated_caller_block_table():
    _assert_legacy_regime_rejected(page_size=16)
    # the unified graph-mode re-plan entry point (design doc "Plan lifecycle")
    assert callable(getattr(PagedAttention, "update", None))
    assert "use_cuda_graph" in inspect.signature(PagedAttention.__init__).parameters


def test_prims_fail_fast_for_unsupported_options():
    """A feature the backend lacks is a resolve-time exclusion with the
    reason, never a NotImplementedError at run: sinks on cuDNN."""
    _assert_legacy_regime_rejected(page_size=16)
    plan_params = inspect.signature(PagedAttention.plan).parameters
    assert "max_sequence_kv" not in plan_params
    with pytest.raises(ValueError, match="attention sinks not supported"):
        resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
            kv_layout="HND",
            causal=True,
            sinks=True,
            backend="cudnn",
        )
    res = resolve_paged_attention(
        device=torch.device(DEVICE),
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=16,
        kv_layout="HND",
        causal=True,
        sinks=True,
    )
    assert "cudnn" in res.excluded and "cudnn" not in res.backends
