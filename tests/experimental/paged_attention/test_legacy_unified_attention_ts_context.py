"""Legacy -> unified: tests/attention/test_attention_ts_context.py

The legacy functions in scope are the ones built on the one-shot paged entry
point ``flashinfer.attention.prims_ts.batch_prefill_with_paged_kv_cache``.
The PrimTS (TensorSpeed) kernel is not a ``PagedAttention`` backend, so the
GPU rows run the legacy fixture on fa2, the backend that serves the whole
legacy envelope (D256, a strided page-table view, an in-page V tail past
kv_len that is NaN).  Same parametrize grid, same fixtures and seeds (the
legacy module's own ``_make_paged_context_case`` / ``_make_native_paged_metadata``
/ ``_poison_invalid_paged_v_tails``), so the node ids equal the legacy ids.
Each GPU case checks the legacy reference (``_assert_context_correct``) at
the legacy tolerance, and the output and LSE against the fp32 paged-attention
oracle.  The legacy ``output_scale`` is the unified ``run(v_scale=)``.  The
contract rows pin the same facts on the unified public surface.

The other 81 legacy functions drive ``BatchPrefillPagedTSWrapper`` /
``BatchPrefillTSWrapper``, the contiguous / packed (ragged) path, variable
windows, MLA or kernel internals: out of scope.
"""

import inspect
import itertools
import math
from dataclasses import replace

import pytest
import torch

from flashinfer.prefill import (
    GraphCapacity,
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from flashinfer.utils import is_sm100a_supported

# the legacy module importorskips the cutlass DSL, as the legacy file does
from tests.attention.test_attention_ts_context import (
    _assert_context_correct,
    _context_reference,
    _cumulative,
    _make_native_paged_metadata,
    _make_paged_context_case,
    _poison_invalid_paged_v_tails,
)

from .paged_attention_reference import reference_paged_prefill

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_attention_ts_context.py"
LEGACY_MAP = [
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_public_surfaces_hide_internal_tuning",
        ["test_attention_ts_context_public_surfaces_hide_internal_tuning"],
        "partial",
        "the same forbidden-token scan over the PagedAttention / metadata / resolve "
        "/ GraphCapacity signatures; 'backend' is allowed (the unified selection "
        "axis)",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_wrapper_has_no_workspace_api",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_wrapper_exposes_compile_oriented_contract",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_plan_rejects_non_bool_zero_tail_contract",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_contiguous_wrapper_exposes_compile_oriented_contract",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_exposes_fixed_table_contract",
        ["test_attention_ts_context_paged_one_shot_exposes_fixed_table_contract"],
        "partial",
        "the fixed-table spelling is PagedAttentionMetadata.dense + run(): names, "
        "kinds and defaults pinned; the one-shot function itself has no unified "
        "analog",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture",
        ["test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture"],
        "partial",
        "plan() rejects under capture; the packed batch_prefill variant is ragged",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper",
        ["test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper"],
        "partial",
        "the metadata keeps the caller's tensors by identity; the stubbed wrapper "
        "plumbing has no unified analog",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_validates_fixed_table",
        ["test_attention_ts_context_paged_one_shot_validates_fixed_table"],
        "partial",
        "same tensors and facts; unified takes max_q_len / max_kv_len explicitly "
        "and validates them instead of deriving them",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata",
        ["test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata"],
        "partial",
        "short-row rejected; invalid-active-page accepted (page ids trusted in- "
        "pool) and kv_len 0 accepted (padding row): contract differences, asserted",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_contiguous_plan_reuses_dynamic_packed_requests",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_variable_window_bounds_are_runtime_state",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_accepts_precomputed_variable_window_cta_starts",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_validates_precomputed_variable_window_cta_starts",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_rejects_cta_starts_for_non_variable_mask",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_plan_compiles_once_for_dynamic_metadata",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_explicit_uniform_plan_compiles_once",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_failed_paged_replan_retains_previous_state",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_failed_contiguous_replan_retains_previous_state",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_run_validate_false_bypasses_validators",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_run_rejects_invalid_runtime_scale_values",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_run_forwards_valid_scales",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_run_keeps_lifecycle_check_without_validation",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_run_rejects_non_bool_validate",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_allows_arbitrary_padding_ids",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_allows_fixed_stride_extra_padding",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_enforces_declared_length_contract",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_defaults_allow_dynamic_lengths",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_enforces_row_strided_table",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_validation_rejects_unsafe_fixed_values",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_fixed_oracle_is_bottom_right_causal",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_packed_oracle_applies_left_window_per_row",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_run_requires_plan",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_pipeline_policy_is_semantic_and_capacity_safe",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_page_window_fits_static_geometry_and_capacity",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_heavy_first_static_raster_policy",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uses_ldtm_stat_default_is_off",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uses_ldtm_stat_default_follows_gpu",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uses_ldtm_stat_schedule_builds",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_clc_policy_is_structural",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_contiguous_clc_policy_is_structural",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_contiguous_persistence_follows_wave_count",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d128_paged_clc_task_graph_is_safe",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_live_paged_clc_uses_distinct_auxiliary_warps",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_uniform_paged_static_scheduler_is_safe",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_public_plan_rejects_unsupported_arch",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_plan_rejects_critical_public_contracts",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_accepts_public_page_sizes",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_rejects_unsupported_page_sizes",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_supported_page_sizes_accuracy",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_zero_fills_nan_v_tail",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_run_rejects_causal_q_longer_than_kv",
        ["test_attention_ts_context_run_rejects_causal_q_longer_than_kv"],
        "partial",
        "paged: plan() rejects Sq > Sk on fa2; packed (ragged) id skips",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_plan_uses_conservative_dynamic_facts",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_v_tail_clear_policy",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_rejects_variable_window_before_compile",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_plan_ignores_aggregate_kv_capacity",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_reserves_int32_work_tile_padding",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_bounded_public_correctness_matrix",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_variable_window_t1_i1_t2_i2",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_variable_window_uses_cta_minimum_start",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_variable_window_clamps_padded_q_rows",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_reuses_compiled_topology_across_batch_sizes",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_paged_context_reuses_compiled_topology_across_batch_sizes",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_variable_window_graph_reloads_runtime_bounds",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_fixed_dense_k_tail_excludes_tma_padding",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_packed_dense_k_bounds_exclude_peer_requests",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uniform_aligned_packed_dense_accuracy",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uniform_packed_offsets_accuracy",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_uniform_packed_window_offsets_accuracy",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_dense_k_mask_accuracy",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_invalid_padding_ids_are_not_dereferenced",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d128_paged_s16k_runtime",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_fixed_window_tail_excludes_left_marker",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_fixed_window_loop_excludes_right_marker",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_fixed_head_paired_window_runtime",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_bf16_fixed_dense_runtime",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_paged_dense_persistent_capacity_runtime",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_fp8_paged_dense_crosses_page_windows",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_paged_head_paired_window_runtime",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_paged_dynamic_causal_runtime",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_graph_replay_reads_updated_fixed_metadata",
        ["test_attention_ts_context_paged_graph_replay_reads_updated_fixed_metadata"],
        "partial",
        "same fixture on fa2 in graph mode; the mutated table is re-planned through "
        "update() before replay (legacy re-reads in place); legacy reference plus "
        "the fp32 oracle; PrimTS tile-config asserts dropped",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_d256_fixed_causal_single_tile_runtime",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_causal_partial_tail_d256",
        ["test_attention_ts_context_paged_one_shot_causal_partial_tail_d256"],
        "equivalent",
        "same fixture (NaN V tails, output_scale as v_scale) on fa2; legacy "
        "reference plus the fp32 oracle (output and LSE)",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_live_q_offsets_expand_causal_domain_on_graph_replay",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_live_zero_offset_qk_redistribution_graph_replay",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_live_k_redistribution_graph_replay",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_window_live_q_redistribution_graph_replay",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_window_graph_replay_writes_fresh_output",
        [],
        "out-of-scope",
        "BatchPrefillPagedTSWrapper API (TensorSpeed wrapper lifecycle / scheduler "
        "policy / validation contract), not the one-shot paged entry point: out of "
        "the round-4 scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_supplied_out_stream_and_cuda_graph",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_mla_prefill",
        [],
        "out-of-scope",
        "contiguous / packed BatchPrefillTSWrapper, variable-window, MLA or "
        "kernel-internal contract: not paged prefill",
    ),
]


OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

# the legacy GPU gate (_REQUIRES_CONTEXT_GPU)
_REQUIRES_CONTEXT_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="PrimTS context attention requires SM100 or SM103",
)


def _dense(case, metadata, *, max_kv_len=None):
    """The legacy native fixed table (``_NativePagedMetadata``) as unified
    dense metadata over the same tensors."""
    qo_cpu = metadata.qo_indptr.cpu()
    kv_cpu = metadata.seq_lens_kv.cpu()
    return PagedAttentionMetadata.dense(
        metadata.qo_indptr,
        metadata.seq_lens_kv,
        metadata.block_tables,
        page_size=case.page_size,
        max_q_len=int((qo_cpu[1:] - qo_cpu[:-1]).max()),
        max_kv_len=int(kv_cpu.max()) if max_kv_len is None else max_kv_len,
        qo_indptr_cpu=qo_cpu,
        kv_seq_lens_cpu=kv_cpu,
    )


# ---------------------------------------------------------------------------
# public API contract
# ---------------------------------------------------------------------------


def test_attention_ts_context_public_surfaces_hide_internal_tuning() -> None:
    surfaces = (
        PagedAttention.__init__,
        PagedAttention.plan,
        PagedAttention.run,
        PagedAttention.update,
        PagedAttentionMetadata.dense,
        PagedAttentionMetadata.csr,
        resolve_paged_attention,
        GraphCapacity.__init__,
    )
    forbidden_token_prefixes = {
        "autotun",
        "clc",
        "config",
        "cta",
        "impl",
        "inst",
        "kernel",
        "mma",
        "pdl",
        "persist",
        "profil",
        "reduc",
        "schedul",
        "split",
        "stag",
        "tile",
        "warp",
    }
    forbidden_token_sequences = (
        ("groups", "tokens", "heads"),
        ("single", "kv"),
        ("tensor", "cores"),
    )
    forbidden_exact_names = {
        "args",
        "enable_pdl",
        "groups_tokens_heads_q",
        "head_dim_per_cta_v",
        "implementation",
        "kernel",
        "mma_variant",
        "num_ctas_per_head_dim",
        "num_insts_kv",
        "separate_reducer_impl",
        "use_cluster_reduction",
        "use_cluster_smem_reduction",
        "use_tensor_cores",
    }
    # the unified selection axis (design doc "Backend selection") is the one
    # legacy-forbidden name the unified API carries on purpose
    allowed_exact_names = {"backend"}
    violations = []
    for surface in surfaces:
        for parameter in inspect.signature(surface).parameters.values():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                violations.append(f"{surface.__qualname__}.**{parameter.name}")
                continue
            tokens = tuple(parameter.name.split("_"))
            has_forbidden_sequence = any(
                tokens[index : index + len(sequence)] == sequence
                for sequence in forbidden_token_sequences
                for index in range(len(tokens) - len(sequence) + 1)
            )
            has_forbidden_token = any(
                token.startswith(prefix)
                for token in tokens
                for prefix in forbidden_token_prefixes
            )
            if parameter.name not in allowed_exact_names and (
                parameter.name in forbidden_exact_names
                or has_forbidden_token
                or has_forbidden_sequence
            ):
                violations.append(f"{surface.__qualname__}.{parameter.name}")

    assert violations == []


def test_attention_ts_context_paged_one_shot_exposes_fixed_table_contract() -> None:
    # the unified fixed-table spelling: PagedAttentionMetadata.dense + run()
    dense = inspect.signature(PagedAttentionMetadata.dense).parameters
    required_parameters = ("qo_indptr", "kv_seq_lens", "block_tables")
    keyword_only_defaults = {
        "page_size": inspect.Parameter.empty,  # no hidden page-size default
        "max_q_len": inspect.Parameter.empty,
        "max_kv_len": inspect.Parameter.empty,
        "qo_indptr_cpu": None,
        "kv_seq_lens_cpu": None,
    }
    assert tuple(dense) == (*required_parameters, *keyword_only_defaults)
    assert all(
        dense[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and dense[name].default is inspect.Parameter.empty
        for name in required_parameters
    )
    assert all(
        dense[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in keyword_only_defaults
    )
    assert {
        name: dense[name].default for name in keyword_only_defaults
    } == keyword_only_defaults

    run = inspect.signature(PagedAttention.run).parameters
    run_keyword_only = ("out", "lse", "sm_scale", "k_scale", "v_scale", "sinks")
    assert tuple(run) == ("self", "q", "kv_cache", *run_keyword_only)
    assert all(
        run[name].kind is inspect.Parameter.KEYWORD_ONLY and run[name].default is None
        for name in run_keyword_only
    )
    # the legacy one-shot defaults (HND, window -1) live on plan()
    plan = inspect.signature(PagedAttention.plan).parameters
    assert plan["kv_layout"].default == "HND"
    assert plan["window_left"].default == -1


def test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture(
    monkeypatch,
) -> None:
    """Planning must not perform host metadata reads during capture."""

    md = PagedAttentionMetadata.dense(
        torch.tensor((0, 1), dtype=torch.int32, device="cuda"),
        torch.tensor((1,), dtype=torch.int32, device="cuda"),
        torch.tensor(((0,),), dtype=torch.int32, device="cuda"),
        page_size=32,
        max_q_len=1,
        max_kv_len=1,
    )
    attn = PagedAttention(torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="cannot run during CUDA graph capture"):
        attn.plan(
            md,
            num_qo_heads=2,
            num_kv_heads=1,
            head_dim_qk=128,
            q_dtype=torch.float16,
            causal=False,
            backend="fa2",
        )
    # the legacy packed batch_prefill() variant is ragged: not paged prefill


def test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper() -> None:
    """The metadata preserves the caller-owned fixed metadata tensors."""

    qo_indptr = torch.tensor((0, 4, 9), dtype=torch.int32, device="cuda")
    block_tables = torch.tensor(
        ((3, 1, -1), (7, 0, 2)), dtype=torch.int32, device="cuda"
    )
    seq_lens_kv = torch.tensor((33, 65), dtype=torch.int32, device="cuda")
    md = PagedAttentionMetadata.dense(
        qo_indptr, seq_lens_kv, block_tables, page_size=32, max_q_len=5, max_kv_len=65
    )

    assert md.qo_indptr is qo_indptr
    assert md.block_tables is block_tables
    assert md.kv_seq_lens is seq_lens_kv
    assert md.batch_size == 2
    assert md.total_q_tokens == 9
    assert md.max_kv_len == 65


def test_attention_ts_context_paged_one_shot_validates_fixed_table() -> None:
    """The legacy geometry facts from the same fixed-table tensors."""

    def build(max_kv_len):
        return PagedAttentionMetadata.dense(
            torch.tensor((0, 1, 3), dtype=torch.int32, device="cuda"),
            torch.tensor((17, 65), dtype=torch.int32, device="cuda"),
            torch.tensor(((4, -1, -1), (1, 3, 0)), dtype=torch.int32, device="cuda"),
            page_size=32,
            max_q_len=2,
            max_kv_len=max_kv_len,
        )

    md = build(65)
    assert md.max_kv_len == 65
    assert md.max_q_len == 2
    assert md.batch_size == 2
    # unified takes the maxima explicitly and validates them (the legacy derives)
    with pytest.raises(ValueError, match=r"max_kv_len \(64\) is smaller"):
        build(64)


@pytest.mark.parametrize(
    ("block_tables", "seq_lens_kv", "match"),
    (
        (((4, -1), (1, 3)), (17, 65), "at least ceil"),
        (((5, -1, -1), (1, 3, 0)), (17, 65), "active block_tables"),
        (((4, -1, -1), (1, 3, 0)), (0, 65), "entries must be positive"),
    ),
    ids=("short-row", "invalid-active-page", "nonpositive-kv-length"),
)
def test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata(
    block_tables: tuple[tuple[int, ...], ...],
    seq_lens_kv: tuple[int, ...],
    match: str,
) -> None:
    qo_indptr_cpu = torch.tensor((0, 1, 3), dtype=torch.int32)
    kv_lens_cpu = torch.tensor(seq_lens_kv, dtype=torch.int32)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    def build():
        return PagedAttentionMetadata.dense(
            qo_indptr_cpu.cuda(),
            kv_lens_cpu.cuda(),
            block_tables,
            page_size=32,
            max_q_len=2,
            max_kv_len=65,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_lens_cpu,
        )

    if match == "at least ceil":
        # short-row: the same check, unified wording
        with pytest.raises(ValueError, match="exceeds block_tables capacity"):
            build()
        return
    md = build()
    if match == "active block_tables":
        # invalid-active-page: accepted. Page ids are trusted in-pool (design
        # doc "Input contract"); a range check would need a device sync.
        assert int(md.block_tables.max()) == 5
        return

    # nonpositive-kv-length: accepted. kv_len 0 is a padding row (design doc
    # "Zero-row contract"); request 1 still computes correctly on fa2.
    torch.manual_seed(0)
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(5, 2, 32, 128, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn_like(k_cache)
    attn = PagedAttention(torch.device("cuda"))
    attn.plan(
        md,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=False,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (k_cache, v_cache))

    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        kv_lens_cpu,
        block_tables,
        32,
        False,
        kv_layout="HND",
    )
    torch.testing.assert_close(out[1:].float(), ref_out[1:], **OUT_TOL)
    torch.testing.assert_close(lse[1:], ref_lse[1:], **LSE_TOL)


# ---------------------------------------------------------------------------
# GPU rows: the legacy fixture on fa2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paged", (False, True), ids=("packed", "paged"))
@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_run_rejects_causal_q_longer_than_kv(paged: bool):
    """Bottom-right causal attention requires Sq <= Sk for every request."""

    if not paged:
        pytest.skip("packed (ragged) BatchPrefillTSWrapper variant: not paged prefill")
    case = _make_paged_context_case(
        q_lengths=(2, 3),
        k_lengths=(3, 2),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        seed=2026071930,
    )
    md = _dense(case, _make_native_paged_metadata(case))
    attn = PagedAttention(torch.device("cuda"))
    with pytest.raises(ValueError, match=r"request 1 has q_len 3 > kv_len 2"):
        attn.plan(
            md,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            kv_layout="HND",
            causal=True,
            backend="fa2",
        )


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
@pytest.mark.parametrize(
    ("head_dim", "plan_k_lengths"),
    (
        pytest.param(128, (65, 33), id="d128-direct-page-ids"),
        pytest.param(256, (1057, 1025), id="d256-staged-page-window"),
    ),
)
def test_attention_ts_context_paged_graph_replay_reads_updated_fixed_metadata(
    head_dim: int,
    plan_k_lengths: tuple[int, int],
) -> None:
    """Captured runs compute the runtime page table / lengths after update().

    The legacy wrapper re-reads the caller's tensors in place on replay; the
    unified graph mode re-plans the mutated tensors into reserved storage
    through ``update()`` (design doc "Plan lifecycle: graph mode")."""

    case = _make_paged_context_case(
        q_lengths=(17, 17),
        k_lengths=plan_k_lengths,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        output_scale=1.0,
        seed=2026090302 + head_dim,
    )
    metadata = _make_native_paged_metadata(
        case, extra_page_columns=1, row_stride_multiplier=2
    )
    assert not metadata.block_tables.is_contiguous()  # the legacy 2x row stride
    # graph mode sizes its reserved storage by the table width
    max_kv_len = int(metadata.block_tables.shape[1]) * case.page_size
    q = case.reference.q
    sm_scale = case.reference.sm_scale

    attn = PagedAttention(torch.device("cuda"), use_cuda_graph=True)
    attn.plan(
        _dense(case, metadata, max_kv_len=max_kv_len),
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"

    # warm up and capture into caller-owned output (legacy _capture_context_graph)
    graph_out = torch.full_like(q, float("nan"))
    graph_lse = torch.empty(q.shape[0], 4, dtype=torch.float32, device="cuda")
    attn.run(
        q, (case.k_cache, case.v_cache), out=graph_out, lse=graph_lse, sm_scale=sm_scale
    )
    torch.cuda.synchronize()
    _assert_context_correct(graph_out, case.reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn.run(
            q,
            (case.k_cache, case.v_cache),
            out=graph_out,
            lse=graph_lse,
            sm_scale=sm_scale,
        )
    metadata_ptrs = (metadata.block_tables.data_ptr(), metadata.seq_lens_kv.data_ptr())
    metadata_stride = metadata.block_tables.stride()

    # the runtime batch: reversed lengths, page ids 1..N in order
    runtime_k_lengths = tuple(reversed(plan_k_lengths))
    runtime_page_counts = tuple(
        math.ceil(length / case.page_size) for length in runtime_k_lengths
    )
    runtime_page_indptr = _cumulative(runtime_page_counts)
    runtime_page_indices = tuple(range(1, runtime_page_indptr[-1] + 1))
    runtime_block_tables = torch.full_like(metadata.block_tables, -911)
    for batch_idx, (begin, end) in enumerate(itertools.pairwise(runtime_page_indptr)):
        runtime_block_tables[batch_idx, : end - begin] = torch.tensor(
            runtime_page_indices[begin:end], dtype=torch.int32, device="cuda"
        )
    metadata.block_tables.copy_(runtime_block_tables)
    metadata.seq_lens_kv.copy_(
        torch.tensor(runtime_k_lengths, dtype=torch.int32, device="cuda")
    )
    assert metadata_ptrs == (
        metadata.block_tables.data_ptr(),
        metadata.seq_lens_kv.data_ptr(),
    )
    assert metadata.block_tables.stride() == metadata_stride
    runtime_md = _dense(case, metadata, max_kv_len=max_kv_len)
    attn.update(runtime_md)

    def gather_logical_cache(cache: torch.Tensor) -> torch.Tensor:
        requests = []
        for batch_idx, k_length in enumerate(runtime_k_lengths):
            page_begin = runtime_page_indptr[batch_idx]
            page_end = runtime_page_indptr[batch_idx + 1]
            page_ids = runtime_page_indices[page_begin:page_end]
            requests.append(
                cache[list(page_ids)]
                .permute(0, 2, 1, 3)
                .reshape(-1, cache.shape[1], cache.shape[3])[:k_length]
            )
        return torch.cat(requests)

    runtime_reference = replace(
        case.reference,
        k=gather_logical_cache(case.k_cache),
        v=gather_logical_cache(case.v_cache),
        kv_indptr=torch.tensor(
            _cumulative(runtime_k_lengths), dtype=torch.int32, device="cuda"
        ),
        k_lengths=runtime_k_lengths,
    )
    expected = _context_reference(runtime_reference)

    graph_out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    # the legacy one-shot cross-check: an eager plan of the runtime metadata
    eager = PagedAttention(torch.device("cuda"))
    eager.plan(
        runtime_md,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=True,
        backend="fa2",
    )
    one_shot_out, _ = eager.run(q, (case.k_cache, case.v_cache), sm_scale=sm_scale)

    _assert_context_correct(graph_out, runtime_reference, expected=expected)
    _assert_context_correct(one_shot_out, runtime_reference, expected=expected)

    # fp32 oracle on the runtime batch: output and base-2 LSE
    ref_out, ref_lse = reference_paged_prefill(
        q,
        case.k_cache,
        case.v_cache,
        runtime_md.qo_indptr_cpu,
        runtime_md.kv_seq_lens_cpu,
        runtime_md.block_tables,
        case.page_size,
        True,
        sm_scale=sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(graph_out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(graph_lse, ref_lse, **LSE_TOL)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_one_shot_causal_partial_tail_d256():
    case = _make_paged_context_case(
        q_lengths=(17, 65),
        # Cross the 128-token KV tile boundary so poisoned V tails also
        # exercise nonzero logical tile/page coordinates.
        k_lengths=(177, 193),
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=256,
        qkv_dtype=torch.float16,
        mask_type="causal",
        seed=2026071520,
    )
    _poison_invalid_paged_v_tails(case)
    metadata = _make_native_paged_metadata(case)
    md = _dense(case, metadata)

    attn = PagedAttention(torch.device("cuda"))
    attn.plan(
        md,
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim_qk=256,
        q_dtype=torch.float16,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    output = torch.full_like(case.reference.q, float("inf"))
    returned, lse = attn.run(
        case.reference.q,
        (case.k_cache, case.v_cache),
        out=output,
        sm_scale=case.reference.sm_scale,
        v_scale=case.reference.output_scale,
    )
    assert returned is output
    assert case.paged_kv_last_page_len.tolist() == [17, 1]
    assert case.paged_kv_indices.tolist() != list(range(case.paged_kv_indices.numel()))
    _assert_context_correct(output, case.reference)

    # fp32 oracle: output (times output_scale) and base-2 LSE.  The oracle
    # multiplies whole pages, so the NaN tail is zeroed for it (0 x NaN).
    ref_out, ref_lse = reference_paged_prefill(
        case.reference.q,
        case.k_cache,
        case.v_cache.nan_to_num(nan=0.0),
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        md.block_tables,
        case.page_size,
        True,
        sm_scale=case.reference.sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(
        output.float(), ref_out * case.reference.output_scale, **OUT_TOL
    )
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
