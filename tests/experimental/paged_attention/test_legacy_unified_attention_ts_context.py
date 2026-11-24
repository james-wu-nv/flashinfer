"""Legacy -> unified: tests/attention/test_attention_ts_context.py

The legacy functions in scope are the ones built on the one-shot paged entry
point ``flashinfer.attention.prims_ts.batch_prefill_with_paged_kv_cache``.

- The contract rows check the same facts on the unified public surface
  (``PagedAttention``, ``PagedAttentionMetadata``, ``resolve_paged_attention``,
  ``GraphCapacity``): no internal tuning knobs, the fixed-table spelling, no
  planning under CUDA-graph capture, the caller's tensors kept, the table
  validated, causal ``q_len > kv_len`` rejected.  Two invalid-metadata cases
  the legacy rejects are accepted by the unified contract (an out-of-pool
  page id, ``kv_len`` 0): expected failures naming that contract.
- The kernel rows (graph replay over an updated fixed table, the D256 causal
  partial tail) run the PrimTS (TensorSpeed) context kernel ``FmhaTs``.  No
  ``PagedAttention`` backend runs it, so they are expected failures naming
  the missing backend, after the legacy SM100 gate.

Same parametrize grids, so the node ids equal the legacy ids.  The other 81
legacy functions drive ``BatchPrefillPagedTSWrapper`` /
``BatchPrefillTSWrapper``, the contiguous / packed (ragged) path, variable
windows, MLA or kernel internals: out of scope.
"""

import inspect

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
    _make_native_paged_metadata,
    _make_paged_context_case,
)


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
        "short-row rejected, as legacy; invalid-active-page and kv_len 0 xfail: the "
        "unified contract accepts them (page ids trusted in-pool, kv_len 0 a padding "
        "row)",
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
        "unsupported-by-design",
        "xfail: no PagedAttention backend runs the PrimTS FmhaTs kernel (API limit)",
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
        "unsupported-by-design",
        "xfail: no PagedAttention backend runs the PrimTS FmhaTs kernel (API limit)",
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


NO_BACKEND_XFAIL = (
    "API limit: no PagedAttention backend runs the PrimTS (TensorSpeed) context "
    "kernel FmhaTs, the legacy subject (the unified backends are fa2, fa3, cudnn, "
    "trtllm-gen and cake)"
)

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
    try:
        build()
    except ValueError:
        return  # rejected, as the legacy requires
    if match == "active block_tables":
        pytest.xfail(
            "API contract: page ids are trusted in-pool (a range check would need a "
            "device sync), so an out-of-pool active page id is accepted"
        )
    pytest.xfail(
        "API contract: kv_len 0 is a padding row (the zero-row contract), so it is "
        "accepted"
    )


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
        qk_dtype=torch.bfloat16,
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
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.arch_blackwell
@_REQUIRES_CONTEXT_GPU
def test_attention_ts_context_paged_one_shot_causal_partial_tail_d256():
    pytest.xfail(NO_BACKEND_XFAIL)
