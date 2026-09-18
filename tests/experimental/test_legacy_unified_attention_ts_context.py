"""Legacy -> unified: tests/attention/test_attention_ts_context.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and every GPU row skips on the ``is_sm100a_supported`` gate
(the CPU-only contract rows run), so the PrimTS context kernels never
executed in PR CI (reports/unified-prefill-round4-20260918/ci-status.md).  On
B200 (nvidia-cutlass-dsl 4.8) the in-scope legacy rows run: 13 passed in
117 s (round-4 legacy run).

Scope (round-4 brief): the functions that use the one-shot paged entry point
``flashinfer.attention.prims_ts.batch_prefill_with_paged_kv_cache``.  The
TensorSpeed (PrimTS) kernel is not a unified backend; what this file converts
is each legacy function's WORKLOAD or CONTRACT: the legacy fixtures
(``_make_paged_context_case`` / ``_make_native_paged_metadata`` /
``_poison_invalid_paged_v_tails`` imported from the legacy module, the legacy
seeds, HND page pools with non-identity page ids) run in the dense form on
every unified backend that resolves (excluded ones recorded with the resolve
reason) and on ``auto`` (recorded), asserted with the legacy reference and
tolerance (``_context_reference`` / ``_assert_context_correct``) and the fp32
oracle.  The legacy ``output_scale`` is the unified ``run(v_scale=)``.  The
81 functions that drive ``BatchPrefillPagedTSWrapper`` / ``BatchPrefillTSWrapper``
or the kernel internals are ``out-of-scope`` rows (TensorSpeed wrapper
lifecycle, scheduler policy, variable-window and MLA contracts), per the brief.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- public surface (``..._public_surfaces_hide_internal_tuning``): the unified
  ``plan()`` / ``run()`` / metadata constructors expose none of the legacy
  tuning tokens; ``backend=`` is a documented selection axis of the unified
  design, not a tuning knob, and is the one legacy-forbidden name it carries.
- fixed-table one-shot contract (``..._exposes_fixed_table_contract``,
  ``..._forwards_fixed_table_to_wrapper``): the unified spelling is the
  metadata object (``PagedAttentionMetadata.dense``) that keeps the caller's
  tensors by identity; the one-shot function itself has no unified analog
  (the unified API is plan / run by design).
- capture guard (``..._one_shot_apis_reject_cuda_graph_capture``): unified
  ``plan()`` refuses to run under capture with its own message.
- metadata validation (``..._validates_fixed_table``,
  ``..._rejects_invalid_fixed_metadata``): the unified metadata takes
  ``max_q_len`` / ``max_kv_len`` explicitly (validated against the host
  mirrors) instead of deriving them; a too-narrow table is rejected (same
  class of check); a page id past the pool is TRUSTED in-pool by contract
  (design doc "Input contract": page ids are not range-checked, a device sync
  would be needed) and a ``kv_len == 0`` request is a legal padding row
  (design doc "Zero-row contract") -- two contract differences, recorded.
- causal envelope (``..._run_rejects_causal_q_longer_than_kv``): unified
  rejects at ``plan()`` with its own message; the packed (ragged) variant is
  out of the paged scope.
- graph replay (``..._paged_graph_replay_reads_updated_fixed_metadata``):
  the legacy wrapper re-reads the caller's table / lengths in place on
  replay; the unified graph mode re-plans through ``update(metadata)`` into
  reserved storage (design doc "Plan lifecycle: graph mode") -- the
  converted row captures, updates with the runtime table / lengths, replays
  and checks the oracle and an eager plan of the same batch.
- poisoned V tails (``..._paged_one_shot_causal_partial_tail_d256`` and the
  added ``test_paged_in_page_tail_past_kv_len_is_masked``): fa2 masks the
  in-page tail past kv_len; trtllm-gen, cake and cuDNN over-read it and
  0 x NaN reaches the output (measured on B200, EXPECT_INPAGE_TAIL_IGNORED:
  non-strict xfail on those backends so the outcome is recorded).  D256
  resolves on fa2 only.
"""

import inspect
import itertools
import math

import pytest
import torch

from flashinfer.prefill import (
    GraphCapacity,
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_INPAGE_TAIL_IGNORED,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    oracle,
    run_on_backends,
    xfail_unless,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_attention_ts_context.py"


LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_public_surfaces_hide_internal_tuning",
        ["test_attention_ts_context_public_surfaces_hide_internal_tuning"],
        "partial",
        "the same forbidden-token scan over the unified public surface "
        "(PagedAttention.__init__ / plan / run, PagedAttentionMetadata.dense / csr, "
        "resolve_paged_attention, GraphCapacity); 'backend' is the one "
        "legacy-forbidden name the unified API carries, as a documented selection "
        "axis",
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
        "the unified fixed-table spelling is PagedAttentionMetadata.dense(qo_indptr, "
        "kv_seq_lens, block_tables, *, page_size, max_q_len, max_kv_len, ...) and "
        "run(q, kv_cache, *, out, lse, sm_scale, k_scale, v_scale, sinks): pinned "
        "here; the one-shot function's defaults (page 32, HND, dense mask) have no "
        "unified analog (plan / run by design)",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture",
        ["test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture"],
        "partial",
        "unified plan() refuses to run while the stream is capturing (RuntimeError "
        "'cannot run during CUDA graph capture'); the legacy one-shot packed variant "
        "is out of the paged scope",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper",
        ["test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper"],
        "partial",
        "the metadata keeps the caller-owned qo_indptr / block_tables / kv_seq_lens "
        "by identity (no copy) and derives max_kv_len 65 / batch 2 facts from the "
        "same tensors; the stubbed wrapper plumbing is native-only",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_validates_fixed_table",
        ["test_attention_ts_context_paged_one_shot_validates_fixed_table"],
        "partial",
        "same tensors (q 3 tokens, table [[4,-1,-1],[1,3,0]], kv (17, 65)): the "
        "unified metadata reports batch 2, total_q 3, max_q 2, max_kv 65; unified "
        "takes max_q_len / max_kv_len explicitly (validated) instead of deriving them",
    ),
    (
        "tests/attention/test_attention_ts_context.py::test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata",
        ["test_attention_ts_context_paged_one_shot_rejects_invalid_fixed_metadata"],
        "partial",
        "same three cases: short-row is rejected ('exceeds block_tables capacity'); "
        "invalid-active-page is ACCEPTED (page ids are trusted in-pool by contract, "
        "no range check without a device sync); nonpositive-kv-length 0 is ACCEPTED "
        "(padding row); both contract differences asserted",
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
        "same fixture (q (2, 3), kv (3, 2), H4:4, D128, bf16, causal): the unified "
        "plan() rejects the causal envelope with its own message ('causal masking "
        "requires q_len_i <= kv_len_i ... request 1 has q_len 3 > kv_len 2'); the "
        "packed (ragged) variant skips as out of the paged scope",
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
        "same fixture (q (17, 17), kv (65, 33) / (1057, 1025), H4:2, D128 / D256, "
        "bf16, causal, seeds 2026090302 + D, non-identity ids, a capacity+1-column "
        "table on a 2x row stride) on every resolving backend: graph-mode plan, "
        "capture, update() with the runtime table / reversed lengths, replay; the "
        "replay matches the legacy fp32 reference of the runtime batch, the oracle "
        "and an eager plan of the same metadata; the legacy re-reads caller tensors "
        "in place, unified re-plans into reserved storage through update()",
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
        [
            "test_attention_ts_context_paged_one_shot_causal_partial_tail_d256",
            "test_paged_in_page_tail_past_kv_len_is_masked",
        ],
        "partial",
        "same fixture (q (17, 65), kv (177, 193), H8:4, D256, fp16, causal, seed "
        "2026071520, output_scale 0.75 as v_scale, NaN-poisoned V tails past kv_len "
        "in the last pages, last_page_len [17, 1]) on fa2 (the only D256 backend) vs "
        "_assert_context_correct and the oracle; the added D128 row shows the tail is "
        "over-read by trtllm-gen / cake / cuDNN (non-strict xfail behind "
        "EXPECT_INPAGE_TAIL_IGNORED)",
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


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


# ---------------------------------------------------------------------------
# legacy fixtures (imported from the legacy module, which importorskips the
# cutlass DSL the PrimTS kernels need; a missing package skips the row)
# ---------------------------------------------------------------------------


def _legacy():
    import tests.attention.test_attention_ts_context as legacy

    return legacy


def _dense_from_native(case, metadata, *, max_kv_len=None):
    """The legacy native fixed table (possibly a strided view with spare
    columns) as unified dense metadata; ``max_kv_len`` may be widened to the
    table's full extent (graph-mode width rule)."""
    kv_cpu = metadata.seq_lens_kv.cpu().to(torch.int32)
    qo_cpu = metadata.qo_indptr.cpu().to(torch.int32)
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


def _run_case(case, md, *, backends=None, include_auto=True, record_property=None):
    ref = case.reference
    kw = dict(
        num_qo_heads=int(ref.q.shape[1]),
        num_kv_heads=int(case.k_cache.shape[1]),
        head_dim_qk=int(ref.q.shape[2]),
        q_dtype=ref.q.dtype,
        kv_layout="HND",
        causal=ref.mask_type == "causal",
        window_left=ref.window_left,
        sm_scale=ref.sm_scale,
        v_scale=ref.output_scale if ref.output_scale != 1.0 else None,
        include_auto=include_auto,
        record_property=record_property,
    )
    if backends is not None:
        kw["backends"] = backends
    return run_on_backends(md, ref.q, (case.k_cache, case.v_cache), **kw)


# ---------------------------------------------------------------------------
# public surface
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
    # the unified API's selection axis (design doc "Backend selection"): the
    # one legacy-forbidden name it carries on purpose
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
    assert "backend" in inspect.signature(PagedAttention.plan).parameters


def test_attention_ts_context_paged_one_shot_exposes_fixed_table_contract() -> None:
    dense = inspect.signature(PagedAttentionMetadata.dense).parameters
    assert tuple(dense) == (
        "qo_indptr",
        "kv_seq_lens",
        "block_tables",
        "page_size",
        "max_q_len",
        "max_kv_len",
        "qo_indptr_cpu",
        "kv_seq_lens_cpu",
    )
    for name in ("qo_indptr", "kv_seq_lens", "block_tables"):
        assert dense[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert dense[name].default is inspect.Parameter.empty
    for name in ("page_size", "max_q_len", "max_kv_len"):
        assert dense[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert dense[name].default is inspect.Parameter.empty  # no hidden defaults
    for name in ("qo_indptr_cpu", "kv_seq_lens_cpu"):
        assert dense[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert dense[name].default is None
    run = inspect.signature(PagedAttention.run).parameters
    assert tuple(run) == (
        "self",
        "q",
        "kv_cache",
        "out",
        "lse",
        "sm_scale",
        "k_scale",
        "v_scale",
        "sinks",
    )
    assert all(
        run[name].kind is inspect.Parameter.KEYWORD_ONLY and run[name].default is None
        for name in ("out", "lse", "sm_scale", "k_scale", "v_scale", "sinks")
    )
    plan = inspect.signature(PagedAttention.plan).parameters
    assert plan["kv_layout"].default == "HND"
    assert plan["causal"].default is True
    assert plan["window_left"].default == -1
    assert plan["lse_mode"].default == "none"
    assert plan["backend"].default == "auto"


def test_attention_ts_context_one_shot_apis_reject_cuda_graph_capture(
    monkeypatch,
) -> None:
    """Unified plan() performs host work and refuses to run under capture."""
    md = dense_metadata(
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([[0]], dtype=torch.int32, device=DEVICE),
        32,
    )
    attn = PagedAttention(torch.device(DEVICE))
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


def test_attention_ts_context_paged_one_shot_forwards_fixed_table_to_wrapper() -> None:
    """The metadata preserves the caller-owned fixed metadata tensors."""
    qo_indptr = torch.tensor((0, 4, 9), dtype=torch.int32, device=DEVICE)
    block_tables = torch.tensor(
        ((3, 1, -1), (7, 0, 2)), dtype=torch.int32, device=DEVICE
    )
    seq_lens_kv = torch.tensor((33, 65), dtype=torch.int32, device=DEVICE)
    md = PagedAttentionMetadata.dense(
        qo_indptr, seq_lens_kv, block_tables, page_size=32, max_q_len=5, max_kv_len=65
    )
    assert md.qo_indptr is qo_indptr
    assert md.block_tables is block_tables
    assert md.kv_seq_lens is seq_lens_kv
    assert md.batch_size == 2 and md.total_q_tokens == 9
    assert md.max_kv_len == 65 and md.max_q_len == 5
    assert md.kv_input_form == "block_tables"


def test_attention_ts_context_paged_one_shot_validates_fixed_table() -> None:
    """The legacy geometry facts from the same fixed-table tensors."""
    md = PagedAttentionMetadata.dense(
        torch.tensor((0, 1, 3), dtype=torch.int32, device=DEVICE),
        torch.tensor((17, 65), dtype=torch.int32, device=DEVICE),
        torch.tensor(((4, -1, -1), (1, 3, 0)), dtype=torch.int32, device=DEVICE),
        page_size=32,
        max_q_len=2,
        max_kv_len=65,
    )
    assert md.max_kv_len == 65
    assert md.max_q_len == 2
    assert md.batch_size == 2
    assert md.total_q_tokens == 3
    # unified takes the maxima explicitly and validates them (the legacy derives)
    with pytest.raises(ValueError, match="max_kv_len \\(64\\) is smaller"):
        PagedAttentionMetadata.dense(
            torch.tensor((0, 1, 3), dtype=torch.int32, device=DEVICE),
            torch.tensor((17, 65), dtype=torch.int32, device=DEVICE),
            torch.tensor(((4, -1, -1), (1, 3, 0)), dtype=torch.int32, device=DEVICE),
            page_size=32,
            max_q_len=2,
            max_kv_len=64,
        )


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
    block_tables,
    seq_lens_kv,
    match: str,
) -> None:
    """The unified metadata's answer to the three legacy rejections."""
    build = lambda: PagedAttentionMetadata.dense(  # noqa: E731
        torch.tensor((0, 1, 3), dtype=torch.int32, device=DEVICE),
        torch.tensor(seq_lens_kv, dtype=torch.int32, device=DEVICE),
        torch.tensor(block_tables, dtype=torch.int32, device=DEVICE),
        page_size=32,
        max_q_len=2,
        max_kv_len=65,
    )
    if match == "at least ceil":
        # same class of check, unified wording
        with pytest.raises(ValueError, match="exceeds block_tables capacity"):
            build()
        return
    md = build()  # contract differences: accepted
    if match == "active block_tables":
        # page ids are trusted in-pool (design doc "Input contract"): no range
        # check without a device sync; the caller owns pool bounds
        assert int(md.block_tables.max()) == 5
    else:
        # kv_len 0 is a padding row (design doc "Zero-row contract")
        assert md.kv_seq_lens_cpu.tolist() == [0, 65]
        q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=DEVICE)
        k = torch.randn(5, 2, 32, 128, dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        results = run_on_backends(
            md,
            q,
            (k, v),
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            kv_layout="HND",
            causal=False,
            backends=("fa2",),
            include_auto=False,
        )
        o_out, _ = oracle(md, q, k, v, causal=False, kv_layout="HND")
        for _n, _s, out, _l in results:
            torch.testing.assert_close(out[1:].float(), o_out[1:], **OUT_TOL)


# ---------------------------------------------------------------------------
# causal envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paged", (False, True), ids=("packed", "paged"))
def test_attention_ts_context_run_rejects_causal_q_longer_than_kv(paged: bool):
    """Bottom-right causal attention requires Sq <= Sk for every request."""
    if not paged:
        pytest.skip("packed (ragged) BatchPrefillTSWrapper variant: not paged")
    legacy = _legacy()
    case = legacy._make_paged_context_case(
        q_lengths=(2, 3),
        k_lengths=(3, 2),
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.bfloat16,
        mask_type="causal",
        seed=2026071930,
    )
    metadata = legacy._make_native_paged_metadata(case)
    md = _dense_from_native(case, metadata)
    attn = PagedAttention(torch.device(DEVICE))
    with pytest.raises(
        ValueError,
        match=r"request 1 has q_len 3 > kv_len 2",
    ):
        attn.plan(
            md,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            causal=True,
            backend="fa2",
        )


# ---------------------------------------------------------------------------
# graph replay with updated fixed metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head_dim", "plan_k_lengths"),
    (
        pytest.param(128, (65, 33), id="d128-direct-page-ids"),
        pytest.param(256, (1057, 1025), id="d256-staged-page-window"),
    ),
)
def test_attention_ts_context_paged_graph_replay_reads_updated_fixed_metadata(
    head_dim: int,
    plan_k_lengths: tuple,
    record_property,
) -> None:
    """Captured runs compute the runtime table / lengths after update()."""
    legacy = _legacy()
    case = legacy._make_paged_context_case(
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
    metadata = legacy._make_native_paged_metadata(
        case, extra_page_columns=1, row_stride_multiplier=2
    )
    assert not metadata.block_tables.is_contiguous()  # the legacy 2x row stride
    width = int(metadata.block_tables.shape[1])
    md1 = _dense_from_native(case, metadata, max_kv_len=width * case.page_size)
    q = case.reference.q
    ref = case.reference
    common = dict(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
    )
    dev = torch.device(DEVICE)
    resolving = []
    for name in ("fa2", "trtllm-gen", "cake", "cudnn", "auto"):
        try:
            res = resolve_paged_attention(
                device=dev,
                page_size=case.page_size,
                need_lse=True,
                backend=name,
                **{k: v for k, v in common.items() if k != "lse_mode"},
            )
        except ValueError as e:
            record_property(f"excluded_{name}", str(e))
            continue
        resolving.append((name, res))
    assert resolving

    # the runtime batch: reversed lengths, page ids 1..N in order (legacy)
    runtime_k_lengths = tuple(reversed(plan_k_lengths))
    runtime_page_counts = tuple(
        math.ceil(length / case.page_size) for length in runtime_k_lengths
    )
    runtime_page_indptr = legacy._cumulative(runtime_page_counts)
    runtime_page_indices = tuple(range(1, runtime_page_indptr[-1] + 1))
    runtime_block_tables = torch.full_like(metadata.block_tables, -911)
    for batch_idx, (begin, end) in enumerate(itertools.pairwise(runtime_page_indptr)):
        runtime_block_tables[batch_idx, : end - begin] = torch.tensor(
            runtime_page_indices[begin:end], dtype=torch.int32, device=DEVICE
        )
    runtime_seq_lens = torch.tensor(runtime_k_lengths, dtype=torch.int32, device=DEVICE)

    def gather_logical_cache(cache):
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

    from dataclasses import replace

    runtime_reference = replace(
        ref,
        k=gather_logical_cache(case.k_cache),
        v=gather_logical_cache(case.v_cache),
        kv_indptr=torch.tensor(
            legacy._cumulative(runtime_k_lengths), dtype=torch.int32, device=DEVICE
        ),
        k_lengths=runtime_k_lengths,
    )
    expected = legacy._context_reference(runtime_reference)

    def plan_or_contiguous(attn, md, res, name, what):
        # the legacy table is a 2x row-stride VIEW; trtllm-gen / cake walk a
        # packed (batch, width) array and decline the view with the reason
        # (design doc "Page-table ABI per backend"); the conversion then hands
        # them the contiguous copy and records the difference
        try:
            attn.plan(md, backend=res, **common) if what == "plan" else attn.update(md)
            return md
        except ValueError as e:
            if "contiguous (batch, width) block_tables" not in str(e):
                raise
            record_property(f"{name}_table_view", "declined: " + str(e)[:80])
            md_c = PagedAttentionMetadata.dense(
                md.qo_indptr,
                md.kv_seq_lens,
                md.block_tables.contiguous(),
                page_size=md.page_size,
                max_q_len=md.max_q_len,
                max_kv_len=md.max_kv_len,
                qo_indptr_cpu=md.qo_indptr_cpu,
                kv_seq_lens_cpu=md.kv_seq_lens_cpu,
            )
            attn.plan(md_c, backend=res, **common) if what == "plan" else attn.update(
                md_c
            )
            return md_c

    for name, res in resolving:
        attn = PagedAttention(dev, use_cuda_graph=True)
        plan_or_contiguous(attn, md1, res, name, "plan")
        if name == "auto":
            record_property("auto_backend", attn.backend)
        out = torch.empty(q.shape[0], 4, head_dim, dtype=q.dtype, device=dev)
        lse = torch.empty(q.shape[0], 4, dtype=torch.float32, device=dev)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                attn.run(
                    q,
                    (case.k_cache, case.v_cache),
                    out=out,
                    lse=lse,
                    sm_scale=ref.sm_scale,
                )
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            attn.run(
                q, (case.k_cache, case.v_cache), out=out, lse=lse, sm_scale=ref.sm_scale
            )
        g.replay()
        torch.cuda.synchronize()
        legacy._assert_context_correct(out, ref)

        # the legacy mutates the caller's table / lengths in place and replays;
        # unified re-plans the new metadata into the reserved storage
        metadata.block_tables.copy_(runtime_block_tables)
        metadata.seq_lens_kv.copy_(runtime_seq_lens)
        md2 = _dense_from_native(case, metadata, max_kv_len=width * case.page_size)
        md2 = plan_or_contiguous(attn, md2, res, name, "update")
        out.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        legacy._assert_context_correct(out, runtime_reference, expected=expected)
        o_out, o_lse = oracle(
            md2,
            q,
            case.k_cache,
            case.v_cache,
            causal=True,
            kv_layout="HND",
            sm_scale=ref.sm_scale,
        )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
        # eager plan of the same runtime metadata (the legacy one-shot cross-check)
        eager = PagedAttention(dev)
        plan_or_contiguous(eager, md2, res, name, "plan")
        e_out, e_lse = eager.run(q, (case.k_cache, case.v_cache), sm_scale=ref.sm_scale)
        legacy._assert_context_correct(e_out, runtime_reference, expected=expected)
        torch.testing.assert_close(e_out.float(), o_out, **OUT_TOL)
        # restore the plan-time table for the next backend
        metadata.block_tables.copy_(
            legacy._make_native_paged_metadata(
                case, extra_page_columns=1, row_stride_multiplier=2
            ).block_tables
        )
        metadata.seq_lens_kv.copy_(
            torch.tensor(plan_k_lengths, dtype=torch.int32, device=DEVICE)
        )


# ---------------------------------------------------------------------------
# poisoned V tails past kv_len
# ---------------------------------------------------------------------------


def test_attention_ts_context_paged_one_shot_causal_partial_tail_d256(record_property):
    legacy = _legacy()
    case = legacy._make_paged_context_case(
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
    legacy._poison_invalid_paged_v_tails(case)
    metadata = legacy._make_native_paged_metadata(case)
    assert case.paged_kv_last_page_len.tolist() == [17, 1]
    assert case.paged_kv_indices.tolist() != list(range(case.paged_kv_indices.numel()))
    md = _dense_from_native(case, metadata)
    results = _run_case(case, md, record_property=record_property)
    assert all(served == "fa2" for _n, served, _o, _l in results)  # D256: fa2 only
    o_out, o_lse = oracle(
        md,
        case.reference.q,
        case.k_cache,
        case.v_cache.nan_to_num(nan=0.0),
        causal=True,
        kv_layout="HND",
        sm_scale=case.reference.sm_scale,
    )
    for _name, _served, out, lse in results:
        legacy._assert_context_correct(out, case.reference)
        torch.testing.assert_close(
            out.float(), o_out * case.reference.output_scale, **OUT_TOL
        )
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", ["fa2", "trtllm-gen", "cake", "cudnn"])
def test_paged_in_page_tail_past_kv_len_is_masked(backend, record_property):
    """The D128 twin of the legacy D256 row on every backend: the in-page V
    tail past kv_len is NaN.  fa2 masks it; trtllm-gen, cake and cuDNN
    over-read it (0 x NaN), recorded as a non-strict xfail behind
    EXPECT_INPAGE_TAIL_IGNORED (measured on B200, 2026-09-18)."""
    legacy = _legacy()
    case = legacy._make_paged_context_case(
        q_lengths=(17, 65),
        k_lengths=(177, 193),
        num_qo_heads=8,
        num_kv_heads=4,
        head_dim=128,
        qkv_dtype=torch.float16,
        mask_type="causal",
        seed=2026071520,
    )
    legacy._poison_invalid_paged_v_tails(case)
    metadata = legacy._make_native_paged_metadata(case)
    md = _dense_from_native(case, metadata)
    results = _run_case(
        case,
        md,
        backends=(backend,),
        include_auto=False,
        record_property=record_property,
    )
    ((_name, _served, out, _lse),) = results
    expected = legacy._context_reference(case.reference)
    finite = bool(torch.isfinite(out.float()).all())
    correct = finite and torch.allclose(out.float(), expected, rtol=1e-2, atol=2e-3)
    record_property("finite_with_nan_tail", finite)
    record_property("correct_with_nan_tail", correct)
    xfail_unless(
        EXPECT_INPAGE_TAIL_IGNORED,
        correct,
        f"{backend} over-reads the in-page V tail past kv_len (0 x NaN / garbage "
        f"reaches the output: finite={finite}); the legacy TensorSpeed kernel masks it",
    )
    legacy._assert_context_correct(out, case.reference, expected=expected)
