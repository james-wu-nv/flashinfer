"""Legacy -> unified: tests/attention/test_cake_fmha.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and its kernel rows skip on the sm_100 / sm_103 gate, so the
Cake context kernels never executed in PR CI (reports/unified-prefill-round4-
20260918/ci-status.md).

Scope: the paged CONTEXT kernel functions (``test_cake_context_*`` that run
``trtllm_batch_context_with_kv_cache(backend="cake")`` through the legacy
trtllm-gen prefill fixture) are converted; the context-route / JIT-spec /
manifest-member / adapter-source tests are ``native-only`` (backend-private
contracts of the Cake product on its prefill path: which cubin member serves a
shape, which binding source is compiled, the TMA descriptor bookkeeping --
nothing the unified API can observe); the decode-route, decode-kernel and
product-packaging tests (manifest, registry, AOT registration, public symbols)
are ``out-of-scope`` (not paged prefill).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_cake_context_bf16_separate_tables_matches_reference: the legacy
  ``uses_shared_paged_kv_idx=False`` layout (K at page 2p, V at 2p + 1 of one
  interleaved pool, a [B, 2, M] table) is two page-id mappings; the unified
  metadata takes exactly one (EXPECT_INDEPENDENT_KV_TABLES): the row asserts
  the rejection and runs the same fixture with the shared table on pinned
  ``cake`` (and on ``auto``, recorded) against the legacy fa2 reference and
  the oracle.
- test_cake_context_fp8_nhd_device_scale_skip_matches_reference: fp8 QKV
  (EXPECT_FP8_Q), device-tensor bmm scales (EXPECT_DEVICE_SCALES: ``run()``
  takes host floats so the call stays sync-free) and skip-softmax
  (EXPECT_SKIP_SOFTMAX) each assert their rejection; there is no numeric path
  to run until the flags flip.
- native-only rows: ``select_cake_fmha_context_route`` / ``get_cake_fmha_
  context_module`` / ``gen_cake_fmha_context_*_module`` contracts.  The
  unified layer sees Cake through ``trtllm_batch_context_with_kv_cache(
  backend="cake")`` only; route selection inside the product is invisible to
  it by design (docs/design_docs/paged_attention_unified_lifecycle.md,
  "Layering and ownership").
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_DEVICE_SCALES,
    EXPECT_FP8_Q,
    EXPECT_INDEPENDENT_KV_TABLES,
    EXPECT_SKIP_SOFTMAX,
    LSE_TOL,
    OUT_TOL,
    assert_legacy_close,
    check_legacy_map,
    check_legacy_map_complete,
    fp8_q_rejected,
    gated,
    independent_tables_rejected,
    legacy_fa2_paged_reference,
    legacy_trtllm_problem,
    oracle,
    plan_legacy_problem,
    resolve_or_skip,
    skip_softmax_knob_present,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_cake_fmha.py"


LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_manifest_is_authenticated_and_complete",
        [],
        "out-of-scope",
        "Cake product packaging (manifest / registry / AOT / symbols): not paged "
        "prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_public_manifest_is_defensive_copy",
        [],
        "out-of-scope",
        "Cake product packaging (manifest / registry / AOT / symbols): not paged "
        "prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_registry_accounts_for_manifest_routes_and_components",
        [],
        "out-of-scope",
        "Cake product packaging (manifest / registry / AOT / symbols): not paged "
        "prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_high_level_selectors_match_pinned_capability_corpus",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "replays the pinned capability corpus (decode + context cells) through the "
        "product's route selectors: Cake context-route / JIT-member selection "
        "contract (backend-private; the unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_capability_replay_covers_q257_exact_context_profile",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_jit_spec_uses_versioned_standalone_sources",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "compat JIT spec sources (context + decode compat module): Cake context "
        "adapter source / ABI contract (backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_jit_selects_one_manifest_member",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_jit_selects_all_exact_manifest_members",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_exact_sink_grid_matches_selector",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_exact_b4_grid_matches_cga_selector",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_other_grids_stay_persistent",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_bf16_absent_selectors_fall_back_to_compat",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_fp16_nhd_jit_selects_one_manifest_member",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_native_fp16_hd512_jit_selects_one_manifest_member",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_quant_bf16q_jit_selects_one_manifest_member",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_quant_fp8_jit_selects_main_and_reducer",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_quant_nvfp4_jit_selects_main_and_reducer",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_bf16_jit_selects_one_manifest_member",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_bf16_exact_profile_selector",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_fp8_jit_selects_one_manifest_member",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); fp8 q is EXPECT_FP8_Q "
        "on the unified side",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_jit_selects_fused_member",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); nvfp4 KV is "
        "EXPECT_NVFP4_KV on the unified side",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_adapters_match_public_feature_abi",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context adapter source / ABI contract (backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_optimized_context_adapters_require_signed_seq_lens",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context adapter source / ABI contract (backend-private csrc "
        "bookkeeping); the unified metadata is int32 by contract",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_typed_launch_adapters_use_typed_tensor_maps",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "context + decode adapter sources: Cake context adapter source / ABI contract "
        "(backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_tma_adapters_track_descriptor_completion",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "context + decode adapter sources (TMA descriptor slot leases): Cake context "
        "adapter source / ABI contract (backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_jit_selects_main_and_support",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); D256 on cake is "
        "EXPECT_TRTLLM_HEAD_DIM_256 on the unified side",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_route_is_optimized_only_on_exact_bf16_domain",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_exact_sink_no_lse_member_falls_back_for_caller_lse",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_decode_candidate_selection_for_adapter_families",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_fp8_decode_route_requires_exact_full_block_bucket",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_fp16_nhd_route_loads_its_authenticated_adapter",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_fp16_hd512_route_loads_its_authenticated_adapter",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_bf16q_route_loads_its_authenticated_adapter",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_fp8_route_loads_its_authenticated_adapter",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_nvfp4_route_loads_authenticated_adapter",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_nvfp4_load_failure_fails_closed_to_compat",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_candidate_selection_for_adapter_families",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_route_requires_exact_workspace",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); the unified workspace "
        "contract is PagedAttention.workspace_requirements()",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_bf16_exact_route_loads_exact_member",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_fp8_route_omits_bf16_exact_profile",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_route_loads_authenticated_chain",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_route_loads_authenticated_chain",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_load_failure_fails_closed",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_decode_route_miss_fails_closed_to_compat",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_context_route_miss_fails_closed_to_compat",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_public_decode_route_miss_canonicalizes_only_pinned_noop_skip",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_public_context_route_miss_canonicalizes_only_pinned_noop_skip",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "skip-softmax threshold canonicalisation on the context FFI path: Cake "
        "context-route / JIT-member selection contract (backend-private; the unified "
        "layer only calls trtllm_batch_context_with_kv_cache(backend='cake')); "
        "skip-softmax is EXPECT_SKIP_SOFTMAX on the unified side",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_public_nvfp4_loader_failure_materializes_compat_scale_abi",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_route_is_optimized_only_on_exact_bf16_domain",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_exact_mask_profile_requires_uniform_runtime_lengths",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_public_entrypoint_forces_cake_backend",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_public_entrypoint_forces_cake_backend",
        ["test_cake_context_route_selection_is_native_only"],
        "native-only",
        "cake_batch_context_with_kv_cache is a thin monkeypatched forwarder to "
        "backend='cake'; the unified spelling is PagedAttention.plan(backend='cake') "
        "(asserted by attn.backend in every converted row)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_public_symbols_are_top_level",
        [],
        "out-of-scope",
        "Cake product packaging (manifest / registry / AOT / symbols): not paged "
        "prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_aot_registers_each_exact_blackwell_target",
        [],
        "out-of-scope",
        "Cake product packaging (manifest / registry / AOT / symbols): not paged "
        "prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_bf16_matches_flashinfer_reference",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_fully_masked_row_returns_zero",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_exact_sink_matches_independent_reference",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_fp16_nhd_matches_flashinfer_reference",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_decode_fp16_hd512_matches_flashinfer_reference",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_bf16_separate_tables_matches_reference",
        ["test_cake_context_bf16_separate_tables_matches_reference"],
        "partial",
        "independent K/V page-id mappings (the legacy interleaved 2p / 2p+1 layout) "
        "have no unified spelling (EXPECT_INDEPENDENT_KV_TABLES, asserted); the same "
        "fixture (NHD, B2, page32, H4:2, bf16, non-causal, q<=7, kv<=31, seed 0) runs "
        "on pinned cake and on auto (recorded) with the shared table vs the legacy "
        "fa2 reference (1e-2, LSE 1e-3) and the oracle",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_fp8_nhd_device_scale_skip_matches_reference",
        ["test_cake_context_fp8_nhd_device_scale_skip_matches_reference"],
        "unsupported-by-design",
        "fp8 QKV (EXPECT_FP8_Q, asserted on cake for the legacy shape H32:4 D128 "
        "page64 NHD), device-tensor scales (EXPECT_DEVICE_SCALES: run() takes host "
        "floats so the call stays sync-free; asserted on the one fp8-KV path, fa2) "
        "and skip-softmax (EXPECT_SKIP_SOFTMAX) each assert their rejection",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_base_decode_cuda_graph_capture_replay",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def test_cake_context_route_selection_is_native_only():
    """The anchor of the native-only rows: the unified layer names Cake as one
    backend (``backend='cake'`` resolves to exactly that name on sm_100 /
    sm_103, with the probe reason elsewhere) and exposes no route, JIT member,
    manifest or adapter concept -- ``Resolution`` carries backend names and
    exclusion reasons only, so the legacy route-selection contracts have no
    unified observable and stay in the native suite."""
    from flashinfer.experimental.paged_attention import CAPABILITIES
    from flashinfer.prefill import Resolution

    assert "cake" in CAPABILITIES
    fields = set(Resolution.__dataclass_fields__)
    assert fields >= {"backends", "excluded", "config"}
    assert not any(
        token in name for name in fields for token in ("route", "member", "manifest")
    )
    res = resolve_or_skip(
        "cake",
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        page_size=32,
        kv_layout="NHD",
        causal=False,
        need_lse=True,
    )
    assert res.backends == ("cake",)


# ---------------------------------------------------------------------------
# test_cake_fmha.py paged context kernel entries
# ---------------------------------------------------------------------------


def test_cake_context_bf16_separate_tables_matches_reference(record_property) -> None:
    p = legacy_trtllm_problem("NHD", 2, 32, 2, 2, "bf16", 7, 31, 128)
    if independent_tables_rejected(p) is not None:
        pytest.fail(
            "EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here"
        )
    assert not EXPECT_INDEPENDENT_KV_TABLES
    # the same fixture with the shared table on cake (the legacy backend) and auto
    ref, lse_ref = legacy_fa2_paged_reference(p, causal=False)
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    o_out = o_lse = None
    for backend in ("cake", "auto"):
        attn, md = plan_legacy_problem(p, backend, causal=False)
        if backend == "auto":
            record_property("auto_backend", attn.backend)
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
        assert_legacy_close(out, ref)
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
        if o_out is None:
            o_out, o_lse = oracle(
                md,
                p["q"],
                p["ref_kv_cache"][:, 0],
                p["ref_kv_cache"][:, 1],
                causal=False,
                kv_layout="NHD",
                sm_scale=p["sm_scale"],
            )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


def test_cake_context_fp8_nhd_device_scale_skip_matches_reference() -> None:
    """fp8 QKV on cake (EXPECT_FP8_Q), device-tensor scales
    (EXPECT_DEVICE_SCALES: the unified run() takes host floats so the call
    stays sync-free; the rejection is shown on the one fp8-KV path that
    exists, fa2) and skip-softmax (EXPECT_SKIP_SOFTMAX)."""
    res = fp8_q_rejected(
        backend="cake",
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        page_size=64,
        kv_layout="NHD",
        causal=True,
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: port the legacy cake fp8 fixture here")
    assert not EXPECT_FP8_Q
    assert skip_softmax_knob_present() == EXPECT_SKIP_SOFTMAX
    # device scales: an fp8-KV plan on fa2, run with tensor scales
    from .test_paged_attention_prototype import make_metadata, make_problem

    p = make_problem(
        seed=1,
        batch_size=2,
        max_q=8,
        max_kv=64,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        page_size=16,
        dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
    )
    resolve_or_skip(
        "fa2",
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        page_size=16,
        causal=True,
        need_lse=False,
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        make_metadata(p),
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        backend="fa2",
    )
    k_scale_t = torch.tensor(p["k_scale"], device=DEVICE, dtype=torch.float32)
    out = gated(
        EXPECT_DEVICE_SCALES,
        lambda: attn.run(
            p["q"],
            (p["k_cache"], p["v_cache"]),
            k_scale=k_scale_t,
            v_scale=p["v_scale"],
        ),
        match="host float",
    )
    if out is not None:
        pytest.fail("EXPECT_DEVICE_SCALES flipped: assert the device-scale result here")
