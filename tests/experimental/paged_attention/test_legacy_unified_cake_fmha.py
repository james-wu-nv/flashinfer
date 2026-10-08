"""Legacy -> unified: tests/attention/test_cake_fmha.py

Only two legacy tests run the Cake paged context kernel
(``trtllm_batch_context_with_kv_cache(backend="cake")`` through the legacy
trtllm-gen prefill fixture), and both exercise a configuration the unified API
does not express: independent K / V page tables, and fp8 QKV with device-tensor
scales and skip-softmax.  Each asserts the rejection on cake and stops.

The context-route / JIT-spec / manifest-member / adapter-source tests are
native-only: which cubin member serves a shape, which binding source is
compiled and the TMA descriptor bookkeeping are backend-private contracts with
no unified observable.  The decode route / kernel tests and the product
packaging tests (manifest, registry, AOT registration, public symbols) are out
of scope.
"""

import inspect

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from flashinfer.utils import get_compute_capability

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cake_fmha.py"
LEGACY_MAP = [
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
        [],
        "native-only",
        "replays the pinned capability corpus (decode + context cells) through the "
        "product's route selectors: Cake context-route / JIT-member selection "
        "contract (backend-private; the unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_capability_replay_covers_q257_exact_context_profile",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_jit_spec_uses_versioned_standalone_sources",
        [],
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
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_bf16_exact_profile_selector",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_fp8_jit_selects_one_manifest_member",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); fp8 q is rejected "
        "by the unified API",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_jit_selects_fused_member",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); nvfp4 KV has "
        "no unified spelling",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_adapters_match_public_feature_abi",
        [],
        "native-only",
        "Cake context adapter source / ABI contract (backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_optimized_context_adapters_require_signed_seq_lens",
        [],
        "native-only",
        "Cake context adapter source / ABI contract (backend-private csrc "
        "bookkeeping); the unified metadata is int32 by contract",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_typed_launch_adapters_use_typed_tensor_maps",
        [],
        "native-only",
        "context + decode adapter sources: Cake context adapter source / ABI contract "
        "(backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_tma_adapters_track_descriptor_completion",
        [],
        "native-only",
        "context + decode adapter sources (TMA descriptor slot leases): Cake context "
        "adapter source / ABI contract (backend-private csrc bookkeeping)",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_jit_selects_main_and_support",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); D256 on cake is "
        "rejected by the unified API",
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
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_route_requires_exact_workspace",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls "
        "trtllm_batch_context_with_kv_cache(backend='cake')); the unified workspace "
        "contract is PagedAttention.workspace_requirements()",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_bf16_exact_route_loads_exact_member",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_fp8_route_omits_bf16_exact_profile",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_hd256_route_loads_authenticated_chain",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_route_loads_authenticated_chain",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_nvfp4_load_failure_fails_closed",
        [],
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
        [],
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
        [],
        "native-only",
        "skip-softmax threshold canonicalisation on the context FFI path: Cake "
        "context-route / JIT-member selection contract (backend-private; the unified "
        "layer only calls trtllm_batch_context_with_kv_cache(backend='cake')); "
        "skip-softmax has no unified spelling",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_public_nvfp4_loader_failure_materializes_compat_scale_abi",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_fmha_context_route_is_optimized_only_on_exact_bf16_domain",
        [],
        "native-only",
        "Cake context-route / JIT-member selection contract (backend-private; the "
        "unified layer only calls trtllm_batch_context_with_kv_cache(backend='cake'))",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_exact_mask_profile_requires_uniform_runtime_lengths",
        [],
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
        [],
        "native-only",
        "cake_batch_context_with_kv_cache is a thin monkeypatched forwarder to "
        "backend='cake'; the unified spelling is "
        "PagedAttention.plan(backend='cake')",
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
        "unsupported-by-design",
        "the subject is independent K/V page tables (legacy [B, 2, M] table, K at "
        "page 2p, V at 2p + 1); the metadata takes one 2-D table shared by K and "
        "V: the legacy table is rejected",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_fp8_nhd_device_scale_skip_matches_reference",
        ["test_cake_context_fp8_nhd_device_scale_skip_matches_reference"],
        "unsupported-by-design",
        "fp8 QKV on cake is rejected at resolve (q dtype); device-tensor scales and "
        "skip-softmax have no unified spelling either",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_base_decode_cuda_graph_capture_replay",
        [],
        "out-of-scope",
        "Cake decode route / kernel: not paged prefill",
    ),
]


DEVICE = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_context_bf16_separate_tables_matches_reference() -> None:
    from tests.attention.test_trtllm_gen_attention_decode import (
        create_kv_cache,
        create_page_table,
        create_query_tensor,
        generate_cumsum_lens,
        generate_seq_lens_prefill,
        prepare_paged_kv_for_kernel,
    )

    if get_compute_capability(DEVICE)[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    # the legacy fixture (NHD, B2, page 32, H4:2, bf16, q <= 7, kv <= 31, D128)
    torch.manual_seed(0)
    q_lens, _, seq_lens = generate_seq_lens_prefill(2, 7, 31)
    create_query_tensor(q_lens, 4, 128, "bf16")  # drawn to keep the RNG order
    q_indptr = generate_cumsum_lens(q_lens)
    kv_cache, _, _, _, _ = create_kv_cache(
        2, seq_lens, 32, 2, 128, "bf16", "bf16", "NHD"
    )
    page_table, _, _ = create_page_table(2, seq_lens, 32)
    # uses_shared_paged_kv_idx=False: a [B, 2, M] table, K at page 2p, V at 2p + 1
    _, table_kv, _ = prepare_paged_kv_for_kernel(kv_cache, page_table, False)
    assert table_kv.shape == (2, 2, page_table.shape[1])

    # the metadata takes one 2-D table shared by K and V
    with pytest.raises(ValueError, match="block_tables must be 2-D"):
        PagedAttentionMetadata.dense(
            q_indptr,
            seq_lens.to(DEVICE),
            table_kv,
            page_size=32,
            max_q_len=int(q_lens.max()),
            max_kv_len=int(seq_lens.max()),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cake_context_fp8_nhd_device_scale_skip_matches_reference() -> None:
    if get_compute_capability(DEVICE)[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")

    # the legacy shape: NHD, page 64, H32:4, D128, causal, fp8 q / KV / output
    with pytest.raises(ValueError, match="cake: unsupported q dtype"):
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=128,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=64,
            kv_layout="NHD",
            causal=True,
            backend="cake",
        )
    # nor is there a skip-softmax knob
    for fn in (PagedAttention.plan, PagedAttention.run):
        params = inspect.signature(fn).parameters
        assert "skip_softmax_threshold_scale_factor" not in params
