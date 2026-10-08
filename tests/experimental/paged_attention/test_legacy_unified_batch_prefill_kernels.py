"""Legacy -> unified: tests/attention/test_batch_prefill_kernels.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the same
shapes, dtypes, page sizes, the same combined ``(pages, 2, ...)`` pool sliced
as ``K = kv[:, 0]`` / ``V = kv[:, 1]`` views, the same random construction
(under a per-row seed; the legacy tests are unseeded), the legacy CSR mapped
losslessly to ``PagedAttentionMetadata`` (``kv_seq_lens[i] = (pages_i - 1) *
page_size + last_page_len[i]``; page size < 8 -> ``.csr``, else ``.dense``).
Each numerical row asserts against the legacy reference at the legacy
tolerance AND against the independent fp32 oracle.  The parametrize axes
keep the legacy names and values; a row id is the legacy node id plus
``-<backend>`` (six backends; a capability-excluded backend skips with the
resolve reason).  The full legacy grids stay under ``slow`` (FI_PARITY_SLOW=1);
the default subset covers every axis value.

CI: legacy file in A10G fixed shard part4 (full) and H100 1/5 sampling; fa3
rows only on H100.  The unified file is not collected by default CI
(``norecursedirs = tests/experimental``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- pos_encoding_mode="ROPE_LLAMA" (an axis of the main, tuple, head_dim_512,
  custom-mask and multi-item grids; the *_rope_large_head NVFP4 rows): the
  unified API has no fused positional encoding -- plan() takes no
  pos_encoding_mode / rope_scale / rope_theta (TypeError, asserted per row)
  because RoPE is the caller's transform (design doc "Feature axes").  The
  ROPE_LLAMA rows run the migration adapter instead: q rotated at positions
  kv_len - q_len + r and the request's K pages at j with
  flashinfer.apply_rope_pos_ids (Llama non-interleaved, theta 1e4), then the
  unified run, compared with the legacy FUSED reference on the unrotated
  tensors at the legacy 1e-3 -- so these rows are ``partial`` (adapter cost
  measured, fused RoPE not expressible).  Support surface: no backend
  declares a positional-encoding axis (_capabilities.py has none).
- use_cuda_graph=True rows of the main / tuple grids were an unconditional
  legacy xfail (workspace overflow); the unified rows run the graph
  lifecycle (GraphCapacity -> workspace_requirements -> warm-up plan ->
  capture -> update() -> replay) on the same fixture and pass.  The scratch
  the contract asks for is large at the big geometries (fa2 plans the
  split-KV scratch for the capacity maxes: up to 4.5 GB at B128 / kv2048 /
  32 heads), which is exactly why the legacy 128 MiB wrapper xfailed.
- ROPE_LLAMA rows with qo_len > kv_len (the non-causal (17, 54, 577) points):
  the fused kernels compute the q position ``kv_len - qo_len + i`` in uint32
  (prefill.cuh, q_frag_apply_llama_rope), so the legacy reference rotates at
  wrapped positions; the adapter's signed positions match the oracle but not
  that reference -- the rows check the oracle and skip the legacy comparison
  with the reason.
- NVFP4 (8 legacy functions): kv_dtype uint8 (packed FP4x2) is not a declared
  KV dtype of any backend (declared: f16/bf16 everywhere, fp8 e4m3/e5m2 on
  fa2) and run() has no kv_cache_sf; the rows assert the resolve rejection
  under EXPECT_NVFP4_KV.  Needs a quantization descriptor (packed dtype,
  scale-factor tensors and their strides, global scales, asymmetric pools).
- multi-item scoring: plan() has no prefix_len_ptr / token_pos_in_items_ptr
  / token_pos_in_items_len / max_item_len_ptr (TypeError, asserted); the
  visible set is expressed with the legacy mask builder as custom_mask
  (causal=False, fa2 = the only mask-capable backend) plus the RoPE adapter,
  against the legacy fused reference -- ``partial`` under
  EXPECT_MULTI_ITEM_SCORING.
- (448, 256) head dims: undeclared pair on every backend (fa2 declares
  64/128/256/512 square); the CTA_TILE_Q plan_info pin is a planner-internal
  contract with no unified observable -- ``native-only``, numerical half
  under EXPECT_HEAD_DIM_448_256.
- fully masked causal rows (q_len 34 > kv_len 1): rejected by the causal
  envelope ("causal masking requires q_len_i <= kv_len_i"); legacy defined
  them as out 0 / LSE -inf.  Needs a fully-masked-row policy in the contract
  -- EXPECT_FULLY_MASKED_ROWS; the fixture runs non-causally as a sanity row.
- fixed_split_size=2 of the lazy-stride-router test has no unified
  counterpart (the split policy is the backend's); the plan-reuse contract
  itself is checked.
- ragged / single-prefill functions of the legacy file are out of scope
  (not paged prefill) and listed as such.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention
from tests.test_helpers.paged_kv import make_padded_paged_kv_view
from tests.test_helpers.test_helpers import ref_single_prefill

from .legacy_unified_helpers import (
    BACKENDS,
    DEVICE,
    EXPECT_FA2_HEAD_DIM_512,
    EXPECT_FULLY_MASKED_ROWS,
    EXPECT_HEAD_DIM_448_256,
    EXPECT_MULTI_ITEM_SCORING,
    LegacyBatch,
    apply_external_rope,
    argnames,
    assert_every_backend_excluded,
    assert_legacy_isclose,
    assert_nvfp4_unsupported,
    assert_oracle,
    assert_rope_kwargs_rejected,
    backend_rows,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    grid,
    legacy_id,
    legacy_reference_single_prefill,
    legacy_uniform_batch,
    param_rows,
    plan_batch,
    plan_signature_params,
    resolve_batch_or_skip,
    run_batch,
    run_batch_graph,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_batch_prefill_kernels.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache",
        ["test_batch_prefill_with_paged_kv_cache"],
        "partial",
        "same 14 legacy axes and values (B12/17/128 x kv54/97/512/2048 x q37/17/127/577 "
        "x page1/5/16 x H4/32:4 x D64/128/256 x causal x NHD x pos NONE/ROPE_LLAMA x "
        "use_cuda_graph x soft cap 0 x LSE x contiguous) plus the backend; combined fp16 "
        "NHD pool as K=kv[:,0]/V=kv[:,1] views; legacy 1e-3 vs per-request "
        "single_prefill_with_kv_cache + oracle + the legacy caller-buffer re-run.  NONE "
        "rows are equivalent; ROPE_LLAMA rows run the external-RoPE adapter against the "
        "fused legacy reference (fused RoPE not expressible); use_cuda_graph=True rows "
        "(a legacy xfail) run the graph lifecycle (workspace sized by "
        "workspace_requirements, up to 4.5 GB) and pass; ROPE rows with qo_len > kv_len "
        "check the oracle only (the fused reference wraps positions in uint32); cudnn / "
        "trtllm-gen / cake run the D128 / page16 cells and skip the rest with the "
        "capability reason.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_plan_reuse",
        ["test_batch_prefill_lazy_stride_router_plan_reuse"],
        "equivalent",
        "same fixture (bf16 /4, B2 q17 kv97 page16 H8:2, padded V view via "
        "make_padded_paged_kv_view), one plan / three runs equal -> unequal -> equal "
        "strides, legacy tolerances (2e-2 vs fp64 ref_single_prefill, 1e-2 pairwise) + "
        "oracle on fa2; fa3 / trtllm-gen / cake must reject-or-correct the unequal run; "
        "the legacy fixed_split_size=2 knob has no counterpart.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_nvfp4",
        ["test_batch_prefill_lazy_stride_router_nvfp4"],
        "unsupported-by-design",
        "NVFP4 KV (uint8 packed + scale factors) is undeclared and run() has no "
        "kv_cache_sf; the stride-router prewarm (prewarm_paged_kv_stride_variant) is a "
        "legacy wrapper knob with no unified counterpart.  EXPECT_NVFP4_KV.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_head_dim_512",
        ["test_batch_prefill_with_paged_kv_cache_head_dim_512"],
        "partial",
        "same axes (causal x pos NONE/ROPE_LLAMA) and fixture (B2 kv97 q17 page16 H4:4 "
        "NHD fp16); (512, 512) is declared on fa2 since WP-T, so the NONE rows run at "
        "the legacy 1e-3 vs single_prefill(backend=fa2) + oracle + caller buffers "
        "(EXPECT_FA2_HEAD_DIM_512 positive branch); ROPE rows via the external adapter; "
        "cudnn / trtllm-gen / cake skip (D512 undeclared).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_tuple_paged_kv_cache",
        ["test_batch_prefill_with_tuple_paged_kv_cache"],
        "partial",
        "as the main grid (D128/256) with two separately allocated fp16 NHD pools; NONE "
        "rows equivalent, ROPE rows through the adapter, graph rows pass.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_custom_mask",
        ["test_batch_prefill_with_paged_kv_cache_custom_mask"],
        "partial",
        "same 12 legacy axes (page1/16, D128/256, H4/32:4, pos NONE/ROPE_LLAMA); the "
        "legacy tril mask as custom_mask with causal=False (the unified mask is ANDed "
        "into the envelope; legacy CUSTOM replaced causal) vs the causal=True plan at "
        "1e-3 (legacy assertion) + oracle for both; ROPE rows on the adapter-rotated "
        "tensors; fa2 is the only mask-capable backend (as for the legacy kernel), the "
        "others skip with 'custom attention mask not supported'.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache",
        [],
        "out-of-scope",
        "ragged (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_head_dim_512",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_custom_mask",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring"],
        "partial",
        "same legacy axes (two item fixtures, page1/5/16, H4/32:4, D128, causal, "
        "ROPE_LLAMA, soft cap 0/30, LSE on/off); plan() has none of the item-position "
        "fields (TypeError asserted, EXPECT_MULTI_ITEM_SCORING); the visible set is "
        "expressed with the legacy mask builder as custom_mask (causal=False) on the "
        "adapter-rotated tensors vs the legacy fused reference (single_prefill with the "
        "same mask, ROPE_LLAMA) at 1e-3 + oracle; fa2 only (mask-capable).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4"],
        "unsupported-by-design",
        "same 9 legacy axes; kv_dtype uint8 (packed FP4x2) is not a declared KV dtype "
        "(every backend excluded with the dtype reason) and run() has no kv_cache_sf; "
        "needs a quantization descriptor.  EXPECT_NVFP4_KV.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views"],
        "unsupported-by-design",
        "as nvfp4 (B2 kv33 q17 page16 H4:2 D128, NHD/HND); additionally needs "
        "independent scale-factor strides.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric"],
        "unsupported-by-design",
        "as nvfp4 (B2 kv99 q33, bf16 q); additionally the (512, 256) / (256, 128) "
        "head-dim pairs are undeclared on every backend.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256",
        ["test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256"],
        "native-only",
        "(448, 256) is not a declared head-dim pair (resolve excludes it; the fp8 row "
        "names the KV dtype first); the CTA_TILE_Q plan_info assertion is a "
        "planner-internal contract with no unified observable -- the numerical half "
        "(fp16 KV, 2e-3 vs fp32) is in place under EXPECT_HEAD_DIM_448_256.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides",
        ["test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides"],
        "equivalent",
        "same fixture (D512 fp16, K and V views of differently padded parents, NHD/HND, "
        "q17/65) on fa2 vs the exact fp32 reference at 2e-3 (legacy) + oracle; D512 is "
        "declared on fa2 since WP-T (EXPECT_FA2_HEAD_DIM_512); other backends skip (D512 "
        "undeclared).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_large_head"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 (B1 kv128 q64 page16 H1:1 fp16).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 (bf16 q).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 plus fused ROPE_LLAMA.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 plus fused ROPE_LLAMA (bf16 q).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4_large_head",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4_rope_large_head",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_single_prefill_torch_compile_cuda_graph",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache under torch.compile, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_ragged_prefill_one_valid_key",
        [],
        "out-of-scope",
        "ragged, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_fully_masked_rows",
        ["test_paged_prefill_fully_masked_rows"],
        "unsupported-by-design",
        "causal with q_len 34 > kv_len 1 is rejected by validate_causal_envelope "
        "('causal masking requires q_len_i <= kv_len_i'); legacy defined the 33 fully "
        "masked rows as out 0 / LSE -inf.  Needs an explicit fully-masked-row policy "
        "(and an oracle without NaN for them); the legacy assertions are in place under "
        "EXPECT_FULLY_MASKED_ROWS.  The same fixture runs non-causally.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_split_kv_empty_chunk",
        ["test_paged_prefill_split_kv_empty_chunk"],
        "equivalent",
        "same fixture (q/10, combined NHD kv/10, B1 q2 kv129 page16 H8:2 D128, fp16 and "
        "bf16), 1e-2 vs fp64 ref_single_prefill for out AND LSE + oracle; dense and CSR "
        "forms (added axis), every backend.",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


# ---------------------------------------------------------------------------
# test_batch_prefill_with_paged_kv_cache / test_batch_prefill_with_tuple_paged_kv_cache
# ---------------------------------------------------------------------------

# legacy decorator order, top-down (the row id reverses it, as pytest does)
MAIN_AXES = dict(
    batch_size=[12, 17, 128],
    kv_len=[54, 97, 512, 2048],
    qo_len=[37, 17, 127, 577],
    page_size=[1, 5, 16],
    num_kv_heads=[4],
    num_qo_heads=[4, 32],
    head_dim=[64, 128, 256],
    causal=[False, True],
    kv_layout=["NHD"],
    pos_encoding_mode=["NONE", "ROPE_LLAMA"],
    use_cuda_graph=[False, True],
    logits_soft_cap=[0.0],
    return_lse=[True],
    contiguous_kv=[True],
)
# default subset: every batch / kv / qo value at least once, including a
# q > kv non-causal point (its causal twin skips exactly as in legacy);
# every other axis is crossed fully
MAIN_DEFAULT_TRIPLES = {
    (12, 54, 37),
    (17, 97, 17),
    (128, 512, 127),
    (12, 2048, 577),
    (17, 54, 577),
}


def _main_default(point):
    return tuple(point[:3]) in MAIN_DEFAULT_TRIPLES


def _run_main_grid(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
    *,
    combined,
    check_caller_buffers,
):
    assert (
        kv_layout == "NHD" and logits_soft_cap == 0.0 and return_lse and contiguous_kv
    )
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")  # legacy skip
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=kv_layout,
        combined=combined,
        seed=seed_of(
            "main" if combined else "tuple",
            batch_size,
            kv_len,
            qo_len,
            page_size,
            num_qo_heads,
            head_dim,
            causal,
        ),
    )
    md = lb.metadata()
    if pos_encoding_mode == "ROPE_LLAMA":
        assert_rope_kwargs_rejected(lb, md)
        q, k = apply_external_rope(lb)
    else:
        q, k = lb.q, lb.k
    if use_cuda_graph:
        attn, out, lse = run_batch_graph(lb, md, backend, q=q, k=k, causal=causal)
    else:
        attn, out, lse = run_batch(lb, md, backend, q=q, k=k, causal=causal)
    assert_oracle(lb, out, lse, causal=causal, q=q, k=k)
    if pos_encoding_mode == "ROPE_LLAMA" and qo_len > kv_len:
        # the fused kernels compute the q position kv_len - qo_len + i in
        # uint32 (include/flashinfer/attention/prefill.cuh,
        # q_frag_apply_llama_rope), so with qo_len > kv_len the legacy
        # reference rotates at wrapped positions; the adapter's signed
        # positions match the oracle but cannot match that reference
        pytest.skip(
            "ROPE_LLAMA with qo_len > kv_len: the legacy fused reference wraps "
            "kv_len - qo_len + i in unsigned arithmetic; the external adapter "
            "(signed positions) matches the oracle, the legacy comparison is "
            "undefined"
        )
    ref = legacy_reference_single_prefill(
        lb, causal=causal, pos_encoding_mode=pos_encoding_mode
    )
    assert_legacy_isclose(out, ref, rtol=1e-3, atol=1e-3)
    if check_caller_buffers and not use_cuda_graph:
        # legacy: a second run into pre-allocated out / lse buffers matches
        out_buf, lse_buf = torch.empty_like(out), torch.empty_like(lse)
        o2, l2 = attn.run(q, (k, lb.v), out=out_buf, lse=lse_buf)
        assert o2 is out_buf and l2 is lse_buf
        assert_legacy_isclose(
            out, out_buf, rtol=1e-3, atol=1e-3, what="caller out buffer"
        )
        assert_legacy_isclose(
            lse, lse_buf, rtol=1e-3, atol=1e-3, what="caller lse buffer"
        )


@pytest.mark.parametrize(
    argnames(MAIN_AXES, "backend"), param_rows(MAIN_AXES, _main_default)
)
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
):
    """Combined fp16 NHD pool as K/V views; NONE rows equivalent, ROPE rows
    through the external adapter, graph rows through the graph lifecycle."""
    _run_main_grid(
        batch_size,
        kv_len,
        qo_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        causal,
        kv_layout,
        pos_encoding_mode,
        use_cuda_graph,
        logits_soft_cap,
        return_lse,
        contiguous_kv,
        backend,
        combined=True,
        check_caller_buffers=True,
    )


@pytest.mark.parametrize(
    argnames(MAIN_AXES, "backend"),
    param_rows(dict(MAIN_AXES, head_dim=[128, 256]), _main_default),
)
def test_batch_prefill_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
):
    """Two separately allocated fp16 NHD pools (the legacy tuple form)."""
    _run_main_grid(
        batch_size,
        kv_len,
        qo_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        causal,
        kv_layout,
        pos_encoding_mode,
        use_cuda_graph,
        logits_soft_cap,
        return_lse,
        contiguous_kv,
        backend,
        combined=False,
        check_caller_buffers=False,
    )


# ---------------------------------------------------------------------------
# test_batch_prefill_lazy_stride_router_plan_reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kv_layout,head_dim,backend",
    backend_rows("NHD", 64, ids="NHD-64") + backend_rows("HND", 128, ids="HND-128"),
)
def test_batch_prefill_lazy_stride_router_plan_reuse(kv_layout, head_dim, backend):
    """One plan, three runs over (k, v_equal), (k, v_unequal), (k, v_equal):
    the unified plan is by construction independent of the pool strides.
    fa2 must run every step (the legacy backend); a backend whose kernels
    require equal K/V stride families (fa3, trtllm-gen, cake) must reject the
    unequal run with a ValueError, never misread it."""
    torch.manual_seed(42)
    batch_size, qo_len, kv_len, page_size = 2, 17, 97, 16
    num_qo_heads, num_kv_heads = 8, 2
    pages_per_request = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_request
    dev = torch.device(DEVICE)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=dev, dtype=torch.bfloat16
    )
    if kv_layout == "NHD":
        cache_shape = (total_pages, page_size, num_kv_heads, head_dim)
    else:
        cache_shape = (total_pages, num_kv_heads, page_size, head_dim)
    k = torch.randn(cache_shape, device=dev, dtype=torch.bfloat16) / 4
    v_equal = torch.randn(cache_shape, device=dev, dtype=torch.bfloat16) / 4
    v_unequal = make_padded_paged_kv_view(v_equal, kv_layout)
    assert k.stride() == v_equal.stride() and k.stride() != v_unequal.stride()

    lb = LegacyBatch(
        q=q,
        k=k,
        v=v_equal,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_request,
        kv_indices_cpu=torch.arange(total_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout=kv_layout,
    )
    md = lb.metadata()
    attn = plan_batch(lb, md, backend, causal=True)
    chosen = attn.backend

    expected = torch.cat(
        [
            ref_single_prefill(lb.request_q(i), *lb.request_kv(i), causal=True)[0]
            for i in range(batch_size)
        ]
    )
    outputs = []
    for step, cache in enumerate(((k, v_equal), (k, v_unequal), (k, v_equal))):
        try:
            out, lse = attn.run(q, cache)
        except ValueError as e:
            assert step == 1 and chosen != "fa2", (
                f"{chosen} rejected a run it must support: {e}"
            )
            assert "stride" in str(e).lower(), e
            outputs.append(None)
            continue
        assert attn.backend == chosen  # no re-plan, no backend switch
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)  # legacy
        assert_oracle(lb, out, lse, causal=True, v=cache[1])
        outputs.append(out)
    assert outputs[0] is not None and outputs[2] is not None
    if outputs[1] is not None:
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-2, atol=1e-2)
    elif chosen == "fa2":
        raise AssertionError("fa2 must run the unequal-stride V view")
    torch.testing.assert_close(outputs[0], outputs[2], rtol=1e-2, atol=1e-2)


def test_batch_prefill_lazy_stride_router_nvfp4():
    """Legacy: B1 q17 kv33 page16 H4:2 D128 fp16 q with a packed NVFP4 pool
    and the prewarm knob.  Unified: the KV dtype is undeclared (rejection)."""
    assert_nvfp4_unsupported(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=16,
        q_dtype=torch.float16,
        causal=True,
        what="lazy_stride_router_nvfp4 (prewarm_paged_kv_stride_variant)",
    )


# ---------------------------------------------------------------------------
# test_batch_prefill_with_paged_kv_cache_head_dim_512
# ---------------------------------------------------------------------------

HD512_AXES = dict(causal=[False, True], pos_encoding_mode=["NONE", "ROPE_LLAMA"])


@pytest.mark.parametrize(
    argnames(HD512_AXES, "backend"), param_rows(HD512_AXES, lambda p: True)
)
def test_batch_prefill_with_paged_kv_cache_head_dim_512(
    causal, pos_encoding_mode, backend
):
    """B2 kv97 q17 page16 H4:4 NHD fp16 at D512: the legacy 1e-3 vs
    single_prefill(backend="fa2") + oracle + the caller-buffer re-run.
    Positive branch of EXPECT_FA2_HEAD_DIM_512 (WP-T declared (512, 512) on
    fa2); the rejection branch is kept for the record."""
    if not EXPECT_FA2_HEAD_DIM_512:
        assert_every_backend_excluded(
            "unsupported head dims (512, 512)",
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=512,
            head_dim_vo=512,
            q_dtype=torch.float16,
            page_size=16,
            kv_layout="NHD",
            causal=causal,
            need_lse=True,
        )
        return
    lb = legacy_uniform_batch(
        batch_size=2,
        kv_len=97,
        qo_len=17,
        page_size=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=512,
        seed=seed_of("hd512", causal),
    )
    md = lb.metadata()
    if pos_encoding_mode == "ROPE_LLAMA":
        assert_rope_kwargs_rejected(lb, md)
        q, k = apply_external_rope(lb)
    else:
        q, k = lb.q, lb.k
    attn, out, lse = run_batch(lb, md, backend, q=q, k=k, causal=causal)
    ref = legacy_reference_single_prefill(
        lb, causal=causal, backend="fa2", pos_encoding_mode=pos_encoding_mode
    )
    assert_legacy_isclose(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=causal, q=q, k=k)
    out_buf, lse_buf = torch.empty_like(out), torch.empty_like(lse)
    attn.run(q, (k, lb.v), out=out_buf, lse=lse_buf)
    torch.testing.assert_close(out, out_buf, rtol=1e-3, atol=1e-3)  # legacy
    torch.testing.assert_close(lse, lse_buf, rtol=1e-3, atol=1e-3)  # legacy


# ---------------------------------------------------------------------------
# test_batch_prefill_with_paged_kv_cache_custom_mask
# ---------------------------------------------------------------------------

CUSTOM_MASK_AXES = dict(
    batch_size=[12, 17, 128],
    kv_len=[54, 97, 512, 2048],
    qo_len=[37, 17, 127, 577],
    page_size=[1, 16],
    num_kv_heads=[4],
    num_qo_heads=[4, 32],
    head_dim=[128, 256],
    kv_layout=["NHD"],
    pos_encoding_mode=["NONE", "ROPE_LLAMA"],
    logits_soft_cap=[0.0],
    return_lse=[True],
    contiguous_kv=[True],
)
CUSTOM_MASK_DEFAULT_TRIPLES = {(12, 54, 37), (17, 97, 17), (128, 512, 127)}


@pytest.mark.parametrize(
    argnames(CUSTOM_MASK_AXES, "backend"),
    param_rows(CUSTOM_MASK_AXES, lambda p: tuple(p[:3]) in CUSTOM_MASK_DEFAULT_TRIPLES),
)
def test_batch_prefill_with_paged_kv_cache_custom_mask(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    backend,
):
    """The legacy bottom-right tril mask passed as ``custom_mask`` (with
    ``causal=False``, since the unified mask is ANDed into the envelope and
    the legacy CUSTOM mode replaced causal) must equal the ``causal=True``
    plan at 1e-3, and both match the oracle."""
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported for custom mask test")  # legacy
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seed=seed_of(
            "custom", batch_size, kv_len, qo_len, page_size, num_qo_heads, head_dim
        ),
    )
    md = lb.metadata()
    if pos_encoding_mode == "ROPE_LLAMA":
        assert_rope_kwargs_rejected(lb, md)
        q, k = apply_external_rope(lb)
    else:
        q, k = lb.q, lb.k
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device=DEVICE),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)
    _, out_custom, lse_custom = run_batch(
        lb, md, backend, q=q, k=k, causal=False, custom_mask=custom_mask
    )
    _, out_causal, lse_causal = run_batch(lb, md, backend, q=q, k=k, causal=True)
    assert_legacy_isclose(out_custom, out_causal, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(
        lb, out_custom, lse_custom, causal=False, custom_mask=custom_mask, q=q, k=k
    )
    assert_oracle(lb, out_causal, lse_causal, causal=True, q=q, k=k)


# ---------------------------------------------------------------------------
# test_batch_prefill_with_paged_kv_cache_multi_item_scoring
# ---------------------------------------------------------------------------

MULTI_ITEM_FIXTURES = [
    # (kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr)
    (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
    (97, 81, 16, list(range(80)) + [0], 97, [79]),
]
MULTI_ITEM_AXES = dict(
    page_size=[1, 5, 16],
    num_kv_heads=[4],
    num_qo_heads=[4, 32],
    head_dim=[128],
    causal=[True],
    kv_layout=["NHD"],
    pos_encoding_mode=["ROPE_LLAMA"],
    logits_soft_cap=[0.0, 30.0],
    return_lse=[True, False],
)


def _multi_item_rows():
    rows = []
    for fi, fixture in enumerate(MULTI_ITEM_FIXTURES):
        kv_len, qo_len, prefix, items, items_len, max_item = fixture
        fixture_id = (
            f"{kv_len}-{qo_len}-{prefix}-token_pos_in_items_ptr{fi}-{items_len}-"
            f"max_item_len_ptr{fi}"
        )
        for point in grid(MULTI_ITEM_AXES):
            for backend in BACKENDS:
                rows.append(
                    pytest.param(
                        1,
                        *fixture,
                        *point,
                        backend,
                        id=f"{legacy_id(MULTI_ITEM_AXES, point)}-{fixture_id}-1-{backend}",
                    )
                )
    return rows


def create_2D_multi_item_mask_dense(
    is_delimiter, sliding_window_size=-1, prefix_cache_len=None
):
    """Verbatim from the legacy test: the multi-item visible set as a dense
    boolean mask (within-item causal, every item sees the prefix, delimiters
    see and are seen by nothing) with the prefix-cache patch prepended."""
    delimiter_idx = is_delimiter.nonzero(as_tuple=True)[0]
    if len(delimiter_idx) == 0:
        return None
    first_delimiter_pos = delimiter_idx[0]
    seq_len = len(is_delimiter)
    pos = torch.arange(seq_len, device=is_delimiter.device)
    group_ids = torch.cumsum(is_delimiter, 0)
    within_group_causal = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0)) & (
        pos.unsqueeze(0) <= pos.unsqueeze(1)
    )
    attention_mask = (
        (
            within_group_causal
            | (
                (pos >= first_delimiter_pos).unsqueeze(1)
                & (pos < first_delimiter_pos).unsqueeze(0)
            )
        )
        & ~is_delimiter.unsqueeze(0)
        & ~is_delimiter.unsqueeze(1)
    )
    if sliding_window_size > 0 and sliding_window_size < len(is_delimiter):
        group_size = torch.sum(within_group_causal & ~is_delimiter.unsqueeze(0), dim=1)
        prefix_window = torch.where(
            pos >= first_delimiter_pos,
            sliding_window_size - group_size,
            torch.where(
                pos < sliding_window_size, first_delimiter_pos, sliding_window_size
            ),
        )
        prefix_start = first_delimiter_pos - prefix_window.unsqueeze(1)
        attention_mask = attention_mask & (pos >= prefix_start)
    if prefix_cache_len:
        patch = torch.ones(
            seq_len, prefix_cache_len, device=is_delimiter.device, dtype=torch.bool
        )
        attention_mask = torch.concat([patch, attention_mask], dim=1)
    return attention_mask.unsqueeze(0).reshape(-1)


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,prefix_len_ptr,token_pos_in_items_ptr,"
    "token_pos_in_items_len,max_item_len_ptr,page_size,num_kv_heads,num_qo_heads,"
    "head_dim,causal,kv_layout,pos_encoding_mode,logits_soft_cap,return_lse,backend",
    _multi_item_rows(),
)
def test_batch_prefill_with_paged_kv_cache_multi_item_scoring(
    batch_size,
    kv_len,
    qo_len,
    prefix_len_ptr,
    token_pos_in_items_ptr,
    token_pos_in_items_len,
    max_item_len_ptr,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    backend,
):
    """plan() has none of the item-position fields (TypeError); the visible
    set is the legacy mask builder's dense mask as ``custom_mask`` with
    ``causal=False``, on the adapter-rotated tensors (the legacy rows are
    ROPE_LLAMA-only), vs the legacy fused reference
    ``single_prefill_with_kv_cache(custom_mask=..., ROPE_LLAMA)`` at 1e-3."""
    params = plan_signature_params()
    for name in (
        "prefix_len_ptr",
        "token_pos_in_items_ptr",
        "token_pos_in_items_len",
        "max_item_len_ptr",
    ):
        assert name not in params, f"plan() grew {name!r}: port the multi-item rows"
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=kv_layout,
        seed=seed_of(
            "multi",
            kv_len,
            qo_len,
            page_size,
            num_qo_heads,
            logits_soft_cap,
            return_lse,
        ),
    )
    md = lb.metadata()
    with pytest.raises(TypeError, match="prefix_len_ptr"):
        PagedAttention(torch.device(DEVICE)).plan(
            md,
            **lb.plan_kwargs(),
            causal=causal,
            prefix_len_ptr=torch.tensor([prefix_len_ptr]).to(torch.uint32).to(DEVICE),
            backend="fa2",
        )
    if EXPECT_MULTI_ITEM_SCORING:
        pytest.fail("EXPECT_MULTI_ITEM_SCORING is set: port the item-position fields")
    mask = create_2D_multi_item_mask_dense(
        is_delimiter=torch.tensor(token_pos_in_items_ptr).to(DEVICE) == 0,
        sliding_window_size=-1,
        prefix_cache_len=prefix_len_ptr,
    )
    assert mask.numel() == qo_len * kv_len
    assert pos_encoding_mode == "ROPE_LLAMA"
    assert_rope_kwargs_rejected(lb, md)
    q, k = apply_external_rope(lb)
    cap = None if logits_soft_cap == 0.0 else logits_soft_cap
    lse_mode = "base2" if return_lse else "none"
    _, out, lse = run_batch(
        lb,
        md,
        backend,
        q=q,
        k=k,
        causal=False,
        custom_mask=mask,
        logits_soft_cap=cap,
        lse_mode=lse_mode,
    )
    ref = legacy_reference_single_prefill(
        lb,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        custom_mask=lambda i: mask,
        pos_encoding_mode=pos_encoding_mode,
    )
    assert_legacy_isclose(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(
        lb,
        out,
        lse,
        causal=False,
        custom_mask=mask,
        logits_soft_cap=cap,
        lse_mode=lse_mode,
        q=q,
        k=k,
    )


# ---------------------------------------------------------------------------
# NVFP4 KV cache (packed uint8 + scale factors): eight legacy functions
# ---------------------------------------------------------------------------

NVFP4_AXES = dict(
    batch_size=[1, 4],
    kv_len=[128, 256],
    qo_len=[64, 128],
    page_size=[16, 64],
    num_kv_heads=[1],
    num_qo_heads=[1],
    head_dim=[128],
    causal=[False],
    q_dtype=[torch.float16, torch.bfloat16],
)


@pytest.mark.parametrize(
    argnames(NVFP4_AXES),
    [
        pytest.param(*point, id=legacy_id(NVFP4_AXES, point))
        for point in grid(NVFP4_AXES)
    ],
)
def test_batch_prefill_with_paged_kv_cache_nvfp4(
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
        pytest.skip("qo_len > kv_len and causal is not supported")  # legacy
    assert_nvfp4_unsupported(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=page_size,
        q_dtype=q_dtype,
        causal=causal,
        what="nvfp4",
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views(kv_layout):
    assert_nvfp4_unsupported(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=16,
        q_dtype=torch.float16,
        causal=True,
        kv_layout=kv_layout,
        what="nvfp4_strided_scale_views (independent scale-factor strides)",
    )


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(512, 256), (256, 128)])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [True])
def test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric(
    head_dim_qk, head_dim_vo, page_size, num_kv_heads, causal
):
    assert_nvfp4_unsupported(
        num_qo_heads=2 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        page_size=page_size,
        q_dtype=torch.bfloat16,
        causal=causal,
        what=f"nvfp4_asymmetric ({head_dim_qk}, {head_dim_vo})",
    )


def _nvfp4_large_head(q_dtype, what):
    assert_nvfp4_unsupported(
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=512,
        head_dim_vo=512,
        page_size=16,
        q_dtype=q_dtype,
        causal=False,
        what=what,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head():
    _nvfp4_large_head(torch.float16, "nvfp4_large_head (D512)")


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16():
    _nvfp4_large_head(torch.bfloat16, "nvfp4_large_head_bf16 (D512)")


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head():
    _nvfp4_large_head(torch.float16, "nvfp4_rope_large_head (D512 + ROPE_LLAMA)")


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16():
    _nvfp4_large_head(torch.bfloat16, "nvfp4_rope_large_head_bf16 (D512 + ROPE_LLAMA)")


# ---------------------------------------------------------------------------
# test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
def test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256(kv_dtype):
    """The (448, 256) pair is undeclared, so resolve excludes it; the
    CTA_TILE_Q plan_info pin stays native.  Flipped: the fp16 numerical half
    at 2e-3 vs fp32 (legacy) + oracle."""
    cfg = dict(
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim_qk=448,
        head_dim_vo=256,
        q_dtype=torch.float16,
        kv_dtype=kv_dtype,
        page_size=16,
        kv_layout="NHD",
        causal=False,
    )
    if not EXPECT_HEAD_DIM_448_256:
        # backends without fp8 KV name the dtype first (checked before head dims)
        assert_every_backend_excluded(
            "unsupported head dims (448, 256)", also=("unsupported kv dtype",), **cfg
        )
        return
    if kv_dtype != torch.float16:
        pytest.skip("the fp8 case is a plan_info (CTA tile) pin: native-only")
    torch.manual_seed(42)
    dev = torch.device(DEVICE)
    batch_size, qo_len, kv_len, page_size, H = 2, 8, 65, 16, 2
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    q = torch.randn(batch_size * qo_len, H, 448, device=dev, dtype=torch.float16)
    k = torch.randn(total_pages, page_size, H, 448, device=dev, dtype=torch.float16)
    v = torch.randn(total_pages, page_size, H, 256, device=dev, dtype=torch.float16)
    lb = LegacyBatch(
        q=q,
        k=k,
        v=v,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_seq,
        kv_indices_cpu=torch.arange(0, total_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout="NHD",
    )
    _, out, lse = run_batch(lb, lb.metadata(), "fa2", causal=False)
    for i in range(batch_size):
        qi = lb.request_q(i).float()
        ki, vi = (t.float() for t in lb.request_kv(i))
        logits = torch.einsum("qhd,khd->hqk", qi, ki) * 448**-0.5
        o_ref_i = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), vi)
        torch.testing.assert_close(
            out[lb.q_indptr_cpu[i] : lb.q_indptr_cpu[i + 1]].float(),
            o_ref_i,
            rtol=2e-3,
            atol=2e-3,
        )
    assert_oracle(lb, out, lse, causal=False)


# ---------------------------------------------------------------------------
# test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides
# ---------------------------------------------------------------------------

SHARED_KV_SMEM_AXES = dict(kv_layout=["NHD", "HND"], qo_len=[17, 65])


@pytest.mark.parametrize(
    argnames(SHARED_KV_SMEM_AXES, "backend"),
    param_rows(SHARED_KV_SMEM_AXES, lambda p: True),
)
def test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides(
    kv_layout, qo_len, backend
):
    """D512 fp16, K and V views of differently padded parents (unequal
    stride families): the exact fp32 reference at 2e-3 (legacy) + oracle."""
    if not EXPECT_FA2_HEAD_DIM_512:
        assert_every_backend_excluded(
            "unsupported head dims (512, 512)",
            num_qo_heads=2,
            num_kv_heads=2,
            head_dim_qk=512,
            head_dim_vo=512,
            q_dtype=torch.float16,
            page_size=16,
            kv_layout=kv_layout,
            causal=True,
            need_lse=True,
        )
        return
    torch.manual_seed(42)
    dev = torch.device(DEVICE)
    head_dim, batch_size, kv_len, page_size, num_kv_heads, num_qo_heads = (
        512,
        2,
        97,
        16,
        2,
        2,
    )
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=dev, dtype=torch.float16
    )

    def padded_pool(num_padding_heads):
        if kv_layout == "NHD":
            parent = torch.randn(
                total_pages,
                page_size,
                num_kv_heads + num_padding_heads,
                head_dim,
                device=dev,
                dtype=torch.float16,
            )
            return parent[:, :, :num_kv_heads, :]
        parent = torch.randn(
            total_pages,
            num_kv_heads + num_padding_heads,
            page_size,
            head_dim,
            device=dev,
            dtype=torch.float16,
        )
        return parent[:, :num_kv_heads, :, :]

    k, v = padded_pool(1), padded_pool(3)
    assert k.stride() != v.stride()
    lb = LegacyBatch(
        q=q,
        k=k,
        v=v,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_seq,
        kv_indices_cpu=torch.arange(0, total_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout=kv_layout,
    )
    _, out, lse = run_batch(lb, lb.metadata(), backend, causal=True)
    sm_scale = head_dim**-0.5
    for i in range(batch_size):
        qi = lb.request_q(i).float()
        ki, vi = (t.float() for t in lb.request_kv(i))
        logits = torch.einsum("qhd,khd->hqk", qi, ki) * sm_scale
        qpos = torch.arange(qo_len, device=dev).unsqueeze(1)
        kpos = torch.arange(kv_len, device=dev).unsqueeze(0)
        logits = logits.masked_fill(
            ~(kpos <= qpos + (kv_len - qo_len)).unsqueeze(0), float("-inf")
        )
        o_ref_i = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), vi)
        torch.testing.assert_close(
            out[lb.q_indptr_cpu[i] : lb.q_indptr_cpu[i + 1]].float(),
            o_ref_i,
            rtol=2e-3,
            atol=2e-3,
        )
    assert_oracle(lb, out, lse, causal=True)


# ---------------------------------------------------------------------------
# test_paged_prefill_fully_masked_rows
# ---------------------------------------------------------------------------

FULLY_MASKED_AXES = dict(dtype=[torch.float16, torch.bfloat16])


def _fully_masked_fixture(dtype):
    qo_len, kv_len = 34, 1
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size, num_pages = 1, 2
    dev = torch.device(DEVICE)
    q = torch.zeros(qo_len, num_qo_heads, head_dim, dtype=dtype, device=dev)
    k_cache = torch.zeros(
        num_pages, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev
    )
    v_cache = torch.ones_like(k_cache)
    return LegacyBatch(
        q=q,
        k=k_cache,
        v=v_cache,
        q_indptr_cpu=torch.tensor([0, qo_len], dtype=torch.int32),
        kv_indptr_cpu=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices_cpu=torch.tensor([1], dtype=torch.int32),
        last_page_len_cpu=torch.tensor([kv_len], dtype=torch.int32),
        page_size=page_size,
        kv_layout="NHD",
    ), qo_len - kv_len


@pytest.mark.parametrize(
    argnames(FULLY_MASKED_AXES, "backend"),
    param_rows(FULLY_MASKED_AXES, lambda p: True),
)
def test_paged_prefill_fully_masked_rows(dtype, backend):
    """q34 / kv1 / page 1.  Today the causal plan is rejected by the
    envelope; the non-causal plan on the same fixture runs (every row attends
    the single all-ones value: out 1, LSE 0).  Flipped: the legacy assertions
    (33 rows out 0 / LSE -inf, last row 1 / 0)."""
    lb, num_masked = _fully_masked_fixture(dtype)
    md = lb.metadata()
    resolve_batch_or_skip(lb, md, backend, causal=True)
    if not EXPECT_FULLY_MASKED_ROWS:
        with pytest.raises(
            ValueError, match="causal masking requires q_len_i <= kv_len_i"
        ):
            PagedAttention(torch.device(DEVICE)).plan(
                md, **lb.plan_kwargs(), causal=True, lse_mode="base2", backend=backend
            )
        _, out, lse = run_batch(lb, md, backend, causal=False)
        assert not out.isnan().any() and not lse.isnan().any()
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=0)
        return
    _, out, lse = run_batch(lb, md, backend, causal=True)
    assert not out.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(
        out[:num_masked], torch.zeros_like(out[:num_masked]), rtol=0, atol=0
    )
    assert torch.isneginf(lse[:num_masked]).all()
    torch.testing.assert_close(
        out[num_masked:], torch.ones_like(out[num_masked:]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        lse[num_masked:], torch.zeros_like(lse[num_masked:]), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# test_paged_prefill_split_kv_empty_chunk
# ---------------------------------------------------------------------------

SPLIT_KV_AXES = dict(dtype=[torch.float16, torch.bfloat16], form=["dense", "csr"])


@pytest.mark.parametrize(
    argnames(SPLIT_KV_AXES, "backend"), param_rows(SPLIT_KV_AXES, lambda p: True)
)
def test_paged_prefill_split_kv_empty_chunk(dtype, form, backend):
    """q2 / kv129 / page16 with the legacy /10 inputs and combined NHD pool:
    finite, and within 1e-2 of the fp64 ``ref_single_prefill`` for out AND
    LSE (legacy), plus oracle; ``form`` (added) runs both paging forms."""
    lb = legacy_uniform_batch(
        batch_size=1,
        kv_len=129,
        qo_len=2,
        page_size=16,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        dtype=dtype,
        fp32_source=False,
        scale=10.0,
        seed=seed_of("split", str(dtype), form),
    )
    md = lb.metadata(form)
    _, out, lse = run_batch(lb, md, backend, causal=True)
    ki, vi = lb.request_kv(0)
    o_ref, lse_ref = ref_single_prefill(lb.q, ki, vi, causal=True)
    assert not out.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(out, o_ref, rtol=1e-2, atol=1e-2)  # legacy
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)  # legacy
    assert_oracle(lb, out, lse, causal=True)
