"""Legacy -> unified parity, group B: trtllm-gen, cake and FMHA v2 paged entries.

The trtllm-gen fixtures are the legacy ones (the helpers of
``tests/attention/test_trtllm_gen_attention_decode.py`` under the legacy
seed), run through ``PagedAttention(backend="trtllm-gen")`` (or ``"cake"``)
and asserted with the legacy reference (fa2 wrapper / sink varlen reference)
at the legacy budget plus the fp32 oracle.  The legacy axes the unified API
cannot express -- fp8 Q, nvfp4 KV, an independent output dtype, skip-softmax,
independent K/V page tables, head dims 256/512 and pages 128-1024 on
trtllm-gen -- assert the clear rejection and flip with the ``EXPECT_*``
flags.  FMHA v2 is not a unified candidate; its chunked-attention semantics
are exact bool masks and run on fa2 as ``partial`` rows.

Default subset: the batch-4 shapes.  ``FI_PARITY_SLOW=1`` adds the legacy
batch-128/256 shapes and the q = kv = 8192 single request.
"""

import inspect

import pytest
import torch

import flashinfer
from flashinfer.experimental.paged_attention import CAPABILITIES
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_CHUNKED_ATTENTION_KNOB,
    EXPECT_DEVICE_SCALES,
    EXPECT_FMHA_V2_CANDIDATE,
    EXPECT_FP8_Q,
    EXPECT_INDEPENDENT_KV_TABLES,
    EXPECT_NVFP4_KV,
    EXPECT_OUTPUT_DTYPE,
    EXPECT_SKIP_SOFTMAX,
    EXPECT_TRTLLM_HEAD_DIM_256,
    EXPECT_TRTLLM_HEAD_DIM_512,
    EXPECT_TRTLLM_LARGE_PAGES,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    dense_metadata,
    gated,
    oracle,
    reference_long,
    resolve_or_skip,
    slow_case,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_MAP = [
    # (legacy nodeid or function, unified test function(s) in this file, status, note)
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill",
        [
            "test_trtllm_batch_prefill",
            "test_trtllm_batch_prefill_head_dim_256",
            "test_trtllm_quantized_dtype_triples_unsupported",
            "test_trtllm_skip_softmax_unsupported",
            "test_trtllm_independent_kv_tables_unsupported",
        ],
        "partial",
        "in-domain axes run with the legacy fixture (seed 0, generate_seq_lens_prefill, "
        "create_kv_cache/page_table, the stacked pool as K=kv[:,0]/V=kv[:,1] views): "
        "HND/NHD x the 10 legacy (batch, page, Hkv, group) shapes x bf16/fp16 x sink x "
        "non-contiguous q x causal, q<=511 / kv<=2047, D128, dense form, pinned "
        "trtllm-gen, legacy fa2 (or sink varlen) reference at 1e-2 + LSE 1e-3 + oracle; "
        "batch 128/256 shapes need FI_PARITY_SLOW=1; D256, fp8/nvfp4 dtype triples, "
        "skip-softmax and uses_shared_paged_kv_idx=False assert the rejection until "
        "EXPECT_TRTLLM_HEAD_DIM_256 / EXPECT_FP8_Q+EXPECT_OUTPUT_DTYPE+EXPECT_NVFP4_KV / "
        "EXPECT_SKIP_SOFTMAX / EXPECT_INDEPENDENT_KV_TABLES",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_lse_contract",
        ["test_trtllm_batch_prefill_lse_contract"],
        "partial",
        "same fixture (HND, B2, page16, H4:2, fp16, q<=64, kv<=128, D128); three of the "
        "legacy (return_lse, provide_lse) cells map to lse_mode base2 / caller lse buffer "
        "(returned tensor IS the buffer, finite, fp32, vs the fa2 LSE at 1e-3 and the "
        "oracle); the (False, True) cell is a contract difference: unified rejects lse= "
        "with lse_mode='none' instead of filling it; the workspace guard-region check "
        "is native-only",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_bs1",
        ["test_trtllm_batch_prefill_bs1"],
        "partial",
        "q = kv = 8192, B1, page16, H64:8, bf16, causal, HND/NHD (FI_PARITY_SLOW=1): "
        "D128 on trtllm-gen vs the legacy fa2 reference (1e-2) and the chunked oracle; "
        "D256 asserts the trtllm-gen rejection (EXPECT_TRTLLM_HEAD_DIM_256) and runs the "
        "same fixture on fa2; skip-softmax / independent tables assert their rejections",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_cubin_variants",
        ["test_trtllm_batch_prefill_cubin_variants_unsupported"],
        "unsupported-by-design",
        "fp8 QKV with bf16/fp16/fp8 output and the sm_107-only spcompress cubins; the "
        "legacy function skips on B200 itself; unified has no fp8 Q / output dtype "
        "(EXPECT_FP8_Q, EXPECT_OUTPUT_DTYPE); cubin selection is a backend-suite contract",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_dynamic_page_size_gqa",
        ["test_trtllm_batch_prefill_dynamic_page_size_gqa"],
        "unsupported-by-design",
        "page 128/256/512/1024 are capability-excluded on trtllm-gen and cake "
        "(EXPECT_TRTLLM_LARGE_PAGES, WP-T); the legacy fixture (B4, H10:2, bf16, q<=257, "
        "kv<=1024, causal) runs on fa2 today vs the legacy fa2 reference and the oracle; "
        "uses_shared_paged_kv_idx=False asserts EXPECT_INDEPENDENT_KV_TABLES",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_head_dim_512",
        ["test_trtllm_batch_prefill_head_dim_512"],
        "unsupported-by-design",
        "D512 is capability-excluded on every unified backend (EXPECT_TRTLLM_HEAD_DIM_512 "
        "here; fa2 D512 is WP-T); the legacy grid (layouts x 3 shapes x 4 dtype triples x "
        "q 1/255/511 x kv 511/2047) asserts the resolve rejection; fp8 triples also need "
        "EXPECT_FP8_Q",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_bf16_separate_tables_matches_reference",
        ["test_cake_context_bf16_separate_tables"],
        "unsupported-by-design",
        "independent K/V page-id mappings (the legacy interleaved 2p / 2p+1 layout) have "
        "no unified spelling (EXPECT_INDEPENDENT_KV_TABLES); the same fixture (NHD, B2, "
        "page32, H4:2, bf16, non-causal, q<=7, kv<=31) runs on cake with the shared "
        "table vs the legacy fa2 reference and the oracle",
    ),
    (
        "tests/attention/test_cake_fmha.py::test_cake_context_fp8_nhd_device_scale_skip_matches_reference",
        ["test_cake_context_fp8_device_scale_skip_unsupported"],
        "unsupported-by-design",
        "fp8 QKV (EXPECT_FP8_Q), device-tensor scales (EXPECT_DEVICE_SCALES: run() takes "
        "host floats so the call stays sync-free) and skip-softmax (EXPECT_SKIP_SOFTMAX) "
        "each assert their rejection",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill",
        ["test_fmha_v2_paged_entry_points_are_native_only"],
        "native-only",
        "FMHA v2 (SM90 / SM12x) is not a unified candidate (EXPECT_FMHA_V2_CANDIDATE); "
        "its paged branches (Q_PAGED_KV_NHD/HND, page 32/128, fp8 + o_dtype, softmax "
        "stats in [max, sum_exp] form) stay in the native suite; an adapter would need a "
        "cc 9/12 capability row, the (q, paged 5-D pool) calling convention or a K/V "
        "pair via [B, 2, M] tables, and an LSE normalization from the stats pair",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_non_interleaved_kv",
        ["test_fmha_v2_paged_entry_points_are_native_only"],
        "native-only",
        "the pre-expanded [B, 2, M] K/V tables are independent page-id mappings "
        "(EXPECT_INDEPENDENT_KV_TABLES) on a backend that is not a unified candidate",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_chunked_attention",
        ["test_fmha_v2_chunked_attention_as_custom_mask"],
        "partial",
        "the paged branch of the legacy grid (B1/4, seq<=1024/4096, H4/32:4, D128, "
        "fp16/bf16, NHD page 32, chunk 64/256; seed 42) with the exact chunked visible "
        "set (col <= row and col >= floor(row / C) * C) as a fa2 custom mask, vs the "
        "legacy chunked_attention_ref_torch at 1e-2 and the masked oracle; the FMHA v2 "
        "entry point and a chunked_attention_size plan axis (EXPECT_CHUNKED_ATTENTION_KNOB) "
        "are not wired; the mask costs O(sum q_i * kv_i) bits",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_chunked_prefill_chunked_attention",
        ["test_fmha_v2_chunked_prefill_chunked_attention_as_custom_mask"],
        "partial",
        "the legacy grid (B1/4, kv<=1024/4096, new<=64/256, H4/32:4, D128, fp16/bf16, "
        "NHD page 32/128, chunk 64/256; seed 42) as a fa2 custom mask over the absolute "
        "positions kv_len - q_len + r, vs chunked_attention_ref_torch at 1e-2 and the "
        "masked oracle; entry point / knob as above",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())


# ---------------------------------------------------------------------------
# the legacy trtllm-gen fixture (tests/attention/test_trtllm_gen_attention_decode.py
# helpers, legacy call order under torch.manual_seed(0))
# ---------------------------------------------------------------------------

_legacy_ws = None


def _legacy_workspace():
    global _legacy_ws
    if _legacy_ws is None:
        _legacy_ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    return _legacy_ws


def _legacy_trtllm_problem(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    dtype_name,
    max_q_len,
    max_kv_len,
    head_dim,
):
    from tests.attention.test_trtllm_gen_attention_decode import (
        create_kv_cache,
        create_page_table,
        create_query_tensor,
        generate_cumsum_lens,
        generate_seq_lens_prefill,
        get_last_page_len,
    )

    torch.manual_seed(0)
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)
    q, _q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, dtype_name)
    q_indptr = generate_cumsum_lens(q_lens)
    kv_cache, _k_scale, _v_scale, ref_kv_cache, _ = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        dtype_name,
        dtype_name,
        kv_layout,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    return dict(
        kv_layout=kv_layout,
        batch_size=batch_size,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16 if dtype_name == "bf16" else torch.float16,
        q=q,
        ref_q=ref_q,
        q_lens=q_lens,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        kv_cache=kv_cache,  # (pages, 2, ...) stacked pool
        ref_kv_cache=ref_kv_cache,
        page_table=page_table,
        all_page_ids=all_page_ids,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sink=sink,
        sm_scale=float(1.0 / (head_dim**0.5)),
    )


def _legacy_fa2_reference(p, *, causal, window_left=-1):
    """The legacy reference for the no-sink rows: the fa2 wrapper on the
    reference pool (the legacy left backend='auto', which is fa2 on B200)."""
    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        _legacy_workspace(), p["kv_layout"], backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr=p["q_indptr"],
        paged_kv_indptr=p["kv_indptr"],
        paged_kv_indices=p["all_page_ids"],
        paged_kv_last_page_len=p["kv_last_page_len"].to(DEVICE),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        page_size=p["page_size"],
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        q_data_type=p["ref_q"].dtype,
        kv_data_type=p["ref_kv_cache"].dtype,
        window_left=window_left,
    )
    return wrapper_ref.run(p["ref_q"], p["ref_kv_cache"], return_lse=True)


def _legacy_sink_reference(p, *, causal, window_left=-1):
    from tests.attention.test_trtllm_gen_attention_decode import flatten_paged_kv
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
        p["ref_kv_cache"],
        p["page_table"],
        p["seq_lens"].to(DEVICE),
        p["page_size"],
        p["kv_last_page_len"],
        p["kv_layout"],
    )
    return sink_attention_unified(
        p["ref_q"],
        k_flat,
        v_flat,
        p["sink"],
        window_left,
        causal,
        p["sm_scale"],
        mode="varlen",
        batch_size=p["batch_size"],
        qo_indptr=p["q_indptr"],
        kv_indptr=kv_indptr_tokens,
    )


def _assert_legacy_close(out, ref, *, rtol=1e-2, atol=1e-2):
    """The legacy assertion: assert_close with the 1e-7 mismatch allowance."""
    from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

    assert_close_with_mismatch_tolerance(
        out.float(),
        ref.float(),
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=int(1e-7 * out.numel()),
    )


def _plan(p, backend, *, causal, lse_mode="base2", use_sinks=False, window_left=-1):
    res = resolve_or_skip(
        backend,
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["dtype"],
        page_size=p["page_size"],
        kv_layout=p["kv_layout"],
        causal=causal,
        need_lse=lse_mode != "none",
        window_left=window_left,
        kv_input_form="block_tables",
        sinks=use_sinks,
    )
    md = dense_metadata(p["q_indptr"], p["seq_lens"], p["page_table"], p["page_size"])
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["dtype"],
        kv_layout=p["kv_layout"],
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        use_sinks=use_sinks,
        backend=res,
    )
    assert attn.backend == backend
    return attn, md


def _trtllm_resolve_kwargs(p, *, causal, backend):
    return dict(
        device=torch.device(DEVICE),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["dtype"],
        page_size=p["page_size"],
        kv_layout=p["kv_layout"],
        causal=causal,
        need_lse=True,
        backend=backend,
    )


TRTLLM_BATCH_PREFILL_SHAPES = [
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    pytest.param(128, 16, 2, 5, marks=pytest.mark.slow),
    pytest.param(128, 32, 4, 1, marks=pytest.mark.slow),
    pytest.param(128, 64, 2, 8, marks=pytest.mark.slow),
    pytest.param(256, 16, 4, 8, marks=pytest.mark.slow),
    pytest.param(256, 32, 2, 8, marks=pytest.mark.slow),
    pytest.param(256, 64, 4, 1, marks=pytest.mark.slow),
    pytest.param(256, 64, 4, 5, marks=pytest.mark.slow),
]


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill: the in-domain axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size", TRTLLM_BATCH_PREFILL_SHAPES
)
@pytest.mark.parametrize("dtype_name", ["bf16", "fp16"])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_batch_prefill(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    dtype_name,
    enable_sink,
    non_contiguous_query,
    causal,
):
    from tests.attention.test_trtllm_gen_attention_decode import (
        flip_coin,
        make_query_non_contiguous,
    )

    if batch_size >= 128:
        slow_case(f"batch {batch_size}, q<=511, kv<=2047")
    p = _legacy_trtllm_problem(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        dtype_name,
        511,
        2047,
        128,
    )
    attn, md = _plan(p, "trtllm-gen", causal=causal, use_sinks=enable_sink)
    q_input = (
        make_query_non_contiguous(p["q"], p["num_qo_heads"], p["head_dim"])
        if non_contiguous_query
        else p["q"].contiguous()
    )
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]  # views of the stacked pool
    assert k.data_ptr() == p["kv_cache"].data_ptr()
    # the legacy coin decides whether the caller supplies the output buffer
    out_buf = None
    if flip_coin(batch_size, page_size, num_kv_heads, head_grp_size, dtype_name):
        out_buf = torch.empty(
            p["q"].shape[0],
            p["num_qo_heads"],
            p["head_dim"],
            dtype=p["dtype"],
            device=DEVICE,
        )
    out, lse = attn.run(
        q_input,
        (k, v),
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
        out=out_buf,
    )
    if out_buf is not None:
        assert out.data_ptr() == out_buf.data_ptr()
    assert lse.dtype == torch.float32 and lse.shape == (
        p["q"].shape[0],
        p["num_qo_heads"],
    )
    assert torch.isfinite(lse).all()

    if enable_sink:
        ref = _legacy_sink_reference(p, causal=causal)
        _assert_legacy_close(out, ref)
    else:
        ref, lse_ref = _legacy_fa2_reference(p, causal=causal)
        _assert_legacy_close(out, ref)
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=causal,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
        sinks=p["sink"] if enable_sink else None,
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_batch_prefill_head_dim_256(kv_layout, causal):
    """The legacy head_dim 256 axis: capability-excluded on trtllm-gen
    (EXPECT_TRTLLM_HEAD_DIM_256); the same legacy fixture runs on fa2, which
    declares (256, 256), so the math is covered today."""
    p = _legacy_trtllm_problem(kv_layout, 4, 16, 2, 1, "bf16", 511, 2047, 256)
    res = gated(
        EXPECT_TRTLLM_HEAD_DIM_256,
        lambda: resolve_paged_attention(
            **_trtllm_resolve_kwargs(p, causal=causal, backend="trtllm-gen")
        ),
        match="unsupported head dims \\(256, 256\\)",
    )
    backend = "trtllm-gen" if res is not None else "fa2"
    attn, md = _plan(p, backend, causal=causal)
    out, lse = attn.run(
        p["q"], (p["kv_cache"][:, 0], p["kv_cache"][:, 1]), sm_scale=p["sm_scale"]
    )
    ref, lse_ref = _legacy_fa2_reference(p, causal=causal)
    _assert_legacy_close(out, ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=causal,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "nvfp4"),
        ("fp8", "nvfp4", "fp8"),
    ],
)
def test_trtllm_quantized_dtype_triples_unsupported(q_dtype, kv_dtype, o_dtype):
    """The legacy fp8 / nvfp4 dtype triples: fp8 Q is rejected at resolve
    (EXPECT_FP8_Q); an output dtype independent of q and packed nvfp4 KV
    with block scale factors have no spelling at all (EXPECT_OUTPUT_DTYPE,
    EXPECT_NVFP4_KV)."""
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=2,
            num_kv_heads=2,
            head_dim_qk=128,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=16,
            kv_layout="HND",
            causal=True,
            need_lse=False,
            backend="trtllm-gen",
        ),
        match="unsupported q dtype",
    )
    run_params = inspect.signature(PagedAttention.run).parameters
    plan_params = inspect.signature(PagedAttention.plan).parameters
    assert (
        "o_dtype" in plan_params or "out_dtype" in run_params
    ) == EXPECT_OUTPUT_DTYPE
    assert ("kv_cache_sf" in run_params) == EXPECT_NVFP4_KV
    if res is not None:
        pytest.fail(
            f"EXPECT_FP8_Q flipped: port the legacy ({q_dtype}, {kv_dtype}, {o_dtype}) "
            "fixture with q_scale / o_scale here"
        )


def test_trtllm_skip_softmax_unsupported():
    """skip_softmax_threshold_scale_factor (approximate softmax) is a
    trtllm-gen launch knob with no unified spelling (EXPECT_SKIP_SOFTMAX)."""
    run_params = inspect.signature(PagedAttention.run).parameters
    plan_params = inspect.signature(PagedAttention.plan).parameters
    present = (
        "skip_softmax_threshold_scale_factor" in run_params
        or "skip_softmax_threshold_scale_factor" in plan_params
    )
    assert present == EXPECT_SKIP_SOFTMAX


def _independent_tables_rejected(p):
    """The legacy interleaved layout (K at page 2p, V at 2p+1, a [B, 2, M]
    table) is two page-id mappings; the metadata takes exactly one.  Returns
    None while EXPECT_INDEPENDENT_KV_TABLES is False."""
    from tests.attention.test_trtllm_gen_attention_decode import (
        prepare_paged_kv_for_kernel,
    )

    (k_i, v_i), table_2, _ = prepare_paged_kv_for_kernel(
        p["kv_cache"], p["page_table"], False
    )
    assert table_2.shape == (p["batch_size"], 2, p["page_table"].shape[1])
    assert not torch.equal(table_2[:, 0], table_2[:, 1])  # K ids != V ids
    md = dense_metadata(p["q_indptr"], p["seq_lens"], p["page_table"], p["page_size"])
    return gated(
        EXPECT_INDEPENDENT_KV_TABLES,
        lambda: PagedAttentionMetadata.dense(
            md.qo_indptr,
            md.kv_seq_lens,
            table_2[:, 0].contiguous(),
            v_block_tables=table_2[:, 1].contiguous(),
            page_size=p["page_size"],
            max_q_len=md.max_q_len,
            max_kv_len=md.max_kv_len,
        ),
        match="v_block_tables",
        exc=TypeError,
    )


def test_trtllm_independent_kv_tables_unsupported():
    p = _legacy_trtllm_problem("HND", 4, 16, 2, 1, "bf16", 511, 2047, 128)
    if _independent_tables_rejected(p) is not None:
        pytest.fail(
            "EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here"
        )


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_lse_contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("provide_lse", [False, True])
def test_trtllm_batch_prefill_lse_contract(return_lse, provide_lse):
    p = _legacy_trtllm_problem("HND", 2, 16, 2, 2, "fp16", 64, 128, 128)
    ref, lse_ref = _legacy_fa2_reference(p, causal=True)
    k, v = p["kv_cache"][:, 0], p["kv_cache"][:, 1]
    provided = (
        torch.full(
            (p["q"].shape[0], p["num_qo_heads"]),
            float("nan"),
            device=DEVICE,
            dtype=torch.float32,
        )
        if provide_lse
        else None
    )
    if not return_lse and not provide_lse:
        attn, md = _plan(p, "trtllm-gen", causal=True, lse_mode="none")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"])
        assert lse is None
    elif not return_lse and provide_lse:
        # contract difference: legacy fills the provided buffer and returns
        # only the output; unified rejects lse= under lse_mode='none' before
        # any launch, and the same buffer is the output of a base2 plan
        attn, md = _plan(p, "trtllm-gen", causal=True, lse_mode="none")
        with pytest.raises(ValueError, match="lse_mode='none'"):
            attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        assert torch.isnan(provided).all()  # untouched
        attn, md = _plan(p, "trtllm-gen", causal=True, lse_mode="base2")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        assert lse.data_ptr() == provided.data_ptr()
    else:
        attn, md = _plan(p, "trtllm-gen", causal=True, lse_mode="base2")
        out, lse = attn.run(p["q"], (k, v), sm_scale=p["sm_scale"], lse=provided)
        if provide_lse:
            assert lse.data_ptr() == provided.data_ptr()
    if lse is not None:
        assert lse.dtype == torch.float32
        assert lse.shape == (p["q"].shape[0], p["num_qo_heads"])
        assert torch.isfinite(lse).all(), (
            "trtllm-gen context kernel produced non-finite LSE"
        )
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    _assert_legacy_close(out, ref)
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=True,
        kv_layout="HND",
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    if lse is not None:
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_bs1: q = kv = 8192
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_bs1(
    kv_layout, head_dim, skips_softmax, uses_shared_paged_kv_idx
):
    if skips_softmax:
        test_trtllm_skip_softmax_unsupported()
        if not EXPECT_SKIP_SOFTMAX:
            return
    slow_case("B1, q = kv = 8192, 64 query heads")
    p = _legacy_trtllm_problem(kv_layout, 1, 16, 8, 8, "bf16", 8192, 8192, head_dim)
    if not uses_shared_paged_kv_idx:
        if _independent_tables_rejected(p) is None:
            return
        pytest.fail(
            "EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here"
        )
    backend = "trtllm-gen"
    if head_dim == 256:
        res = gated(
            EXPECT_TRTLLM_HEAD_DIM_256,
            lambda: resolve_paged_attention(
                **_trtllm_resolve_kwargs(p, causal=True, backend="trtllm-gen")
            ),
            match="unsupported head dims \\(256, 256\\)",
        )
        backend = "trtllm-gen" if res is not None else "fa2"
    attn, md = _plan(p, backend, causal=True)
    out, lse = attn.run(
        p["q"], (p["kv_cache"][:, 0], p["kv_cache"][:, 1]), sm_scale=p["sm_scale"]
    )
    ref, lse_ref = _legacy_fa2_reference(p, causal=True)
    _assert_legacy_close(out, ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    o_out, o_lse = reference_long(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=True,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_cubin_variants (fp8 QKV; spcompress is sm_107-only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [("fp8", "fp8", "bf16"), ("fp8", "fp8", "fp16"), ("fp8", "fp8", "fp8")],
)
@pytest.mark.parametrize("head_dim,window_left", [(128, -1), (256, -1), (128, 127)])
def test_trtllm_batch_prefill_cubin_variants_unsupported(
    q_dtype, kv_dtype, o_dtype, head_dim, window_left
):
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=2,
            num_kv_heads=2,
            head_dim_qk=head_dim,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=16,
            kv_layout="HND",
            causal=True,
            window_left=window_left,
            need_lse=False,
            backend="trtllm-gen",
        ),
        match="unsupported q dtype",
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: port the legacy fp8 QKV fixture here")


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_dynamic_page_size_gqa: pages 128..1024
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_dynamic_page_size_gqa(
    page_size, uses_shared_paged_kv_idx
):
    p = _legacy_trtllm_problem("HND", 4, page_size, 2, 5, "bf16", 257, 1024, 128)
    if not uses_shared_paged_kv_idx:
        if _independent_tables_rejected(p) is None:
            return
        pytest.fail(
            "EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here"
        )
    backends = []
    for backend in ("trtllm-gen", "cake"):
        res = gated(
            EXPECT_TRTLLM_LARGE_PAGES,
            lambda backend=backend: resolve_paged_attention(
                **_trtllm_resolve_kwargs(p, causal=True, backend=backend)
            ),
            match=f"unsupported page_size {page_size}",
        )
        if res is not None:
            backends.append(backend)
    backends.append("fa2")  # runs any page size today
    ref, lse_ref = _legacy_fa2_reference(p, causal=True)
    for backend in backends:
        attn, md = _plan(p, backend, causal=True)
        out, lse = attn.run(
            p["q"], (p["kv_cache"][:, 0], p["kv_cache"][:, 1]), sm_scale=p["sm_scale"]
        )
        _assert_legacy_close(out, ref)
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
        o_out, o_lse = oracle(
            md,
            p["q"],
            p["ref_kv_cache"][:, 0],
            p["ref_kv_cache"][:, 1],
            causal=True,
            kv_layout="HND",
            sm_scale=p["sm_scale"],
        )
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_trtllm_batch_prefill_head_dim_512
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [(4, 16, 2, 1), (4, 32, 4, 5), (128, 16, 2, 8)],
)
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("max_q_len", [1, 255, 511])
@pytest.mark.parametrize("max_kv_len", [511, 2047])
def test_trtllm_batch_prefill_head_dim_512(
    kv_layout,
    batch_size,
    page_size,
    num_kv_heads,
    head_grp_size,
    q_dtype,
    kv_dtype,
    o_dtype,
    max_q_len,
    max_kv_len,
):
    """D512 (Gemma-style full attention) on trtllm-gen: the legacy grid's
    axes assert the resolve rejection (fp8 triples: the q dtype is rejected
    first).  With EXPECT_TRTLLM_HEAD_DIM_512 the bf16/fp16 rows run the
    legacy fixture vs the oracle (the legacy reference for D512 was torch
    SDPA on the dequantized pool: the oracle here)."""
    fp8 = q_dtype == "fp8"
    if fp8 and not EXPECT_FP8_Q:
        flag, match = False, "unsupported q dtype"
    else:
        flag, match = EXPECT_TRTLLM_HEAD_DIM_512, "unsupported head dims \\(512, 512\\)"
    res = gated(
        flag,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=num_kv_heads * head_grp_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=512,
            q_dtype=torch.float8_e4m3fn
            if fp8
            else (torch.bfloat16 if q_dtype == "bf16" else torch.float16),
            kv_dtype=torch.float8_e4m3fn if fp8 else None,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=True,
            need_lse=True,
            backend="trtllm-gen",
        ),
        match=match,
    )
    if res is None:
        return
    if fp8:
        pytest.fail("EXPECT_FP8_Q flipped: port the fp8 D512 fixture with q_scale here")
    if batch_size >= 128:
        slow_case(f"batch {batch_size}, D512")
    p = _legacy_trtllm_problem(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        512,
    )
    attn, md = _plan(p, "trtllm-gen", causal=True)
    out, lse = attn.run(
        p["q"], (p["kv_cache"][:, 0], p["kv_cache"][:, 1]), sm_scale=p["sm_scale"]
    )
    o_out, o_lse = oracle(
        md,
        p["q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=True,
        kv_layout=kv_layout,
        sm_scale=p["sm_scale"],
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_cake_fmha.py paged context entries
# ---------------------------------------------------------------------------


def test_cake_context_bf16_separate_tables():
    p = _legacy_trtllm_problem("NHD", 2, 32, 2, 2, "bf16", 7, 31, 128)
    if _independent_tables_rejected(p) is not None:
        pytest.fail(
            "EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] fixture here"
        )
    # the same fixture with the shared table on cake (the legacy backend)
    attn, md = _plan(p, "cake", causal=False)
    out, lse = attn.run(
        p["q"], (p["kv_cache"][:, 0], p["kv_cache"][:, 1]), sm_scale=p["sm_scale"]
    )
    ref, lse_ref = _legacy_fa2_reference(p, causal=False)
    _assert_legacy_close(out, ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
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


def test_cake_context_fp8_device_scale_skip_unsupported():
    """fp8 QKV on cake (EXPECT_FP8_Q), device-tensor scales
    (EXPECT_DEVICE_SCALES: the unified run() takes host floats so the call
    stays sync-free; the rejection is shown on the one fp8-KV path that
    exists, fa2) and skip-softmax (EXPECT_SKIP_SOFTMAX)."""
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=128,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=64,
            kv_layout="NHD",
            causal=True,
            need_lse=False,
            backend="cake",
        ),
        match="unsupported q dtype",
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: port the legacy cake fp8 fixture here")
    test_trtllm_skip_softmax_unsupported()
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


# ---------------------------------------------------------------------------
# test_fmha_v2_prefill.py paged entries
# ---------------------------------------------------------------------------


def test_fmha_v2_paged_entry_points_are_native_only():
    """FMHA v2 (trtllm_fmha_v2_prefill, SM90 / SM12x) is not a unified
    candidate.  What an adapter would need: a capability row for cc 9 / 12
    (head dims 128/256, pages 32/128, NHD/HND), the (q, 5-D paged pool) or
    (q, (k, v)) + [B, 2, M] calling convention mapped from one page table,
    the softmax-stats pair [max, sum_exp] normalized to the LSE contract
    (arch-dependent scaling of the stored max), and the chunked mask as a
    plan axis.  The independent-table branch also needs
    EXPECT_INDEPENDENT_KV_TABLES."""
    cap = gated(
        EXPECT_FMHA_V2_CANDIDATE,
        lambda: CAPABILITIES["fmha_v2"],
        match="fmha_v2",
        exc=KeyError,
    )
    if cap is not None:
        pytest.fail("EXPECT_FMHA_V2_CANDIDATE flipped: port the legacy paged grid here")


def _chunked_mask(q_lens, kv_lens, chunk):
    """The exact FMHA v2 chunked visible set, flattened per request in
    request order: for query r of a request (absolute row = kv_len - q_len +
    r), allowed iff col <= row and col >= floor(row / chunk) * chunk."""
    parts = []
    for lq, lkv in zip(q_lens, kv_lens, strict=True):
        lq, lkv = int(lq), int(lkv)
        rows = torch.arange(lq, device=DEVICE).unsqueeze(1) + (lkv - lq)
        cols = torch.arange(lkv, device=DEVICE).unsqueeze(0)
        allowed = (cols <= rows) & (cols >= (rows // chunk) * chunk)
        parts.append(allowed.reshape(-1))
    return torch.cat(parts)


def _fa2_masked(md, q, k, v, *, num_qo_heads, num_kv_heads, head_dim, dtype, mask):
    res = resolve_or_skip(
        "fa2",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        page_size=md.page_size,
        kv_layout="NHD",
        causal=True,
        need_lse=True,
        custom_mask=True,
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        custom_mask=mask,
        backend=res,
    )
    assert attn.backend == "fa2"
    return attn.run(q, (k, v))


def _legacy_fmha_v2_paged_pool(num_pages, page_size, num_kv_heads, head_dim, dtype):
    return torch.randn(
        num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [1024, 4096])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_fmha_v2_chunked_attention_as_custom_mask(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_size,
    chunked_attention_size,
):
    from tests.attention.test_fmha_v2_prefill import chunked_attention_ref_torch

    plan_params = inspect.signature(PagedAttention.plan).parameters
    assert ("chunked_attention_size" in plan_params) == EXPECT_CHUNKED_ATTENTION_KNOB

    # ---- the legacy Q_PAGED_KV_NHD fixture ----
    torch.manual_seed(42)
    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=DEVICE,
    )
    max_kv_len = seq_lens.max().item()
    cum_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    total_tokens = cum_seq_lens[-1].item()
    sm_scale = 1.0 / (head_dim**0.5)
    max_num_blocks = (max_kv_len + page_size - 1) // page_size
    num_pages = batch_size * max_num_blocks
    paged_kv_cache = _legacy_fmha_v2_paged_pool(
        num_pages, page_size, num_kv_heads, head_dim, dtype
    )
    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=DEVICE
    )
    for i in range(batch_size):
        num_blocks_needed = (seq_lens[i].item() + page_size - 1) // page_size
        block_tables[i, :num_blocks_needed] = torch.arange(
            i * max_num_blocks, i * max_num_blocks + num_blocks_needed, device=DEVICE
        )

    seq_cpu = seq_lens.cpu()
    mask = _chunked_mask(seq_cpu, seq_cpu, chunked_attention_size)
    md = dense_metadata(cum_seq_lens, seq_cpu, block_tables, page_size)
    k, v = paged_kv_cache[:, 0], paged_kv_cache[:, 1]
    out, lse = _fa2_masked(
        md,
        q,
        k,
        v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        mask=mask,
    )
    output_ref = chunked_attention_ref_torch(
        (q, paged_kv_cache),
        seq_lens=seq_lens,
        cum_seq_lens_q=cum_seq_lens,
        sm_scale=sm_scale,
        chunked_attention_size=chunked_attention_size,
        block_tables=block_tables,
    )
    torch.testing.assert_close(
        out.float(), output_ref.float(), rtol=1e-2, atol=1e-2
    )  # legacy
    o_out, o_lse = oracle(md, q, k, v, causal=True, kv_layout="NHD", custom_mask=mask)
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    # the chunk boundary really excluded keys: differs from plain causal
    plain_out, _ = oracle(md, q, k, v, causal=True, kv_layout="NHD")
    assert not torch.allclose(o_out, plain_out, **OUT_TOL)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_kv_len", [1024, 4096])
@pytest.mark.parametrize("max_new_tokens", [64, 256])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_fmha_v2_chunked_prefill_chunked_attention_as_custom_mask(
    batch_size,
    max_kv_len,
    max_new_tokens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_size,
    chunked_attention_size,
):
    from tests.attention.test_fmha_v2_prefill import chunked_attention_ref_torch

    # ---- the legacy fixture (Q < KV, Q_PAGED_KV_NHD) ----
    torch.manual_seed(42)
    kv_seq_lens = torch.randint(
        max_kv_len // 2, max_kv_len + 1, (batch_size,), dtype=torch.int32, device=DEVICE
    )
    q_seq_lens = torch.randint(
        max(1, max_new_tokens // 2),
        max_new_tokens + 1,
        (batch_size,),
        dtype=torch.int32,
        device=DEVICE,
    )
    q_seq_lens = torch.minimum(q_seq_lens, kv_seq_lens)
    actual_max_kv_len = kv_seq_lens.max().item()
    cum_seq_lens_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens_kv[1:] = torch.cumsum(kv_seq_lens, dim=0)
    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens_q[1:] = torch.cumsum(q_seq_lens, dim=0)
    total_q_tokens = cum_seq_lens_q[-1].item()
    sm_scale = 1.0 / (head_dim**0.5)
    max_num_blocks = (actual_max_kv_len + page_size - 1) // page_size
    num_pages = batch_size * max_num_blocks
    paged_kv_cache = _legacy_fmha_v2_paged_pool(
        num_pages, page_size, num_kv_heads, head_dim, dtype
    )
    q = torch.randn(total_q_tokens, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=DEVICE
    )
    for i in range(batch_size):
        num_blocks_needed = (kv_seq_lens[i].item() + page_size - 1) // page_size
        block_tables[i, :num_blocks_needed] = torch.arange(
            i * max_num_blocks, i * max_num_blocks + num_blocks_needed, device=DEVICE
        )

    mask = _chunked_mask(q_seq_lens.cpu(), kv_seq_lens.cpu(), chunked_attention_size)
    md = dense_metadata(cum_seq_lens_q, kv_seq_lens.cpu(), block_tables, page_size)
    k, v = paged_kv_cache[:, 0], paged_kv_cache[:, 1]
    out, lse = _fa2_masked(
        md,
        q,
        k,
        v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        mask=mask,
    )
    output_ref = chunked_attention_ref_torch(
        (q, paged_kv_cache),
        seq_lens=kv_seq_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        sm_scale=sm_scale,
        chunked_attention_size=chunked_attention_size,
        cum_seq_lens_kv=cum_seq_lens_kv,
        block_tables=block_tables,
    )
    torch.testing.assert_close(
        out.float(), output_ref.float(), rtol=1e-2, atol=1e-2
    )  # legacy
    o_out, o_lse = oracle(md, q, k, v, causal=True, kv_layout="NHD", custom_mask=mask)
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
