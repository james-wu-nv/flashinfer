"""Legacy -> unified: tests/attention/test_trtllm_gen_attention_prefill.py

The legacy paged-context tests call ``trtllm_batch_context_with_kv_cache``
(the trtllm-gen context kernel) through one helper,
``_test_trtllm_batch_prefill``.  The helper here keeps the legacy one's
signature, skips, fixture (the ``test_trtllm_gen_attention_decode`` helpers
under seed 0, in the legacy RNG order), reference and assertions, and runs
the kernel through ``PagedAttention`` instead.  The tests and grids are the
legacy ones, so the node ids equal the legacy ids.

The legacy file tests one kernel against the fa2 wrapper, so only trtllm-gen
runs (an fa2 row would compare fa2 with itself).  Like the legacy call, the
plan requests the LSE only where the legacy checks it (16-bit q, no sinks, no
skip-softmax), and the run uses the legacy 256 MiB workspace buffer, so the
legacy check of the guard region after the softmax-stats slab applies
unchanged.  The legacy then runs ``BatchPrefillWithPagedKVCacheWrapper``
(``backend="trtllm-gen"``) and compares it with the function: a check of two
legacy APIs against each other, dropped here.

Configurations ``PagedAttention`` cannot express are expected failures that
name the missing API: skip-softmax, separate K / V page tables, NVFP4 KV,
NVFP4 output, an output dtype other than q's (fp8 q with a 16-bit output) and
the SM107 spcompress cubins.  fp8 q and head dims 256 / 512 are not declared
for trtllm-gen, so the resolver skips those rows.

The five ``test_trtllm_gen_prefill*`` tests call
``trtllm_ragged_attention_deepseek`` (ragged MLA prefill) and are out of scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from flashinfer.utils import get_compute_capability
from tests.attention.test_trtllm_gen_attention_decode import (
    DTYPE_MAP,
    GPU_DEVICE,
    TRTLLM_GEN_WORKSPACE_CHECK_BYTES,
    _skip_if_not_blackwell,
    create_kv_cache,
    create_output,
    create_page_table,
    create_query_tensor,
    create_workspace_buffers,
    flatten_paged_kv,
    flip_coin,
    generate_cumsum_lens,
    generate_seq_lens_prefill,
    get_last_page_len,
    make_query_non_contiguous,
    sdpa_paged_reference,
    trtllm_gen_workspace_softmax_end_bytes_context,
    workspace_size,
)
from tests.attention.test_trtllm_gen_attention_prefill import (
    _SPCOMPRESS_TILE_GEOMETRY,
)
from tests.test_helpers.sink_attention_reference import sink_attention_unified
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_trtllm_gen_attention_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill",
        ["test_trtllm_batch_prefill"],
        "partial",
        "same grid, seed and tensors on trtllm-gen, legacy 256 MiB workspace, LSE "
        "where the legacy checks it; bf16 / fp16 D128 rows check the legacy reference "
        "(1e-2, LSE 1e-3) and the workspace guard; xfail: skip-softmax, separate K/V "
        "tables, NVFP4 KV / output, fp8 q with a 16-bit output; resolver skip: fp8 q, "
        "D256; the wrapper-vs-function cross-check is dropped",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_lse_contract",
        ["test_trtllm_batch_prefill_lse_contract"],
        "equivalent",
        "same fixture; return_lse / provide_lse map to lse_mode base2 and a caller "
        "lse= buffer; the legacy LSE (1e-3), finiteness and workspace guard checks; "
        "the wrapper-vs-function cross-check is dropped",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_bs1",
        ["test_trtllm_batch_prefill_bs1"],
        "partial",
        "same grid (batch 1, q and kv up to 8192) on trtllm-gen; D128 rows check the "
        "legacy reference, LSE and workspace guard; xfail: skip-softmax, separate K/V "
        "tables; resolver skip: D256",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_cubin_variants",
        ["test_trtllm_batch_prefill_cubin_variants"],
        "unsupported-by-design",
        "the legacy skips outside SM107; on SM107 xfail: plan() has no uses_spcompress "
        "(and fp8 q is not declared for trtllm-gen)",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_dynamic_page_size_gqa",
        ["test_trtllm_batch_prefill_dynamic_page_size_gqa"],
        "partial",
        "pages 128-1024 on trtllm-gen vs the legacy reference, LSE and workspace "
        "guard; separate K/V tables xfail",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_head_dim_512",
        ["test_trtllm_batch_prefill_head_dim_512"],
        "partial",
        "(512, 512) and fp8 q are not declared for trtllm-gen (the legacy runs both "
        "on B200): resolver skip; xfail: skip-softmax, separate K/V tables, fp8 q with "
        "a bf16 output",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill",
        [],
        "out-of-scope",
        "trtllm_ragged_attention_deepseek (ragged MLA prefill), not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_use_fp16_softmax",
        [],
        "out-of-scope",
        "ragged MLA prefill (SM107-only fp16-softmax cubin variant), not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_fp8",
        [],
        "out-of-scope",
        "ragged MLA prefill with fp8 inputs on the cute-dsl backend, not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_bs1",
        [],
        "out-of-scope",
        "ragged MLA prefill, not paged",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_gen_prefill_glm5",
        [],
        "out-of-scope",
        "ragged MLA prefill with the GLM-5 MHA dimensions, not paged",
    ),
]

DEVICE = torch.device("cuda:0")

SPCOMPRESS_XFAIL = (
    "API limit: plan() has no uses_spcompress (the SM107 spcompress cubins); "
    "fp8 q is not declared for trtllm-gen either"
)
SKIP_SOFTMAX_XFAIL = (
    "API limit: PagedAttention has no skip-softmax threshold "
    "(skip_softmax_threshold_scale_factor)"
)
SPLIT_KV_TABLE_XFAIL = (
    "API limit: PagedAttentionMetadata takes one page table shared by K and V; "
    "no separate K / V page ids (uses_shared_paged_kv_idx=False)"
)
NVFP4_KV_XFAIL = (
    "API limit: PagedAttention has no NVFP4 KV cache (packed FP4 KV and its "
    "kv_cache_sf block scales)"
)
NVFP4_OUT_XFAIL = (
    "API limit: run() has no NVFP4 output (packed FP4 output and its scale factors)"
)
O_DTYPE_XFAIL = (
    "API limit: run() writes the output in q's dtype; no output dtype for fp8 q "
    "with a bf16 / fp16 output"
)


def _test_trtllm_batch_prefill(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    device_scale: float,
    head_dim: int,
    non_contiguous_query: bool = False,
    skips_softmax: bool = False,
    uses_shared_paged_kv_idx: bool = True,
    return_lse: bool | None = None,
    provide_lse: bool = False,
    uses_spcompress: bool = False,
):
    # enable_pdl (None in every legacy row) and device_scale (host scales for
    # the 16-bit rows that run) have no PagedAttention counterpart.

    # the legacy skips
    compute_capability = get_compute_capability(torch.device(device="cuda"))
    if compute_capability[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")
    if not causal and window_left >= 0:
        pytest.skip("Non-causal paged trtllm-gen tests only cover dense attention")

    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )

    # NVFP4 KV cache constraints
    if kv_dtype == "nvfp4":
        if compute_capability == (10, 7):
            pytest.skip("KV Cache NVFP4 is not supported on SM107")
        if q_dtype != "fp8":
            pytest.skip("NVFP4 KV cache requires FP8 query")
        if o_dtype != "fp8":
            pytest.skip("NVFP4 KV cache only supports FP8 output")

    if uses_spcompress and compute_capability != (10, 7):
        pytest.skip(
            "use_fp16_softmax / uses_spcompress are only supported on SM107 (Rubin)"
        )

    if uses_spcompress:
        if q_dtype != "fp8":
            pytest.skip("uses_spcompress requires FP8 Q")
        if kv_dtype == "nvfp4":
            pytest.skip("uses_spcompress is not shipped for NVFP4 KV")
        if (head_dim, q_dtype, kv_dtype) not in _SPCOMPRESS_TILE_GEOMETRY:
            pytest.skip(
                f"no spcompress tile geometry for "
                f"(head_dim={head_dim}, q_dtype={q_dtype}, kv_dtype={kv_dtype})"
            )

    # what PagedAttention cannot express
    if uses_spcompress:
        pytest.xfail(SPCOMPRESS_XFAIL)
    if skips_softmax:
        pytest.xfail(SKIP_SOFTMAX_XFAIL)
    if not uses_shared_paged_kv_idx:
        pytest.xfail(SPLIT_KV_TABLE_XFAIL)
    if kv_dtype == "nvfp4":
        pytest.xfail(NVFP4_KV_XFAIL)
    if o_dtype == "nvfp4":
        pytest.xfail(NVFP4_OUT_XFAIL)
    if o_dtype != q_dtype:
        pytest.xfail(O_DTYPE_XFAIL)

    # Validate LSE only for paths where we have a reliable reference and a supported
    # (non-FP8 / non-FP4) numerical regime. Sink attention doesn't populate lse_ref.
    check_lse = (
        not enable_sink
        and not skips_softmax
        and o_dtype != "nvfp4"
        and kv_dtype != "nvfp4"
        and q_dtype != "fp8"
        and not uses_spcompress
    )
    if (return_lse or provide_lse) and not check_lse:
        pytest.skip("LSE contract validation requires a reliable LSE reference")
    effective_return_lse = check_lse if return_lse is None else return_lse
    expects_lse = check_lse and (effective_return_lse or provide_lse)

    num_qo_heads = num_kv_heads * head_grp_size
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=DTYPE_MAP[q_dtype],
            kv_dtype=DTYPE_MAP[kv_dtype],
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=expects_lse,
            window_left=window_left,
            sinks=enable_sink,
            backend="trtllm-gen",
        )
    except ValueError as e:
        pytest.skip(f"trtllm-gen not runnable here: {e}")
    if q_dtype == "fp8":
        raise AssertionError(
            "trtllm-gen now declares fp8 q: port the legacy fp8 path (q_scale folded "
            "into sm_scale, 1 / o_scale into v_scale, the fp8 tolerances)"
        )

    # Set up test parameters
    torch.manual_seed(0)

    # Generate random sequence lengths
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)

    # Create query tensor and related data
    q, q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = generate_cumsum_lens(q_lens)

    # Create KV cache and related data
    kv_cache, k_scale, v_scale, ref_kv_cache, kv_cache_sf = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" or kv_dtype == "nvfp4" else q_dtype,
        kv_layout,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)

    workspace_buffer, workspace_buffer_ref = create_workspace_buffers(GPU_DEVICE)

    # Create output tensor and related data
    create_out_tensor = flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype
    )
    can_infer_type = q.dtype == DTYPE_MAP[o_dtype] or create_out_tensor
    create_out_dtype = not can_infer_type or flip_coin(
        batch_size, page_size, num_kv_heads, head_grp_size, o_dtype, q_dtype
    )
    out, out_dtype, o_scale, o_sf_scale, o_sf_vec_size = create_output(
        q, o_dtype, create_out_tensor, create_out_dtype
    )

    sm_scale = float(1.0 / (head_dim**0.5))

    # Build reference output
    plan_params = {
        "qo_indptr": q_indptr,
        "paged_kv_indptr": kv_indptr,
        "paged_kv_indices": all_page_ids,
        "paged_kv_last_page_len": kv_last_page_len.to(GPU_DEVICE),
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim_qk": head_dim,
        "page_size": page_size,
        "causal": causal,
        "pos_encoding_mode": "NONE",
        "logits_soft_cap": 0.0,
        "q_data_type": ref_q.dtype,
        "kv_data_type": ref_kv_cache.dtype,
        "window_left": window_left,
    }
    lse_ref = None
    sink = torch.rand(num_qo_heads, device=GPU_DEVICE, dtype=torch.float32) * 5
    if head_dim > 256:
        # FlashInfer's own FA2/FA3 kernels don't support head_dim > 256;
        # fall back to a PyTorch SDPA reference (causal/windowed only, no sink support).
        assert not enable_sink, (
            "SDPA fallback does not model attention sinks; skip sink+head_dim>256 cases"
        )
        output_ref = sdpa_paged_reference(
            ref_q,
            ref_kv_cache,
            q_lens,
            seq_lens,
            page_table,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            kv_layout,
            window_left,
        )
    elif not enable_sink:
        wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer_ref, kv_layout
        )
        wrapper_ref.plan(**plan_params)
        output_ref, lse_ref = wrapper_ref.run(ref_q, ref_kv_cache, return_lse=True)
    else:
        # Construct flat K/V via helper
        k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
            ref_kv_cache,
            page_table,
            seq_lens.to(GPU_DEVICE),
            page_size,
            kv_last_page_len,
            kv_layout,
        )
        output_ref = sink_attention_unified(
            ref_q,
            k_flat,
            v_flat,
            sink,
            window_left,
            causal,
            sm_scale,
            mode="varlen",
            batch_size=batch_size,
            qo_indptr=q_indptr,
            kv_indptr=kv_indptr_tokens,
        )

    # Run trtllm-gen through PagedAttention, on the legacy workspace buffer
    max_q_len_val = torch.max(q_lens).item()
    metadata = PagedAttentionMetadata.dense(
        q_indptr,
        seq_lens.to(GPU_DEVICE),
        page_table,
        page_size=page_size,
        max_q_len=max_q_len_val,
        max_kv_len=torch.max(seq_lens).item(),
    )
    attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=q.dtype,
        kv_dtype=kv_cache.dtype,
        kv_layout=kv_layout,
        causal=causal,
        window_left=window_left,
        lse_mode="base2" if expects_lse else "none",
        use_sinks=enable_sink,
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"

    # Optionally make query non-contiguous for testing stride support
    if non_contiguous_query:
        q_input = make_query_non_contiguous(q, num_qo_heads, head_dim)
    else:
        q_input = q.contiguous()

    provided_lse = None
    if expects_lse:
        # Allocate LSE on the caller side so we can pre-populate it with NaNs
        # and catch missed writes. Shape is [total_qo_tokens, num_qo_heads].
        provided_lse = (
            torch.full(
                (ref_q.shape[0], num_qo_heads),
                float("nan"),
                device=GPU_DEVICE,
                dtype=torch.float32,
            )
            if provide_lse
            else None
        )
        # Zero out the guard region that sits immediately after the softmax
        # slab. If the kernel writes out of bounds we'll notice it flip to
        # non-zero below.
        softmax_end = trtllm_gen_workspace_softmax_end_bytes_context(
            num_qo_heads, batch_size, max_q_len_val
        )
        guard_end = min(softmax_end + TRTLLM_GEN_WORKSPACE_CHECK_BYTES, workspace_size)
        workspace_buffer[softmax_end:guard_end].zero_()

    # the legacy scales are bmm1 = q_scale * k_scale * sm_scale and
    # bmm2 = v_scale / o_scale; q_scale = o_scale = 1 for the 16-bit rows that
    # get here
    output, lse_out = attn.run(
        q_input,
        (kv_cache[:, 0], kv_cache[:, 1]),
        out=out,
        lse=provided_lse,
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sinks=(sink if enable_sink else None),
    )
    if expects_lse:
        if provide_lse:
            assert lse_out.data_ptr() == provided_lse.data_ptr()
        assert lse_out.dtype == torch.float32
        assert lse_out.shape == (ref_q.shape[0], num_qo_heads)
        assert torch.isfinite(lse_out).all(), (
            "trtllm-gen context kernel produced non-finite LSE"
        )
        if lse_ref is not None:
            torch.testing.assert_close(lse_out, lse_ref.float(), rtol=1e-3, atol=1e-3)
        # Softmax slab and its guard region remain zero-initialized outside writes.
        assert (workspace_buffer[softmax_end:guard_end].cpu().numpy() == 0).all(), (
            "trtllm-gen context kernel wrote past the softmax slab"
        )
    else:
        assert lse_out is None

    # the legacy tolerance for 16-bit q / KV / output (the fp8 and nvfp4 rows
    # do not get here)
    rtol, atol = 1e-2, 1e-2
    allowed_mismatch_rate = 1e-7
    # Calculate max allowed mismatched elements based on tensor size
    total_elements = (output.float() * o_scale).numel()
    max_mismatched_elements = int(allowed_mismatch_rate * total_elements)

    assert_close_with_mismatch_tolerance(
        output.float() * o_scale,
        output_ref.float(),
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=max_mismatched_elements,
    )


TRTLLM_BATCH_PREFILL_SHAPES = [
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    (128, 16, 2, 5),
    (128, 32, 4, 1),
    (128, 64, 2, 8),
    (256, 16, 4, 8),
    (256, 32, 2, 8),
    (256, 64, 4, 1),
    (256, 64, 4, 5),
]


TRTLLM_BATCH_PREFILL_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
]


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    TRTLLM_BATCH_PREFILL_SHAPES,
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    TRTLLM_BATCH_PREFILL_DTYPES,
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_q_len", [511])
@pytest.mark.parametrize("max_kv_len", [2047])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("non_contiguous_query", [False, True])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
@pytest.mark.parametrize("causal", [True, False])
def test_trtllm_batch_prefill(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    causal: bool,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        causal,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        kv_dtype in ("fp8", "nvfp4"),
        head_dim,
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("provide_lse", [False, True])
def test_trtllm_batch_prefill_lse_contract(return_lse, provide_lse):
    _test_trtllm_batch_prefill(
        "HND",
        2,
        16,
        2,
        2,
        True,
        -1,
        "fp16",
        "fp16",
        "fp16",
        False,
        False,
        64,
        128,
        False,
        128,
        uses_shared_paged_kv_idx=True,
        return_lse=return_lse,
        provide_lse=provide_lse,
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (1, 16, 8, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [8192])
@pytest.mark.parametrize("max_kv_len", [8192])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_bs1(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        True,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        False,
        head_dim,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
    ],
)
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp8", "fp8", "bf16"),
        ("fp8", "fp8", "fp16"),
        ("fp8", "fp8", "fp8"),
    ],
)
@pytest.mark.parametrize(
    "head_dim,window_left",
    [
        (128, -1),
        (256, -1),
        (128, 127),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False, True])
@pytest.mark.parametrize("max_q_len", [511, 3023])
@pytest.mark.parametrize("max_kv_len", [2047, 8192])
def test_trtllm_batch_prefill_cubin_variants(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        True,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        False,
        head_dim,
        uses_spcompress=True,
    )


@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_dynamic_page_size_gqa(
    page_size: int,
    uses_shared_paged_kv_idx: bool,
) -> None:
    _skip_if_not_blackwell()
    _test_trtllm_batch_prefill(
        "HND",
        batch_size=4,
        page_size=page_size,
        num_kv_heads=2,
        head_grp_size=5,
        causal=True,
        window_left=-1,
        q_dtype="bf16",
        o_dtype="bf16",
        kv_dtype="bf16",
        enable_pdl=None,
        enable_sink=False,
        max_q_len=257,
        max_kv_len=1024,
        device_scale=False,
        head_dim=128,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )


@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize(
    "batch_size,page_size,num_kv_heads,head_grp_size",
    [
        (4, 16, 2, 1),
        (4, 32, 4, 5),
        (128, 16, 2, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("fp8", "fp8", "fp8"),
        ("fp8", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [None])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_q_len", [1, 255, 511])
@pytest.mark.parametrize("max_kv_len", [511, 2047])
@pytest.mark.parametrize("head_dim", [512])
@pytest.mark.parametrize("non_contiguous_query", [False])
@pytest.mark.parametrize("skips_softmax", [False, True])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_head_dim_512(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    window_left: int,
    q_dtype: str,
    o_dtype: str,
    kv_dtype: str,
    enable_pdl: bool,
    enable_sink: bool,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    non_contiguous_query: bool,
    skips_softmax: bool,
    uses_shared_paged_kv_idx: bool,
):
    _test_trtllm_batch_prefill(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        True,
        window_left,
        q_dtype,
        o_dtype,
        kv_dtype,
        enable_pdl,
        enable_sink,
        max_q_len,
        max_kv_len,
        kv_dtype in ("fp8", "nvfp4"),
        head_dim,
        non_contiguous_query=non_contiguous_query,
        skips_softmax=skips_softmax,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
    )
