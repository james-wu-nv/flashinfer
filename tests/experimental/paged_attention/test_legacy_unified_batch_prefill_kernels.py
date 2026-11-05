"""Legacy -> unified: tests/attention/test_batch_prefill_kernels.py

Every paged-prefill test of the legacy file runs its legacy fixture through
``PagedAttention``.  The legacy tests call ``BatchPrefillWithPagedKVCacheWrapper``
with ``backend="fa2"`` or ``"auto"`` (fa2 here, fa3 on SM90a).

- Attention-correctness tests (the main grid, the tuple-KV grid, custom mask,
  head_dim 512) run once per backend: fa2, fa3, cudnn and trtllm-gen; node ids
  are the legacy ids with a ``-<backend>`` suffix.  A backend that cannot run
  a case on this GPU is skipped with the resolver's reason.
- Tests of an fa2 implementation detail (the lazy stride router, the shared
  K/V shared-memory layout, the CTA-tile probe, the split-KV merge) run fa2
  only, with the legacy ids.
- Configurations ``PagedAttention`` cannot express are expected failures
  (``pytest.xfail``) that name the missing API: fused RoPE
  (``pos_encoding_mode``), NVFP4 KV, multi-item scoring, and causal rows with
  ``q_len > kv_len`` (fully masked rows).

The legacy page ids become ``PagedAttentionMetadata.csr`` (legal at every page
size; backends that need a dense table derive it).  Like the legacy runs, the
plans request the LSE only where the legacy run returns it, and use the legacy
workspace sizes.  Each case checks the legacy reference at the legacy
tolerance; the fp32 oracle is added only where the legacy check is relative
(custom mask vs causal).

The legacy ``use_cuda_graph=True`` rows were an unconditional xfail (the
wrapper's workspace overflowed); here they run the unified graph lifecycle
with a workspace sized by ``workspace_requirements``.  Ragged, single-prefill
and torch.compile functions of the legacy file are out of scope.
"""

import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    GraphCapacity,
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from flashinfer.utils import get_compute_capability
from tests.test_helpers.paged_kv import make_padded_paged_kv_view
from tests.test_helpers.test_helpers import ref_single_prefill

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_prefill_kernels.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache",
        ["test_batch_prefill_with_paged_kv_cache"],
        "partial",
        "same grid and combined fp16 pool on fa2 / fa3 / cudnn / trtllm-gen; legacy "
        "1e-3 vs single_prefill + caller-buffer re-run; use_cuda_graph rows (a legacy "
        "xfail) run the graph lifecycle; ROPE_LLAMA rows xfail (no fused RoPE)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_plan_reuse",
        ["test_batch_prefill_lazy_stride_router_plan_reuse"],
        "partial",
        "same fixture on fa2, one plan over equal / unequal / equal V strides, plan "
        "info unchanged across runs, legacy tolerances; plan() has no "
        "fixed_split_size, so the legacy forced split-KV is not reproduced",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_nvfp4",
        ["test_batch_prefill_lazy_stride_router_nvfp4"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_head_dim_512",
        ["test_batch_prefill_with_paged_kv_cache_head_dim_512"],
        "partial",
        "same fixture on fa2 / fa3 / cudnn / trtllm-gen; legacy 1e-3 vs single_prefill "
        "and the caller-buffer re-run (output and LSE); ROPE_LLAMA rows xfail",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_tuple_paged_kv_cache",
        ["test_batch_prefill_with_tuple_paged_kv_cache"],
        "partial",
        "same grid and separate fp16 K / V pools on fa2 / fa3 / cudnn / trtllm-gen; "
        "legacy 1e-3 vs single_prefill; use_cuda_graph rows run the graph lifecycle; "
        "ROPE_LLAMA rows xfail",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_custom_mask",
        ["test_batch_prefill_with_paged_kv_cache_custom_mask"],
        "partial",
        "same grid on fa2 / fa3 / cudnn / trtllm-gen; custom-mask vs causal at the "
        "legacy 1e-3 plus the fp32 oracle on both; ROPE_LLAMA rows xfail",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_head_dim_512",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_custom_mask",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring",
        ["test_batch_prefill_with_paged_kv_cache_multi_item_scoring"],
        "unsupported-by-design",
        "xfail: plan() has no multi-item scoring inputs (and no fused RoPE), an API "
        "limit",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256",
        ["test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256"],
        "native-only",
        "fa2 CTA-tile probe at head dims (448, 256), which fa2 does not declare: "
        "skipped with the resolver's reason",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides",
        ["test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides"],
        "equivalent",
        "same fixture on fa2 (D512 K / V views with different strides); legacy exact "
        "fp32 reference at 2e-3",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_large_head"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV (and no fused RoPE) in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16"],
        "unsupported-by-design",
        "xfail: no NVFP4 KV (and no fused RoPE) in PagedAttention (API limit)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4_large_head",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_ragged_kv_cache_nvfp4_rope_large_head",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_single_prefill_torch_compile_cuda_graph",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache under torch.compile, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_ragged_prefill_one_valid_key",
        [],
        "out-of-scope",
        "ragged KV, not paged prefill",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_fully_masked_rows",
        ["test_paged_prefill_fully_masked_rows"],
        "unsupported-by-design",
        "xfail: causal plans require q_len <= kv_len per request (the fully masked "
        "rows are rejected by the API contract)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_split_kv_empty_chunk",
        ["test_paged_prefill_split_kv_empty_chunk"],
        "equivalent",
        "same fixture on fa2, legacy 1e-2 vs ref_single_prefill on output and LSE",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]
DEVICE = torch.device("cuda:0")
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

ROPE_XFAIL = "API limit: PagedAttention.plan() has no fused RoPE (pos_encoding_mode)"
NVFP4_XFAIL = (
    "API limit: PagedAttention has no NVFP4 KV cache (packed uint8 FP4 KV and "
    "its kv_cache_sf scale factors)"
)


def skip_if_head_dim_unsupported(head_dim):
    # the legacy gate: 16-bit FA2 head_dim > 256 uses the Ampere+ large-head path
    if head_dim > 256 and get_compute_capability(DEVICE)[0] < 8:
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")


def _accumulate_mismatch_count(mismatch_counts, out, ref, rtol, atol):
    """The legacy per-request comparison: record the mismatch count on
    device, check once per test."""
    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    close = torch.isclose(out.float(), ref.float(), rtol=rtol, atol=atol)
    mismatch_counts.append((~close).sum())


def _assert_no_ref_mismatch(mismatch_counts):
    counts = torch.stack(mismatch_counts).cpu()  # single device sync
    if counts.any():
        bad = counts.nonzero().flatten().tolist()
        raise AssertionError(
            f"output mismatches reference in request(s) {bad}; "
            f"mismatched elements per request: {[int(counts[i]) for i in bad]}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
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
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    if pos_encoding_mode != "NONE":
        pytest.xfail(ROPE_XFAIL)
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=return_lse,
            logits_soft_cap=logits_soft_cap or None,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.half()
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    kv_cache = (kv_data[:, 0], kv_data[:, 1])  # PagedAttention takes the (K, V) pair

    metadata = PagedAttentionMetadata.csr(
        q_indptr_cpu.to(0),
        torch.full((batch_size,), kv_len, dtype=torch.int32).to(0),  # kv_seq_lens
        kv_indices_cpu.to(0),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        logits_soft_cap=logits_soft_cap or None,
    )
    if not use_cuda_graph:
        workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
        )
        attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
        attn.plan(
            metadata,
            **plan_kw,
            lse_mode="base2" if return_lse else "none",
            backend=backend,
        )
        assert attn.backend == backend
        o, lse = attn.run(q, kv_cache)

        # test with pre-allocated output
        o_buffer = torch.empty_like(o)
        if return_lse:
            lse_buffer = torch.empty_like(lse)
            attn.run(q, kv_cache, out=o_buffer, lse=lse_buffer)
        else:
            attn.run(q, kv_cache, out=o_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    else:
        # The unified graph lifecycle: a capacity for this batch and a
        # workspace sized for it (the legacy 256 MiB buffer overflowed, which
        # is why the legacy xfails these rows), a warm-up plan (kv_len capped
        # at qo_len, since causal needs kv >= q), warm-up on a side stream,
        # capture, update() to the real batch, replay.
        capacity = GraphCapacity(
            batch_size=batch_size,
            total_q_tokens=batch_size * qo_len,
            max_q_len=qo_len,
            max_kv_len=kv_len,
            page_size=page_size,
            kv_input_form="page_indices",
            flat_capacity=total_num_pages,
        )
        workspace_buffer = torch.empty(
            PagedAttention.workspace_requirements(
                capacity, device=DEVICE, **plan_kw, need_lse=return_lse, backend=backend
            ),
            dtype=torch.uint8,
            device="cuda:0",
        )
        attn = PagedAttention(
            DEVICE, graph_capacity=capacity, workspace_buffer=workspace_buffer
        )
        warmup_kv_len = min(qo_len, kv_len)
        warmup_pages = (warmup_kv_len + page_size - 1) // page_size
        metadata_warmup = PagedAttentionMetadata.csr(
            q_indptr_cpu.to(0),
            torch.full((batch_size,), warmup_kv_len, dtype=torch.int32).to(0),
            # each request's first warmup_pages pages, in request order
            kv_indices_cpu.reshape(batch_size, num_pages_per_seq)[:, :warmup_pages]
            .reshape(-1)
            .to(0),
            page_size=page_size,
            max_q_len=qo_len,
            max_kv_len=warmup_kv_len,
        )
        attn.plan(
            metadata_warmup,
            **plan_kw,
            lse_mode="base2" if return_lse else "none",
            backend=backend,
        )
        assert attn.backend == backend

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o, _ = attn.run(q, kv_cache)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o, _ = attn.run(q, kv_cache)

        attn.update(metadata)

        g.replay()

    mismatch_counts = []
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, :, : kv_last_page_len_cpu[i]
                    ]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        _accumulate_mismatch_count(mismatch_counts, o_i, o_ref_i, rtol=1e-3, atol=1e-3)
    _assert_no_ref_mismatch(mismatch_counts)


@pytest.mark.parametrize("kv_layout,head_dim", [("NHD", 64), ("HND", 128)])
def test_batch_prefill_lazy_stride_router_plan_reuse(kv_layout, head_dim):
    """Reuse one equal-primary plan across equal/unequal/equal paged runs
    (fa2's stride router)."""
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim_qk=head_dim,
            q_dtype=torch.bfloat16,
            page_size=16,
            kv_layout=kv_layout,
            causal=True,
            kv_input_form="page_indices",
            backend="fa2",
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")
    torch.manual_seed(42)
    batch_size, qo_len, kv_len, page_size = 2, 17, 97, 16
    num_qo_heads, num_kv_heads = 8, 2
    pages_per_request = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_request

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if kv_layout == "NHD":
        cache_shape = (total_pages, page_size, num_kv_heads, head_dim)
        to_dense = lambda cache: cache.reshape(-1, num_kv_heads, head_dim)
    else:
        cache_shape = (total_pages, num_kv_heads, page_size, head_dim)
        to_dense = lambda cache: cache.permute(0, 2, 1, 3).reshape(
            -1, num_kv_heads, head_dim
        )

    k = torch.randn(cache_shape, device="cuda", dtype=torch.bfloat16) / 4
    v_equal = torch.randn(cache_shape, device="cuda", dtype=torch.bfloat16) / 4
    v_unequal = make_padded_paged_kv_view(v_equal, kv_layout)
    assert k.shape == v_equal.shape == v_unequal.shape
    assert k.stride() == v_equal.stride()
    assert k.stride() != v_unequal.stride()

    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda", dtype=torch.int32) * qo_len
    )
    kv_indices = torch.arange(total_pages, device="cuda", dtype=torch.int32)

    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        torch.full((batch_size,), kv_len, device="cuda", dtype=torch.int32),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(
        DEVICE,
        workspace_buffer=torch.empty(
            128 * 1024 * 1024, device="cuda", dtype=torch.uint8
        ),
    )
    # The legacy plan also passes fixed_split_size=2 to force split-KV;
    # PagedAttention.plan() has no such knob, so this plan uses the default
    # split heuristic.
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        kv_layout=kv_layout,
        causal=True,
        backend="fa2",
    )
    assert attn.backend == "fa2"
    # fa2's plan info, read from the wrapper the fa2 backend holds
    plan_info = tuple(attn._impl._active._active._plan_info)
    outputs = [
        attn.run(q, cache)[0] for cache in ((k, v_equal), (k, v_unequal), (k, v_equal))
    ]
    assert tuple(attn._impl._active._active._plan_info) == plan_info

    expected_batches = []
    for batch_idx in range(batch_size):
        q_i = q[batch_idx * qo_len : (batch_idx + 1) * qo_len]
        page_slice = slice(
            batch_idx * pages_per_request,
            (batch_idx + 1) * pages_per_request,
        )
        expected_i, _ = ref_single_prefill(
            q_i,
            to_dense(k[page_slice])[:kv_len],
            to_dense(v_equal[page_slice])[:kv_len],
            causal=True,
        )
        expected_batches.append(expected_i)
    expected = torch.cat(expected_batches)
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=1e-2, atol=1e-2)


def test_batch_prefill_lazy_stride_router_nvfp4():
    pytest.xfail(NVFP4_XFAIL)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_with_paged_kv_cache_head_dim_512(
    causal,
    pos_encoding_mode,
    backend,
):
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)
    if pos_encoding_mode != "NONE":
        pytest.xfail(ROPE_XFAIL)
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=16,
            kv_layout="NHD",
            causal=causal,
            need_lse=True,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    batch_size = 2
    kv_len = 97
    qo_len = 17
    page_size = 16
    num_kv_heads = 4
    num_qo_heads = 4
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
    kv_data = kv_data_fp32.half()
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    kv_cache = (kv_data[:, 0], kv_data[:, 1])

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    metadata = PagedAttentionMetadata.csr(
        q_indptr_cpu.to(0),
        torch.full((batch_size,), kv_len, dtype=torch.int32).to(0),  # kv_seq_lens
        kv_indices_cpu.to(0),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        lse_mode="base2",
        backend=backend,
    )
    assert attn.backend == backend
    o, lse = attn.run(q, kv_cache)

    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    attn.run(q, kv_cache, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    mismatch_counts = []
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 0].reshape(
                    -1, num_kv_heads, head_dim
                ),
                kv_data_fp32[
                    kv_indptr_cpu[i + 1] - 1, 0, : kv_last_page_len_cpu[i], :
                ].reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1, 1].reshape(
                    -1, num_kv_heads, head_dim
                ),
                kv_data_fp32[
                    kv_indptr_cpu[i + 1] - 1, 1, : kv_last_page_len_cpu[i], :
                ].reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            backend="fa2",
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        _accumulate_mismatch_count(mismatch_counts, o_i, o_ref_i, rtol=1e-3, atol=1e-3)
    _assert_no_ref_mismatch(mismatch_counts)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
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
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    if pos_encoding_mode != "NONE":
        pytest.xfail(ROPE_XFAIL)
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=return_lse,
            logits_soft_cap=logits_soft_cap or None,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1).int() * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
            for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].half() for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr_cpu = torch.arange(0, batch_size + 1).int() * num_pages_per_seq
    kv_indices_cpu = torch.arange(0, total_num_pages).int()
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )

    metadata = PagedAttentionMetadata.csr(
        q_indptr_cpu.to(0),
        torch.full((batch_size,), kv_len, dtype=torch.int32).to(0),  # kv_seq_lens
        kv_indices_cpu.to(0),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        logits_soft_cap=logits_soft_cap or None,
    )
    if not use_cuda_graph:
        workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
        )
        attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
        attn.plan(
            metadata,
            **plan_kw,
            lse_mode="base2" if return_lse else "none",
            backend=backend,
        )
        assert attn.backend == backend
        o, _ = attn.run(q, kv_data)
    else:
        # the unified graph lifecycle (see test_batch_prefill_with_paged_kv_cache)
        capacity = GraphCapacity(
            batch_size=batch_size,
            total_q_tokens=batch_size * qo_len,
            max_q_len=qo_len,
            max_kv_len=kv_len,
            page_size=page_size,
            kv_input_form="page_indices",
            flat_capacity=total_num_pages,
        )
        workspace_buffer = torch.empty(
            PagedAttention.workspace_requirements(
                capacity, device=DEVICE, **plan_kw, need_lse=return_lse, backend=backend
            ),
            dtype=torch.uint8,
            device="cuda:0",
        )
        attn = PagedAttention(
            DEVICE, graph_capacity=capacity, workspace_buffer=workspace_buffer
        )
        warmup_kv_len = min(qo_len, kv_len)
        warmup_pages = (warmup_kv_len + page_size - 1) // page_size
        metadata_warmup = PagedAttentionMetadata.csr(
            q_indptr_cpu.to(0),
            torch.full((batch_size,), warmup_kv_len, dtype=torch.int32).to(0),
            # each request's first warmup_pages pages, in request order
            kv_indices_cpu.reshape(batch_size, num_pages_per_seq)[:, :warmup_pages]
            .reshape(-1)
            .to(0),
            page_size=page_size,
            max_q_len=qo_len,
            max_kv_len=warmup_kv_len,
        )
        attn.plan(
            metadata_warmup,
            **plan_kw,
            lse_mode="base2" if return_lse else "none",
            backend=backend,
        )
        assert attn.backend == backend
        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o, _ = attn.run(q, kv_data)
        torch.cuda.current_stream().wait_stream(s)
        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o, _ = attn.run(q, kv_data)
        attn.update(metadata)
        g.replay()

    k_cache, v_cache = kv_data_fp32
    mismatch_counts = []
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                k_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        vi = torch.cat(
            [
                v_cache[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr_cpu[i + 1] - 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).half()
        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi,
            ki,
            vi,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        _accumulate_mismatch_count(mismatch_counts, o_i, o_ref_i, rtol=1e-3, atol=1e-3)
    _assert_no_ref_mismatch(mismatch_counts)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [12, 17, 128])
@pytest.mark.parametrize("kv_len", [54, 97, 512, 2048])
@pytest.mark.parametrize("qo_len", [37, 17, 127, 577])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [True])
@pytest.mark.parametrize("contiguous_kv", [True])
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
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported for custom mask test")
    if pos_encoding_mode != "NONE":
        pytest.xfail(ROPE_XFAIL)
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=False,
            need_lse=return_lse,
            logits_soft_cap=logits_soft_cap or None,
            custom_mask=True,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data = torch.randn(*kv_shape, dtype=torch.float16, device="cuda:0")
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data = torch.randn(*kv_shape, dtype=torch.float16, device="cuda:0")
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_cache = (kv_data[:, 0], kv_data[:, 1])

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    metadata = PagedAttentionMetadata.csr(
        q_indptr,
        torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda:0"),
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        logits_soft_cap=logits_soft_cap or None,
        lse_mode="base2" if return_lse else "none",
        backend=backend,
    )
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device="cuda:0"),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)

    # use custom mask (the legacy plan passes no causal: non-causal + the mask)
    attn.plan(metadata, **plan_kw, causal=False, custom_mask=custom_mask)
    assert attn.backend == backend
    o_custom, lse_custom = attn.run(q, kv_cache)

    # use causal
    attn.plan(metadata, **plan_kw, causal=True)
    assert attn.backend == backend
    o_causal, lse_causal = attn.run(q, kv_cache)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)

    # fp32 oracle (the legacy check compares the two runs with each other)
    for out, causal, mask in ((o_custom, False, custom_mask), (o_causal, True, None)):
        ref_out, _ = reference_paged_prefill(
            q,
            *kv_cache,
            q_indptr.cpu(),
            torch.full((batch_size,), kv_len, dtype=torch.int32),
            None,
            page_size,
            causal,
            kv_layout=kv_layout,
            kv_page_indices=kv_indices,
            custom_mask=mask,
        )
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr",
    [
        (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
        (97, 81, 16, list(range(80)) + [0], 97, [79]),
    ],
)
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["ROPE_LLAMA"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
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
):
    pytest.xfail(
        "API limit: PagedAttention.plan() has no multi-item scoring inputs "
        "(prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, "
        "max_item_len_ptr) and no fused RoPE"
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
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
        pytest.skip("qo_len > kv_len and causal is not supported")
    pytest.xfail(NVFP4_XFAIL)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views(kv_layout):
    pytest.xfail(NVFP4_XFAIL)


@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(512, 256), (256, 128)])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [2, 8])
@pytest.mark.parametrize("causal", [True])
def test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric(
    head_dim_qk,
    head_dim_vo,
    page_size,
    num_kv_heads,
    causal,
):
    skip_if_head_dim_unsupported(head_dim_qk)
    if get_compute_capability(DEVICE)[0] < 10:
        pytest.skip(
            "asymmetric NVFP4 KV prefill uses the NVFP4 KV quantization kernel, "
            "which requires SM100 or newer"
        )
    pytest.xfail(NVFP4_XFAIL)


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
def test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256(kv_dtype):
    """fa2's CTA_TILE_Q pick at head dims (448, 256)."""
    head_dim_qk = 448
    head_dim_vo = 256
    skip_if_head_dim_unsupported(head_dim_qk)
    if kv_dtype.itemsize == 1 and get_compute_capability(DEVICE)[0] < 10:
        pytest.skip("FP8 KV with head_dim > 256 requires SM100 or newer")
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "shared_memory_per_block_optin", None) is None:
        pytest.skip("torch does not expose shared_memory_per_block_optin")
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=2,
            num_kv_heads=2,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=torch.float16,
            kv_dtype=kv_dtype,
            page_size=16,
            kv_layout="NHD",
            causal=False,
            backend="fa2",
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")
    raise AssertionError(
        "fa2 now declares head dims (448, 256): port the legacy CTA-tile probe"
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("qo_len", [17, 65])
def test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides(kv_layout, qo_len):
    """D512 fp16 K and V pools as views of differently padded parents: V rows
    must be addressed with V's strides (fa2's shared K/V shared-memory
    layout)."""
    head_dim = 512
    skip_if_head_dim_unsupported(head_dim)
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=2,
            num_kv_heads=2,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            page_size=16,
            kv_layout=kv_layout,
            causal=True,
            kv_input_form="page_indices",
            backend="fa2",
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")

    torch.manual_seed(42)
    batch_size = 2
    kv_len = 97
    page_size = 16
    num_kv_heads = 2
    num_qo_heads = 2  # group_size 1: avg_packed_qo_len == qo_len
    causal = True

    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)

    def padded_pool(num_padding_heads):
        if kv_layout == "NHD":
            parent = torch.randn(
                total_num_pages,
                page_size,
                num_kv_heads + num_padding_heads,
                head_dim,
                device="cuda:0",
                dtype=torch.float16,
            )
            return parent[:, :, :num_kv_heads, :]
        parent = torch.randn(
            total_num_pages,
            num_kv_heads + num_padding_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        return parent[:, :num_kv_heads, :, :]

    k = padded_pool(1)
    v = padded_pool(3)
    assert k.shape == v.shape
    assert not k.is_contiguous() and not v.is_contiguous()
    # The point of the test: genuinely different stride families.
    assert k.stride() != v.stride()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    metadata = PagedAttentionMetadata.csr(
        q_indptr_cpu.to("cuda:0"),
        torch.full((batch_size,), kv_len, dtype=torch.int32).to("cuda:0"),
        kv_indices_cpu.to("cuda:0"),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(DEVICE, workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, _ = attn.run(q, (k, v))
    assert o.shape == (batch_size * qo_len, num_qo_heads, head_dim)

    # Exact float32 reference on the logical (view) K/V values.
    sm_scale = head_dim**-0.5
    perm = (0, 1, 2, 3) if kv_layout == "NHD" else (0, 2, 1, 3)
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()
        ki = (
            k[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
            .permute(*perm)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .float()
        )
        vi = (
            v[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
            .permute(*perm)
            .reshape(-1, num_kv_heads, head_dim)[:kv_len]
            .float()
        )
        logits = torch.einsum("qhd,khd->hqk", qi, ki) * sm_scale
        if causal:
            qpos = torch.arange(qo_len, device="cuda:0").unsqueeze(1)
            kpos = torch.arange(kv_len, device="cuda:0").unsqueeze(0)
            allowed = kpos <= qpos + (kv_len - qo_len)
            logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        o_ref_i = torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), vi)
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]].float()
        torch.testing.assert_close(o_i, o_ref_i, rtol=2e-3, atol=2e-3)


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head():
    skip_if_head_dim_unsupported(512)
    pytest.xfail(NVFP4_XFAIL)


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16():
    skip_if_head_dim_unsupported(512)
    pytest.xfail(NVFP4_XFAIL)


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head():
    skip_if_head_dim_unsupported(512)
    pytest.xfail(NVFP4_XFAIL + "; no fused RoPE either")


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16():
    skip_if_head_dim_unsupported(512)
    pytest.xfail(NVFP4_XFAIL + "; no fused RoPE either")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_fully_masked_rows(dtype):
    """Legacy (#4452): causal q_len 34 > kv_len 1 leaves 33 fully masked rows
    (out 0, LSE -inf)."""
    pytest.xfail(
        "API contract: a causal plan requires q_len <= kv_len per request, so "
        "the fully masked rows are rejected"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_split_kv_empty_chunk(dtype):
    # Multi-token causal prefill (qo_len=2, kv_len=129) with split-KV: the
    # first token's rows have no attendable key in the last kv chunk, so the
    # merge path must combine a finite partial state with an empty one
    # (partial lse=-inf, d=0) without producing NaN.
    bs, qo_len, kv_len = 1, 2, 129
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128
    page_size = 16
    try:
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            page_size=page_size,
            kv_layout="NHD",
            causal=True,
            need_lse=True,
            kv_input_form="page_indices",
            backend="fa2",
        )
    except ValueError as e:
        pytest.skip(f"fa2 not runnable here: {e}")
    pages_per = (kv_len + page_size - 1) // page_size
    q = (
        torch.randn(bs * qo_len, num_qo_heads, head_dim, dtype=dtype, device="cuda")
        / 10
    )
    kv_data = (
        torch.randn(
            pages_per, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device="cuda"
        )
        / 10
    )
    qo_indptr = torch.tensor([0, bs * qo_len], dtype=torch.int32, device="cuda")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        torch.tensor([kv_len], dtype=torch.int32, device="cuda"),
        torch.arange(pages_per, dtype=torch.int32, device="cuda"),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    attn = PagedAttention(DEVICE, workspace_buffer=workspace)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_dtype=dtype,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, lse = attn.run(q, (kv_data[:, 0], kv_data[:, 1]))

    k = kv_data[:, 0].reshape(-1, num_kv_heads, head_dim)
    v = kv_data[:, 1].reshape(-1, num_kv_heads, head_dim)
    o_ref, lse_ref = ref_single_prefill(q, k[:kv_len], v[:kv_len], causal=True)
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)
