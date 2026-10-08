"""Legacy -> unified: tests/attention/test_batch_prefill_kernels.py

Every paged-prefill test of the legacy file runs its legacy fixture through
``PagedAttention`` pinned to fa2: the legacy tests call
``BatchPrefillWithPagedKVCacheWrapper`` with ``backend="fa2"`` or ``"auto"``,
which picks fa2 on this device.  Same grids, same tensors in the same RNG
order, so the node ids equal the legacy ids.  The legacy CSR maps losslessly
to ``PagedAttentionMetadata`` (every request holds ``kv_len`` tokens on its
own pages; page size < 8 -> ``.csr``, else ``.dense``).  Each case checks the
legacy reference at the legacy tolerance, and the output and LSE against the
fp32 paged-attention oracle.

The legacy ``use_cuda_graph=True`` rows were an unconditional xfail (the
wrapper's workspace overflowed); here they run the graph lifecycle with a
workspace sized by ``workspace_requirements`` and must pass.

Not expressible, asserted as rejections: fused RoPE (``pos_encoding_mode``),
NVFP4 KV, multi-item scoring, the (448, 256) head dims and causal rows with
``q_len > kv_len``.  Ragged, single-prefill and torch.compile functions of the
legacy file are out of scope.
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
        "same grid and combined fp16 NHD pool on fa2; NONE rows: legacy 1e-3 vs "
        "single_prefill + caller-buffer re-run + oracle, use_cuda_graph rows (a legacy "
        "xfail) run the graph lifecycle and pass; ROPE_LLAMA rows assert the plan() "
        "TypeError (no fused RoPE)",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_plan_reuse",
        ["test_batch_prefill_lazy_stride_router_plan_reuse"],
        "equivalent",
        "same fixture on fa2, one plan over equal / unequal / equal V strides at the "
        "legacy tolerances plus the oracle; the fixed_split_size knob has no "
        "counterpart",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_nvfp4",
        ["test_batch_prefill_lazy_stride_router_nvfp4"],
        "unsupported-by-design",
        "NVFP4 KV (uint8) is not a declared KV dtype: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_head_dim_512",
        ["test_batch_prefill_with_paged_kv_cache_head_dim_512"],
        "partial",
        "same fixture on fa2 at D512; NONE rows: legacy 1e-3 vs single_prefill(fa2) + "
        "caller buffers + oracle; ROPE_LLAMA rows assert the plan() TypeError",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_tuple_paged_kv_cache",
        ["test_batch_prefill_with_tuple_paged_kv_cache"],
        "partial",
        "as the main grid with two separate fp16 NHD pools",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_custom_mask",
        ["test_batch_prefill_with_paged_kv_cache_custom_mask"],
        "partial",
        "same grid on fa2; the tril mask as custom_mask equals the causal plan at 1e-3 "
        "plus the oracle for both; ROPE_LLAMA rows assert the plan() TypeError",
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
        "plan() has no multi-item fields (prefix_len_ptr, ...): TypeError asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4"],
        "unsupported-by-design",
        "NVFP4 KV (uint8) is not a declared KV dtype: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views"],
        "unsupported-by-design",
        "NVFP4 KV: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric"],
        "unsupported-by-design",
        "NVFP4 KV: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256",
        ["test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256"],
        "native-only",
        "pins a planner-internal CTA tile; (448, 256) is not a declared head-dim pair: "
        "resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides",
        ["test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides"],
        "equivalent",
        "same fixture on fa2 (D512, unequal K/V strides), legacy fp32 reference at "
        "2e-3 plus the oracle",
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
        "NVFP4 KV: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16"],
        "unsupported-by-design",
        "NVFP4 KV: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head"],
        "unsupported-by-design",
        "NVFP4 KV and fused RoPE: resolve rejection asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
        ["test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16"],
        "unsupported-by-design",
        "NVFP4 KV and fused RoPE: resolve rejection asserted",
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
        "causal with q_len 34 > kv_len 1 is rejected by plan() (no fully-masked-row "
        "policy): ValueError asserted",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_split_kv_empty_chunk",
        ["test_paged_prefill_split_kv_empty_chunk"],
        "equivalent",
        "same fixture on fa2, legacy 1e-2 vs ref_single_prefill on output and LSE plus "
        "the oracle",
    ),
]

DEVICE = torch.device("cuda:0")
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _skip_if_head_dim_unsupported(head_dim):
    # the legacy gate: 16-bit FA2 head_dim > 256 uses the Ampere+ large-head path
    if head_dim > 256 and get_compute_capability(DEVICE)[0] < 8:
        pytest.skip("16-bit FA2 head_dim > 256 is only supported on SM80 or newer")


def _uniform_metadata(batch_size, qo_len, kv_len, page_size, pages_per_seq):
    """The legacy uniform batch (request i owns pages [i * pages_per_seq,
    (i + 1) * pages_per_seq) and attends its first ``kv_len`` tokens) in
    unified form: ``.csr`` below page size 8, else the ``.dense`` table."""
    table = torch.arange(batch_size * pages_per_seq, dtype=torch.int32).reshape(
        batch_size, pages_per_seq
    )
    qo_indptr_cpu = torch.arange(batch_size + 1, dtype=torch.int32) * qo_len
    kv_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)
    common = dict(
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    if page_size < 8:
        used = (kv_len + page_size - 1) // page_size
        page_ids = table[:, :used].reshape(-1)
        return PagedAttentionMetadata.csr(
            qo_indptr_cpu.to(DEVICE),
            kv_lens_cpu.to(DEVICE),
            page_ids.to(DEVICE),
            **common,
        )
    return PagedAttentionMetadata.dense(
        qo_indptr_cpu.to(DEVICE), kv_lens_cpu.to(DEVICE), table.to(DEVICE), **common
    )


def _assert_oracle(out, lse, q, k_cache, v_cache, md, causal, kv_layout, **kw):
    """fp32 oracle: output and base-2 LSE."""
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        md.block_tables,
        md.page_size,
        causal,
        kv_layout=kv_layout,
        kv_page_indices=md.kv_page_indices,
        **kw,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def _single_prefill_reference(q, k_cache, v_cache, qo_len, kv_len, **kw):
    """The legacy reference: ``single_prefill_with_kv_cache`` per request on
    its K/V gathered from its pages (NHD pools, uniform batch)."""
    batch_size = q.shape[0] // qo_len
    pages_per_seq = k_cache.shape[0] // batch_size
    num_kv_heads, head_dim = k_cache.shape[-2:]
    outs = []
    for i in range(batch_size):
        pages = slice(i * pages_per_seq, (i + 1) * pages_per_seq)
        ki = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        vi = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        qi = q[i * qo_len : (i + 1) * qo_len]
        outs.append(flashinfer.prefill.single_prefill_with_kv_cache(qi, ki, vi, **kw))
    return torch.cat(outs)


def _run_cuda_graph(q, kv_cache, md, warmup_md, **plan_kw):
    """The legacy ``use_cuda_graph`` flow on the unified graph lifecycle:
    plan a warm-up batch, warm up on a side stream, capture one ``run()``,
    ``update()`` to the real batch, replay.  The workspace is sized by
    ``workspace_requirements`` for the batch's capacity (several GiB at the
    largest geometries, where the legacy wrapper's workspace overflowed)."""
    if md.block_tables is not None:
        paging = dict(table_width=md.block_tables.shape[1])
    else:
        paging = dict(
            kv_input_form="page_indices", flat_capacity=md.kv_page_indices.numel()
        )
    cap = GraphCapacity(
        batch_size=md.kv_seq_lens_cpu.numel(),
        total_q_tokens=q.shape[0],
        max_q_len=md.max_q_len,
        max_kv_len=md.max_kv_len,
        page_size=md.page_size,
        **paging,
    )
    nbytes = PagedAttention.workspace_requirements(
        cap, device=DEVICE, **plan_kw, need_lse=True, backend="fa2"
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)
    attn = PagedAttention(DEVICE, graph_capacity=cap, workspace_buffer=workspace)
    attn.plan(warmup_md, **plan_kw, lse_mode="base2", backend="fa2")
    assert attn.backend == "fa2"
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            attn.run(q, kv_cache)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(q, kv_cache)
    attn.update(md)
    g.replay()
    return out, lse


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
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
    )
    if pos_encoding_mode != "NONE":
        # RoPE is the caller's transform: plan() has no fused positional encoding
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            PagedAttention(DEVICE).plan(
                md, **plan_kw, pos_encoding_mode=pos_encoding_mode
            )
        return

    # the legacy fixture (unseeded there; same draws in the same order)
    torch.manual_seed(0)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]  # NHD
    kv_data = torch.randn(*kv_shape, dtype=torch.float32, device=DEVICE).half()
    kv_cache = (kv_data[:, 0], kv_data[:, 1])

    if use_cuda_graph:
        # the legacy warm-up planned a shorter batch; causal needs kv >= q
        warmup_md = _uniform_metadata(
            batch_size, qo_len, min(qo_len, kv_len), page_size, num_pages_per_seq
        )
        o, lse = _run_cuda_graph(q, kv_cache, md, warmup_md, **plan_kw)
    else:
        attn = PagedAttention(DEVICE)
        attn.plan(md, **plan_kw, lse_mode="base2", backend="fa2")
        assert attn.backend == "fa2"
        o, lse = attn.run(q, kv_cache)

        # legacy: a second run into pre-allocated out / lse buffers
        o_buffer, lse_buffer = torch.empty_like(o), torch.empty_like(lse)
        attn.run(q, kv_cache, out=o_buffer, lse=lse_buffer)
        torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)

    # legacy reference at the legacy tolerance
    o_ref = _single_prefill_reference(
        q, *kv_cache, qo_len, kv_len, causal=causal, logits_soft_cap=logits_soft_cap
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)

    _assert_oracle(o, lse, q, *kv_cache, md, causal, kv_layout)


@pytest.mark.parametrize("kv_layout,head_dim", [("NHD", 64), ("HND", 128)])
def test_batch_prefill_lazy_stride_router_plan_reuse(kv_layout, head_dim):
    """Reuse one plan across equal/unequal/equal V stride runs."""
    torch.manual_seed(42)
    batch_size, qo_len, kv_len, page_size = 2, 17, 97, 16
    num_qo_heads, num_kv_heads = 8, 2
    pages_per_request = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_request

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    if kv_layout == "NHD":
        cache_shape = (total_pages, page_size, num_kv_heads, head_dim)
        to_dense = lambda cache: cache.reshape(-1, num_kv_heads, head_dim)
    else:
        cache_shape = (total_pages, num_kv_heads, page_size, head_dim)
        to_dense = lambda cache: cache.permute(0, 2, 1, 3).reshape(
            -1, num_kv_heads, head_dim
        )

    k = torch.randn(cache_shape, device=DEVICE, dtype=torch.bfloat16) / 4
    v_equal = torch.randn(cache_shape, device=DEVICE, dtype=torch.bfloat16) / 4
    v_unequal = make_padded_paged_kv_view(v_equal, kv_layout)
    assert k.shape == v_equal.shape == v_unequal.shape
    assert k.stride() == v_equal.stride()
    assert k.stride() != v_unequal.stride()

    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, pages_per_request)
    attn = PagedAttention(DEVICE)
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        kv_layout=kv_layout,
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    caches = [(k, v_equal), (k, v_unequal), (k, v_equal)]
    results = [attn.run(q, cache) for cache in caches]
    outputs = [out for out, _ in results]

    # legacy reference at the legacy tolerances
    expected_batches = []
    for batch_idx in range(batch_size):
        q_i = q[batch_idx * qo_len : (batch_idx + 1) * qo_len]
        page_slice = slice(
            batch_idx * pages_per_request, (batch_idx + 1) * pages_per_request
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

    for (out, lse), (k_cache, v_cache) in zip(results, caches, strict=True):
        _assert_oracle(out, lse, q, k_cache, v_cache, md, True, kv_layout)


def _assert_nvfp4_rejected(**cfg):
    """A packed NVFP4 KV cache (uint8 FP4x2 + scale factors) is not a
    declared KV dtype of fa2."""
    with pytest.raises(ValueError, match="unsupported kv dtype torch.uint8"):
        resolve_paged_attention(
            device=DEVICE,
            kv_dtype=torch.uint8,
            kv_layout=cfg.pop("kv_layout", "NHD"),
            kv_input_form="page_indices",  # legal at every page size
            backend="fa2",
            **cfg,
        )


def test_batch_prefill_lazy_stride_router_nvfp4():
    _assert_nvfp4_rejected(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.float16,
        page_size=16,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_with_paged_kv_cache_head_dim_512(
    causal,
    pos_encoding_mode,
):
    head_dim = 512
    _skip_if_head_dim_unsupported(head_dim)

    batch_size = 2
    kv_len = 97
    qo_len = 17
    page_size = 16
    num_kv_heads = 4
    num_qo_heads = 4
    kv_layout = "NHD"
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
    )
    if pos_encoding_mode != "NONE":
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            PagedAttention(DEVICE).plan(
                md, **plan_kw, pos_encoding_mode=pos_encoding_mode
            )
        return

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    kv_data = torch.randn(*kv_shape, dtype=torch.float32, device=DEVICE).half()
    kv_cache = (kv_data[:, 0], kv_data[:, 1])

    attn = PagedAttention(DEVICE)
    attn.plan(md, **plan_kw, lse_mode="base2", backend="fa2")
    assert attn.backend == "fa2"
    o, lse = attn.run(q, kv_cache)

    o_buffer, lse_buffer = torch.empty_like(o), torch.empty_like(lse)
    attn.run(q, kv_cache, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)

    o_ref = _single_prefill_reference(
        q, *kv_cache, qo_len, kv_len, causal=causal, backend="fa2"
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)

    _assert_oracle(o, lse, q, *kv_cache, md, causal, kv_layout)


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
):
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
    )
    if pos_encoding_mode != "NONE":
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            PagedAttention(DEVICE).plan(
                md, **plan_kw, pos_encoding_mode=pos_encoding_mode
            )
        return

    torch.manual_seed(0)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]  # NHD
    kv_data_fp32 = [
        torch.randn(*kv_shape, dtype=torch.float32, device=DEVICE) for _ in range(2)
    ]
    kv_cache = tuple(kv_data_fp32[i].half() for i in range(2))

    if use_cuda_graph:
        warmup_md = _uniform_metadata(
            batch_size, qo_len, min(qo_len, kv_len), page_size, num_pages_per_seq
        )
        o, lse = _run_cuda_graph(q, kv_cache, md, warmup_md, **plan_kw)
    else:
        attn = PagedAttention(DEVICE)
        attn.plan(md, **plan_kw, lse_mode="base2", backend="fa2")
        assert attn.backend == "fa2"
        o, lse = attn.run(q, kv_cache)

    o_ref = _single_prefill_reference(
        q, *kv_cache, qo_len, kv_len, causal=causal, logits_soft_cap=logits_soft_cap
    )
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)

    _assert_oracle(o, lse, q, *kv_cache, md, causal, kv_layout)


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
):
    if qo_len > kv_len:
        pytest.skip("qo_len > kv_len is not supported for custom mask test")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_layout=kv_layout,
        lse_mode="base2",
        backend="fa2",
    )
    if pos_encoding_mode != "NONE":
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            PagedAttention(DEVICE).plan(
                md, **plan_kw, pos_encoding_mode=pos_encoding_mode
            )
        return

    torch.manual_seed(0)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]  # NHD
    kv_data = torch.randn(*kv_shape, dtype=torch.float16, device=DEVICE)
    kv_cache = (kv_data[:, 0], kv_data[:, 1])
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device=DEVICE),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)

    # use custom mask (non-causal: the unified mask is ANDed into the envelope)
    attn = PagedAttention(DEVICE)
    attn.plan(md, **plan_kw, causal=False, custom_mask=custom_mask)
    assert attn.backend == "fa2"
    o_custom, lse_custom = attn.run(q, kv_cache)

    # use causal
    attn.plan(md, **plan_kw, causal=True)
    assert attn.backend == "fa2"
    o_causal, lse_causal = attn.run(q, kv_cache)
    torch.testing.assert_close(o_custom, o_causal, rtol=1e-3, atol=1e-3)

    _assert_oracle(
        o_custom,
        lse_custom,
        q,
        *kv_cache,
        md,
        False,
        kv_layout,
        custom_mask=custom_mask,
    )
    _assert_oracle(o_causal, lse_causal, q, *kv_cache, md, True, kv_layout)


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
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    # plan() has no multi-item scoring fields (nor fused RoPE)
    with pytest.raises(TypeError, match="prefix_len_ptr"):
        PagedAttention(DEVICE).plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_layout=kv_layout,
            causal=causal,
            logits_soft_cap=logits_soft_cap or None,
            lse_mode="base2" if return_lse else "none",
            backend="fa2",
            prefix_len_ptr=torch.tensor(prefix_len_ptr).to(torch.uint32).to(DEVICE),
            token_pos_in_items_ptr=torch.tensor(token_pos_in_items_ptr)
            .to(torch.uint16)
            .to(DEVICE),
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=torch.tensor(max_item_len_ptr).to(torch.uint16).to(DEVICE),
            pos_encoding_mode=pos_encoding_mode,
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
    _assert_nvfp4_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=q_dtype,
        page_size=page_size,
        causal=causal,
    )


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
def test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views(kv_layout):
    _assert_nvfp4_rejected(
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.float16,
        page_size=16,
        kv_layout=kv_layout,
    )


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
    _skip_if_head_dim_unsupported(head_dim_qk)
    if get_compute_capability(DEVICE)[0] < 10:
        pytest.skip(
            "asymmetric NVFP4 KV prefill uses the NVFP4 KV quantization kernel, "
            "which requires SM100 or newer"
        )
    _assert_nvfp4_rejected(
        num_qo_heads=2 * num_kv_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        q_dtype=torch.bfloat16,
        page_size=page_size,
        causal=causal,
    )


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
def test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256(kv_dtype):
    """The legacy test pins fa2's CTA_TILE_Q pick at (448, 256), a head-dim
    pair the unified API does not declare."""
    head_dim_qk = 448
    head_dim_vo = 256
    _skip_if_head_dim_unsupported(head_dim_qk)
    if kv_dtype.itemsize == 1 and get_compute_capability(DEVICE)[0] < 10:
        pytest.skip("FP8 KV with head_dim > 256 requires SM100 or newer")
    props = torch.cuda.get_device_properties(0)
    if getattr(props, "shared_memory_per_block_optin", None) is None:
        pytest.skip("torch does not expose shared_memory_per_block_optin")

    with pytest.raises(ValueError, match=r"unsupported head dims \(448, 256\)"):
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


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("qo_len", [17, 65])
def test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides(kv_layout, qo_len):
    """D512 fp16 K and V pools as views of differently padded parents: V rows
    must be addressed with V's strides."""
    head_dim = 512
    _skip_if_head_dim_unsupported(head_dim)

    torch.manual_seed(42)
    batch_size = 2
    kv_len = 97
    page_size = 16
    num_kv_heads = 2
    num_qo_heads = 2  # group_size 1: avg_packed_qo_len == qo_len
    causal = True

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    def padded_pool(num_padding_heads):
        if kv_layout == "NHD":
            parent = torch.randn(
                total_num_pages,
                page_size,
                num_kv_heads + num_padding_heads,
                head_dim,
                device=DEVICE,
                dtype=torch.float16,
            )
            return parent[:, :, :num_kv_heads, :]
        parent = torch.randn(
            total_num_pages,
            num_kv_heads + num_padding_heads,
            page_size,
            head_dim,
            device=DEVICE,
            dtype=torch.float16,
        )
        return parent[:, :num_kv_heads, :, :]

    k = padded_pool(1)
    v = padded_pool(3)
    assert k.shape == v.shape
    assert not k.is_contiguous() and not v.is_contiguous()
    assert k.stride() != v.stride()

    md = _uniform_metadata(batch_size, qo_len, kv_len, page_size, num_pages_per_seq)
    attn = PagedAttention(DEVICE)
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=causal,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, lse = attn.run(q, (k, v))
    assert o.shape == (batch_size * qo_len, num_qo_heads, head_dim)

    # legacy: exact float32 reference on the logical (view) K/V values
    sm_scale = head_dim**-0.5
    perm = (0, 1, 2, 3) if kv_layout == "NHD" else (0, 2, 1, 3)
    for i in range(batch_size):
        pages = slice(i * num_pages_per_seq, (i + 1) * num_pages_per_seq)
        qi = q[i * qo_len : (i + 1) * qo_len].float()
        ki = k[pages].permute(*perm).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        vi = v[pages].permute(*perm).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        logits = torch.einsum("qhd,khd->hqk", qi, ki.float()) * sm_scale
        qpos = torch.arange(qo_len, device=DEVICE).unsqueeze(1)
        kpos = torch.arange(kv_len, device=DEVICE).unsqueeze(0)
        allowed = kpos <= qpos + (kv_len - qo_len)
        logits = logits.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        o_ref_i = torch.einsum(
            "hqk,khd->qhd", torch.softmax(logits, dim=-1), vi.float()
        )
        o_i = o[i * qo_len : (i + 1) * qo_len].float()
        torch.testing.assert_close(o_i, o_ref_i, rtol=2e-3, atol=2e-3)

    _assert_oracle(o, lse, q, k, v, md, causal, kv_layout)


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head():
    _skip_if_head_dim_unsupported(512)
    _assert_nvfp4_rejected(
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=512,
        q_dtype=torch.float16,
        page_size=16,
        causal=False,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16():
    _skip_if_head_dim_unsupported(512)
    _assert_nvfp4_rejected(
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=512,
        q_dtype=torch.bfloat16,
        page_size=16,
        causal=False,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head():
    _skip_if_head_dim_unsupported(512)
    # fused ROPE_LLAMA is not expressible either (no plan() argument)
    _assert_nvfp4_rejected(
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=512,
        q_dtype=torch.float16,
        page_size=16,
        causal=False,
    )


def test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16():
    _skip_if_head_dim_unsupported(512)
    _assert_nvfp4_rejected(
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim_qk=512,
        q_dtype=torch.bfloat16,
        page_size=16,
        causal=False,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_fully_masked_rows(dtype):
    """Legacy: causal q_len 34 > kv_len 1 leaves 33 fully masked rows (out 0,
    LSE -inf).  The unified causal envelope rejects q_len > kv_len."""
    qo_len, kv_len = 34, 1
    num_qo_heads, num_kv_heads, head_dim = 32, 8, 128
    page_size = 1
    md = PagedAttentionMetadata.csr(
        torch.tensor([0, qo_len], dtype=torch.int32, device=DEVICE),
        torch.tensor([kv_len], dtype=torch.int32, device=DEVICE),
        torch.tensor([1], dtype=torch.int32, device=DEVICE),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )
    with pytest.raises(ValueError, match="causal masking requires q_len_i <= kv_len_i"):
        PagedAttention(DEVICE).plan(
            md,
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_paged_prefill_split_kv_empty_chunk(dtype):
    """Multi-token causal prefill (q 2, kv 129) where a split-KV chunk is
    empty for the first token: the merge must not produce NaN."""
    bs, qo_len, kv_len = 1, 2, 129
    num_qo_heads, num_kv_heads, head_dim = 8, 2, 128
    page_size = 16
    pages_per = (kv_len + page_size - 1) // page_size
    q = (
        torch.randn(bs * qo_len, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
        / 10
    )
    kv_data = (
        torch.randn(
            pages_per, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=DEVICE
        )
        / 10
    )
    kv_cache = (kv_data[:, 0], kv_data[:, 1])

    md = _uniform_metadata(bs, qo_len, kv_len, page_size, pages_per)
    attn = PagedAttention(DEVICE)
    attn.plan(
        md,
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
    o, lse = attn.run(q, kv_cache)

    k = kv_data[:, 0].reshape(-1, num_kv_heads, head_dim)
    v = kv_data[:, 1].reshape(-1, num_kv_heads, head_dim)
    o_ref, lse_ref = ref_single_prefill(q, k[:kv_len], v[:kv_len], causal=True)
    assert not o.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)

    _assert_oracle(o, lse, q, *kv_cache, md, True, "NHD")
