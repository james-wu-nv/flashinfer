"""Legacy -> unified: tests/attention/test_trtllm_gen_attention_prefill.py

The paged-context tests run the legacy fixture (``_test_trtllm_batch_prefill``:
the ``test_trtllm_gen_attention_decode`` helpers under seed 0) through
``PagedAttention`` pinned to trtllm-gen, the backend the legacy test calls
(``trtllm_batch_context_with_kv_cache``).  Same grids, so the node ids equal
the legacy ids.  Each bf16 / fp16 row checks the legacy reference (the paged
prefill wrapper, or ``sink_attention_unified`` for sink rows) at the legacy
tolerance, and the output and LSE against the fp32 paged-attention oracle.

Rows outside the trtllm-gen envelope assert the rejection and stop: fp8 q
(which also covers the fp8 / nvfp4 output and the nvfp4 KV triples), head dims
256 and 512, skip-softmax (no plan / run knob), the spcompress cubins (no
knob) and independent K / V page tables (the metadata takes one 2-D table).

The five ``test_trtllm_gen_prefill*`` tests call
``trtllm_ragged_attention_deepseek`` (ragged MLA prefill) and are out of scope.
"""

from types import SimpleNamespace

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
    create_kv_cache,
    create_page_table,
    create_query_tensor,
    create_workspace_buffers,
    flatten_paged_kv,
    flip_coin,
    generate_cumsum_lens,
    generate_seq_lens_prefill,
    get_last_page_len,
    make_query_non_contiguous,
    prepare_paged_kv_for_kernel,
)
from tests.test_helpers.sink_attention_reference import sink_attention_unified
from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_trtllm_gen_attention_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill",
        ["test_trtllm_batch_prefill"],
        "partial",
        "same grid, seed and tensors on trtllm-gen; bf16 / fp16 D128 rows check the "
        "legacy reference at 1e-2 (LSE 1e-3) plus the fp32 oracle; fp8 q (all fp8 / "
        "nvfp4 triples), D256, skip-softmax and independent K/V tables assert the "
        "rejection; the legacy wrapper-vs-function cross-check is not mirrored",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_lse_contract",
        ["test_trtllm_batch_prefill_lse_contract"],
        "partial",
        "same fixture; return_lse / provide_lse map to lse_mode base2 and a caller "
        "lse= buffer (returned as is, checked vs the legacy LSE and the oracle); the "
        "workspace guard-region check is a native-only contract",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_bs1",
        ["test_trtllm_batch_prefill_bs1"],
        "partial",
        "same grid (q 8192 over kv 16384); D128 rows run on trtllm-gen vs the legacy "
        "reference and a query-chunked fp32 oracle; D256, skip-softmax and "
        "independent K/V tables assert the rejection",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_cubin_variants",
        ["test_trtllm_batch_prefill_cubin_variants"],
        "unsupported-by-design",
        "fp8 QKV through the SM107-only spcompress cubins: fp8 q is rejected at "
        "resolve and there is no spcompress knob (asserted on every SM10x; the "
        "legacy skips outside SM107)",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_dynamic_page_size_gqa",
        ["test_trtllm_batch_prefill_dynamic_page_size_gqa"],
        "partial",
        "pages 128-1024 on trtllm-gen vs the legacy reference and the oracle; "
        "independent K/V tables assert the rejection",
    ),
    (
        "tests/attention/test_trtllm_gen_attention_prefill.py::test_trtllm_batch_prefill_head_dim_512",
        ["test_trtllm_batch_prefill_head_dim_512"],
        "unsupported-by-design",
        "trtllm-gen declares no (512, 512); every id asserts the resolve rejection "
        "(fp8 q rows: the q dtype rejection)",
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
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _skip_unless_sm10x():
    if get_compute_capability(DEVICE)[0] != 10:
        pytest.skip("These tests are only guaranteed to work on SM100 and SM103 GPUs.")


def _legacy_fixture(
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
    """``_test_trtllm_batch_prefill``'s bf16 / fp16 tensors in the legacy RNG
    order (seed 0; q, stacked (pages, 2, ...) KV pool, page ids, sink), plus
    the dense metadata over the legacy page table."""
    torch.manual_seed(0)
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)
    q, _, _ = create_query_tensor(q_lens, num_qo_heads, head_dim, dtype_name)
    q_indptr = generate_cumsum_lens(q_lens)
    kv_cache, _, _, _, _ = create_kv_cache(
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

    qo_indptr_cpu = q_indptr.cpu()
    md = PagedAttentionMetadata.dense(
        q_indptr,
        seq_lens.to(DEVICE),
        page_table,
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(seq_lens.max()),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=seq_lens,
    )
    return SimpleNamespace(
        kv_layout=kv_layout,
        batch_size=batch_size,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=DTYPE_MAP[dtype_name],
        q=q,
        q_lens=q_lens,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_cache=kv_cache,
        page_table=page_table,
        all_page_ids=all_page_ids,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sink=sink,
        sm_scale=float(1.0 / (head_dim**0.5)),
        md=md,
    )


def _legacy_reference(fx, *, causal, enable_sink):
    """The legacy reference: the paged prefill wrapper (output, LSE), or
    ``sink_attention_unified`` on the flattened pool for sink rows (no LSE)."""
    if enable_sink:
        k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
            fx.kv_cache,
            fx.page_table,
            fx.seq_lens.to(DEVICE),
            fx.page_size,
            fx.kv_last_page_len,
            fx.kv_layout,
        )
        out = sink_attention_unified(
            fx.q,
            k_flat,
            v_flat,
            fx.sink,
            -1,
            causal,
            fx.sm_scale,
            mode="varlen",
            batch_size=fx.batch_size,
            qo_indptr=fx.q_indptr,
            kv_indptr=kv_indptr_tokens,
        )
        return out, None
    _, workspace_buffer_ref = create_workspace_buffers(DEVICE)
    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, fx.kv_layout
    )
    wrapper_ref.plan(
        qo_indptr=fx.q_indptr,
        paged_kv_indptr=fx.kv_indptr,
        paged_kv_indices=fx.all_page_ids,
        paged_kv_last_page_len=fx.kv_last_page_len.to(DEVICE),
        num_qo_heads=fx.num_qo_heads,
        num_kv_heads=fx.num_kv_heads,
        head_dim_qk=fx.head_dim,
        page_size=fx.page_size,
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        q_data_type=fx.dtype,
        kv_data_type=fx.dtype,
        window_left=-1,
    )
    return wrapper_ref.run(fx.q, fx.kv_cache, return_lse=True)


def _assert_legacy_close(out, ref):
    """The legacy output assertion (1e-2, mismatch rate 1e-7)."""
    assert_close_with_mismatch_tolerance(
        out.float(),
        ref.float(),
        rtol=1e-2,
        atol=1e-2,
        max_mismatched_elements=int(1e-7 * out.numel()),
    )


def _assert_independent_kv_tables_rejected(fx):
    """The legacy uses_shared_paged_kv_idx=False table is [B, 2, M] (K at page
    2p, V at 2p + 1); the metadata takes one 2-D table shared by K and V."""
    _, table_kv, _ = prepare_paged_kv_for_kernel(fx.kv_cache, fx.page_table, False)
    with pytest.raises(ValueError, match="block_tables must be 2-D"):
        PagedAttentionMetadata.dense(
            fx.q_indptr,
            fx.seq_lens.to(DEVICE),
            table_kv,
            page_size=fx.page_size,
            max_q_len=fx.md.max_q_len,
            max_kv_len=fx.md.max_kv_len,
        )


def _no_skip_softmax_knob():
    """plan() and run() have no skip-softmax threshold: passing it is a TypeError."""
    attn = PagedAttention(torch.device("cuda"))
    for fn in (attn.plan, attn.run):
        with pytest.raises(TypeError, match="skip_softmax_threshold_scale_factor"):
            fn(None, None, skip_softmax_threshold_scale_factor=1.0)


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
    # the legacy skips
    _skip_unless_sm10x()
    if not causal and window_left >= 0:
        pytest.skip("Non-causal paged trtllm-gen tests only cover dense attention")
    if skips_softmax and q_dtype != kv_dtype:
        pytest.skip(
            "skips_softmax does not currently support Q and Kv types being different"
        )
    if kv_dtype == "nvfp4":
        if q_dtype != "fp8":
            pytest.skip("NVFP4 KV cache requires FP8 query")
        if o_dtype != "fp8":
            pytest.skip("NVFP4 KV cache only supports FP8 output")

    num_qo_heads = num_kv_heads * head_grp_size
    resolve_kw = dict(
        device=DEVICE,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        backend="trtllm-gen",
    )
    if q_dtype == "fp8":
        # also the fp8 / nvfp4 output and nvfp4 KV triples: all take fp8 q
        with pytest.raises(ValueError, match="unsupported q dtype"):
            resolve_paged_attention(
                q_dtype=torch.float8_e4m3fn, kv_dtype=torch.float8_e4m3fn, **resolve_kw
            )
        return
    if head_dim == 256:
        with pytest.raises(ValueError, match=r"unsupported head dims \(256, 256\)"):
            resolve_paged_attention(q_dtype=DTYPE_MAP[q_dtype], **resolve_kw)
        return
    if skips_softmax:
        _no_skip_softmax_knob()
        return

    fx = _legacy_fixture(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        head_dim,
    )
    if not uses_shared_paged_kv_idx:
        _assert_independent_kv_tables_rejected(fx)
        return

    attn = PagedAttention(DEVICE)
    attn.plan(
        fx.md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=fx.dtype,
        kv_layout=kv_layout,
        causal=causal,
        lse_mode="base2",
        use_sinks=enable_sink,
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"
    q_input = (
        make_query_non_contiguous(fx.q, num_qo_heads, head_dim)
        if non_contiguous_query
        else fx.q.contiguous()
    )
    # the legacy coin decides whether the caller supplies the output buffer
    out_buf = None
    if flip_coin(batch_size, page_size, num_kv_heads, head_grp_size, o_dtype):
        out_buf = torch.empty_like(fx.q)
    k_cache, v_cache = fx.kv_cache[:, 0], fx.kv_cache[:, 1]
    out, lse = attn.run(
        q_input,
        (k_cache, v_cache),
        out=out_buf,
        sm_scale=fx.sm_scale,
        sinks=fx.sink if enable_sink else None,
    )
    if out_buf is not None:
        assert out.data_ptr() == out_buf.data_ptr()

    # legacy reference at the legacy tolerance
    out_ref, lse_ref = _legacy_reference(fx, causal=causal, enable_sink=enable_sink)
    _assert_legacy_close(out, out_ref)
    if lse_ref is not None:
        torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)

    # fp32 oracle: output and base-2 LSE
    ref_out, ref_lse = reference_paged_prefill(
        fx.q,
        k_cache,
        v_cache,
        fx.qo_indptr_cpu,
        fx.seq_lens,
        fx.page_table,
        page_size,
        causal,
        sm_scale=fx.sm_scale,
        kv_layout=kv_layout,
        sinks=fx.sink if enable_sink else None,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("provide_lse", [False, True])
def test_trtllm_batch_prefill_lse_contract(return_lse, provide_lse):
    _skip_unless_sm10x()
    fx = _legacy_fixture("HND", 2, 16, 2, 2, "fp16", 64, 128, 128)
    want_lse = return_lse or provide_lse
    provided_lse = None
    if provide_lse:
        # NaN-filled so a missed write shows
        provided_lse = torch.full(
            (fx.q.shape[0], fx.num_qo_heads),
            float("nan"),
            device=DEVICE,
            dtype=torch.float32,
        )

    attn = PagedAttention(DEVICE)
    attn.plan(
        fx.md,
        num_qo_heads=fx.num_qo_heads,
        num_kv_heads=fx.num_kv_heads,
        head_dim_qk=fx.head_dim,
        q_dtype=fx.dtype,
        kv_layout="HND",
        causal=True,
        lse_mode="base2" if want_lse else "none",
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"
    k_cache, v_cache = fx.kv_cache[:, 0], fx.kv_cache[:, 1]
    out, lse = attn.run(
        fx.q, (k_cache, v_cache), lse=provided_lse, sm_scale=fx.sm_scale
    )

    out_ref, lse_ref = _legacy_reference(fx, causal=True, enable_sink=False)
    _assert_legacy_close(out, out_ref)
    ref_out, ref_lse = reference_paged_prefill(
        fx.q,
        k_cache,
        v_cache,
        fx.qo_indptr_cpu,
        fx.seq_lens,
        fx.page_table,
        16,
        True,
        sm_scale=fx.sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    if not want_lse:
        assert lse is None
        return
    if provide_lse:
        assert lse.data_ptr() == provided_lse.data_ptr()
    assert lse.dtype == torch.float32
    assert lse.shape == (fx.q.shape[0], fx.num_qo_heads)
    assert torch.isfinite(lse).all(), (
        "trtllm-gen context kernel produced non-finite LSE"
    )
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


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
    _skip_unless_sm10x()
    num_qo_heads = num_kv_heads * head_grp_size
    if head_dim == 256:
        with pytest.raises(ValueError, match=r"unsupported head dims \(256, 256\)"):
            resolve_paged_attention(
                device=DEVICE,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_dtype=DTYPE_MAP[q_dtype],
                page_size=page_size,
                kv_layout=kv_layout,
                backend="trtllm-gen",
            )
        return
    if skips_softmax:
        _no_skip_softmax_knob()
        return

    fx = _legacy_fixture(
        kv_layout,
        batch_size,
        page_size,
        num_kv_heads,
        head_grp_size,
        q_dtype,
        max_q_len,
        max_kv_len,
        head_dim,
    )
    if not uses_shared_paged_kv_idx:
        _assert_independent_kv_tables_rejected(fx)
        return

    attn = PagedAttention(DEVICE)
    attn.plan(
        fx.md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=fx.dtype,
        kv_layout=kv_layout,
        causal=True,
        lse_mode="base2",
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"
    k_cache, v_cache = fx.kv_cache[:, 0], fx.kv_cache[:, 1]
    out, lse = attn.run(fx.q, (k_cache, v_cache), sm_scale=fx.sm_scale)

    out_ref, lse_ref = _legacy_reference(fx, causal=True, enable_sink=False)
    _assert_legacy_close(out, out_ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)

    # fp32 oracle in query chunks: the full 64 x 8192 x 16384 score matrix
    # does not fit.  Rows [a, b) under bottom-right causal masking see the
    # first kv_len - q_len + b keys, exactly as a request of length b - a.
    q_len, kv_len = int(fx.q_lens[0]), int(fx.seq_lens[0])
    ref_out, ref_lse = [], []
    for a in range(0, q_len, 256):
        b = min(a + 256, q_len)
        o, l = reference_paged_prefill(
            fx.q[a:b],
            k_cache,
            v_cache,
            torch.tensor([0, b - a]),
            torch.tensor([kv_len - q_len + b]),
            fx.page_table,
            page_size,
            True,
            sm_scale=fx.sm_scale,
            kv_layout=kv_layout,
        )
        ref_out.append(o)
        ref_lse.append(l)
    torch.testing.assert_close(out.float(), torch.cat(ref_out), **OUT_TOL)
    torch.testing.assert_close(lse, torch.cat(ref_lse), **LSE_TOL)


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
    # The legacy also skips outside SM107 (where the spcompress cubins ship);
    # the unified rejection does not depend on them, so it is asserted on every
    # SM10x.
    _skip_unless_sm10x()
    with pytest.raises(ValueError, match="unsupported q dtype"):
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_kv_heads * head_grp_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=page_size,
            kv_layout=kv_layout,
            window_left=window_left,
            sinks=enable_sink,
            backend="trtllm-gen",
        )
    with pytest.raises(TypeError, match="uses_spcompress"):
        PagedAttention(torch.device("cuda")).plan(None, uses_spcompress=True)


@pytest.mark.parametrize("page_size", [128, 256, 512, 1024])
@pytest.mark.parametrize("uses_shared_paged_kv_idx", [True, False])
def test_trtllm_batch_prefill_dynamic_page_size_gqa(
    page_size: int,
    uses_shared_paged_kv_idx: bool,
) -> None:
    _skip_unless_sm10x()
    fx = _legacy_fixture("HND", 4, page_size, 2, 5, "bf16", 257, 1024, 128)
    if not uses_shared_paged_kv_idx:
        _assert_independent_kv_tables_rejected(fx)
        return

    attn = PagedAttention(DEVICE)
    attn.plan(
        fx.md,
        num_qo_heads=10,
        num_kv_heads=2,
        head_dim_qk=128,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
        backend="trtllm-gen",
    )
    assert attn.backend == "trtllm-gen"
    k_cache, v_cache = fx.kv_cache[:, 0], fx.kv_cache[:, 1]
    out, lse = attn.run(fx.q, (k_cache, v_cache), sm_scale=fx.sm_scale)

    out_ref, lse_ref = _legacy_reference(fx, causal=True, enable_sink=False)
    _assert_legacy_close(out, out_ref)
    torch.testing.assert_close(lse, lse_ref.float(), rtol=1e-3, atol=1e-3)

    ref_out, ref_lse = reference_paged_prefill(
        fx.q,
        k_cache,
        v_cache,
        fx.qo_indptr_cpu,
        fx.seq_lens,
        fx.page_table,
        page_size,
        True,
        sm_scale=fx.sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


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
    _skip_unless_sm10x()
    if q_dtype == "fp8":
        q_torch, kv_torch, match = (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            "unsupported q dtype",
        )
    else:
        q_torch = kv_torch = DTYPE_MAP[q_dtype]
        match = r"unsupported head dims \(512, 512\)"
    with pytest.raises(ValueError, match=match):
        resolve_paged_attention(
            device=DEVICE,
            num_qo_heads=num_kv_heads * head_grp_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=q_torch,
            kv_dtype=kv_torch,
            page_size=page_size,
            kv_layout=kv_layout,
            backend="trtllm-gen",
        )
