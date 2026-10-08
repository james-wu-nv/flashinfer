"""Legacy -> unified: tests/attention/test_sm120_fmha_paged.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane collects it at
1/5 sampling and every row skips on the compute-capability 12.0 gate, so the
SM120 fp8 paged FMHA never executed in PR CI (reports/unified-prefill-round4-
20260918/ci-status.md).  On B200 the legacy file skips entirely (16 skipped).

The SM120 FP8 paged FMHA (``sm120_fmha_fp8_paged_prefill``, cute-dsl, SM120
only) is not a unified backend and its fixtures are fp8 q / K / V with an fp16
output.  What the unified API can express of each legacy row is its SHAPE
workload: the same (B, Sq, Skv, Hq, Hkv, D, page_size, mask) with the legacy
integer-valued fixture (``randint(-2, 3)``, exact in bf16 as in e4m3) run as a
bf16 q / K / V twin on every unified backend that resolves (D64 / D256: fa2
only; D128: fa2, trtllm-gen, cake, cuDNN; D32: nobody) and on ``auto``
(recorded), against the legacy ``_ref_paged_fmha_single`` reference (fp32,
bottom-right causal, the legacy 0.2 / 0.2 budget) and the fp32 oracle.  Each
row also asserts the rejections of the axes the unified API lacks: fp8 q
(EXPECT_FP8_Q), an output dtype independent of q (EXPECT_OUTPUT_DTYPE) and
head_dim 32 (EXPECT_HEAD_DIM_32: no backend declares (32, 32)).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- fp8 q with fp16 output: the whole legacy numeric regime (fp8 e4m3 inputs,
  fp16 output, 0.2 tolerance) is outside the unified API (EXPECT_FP8_Q,
  EXPECT_OUTPUT_DTYPE); the bf16 twin covers layout, GQA, causal / non-causal,
  variable KV lengths and packed varlen q.
- head_dim 32: capability-excluded everywhere (EXPECT_HEAD_DIM_32).
- test_sm120_paged_compile_cache: a cute-dsl compile-cache contract of the
  SM120 kernel (native-only); the unified layer has no compile cache of its
  own (backend JIT modules are functools-cached inside each backend).
"""

import inspect
import math

import pytest
import torch

from flashinfer.experimental.paged_attention import CAPABILITIES
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_HEAD_DIM_32,
    EXPECT_OUTPUT_DTYPE,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    fp8_q_rejected,
    gated,
    oracle,
    output_dtype_knob_present,
    run_on_backends,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_sm120_fmha_paged.py"


LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_uniform_q",
        ["test_sm120_paged_uniform_q"],
        "partial",
        "same grid (5 shapes: D32 / D128 MHA / GQA 4:1 / D64 / D256 three-stage KV "
        "ring x causal); the legacy fp8 fixture's bf16 twin (same randint(-2, 3) "
        "values, exact in both formats; per-row seed, the legacy is unseeded) on "
        "every resolving unified backend and auto (recorded) vs the legacy "
        "_ref_paged_fmha_single reference (0.2 / 0.2) and the oracle; fp8 q / fp16 "
        "output assert EXPECT_FP8_Q / EXPECT_OUTPUT_DTYPE; the D32 row asserts "
        "EXPECT_HEAD_DIM_32 (no backend declares (32, 32)) and runs nothing",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_variable_kv_lengths",
        ["test_sm120_paged_variable_kv_lengths"],
        "partial",
        "same fixture (B2, Sq 128, Skv 128, H4:4, D128, page 64, kv lengths (64, "
        "128), non-causal); the legacy fp8 fixture's bf16 twin (same randint(-2, 3) "
        "values, exact in both formats; per-row seed, the legacy is unseeded) on "
        "every resolving unified backend and auto (recorded) vs the legacy "
        "_ref_paged_fmha_single reference (0.2 / 0.2) and the oracle; fp8 q / fp16 "
        "output assert EXPECT_FP8_Q / EXPECT_OUTPUT_DTYPE",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_varlen_q",
        ["test_sm120_paged_varlen_q"],
        "partial",
        "same grid (MHA / GQA 4:1 x causal; packed q (64, 96), kv 128, page 64); the "
        "legacy fp8 fixture's bf16 twin (same randint(-2, 3) values, exact in both "
        "formats; per-row seed, the legacy is unseeded) on every resolving unified "
        "backend and auto (recorded) vs the legacy _ref_paged_fmha_single reference "
        "(0.2 / 0.2) and the oracle; fp8 q / fp16 output assert EXPECT_FP8_Q / "
        "EXPECT_OUTPUT_DTYPE",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_compile_cache",
        ["test_sm120_fmha_is_not_a_unified_backend"],
        "native-only",
        "compile-cache hit accounting of compile_sm120_fmha_fp8_paged_kernel: a "
        "cute-dsl contract of the SM120 kernel; the unified layer has no compile "
        "cache of its own",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def test_sm120_fmha_is_not_a_unified_backend():
    """The anchor of the native-only row: no SM120 cute-dsl FMHA backend in
    the capability table (cc 12 is served by fa2 and cuDNN), no compile-cache
    surface on PagedAttention."""
    assert not any("sm120" in name or "cute" in name for name in CAPABILITIES)
    assert {n for n, c in CAPABILITIES.items() if 12 in c.cc_majors} == {"fa2", "cudnn"}
    assert not any(
        hasattr(PagedAttention, attr) for attr in ("cache_info", "cache_clear")
    )


# ---------------------------------------------------------------------------
# the legacy fixture (bf16 twin of the fp8 randint(-2, 3) tensors)
# ---------------------------------------------------------------------------


def _make_vals(shape, dtype=torch.bfloat16):
    return torch.randint(-2, 3, shape, dtype=torch.float32, device=DEVICE).to(dtype)


def _make_paged_kv(k_dense, v_dense, page_size):
    """The legacy conversion: one combined HND cache whose K / V plane views
    share one physical page-id table."""
    B, Skv, Hkv, D = k_dense.shape
    assert Skv % page_size == 0
    pages_per_seq = Skv // page_size
    kv_pool = torch.empty(
        B * pages_per_seq, 2, Hkv, page_size, D, dtype=k_dense.dtype, device=DEVICE
    )
    k_pages = k_dense.reshape(B, pages_per_seq, page_size, Hkv, D).permute(
        0, 1, 3, 2, 4
    )
    v_pages = v_dense.reshape(B, pages_per_seq, page_size, Hkv, D).permute(
        0, 1, 3, 2, 4
    )
    kv_pool[:, 0].copy_(k_pages.reshape_as(kv_pool[:, 0]))
    kv_pool[:, 1].copy_(v_pages.reshape_as(kv_pool[:, 1]))
    k_pool, v_pool = kv_pool.unbind(dim=1)
    block_tables = torch.arange(
        B * pages_per_seq, dtype=torch.int32, device=DEVICE
    ).reshape(B, pages_per_seq)
    return k_pool, v_pool, block_tables


def _ref_paged_fmha_single(q_b, k_b, v_b, sm_scale, is_causal, kv_len=None):
    """The legacy fp32 reference for one batch item (bottom-right causal)."""
    sq, Hq, D = q_b.shape
    skv = k_b.shape[0]
    if kv_len is None:
        kv_len = skv
    Hkv = k_b.shape[1]
    q_f = q_b.float().permute(1, 0, 2)
    k_f = k_b.float().permute(1, 0, 2)
    v_f = v_b.float().permute(1, 0, 2)
    if Hq != Hkv:
        k_f = k_f.repeat_interleave(Hq // Hkv, dim=0)
        v_f = v_f.repeat_interleave(Hq // Hkv, dim=0)
    scores = torch.einsum("hqd,hkd->hqk", q_f, k_f) * sm_scale
    if kv_len < skv:
        scores[:, :, kv_len:] = float("-inf")
    if is_causal:
        q_offset = kv_len - sq
        q_idx = (torch.arange(sq, device=DEVICE) + q_offset).view(-1, 1)
        k_idx = torch.arange(kv_len, device=DEVICE).view(1, -1)
        causal_mask = k_idx > q_idx
        scores[:, :sq, :kv_len] = scores[:, :sq, :kv_len].masked_fill(
            causal_mask, float("-inf")
        )
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->hqd", attn, v_f).permute(1, 0, 2)


def _tol():
    return dict(atol=0.2, rtol=0.2)


def _assert_fp8_axes_rejected(
    *, head_dim, page_size, causal, num_qo_heads, num_kv_heads
):
    """The legacy regime: fp8 e4m3 q / K / V, fp16 output."""
    assert output_dtype_knob_present() == EXPECT_OUTPUT_DTYPE
    res = fp8_q_rejected(
        backend="fa2",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        kv_layout="HND",
        causal=causal,
    )
    if res is not None:
        pytest.fail("EXPECT_FP8_Q flipped: run the legacy fp8 fixture here")


def _head_dim_32_rejected(*, num_qo_heads, num_kv_heads, page_size, causal) -> bool:
    """(32, 32) is declared by no backend (EXPECT_HEAD_DIM_32)."""
    res = gated(
        EXPECT_HEAD_DIM_32,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=32,
            q_dtype=torch.bfloat16,
            page_size=page_size,
            kv_layout="HND",
            causal=causal,
            need_lse=True,
        ),
        match="unsupported head dims \\(32, 32\\)",
    )
    if res is not None:
        pytest.fail("EXPECT_HEAD_DIM_32 flipped: run the D32 twin here")
    return True


def _run_twin(
    q,
    k_dense,
    v_dense,
    *,
    cu_seqlens_q,
    seqlens_kv,
    page_size,
    is_causal,
    record_property,
):
    """The bf16 twin on every resolving backend + auto vs the legacy
    reference and the oracle.  ``q`` is packed (total_q, Hq, D)."""
    B, Skv, Hkv, D = k_dense.shape
    Hq = q.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    k_pool, v_pool, block_tables = _make_paged_kv(k_dense, v_dense, page_size)
    md = dense_metadata(cu_seqlens_q, seqlens_kv, block_tables, page_size)
    results = run_on_backends(
        md,
        q,
        (k_pool, v_pool),
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        q_dtype=q.dtype,
        kv_layout="HND",
        causal=is_causal,
        sm_scale=sm_scale,
        record_property=record_property,
    )
    refs = []
    for b in range(B):
        s, e = int(cu_seqlens_q[b]), int(cu_seqlens_q[b + 1])
        refs.append(
            _ref_paged_fmha_single(
                q[s:e], k_dense[b], v_dense[b], sm_scale, is_causal, int(seqlens_kv[b])
            )
        )
    ref = torch.cat(refs)
    o_out, o_lse = oracle(
        md, q, k_pool, v_pool, causal=is_causal, kv_layout="HND", sm_scale=sm_scale
    )
    for _name, _served, out, lse in results:
        torch.testing.assert_close(out.float(), ref, **_tol())  # legacy budget
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    return results


# ---------------------------------------------------------------------------
# Uniform packed Q + paged KV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B,Sq,Skv,Hq,Hkv,D,page_size",
    [
        (1, 128, 128, 4, 4, 32, 64),  # head_dim=32
        (1, 128, 128, 8, 8, 128, 64),  # MHA, 2 pages/seq
        (2, 64, 128, 8, 2, 128, 64),  # GQA 4:1
        (1, 128, 128, 4, 4, 64, 64),  # head_dim=64
        (1, 129, 384, 2, 1, 256, 64),  # D=256 three-stage KV ring
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_uniform_q(
    B, Sq, Skv, Hq, Hkv, D, page_size, is_causal, record_property
):
    """Uniform Q packed as (B * Sq, Hq, D) + paged KV (bf16 twin)."""
    if D == 32:
        _head_dim_32_rejected(
            num_qo_heads=Hq, num_kv_heads=Hkv, page_size=page_size, causal=is_causal
        )
        return
    _assert_fp8_axes_rejected(
        head_dim=D,
        page_size=page_size,
        causal=is_causal,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
    )
    torch.manual_seed(B * 1000003 + Sq * 10007 + Skv * 101 + Hq * 13 + D)
    q_dense = _make_vals((B, Sq, Hq, D))
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_vals((B, Skv, Hkv, D))
    v = _make_vals((B, Skv, Hkv, D))
    seqlens_kv = torch.full((B,), Skv, dtype=torch.int32)
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32) * Sq
    _run_twin(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        seqlens_kv=seqlens_kv,
        page_size=page_size,
        is_causal=is_causal,
        record_property=record_property,
    )


# ---------------------------------------------------------------------------
# Paged KV with variable KV lengths (seqlens_kv < Skv)
# ---------------------------------------------------------------------------


def test_sm120_paged_variable_kv_lengths(record_property):
    """Paged KV with different actual KV lengths per batch item (bf16 twin)."""
    B, Sq, Skv, Hq, Hkv, D, page_size = 2, 128, 128, 4, 4, 128, 64
    _assert_fp8_axes_rejected(
        head_dim=D, page_size=page_size, causal=False, num_qo_heads=Hq, num_kv_heads=Hkv
    )
    torch.manual_seed(20260918)
    q_dense = _make_vals((B, Sq, Hq, D))
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_vals((B, Skv, Hkv, D))
    v = _make_vals((B, Skv, Hkv, D))
    seqlens_kv = torch.tensor([64, 128], dtype=torch.int32)
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32) * Sq
    # non-causal (the legacy row): q 128 over kv 64 is fine without the envelope
    _run_twin(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        seqlens_kv=seqlens_kv,
        page_size=page_size,
        is_causal=False,
        record_property=record_property,
    )


# ---------------------------------------------------------------------------
# Packed-varlen Q + paged KV
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "Hq,Hkv,D,page_size",
    [
        (8, 8, 128, 64),  # MHA
        (8, 2, 128, 64),  # GQA 4:1
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_varlen_q(Hq, Hkv, D, page_size, is_causal, record_property):
    """Packed-varlen Q (total_q, Hq, D) + paged KV (bf16 twin)."""
    _assert_fp8_axes_rejected(
        head_dim=D,
        page_size=page_size,
        causal=is_causal,
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
    )
    q_lens = [64, 96]
    kv_len = 128
    B = len(q_lens)
    torch.manual_seed(20260919 + Hkv)
    q_packed = _make_vals((sum(q_lens), Hq, D))
    k_dense = _make_vals((B, kv_len, Hkv, D))
    v_dense = _make_vals((B, kv_len, Hkv, D))
    cu_seqlens_q = torch.tensor(
        [0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32
    )
    seqlens_kv = torch.full((B,), kv_len, dtype=torch.int32)
    _run_twin(
        q_packed,
        k_dense,
        v_dense,
        cu_seqlens_q=cu_seqlens_q,
        seqlens_kv=seqlens_kv,
        page_size=page_size,
        is_causal=is_causal,
        record_property=record_property,
    )


def test_sm120_paged_compile_cache():
    """native-only (see the anchor test): the unified API has no compile
    cache surface; this row records that repeated plans of one configuration
    are legal and idempotent on fa2."""
    assert not any(
        "cache" in name for name in inspect.signature(PagedAttention.plan).parameters
    )
    B, Sq, Skv, Hq, Hkv, D, page_size = 1, 128, 128, 4, 4, 128, 64
    torch.manual_seed(7)
    q = _make_vals((B * Sq, Hq, D))
    k = _make_vals((B, Skv, Hkv, D))
    v = _make_vals((B, Skv, Hkv, D))
    k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
    md = dense_metadata(
        torch.arange(B + 1, dtype=torch.int32) * Sq,
        torch.full((B,), Skv, dtype=torch.int32),
        block_tables,
        page_size,
    )
    attn = PagedAttention(torch.device(DEVICE))
    outs = []
    for _ in range(3):
        attn.plan(
            md,
            num_qo_heads=Hq,
            num_kv_heads=Hkv,
            head_dim_qk=D,
            q_dtype=torch.bfloat16,
            causal=True,
            backend="fa2",
        )
        out, _ = attn.run(q, (k_pool, v_pool))
        outs.append(out)
    for out in outs[1:]:
        torch.testing.assert_close(out, outs[0], rtol=0, atol=0)
