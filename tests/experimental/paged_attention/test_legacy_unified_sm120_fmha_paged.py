"""Legacy -> unified: tests/attention/test_sm120_fmha_paged.py

The legacy subject, ``sm120_fmha_fp8_paged_prefill`` (cute-dsl, SM120 only),
is not a unified backend, and the unified API has no fp8 q.  The conversion
runs a SUBSTITUTE on fa2 (which serves SM100 and SM120 alike, so the rows run
where the legacy skips): the legacy fp8 e4m3 K / V caches as they are, and the
legacy fp8 q values cast to fp16 (``randint(-2, 3)``, exact in both formats),
with the legacy fp16 output dtype.  Same grids and RNG order as the legacy
(which is unseeded), so the node ids equal the legacy ids.  Each case checks
the legacy fp32 reference (``_ref_paged_fmha_single``, verbatim) at the legacy
0.2 tolerance and the fp32 paged-attention oracle.

head_dim 32 is not a unified head dim: those rows assert the rejection.  The
compile-cache test is a contract of the SM120 cute-dsl kernel (native-only).
"""

import math

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_sm120_fmha_paged.py"
LEGACY_MAP = [
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_uniform_q",
        ["test_sm120_paged_uniform_q"],
        "partial",
        "substitute: fa2 with the legacy fp8 K / V and fp16 q (the legacy fp8 q values); "
        "legacy reference at 0.2 plus the fp32 oracle; the head_dim 32 rows assert the "
        "rejection",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_variable_kv_lengths",
        ["test_sm120_paged_variable_kv_lengths"],
        "partial",
        "substitute: fa2 with the legacy fp8 K / V and fp16 q (the legacy fp8 q values); "
        "legacy reference at 0.2 plus the fp32 oracle",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_varlen_q",
        ["test_sm120_paged_varlen_q"],
        "partial",
        "substitute: fa2 with the legacy fp8 K / V and fp16 q (the legacy fp8 q values); "
        "legacy reference at 0.2 plus the fp32 oracle",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_compile_cache",
        [],
        "native-only",
        "compile-cache hit accounting of compile_sm120_fmha_fp8_paged_kernel, a "
        "contract of the SM120 cute-dsl kernel",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)


def _make_paged_kv(k_dense, v_dense, page_size):
    """Verbatim from the legacy file: one combined HND cache, returned as its
    K / V plane views plus the shared block table."""
    B, Skv, Hkv, D = k_dense.shape
    assert Skv % page_size == 0, f"Skv={Skv} must be divisible by page_size={page_size}"
    pages_per_seq = Skv // page_size

    kv_pool = torch.empty(
        B * pages_per_seq,
        2,
        Hkv,
        page_size,
        D,
        dtype=k_dense.dtype,
        device=k_dense.device,
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
        B * pages_per_seq, dtype=torch.int32, device="cuda"
    ).reshape(B, pages_per_seq)

    return k_pool.cuda(), v_pool.cuda(), block_tables


def _ref_paged_fmha_single(q_b, k_b, v_b, sm_scale, is_causal, kv_len=None):
    """Verbatim from the legacy file: fp32 reference for one batch item,
    bottom-right causal."""
    sq, Hq, D = q_b.shape
    skv = k_b.shape[0]
    if kv_len is None:
        kv_len = skv
    Hkv = k_b.shape[1]

    q_f = q_b.float().permute(1, 0, 2)  # (Hq, sq, D)
    k_f = k_b.float().permute(1, 0, 2)  # (Hkv, skv, D)
    v_f = v_b.float().permute(1, 0, 2)
    if Hq != Hkv:
        k_f = k_f.repeat_interleave(Hq // Hkv, dim=0)
        v_f = v_f.repeat_interleave(Hq // Hkv, dim=0)

    scores = torch.einsum("hqd,hkd->hqk", q_f, k_f) * sm_scale
    if kv_len < skv:
        scores[:, :, kv_len:] = float("-inf")
    if is_causal:
        q_offset = kv_len - sq
        q_idx = (torch.arange(sq, device=q_b.device) + q_offset).view(-1, 1)
        k_idx = torch.arange(kv_len, device=q_b.device).view(1, -1)
        causal_mask = k_idx > q_idx  # (sq, kv_len)
        scores[:, :sq, :kv_len] = scores[:, :sq, :kv_len].masked_fill(
            causal_mask, float("-inf")
        )
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,hkd->hqd", attn, v_f).permute(1, 0, 2)  # (sq, Hq, D)


def _make_fp8(shape, dtype=torch.float8_e4m3fn):
    return torch.randint(-2, 3, shape, dtype=torch.float32, device="cuda").to(dtype)


def _tol():
    return dict(atol=0.2, rtol=0.2)


def _metadata(cu_seqlens_q, seqlens_kv, block_tables, page_size):
    """The legacy (cu_seqlens_q, seqlens_kv, block_tables) as dense metadata."""
    qo_indptr_cpu, kv_lens_cpu = cu_seqlens_q.cpu(), seqlens_kv.cpu()
    return PagedAttentionMetadata.dense(
        cu_seqlens_q,
        seqlens_kv,
        block_tables,
        page_size=page_size,
        max_q_len=int(qo_indptr_cpu.diff().max()),
        max_kv_len=int(kv_lens_cpu.max()),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


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
def test_sm120_paged_uniform_q(B, Sq, Skv, Hq, Hkv, D, page_size, is_causal):
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    # the legacy fixture, verbatim (same RNG order: q, k, v)
    q_dense = _make_fp8((B, Sq, Hq, D), in_dtype)
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v = _make_fp8((B, Skv, Hkv, D), in_dtype)
    k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
    seqlens_kv = torch.full((B,), Skv, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * Sq

    md = _metadata(cu_seqlens_q, seqlens_kv, block_tables, page_size)
    attn = PagedAttention(torch.device("cuda:0"))
    plan_kwargs = dict(
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        q_dtype=out_dtype,
        kv_dtype=in_dtype,
        kv_layout="HND",
        causal=is_causal,
        backend="fa2",
    )
    if D == 32:
        with pytest.raises(ValueError, match=r"fa2: unsupported head dims \(32, 32\)"):
            attn.plan(md, **plan_kwargs)
        return
    attn.plan(md, **plan_kwargs)
    assert attn.backend == "fa2"
    o, _ = attn.run(q.to(out_dtype), (k_pool, v_pool), sm_scale=sm_scale)
    assert o.dtype == out_dtype

    # legacy reference at the legacy tolerance
    ref = (
        torch.stack(
            [
                _ref_paged_fmha_single(q_dense[b], k[b], v[b], sm_scale, is_causal, Skv)
                for b in range(B)
            ]
        )
        .reshape_as(o)
        .to(out_dtype)
    )
    torch.testing.assert_close(o, ref, **_tol())

    # fp32 oracle
    ref_out, _ = reference_paged_prefill(
        q,
        k_pool,
        v_pool,
        cu_seqlens_q.cpu(),
        seqlens_kv.cpu(),
        block_tables,
        page_size,
        is_causal,
        sm_scale=sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)


def test_sm120_paged_variable_kv_lengths():
    B, Sq, Skv, Hq, Hkv, D, page_size = 2, 128, 128, 4, 4, 128, 64
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    # the legacy fixture, verbatim (same RNG order: q, k, v)
    q_dense = _make_fp8((B, Sq, Hq, D), in_dtype)
    q = q_dense.reshape(B * Sq, Hq, D)
    k = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v = _make_fp8((B, Skv, Hkv, D), in_dtype)
    k_pool, v_pool, block_tables = _make_paged_kv(k, v, page_size)
    seqlens_kv = torch.tensor([64, 128], dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(B + 1, dtype=torch.int32, device="cuda") * Sq

    attn = PagedAttention(torch.device("cuda:0"))
    attn.plan(
        _metadata(cu_seqlens_q, seqlens_kv, block_tables, page_size),
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        q_dtype=out_dtype,
        kv_dtype=in_dtype,
        kv_layout="HND",
        causal=False,
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, _ = attn.run(q.to(out_dtype), (k_pool, v_pool), sm_scale=sm_scale)

    # legacy reference at the legacy tolerance
    ref = (
        torch.stack(
            [
                _ref_paged_fmha_single(
                    q_dense[b], k[b], v[b], sm_scale, False, int(seqlens_kv[b].item())
                )
                for b in range(B)
            ]
        )
        .reshape_as(o)
        .to(out_dtype)
    )
    torch.testing.assert_close(o, ref, **_tol())

    # fp32 oracle
    ref_out, _ = reference_paged_prefill(
        q,
        k_pool,
        v_pool,
        cu_seqlens_q.cpu(),
        seqlens_kv.cpu(),
        block_tables,
        page_size,
        False,
        sm_scale=sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)


@pytest.mark.parametrize(
    "Hq,Hkv,D,page_size",
    [
        (8, 8, 128, 64),  # MHA
        (8, 2, 128, 64),  # GQA 4:1
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_varlen_q(Hq, Hkv, D, page_size, is_causal):
    sm_scale = 1.0 / math.sqrt(D)
    in_dtype, out_dtype = torch.float8_e4m3fn, torch.float16

    # the legacy fixture, verbatim (same RNG order: q, k, v)
    q_lens = [64, 96]
    kv_len = 128
    B = len(q_lens)
    Skv = kv_len
    total_q = sum(q_lens)
    cu_seqlens_q = torch.tensor(
        [0, q_lens[0], q_lens[0] + q_lens[1]], dtype=torch.int32, device="cuda"
    )
    seqlens_kv = torch.full((B,), kv_len, dtype=torch.int32, device="cuda")
    q_packed = _make_fp8((total_q, Hq, D), in_dtype)
    k_dense = _make_fp8((B, Skv, Hkv, D), in_dtype)
    v_dense = _make_fp8((B, Skv, Hkv, D), in_dtype)
    k_pool, v_pool, block_tables = _make_paged_kv(k_dense, v_dense, page_size)

    attn = PagedAttention(torch.device("cuda:0"))
    attn.plan(
        _metadata(cu_seqlens_q, seqlens_kv, block_tables, page_size),
        num_qo_heads=Hq,
        num_kv_heads=Hkv,
        head_dim_qk=D,
        q_dtype=out_dtype,
        kv_dtype=in_dtype,
        kv_layout="HND",
        causal=is_causal,
        backend="fa2",
    )
    assert attn.backend == "fa2"
    o, _ = attn.run(q_packed.to(out_dtype), (k_pool, v_pool), sm_scale=sm_scale)

    # legacy reference at the legacy tolerance, per request
    for b in range(B):
        q_start = int(cu_seqlens_q[b])
        q_end = int(cu_seqlens_q[b + 1])
        q_b = q_packed[q_start:q_end]
        k_b = k_dense[b, :kv_len]
        v_b = v_dense[b, :kv_len]
        ref_b = _ref_paged_fmha_single(q_b, k_b, v_b, sm_scale, is_causal, kv_len).to(
            out_dtype
        )
        torch.testing.assert_close(o[q_start:q_end], ref_b, **_tol())

    # fp32 oracle
    ref_out, _ = reference_paged_prefill(
        q_packed,
        k_pool,
        v_pool,
        cu_seqlens_q.cpu(),
        seqlens_kv.cpu(),
        block_tables,
        page_size,
        is_causal,
        sm_scale=sm_scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
