"""Legacy paged-prefill tests -> unified ``PagedAttention`` parity, group A (gap rows).

The legacy entries here exercise something the unified API cannot express
today (a fused positional encoding, head_dim 512 / (448, 256), multi-item
scoring positions, an NVFP4 KV cache, causal rows with q_len > kv_len,
k/v scales on a float KV cache, a batch-invariant split policy, the legacy
wrapper's exact float/int workspace split).  Each row asserts that the
unified API REJECTS the legacy input with a clear ``ValueError`` (or
``TypeError`` for an argument the contract does not have), or that
``resolve_paged_attention`` excludes every backend with the reason -- never
that it silently runs something else.

Every rejection row is written against an ``EXPECT_*`` constant so that the
capability extension (WP-T: head_dim 512, float-KV scales, ...) flips the
row into the positive legacy check: the legacy fixture, the legacy reference
method and the legacy tolerance are already in place under the ``True``
branch, plus the independent fp32 oracle.  Where the legacy semantics can be
reproduced through the existing contract (a migration adapter: ``sm_scale *=
k_scale`` / ``out *= v_scale``; the multi-item visible set as a boolean
``custom_mask``; external RoPE on q and the K pages), a second ``partial``
row runs that adapter against the legacy reference so the cost of the
adapter is measured, not asserted away.

``LEGACY_MAP`` follows PLAN.md §2; the fixture / oracle helpers come from
``test_paged_attention_legacy_parity_a_core.py``.
"""

import inspect
import math

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .test_paged_attention_legacy_parity_a_core import (
    BACKENDS,
    DEVICE,
    EXPECT_FLOAT_KV_SCALES,
    EXPECT_FULLY_MASKED_ROWS,
    LegacyBatch,
    _grid,
    _param_rows,
    _seed,
    assert_legacy_close,
    assert_oracle,
    legacy_reference_single_prefill,
    legacy_uniform_batch,
    resolve_or_skip,
    unified_plan,
    unified_run,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (legacy nodeid or function, unified test function(s) in this file, status, note)
LEGACY_MAP = [
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_head_dim_512",
        ["test_legacy_head_dim_512"],
        "unsupported-by-design",
        "head_dim 512 is not in any backend's declared head_dims (fa2 declares 64/128/256): "
        "resolve excludes every backend with 'unsupported head dims (512, 512)'; needs the "
        "fa2 capability row for (512, 512) (WP-T) -- the legacy fixture (B2 kv97 q17 page16 "
        "H4:4 NHD fp16, causal x NONE) and 1e-3 check are in place under EXPECT_HEAD_DIM_512; "
        "ROPE_LLAMA rows stay unsupported.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides",
        ["test_legacy_shared_kv_smem_unequal_kv_strides"],
        "unsupported-by-design",
        "D512 shared-KV-smem producer path: capability-excluded like head_dim 512; the "
        "padded-parent K/V pools (unequal stride families, NHD/HND, q17/65) and the exact "
        "fp32 reference at 2e-3 are in place under EXPECT_HEAD_DIM_512.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256",
        ["test_legacy_cta_tile_q_probe_qk448_vo256"],
        "native-only",
        "(448, 256) is not a declared head-dim pair (resolve excludes it); the CTA_TILE_Q "
        "plan_info assertion is a planner-internal contract with no unified observable -- "
        "the numerical half (fp16 KV, 2e-3 vs fp32) is in place under EXPECT_HEAD_DIM_448_256.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_multi_item_scoring",
        [
            "test_legacy_multi_item_scoring_kwargs_unsupported",
            "test_legacy_multi_item_scoring_via_custom_mask",
        ],
        "unsupported-by-design",
        "plan() has no prefix_len_ptr / token_pos_in_items_ptr / token_pos_in_items_len / "
        "max_item_len_ptr (TypeError); the legacy rows are ROPE_LLAMA-only, which the unified "
        "API cannot apply.  The visible set itself IS expressible: the legacy mask builder "
        "(create_2D_multi_item_mask_dense) as custom_mask with causal=False reproduces the "
        "legacy reference (single_prefill with the same mask, pos NONE) at 1e-3 with soft cap "
        "0/30 and LSE on/off on fa2 -- the partial row; needs a positional-encoding axis "
        "(fused RoPE or an external-RoPE contract) plus item-position fields to be equivalent.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "kv_dtype uint8 (packed FP4x2) is not a declared KV dtype and run() has no "
        "kv_cache_sf; needs a quantization descriptor (packed dtype, scale-factor tensors, "
        "layout, global scales).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_strided_scale_views",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4; additionally needs independent scale-factor strides.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_asymmetric",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4; additionally (512, 256) / (256, 128) head-dim pairs are undeclared.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_nvfp4",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4; the stride-router prewarm (prewarm_paged_kv_stride_variant) is a legacy "
        "wrapper knob with no unified counterpart.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_large_head_bf16",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 (bf16 q).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 plus fused ROPE_LLAMA.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_nvfp4_rope_large_head_bf16",
        ["test_legacy_nvfp4_kv_unsupported"],
        "unsupported-by-design",
        "as nvfp4 plus head_dim 512 plus fused ROPE_LLAMA (bf16 q).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_fully_masked_rows",
        ["test_legacy_fully_masked_rows"],
        "unsupported-by-design",
        "causal with q_len 34 > kv_len 1 is rejected by validate_causal_envelope "
        "('causal masking requires q_len_i <= kv_len_i'); legacy defined the 33 fully masked "
        "rows as out 0 / LSE -inf.  Needs an explicit fully-masked-row policy in the contract "
        "(and an oracle that does not produce NaN for them); the legacy assertions are in place "
        "under EXPECT_FULLY_MASKED_ROWS.  The same fixture runs non-causally (sanity).",
    ),
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_effect",
        ["test_legacy_kv_scale_forwarding_effect"],
        "unsupported-by-design",
        "k_scale / v_scale on a fp16/bf16 KV cache are rejected ('per-tensor KV scales apply "
        "to fp8 KV caches only'); the migration adapter (sm_scale = k_scale / sqrt(D), out * "
        "v_scale) reproduces the legacy effect; needs the float-KV scale fold (WP-T) -- legacy "
        "assertion in place under EXPECT_FLOAT_KV_SCALES.",
    ),
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_math_property",
        ["test_legacy_kv_scale_forwarding_math_property"],
        "unsupported-by-design",
        "as above; the three legacy cases (k only, v only, both) are checked through the "
        "adapter against the legacy wrapper's k_scale/v_scale outputs at the legacy tolerance "
        "(rtol 1e-2, atol 1e-3) and against the unified q*k_scale / out*v_scale identities.",
    ),
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_prefill_tensor_cores",
        ["test_legacy_batch_invariant_fixed_split_size"],
        "unsupported-by-design",
        "plan() has no fixed_split_size / disable_split_kv (TypeError) and the contract "
        "promises no batch invariance; the legacy fixture (pos NONE rows) is run at batch B "
        "and at the first 2 requests, both against the oracle, and the bitwise comparison is "
        "an xfail-if-unequal until a determinism policy exists (EXPECT_BATCH_INVARIANT).",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers",
        ["test_legacy_workspace_fixed_split_exact_buffers"],
        "partial",
        "fixed_split_size is not expressible (TypeError); the unified sizing contract is "
        "workspace_requirements() (one conservative byte bound, not an exact float/int split) "
        "and a buffer of exactly that size plans and runs the legacy geometry (B3 q64 kv1024 "
        "page16 H16:4 D128) against the oracle.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers",
        ["test_legacy_workspace_cuda_graph_exact_buffers"],
        "partial",
        "graph mode: workspace_requirements(use_cuda_graph=True) for the legacy geometry is "
        "exact for fa2 (a buffer of that size captures and replays against the oracle; 16 "
        "bytes less is rejected at plan naming the bytes); the legacy int-workspace half is "
        "instance-owned in the unified API.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer",
        ["test_legacy_workspace_rejects_unaligned_buffer"],
        "partial",
        "an unaligned caller workspace is rejected at plan() on fa2 with the legacy wrapper's "
        "'float_workspace_buffer must be 16-byte aligned' (not at construction: the unified "
        "constructor validates dtype/device/shape only); other backends are reject-or-correct.",
    ),
]

# Flip constants (see the module docstring); the two shared with the core
# file are imported from there.
EXPECT_HEAD_DIM_512 = True  # fa2 (512, 512) declared by WP-T (a466bd67)
EXPECT_HEAD_DIM_448_256 = False
EXPECT_NVFP4_KV = False
EXPECT_BATCH_INVARIANT = False
EXPECT_ROPE = False
EXPECT_MULTI_ITEM = False

MB = 1024 * 1024


def _plan_params():
    return inspect.signature(PagedAttention.plan).parameters


def _assert_every_backend_excluded(match, *, also=(), **cfg):
    """Every backend is excluded and each reason names the axis (``match``),
    one of the ``also`` axes the configuration violates first, or the
    device's compute capability; ``auto`` raises the aggregated ValueError."""
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        resolve_paged_attention(device=torch.device(DEVICE), backend="auto", **cfg)
    assert match in str(ei.value), str(ei.value)
    accepted = (match, "compute capability", *also)
    for backend in BACKENDS[:-1]:
        with pytest.raises(ValueError) as ei:
            resolve_paged_attention(device=torch.device(DEVICE), backend=backend, **cfg)
        msg = str(ei.value)
        assert any(a in msg for a in accepted), f"{backend}: {msg}"


# ---------------------------------------------------------------------------
# positional encoding (an axis of the main / tuple / custom-mask / multi-item
# legacy grids; no legacy function of its own)
# ---------------------------------------------------------------------------


def test_legacy_rope_pos_encoding_unsupported():
    """The legacy ``pos_encoding_mode="ROPE_LLAMA"`` rows: the unified plan()
    has no positional-encoding argument (RoPE is the caller's, applied before
    the call), so the legacy kwargs are a TypeError, never a silent NONE."""
    params = _plan_params()
    for name in ("pos_encoding_mode", "rope_scale", "rope_theta"):
        assert name not in params, f"plan() grew {name!r}: port the ROPE_LLAMA rows"
    lb = legacy_uniform_batch(
        batch_size=2,
        kv_len=97,
        qo_len=17,
        page_size=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        seed=_seed("rope-reject"),
    )
    with pytest.raises(TypeError, match="pos_encoding_mode"):
        PagedAttention(torch.device(DEVICE)).plan(
            lb.metadata(),
            **lb.plan_kwargs(),
            causal=True,
            pos_encoding_mode="ROPE_LLAMA",
            backend="fa2",
        )
    if EXPECT_ROPE:
        pytest.fail("EXPECT_ROPE is set: add the fused-RoPE plan argument to this row")


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_rope_llama_external_adapter(backend, causal):
    """The migration adapter for the ROPE_LLAMA rows: rotate q (position
    ``kv_len - q_len + r``) and the request's K pages (position ``j``) with
    ``flashinfer.apply_rope_pos_ids`` (Llama non-interleaved, theta 1e4,
    scale 1 -- the fused kernel's defaults), then run the unified API on the
    rotated tensors.  Compared with the legacy fused reference
    ``single_prefill_with_kv_cache(pos_encoding_mode="ROPE_LLAMA")`` on the
    UNROTATED inputs at the legacy tolerance, and with the oracle on the
    rotated ones.  Legacy tuple fixture point B2 kv97 q17 page16 H4:4 D128."""
    lb = legacy_uniform_batch(
        batch_size=2,
        kv_len=97,
        qo_len=17,
        page_size=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        combined=False,
        seed=_seed("rope-adapter", causal),
    )
    md = lb.metadata()
    resolve_or_skip(lb, md, backend, causal=causal)
    # legacy fused reference on the unrotated q / K / V
    ref = torch.cat(
        [
            flashinfer.prefill.single_prefill_with_kv_cache(
                lb.request_q(i),
                *lb.request_kv(i),
                causal=causal,
                pos_encoding_mode="ROPE_LLAMA",
            )
            for i in range(lb.batch_size)
        ]
    )
    # external RoPE: q rows at their absolute positions, K pages in place
    q_rot = torch.empty_like(lb.q)
    k_rot = lb.k.clone()
    P, Hk, D = lb.page_size, lb.num_kv_heads, lb.head_dim_qk
    k_rows = k_rot.view(-1, Hk, D)  # NHD pool rows == token rows
    for i in range(lb.batch_size):
        s, e = int(lb.q_indptr_cpu[i]), int(lb.q_indptr_cpu[i + 1])
        lq, lkv = e - s, int(lb.kv_seq_lens_cpu[i])
        pos_q = torch.arange(lkv - lq, lkv, dtype=torch.int32, device=lb.q.device)
        q_i = lb.q[s:e]
        q_rot[s:e] = flashinfer.apply_rope_pos_ids(q_i, q_i[:, :Hk], pos_q)[0]
        page0 = int(lb.page_ids(i)[0])
        rows = slice(
            page0 * P, page0 * P + lkv
        )  # arange page ids: rows are consecutive
        k_i = k_rows[rows]
        pos_k = torch.arange(0, lkv, dtype=torch.int32, device=lb.q.device)
        k_rows[rows] = flashinfer.apply_rope_pos_ids(k_i, k_i, pos_k)[1]
    _, out, lse = unified_run(lb, md, backend, q=q_rot, k=k_rot, causal=causal)
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=causal, q=q_rot, k=k_rot)


# ---------------------------------------------------------------------------
# head_dim 512 and the (448, 256) probe
# ---------------------------------------------------------------------------


def _head_dim_512_cfg(**over):
    cfg = dict(
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=512,
        head_dim_vo=512,
        q_dtype=torch.float16,
        page_size=16,
        kv_layout="NHD",
        need_lse=True,
    )
    cfg.update(over)
    return cfg


@pytest.mark.parametrize("causal", [False, True])
def test_legacy_head_dim_512(causal):
    """``test_batch_prefill_with_paged_kv_cache_head_dim_512`` (NONE rows).
    Today: every backend excluded with the head-dim reason.  Flipped: the
    legacy fixture on fa2 at 1e-3 vs single_prefill(backend="fa2") + oracle
    + the legacy caller-buffer re-run."""
    if not EXPECT_HEAD_DIM_512:
        _assert_every_backend_excluded(
            "unsupported head dims (512, 512)", **_head_dim_512_cfg(causal=causal)
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
        seed=_seed("hd512", causal),
    )
    md = lb.metadata()
    attn, out, lse = unified_run(lb, md, "fa2", causal=causal)
    ref = legacy_reference_single_prefill(lb, causal=causal, backend="fa2")
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=causal)
    out_buf, lse_buf = torch.empty_like(out), torch.empty_like(lse)
    attn.run(lb.q, (lb.k, lb.v), out=out_buf, lse=lse_buf)
    torch.testing.assert_close(out, out_buf, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buf, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("qo_len", [17, 65])
def test_legacy_shared_kv_smem_unequal_kv_strides(kv_layout, qo_len):
    """``test_batch_prefill_paged_shared_kv_smem_unequal_kv_strides``: D512
    fp16, K and V views of differently padded parents (unequal stride
    families).  Today: capability-excluded.  Flipped: fa2 vs the exact fp32
    reference at 2e-3 (legacy) + oracle."""
    if not EXPECT_HEAD_DIM_512:
        _assert_every_backend_excluded(
            "unsupported head dims (512, 512)",
            **_head_dim_512_cfg(num_qo_heads=2, num_kv_heads=2, kv_layout=kv_layout),
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
    _, out, lse = unified_run(lb, lb.metadata(), "fa2", causal=True)
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


@pytest.mark.parametrize("kv_dtype", [torch.float16, torch.float8_e4m3fn])
def test_legacy_cta_tile_q_probe_qk448_vo256(kv_dtype):
    """``test_batch_prefill_paged_cta_tile_q_smem_probe_qk448_vo256``: the
    (448, 256) pair is undeclared, so resolve excludes it; the CTA_TILE_Q
    plan_info pin stays native.  Flipped: the fp16 numerical half at 2e-3."""
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
        _assert_every_backend_excluded(
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
    _, out, lse = unified_run(lb, lb.metadata(), "fa2", causal=False)
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
# multi-item scoring
# ---------------------------------------------------------------------------

MULTI_ITEM_FIXTURES = [
    # (kv_len, qo_len, prefix_len_ptr, token_pos_in_items_ptr, token_pos_in_items_len, max_item_len_ptr)
    (54, 37, 17, list(range(17)) + list(range(19)) + [0], 100, [18]),
    (97, 81, 16, list(range(80)) + [0], 97, [79]),
]


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


def test_legacy_multi_item_scoring_kwargs_unsupported():
    """plan() has none of the multi-item fields; passing them is a TypeError."""
    params = _plan_params()
    for name in (
        "prefix_len_ptr",
        "token_pos_in_items_ptr",
        "token_pos_in_items_len",
        "max_item_len_ptr",
    ):
        assert name not in params, f"plan() grew {name!r}: port the multi-item rows"
    kv_len, qo_len, prefix, items, items_len, max_item = MULTI_ITEM_FIXTURES[0]
    lb = legacy_uniform_batch(
        batch_size=1,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        seed=_seed("multi-reject"),
    )
    with pytest.raises(TypeError, match="prefix_len_ptr"):
        PagedAttention(torch.device(DEVICE)).plan(
            lb.metadata(),
            **lb.plan_kwargs(),
            causal=True,
            prefix_len_ptr=torch.tensor(prefix).to(torch.uint32).to(DEVICE),
            backend="fa2",
        )
    if EXPECT_MULTI_ITEM:
        pytest.fail(
            "EXPECT_MULTI_ITEM is set: add the item-position fields to this row"
        )


@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("page_size", [1, 5, 16])
@pytest.mark.parametrize(
    "fixture", [0, 1], ids=["kv54-q37-prefix17", "kv97-q81-prefix16"]
)
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_legacy_multi_item_scoring_via_custom_mask(
    backend, fixture, page_size, num_qo_heads, logits_soft_cap, return_lse
):
    """The legacy multi-item grid (B1, H4/32:4, D128, page 1/5/16, soft cap
    0/30, LSE on/off) with its visible set expressed as ``custom_mask`` and
    ``causal=False``; pos_encoding_mode NONE instead of the legacy
    ROPE_LLAMA (the one axis that cannot be expressed).  Legacy reference:
    ``single_prefill_with_kv_cache(custom_mask=...)`` at 1e-3 + oracle."""
    kv_len, qo_len, prefix, items, items_len, max_item = MULTI_ITEM_FIXTURES[fixture]
    lb = legacy_uniform_batch(
        batch_size=1,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=4,
        head_dim=128,
        fp32_source=False,
        seed=_seed(
            "multi", fixture, page_size, num_qo_heads, logits_soft_cap, return_lse
        ),
    )
    md = lb.metadata()
    mask = create_2D_multi_item_mask_dense(
        is_delimiter=torch.tensor(items).to(DEVICE) == 0,
        sliding_window_size=-1,
        prefix_cache_len=prefix,
    )
    assert mask.numel() == qo_len * kv_len
    cap = None if logits_soft_cap == 0.0 else logits_soft_cap
    _, out, lse = unified_run(
        lb,
        md,
        backend,
        causal=False,
        custom_mask=mask,
        logits_soft_cap=cap,
        lse_mode="base2" if return_lse else "none",
    )
    ref = legacy_reference_single_prefill(
        lb, causal=True, logits_soft_cap=logits_soft_cap, custom_mask=lambda i: mask
    )
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(
        lb,
        out,
        lse,
        causal=False,
        custom_mask=mask,
        logits_soft_cap=cap,
        lse_mode="base2" if return_lse else "none",
    )


# ---------------------------------------------------------------------------
# NVFP4 KV cache (packed uint8 + scale factors)
# ---------------------------------------------------------------------------

NVFP4_LEGACY_ENTRIES = [
    # id (legacy function suffix), (num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, page_size, q_dtype, extra_axis)
    ("nvfp4", (1, 1, 128, 128, 16, torch.float16, None)),
    (
        "nvfp4_strided_scale_views",
        (4, 2, 128, 128, 16, torch.float16, "independent scale-factor strides"),
    ),
    (
        "nvfp4_asymmetric",
        (8, 2, 512, 256, 16, torch.bfloat16, "asymmetric (512, 256) pools"),
    ),
    (
        "lazy_stride_router_nvfp4",
        (4, 2, 128, 128, 16, torch.float16, "prewarm_paged_kv_stride_variant"),
    ),
    ("nvfp4_large_head", (1, 1, 512, 512, 16, torch.float16, "head_dim 512")),
    ("nvfp4_large_head_bf16", (1, 1, 512, 512, 16, torch.bfloat16, "head_dim 512")),
    (
        "nvfp4_rope_large_head",
        (1, 1, 512, 512, 16, torch.float16, "head_dim 512 + ROPE_LLAMA"),
    ),
    (
        "nvfp4_rope_large_head_bf16",
        (1, 1, 512, 512, 16, torch.bfloat16, "head_dim 512 + ROPE_LLAMA"),
    ),
]


@pytest.mark.parametrize(
    "entry", NVFP4_LEGACY_ENTRIES, ids=[e[0] for e in NVFP4_LEGACY_ENTRIES]
)
def test_legacy_nvfp4_kv_unsupported(entry):
    """Every NVFP4 legacy entry: a packed uint8 KV cache is not a declared KV
    dtype (every backend excluded with the dtype -- or, for the D512 /
    asymmetric entries, the head-dim -- reason) and run() has no
    ``kv_cache_sf``.  Flipping ``EXPECT_NVFP4_KV`` marks this row for the
    quantization-descriptor extension."""
    name, (hq, hk, dqk, dvo, page, q_dtype, extra) = entry
    cfg = dict(
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=dqk,
        head_dim_vo=dvo,
        q_dtype=q_dtype,
        kv_dtype=torch.uint8,
        page_size=page,
        kv_layout="NHD",
        causal=False,
    )
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        resolve_paged_attention(device=torch.device(DEVICE), backend="auto", **cfg)
    msg = str(ei.value)
    assert (
        "unsupported kv dtype torch.uint8" in msg or "unsupported head dims" in msg
    ), msg
    assert "kv_cache_sf" not in inspect.signature(PagedAttention.run).parameters
    if EXPECT_NVFP4_KV:
        pytest.fail(
            f"EXPECT_NVFP4_KV is set: port the legacy {name} fixture ({extra or 'base'})"
        )


# ---------------------------------------------------------------------------
# fully masked causal rows (q_len > kv_len)
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_fully_masked_rows(backend, dtype):
    """``test_paged_prefill_fully_masked_rows``: q34 / kv1 / page 1.  Today the
    causal plan is rejected by the envelope; the non-causal plan on the same
    fixture runs (every row attends the single all-ones value: out 1, LSE 0).
    Flipped: the legacy assertions (33 rows out 0 / LSE -inf, last row 1 / 0)."""
    lb, num_masked = _fully_masked_fixture(dtype)
    md = lb.metadata()
    resolve_or_skip(lb, md, backend, causal=True)
    if not EXPECT_FULLY_MASKED_ROWS:
        with pytest.raises(
            ValueError, match="causal masking requires q_len_i <= kv_len_i"
        ):
            PagedAttention(torch.device(DEVICE)).plan(
                md, **lb.plan_kwargs(), causal=True, lse_mode="base2", backend=backend
            )
        _, out, lse = unified_run(lb, md, backend, causal=False)
        assert not out.isnan().any() and not lse.isnan().any()
        torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
        torch.testing.assert_close(lse, torch.zeros_like(lse), rtol=0, atol=0)
        return
    _, out, lse = unified_run(lb, md, backend, causal=True)
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
# k_scale / v_scale on a float KV cache
# ---------------------------------------------------------------------------


def _kv_scale_fixture(dtype, n_ctx, seed):
    torch.manual_seed(seed)
    H_QO, H_KV, HEAD_DIM, PAGE_SIZE = 1, 1, 64, 16
    max_num_pages = (n_ctx + PAGE_SIZE - 1) // PAGE_SIZE
    dev = torch.device(DEVICE)
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device=dev
    )
    v_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device=dev
    )
    q = torch.randn(n_ctx, H_QO, HEAD_DIM, dtype=dtype, device=dev)
    return LegacyBatch(
        q=q,
        k=k_cache,
        v=v_cache,
        q_indptr_cpu=torch.tensor([0, n_ctx], dtype=torch.int32),
        kv_indptr_cpu=torch.tensor([0, max_num_pages], dtype=torch.int32),
        kv_indices_cpu=torch.arange(max_num_pages, dtype=torch.int32),
        last_page_len_cpu=torch.tensor(
            [n_ctx % PAGE_SIZE or PAGE_SIZE], dtype=torch.int32
        ),
        page_size=PAGE_SIZE,
        kv_layout="NHD",
    )


def _legacy_kv_scale_wrapper(lb):
    """The legacy wrapper planned as in ``test_batch_prefill.py``.  The legacy
    test then called the deprecated ``forward_return_lse``, which silently
    resets the plan's ``causal`` to False and ``sm_scale`` to the default
    before running; the parity rows call ``run(..., return_lse=True)`` so the
    reference keeps the causal plan the legacy test declared."""
    dev = torch.device(DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(16 * MB, dtype=torch.uint8, device=dev)
    )
    wrapper.plan(
        lb.q_indptr_cpu.to(dev),
        lb.kv_indptr_cpu.to(dev),
        lb.kv_indices_cpu.to(dev),
        lb.last_page_len_cpu.to(dev),
        lb.num_qo_heads,
        lb.num_kv_heads,
        lb.head_dim_qk,
        lb.page_size,
        causal=True,
        q_data_type=lb.dtype,
        kv_data_type=lb.dtype,
    )
    return wrapper


def _adapter_run(lb, md, backend, *, k_scale=None, v_scale=None, q=None):
    """The migration adapter for float-KV scales through the unified
    contract: ``sm_scale = k_scale / sqrt(D)`` and ``out * v_scale``."""
    sm_scale = None if k_scale is None else k_scale / math.sqrt(lb.head_dim_qk)
    attn, out, lse = unified_run(lb, md, backend, q=q, causal=True, sm_scale=sm_scale)
    if v_scale is not None:
        out = (out.float() * v_scale).to(out.dtype)
    return attn, out, lse


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_kv_scale_forwarding_effect(backend, dtype):
    """``test_kv_scale_forwarding_effect``: scales (0.1, 0.1) vs (2.0, 2.0)
    must change the output.  Today the scales are rejected on a float KV
    cache and the adapter carries the legacy assertion."""
    lb = _kv_scale_fixture(dtype, n_ctx=8, seed=42)
    md = lb.metadata()
    attn = unified_plan(lb, md, backend, causal=True)
    if not EXPECT_FLOAT_KV_SCALES:
        with pytest.raises(ValueError, match="KV scales apply to fp8 KV caches only"):
            attn.run(lb.q, (lb.k, lb.v), k_scale=0.1, v_scale=0.1)
        _, out1, _ = _adapter_run(lb, md, backend, k_scale=0.1, v_scale=0.1)
        _, out2, _ = _adapter_run(lb, md, backend, k_scale=2.0, v_scale=2.0)
    else:
        out1, _ = attn.run(lb.q, (lb.k, lb.v), k_scale=0.1, v_scale=0.1)
        out2, _ = attn.run(lb.q, (lb.k, lb.v), k_scale=2.0, v_scale=2.0)
    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different."
    )
    # and the adapter matches the legacy wrapper's own scaled output
    wrapper = _legacy_kv_scale_wrapper(lb)
    ref1, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=0.1, v_scale=0.1)
    ref2, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=2.0, v_scale=2.0)
    torch.testing.assert_close(out1, ref1, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_kv_scale_forwarding_math_property(backend, dtype):
    """``test_kv_scale_forwarding_math_property``: k_scale == scaling q,
    v_scale == scaling the output, both together -- through the adapter
    against the legacy wrapper's k_scale/v_scale outputs and against the
    unified identities, at the legacy tolerance (rtol 1e-2, atol 1e-3)."""
    lb = _kv_scale_fixture(dtype, n_ctx=128, seed=0)
    md = lb.metadata()
    k_scale, v_scale = 0.5, 2.0
    attn = unified_plan(lb, md, backend, causal=True)
    wrapper = _legacy_kv_scale_wrapper(lb)
    if EXPECT_FLOAT_KV_SCALES:
        run = lambda **kw: attn.run(lb.q, (lb.k, lb.v), **kw)[0]  # noqa: E731
        out1, out2, out3 = (
            run(k_scale=k_scale),
            run(v_scale=v_scale),
            run(k_scale=k_scale, v_scale=v_scale),
        )
    else:
        with pytest.raises(ValueError, match="KV scales apply to fp8 KV caches only"):
            attn.run(lb.q, (lb.k, lb.v), k_scale=k_scale)
        with pytest.raises(ValueError, match="KV scales apply to fp8 KV caches only"):
            attn.run(lb.q, (lb.k, lb.v), v_scale=v_scale)
        out1 = _adapter_run(lb, md, backend, k_scale=k_scale)[1]
        out2 = _adapter_run(lb, md, backend, v_scale=v_scale)[1]
        out3 = _adapter_run(lb, md, backend, k_scale=k_scale, v_scale=v_scale)[1]
    # legacy identities, in unified terms
    base, _ = attn.run(lb.q, (lb.k, lb.v))
    scaled_q, _ = attn.run((lb.q * k_scale).to(lb.dtype), (lb.k, lb.v))
    torch.testing.assert_close(out1, scaled_q, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        out2, (base.float() * v_scale).to(base.dtype), rtol=1e-2, atol=1e-3
    )
    torch.testing.assert_close(
        out3, (scaled_q.float() * v_scale).to(base.dtype), rtol=1e-2, atol=1e-3
    )
    # and the legacy wrapper's own k_scale / v_scale outputs
    ref1, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=k_scale)
    ref2, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, v_scale=v_scale)
    ref3, _ = wrapper.run(
        lb.q, (lb.k, lb.v), return_lse=True, k_scale=k_scale, v_scale=v_scale
    )
    torch.testing.assert_close(out1, ref1, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(out2, ref2, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(out3, ref3, rtol=1e-2, atol=1e-3)
    assert_oracle(lb, base, None, causal=True, lse_mode="none")


# ---------------------------------------------------------------------------
# batch invariance (fixed_split_size / disable_split_kv)
# ---------------------------------------------------------------------------

INVARIANT_AXES = dict(
    batch_size=[3, 4],
    kv_len=[4096, 5000],
    qo_len=[128, 256],
    disable_split_kv=[True, False],
    page_size=[1, 8, 16],
    group_size=[1, 4, 8],
    head_dim=[128, 256],
    kv_layout=["HND", "NHD"],
)
INVARIANT_DEFAULT = {
    (3, 4096, 128, True, 1, 1, 128, "HND"),
    (4, 5000, 256, False, 8, 4, 256, "NHD"),
    (3, 5000, 128, False, 16, 8, 128, "NHD"),
    (4, 4096, 256, True, 16, 1, 256, "HND"),
}


def test_legacy_batch_invariant_kwargs_unsupported():
    params = _plan_params()
    for name in ("fixed_split_size", "disable_split_kv"):
        assert name not in params, (
            f"plan() grew {name!r}: port the batch-invariant rows"
        )
    lb = legacy_uniform_batch(
        batch_size=2,
        kv_len=97,
        qo_len=17,
        page_size=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        seed=_seed("invariant-reject"),
    )
    with pytest.raises(TypeError, match="fixed_split_size"):
        PagedAttention(torch.device(DEVICE)).plan(
            lb.metadata(),
            **lb.plan_kwargs(),
            causal=True,
            fixed_split_size=2048,
            backend="fa2",
        )


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,disable_split_kv,page_size,group_size,head_dim,kv_layout,backend",
    _param_rows(_grid(INVARIANT_AXES), lambda p: tuple(p) in INVARIANT_DEFAULT),
)
def test_legacy_batch_invariant_fixed_split_size(
    batch_size,
    kv_len,
    qo_len,
    disable_split_kv,
    page_size,
    group_size,
    head_dim,
    kv_layout,
    backend,
):
    """``test_batch_prefill_tensor_cores`` (pos NONE rows): the legacy fixture
    (fp16 combined pool /10, invariant_bs 2) at batch B and at its first two
    requests, each against the oracle; the legacy bitwise assertion is an
    xfail-if-unequal until the API has a determinism policy (the legacy
    ``fixed_split_size`` / ``disable_split_kv`` knobs have no counterpart:
    ``disable_split_kv`` is carried as a parameter only)."""
    invariant_bs = 2
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=4 * group_size,
        num_kv_heads=4,
        head_dim=head_dim,
        kv_layout=kv_layout,
        fp32_source=False,
        scale=10.0,
        seed=_seed(
            "invariant",
            batch_size,
            kv_len,
            qo_len,
            disable_split_kv,
            page_size,
            group_size,
            head_dim,
            kv_layout,
        ),
    )
    md = lb.metadata()
    _, out, lse = unified_run(lb, md, backend, causal=False)
    assert_oracle(lb, out, lse, causal=False)
    sub = LegacyBatch(
        q=lb.q[: invariant_bs * qo_len],
        k=lb.k,
        v=lb.v,
        q_indptr_cpu=lb.q_indptr_cpu[: invariant_bs + 1],
        kv_indptr_cpu=lb.kv_indptr_cpu[: invariant_bs + 1],
        kv_indices_cpu=lb.kv_indices_cpu,
        last_page_len_cpu=lb.last_page_len_cpu[:invariant_bs],
        page_size=page_size,
        kv_layout=kv_layout,
    )
    _, out_inv, lse_inv = unified_run(sub, sub.metadata(), backend, causal=False)
    assert_oracle(sub, out_inv, lse_inv, causal=False)
    n = invariant_bs * qo_len
    equal = torch.equal(out[:n], out_inv) and torch.equal(lse[:n], lse_inv)
    if EXPECT_BATCH_INVARIANT:
        assert equal, "batch invariance promised but the prefix batch differs bitwise"
    elif not equal:
        pytest.xfail(
            f"[{backend}] no batch-invariance contract: prefix batch differs bitwise "
            f"(max |out| diff {(out[:n].float() - out_inv.float()).abs().max().item():.2e}, "
            f"max |lse| diff {(lse[:n] - lse_inv).abs().max().item():.2e}); "
            "legacy pinned it with fixed_split_size / disable_split_kv"
        )


# ---------------------------------------------------------------------------
# workspace sizing (exact buffers, cuda graph, alignment)
# ---------------------------------------------------------------------------


def _workspace_fixture(seed):
    """The legacy ``_run_batch_prefill_workspace_size_plan`` geometry with
    fp16 NHD pools (the legacy test plans only; here the buffer must also run)."""
    return legacy_uniform_batch(
        batch_size=3,
        kv_len=1024,
        qo_len=64,
        page_size=16,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=128,
        combined=False,
        seed=seed,
    )


def _workspace_capacity(lb):
    from flashinfer.prefill import GraphCapacity

    width = int(lb.kv_indptr_cpu.diff().max())
    return GraphCapacity(
        batch_size=lb.batch_size,
        total_q_tokens=int(lb.q_indptr_cpu[-1]),
        max_q_len=int(lb.q_lens_cpu.max()),
        max_kv_len=width * lb.page_size,
        page_size=lb.page_size,
        table_width=width,
    )


def _requirements(lb, *, use_cuda_graph, backend="fa2"):
    return PagedAttention.workspace_requirements(
        _workspace_capacity(lb),
        device=torch.device(DEVICE),
        **lb.plan_kwargs(),
        causal=True,
        need_lse=True,
        use_cuda_graph=use_cuda_graph,
        backend=backend,
    )


def test_legacy_workspace_fixed_split_exact_buffers():
    """``..._plans_fixed_split_with_exact_buffers``: fixed_split_size is not
    a unified argument; the eager bound of ``workspace_requirements`` for the
    legacy geometry is positive and a buffer of exactly that size plans AND
    runs on fa2 against the oracle (the legacy test only planned)."""
    assert "fixed_split_size" not in _plan_params()
    lb = _workspace_fixture(_seed("ws-eager"))
    md = lb.metadata()
    resolve_or_skip(lb, md, "fa2")
    nbytes = _requirements(lb, use_cuda_graph=False)
    assert nbytes > 0
    dev = torch.device(DEVICE)
    attn = PagedAttention(
        dev, workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=dev)
    )
    unified_plan(lb, md, "fa2", attn=attn)
    out, lse = attn.run(lb.q, (lb.k, lb.v))
    assert_oracle(lb, out, lse, causal=True)


def test_legacy_workspace_cuda_graph_exact_buffers():
    """``..._plans_cuda_graph_with_exact_buffers``: the graph-mode bound for
    the legacy geometry captures and replays on a buffer of exactly that size
    (against the oracle); 16 bytes less is rejected at plan naming the bytes."""
    lb = _workspace_fixture(_seed("ws-graph"))
    width = int(lb.kv_indptr_cpu.diff().max())
    md = lb.metadata(table_width=width)
    resolve_or_skip(lb, md, "fa2")
    nbytes = _requirements(lb, use_cuda_graph=True)
    assert nbytes > 0
    dev = torch.device(DEVICE)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    attn = PagedAttention(
        dev, graph_capacity=_workspace_capacity(lb), workspace_buffer=ws
    )
    unified_plan(lb, md, "fa2", attn=attn)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    attn.update(md)
    g.replay()
    torch.cuda.synchronize()
    assert_oracle(lb, out, lse, causal=True)
    short = PagedAttention(
        dev, graph_capacity=_workspace_capacity(lb), workspace_buffer=ws[: nbytes - 16]
    )
    with pytest.raises(ValueError, match=rf"needs {nbytes} bytes"):
        short.plan(md, **lb.plan_kwargs(), causal=True, lse_mode="base2", backend="fa2")


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_workspace_rejects_unaligned_buffer(backend):
    """``..._rejects_unaligned_workspace_buffer``: a caller buffer whose data
    pointer is not 16-byte aligned.  The unified constructor accepts any
    contiguous 1-D byte tensor; the rejection surfaces at plan() from the
    fa2 wrapper ('float_workspace_buffer must be 16-byte aligned').  Other
    backends: reject with a ValueError or run correctly, never misread."""
    lb = _workspace_fixture(_seed("ws-unaligned"))
    md = lb.metadata()
    resolve_or_skip(lb, md, backend)
    dev = torch.device(DEVICE)
    ws = torch.empty(32 * MB + 1, dtype=torch.uint8, device=dev)[1:]
    assert ws.data_ptr() % 16 != 0
    attn = PagedAttention(dev, workspace_buffer=ws)
    try:
        unified_plan(lb, md, backend, attn=attn)
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    except ValueError as e:
        assert "aligned" in str(e), e
        if backend in ("fa2", "auto") and attn.backend in (None, "fa2"):
            assert "float_workspace_buffer must be 16-byte aligned" in str(e)
        return
    torch.cuda.synchronize()
    assert_oracle(lb, out, lse, causal=True)
    pytest.skip(
        f"[{attn.backend}] accepts the unaligned workspace and matches the oracle; "
        "the legacy alignment rejection is fa2-specific"
    )
