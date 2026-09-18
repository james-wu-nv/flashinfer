"""Legacy paged-prefill tests -> unified ``PagedAttention`` parity, group A (core rows).

Every test here re-creates a legacy test's OWN fixture -- the same shapes,
dtypes, layouts, page sizes, the same combined ``(pages, 2, ...)`` pool sliced
as ``K = kv[:, 0]`` / ``V = kv[:, 1]`` views without copies, the same random
construction (under a per-row seed, since the legacy tests are unseeded) --
and runs it through ``flashinfer.prefill.PagedAttention`` with the lossless
metadata mapping of ``reports/unified-test-review-20260917/03 §2``:

    kv_seq_lens[i] = (pages_i - 1) * page_size + last_page_len[i]
    page_size < 8  -> PagedAttentionMetadata.csr(legacy indices)
    page_size >= 8 -> PagedAttentionMetadata.dense(indices reshaped per request)

Each row asserts TWICE:

- against the LEGACY reference method with the LEGACY tolerance (the same
  per-request ``single_prefill_with_kv_cache`` / ``ref_single_prefill`` /
  legacy-wrapper comparison the legacy test makes), and
- against the independent fp32 oracle (``paged_attention_reference.py``, at
  the suite's oracle budget ``OUT_TOL`` / ``LSE_TOL``), so a bug shared by the
  legacy kernel and the unified backend cannot hide behind the first check.

Backends are pinned explicitly (``attn.backend`` is asserted) and ``auto`` is
run where the legacy test used the default backend.  A row whose backend is
capability-excluded skips with the resolve reason -- the manifest status of
that row is ``partial`` for that backend, never ``equivalent``.  The full
legacy parameter grids are kept under the ``slow`` marker (legacy backend
only), which ``tests/conftest.py`` skips unless ``FI_PARITY_SLOW=1``; the
default run covers a deterministic subset in which every axis value appears
at least once.  Shapes are never shrunk.

``LEGACY_MAP`` below is the per-file crosswalk (format: PLAN.md §2); the
gap rows (RoPE, head_dim 512, NVFP4, ...) live in
``test_paged_attention_legacy_parity_a_features.py``.
"""

import itertools
import zlib

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)
from tests.test_helpers.paged_kv import make_padded_paged_kv_view
from tests.test_helpers.test_helpers import ref_single_prefill

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import LSE_TOL, OUT_TOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (legacy nodeid or function, unified test function(s) in this file, status, note)
LEGACY_MAP = [
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache",
        ["test_legacy_main_paged_grid", "test_legacy_main_paged_cuda_graph_row"],
        "partial",
        "pos_encoding_mode=NONE rows: same grid (B12/17/128 x kv54/97/512/2048 x "
        "q37/17/127/577 x page1/5/16 x H4/32:4 x D64/128/256 x causal), same fp16 "
        "combined NHD pool as K=kv[:,0]/V=kv[:,1] views, legacy tolerance 1e-3 vs "
        "per-request single_prefill_with_kv_cache + oracle, fa2 pinned and auto; "
        "ROPE_LLAMA rows unsupported (features file); use_cuda_graph=True rows were "
        "an unconditional legacy xfail -- the unified graph row (plan/capture/update/"
        "replay on the same fixture) passes; cudnn/trtllm-gen/cake run the D128/page16 "
        "cells and skip the rest with the capability reason.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_tuple_paged_kv_cache",
        ["test_legacy_tuple_paged_grid"],
        "partial",
        "NONE rows: same grid (D128/256) with the two separately allocated fp16 NHD "
        "pools, legacy tolerance vs single_prefill + oracle; ROPE rows unsupported; "
        "cuda-graph rows were a legacy xfail.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_lazy_stride_router_plan_reuse",
        ["test_legacy_lazy_stride_router_plan_reuse"],
        "equivalent",
        "same fixture (bf16 /4, B2 q17 kv97 page16 H8:2, padded V view via "
        "make_padded_paged_kv_view), one plan / three runs equal->unequal->equal "
        "strides, legacy tolerances (2e-2 vs fp64 ref_single_prefill, 1e-2 pairwise) "
        "+ oracle on fa2; fa3/trtllm-gen/cake must reject-or-correct the unequal "
        "run; fixed_split_size=2 has no unified counterpart (the split policy is the "
        "backend's, see features file).",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_batch_prefill_with_paged_kv_cache_custom_mask",
        ["test_legacy_custom_mask_grid"],
        "partial",
        "NONE rows: same grid (page1/16, D128/256, H4/32:4); the legacy tril mask as "
        "custom_mask with causal=False vs the causal=True plan at 1e-3 (legacy "
        "assertion) + oracle for both; fa2 is the only mask-capable backend, as for "
        "the legacy kernel; ROPE rows unsupported.",
    ),
    (
        "tests/attention/test_batch_prefill_kernels.py::test_paged_prefill_split_kv_empty_chunk",
        ["test_legacy_split_kv_empty_chunk"],
        "equivalent",
        "same fixture (q/10, combined NHD kv/10, B1 q2 kv129 page16 H8:2 D128, fp16 "
        "and bf16), 1e-2 vs fp64 ref_single_prefill for out AND LSE + oracle; dense "
        "and CSR forms, every backend.",
    ),
    (
        "tests/attention/test_non_contiguous_prefill.py::test_batch_paged_prefill_packed_input",
        ["test_legacy_packed_q_input"],
        "equivalent",
        "same grid (B1/19/99 x page1/5 x seq1/7/127/257 x Hkv1/4/8 x Hq4/8 x "
        "D64/128/256 x causal); q is the head slice of a fused QKV buffer; slice vs "
        "q.contiguous() at legacy tolerance (rtol 1e-3, atol 2e-3) + oracle for both; "
        "page 1/5 => CSR form, so only the CSR-native fa backends resolve (as in the "
        "legacy fa2 run); cudnn rejects strided q by contract.",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_paged_prefill_sliding_window",
        ["test_legacy_sliding_window_grid"],
        "partial",
        "same grid (B12/17/30 x kv54/397/1177 x q1/37/47 x window13/33/111 x "
        "Hkv1/4/8 x Hq4/8 x D64/128/256/512 x page1/16 x fa2/auto), legacy tolerance "
        "1e-3 vs single_prefill(window_left, backend=fa2) + oracle; head_dim 512 rows "
        "are capability-excluded (WP-T extension), so they skip with the reason.",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_correctness",
        ["test_legacy_batch_attention_correctness"],
        "partial",
        "same seq configs (incl. the numpy-seeded random 256-request batch) and axes "
        "(page1/8/16, Hkv1/4, group1/4/7/8, D64/128/256, v_scale, causal, HND/NHD, "
        "bf16/fp16, softcap 0/50); unified vs the legacy fa2 wrapper at 1e-2 (out and "
        "LSE) + chunked oracle; v_scale=2.0 rows: the unified API rejects float-KV "
        "scales, the migration adapter out*v_scale is checked instead; the causal "
        "q>kv config (kv2/q235 + kv1/q13353) is rejected by the causal envelope "
        "(fully-masked rows, features file); softcap 50 rows resolve on fa2 only; the "
        "BatchAttention holistic scheduler itself is not a unified backend.",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_with_noncontiguous_q",
        ["test_legacy_batch_attention_noncontiguous_q"],
        "partial",
        "same fixture ((146,146), page 1, D64 bf16 NHD, q = first chunk of a 2D-wide "
        "buffer); numerics vs the legacy fa2 wrapper at 1e-2 + oracle; the holistic "
        "scheduler is not exercised.",
    ),
    (
        "tests/attention/test_shared_prefix_kernels.py::test_batch_attention_with_shared_prefix_paged_kv_cache",
        [
            "test_legacy_shared_prefix_one_shot",
            "test_legacy_shared_prefix_composed_merge",
        ],
        "partial",
        "same fixture (append_paged_kv_cache pool, decode/append stages, B12/17, "
        "unique37/17, shared128/512/2048, H8/16, D128/256, page1/16): one-shot paged "
        "attention over shared+unique pages vs MultiLevelCascadeAttentionWrapper at "
        "1e-3 + oracle; the two-level composition (two PagedAttention runs + "
        "merge_state on base-2 LSE) vs the cascade at 1e-3; the merge is an external "
        "composition, not a PagedAttention feature.",
    ),
    (
        "tests/attention/test_shared_prefix_kernels.py::test_merge_state_in_place_with_mask",
        ["test_legacy_merge_state_in_place_is_native_only"],
        "native-only",
        "merge-operator contract; PagedAttention has no merge entry point and the "
        "legacy operator test is retained.",
    ),
]

DEVICE = "cuda:0"
BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "cake", "auto"]
slow = pytest.mark.slow

# Flip constants for rows the unified API cannot express today (the features
# file asserts the rejection while False; WP-T style extensions flip them).
EXPECT_FLOAT_KV_SCALES = False  # k_scale / v_scale on a fp16/bf16 KV cache
EXPECT_FULLY_MASKED_ROWS = False  # causal rows with q_len > kv_len (out 0 / LSE -inf)


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _param_rows(rows, default_pred, *, slow_backends=("fa2",)):
    """Cross legacy grid points with backends.  Default-subset points run on
    every backend; the rest of the legacy grid is kept under ``slow`` on the
    legacy backend(s) only, so the full legacy grid stays runnable without
    multiplying it by six."""
    out = []
    for point in rows:
        default = default_pred(point)
        for backend in BACKENDS:
            if not default and backend not in slow_backends:
                continue
            out.append(
                pytest.param(
                    *point,
                    backend,
                    marks=() if default else (slow,),
                    id="-".join(str(x) for x in point) + f"-{backend}",
                )
            )
    return out


# ---------------------------------------------------------------------------
# legacy fixture + lossless metadata mapping
# ---------------------------------------------------------------------------


class LegacyBatch:
    """A legacy paged-prefill batch: the legacy wrapper's CSR metadata
    (``qo_indptr`` / ``paged_kv_indptr`` / ``paged_kv_indices`` /
    ``paged_kv_last_page_len``) plus the K/V pools as the legacy test holds
    them (combined-pool views or separate pools).  ``metadata()`` performs the
    lossless mapping to the unified canonical form."""

    def __init__(
        self,
        *,
        q,
        k,
        v,
        q_indptr_cpu,
        kv_indptr_cpu,
        kv_indices_cpu,
        last_page_len_cpu,
        page_size,
        kv_layout,
    ):
        self.q, self.k, self.v = q, k, v
        self.q_indptr_cpu = q_indptr_cpu.to(torch.int32)
        self.kv_indptr_cpu = kv_indptr_cpu.to(torch.int32)
        self.kv_indices_cpu = kv_indices_cpu.to(torch.int32)
        self.last_page_len_cpu = last_page_len_cpu.to(torch.int32)
        self.page_size = page_size
        self.kv_layout = kv_layout
        pages = self.kv_indptr_cpu.diff()
        # 03 §2.1: non-empty request -> (pages - 1) * P + last; zero pages -> 0
        self.kv_seq_lens_cpu = torch.where(
            pages > 0,
            (pages - 1) * page_size + self.last_page_len_cpu,
            torch.zeros_like(pages),
        ).to(torch.int32)
        self.q_lens_cpu = self.q_indptr_cpu.diff()

    # ---- static configuration ----
    @property
    def batch_size(self):
        return int(self.kv_seq_lens_cpu.shape[0])

    @property
    def num_qo_heads(self):
        return int(self.q.shape[1])

    @property
    def num_kv_heads(self):
        return int(self.k.shape[2] if self.kv_layout == "NHD" else self.k.shape[1])

    @property
    def head_dim_qk(self):
        return int(self.k.shape[3])

    @property
    def head_dim_vo(self):
        return int(self.v.shape[3])

    @property
    def dtype(self):
        return self.q.dtype

    def plan_kwargs(self):
        return dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim_qk,
            head_dim_vo=self.head_dim_vo,
            q_dtype=self.dtype,
            kv_dtype=self.k.dtype,
            kv_layout=self.kv_layout,
        )

    # ---- per-request views (the legacy reference's gather) ----
    def page_ids(self, i):
        return self.kv_indices_cpu[
            int(self.kv_indptr_cpu[i]) : int(self.kv_indptr_cpu[i + 1])
        ]

    def request_q(self, i):
        return self.q[int(self.q_indptr_cpu[i]) : int(self.q_indptr_cpu[i + 1])]

    def request_kv(self, i):
        """(kv_len, Hkv, D) K and V of request ``i`` gathered from the pools --
        the legacy tests' cat(full pages, last page[:last_page_len])."""
        kv_len = int(self.kv_seq_lens_cpu[i])
        ids = self.page_ids(i).to(self.k.device, torch.long)

        def gather(pool):
            pages = pool[ids]
            if self.kv_layout == "HND":
                pages = pages.permute(0, 2, 1, 3)
            return pages.reshape(-1, pages.shape[-2], pages.shape[-1])[:kv_len]

        return gather(self.k), gather(self.v)

    # ---- lossless mapping to the unified form ----
    def metadata(self, form=None, device=DEVICE, table_width=None):
        """``table_width`` pads the dense table (CUDA-graph capacities need
        every batch's table at the captured width)."""
        form = form or ("csr" if self.page_size < 8 else "dense")
        dev = torch.device(device)
        common = dict(
            page_size=self.page_size,
            max_q_len=int(self.q_lens_cpu.max()),
            max_kv_len=max(int(self.kv_seq_lens_cpu.max()), 1),
            qo_indptr_cpu=self.q_indptr_cpu,
            kv_seq_lens_cpu=self.kv_seq_lens_cpu,
        )
        if form == "csr":
            return PagedAttentionMetadata.csr(
                self.q_indptr_cpu.to(dev),
                self.kv_seq_lens_cpu.to(dev),
                self.kv_indices_cpu.to(dev),
                **common,
            )
        assert form == "dense", form
        pages = self.kv_indptr_cpu.diff()
        width = int(pages.max()) if table_width is None else table_width
        assert width >= int(pages.max())
        table = torch.zeros(self.batch_size, width, dtype=torch.int32)
        for i in range(self.batch_size):
            table[i, : int(pages[i])] = self.page_ids(i)
        return PagedAttentionMetadata.dense(
            self.q_indptr_cpu.to(dev),
            self.kv_seq_lens_cpu.to(dev),
            table.to(dev),
            **common,
        )


def legacy_uniform_batch(
    *,
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seed,
    kv_layout="NHD",
    dtype=torch.float16,
    combined=True,
    fp32_source=True,
    scale=1.0,
    device=DEVICE,
):
    """The legacy ``test_batch_prefill_kernels`` fixture: uniform lengths,
    ``paged_kv_indices = arange(total_pages)``, last page ``(kv_len - 1) % P +
    1``; a combined ``(pages, 2, ...)`` pool (K/V are its views) or two pools;
    K/V drawn in fp32 and rounded (``fp32_source``) as the legacy main test
    does, or directly in ``dtype``."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=dev, dtype=dtype
    )
    if scale != 1.0:
        q = q / scale
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    if kv_layout == "HND":
        pool_shape = (num_kv_heads, page_size, head_dim)
    else:
        pool_shape = (page_size, num_kv_heads, head_dim)
    src_dtype = torch.float32 if fp32_source else dtype

    def draw(shape):
        t = torch.randn(*shape, dtype=src_dtype, device=dev)
        if scale != 1.0:
            t = t / scale
        return t.to(dtype)

    if combined:
        kv = draw((total_pages, 2, *pool_shape))
        k, v = kv[:, 0], kv[:, 1]  # views, no copy
        assert k.data_ptr() == kv.data_ptr() and not k.is_contiguous()
    else:
        k = draw((total_pages, *pool_shape))
        v = draw((total_pages, *pool_shape))
    return LegacyBatch(
        q=q,
        k=k,
        v=v,
        q_indptr_cpu=q_indptr_cpu,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_seq,
        kv_indices_cpu=torch.arange(0, total_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout=kv_layout,
    )


def legacy_reference_single_prefill(
    lb, *, causal, window_left=-1, logits_soft_cap=0.0, backend="auto", custom_mask=None
):
    """The legacy reference method: ``single_prefill_with_kv_cache`` per
    request on the legacy per-request K/V gather, concatenated."""
    outs = []
    for i in range(lb.batch_size):
        ki, vi = lb.request_kv(i)
        kw = {}
        if custom_mask is not None:
            kw["custom_mask"] = custom_mask(i)
        outs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                lb.request_q(i),
                ki,
                vi,
                causal=causal,
                pos_encoding_mode="NONE",
                logits_soft_cap=logits_soft_cap,
                window_left=window_left,
                backend=backend,
                **kw,
            )
        )
    return torch.cat(outs)


def assert_legacy_close(out, ref, *, rtol, atol, what="output"):
    """The legacy assertion: ``torch.isclose`` mismatch count, one sync."""
    assert out.shape == ref.shape, (
        f"{what}: shape {tuple(out.shape)} vs {tuple(ref.shape)}"
    )
    close = torch.isclose(out.float(), ref.float(), rtol=rtol, atol=atol)
    bad = int((~close).sum())
    if bad:
        diff = (out.float() - ref.float()).abs()
        raise AssertionError(
            f"{what}: {bad}/{out.numel()} elements outside the legacy tolerance "
            f"rtol={rtol} atol={atol}; max abs diff {diff.max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# oracle (fp32, independent) with query-row chunking for long requests
# ---------------------------------------------------------------------------


def oracle(
    lb,
    *,
    causal,
    window_left=-1,
    sm_scale=None,
    lse_base="2",
    custom_mask=None,
    logits_soft_cap=None,
    q=None,
    k=None,
    v=None,
    max_rows=2048,
):
    """``reference_paged_prefill`` on the legacy batch.  A request longer than
    ``max_rows`` query rows is split into row chunks, each a pseudo-request
    whose KV is truncated to ``kv_len - q_len + chunk_end`` (bottom-right
    causal alignment and the sliding window depend only on the absolute
    query position, which the truncation preserves), so the (H, lq, lkv)
    score tensor of the 8190x7939 legacy config stays bounded.  Rows carrying
    a custom mask, or non-causal windows, are never chunked."""
    q = lb.q if q is None else q
    k = lb.k if k is None else k
    v = lb.v if v is None else v
    P = lb.page_size
    q_lens, kv_lens, pages = [], [], []
    for i in range(lb.batch_size):
        lq, lkv = int(lb.q_lens_cpu[i]), int(lb.kv_seq_lens_cpu[i])
        ids = lb.page_ids(i)
        chunk = custom_mask is None and lq > max_rows and (causal or window_left < 0)
        if not chunk:
            q_lens.append(lq)
            kv_lens.append(lkv)
            pages.append(ids[: (lkv + P - 1) // P])
            continue
        for a in range(0, lq, max_rows):
            b = min(a + max_rows, lq)
            kv_b = lkv - lq + b if causal else lkv
            q_lens.append(b - a)
            kv_lens.append(kv_b)
            pages.append(ids[: (kv_b + P - 1) // P])
    qo_indptr_cpu = torch.tensor(
        [0] + list(itertools.accumulate(q_lens)), dtype=torch.int32
    )
    kv_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32)
    flat = torch.cat(pages).to(q.device, torch.int32)
    return reference_paged_prefill(
        q.contiguous(),
        k,
        v,
        qo_indptr_cpu,
        kv_lens_cpu,
        None,
        P,
        causal,
        sm_scale=sm_scale,
        window_left=window_left,
        kv_layout=lb.kv_layout,
        kv_page_indices=flat,
        lse_base=lse_base,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
    )


def assert_oracle(lb, out, lse, *, causal, lse_mode="base2", **kw):
    ref_out, ref_lse = oracle(
        lb, causal=causal, lse_base="e" if lse_mode == "basee" else "2", **kw
    )
    assert torch.isfinite(out).all(), "non-finite output"
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    if lse_mode == "none":
        assert lse is None
    else:
        assert (
            lse.shape == (out.shape[0], lb.num_qo_heads) and lse.dtype == torch.float32
        )
        assert torch.isfinite(lse).all(), "non-finite LSE"
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    return ref_out, ref_lse


# ---------------------------------------------------------------------------
# unified plan / run with explicit backend pinning
# ---------------------------------------------------------------------------


def resolve_or_skip(
    lb,
    md,
    backend,
    *,
    causal=True,
    window_left=-1,
    need_lse=True,
    logits_soft_cap=None,
    custom_mask=False,
):
    """Resolution for the legacy configuration, or a skip that records WHY
    this backend cannot run the row (the manifest reads it as ``partial``)."""
    try:
        return resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=lb.num_qo_heads,
            num_kv_heads=lb.num_kv_heads,
            head_dim_qk=lb.head_dim_qk,
            head_dim_vo=lb.head_dim_vo,
            q_dtype=lb.dtype,
            kv_dtype=lb.k.dtype,
            page_size=lb.page_size,
            kv_layout=lb.kv_layout,
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=md.kv_input_form,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"[{backend}] capability-excluded: {e}")


def unified_plan(
    lb,
    md,
    backend,
    *,
    causal=True,
    window_left=-1,
    lse_mode="base2",
    logits_soft_cap=None,
    custom_mask=None,
    attn=None,
):
    res = resolve_or_skip(
        lb,
        md,
        backend,
        causal=causal,
        window_left=window_left,
        need_lse=lse_mode != "none",
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask is not None,
    )
    attn = attn if attn is not None else PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        **lb.plan_kwargs(),
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        backend=backend,
    )
    if backend == "auto":
        assert attn.backend in res.backends, attn.explain()
    else:
        assert attn.backend == backend, attn.explain()
    return attn


def unified_run(lb, md, backend, *, q=None, k=None, v=None, sm_scale=None, **plan_kw):
    attn = unified_plan(lb, md, backend, **plan_kw)
    out, lse = attn.run(
        lb.q if q is None else q,
        (lb.k if k is None else k, lb.v if v is None else v),
        sm_scale=sm_scale,
    )
    return attn, out, lse


# ---------------------------------------------------------------------------
# test_batch_prefill_kernels.py :: main paged grid (combined pool)
# ---------------------------------------------------------------------------

MAIN_AXES = dict(
    batch_size=[12, 17, 128],
    kv_len=[54, 97, 512, 2048],
    qo_len=[37, 17, 127, 577],
    page_size=[1, 5, 16],
    num_kv_heads=[4],
    num_qo_heads=[4, 32],
    head_dim=[64, 128, 256],
    causal=[False, True],
)
# default subset: every batch / kv / qo value at least once, including a
# q > kv non-causal point (its causal twin skips exactly as in legacy)
MAIN_DEFAULT_TRIPLES = {
    (12, 54, 37),
    (17, 97, 17),
    (128, 512, 127),
    (12, 2048, 577),
    (17, 54, 577),
}


def _grid(axes, **override):
    axes = dict(axes, **override)
    return list(itertools.product(*axes.values()))


def _main_default(point):
    return tuple(point[:3]) in MAIN_DEFAULT_TRIPLES


def _run_legacy_main_grid(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    backend,
    *,
    combined,
    check_caller_buffers,
):
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
        combined=combined,
        seed=_seed(
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
    attn, out, lse = unified_run(lb, md, backend, causal=causal)
    ref = legacy_reference_single_prefill(lb, causal=causal)
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=causal)
    if check_caller_buffers:
        # legacy: a second run into pre-allocated out / lse buffers matches
        out_buf, lse_buf = torch.empty_like(out), torch.empty_like(lse)
        o2, l2 = attn.run(lb.q, (lb.k, lb.v), out=out_buf, lse=lse_buf)
        assert o2 is out_buf and l2 is lse_buf
        assert_legacy_close(
            out, out_buf, rtol=1e-3, atol=1e-3, what="caller out buffer"
        )
        assert_legacy_close(
            lse, lse_buf, rtol=1e-3, atol=1e-3, what="caller lse buffer"
        )


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,page_size,num_kv_heads,num_qo_heads,head_dim,causal,backend",
    _param_rows(_grid(MAIN_AXES), _main_default),
)
def test_legacy_main_paged_grid(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    backend,
):
    """``test_batch_prefill_with_paged_kv_cache`` (pos_encoding_mode=NONE,
    use_cuda_graph=False): combined fp16 NHD pool as K/V views."""
    _run_legacy_main_grid(
        batch_size,
        kv_len,
        qo_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        causal,
        backend,
        combined=True,
        check_caller_buffers=True,
    )


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,page_size,num_kv_heads,num_qo_heads,head_dim,causal,backend",
    _param_rows(_grid(MAIN_AXES, head_dim=[128, 256]), _main_default),
)
def test_legacy_tuple_paged_grid(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    backend,
):
    """``test_batch_prefill_with_tuple_paged_kv_cache``: two separately
    allocated fp16 NHD pools."""
    _run_legacy_main_grid(
        batch_size,
        kv_len,
        qo_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        causal,
        backend,
        combined=False,
        check_caller_buffers=False,
    )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_main_paged_cuda_graph_row(backend, causal):
    """The legacy main test's ``use_cuda_graph=True`` rows were an unconditional
    xfail (workspace overflow).  The unified lifecycle on the same fixture
    (B12 kv97 q17 page16 H32:4 D128 fp16 combined NHD): plan a warm-up batch
    inside a capacity sized for the real one, capture, ``update()`` to the
    legacy batch, replay, legacy tolerance + oracle."""
    from flashinfer.prefill import GraphCapacity

    lb = legacy_uniform_batch(
        batch_size=12,
        kv_len=97,
        qo_len=17,
        page_size=16,
        num_qo_heads=32,
        num_kv_heads=4,
        head_dim=128,
        seed=_seed("main-graph", causal),
    )
    width = int(lb.kv_indptr_cpu.diff().max())
    md = lb.metadata(table_width=width)
    resolve_or_skip(lb, md, backend, causal=causal)
    # legacy warm-up plan: one page per request, last_page_len = page_size
    warm = LegacyBatch(
        q=lb.q,
        k=lb.k,
        v=lb.v,
        q_indptr_cpu=lb.q_indptr_cpu,
        kv_indptr_cpu=torch.arange(0, lb.batch_size + 1, dtype=torch.int32),
        kv_indices_cpu=torch.arange(0, lb.batch_size, dtype=torch.int32),
        last_page_len_cpu=torch.full((lb.batch_size,), lb.page_size, dtype=torch.int32),
        page_size=lb.page_size,
        kv_layout=lb.kv_layout,
    )
    if causal:
        # the warm-up batch has kv 16 < q 17: legacy's warm-up plan ran it
        # anyway; the unified causal envelope rejects it, so warm up on a
        # legal prefix (kv = 2 pages) instead
        warm = LegacyBatch(
            q=lb.q,
            k=lb.k,
            v=lb.v,
            q_indptr_cpu=lb.q_indptr_cpu,
            kv_indptr_cpu=torch.arange(0, lb.batch_size + 1, dtype=torch.int32) * 2,
            kv_indices_cpu=torch.arange(0, 2 * lb.batch_size, dtype=torch.int32),
            last_page_len_cpu=torch.full(
                (lb.batch_size,), lb.page_size, dtype=torch.int32
            ),
            page_size=lb.page_size,
            kv_layout=lb.kv_layout,
        )
    cap = GraphCapacity(
        batch_size=lb.batch_size,
        total_q_tokens=int(lb.q_indptr_cpu[-1]),
        max_q_len=int(lb.q_lens_cpu.max()),
        max_kv_len=width * lb.page_size,
        page_size=lb.page_size,
        table_width=width,
    )
    dev = torch.device(DEVICE)
    attn = PagedAttention(dev, graph_capacity=cap)
    unified_plan(
        lb, warm.metadata("dense", table_width=width), backend, causal=causal, attn=attn
    )
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out, lse = attn.run(lb.q, (lb.k, lb.v))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    attn.update(md)
    g.replay()
    torch.cuda.synchronize()
    ref = legacy_reference_single_prefill(lb, causal=causal)
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=causal)


# ---------------------------------------------------------------------------
# test_batch_prefill_kernels.py :: lazy stride router plan reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_layout,head_dim", [("NHD", 64), ("HND", 128)])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_lazy_stride_router_plan_reuse(kv_layout, head_dim, backend):
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
    attn = unified_plan(lb, md, backend, causal=True)
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


# ---------------------------------------------------------------------------
# test_batch_prefill_kernels.py :: custom mask grid
# ---------------------------------------------------------------------------

CUSTOM_MASK_AXES = dict(
    batch_size=[12, 17, 128],
    kv_len=[54, 97, 512, 2048],
    qo_len=[37, 17, 127, 577],
    page_size=[1, 16],
    num_kv_heads=[4],
    num_qo_heads=[4, 32],
    head_dim=[128, 256],
)
CUSTOM_MASK_DEFAULT_TRIPLES = {(12, 54, 37), (17, 97, 17), (128, 512, 127)}


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,page_size,num_kv_heads,num_qo_heads,head_dim,backend",
    _param_rows(
        _grid(CUSTOM_MASK_AXES), lambda p: tuple(p[:3]) in CUSTOM_MASK_DEFAULT_TRIPLES
    ),
)
def test_legacy_custom_mask_grid(
    batch_size, kv_len, qo_len, page_size, num_kv_heads, num_qo_heads, head_dim, backend
):
    """``test_batch_prefill_with_paged_kv_cache_custom_mask``: the legacy
    bottom-right tril mask passed as ``custom_mask`` (with ``causal=False``,
    since the unified mask is ANDed into the envelope and the legacy CUSTOM
    mode replaced causal) must equal the ``causal=True`` plan at 1e-3, and
    both match the oracle."""
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
        fp32_source=False,
        seed=_seed(
            "custom", batch_size, kv_len, qo_len, page_size, num_qo_heads, head_dim
        ),
    )
    md = lb.metadata()
    custom_mask = torch.tril(
        torch.full((batch_size, qo_len, kv_len), True, device=DEVICE),
        diagonal=(kv_len - qo_len),
    ).reshape(-1)
    _, out_custom, lse_custom = unified_run(
        lb, md, backend, causal=False, custom_mask=custom_mask
    )
    _, out_causal, lse_causal = unified_run(lb, md, backend, causal=True)
    assert_legacy_close(out_custom, out_causal, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(lb, out_custom, lse_custom, causal=False, custom_mask=custom_mask)
    assert_oracle(lb, out_causal, lse_causal, causal=True)


# ---------------------------------------------------------------------------
# test_batch_prefill_kernels.py :: split-KV empty chunk fixture
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("form", ["dense", "csr"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_split_kv_empty_chunk(backend, dtype, form):
    """``test_paged_prefill_split_kv_empty_chunk``: q2 / kv129 / page16 with
    the legacy /10 inputs and combined NHD pool; finite, and within 1e-2 of
    the fp64 ``ref_single_prefill`` for out AND LSE (legacy), plus oracle."""
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
        seed=_seed("split", str(dtype), form),
    )
    md = lb.metadata(form)
    _, out, lse = unified_run(lb, md, backend, causal=True)
    ki, vi = lb.request_kv(0)
    o_ref, lse_ref = ref_single_prefill(lb.q, ki, vi, causal=True)
    assert not out.isnan().any() and not lse.isnan().any()
    torch.testing.assert_close(out, o_ref, rtol=1e-2, atol=1e-2)  # legacy
    torch.testing.assert_close(lse, lse_ref, rtol=1e-2, atol=1e-2)  # legacy
    assert_oracle(lb, out, lse, causal=True)


# ---------------------------------------------------------------------------
# test_non_contiguous_prefill.py :: packed (fused QKV) query view
# ---------------------------------------------------------------------------

PACKED_AXES = dict(
    batch_size=[1, 19, 99],
    page_size=[1, 5],
    seq_len=[1, 7, 127, 257],
    num_kv_heads=[1, 4, 8],
    num_qo_heads=[4, 8],
    head_dim=[64, 128, 256],
    causal=[True, False],
)
PACKED_DEFAULT = {(1, 1), (19, 7), (99, 127), (19, 257)}  # (batch_size, seq_len)


def _packed_default(p):
    b, page, seq, hk, hq, hd, causal = p
    return (b, seq) in PACKED_DEFAULT and (hk, hq) in {(1, 4), (4, 8), (8, 8)}


@pytest.mark.parametrize(
    "batch_size,page_size,seq_len,num_kv_heads,num_qo_heads,head_dim,causal,backend",
    _param_rows(_grid(PACKED_AXES), _packed_default),
)
def test_legacy_packed_q_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    backend,
):
    """``test_batch_paged_prefill_packed_input``: q is the head slice of a
    fused QKV projection (token stride (Hq + 2 Hkv) D); the slice and its
    ``.contiguous()`` copy must agree at the legacy tolerance and both match
    the oracle.  cuDNN rejects the strided view by contract (a packed copy
    is the documented answer), which the row records instead of misreading."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")  # legacy
    torch.manual_seed(
        _seed(
            "packed",
            batch_size,
            page_size,
            seq_len,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            causal,
        )
    )
    dev = torch.device(DEVICE)
    nnz = batch_size * seq_len
    pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * pages_per_req
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=dev
    )
    v_cache = torch.randn_like(k_cache)
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device=dev,
    )
    q, _, _ = qkv_packed.split(
        (num_qo_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim),
        dim=-1,
    )
    q = q.view(-1, num_qo_heads, head_dim)
    # a single token (nnz == 1) is contiguous by PyTorch's definition; every
    # other point is the strided fused-QKV slice
    assert q.stride(-1) == 1 and (nnz == 1 or not q.is_contiguous())
    lb = LegacyBatch(
        q=q,
        k=k_cache,
        v=v_cache,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * seq_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_req,
        kv_indices_cpu=torch.arange(num_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (seq_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout="NHD",
    )
    md = lb.metadata()
    attn = unified_plan(lb, md, backend, causal=causal)
    from flashinfer.experimental.paged_attention import CAPABILITIES

    if CAPABILITIES[attn.backend].requires_contiguous_q and not q.is_contiguous():
        with pytest.raises(ValueError, match="requires packed q"):
            attn.run(q, (k_cache, v_cache))
        out_c, lse_c = attn.run(q.contiguous(), (k_cache, v_cache))
        assert_oracle(lb, out_c, lse_c, causal=causal)
        pytest.skip(
            f"[{attn.backend}] rejects the fused-QKV head slice by contract "
            "(requires packed q); the contiguous copy matches the oracle"
        )
    out_packed, lse_packed = attn.run(q, (k_cache, v_cache))
    out_contig, lse_contig = attn.run(q.contiguous(), (k_cache, v_cache))
    torch.testing.assert_close(out_packed, out_contig, rtol=1e-3, atol=2e-3)  # legacy
    assert_oracle(lb, out_packed, lse_packed, causal=causal)
    assert_oracle(lb, out_contig, lse_contig, causal=causal)


# ---------------------------------------------------------------------------
# test_sliding_window.py :: batch paged prefill sliding window
# ---------------------------------------------------------------------------

SWA_AXES = dict(
    batch_size=[12, 17, 30],
    kv_len=[54, 397, 1177],
    qo_len=[1, 37, 47],
    window_left=[13, 33, 111],
    num_kv_heads=[1, 4, 8],
    num_qo_heads=[4, 8],
    head_dim=[64, 128, 256, 512],
    page_size=[1, 16],
)
SWA_DEFAULT_QUADS = {
    (12, 54, 1, 13),
    (17, 397, 37, 33),
    (30, 1177, 47, 111),
    (12, 1177, 37, 13),
}


def _swa_default(p):
    b, kv, qo, w, hk, hq, hd, page = p
    return (b, kv, qo, w) in SWA_DEFAULT_QUADS and (hk, hq) in {(1, 4), (4, 8), (8, 8)}


@pytest.mark.parametrize(
    "batch_size,kv_len,qo_len,window_left,num_kv_heads,num_qo_heads,head_dim,page_size,backend",
    _param_rows(_grid(SWA_AXES), _swa_default, slow_backends=("fa2", "auto")),
)
def test_legacy_sliding_window_grid(
    batch_size,
    kv_len,
    qo_len,
    window_left,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    backend,
):
    """``test_batch_paged_prefill_sliding_window``: fp16 NHD pools, causal
    sliding window; legacy tolerance vs ``single_prefill_with_kv_cache(
    window_left, causal=True, backend="fa2")`` + oracle.  ``window_left`` has
    the same meaning in both APIs (keys ``j >= p - window_left``)."""
    if num_qo_heads < num_kv_heads:
        pytest.skip("num_qo_heads < num_kv_heads is not supported")  # legacy
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        combined=False,
        fp32_source=False,
        seed=_seed(
            "swa",
            batch_size,
            kv_len,
            qo_len,
            window_left,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            page_size,
        ),
    )
    md = lb.metadata()
    _, out, lse = unified_run(lb, md, backend, causal=True, window_left=window_left)
    ref = legacy_reference_single_prefill(
        lb, causal=True, window_left=window_left, backend="fa2"
    )
    assert_legacy_close(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=True, window_left=window_left)


# ---------------------------------------------------------------------------
# test_batch_attention.py :: holistic correctness (legacy fa2 wrapper as ref)
# ---------------------------------------------------------------------------


def _build_seq_len_configs():
    """Verbatim from ``tests/attention/test_batch_attention.py`` (including the
    numpy-seeded random 256-request batch)."""
    np.random.seed(42)
    torch.manual_seed(42)
    seq_len_configs = [
        [(146, 146)],
        [(67, 67)],
        [(8190, 7939)],
        [(2048, 1)] * 77,  # decode-only
        [(4099, 129)] * 2,  # prefill-only
        [(600, 1)] * 132 * 2 + [(5000, 3)] * 128,
        [(1024, 1)] * 100 + [(8192, 17)] * 8,  # speculative decode
        [(766, 2)] * 99 + [(1024, 512)] * 1,  # chunked prefill
        [(2, 235)] + [(1, 13353)],  # real workload
    ]
    bsz, stride, sparsity = 256, 16, 0.05
    full_kv_len = np.random.randint(1000, 11000, size=bsz)
    seq_len = []
    for i in range(bsz):
        if i % stride == 0:
            kv_len, qo_len = full_kv_len[i], stride + 1
        else:
            kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
        seq_len.append((int(kv_len), int(qo_len)))
    seq_len_configs.append(seq_len)
    return seq_len_configs


SEQ_LEN_CONFIGS = _build_seq_len_configs()
HOLISTIC_AXES = dict(
    config=list(range(len(SEQ_LEN_CONFIGS))),
    page_block_size=[1, 8, 16],
    num_kv_heads=[1, 4],
    gqa_group_size=[1, 4, 7, 8],
    head_dim=[64, 128, 256],
    v_scale=[2.0, None],
    causal=[False, True],
    layout=["HND", "NHD"],
    test_dtype=["bf16", "fp16"],
    logits_soft_cap=[0.0, 50.0],
)
_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}


def _holistic_default_rows():
    """A rotation through the legacy axes: each seq config twice, every other
    axis value at least once, the q > kv config paired with causal=True."""
    rows = []
    ax = HOLISTIC_AXES
    for idx in range(2 * len(SEQ_LEN_CONFIGS)):
        rows.append(
            (
                idx % len(SEQ_LEN_CONFIGS),
                ax["page_block_size"][idx % 3],
                ax["num_kv_heads"][(idx // 3) % 2],
                ax["gqa_group_size"][(idx // 2) % 4],
                ax["head_dim"][(idx // 4) % 3],
                ax["v_scale"][(idx // 5) % 2],
                ax["causal"][(idx + 1) % 2],
                ax["layout"][(idx // 6) % 2],
                ax["test_dtype"][(idx // 7) % 2],
                ax["logits_soft_cap"][(idx // 9) % 2],
            )
        )
    return rows


HOLISTIC_DEFAULT = set(_holistic_default_rows())


def _legacy_batch_attention_fixture(
    kv_lens,
    qo_lens,
    *,
    page_block_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    layout,
    test_dtype,
    seed,
    is_chunked_q=False,
):
    """``_run_attention``'s fixture: q ~ U[0,1), combined randn pool, page
    ids ``arange(num_blocks)``."""
    torch.manual_seed(seed)
    dev = torch.device(DEVICE)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)
    seq_lens_blocks = (seq_lens + page_block_size - 1) // page_block_size
    q_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(seq_lens_blocks, 0, dtype=torch.int32),
        ]
    )
    num_blocks = int(kv_indptr[-1])
    total_q = int(q_indptr[-1])
    if is_chunked_q:
        q_base = torch.rand(
            total_q, num_qo_heads, head_dim * 2, dtype=test_dtype, device=dev
        )
        q = torch.chunk(q_base, 2, dim=-1)[0]
    else:
        q = torch.rand(total_q, num_qo_heads, head_dim, dtype=test_dtype, device=dev)
    if layout == "NHD":
        kv_data = torch.randn(
            num_blocks,
            2,
            page_block_size,
            num_kv_heads,
            head_dim,
            dtype=test_dtype,
            device=dev,
        )
    else:
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_block_size,
            head_dim,
            dtype=test_dtype,
            device=dev,
        )
    lb = LegacyBatch(
        q=q,
        k=kv_data[:, 0],
        v=kv_data[:, 1],
        q_indptr_cpu=q_indptr,
        kv_indptr_cpu=kv_indptr,
        kv_indices_cpu=torch.arange(num_blocks, dtype=torch.int32),
        last_page_len_cpu=(seq_lens - 1) % page_block_size + 1,
        page_size=page_block_size,
        kv_layout=layout,
    )
    return lb, kv_data


def _legacy_old_scheduler(lb, kv_data, *, causal, logits_soft_cap, v_scale):
    """The legacy reference of ``test_batch_attention``: the fa2
    ``BatchPrefillWithPagedKVCacheWrapper`` ("old scheduler")."""
    dev = torch.device(DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
        kv_layout=lb.kv_layout,
        backend="fa2",
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
        causal=causal,
        q_data_type=lb.dtype,
        kv_data_type=lb.dtype,
        logits_soft_cap=logits_soft_cap,
    )
    return wrapper.run(lb.q, kv_data, return_lse=True, v_scale=v_scale)


@pytest.mark.parametrize(
    "config,page_block_size,num_kv_heads,gqa_group_size,head_dim,v_scale,causal,layout,test_dtype,logits_soft_cap,backend",
    _param_rows(_grid(HOLISTIC_AXES), lambda p: tuple(p) in HOLISTIC_DEFAULT),
)
def test_legacy_batch_attention_correctness(
    config,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    v_scale,
    causal,
    layout,
    test_dtype,
    logits_soft_cap,
    backend,
):
    """``test_batch_attention_correctness``: the unified API on the legacy
    fixture vs the legacy fa2 wrapper (out and LSE at 1e-2) + chunked oracle.
    v_scale rows: the unified API takes KV scales for fp8 KV only, so the
    rejection is asserted and the migration adapter ``out * v_scale`` is
    compared instead (flip ``EXPECT_FLOAT_KV_SCALES`` when extended)."""
    pairs = SEQ_LEN_CONFIGS[config]
    kv_lens, qo_lens = [p[0] for p in pairs], [p[1] for p in pairs]
    dtype = _DT[test_dtype]
    lb, kv_data = _legacy_batch_attention_fixture(
        kv_lens,
        qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_kv_heads * gqa_group_size,
        head_dim=head_dim,
        layout=layout,
        test_dtype=dtype,
        seed=_seed(
            "holistic",
            config,
            page_block_size,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            causal,
            layout,
            test_dtype,
            logits_soft_cap,
        ),
    )
    md = lb.metadata()
    cap = None if logits_soft_cap == 0.0 else logits_soft_cap
    if (
        causal
        and bool((lb.q_lens_cpu > lb.kv_seq_lens_cpu).any())
        and not EXPECT_FULLY_MASKED_ROWS
    ):
        resolve_or_skip(lb, md, backend, causal=True, logits_soft_cap=cap)
        with pytest.raises(
            ValueError, match="causal masking requires q_len_i <= kv_len_i"
        ):
            PagedAttention(torch.device(DEVICE)).plan(
                md,
                **lb.plan_kwargs(),
                causal=True,
                lse_mode="base2",
                logits_soft_cap=cap,
                backend=backend,
            )
        pytest.skip(
            "causal q_len > kv_len rows (fully masked query rows) are rejected by the "
            "unified causal envelope; legacy returned out 0 / LSE -inf for them "
            "(features file: test_legacy_fully_masked_rows)"
        )
    attn = unified_plan(lb, md, backend, causal=causal, logits_soft_cap=cap)
    ref_out, ref_lse = _legacy_old_scheduler(
        lb, kv_data, causal=causal, logits_soft_cap=logits_soft_cap, v_scale=v_scale
    )
    if v_scale is not None and not EXPECT_FLOAT_KV_SCALES:
        with pytest.raises(ValueError, match="KV scales apply to fp8 KV caches only"):
            attn.run(lb.q, (lb.k, lb.v), v_scale=v_scale)
        out, lse = attn.run(lb.q, (lb.k, lb.v))
        out_scaled = (out.float() * v_scale).to(out.dtype)  # migration adapter
        torch.testing.assert_close(out_scaled, ref_out, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)
        assert_oracle(lb, out, lse, causal=causal, logits_soft_cap=cap)
        return
    out, lse = attn.run(lb.q, (lb.k, lb.v), v_scale=v_scale)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)  # legacy
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)  # legacy
    assert_oracle(lb, out, lse, causal=causal, logits_soft_cap=cap)


@pytest.mark.parametrize("backend", BACKENDS)
def test_legacy_batch_attention_noncontiguous_q(backend):
    """``test_batch_attention_with_noncontiguous_q``: q is the first half of a
    ``(n, 1, 2 * D)`` buffer (token and head stride 2D, unit inner stride) on
    the (146, 146) config, page 1, D64 bf16 NHD."""
    pairs = SEQ_LEN_CONFIGS[0]
    lb, kv_data = _legacy_batch_attention_fixture(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        page_block_size=1,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=64,
        layout="NHD",
        test_dtype=torch.bfloat16,
        seed=_seed("holistic-noncontig"),
        is_chunked_q=True,
    )
    assert not lb.q.is_contiguous() and lb.q.stride(-1) == 1
    md = lb.metadata()
    attn = unified_plan(lb, md, backend, causal=True)
    from flashinfer.experimental.paged_attention import CAPABILITIES

    if CAPABILITIES[attn.backend].requires_contiguous_q:
        with pytest.raises(ValueError, match="requires packed q"):
            attn.run(lb.q, (lb.k, lb.v))
        pytest.skip(f"[{attn.backend}] rejects the strided q view by contract")
    out, lse = attn.run(lb.q, (lb.k, lb.v))
    ref_out, ref_lse = _legacy_old_scheduler(
        lb, kv_data, causal=True, logits_soft_cap=0.0, v_scale=None
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)
    assert_oracle(lb, out, lse, causal=True)


# ---------------------------------------------------------------------------
# test_shared_prefix_kernels.py :: shared prefix (one-shot and composed)
# ---------------------------------------------------------------------------

SHARED_PREFIX_AXES = dict(
    stage=["decode", "append"],
    batch_size=[12, 17],
    unique_kv_len=[37, 17],
    shared_kv_len=[128, 512, 2048],
    num_heads=[8, 16],
    causal=[False],
    head_dim=[128, 256],
    page_size=[1, 16],
)


def _shared_default(p):
    stage, b, unique, shared, heads, causal, hd, page = p
    return (b, unique, shared, heads) in {
        (12, 37, 128, 8),
        (17, 17, 512, 16),
        (12, 17, 2048, 8),
    }


def _ceil_div(a, b):
    return (a + b - 1) // b


def _shared_prefix_fixture(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    head_dim,
    page_size,
    seed,
):
    """Verbatim legacy construction: q, k/v_shared, k/v_unique, and the pool
    filled with ``append_paged_kv_cache`` (shared pages first)."""
    torch.manual_seed(seed)
    dev = torch.device(DEVICE)
    kv_layout = "NHD"
    if stage == "append":
        q = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
        q_indptr = torch.arange(0, batch_size + 1).to(dev).int() * unique_kv_len
    else:
        q = torch.randn(batch_size, num_heads, head_dim).to(dev).half()
        q_indptr = torch.arange(0, batch_size + 1).to(dev).int()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(dev).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(dev).half()
    k_unique = (
        torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
    )
    v_unique = (
        torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
    )
    n_shared_pages = _ceil_div(shared_kv_len, page_size)
    n_unique_pages = _ceil_div(unique_kv_len, page_size)
    kv_data = (
        torch.zeros(
            n_shared_pages + batch_size * n_unique_pages,
            2,
            page_size,
            num_heads,
            head_dim,
        )
        .to(dev)
        .half()
    )
    shared_kv_indices = torch.arange(0, n_shared_pages).to(dev).int()
    shared_append_indptr = torch.arange(0, 2).to(dev).int() * shared_kv_len
    shared_kv_indptr = torch.arange(0, 2).to(dev).int() * n_shared_pages
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(dev)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            k_shared.shape[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )
    unique_kv_indices = (
        torch.arange(0, batch_size * n_unique_pages).to(dev).int() + n_shared_pages
    )
    unique_append_indptr = torch.arange(0, batch_size + 1).to(dev).int() * unique_kv_len
    unique_kv_indptr = torch.arange(0, batch_size + 1).to(dev).int() * n_unique_pages
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(dev)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            k_unique.shape[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )
    return dict(
        q=q,
        q_indptr=q_indptr,
        kv_data=kv_data,
        shared=(shared_kv_indptr, shared_kv_indices, shared_last_page_len),
        unique=(unique_kv_indptr, unique_kv_indices, unique_last_page_len),
        n_shared_pages=n_shared_pages,
        n_unique_pages=n_unique_pages,
        kv_layout=kv_layout,
    )


def _legacy_cascade_reference(
    f, *, batch_size, num_heads, head_dim, page_size, stage, causal
):
    """The legacy reference: ``MultiLevelCascadeAttentionWrapper`` (2 levels)."""
    dev = torch.device(DEVICE)
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(dev), f["kv_layout"]
    )
    qo_indptr_top = torch.tensor([0, f["q"].shape[0]], dtype=torch.int32).to(dev)
    kw = {} if stage == "decode" else dict(causal=causal)
    wrapper.plan(
        [qo_indptr_top, f["q_indptr"]],
        [f["shared"][0], f["unique"][0]],
        [f["shared"][1], f["unique"][1]],
        [f["shared"][2], f["unique"][2]],
        num_heads,
        num_heads,
        head_dim,
        page_size,
        **kw,
    )
    return wrapper.run(f["q"], f["kv_data"])


def _shared_prefix_batches(f, *, batch_size, shared_kv_len, unique_kv_len, page_size):
    """Three legacy-shaped batches over the SAME pool: the one-shot
    (shared pages ++ unique pages per request), the shared-only and the
    unique-only segment (for the composed check)."""
    n_s, n_u = f["n_shared_pages"], f["n_unique_pages"]
    q_indptr_cpu = f["q_indptr"].cpu()
    k, v = f["kv_data"][:, 0], f["kv_data"][:, 1]
    shared_ids = torch.arange(0, n_s, dtype=torch.int32)
    full_ids, full_indptr = [], [0]
    for i in range(batch_size):
        full_ids.append(shared_ids)
        full_ids.append(
            torch.arange(n_s + i * n_u, n_s + (i + 1) * n_u, dtype=torch.int32)
        )
        full_indptr.append(full_indptr[-1] + n_s + n_u)
    common = dict(
        q=f["q"],
        k=k,
        v=v,
        q_indptr_cpu=q_indptr_cpu,
        page_size=page_size,
        kv_layout=f["kv_layout"],
    )
    one_shot = LegacyBatch(
        kv_indptr_cpu=torch.tensor(full_indptr, dtype=torch.int32),
        kv_indices_cpu=torch.cat(full_ids),
        last_page_len_cpu=torch.full(
            (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        **common,
    )
    shared_only = LegacyBatch(
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * n_s,
        kv_indices_cpu=shared_ids.repeat(batch_size),
        last_page_len_cpu=torch.full(
            (batch_size,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        **common,
    )
    unique_only = LegacyBatch(
        kv_indptr_cpu=f["unique"][0].cpu(),
        kv_indices_cpu=f["unique"][1].cpu(),
        last_page_len_cpu=f["unique"][2].cpu(),
        **common,
    )
    return one_shot, shared_only, unique_only


@pytest.mark.parametrize(
    "stage,batch_size,unique_kv_len,shared_kv_len,num_heads,causal,head_dim,page_size,backend",
    _param_rows(_grid(SHARED_PREFIX_AXES), _shared_default),
)
def test_legacy_shared_prefix_one_shot(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
    backend,
):
    """``test_batch_attention_with_shared_prefix_paged_kv_cache`` as ONE paged
    attention over ``shared pages ++ unique pages`` per request (the shared
    pages are the same physical pages for every request) vs the legacy
    ``MultiLevelCascadeAttentionWrapper`` at 1e-3 + oracle."""
    f = _shared_prefix_fixture(
        stage,
        batch_size,
        unique_kv_len,
        shared_kv_len,
        num_heads,
        head_dim,
        page_size,
        seed=_seed(
            "shared",
            stage,
            batch_size,
            unique_kv_len,
            shared_kv_len,
            num_heads,
            head_dim,
            page_size,
        ),
    )
    one_shot, _, _ = _shared_prefix_batches(
        f,
        batch_size=batch_size,
        shared_kv_len=shared_kv_len,
        unique_kv_len=unique_kv_len,
        page_size=page_size,
    )
    md = one_shot.metadata()
    _, out, lse = unified_run(one_shot, md, backend, causal=causal)
    ref = _legacy_cascade_reference(
        f,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        stage=stage,
        causal=causal,
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(one_shot, out, lse, causal=causal)


@pytest.mark.parametrize(
    "stage,batch_size,unique_kv_len,shared_kv_len,num_heads,causal,head_dim,page_size,backend",
    _param_rows(_grid(SHARED_PREFIX_AXES), _shared_default),
)
def test_legacy_shared_prefix_composed_merge(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
    backend,
):
    """The legacy two-level path composed from the unified API: one
    ``PagedAttention`` run over the shared pages, one over the unique pages
    (both ``lse_mode="base2"``, the unit ``merge_state`` consumes), merged
    with ``flashinfer.merge_state``; vs the cascade wrapper at 1e-3 and the
    one-shot oracle.  The merge is an external composition."""
    f = _shared_prefix_fixture(
        stage,
        batch_size,
        unique_kv_len,
        shared_kv_len,
        num_heads,
        head_dim,
        page_size,
        seed=_seed(
            "shared",
            stage,
            batch_size,
            unique_kv_len,
            shared_kv_len,
            num_heads,
            head_dim,
            page_size,
        ),
    )
    one_shot, shared_only, unique_only = _shared_prefix_batches(
        f,
        batch_size=batch_size,
        shared_kv_len=shared_kv_len,
        unique_kv_len=unique_kv_len,
        page_size=page_size,
    )
    _, o_s, s_s = unified_run(
        shared_only, shared_only.metadata(), backend, causal=False
    )
    _, o_u, s_u = unified_run(
        unique_only, unique_only.metadata(), backend, causal=causal
    )
    out, lse = flashinfer.merge_state(o_s, s_s, o_u, s_u)
    ref = _legacy_cascade_reference(
        f,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        stage=stage,
        causal=causal,
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(one_shot, out, lse, causal=causal)


def test_legacy_merge_state_in_place_is_native_only():
    """``test_merge_state_in_place_with_mask`` is a merge-operator contract:
    the unified API has no merge entry point (composition uses the cascade
    operators, see the composed test above) and the legacy operator stays."""
    assert not hasattr(PagedAttention, "merge")
    assert callable(flashinfer.merge_state_in_place) and callable(
        flashinfer.merge_state
    )
    import inspect

    params = inspect.signature(PagedAttention.run).parameters
    assert "lse" in params and "sinks" in params and "merge" not in params
