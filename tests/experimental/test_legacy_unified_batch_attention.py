"""Legacy -> unified: tests/attention/test_batch_attention.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the
``_run_attention`` construction (q ~ U[0, 1), combined randn pool as
``K = kv[:, 0]`` / ``V = kv[:, 1]`` views, page ids ``arange``), the ten
legacy sequence-length configs including the numpy-seeded random 256-request
batch, the legacy CSR mapped losslessly (page 1 -> ``.csr``, page 8 / 16 ->
``.dense``).  The legacy subject was the holistic ``BatchAttention``
scheduler compared with the fa2 ``BatchPrefillWithPagedKVCacheWrapper`` ("old
scheduler") at rtol/atol 1e-2 for output and LSE; the unified rows compare
``PagedAttention`` with that same legacy reference at the legacy tolerance
and with the fp32 oracle (query-row chunked for the 8190 x 7939 config).
The parametrize axes keep the legacy names and values (``seq_len_pairs`` and
``test_dtype`` ids are pytest's ``<name><index>``); the row id is the legacy
node id plus ``-<backend>``; the full 23040-point grid is under ``slow``.

CI: legacy file only in the H100 1/5 sampling lane; its three functions are
xfail on SM120.  The unified file is not collected by default CI.

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- the holistic ``BatchAttention`` scheduler itself is not a unified backend
  (the unified API dispatches to fa2 / fa3 / cuDNN / trtllm-gen / cake); the
  numerics of every legacy point are covered, the scheduler is not.
- v_scale=2.0 rows: ``run(v_scale=)`` on a fp16 / bf16 cache is accepted
  since WP-T (EXPECT_FLOAT_KV_SCALES; the output is multiplied), so the rows
  are positive; the rejection branch is kept for the record.
- the "real workload" config ``[(2, 235), (1, 13353)]`` with causal=True has
  q_len > kv_len in every request: the unified causal envelope rejects it
  ("causal masking requires q_len_i <= kv_len_i") where the legacy kernels
  defined the fully masked rows as out 0 / LSE -inf.  Needs a fully-masked-row
  policy (EXPECT_FULLY_MASKED_ROWS); the rows assert the rejection and skip.
- logits_soft_cap=50 rows resolve on fa2 / fa3 only (cuDNN, trtllm-gen and
  cake declare no soft cap); page 1 rows resolve on the CSR-native backends
  only; D64 / D256 and page 8 skip on trtllm-gen / cake (declared D128 and
  pages 16..1024).
- NVFP4 (``test_batch_attention_nvfp4``): kv_dtype uint8 is not a declared KV
  dtype and run() has no kv_cache_sf (EXPECT_NVFP4_KV).
"""

import numpy as np
import pytest
import torch

from flashinfer.experimental.paged_attention import CAPABILITIES
from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FLOAT_KV_SCALES,
    EXPECT_FULLY_MASKED_ROWS,
    LegacyBatch,
    argnames,
    assert_nvfp4_unsupported,
    assert_oracle,
    backend_rows,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    grid,
    legacy_id,
    legacy_paged_wrapper,
    param_rows,
    plan_batch,
    resolve_batch_or_skip,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_batch_attention.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_with_noncontiguous_q",
        ["test_batch_attention_with_noncontiguous_q"],
        "partial",
        "same fixture ((146,146), page 1, D64 bf16 NHD, q = first chunk of a 2D-wide "
        "buffer); numerics vs the legacy fa2 wrapper at 1e-2 (out and LSE) + oracle; the "
        "holistic scheduler is not exercised; a backend that requires packed q rejects "
        "the view by contract (recorded).",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_correctness",
        ["test_batch_attention_correctness"],
        "partial",
        "same 10 legacy axes (10 seq configs incl. the numpy-seeded random 256-request "
        "batch, page1/8/16, Hkv1/4, group1/4/7/8, D64/128/256, v_scale 2.0/None, causal, "
        "HND/NHD, bf16/fp16, softcap 0/50); unified vs the legacy fa2 wrapper at 1e-2 "
        "(out and LSE) + chunked oracle; v_scale rows positive since WP-T "
        "(EXPECT_FLOAT_KV_SCALES); the causal q>kv config (kv2/q235 + kv1/q13353) is "
        "rejected by the causal envelope (EXPECT_FULLY_MASKED_ROWS, asserted + skip); "
        "softcap 50 rows resolve on fa2 only; the BatchAttention holistic scheduler "
        "itself is not a unified backend.",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_nvfp4",
        ["test_batch_attention_nvfp4"],
        "unsupported-by-design",
        "same 9 legacy axes (B1/4 x kv128/256 x q64/128 x page16/64 x H1:1 x D128 x "
        "non-causal x fp16/bf16 q); kv_dtype uint8 (packed FP4x2) is not a declared KV "
        "dtype and run() has no kv_cache_sf; EXPECT_NVFP4_KV.",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


# ---------------------------------------------------------------------------
# the legacy configuration generator and fixture (verbatim)
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
    seq_len_pairs=SEQ_LEN_CONFIGS,
    page_block_size=[1, 8, 16],
    num_kv_heads=[1, 4],
    gqa_group_size=[1, 4, 7, 8],
    head_dim=[64, 128, 256],
    v_scale=[2.0, None],
    causal=[False, True],
    layout=["HND", "NHD"],
    test_dtype=[torch.bfloat16, torch.float16],
    logits_soft_cap=[0.0, 50.0],
)


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


def _holistic_default(point):
    return (SEQ_LEN_CONFIGS.index(point[0]), *point[1:]) in HOLISTIC_DEFAULT


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
    ids ``arange(num_blocks)``.  Returns the batch and the combined pool the
    legacy wrapper consumes."""
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
    wrapper = legacy_paged_wrapper(
        lb, backend="fa2", causal=causal, logits_soft_cap=logits_soft_cap
    )
    return wrapper.run(lb.q, kv_data, return_lse=True, v_scale=v_scale)


# ---------------------------------------------------------------------------
# test_batch_attention_with_noncontiguous_q
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", backend_rows())
def test_batch_attention_with_noncontiguous_q(backend):
    """q is the first half of a ``(n, 1, 2 * D)`` buffer (token and head
    stride 2D, unit inner stride) on the (146, 146) config, page 1, D64 bf16
    NHD, causal; vs the legacy fa2 wrapper at 1e-2 + oracle."""
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
        seed=seed_of("holistic-noncontig"),
        is_chunked_q=True,
    )
    assert not lb.q.is_contiguous() and lb.q.stride(-1) == 1
    md = lb.metadata()
    attn = plan_batch(lb, md, backend, causal=True)
    if CAPABILITIES[attn.backend].requires_contiguous_q:
        with pytest.raises(ValueError, match="requires packed q"):
            attn.run(lb.q, (lb.k, lb.v))
        pytest.skip(f"[{attn.backend}] rejects the strided q view by contract")
    out, lse = attn.run(lb.q, (lb.k, lb.v))
    ref_out, ref_lse = _legacy_old_scheduler(
        lb, kv_data, causal=True, logits_soft_cap=0.0, v_scale=None
    )
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)  # legacy
    torch.testing.assert_close(lse, ref_lse, rtol=1e-2, atol=1e-2)  # legacy
    assert_oracle(lb, out, lse, causal=True)


# ---------------------------------------------------------------------------
# test_batch_attention_correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    argnames(HOLISTIC_AXES, "backend"), param_rows(HOLISTIC_AXES, _holistic_default)
)
def test_batch_attention_correctness(
    seq_len_pairs,
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
    """The unified API on the legacy fixture vs the legacy fa2 wrapper (out
    and LSE at 1e-2) + chunked oracle.  v_scale rows run ``run(v_scale=)``
    (positive branch of EXPECT_FLOAT_KV_SCALES); the causal q > kv config
    asserts the envelope rejection and skips (EXPECT_FULLY_MASKED_ROWS)."""
    kv_lens, qo_lens = [p[0] for p in seq_len_pairs], [p[1] for p in seq_len_pairs]
    config = SEQ_LEN_CONFIGS.index(seq_len_pairs)
    lb, kv_data = _legacy_batch_attention_fixture(
        kv_lens,
        qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_kv_heads * gqa_group_size,
        head_dim=head_dim,
        layout=layout,
        test_dtype=test_dtype,
        seed=seed_of(
            "holistic",
            config,
            page_block_size,
            num_kv_heads,
            gqa_group_size,
            head_dim,
            causal,
            layout,
            str(test_dtype),
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
        resolve_batch_or_skip(lb, md, backend, causal=True, logits_soft_cap=cap)
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
            "(EXPECT_FULLY_MASKED_ROWS)"
        )
    attn = plan_batch(lb, md, backend, causal=causal, logits_soft_cap=cap)
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
    # the oracle sees the unscaled V; v_scale multiplies the output (exactly,
    # in fp32) so undo it before the independent check
    out_for_oracle = (out.float() / v_scale).to(out.dtype) if v_scale else out
    assert_oracle(lb, out_for_oracle, lse, causal=causal, logits_soft_cap=cap)


# ---------------------------------------------------------------------------
# test_batch_attention_nvfp4
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
def test_batch_attention_nvfp4(
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
    """BatchAttention with an NVFP4 KV cache: the packed uint8 KV dtype is
    undeclared on every unified backend and run() has no kv_cache_sf."""
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
        what="batch_attention_nvfp4",
    )
