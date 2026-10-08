"""Legacy -> unified: tests/attention/test_batch_attention.py

The legacy tests compare the holistic ``BatchAttention`` scheduler (fa2-family
kernels on this hardware) with the fa2 ``BatchPrefillWithPagedKVCacheWrapper``
("old scheduler").  Here ``PagedAttention`` pinned to fa2 takes the place of
``BatchAttention`` on the legacy fixture: same grid, same tensors (the legacy
seeds only at collection, in ``_build_seq_len_configs``), so the node ids
equal the legacy ids.  Each case checks the output and LSE against the legacy
fa2 wrapper at the legacy 1e-2 tolerance, and against the fp32 oracle.

Two kinds of legacy case the unified API cannot express assert the rejection:
the causal "real workload" config (q_len > kv_len, fully masked query rows,
outside the causal envelope) and the NVFP4 KV cache (uint8 is not a KV dtype).
The legacy SM120 xfail (BatchAttention's tile size) is not carried over.
"""

import math

import numpy as np
import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_attention.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_with_noncontiguous_q",
        ["test_batch_attention_with_noncontiguous_q"],
        "equivalent",
        "same fixture (q a half-width view) on fa2 in place of BatchAttention; legacy "
        "fa2 wrapper at 1e-2 (output and LSE) plus the fp32 oracle",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_correctness",
        ["test_batch_attention_correctness"],
        "partial",
        "same grid and tensors on fa2 in place of BatchAttention; legacy fa2 wrapper at "
        "1e-2 (output and LSE) plus the fp32 oracle; the causal q_len > kv_len config "
        "(fully masked rows) asserts the causal-envelope rejection",
    ),
    (
        "tests/attention/test_batch_attention.py::test_batch_attention_nvfp4",
        ["test_batch_attention_nvfp4"],
        "unsupported-by-design",
        "NVFP4 KV (packed uint8 + scale factors) is not a unified KV dtype; asserts the "
        "fa2 rejection",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _build_seq_len_configs():
    """Verbatim from the legacy file (seeds numpy and torch with 42)."""
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
        seq_len.append((kv_len, qo_len))
    seq_len_configs.append(seq_len)

    return seq_len_configs


def _oracle(q, k_cache, v_cache, kv_lens, qo_lens, page_size, causal, layout, cap):
    """``reference_paged_prefill`` with each request cut into chunks of at most
    1024 query rows (a causal chunk's KV ends at its last row's bottom-right
    position), so the 8190 x 7939 config's fp32 scores stay small.  The legacy
    page ids are ``arange``: request i owns a contiguous run of pages."""
    q_lens, chunk_kv_lens, page_ids = [], [], []
    first_page = 0
    for lq, lkv in zip(qo_lens, kv_lens, strict=True):
        for a in range(0, lq, 1024):
            b = min(a + 1024, lq)
            kv_b = lkv - lq + b if causal else lkv
            q_lens.append(b - a)
            chunk_kv_lens.append(kv_b)
            page_ids.append(
                torch.arange(first_page, first_page + math.ceil(kv_b / page_size))
            )
        first_page += math.ceil(lkv / page_size)
    return reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        torch.tensor([0, *np.cumsum(q_lens)], dtype=torch.int32),
        torch.tensor(chunk_kv_lens, dtype=torch.int32),
        None,
        page_size,
        causal,
        kv_layout=layout,
        kv_page_indices=torch.cat(page_ids).to(q.device, torch.int32),
        logits_soft_cap=cap,
    )


def _run_attention(
    kv_lens,
    qo_lens,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    v_scale=None,
    layout="NHD",
    test_dtype=torch.bfloat16,
    logits_soft_cap=0.0,
    device="cuda",
    causal=True,
    is_chunked_q=False,
):
    """The legacy ``_run_attention`` with ``PagedAttention`` (fa2) in place of
    ``BatchAttention``: the legacy fixture, the legacy fa2 wrapper as the
    reference at the legacy tolerance, then the fp32 oracle."""
    # the legacy fixture, verbatim (same RNG order: q, kv_data)
    dev = torch.device(device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32, device=dev)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat(
        [torch.tensor([0], device=dev), torch.cumsum(q_lens, 0)], dim=0
    ).int()
    kv_indptr = torch.cat(
        [torch.tensor([0], device=dev), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    if is_chunked_q:
        q_base = torch.rand(
            q_indptr[-1].item(),
            num_qo_heads,
            head_dim * 2,
            dtype=test_dtype,
            device=dev,
        )
        q = torch.chunk(q_base, 2, dim=-1)[0]
    else:
        q = torch.rand(
            q_indptr[-1].item(), num_qo_heads, head_dim, dtype=test_dtype, device=dev
        )
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
    elif layout == "HND":
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_block_size,
            head_dim,
            dtype=test_dtype,
            device=dev,
        )

    # unified: the legacy CSR as-is (kv_page_indices = arange), K / V as the
    # two plane views of the combined pool
    md = PagedAttentionMetadata.csr(
        q_indptr,
        seq_lens,
        torch.arange(num_blocks, device=dev).int(),
        page_size=page_block_size,
        max_q_len=max(qo_lens),
        max_kv_len=int(max(kv_lens)),
        qo_indptr_cpu=q_indptr.cpu(),
        kv_seq_lens_cpu=seq_lens.cpu(),
    )
    attn = PagedAttention(dev)
    plan_kwargs = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=test_dtype,
        kv_layout=layout,
        causal=causal,
        lse_mode="base2",
        logits_soft_cap=logits_soft_cap or None,
        backend="fa2",
    )
    if causal and any(lq > lkv for lq, lkv in zip(qo_lens, kv_lens, strict=True)):
        # fully masked query rows: legacy returns out 0 / LSE -inf for them,
        # the unified causal envelope rejects the batch
        with pytest.raises(
            ValueError, match="causal masking requires q_len_i <= kv_len_i"
        ):
            attn.plan(md, **plan_kwargs)
        return
    attn.plan(md, **plan_kwargs)
    assert attn.backend == "fa2"
    out, lse = attn.run(q, (kv_data[:, 0], kv_data[:, 1]), v_scale=v_scale)

    # legacy reference: the fa2 wrapper ("old scheduler") at the legacy tolerance
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
        kv_layout=layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr,
        kv_indptr,
        torch.arange(num_blocks, device=dev).int(),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_old, lse_old = wrapper_old.run(q, kv_data, return_lse=True, v_scale=v_scale)
    torch.testing.assert_close(out_old, out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse_old, lse, rtol=1e-2, atol=1e-2)

    # fp32 oracle: output (v_scale multiplies it) and base-2 LSE
    ref_out, ref_lse = _oracle(
        q,
        kv_data[:, 0],
        kv_data[:, 1],
        [int(n) for n in kv_lens],
        list(qo_lens),
        page_block_size,
        causal,
        layout,
        logits_soft_cap or None,
    )
    if v_scale is not None:
        ref_out = ref_out * v_scale
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_batch_attention_with_noncontiguous_q():
    # Pick the first sequence length config's first pair
    seq_len_pairs = _build_seq_len_configs()[0]
    kv_lens = [p[0] for p in seq_len_pairs]
    qo_lens = [p[1] for p in seq_len_pairs]

    _run_attention(
        kv_lens=kv_lens,
        qo_lens=qo_lens,
        page_block_size=1,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=64,
        v_scale=None,
        causal=True,
        layout="NHD",
        test_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
        device="cuda",
        is_chunked_q=True,
    )


@pytest.mark.parametrize("seq_len_pairs", _build_seq_len_configs())
@pytest.mark.parametrize("page_block_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 7, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("v_scale", [2.0, None])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
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
):
    num_qo_heads = num_kv_heads * gqa_group_size
    kv_lens = [p[0] for p in seq_len_pairs]
    qo_lens = [p[1] for p in seq_len_pairs]

    _run_attention(
        kv_lens=kv_lens,
        qo_lens=qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        v_scale=v_scale,
        causal=causal,
        layout=layout,
        test_dtype=test_dtype,
        logits_soft_cap=logits_soft_cap,
        device="cuda",
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
    """NVFP4 KV (packed FP4x2 uint8 pages plus FP8 scale factors) is not a
    unified KV dtype: plan() rejects it."""
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    device = torch.device("cuda:0")
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    qo_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len
    kv_lens_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)
    md = PagedAttentionMetadata.dense(
        qo_indptr_cpu.to(device),
        kv_lens_cpu.to(device),
        torch.arange(
            batch_size * num_pages_per_seq, dtype=torch.int32, device=device
        ).reshape(batch_size, num_pages_per_seq),
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    with pytest.raises(ValueError, match="fa2: unsupported kv dtype torch.uint8"):
        PagedAttention(device).plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=q_dtype,
            kv_dtype=torch.uint8,
            kv_layout="NHD",
            causal=causal,
            backend="fa2",
        )
