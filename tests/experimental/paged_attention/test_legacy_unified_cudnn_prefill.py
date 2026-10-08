"""Legacy -> unified: tests/attention/test_cudnn_prefill.py

``test_cudnn_prefill`` runs the legacy fixture through ``PagedAttention``
pinned to cudnn, the backend the legacy wrapper is built with.  Same grid,
same seed, same tensors (the ``as_strided`` combined pool handed over as the
same K/V views, the legacy block table), so the node ids equal the legacy
ids.  Each case checks the legacy reference (the fa2 wrapper on the combined
pool) at the legacy tolerance, and the output against the fp32 paged-attention
oracle; the legacy ``return_lse`` axis (unused by the legacy body) plans the
LSE, which is checked against the oracle too.

``test_cudnn_prefill_fp8`` quantizes the QUERY to e4m3 with a ``q_scale``
tensor and asks for a bf16 output.  ``PagedAttention`` admits fp16 / bf16 q
only (fp8 is a KV-only axis), so each case asserts the cudnn plan rejection.
"""

import cudnn
import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import get_compute_capability

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cudnn_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill",
        ["test_cudnn_prefill"],
        "equivalent",
        "same grid, seed and tensors (as_strided pool views, legacy block table) on "
        "cudnn; legacy fa2 reference at the legacy tolerance plus the fp32 oracle "
        "(output, and LSE when return_lse)",
    ),
    (
        "tests/attention/test_cudnn_prefill.py::test_cudnn_prefill_fp8",
        ["test_cudnn_prefill_fp8"],
        "unsupported-by-design",
        "fp8 q with q_scale / o_data_type has no unified spelling (q is fp16/bf16 "
        "only); each legacy id asserts the cudnn plan rejection",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    # the legacy fixture, verbatim
    seed = 1
    torch.manual_seed(seed)
    device = "cuda:0"
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size, 1, 1, 1), dtype=torch.int32, device=device
    )
    cumsum_s_qo = torch.sum(actual_seq_lens_q)
    q = torch.randn(
        cumsum_s_qo, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache_shape = (total_num_pages, 2, num_kv_heads, page_size, head_dim)
    kv_cache = torch.randn(size=kv_cache_shape, dtype=torch.bfloat16).to(device)
    kv_cache = kv_cache.as_strided(
        kv_cache.shape,
        (
            2 * page_size * num_kv_heads * head_dim,
            page_size * num_kv_heads * head_dim,
            head_dim,
            num_kv_heads * head_dim,
            1,
        ),
    )
    k_cache_view = kv_cache[:, 0, :, :, :]
    v_cache_view = kv_cache[:, 1, :, :, :]
    v_cache = v_cache_view.as_strided(
        v_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    k_cache = k_cache_view.as_strided(
        k_cache_view.shape,
        (2 * page_size * num_kv_heads * head_dim, head_dim, num_kv_heads * head_dim, 1),
    )
    kv_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(
                (actual_seq_lens_kv.flatten() + page_size - 1) // page_size,
                dim=0,
            ),
        ]
    ).int()
    kv_indices = torch.zeros(kv_indptr[-1], device=device, dtype=torch.int32)
    for i in range(len(kv_indptr) - 1):
        start_idx = kv_indptr[i]
        end_idx = kv_indptr[i + 1]
        kv_indices[start_idx:end_idx] = torch.arange(
            i * num_pages_per_seq,
            i * num_pages_per_seq + (end_idx - start_idx),
            device=device,
        )
    kv_last_page_len = torch.where(
        actual_seq_lens_kv.flatten() % page_size == 0,
        torch.full((batch_size,), page_size, device=device),
        actual_seq_lens_kv.flatten() % page_size,
    ).int()
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int,
        device=device,
    )
    scale = float(1.0 / (head_dim**0.5))
    qo_indptr = torch.cat(
        [
            torch.tensor([0], device=device),
            torch.cumsum(actual_seq_lens_q.view(-1), dim=0),
        ]
    ).int()

    # unified: the legacy K/V views (no copy) and block table on cudnn
    qo_indptr_cpu = qo_indptr.cpu()
    kv_lens_cpu = actual_seq_lens_kv.view(-1).cpu()
    md = PagedAttentionMetadata.dense(
        qo_indptr,
        actual_seq_lens_kv.view(-1),
        block_tables,
        page_size=page_size,
        max_q_len=s_qo,
        max_kv_len=s_kv,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    attn = PagedAttention(torch.device(device))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="HND",
        causal=causal,
        lse_mode="base2" if return_lse else "none",
        backend="cudnn",
    )
    assert attn.backend == "cudnn"
    output, lse = attn.run(q, (k_cache, v_cache), sm_scale=scale)

    # legacy reference at the legacy tolerance: the fa2 wrapper on the pool
    workspace_buffer_ref = torch.empty(
        128 * 1024 * 1024, dtype=torch.int8, device=device
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer_ref, "HND", backend="fa2"
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        causal=causal,
        q_data_type=torch.bfloat16,
    )
    output_ref = wrapper.run(q, kv_cache)
    torch.testing.assert_close(output, output_ref, atol=3e-3, rtol=1e-2)

    # fp32 oracle: output, and the base-2 LSE when planned
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        kv_lens_cpu,
        block_tables,
        page_size,
        causal,
        sm_scale=scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(output.float(), ref_out, **OUT_TOL)
    if return_lse:
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    else:
        assert lse is None


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [8, 17, 700])
@pytest.mark.parametrize("s_kv", [8, 32, 1066])
@pytest.mark.parametrize("page_size", [8, 16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("is_cuda_graph_compatible", [True])
def test_cudnn_prefill_fp8(
    batch_size,
    s_qo,
    s_kv,
    page_size,
    num_kv_heads,
    num_qo_heads,
    causal,
    return_lse,
    is_cuda_graph_compatible,
):
    if cudnn.backend_version() < 91701:
        pytest.skip("cuDNN backend version is less than 9.17.1, skipping test")

    head_dim = 128
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test")

    device = "cuda:0"
    major, _ = get_compute_capability(torch.device(device))
    if major != 10:
        pytest.skip(
            f"cuDNN FP8 prefill is not supported on compute capability {major}, skipping test"
        )

    # the legacy shapes (seed 1); fp8 q is rejected at plan, before any data
    torch.manual_seed(1)
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    qo_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=device), actual_seq_lens_q.cumsum(0)]
    ).int()
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    block_tables = torch.arange(
        batch_size * num_pages_per_seq, dtype=torch.int32, device=device
    ).view(batch_size, num_pages_per_seq)
    md = PagedAttentionMetadata.dense(
        qo_indptr,
        actual_seq_lens_kv,
        block_tables,
        page_size=page_size,
        max_q_len=s_qo,
        max_kv_len=s_kv,
        qo_indptr_cpu=qo_indptr.cpu(),
        kv_seq_lens_cpu=actual_seq_lens_kv.cpu(),
    )
    attn = PagedAttention(torch.device(device))
    with pytest.raises(ValueError, match="cudnn: unsupported q dtype"):
        attn.plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            kv_layout="HND",
            causal=causal,
            lse_mode="base2" if return_lse else "none",
            backend="cudnn",
        )
