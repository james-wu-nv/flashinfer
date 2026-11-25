"""Legacy -> unified: tests/attention/test_fp8_prefill.py

The two paged-prefill tests run the legacy fixture through ``PagedAttention``
on fa2, fa3, cudnn and trtllm-gen; node ids are the legacy ids with a
``-<backend>`` suffix, and a backend that cannot run a case on this GPU is
skipped with the resolver's reason.  The legacy tests pin fa2.  The combined
``(pages, 2, ...)`` pool is passed as the ``kv[:, 0]`` / ``kv[:, 1]`` views and
the legacy page indices as ``PagedAttentionMetadata.csr``; like the legacy
plans the runs are non-causal, and like the legacy runs they request no LSE.

``test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale`` compares the
fp8-KV run (calibration scales) with the same backend on the fp16 pool, as the
legacy test does; that comparison is relative, so the fp8 output is also
checked against the fp32 oracle on the dequantized pool.

``test_paged_fp8_h512_long_q_keeps_cta32`` keeps the legacy tile check on fa2:
``_plan_info[CTA_TILE_Q]`` of the wrapper the unified fa2 backend holds (a
private attribute; ``PagedAttention`` exposes no plan info).

The decode, ragged and single-prefill tests are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_fp8_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale",
        ["test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale"],
        "equivalent",
        "same grid, seed and tensors on fa2 / fa3 / cudnn / trtllm-gen (non-causal, "
        "no LSE, like the legacy run); the fp8 run against the same backend on the "
        "fp16 pool at the legacy tolerance, plus the fp32 oracle on the "
        "dequantized pool",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_prefill_with_ragged_kv_cache_fp8",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_batch_decode_with_prefill_with_paged_kv_cache",
        [],
        "out-of-scope",
        "decode test (q_len 1 against the decode wrapper); out of the prefill scope for now",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_ragged_fp8_h512_long_q_keeps_cta32",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_paged_fp8_h512_long_q_keeps_cta32",
        ["test_paged_fp8_h512_long_q_keeps_cta32"],
        "equivalent",
        "same fixture on fa2 / fa3 / cudnn / trtllm-gen; the legacy CTA_TILE_Q == 32 "
        "check on fa2 (read from the wrapper the fa2 backend holds) and the legacy "
        "finite-output assertion, plus the fp32 oracle",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_single_prefill_fp8_h512_long_q",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill",
    ),
    (
        "tests/attention/test_fp8_prefill.py::test_ragged_fp8_calibration_scales",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)

_PLAN_INFO_CTA_TILE_Q_IDX = 3  # PrefillPlanInfo::ToVector layout (scheduler.cuh)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("qo_len", [1, 7, 53])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_batch_prefill_with_paged_kv_cache_fp8_calibration_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    dtype,
    backend,
):
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_dtype=dtype,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=False,
            need_lse=False,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    torch.manual_seed(42)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        0.05
        * torch.randn(
            total_num_pages, 2, num_kv_heads, page_size, head_dim, dtype=torch.float16
        ).to(0)
        if kv_layout == "HND"
        else 0.05
        * torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        ).to(0)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        kv_seq_lens,
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    attn_f16 = PagedAttention(workspace_buffer)
    attn_f16.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float16,
        kv_layout=kv_layout,
        causal=False,
        backend=backend,
    )
    o_fp16, _ = attn_f16.run(q, (kv_data[:, 0], kv_data[:, 1]))
    k_data, v_data = torch.chunk(kv_data, 2, dim=1)
    k_scale = k_data.amax().item() / 256
    v_scale = v_data.amax().item() / 256

    k_fp8 = (k_data / k_scale).to(dtype)
    v_fp8 = (v_data / v_scale).to(dtype)
    kv_data_fp8 = torch.cat([k_fp8, v_fp8], dim=1)

    attn_f8 = PagedAttention(workspace_buffer)
    attn_f8.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=dtype,
        kv_layout=kv_layout,
        causal=False,
        backend=backend,
    )
    assert attn_f8.backend == backend
    o_fp8, _ = attn_f8.run(
        q,
        (kv_data_fp8[:, 0], kv_data_fp8[:, 1]),
        k_scale=k_scale,
        v_scale=v_scale,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=1e-2, rtol=2e-1)

    # fp32 oracle on the dequantized pool (the legacy check compares two runs)
    ref_out, _ = reference_paged_prefill(
        q,
        k_fp8[:, 0].float() * k_scale,
        v_fp8[:, 0].float() * v_scale,
        qo_indptr.cpu(),
        kv_seq_lens.cpu(),
        None,
        page_size,
        False,
        kv_layout=kv_layout,
        kv_page_indices=kv_indices,
    )
    torch.testing.assert_close(o_fp8.float(), ref_out, **OUT_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_paged_fp8_h512_long_q_keeps_cta32(backend):
    head_dim = 512
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_dtype=torch.float8_e4m3fn,
            page_size=16,
            kv_layout="NHD",
            causal=False,
            need_lse=False,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")
    torch.manual_seed(42)
    batch_size = 2
    qo_len = kv_len = 128
    page_size = 16
    num_qo_heads = num_kv_heads = 4
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16
    ).to(0)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data_fp8 = (
        torch.randn(
            total_num_pages, 2, page_size, num_kv_heads, head_dim, dtype=torch.float16
        )
        .to(0)
        .to(torch.float8_e4m3fn)
    )
    qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_seq_lens = torch.full((batch_size,), kv_len, dtype=torch.int32).to(0)
    metadata = PagedAttentionMetadata.csr(
        qo_indptr,
        kv_seq_lens,
        kv_indices,
        page_size=page_size,
        max_q_len=qo_len,
        max_kv_len=kv_len,
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    attn = PagedAttention(workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.float16,
        kv_dtype=torch.float8_e4m3fn,
        kv_layout="NHD",
        causal=False,
        backend=backend,
    )
    assert attn.backend == backend
    if backend == "fa2":
        # fa2's own tile choice, on the wrapper the fa2 backend holds
        plan_info = attn._impl._active._active._plan_info
        assert plan_info[_PLAN_INFO_CTA_TILE_Q_IDX] == 32
    o, _ = attn.run(q, (kv_data_fp8[:, 0], kv_data_fp8[:, 1]))
    assert torch.isfinite(o).all()

    # fp32 oracle (the legacy test checks only that the output is finite)
    ref_out, _ = reference_paged_prefill(
        q,
        kv_data_fp8[:, 0].float(),
        kv_data_fp8[:, 1].float(),
        qo_indptr.cpu(),
        kv_seq_lens.cpu(),
        None,
        page_size,
        False,
        kv_layout="NHD",
        kv_page_indices=kv_indices,
    )
    torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
