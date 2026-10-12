"""Legacy -> unified: tests/attention/test_batch_prefill.py

Both legacy tests check that ``k_scale`` / ``v_scale`` reach the kernel.  The
unified tests make the same runs through ``PagedAttention`` on the legacy
fixture (same seed, same tensors) and keep the legacy assertions, once per
backend: fa2, fa3, cudnn and trtllm-gen.  A backend that cannot run the
fixture on this GPU is skipped with the resolver's reason.  The legacy
wrapper's ``backend="auto"`` runs fa2 (fa3 on SM90a), so that row runs the
legacy kernel.

The legacy tests plan ``causal=True`` but run through the deprecated
``forward_return_lse``, which resets ``causal`` to its default ``False``
before running; the unified rows therefore plan ``causal=False``.  They use
the legacy 16 MiB workspace and a base-2 LSE, as ``forward_return_lse`` does.

The legacy assertions only compare runs with each other, so every run is also
checked against the fp32 oracle.
"""

import math

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
LEGACY_SOURCE = "tests/attention/test_batch_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_effect",
        ["test_kv_scale_forwarding_effect"],
        "equivalent",
        "same fixture on fa2 / fa3 / cudnn / trtllm-gen, non-causal like the "
        "legacy forward_return_lse run; the legacy assertion plus the fp32 "
        "oracle on both scaled runs",
    ),
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_math_property",
        ["test_kv_scale_forwarding_math_property"],
        "equivalent",
        "same fixture on fa2 / fa3 / cudnn / trtllm-gen, non-causal like the "
        "legacy forward_return_lse run; the three legacy identities at rtol "
        "1e-2 / atol 1e-3 plus the fp32 oracle",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_effect(dtype, backend):
    torch.manual_seed(42)

    H_QO, H_KV, N_CTX, HEAD_DIM, PAGE_SIZE = 1, 1, 8, 64, 16
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    try:
        resolve_paged_attention(
            device=torch.device("cuda"),
            num_qo_heads=H_QO,
            num_kv_heads=H_KV,
            head_dim_qk=HEAD_DIM,
            q_dtype=dtype,
            page_size=PAGE_SIZE,
            kv_layout="NHD",
            causal=False,
            need_lse=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    # Create paged KV cache
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    paged_kv_cache = (k_cache, v_cache)

    # Create query tensor and metadata (one request on pages 0 .. max_num_pages-1)
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    kv_seq_lens = torch.tensor([N_CTX], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")[None]
    metadata = PagedAttentionMetadata.dense(
        qo_indptr,
        kv_seq_lens,
        block_tables,
        page_size=PAGE_SIZE,
        max_q_len=N_CTX,
        max_kv_len=N_CTX,
    )

    workspace_buffer = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    attn = PagedAttention(q.device, workspace_buffer=workspace_buffer)
    attn.plan(
        metadata,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=False,
        lse_mode="base2",
        backend=backend,
    )
    assert attn.backend == backend

    out1, lse1 = attn.run(q, paged_kv_cache, k_scale=0.1, v_scale=0.1)
    out2, lse2 = attn.run(q, paged_kv_cache, k_scale=2.0, v_scale=2.0)

    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different. "
        "This may indicate that the arguments are not passed correctly."
    )

    # fp32 oracle: k_scale folds into the softmax scale, v_scale multiplies
    # the output.
    for out, lse, scale in ((out1, lse1, 0.1), (out2, lse2, 2.0)):
        ref_out, ref_lse = reference_paged_prefill(
            q,
            k_cache,
            v_cache,
            qo_indptr.cpu(),
            kv_seq_lens.cpu(),
            block_tables,
            PAGE_SIZE,
            False,  # causal
            sm_scale=scale / math.sqrt(HEAD_DIM),
            kv_layout="NHD",
        )
        torch.testing.assert_close(out.float(), ref_out * scale, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_math_property(dtype: torch.dtype, backend):
    torch.manual_seed(0)

    # ---------------- parameters ----------------
    N_CTX, PAGE_SIZE = 128, 16
    H_QO, H_KV, HEAD_DIM = 1, 1, 64  # Explicitly specify H_QO
    max_num_pages = (N_CTX + PAGE_SIZE - 1) // PAGE_SIZE

    try:
        resolve_paged_attention(
            device=torch.device("cuda"),
            num_qo_heads=H_QO,
            num_kv_heads=H_KV,
            head_dim_qk=HEAD_DIM,
            q_dtype=dtype,
            page_size=PAGE_SIZE,
            kv_layout="NHD",
            causal=False,
            need_lse=True,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")

    # ---------------- paged KV cache ----------------
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)
    paged_kv_cache = (k_cache, v_cache)

    # ---------------- query and metadata ----------------
    q = torch.randn(N_CTX, H_QO, HEAD_DIM, dtype=dtype, device="cuda")
    qo_indptr = torch.tensor([0, N_CTX], dtype=torch.int32, device="cuda")
    kv_seq_lens = torch.tensor([N_CTX], dtype=torch.int32, device="cuda")
    block_tables = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")[None]
    metadata = PagedAttentionMetadata.dense(
        qo_indptr,
        kv_seq_lens,
        block_tables,
        page_size=PAGE_SIZE,
        max_q_len=N_CTX,
        max_kv_len=N_CTX,
    )

    # ---------------- PagedAttention ----------------
    workspace = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    attn = PagedAttention(q.device, workspace_buffer=workspace)
    attn.plan(
        metadata,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=False,
        lse_mode="base2",
        backend=backend,
    )
    assert attn.backend == backend

    # ---------------- scale factors ----------------
    k_scale = 0.5
    v_scale = 2.0

    # -------- case 1: k_scale only ----------
    out1, lse1 = attn.run(q, paged_kv_cache, k_scale=k_scale)
    out1_ref, _ = attn.run(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-2, atol=1e-3)

    # -------- case 2: v_scale only ----------
    out2, lse2 = attn.run(q, paged_kv_cache, v_scale=v_scale)
    out2_ref, lse2_ref = attn.run(q, paged_kv_cache)
    torch.testing.assert_close(out2, out2_ref * v_scale, rtol=1e-2, atol=1e-3)

    # -------- case 3: both k_scale and v_scale ----------
    out3, lse3 = attn.run(q, paged_kv_cache, k_scale=k_scale, v_scale=v_scale)
    out3_ref, _ = attn.run(q * k_scale, paged_kv_cache)
    torch.testing.assert_close(out3, out3_ref * v_scale, rtol=1e-2, atol=1e-3)

    # -------- fp32 oracle ----------
    # k_scale folds into the softmax scale, v_scale multiplies the output
    for out, lse, k, v in (
        (out1, lse1, k_scale, 1.0),
        (out2, lse2, 1.0, v_scale),
        (out2_ref, lse2_ref, 1.0, 1.0),
        (out3, lse3, k_scale, v_scale),
    ):
        ref_out, ref_lse = reference_paged_prefill(
            q,
            k_cache,
            v_cache,
            qo_indptr.cpu(),
            kv_seq_lens.cpu(),
            block_tables,
            PAGE_SIZE,
            False,  # causal
            sm_scale=k / math.sqrt(HEAD_DIM),
            kv_layout="NHD",
        )
        torch.testing.assert_close(out.float(), ref_out * v, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
