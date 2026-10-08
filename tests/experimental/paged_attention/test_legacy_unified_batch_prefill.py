"""Legacy -> unified: tests/attention/test_batch_prefill.py

Both legacy tests run their fixture (one request, D64, page 16, two NHD pools)
through ``PagedAttention`` pinned to fa2, the backend the legacy wrapper picks
on this GPU.  Same grid, same seed, same tensors, so the node ids equal the
legacy ids.  ``k_scale`` / ``v_scale`` on a fp16 / bf16 cache are ``run()``
arguments; each case keeps the legacy assertion at the legacy tolerance and
checks every scaled output and LSE against the fp32 oracle (``k_scale`` folds
into the softmax scale, ``v_scale`` multiplies the output).

The legacy tests plan ``causal=True`` and then call the deprecated
``forward_return_lse``, which resets ``causal`` to False before running; the
unified rows keep the causal plan the legacy test declares.
"""

import math

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_effect",
        ["test_kv_scale_forwarding_effect"],
        "equivalent",
        "same fixture on fa2; legacy assertion (scales 0.1 vs 2.0 change the "
        "output) plus the fp32 oracle on both scaled runs (output and LSE)",
    ),
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_math_property",
        ["test_kv_scale_forwarding_math_property"],
        "equivalent",
        "same fixture on fa2; the three legacy identities at rtol 1e-2 / atol "
        "1e-3 plus the fp32 oracle on every run (output and LSE)",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _fixture(dtype, n_ctx, seed):
    """The legacy fixture, verbatim (same RNG order: k, v, q), plus its
    unified metadata and an fa2 plan."""
    torch.manual_seed(seed)
    H_QO, H_KV, HEAD_DIM, PAGE_SIZE = 1, 1, 64, 16
    max_num_pages = (n_ctx + PAGE_SIZE - 1) // PAGE_SIZE
    k_cache = torch.randn(
        max_num_pages, PAGE_SIZE, H_KV, HEAD_DIM, dtype=dtype, device="cuda"
    )
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(n_ctx, H_QO, HEAD_DIM, dtype=dtype, device="cuda")

    qo_indptr_cpu = torch.tensor([0, n_ctx], dtype=torch.int32)
    kv_lens_cpu = torch.tensor([n_ctx], dtype=torch.int32)
    block_tables = torch.arange(max_num_pages, dtype=torch.int32, device="cuda")[None]
    md = PagedAttentionMetadata.dense(
        qo_indptr_cpu.cuda(),
        kv_lens_cpu.cuda(),
        block_tables,
        page_size=PAGE_SIZE,
        max_q_len=n_ctx,
        max_kv_len=n_ctx,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    attn = PagedAttention(torch.device("cuda"))
    attn.plan(
        md,
        num_qo_heads=H_QO,
        num_kv_heads=H_KV,
        head_dim_qk=HEAD_DIM,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        backend="fa2",
    )
    assert attn.backend == "fa2"

    def oracle(out, lse, k_scale=1.0, v_scale=1.0):
        ref_out, ref_lse = reference_paged_prefill(
            q,
            k_cache,
            v_cache,
            qo_indptr_cpu,
            kv_lens_cpu,
            block_tables,
            PAGE_SIZE,
            True,
            sm_scale=k_scale / math.sqrt(HEAD_DIM),
            kv_layout="NHD",
        )
        torch.testing.assert_close(out.float(), ref_out * v_scale, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)

    return q, (k_cache, v_cache), attn, oracle


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_effect(dtype):
    q, kv, attn, oracle = _fixture(dtype, n_ctx=8, seed=42)

    out1, lse1 = attn.run(q, kv, k_scale=0.1, v_scale=0.1)
    out2, lse2 = attn.run(q, kv, k_scale=2.0, v_scale=2.0)

    # legacy assertion
    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different."
    )
    oracle(out1, lse1, k_scale=0.1, v_scale=0.1)
    oracle(out2, lse2, k_scale=2.0, v_scale=2.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_kv_scale_forwarding_math_property(dtype: torch.dtype):
    q, kv, attn, oracle = _fixture(dtype, n_ctx=128, seed=0)
    k_scale, v_scale = 0.5, 2.0

    # case 1: k_scale only == scaling q
    out1, lse1 = attn.run(q, kv, k_scale=k_scale)
    out1_ref, _ = attn.run(q * k_scale, kv)
    torch.testing.assert_close(out1, out1_ref, rtol=1e-2, atol=1e-3)

    # case 2: v_scale only == scaling the output
    out2, lse2 = attn.run(q, kv, v_scale=v_scale)
    out2_ref, lse2_ref = attn.run(q, kv)
    torch.testing.assert_close(out2, out2_ref * v_scale, rtol=1e-2, atol=1e-3)

    # case 3: both
    out3, lse3 = attn.run(q, kv, k_scale=k_scale, v_scale=v_scale)
    torch.testing.assert_close(out3, out1_ref * v_scale, rtol=1e-2, atol=1e-3)

    oracle(out2_ref, lse2_ref)
    oracle(out1, lse1, k_scale=k_scale)
    oracle(out2, lse2, v_scale=v_scale)
    oracle(out3, lse3, k_scale=k_scale, v_scale=v_scale)
