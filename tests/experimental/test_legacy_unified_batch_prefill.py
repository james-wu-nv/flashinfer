"""Legacy -> unified: tests/attention/test_batch_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the two
separately allocated fp16 / bf16 NHD pools (B1, D64, page 16, N_CTX 8 / 128),
the legacy CSR mapped losslessly to ``PagedAttentionMetadata.dense`` (page
16), the legacy assertions and tolerances (rtol 1e-2, atol 1e-3 against the
legacy wrapper's own ``k_scale`` / ``v_scale`` outputs and the legacy scaling
identities) plus the independent fp32 oracle on the scaled attention.  The
``dtype`` axis keeps the legacy values; the row id is the legacy node id plus
``-<backend>``.

CI: legacy file only in the H100 1/5 sampling lane (no A10G fixed shard).
The unified file is not collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- none left open.  ``k_scale`` / ``v_scale`` on a fp16 / bf16 KV cache are
  accepted by ``run()`` since WP-T (EXPECT_FLOAT_KV_SCALES): ``k_scale`` folds
  into the softmax scale the kernel launches with and ``v_scale`` multiplies
  the output, on every backend (design doc "Feature axes").  The rejection
  branch of the flag is kept for the record.
- legacy finding: the legacy tests plan ``causal=True`` and then call the
  deprecated ``forward_return_lse``, which resets ``causal`` to False and
  ``sm_scale`` to None before running.  The unified rows keep the declared
  causal plan and compare against ``wrapper.run(..., return_lse=True)`` on the
  same plan, so the reference is the plan the legacy test declared.
"""

import math

import pytest
import torch

from .legacy_unified_helpers import (
    EXPECT_FLOAT_KV_SCALES,
    LegacyBatch,
    argnames,
    assert_oracle,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    legacy_paged_wrapper,
    param_rows,
    plan_batch,
    run_batch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_batch_prefill.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_effect",
        ["test_kv_scale_forwarding_effect"],
        "equivalent",
        "same fixture (seed 42, B1 N_CTX 8 D64 page16 H1:1, two NHD pools, fp16 / bf16) "
        "and the legacy assertion (scales 0.1/0.1 vs 2.0/2.0 change the output); "
        "run(k_scale, v_scale) is the unified surface since WP-T "
        "(EXPECT_FLOAT_KV_SCALES); both outputs additionally match the legacy wrapper's "
        "own k_scale/v_scale outputs at rtol 1e-2 / atol 1e-3 and the oracle with "
        "sm_scale = k_scale / sqrt(D) and out / v_scale; every backend.",
    ),
    (
        "tests/attention/test_batch_prefill.py::test_kv_scale_forwarding_math_property",
        ["test_kv_scale_forwarding_math_property"],
        "equivalent",
        "same fixture (seed 0, N_CTX 128) and the three legacy identities (k_scale == "
        "scaling q, v_scale == scaling the output, both) at the legacy rtol 1e-2 / atol "
        "1e-3, through run(k_scale, v_scale); plus the legacy wrapper's own outputs and "
        "the oracle on the unscaled plan; every backend.",
    ),
]

KV_SCALE_AXES = dict(dtype=[torch.float16, torch.bfloat16])
LEGACY_TOL = dict(rtol=1e-2, atol=1e-3)


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


def _kv_scale_fixture(dtype, n_ctx, seed):
    """Verbatim legacy construction: two NHD pools, one request of n_ctx."""
    torch.manual_seed(seed)
    H_QO, H_KV, HEAD_DIM, PAGE_SIZE = 1, 1, 64, 16
    max_num_pages = (n_ctx + PAGE_SIZE - 1) // PAGE_SIZE
    dev = torch.device("cuda")
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


def _scaled_run(lb, md, backend, attn, *, k_scale=None, v_scale=None):
    """``run(k_scale, v_scale)`` on the planned instance (positive branch), or
    the rejection plus the migration adapter (``sm_scale = k_scale /
    sqrt(D)``, ``out * v_scale``) while the flag is off."""
    if EXPECT_FLOAT_KV_SCALES:
        return attn.run(lb.q, (lb.k, lb.v), k_scale=k_scale, v_scale=v_scale)
    with pytest.raises(ValueError, match="KV scales apply to fp8 KV caches only"):
        attn.run(lb.q, (lb.k, lb.v), k_scale=k_scale, v_scale=v_scale)
    sm_scale = None if k_scale is None else k_scale / math.sqrt(lb.head_dim_qk)
    _, out, lse = run_batch(lb, md, backend, causal=True, sm_scale=sm_scale)
    if v_scale is not None:
        out = (out.float() * v_scale).to(out.dtype)
    return out, lse


def _assert_scaled_oracle(lb, out, lse, *, k_scale=None, v_scale=None):
    """The oracle sees ``k_scale`` as the softmax scale and ``v_scale`` undone
    on the output (both exact in fp32 for the legacy scale values)."""
    sm_scale = None if k_scale is None else k_scale / math.sqrt(lb.head_dim_qk)
    out_for_oracle = out if v_scale is None else (out.float() / v_scale).to(out.dtype)
    assert_oracle(lb, out_for_oracle, lse, causal=True, sm_scale=sm_scale)


@pytest.mark.parametrize(
    argnames(KV_SCALE_AXES, "backend"), param_rows(KV_SCALE_AXES, lambda p: True)
)
def test_kv_scale_forwarding_effect(dtype, backend):
    """Scales (0.1, 0.1) vs (2.0, 2.0) must change the output (legacy); each
    output matches the legacy wrapper's k_scale/v_scale output and the oracle."""
    lb = _kv_scale_fixture(dtype, n_ctx=8, seed=42)
    md = lb.metadata()
    attn = plan_batch(lb, md, backend, causal=True)
    out1, lse1 = _scaled_run(lb, md, backend, attn, k_scale=0.1, v_scale=0.1)
    out2, lse2 = _scaled_run(lb, md, backend, attn, k_scale=2.0, v_scale=2.0)
    assert not torch.allclose(out1, out2, atol=1e-3), (
        "Output should change when k_scale/v_scale values are different."
    )  # legacy assertion
    wrapper = legacy_paged_wrapper(lb, workspace_mb=16)
    ref1, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=0.1, v_scale=0.1)
    ref2, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=2.0, v_scale=2.0)
    torch.testing.assert_close(out1, ref1, **LEGACY_TOL)
    torch.testing.assert_close(out2, ref2, **LEGACY_TOL)
    _assert_scaled_oracle(lb, out1, lse1, k_scale=0.1, v_scale=0.1)
    _assert_scaled_oracle(lb, out2, lse2, k_scale=2.0, v_scale=2.0)


@pytest.mark.parametrize(
    argnames(KV_SCALE_AXES, "backend"), param_rows(KV_SCALE_AXES, lambda p: True)
)
def test_kv_scale_forwarding_math_property(dtype, backend):
    """k_scale == scaling q, v_scale == scaling the output, both together
    (legacy identities at rtol 1e-2 / atol 1e-3), against the unified runs,
    the legacy wrapper's own k_scale/v_scale outputs and the oracle."""
    lb = _kv_scale_fixture(dtype, n_ctx=128, seed=0)
    md = lb.metadata()
    k_scale, v_scale = 0.5, 2.0
    attn = plan_batch(lb, md, backend, causal=True)
    out1, lse1 = _scaled_run(lb, md, backend, attn, k_scale=k_scale)
    out2, lse2 = _scaled_run(lb, md, backend, attn, v_scale=v_scale)
    out3, lse3 = _scaled_run(lb, md, backend, attn, k_scale=k_scale, v_scale=v_scale)
    # the legacy identities, in unified terms
    base, base_lse = attn.run(lb.q, (lb.k, lb.v))
    scaled_q, _ = attn.run((lb.q * k_scale).to(lb.dtype), (lb.k, lb.v))
    torch.testing.assert_close(out1, scaled_q, **LEGACY_TOL)  # case 1: k_scale only
    torch.testing.assert_close(
        out2, (base.float() * v_scale).to(base.dtype), **LEGACY_TOL
    )  # case 2: v_scale only
    torch.testing.assert_close(
        out3, (scaled_q.float() * v_scale).to(base.dtype), **LEGACY_TOL
    )  # case 3: both
    # and the legacy wrapper's own k_scale / v_scale outputs
    wrapper = legacy_paged_wrapper(lb, workspace_mb=16)
    ref1, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, k_scale=k_scale)
    ref2, _ = wrapper.run(lb.q, (lb.k, lb.v), return_lse=True, v_scale=v_scale)
    ref3, _ = wrapper.run(
        lb.q, (lb.k, lb.v), return_lse=True, k_scale=k_scale, v_scale=v_scale
    )
    torch.testing.assert_close(out1, ref1, **LEGACY_TOL)
    torch.testing.assert_close(out2, ref2, **LEGACY_TOL)
    torch.testing.assert_close(out3, ref3, **LEGACY_TOL)
    assert_oracle(lb, base, base_lse, causal=True)
    _assert_scaled_oracle(lb, out1, lse1, k_scale=k_scale)
    _assert_scaled_oracle(lb, out2, lse2, v_scale=v_scale)
    _assert_scaled_oracle(lb, out3, lse3, k_scale=k_scale, v_scale=v_scale)
