"""Legacy -> unified: tests/attention/test_batch_invariant_fa2.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the fp16
combined pool ``/10`` as ``K = kv[:, 0]`` / ``V = kv[:, 1]`` views (HND and
NHD), uniform lengths, the legacy CSR mapped losslessly (page 1 -> ``.csr``,
page 8 / 16 -> ``.dense``), the batch of B requests and its first
``invariant_bs`` requests as a batch of their own.  The parametrize axes keep
the legacy names and values; the row id is the legacy node id plus
``-<backend>``; the full 1152-point grid is under ``slow``.

CI: legacy file only in the H100 1/5 sampling lane.  The unified file is not
collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- ``fixed_split_size`` / ``disable_split_kv``: plan() has neither argument
  (TypeError, asserted); the split-KV schedule is the backend's own and the
  contract promises no batch invariance (design doc: "Known limitations" has
  no determinism item; ``_capabilities.py`` declares no determinism axis).
  The legacy bitwise assertion (batch B vs its first two requests) is
  therefore an xfail-if-unequal until a determinism policy exists
  (EXPECT_BATCH_INVARIANT); both batches are checked against the oracle.
  Measured on B200 (round 3): fa2 differs bitwise by <= 3.8e-6 (output) /
  1.5e-5 (LSE) on this fixture -- the split-KV schedule depends on the batch.
- pos_encoding_mode="ROPE_LLAMA" rows: no fused positional encoding in the
  unified API (EXPECT_ROPE); the rows rotate q and the K pages with the
  external adapter and run the same invariance check on the rotated tensors.
- ``test_batch_decode_tensor_cores`` uses the decode wrapper
  (BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores=True): decode-only
  API, out of scope for this conversion.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_BATCH_INVARIANT,
    apply_external_rope,
    argnames,
    assert_oracle,
    assert_rope_kwargs_rejected,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    legacy_uniform_batch,
    param_rows,
    plan_signature_params,
    run_batch,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_batch_invariant_fa2.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_decode_tensor_cores",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True): decode-only API.",
    ),
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_prefill_tensor_cores",
        ["test_batch_prefill_tensor_cores"],
        "unsupported-by-design",
        "same 12 legacy axes (B3/4, invariant_bs 2, kv4096/5000, q128/256, "
        "fixed_split_size 2048, disable_split_kv, page1/8/16, Hkv4, group1/4/8, "
        "D128/256, HND/NHD, pos NONE/ROPE_LLAMA); plan() has no fixed_split_size / "
        "disable_split_kv (TypeError asserted) and the contract promises no batch "
        "invariance: the legacy fixture is run at batch B and at its first 2 requests, "
        "both against the oracle, and the bitwise comparison is an xfail-if-unequal "
        "until a determinism policy exists (EXPECT_BATCH_INVARIANT); ROPE rows through "
        "the external adapter.",
    ),
]

INVARIANT_AXES = dict(
    batch_size=[3, 4],
    invariant_bs=[2],
    kv_len=[4096, 5000],
    qo_len=[128, 256],
    fixed_split_size=[2048],
    disable_split_kv=[True, False],
    page_size=[1, 8, 16],
    num_kv_heads=[4],
    group_size=[1, 4, 8],
    head_dim=[128, 256],
    kv_layout=["HND", "NHD"],
    pos_encoding_mode=["NONE", "ROPE_LLAMA"],
)
# (batch_size, kv_len, qo_len, disable_split_kv, page_size, group_size, head_dim, kv_layout)
INVARIANT_DEFAULT = {
    (3, 4096, 128, True, 1, 1, 128, "HND"),
    (4, 5000, 256, False, 8, 4, 256, "NHD"),
    (3, 5000, 128, False, 16, 8, 128, "NHD"),
    (4, 4096, 256, True, 16, 1, 256, "HND"),
}


def _invariant_default(p):
    b, inv, kv, qo, fss, dsk, page, hk, group, hd, layout, pos = p
    return (b, kv, qo, dsk, page, group, hd, layout) in INVARIANT_DEFAULT


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


@pytest.mark.parametrize(
    argnames(INVARIANT_AXES, "backend"),
    param_rows(INVARIANT_AXES, _invariant_default),
)
def test_batch_prefill_tensor_cores(
    batch_size,
    invariant_bs,
    kv_len,
    qo_len,
    fixed_split_size,
    disable_split_kv,
    page_size,
    num_kv_heads,
    group_size,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    backend,
):
    """The legacy fixture (fp16 combined pool /10) at batch B and at its
    first ``invariant_bs`` requests, each against the oracle; the legacy
    bitwise assertion is an xfail-if-unequal until the API has a
    determinism policy (the legacy ``fixed_split_size`` /
    ``disable_split_kv`` knobs have no counterpart and are carried as
    parameters only)."""
    lb = legacy_uniform_batch(
        batch_size=batch_size,
        kv_len=kv_len,
        qo_len=qo_len,
        page_size=page_size,
        num_qo_heads=num_kv_heads * group_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kv_layout=kv_layout,
        fp32_source=False,
        scale=10.0,
        seed=seed_of(
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
    params = plan_signature_params()
    for name in ("fixed_split_size", "disable_split_kv"):
        assert name not in params, f"plan() grew {name!r}: port the split knobs"
    with pytest.raises(TypeError, match="fixed_split_size"):
        PagedAttention(torch.device(DEVICE)).plan(
            md,
            **lb.plan_kwargs(),
            causal=False,
            fixed_split_size=None if disable_split_kv else fixed_split_size,
            disable_split_kv=disable_split_kv,
            backend="fa2",
        )
    if pos_encoding_mode == "ROPE_LLAMA":
        assert_rope_kwargs_rejected(lb, md)
        q, k = apply_external_rope(lb)
    else:
        q, k = lb.q, lb.k
    _, out, lse = run_batch(lb, md, backend, q=q, k=k, causal=False)
    assert_oracle(lb, out, lse, causal=False, q=q, k=k)
    sub = lb.prefix(invariant_bs)
    n = invariant_bs * qo_len
    _, out_inv, lse_inv = run_batch(
        sub, sub.metadata(), backend, q=q[:n], k=k, causal=False
    )
    assert_oracle(sub, out_inv, lse_inv, causal=False, q=q[:n], k=k)
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
