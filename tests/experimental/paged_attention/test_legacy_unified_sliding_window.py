"""Legacy -> unified: tests/attention/test_sliding_window.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the two
fp16 NHD pools, uniform lengths, ``paged_kv_indices = arange``, the legacy CSR
mapped losslessly (page 1 -> ``.csr``, page 16 -> ``.dense``), causal sliding
window with the same ``window_left`` meaning (keys ``j >= p - window_left``).
The row asserts the legacy comparison (``single_prefill_with_kv_cache(
window_left, causal=True, backend="fa2")`` per request at rtol/atol 1e-3) and
the fp32 oracle.  The parametrize axes keep the legacy names and values; the
legacy grid had ``backend`` (fa2 / auto) as its innermost axis, so the row id
starts with the backend and the fa2 / auto rows carry exactly the legacy
node ids; the other four backends are extra rows.  The full grid (7776 legacy
points x fa2 / auto) is under ``slow``.

CI: legacy file in A10G fixed shard part1 (full) and H100 1/5 sampling.  The
unified file is not collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- head_dim 512 rows: (512, 512) is declared on fa2 since WP-T
  (EXPECT_FA2_HEAD_DIM_512), so they run on fa2 / auto; the legacy
  ``skip_if_head_dim_unsupported`` (SM80+) is moot on every device the
  unified backends declare.  cudnn / trtllm-gen / cake skip (D512 undeclared;
  cudnn also declares no sliding window at all).
- decode / single-prefill / ragged functions of the legacy file are out of
  scope (not paged prefill).
"""

import pytest
import torch

from .legacy_unified_helpers import (
    argnames,
    assert_legacy_isclose,
    assert_oracle,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    legacy_reference_single_prefill,
    legacy_uniform_batch,
    param_rows,
    run_batch,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_sliding_window.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_sliding_window.py::test_single_decode_sliding_window",
        [],
        "out-of-scope",
        "single_decode_with_kv_cache, not paged prefill.",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_decode_sliding_window",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper (decode-only API), not paged prefill.",
    ),
    (
        "tests/attention/test_sliding_window.py::test_single_decode_prefill_sliding_window_match",
        [],
        "out-of-scope",
        "single decode vs single prefill, not paged prefill.",
    ),
    (
        "tests/attention/test_sliding_window.py::test_single_prefill_sliding_window",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache, not paged prefill.",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_paged_prefill_sliding_window",
        ["test_batch_paged_prefill_sliding_window"],
        "equivalent",
        "same grid (B12/17/30 x kv54/397/1177 x q1/37/47 x window13/33/111 x Hkv1/4/8 x "
        "Hq4/8 x D64/128/256/512 x page1/16 x backend), two fp16 NHD pools, legacy "
        "tolerance 1e-3 vs single_prefill(window_left, causal, backend=fa2) + oracle; "
        "fa2 / auto rows carry the legacy node ids, the other backends are extra rows; "
        "D512 runs on fa2 since WP-T (EXPECT_FA2_HEAD_DIM_512); cudnn skips every row "
        "(no sliding window).",
    ),
    (
        "tests/attention/test_sliding_window.py::test_batch_ragged_prefill_sliding_window",
        [],
        "out-of-scope",
        "ragged (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill.",
    ),
]

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


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


@pytest.mark.parametrize(
    argnames(SWA_AXES, "backend"),
    param_rows(
        SWA_AXES, _swa_default, slow_backends=("fa2", "auto"), backend_first=True
    ),
)
def test_batch_paged_prefill_sliding_window(
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
    """fp16 NHD pools, causal sliding window; legacy tolerance vs
    ``single_prefill_with_kv_cache(window_left, causal=True, backend="fa2")``
    + oracle.  ``window_left`` has the same meaning in both APIs."""
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
        seed=seed_of(
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
    _, out, lse = run_batch(lb, md, backend, causal=True, window_left=window_left)
    ref = legacy_reference_single_prefill(
        lb, causal=True, window_left=window_left, backend="fa2"
    )
    assert_legacy_isclose(out, ref, rtol=1e-3, atol=1e-3)
    assert_oracle(lb, out, lse, causal=True, window_left=window_left)
