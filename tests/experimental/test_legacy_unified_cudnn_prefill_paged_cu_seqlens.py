"""Legacy -> unified: tests/attention/test_cudnn_prefill_paged_cu_seqlens.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only, and only if the runner
image ships cudnn-frontend; the unified file is in no default lane
(tests/experimental is excluded by norecursedirs).

The legacy test drives ONE native call (paged, token-unit ``batch_offsets_q``)
through both cuDNN sequence-length paths by monkeypatching the private version
gate ``flashinfer.cudnn.prefill._cudnn_supports_direct_seqlens``: forced off
(the element-offset conversion path) and forced on for the mixed form (the
direct ``cu_seq_len_q`` + ``seq_len_kv`` path), and asserts the two agree at
1e-2.  The unified cuDNN backend issues exactly that native call
(``batch_offsets_units="tokens"``, dense block table), so the same monkeypatch
steers it: here two ``PagedAttention(backend="cudnn")`` plans, one under each
gate, on the imported legacy fixture (``_make_paged_inputs``: seed 1, the
``as_strided`` combined pool as K/V views, the legacy block table); the
legacy assertion (direct vs conversion at 1e-2) plus both outputs against
the fp32 oracle.  Default run = the full legacy grid (64 ids).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- none: the one legacy function is fully covered; the version gate it patches
  is private to flashinfer.cudnn.prefill (the legacy test patches it too).
"""

import pytest
import torch

from flashinfer.cudnn import prefill as cudnn_prefill

from .legacy_unified_helpers import (
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py::test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy",
        ["test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy"],
        "equivalent",
        "same grid (B1/4, (s_qo, s_kv) (64, 512)/(17, 200), page 16/64, Hkv 1/2, H8, "
        "D128, causal), the legacy fixture imported (_make_paged_inputs, seed 1, "
        "as_strided pool views, legacy table), the legacy skips inherited; the legacy "
        "monkeypatch of the private version gate steers the unified cudnn backend's "
        "native call the same way: one plan under the forced-off gate (element-offset "
        "conversion path), one under the mixed-form gate (direct path, its consultation "
        "recorded as legacy does); direct vs conversion at the legacy 1e-2 budget and "
        "both vs the oracle; lse_mode none as legacy return_lse=False",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(64, 512), (17, 200)])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy(
    monkeypatch, batch_size, s_qo, s_kv, page_size, num_kv_heads, causal
):
    from tests.attention.test_cudnn_prefill_paged_cu_seqlens import (
        _make_paged_inputs,
        _mixed_paged_supported,
    )

    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not _mixed_paged_supported():
        pytest.skip("cuDNN backend/frontend too old for mixed-form paged seqlens")

    device = "cuda:0"
    num_qo_heads, head_dim = 8, 128
    inp = _make_paged_inputs(
        batch_size, s_qo, s_kv, page_size, num_qo_heads, num_kv_heads, head_dim, device
    )
    scale = float(head_dim**-0.5)
    md = dense_metadata(
        inp["qo_indptr"],
        inp["actual_seq_lens_kv"].view(-1),
        inp["block_tables"],
        page_size,
    )

    def run():
        # plan AND run under the active gate: the backend issues the native
        # call from run(), which consults the gate for the paged (mixed) form
        attn = plan_pinned(
            "cudnn",
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.bfloat16,
            kv_layout="HND",
            causal=causal,
            lse_mode="none",  # legacy return_lse=False
        )
        out, lse = attn.run(inp["q"], (inp["k_cache"], inp["v_cache"]), sm_scale=scale)
        assert lse is None
        return out

    # Legacy paged path: force the gate off -> element-offset conversion.
    monkeypatch.setattr(
        cudnn_prefill,
        "_cudnn_supports_direct_seqlens",
        lambda dtype, *, mixed=False: False,
    )
    out_legacy = run()

    # Direct paged path: force the gate on for the mixed-form request only,
    # recording the consultations (as the legacy test does).
    mixed_calls = []

    def _gate_direct(dtype, *, mixed=False):
        mixed_calls.append(mixed)
        return mixed

    monkeypatch.setattr(cudnn_prefill, "_cudnn_supports_direct_seqlens", _gate_direct)
    out_direct = run()

    assert True in mixed_calls, (
        "paged dispatch did not consult _cudnn_supports_direct_seqlens with "
        "mixed=True, so the direct mixed-form path was not exercised"
    )
    torch.testing.assert_close(out_direct, out_legacy, atol=1e-2, rtol=1e-2)  # legacy
    ref_out, _ = oracle(
        md,
        inp["q"],
        inp["k_cache"],
        inp["v_cache"],
        causal=causal,
        kv_layout="HND",
        sm_scale=scale,
    )
    torch.testing.assert_close(out_direct.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(out_legacy.float(), ref_out, **OUT_TOL)
