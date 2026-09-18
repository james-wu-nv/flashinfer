"""Legacy -> unified: tests/attention/test_non_contiguous_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: q is the
head slice of a fused QKV projection (token stride ``(Hq + 2 Hkv) * D``), the
two fp16 NHD pools, the legacy CSR mapped losslessly (page 1 / 5 -> ``.csr``).
The row asserts the legacy comparison (slice vs ``q.contiguous()`` at rtol
1e-3 / atol 2e-3) and both outputs against the fp32 oracle.  The parametrize
axes keep the legacy names and values; the row id is the legacy node id plus
``-<backend>``; the full legacy grid is under ``slow``.

CI: legacy file only in the H100 1/5 sampling lane.  The unified file is not
collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- page sizes 1 and 5 select the flat page-id form, so only the CSR-native
  backends (fa2, fa3) resolve -- exactly the backends the legacy test ran;
  cudnn / trtllm-gen / cake skip with the dense-derivation reason.
- cuDNN declares ``requires_contiguous_q`` (its graph scales the token-unit
  ragged offsets by Hq * D, so a fused-QKV head slice is silently wrong);
  the contract rejects the strided view with "requires packed q" and the
  row records that instead of misreading (a packed copy is the documented
  migration).  It never resolves here anyway (page 1 / 5).
- single-prefill and ragged functions of the legacy file are out of scope.
"""

import pytest
import torch

from flashinfer.experimental.paged_attention import CAPABILITIES

from .legacy_unified_helpers import (
    DEVICE,
    LegacyBatch,
    argnames,
    assert_oracle,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    param_rows,
    plan_batch,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_non_contiguous_prefill.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_non_contiguous_prefill.py::test_single_prefill_packed_input",
        [],
        "out-of-scope",
        "single_prefill_with_kv_cache on a packed QKV view, not paged prefill.",
    ),
    (
        "tests/attention/test_non_contiguous_prefill.py::test_batch_ragged_prefill_packed_input",
        [],
        "out-of-scope",
        "ragged (BatchPrefillWithRaggedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_non_contiguous_prefill.py::test_batch_paged_prefill_packed_input",
        ["test_batch_paged_prefill_packed_input"],
        "equivalent",
        "same grid (B1/19/99 x page1/5 x seq1/7/127/257 x Hkv1/4/8 x Hq4/8 x D64/128/256 "
        "x causal); q is the head slice of a fused QKV buffer; slice vs q.contiguous() "
        "at the legacy tolerance (rtol 1e-3, atol 2e-3) + oracle for both; page 1/5 => "
        "CSR form, so only the CSR-native fa backends resolve (as in the legacy fa2 "
        "run); cudnn rejects strided q by contract.",
    ),
]

PACKED_AXES = dict(
    batch_size=[1, 19, 99],
    page_size=[1, 5],
    seq_len=[1, 7, 127, 257],
    num_kv_heads=[1, 4, 8],
    num_qo_heads=[4, 8],
    head_dim=[64, 128, 256],
    causal=[True, False],
)
PACKED_DEFAULT = {(1, 1), (19, 7), (99, 127), (19, 257)}  # (batch_size, seq_len)


def _packed_default(p):
    b, page, seq, hk, hq, hd, causal = p
    return (b, seq) in PACKED_DEFAULT and (hk, hq) in {(1, 4), (4, 8), (8, 8)}


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


@pytest.mark.parametrize(
    argnames(PACKED_AXES, "backend"), param_rows(PACKED_AXES, _packed_default)
)
def test_batch_paged_prefill_packed_input(
    batch_size,
    page_size,
    seq_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    backend,
):
    """q is the head slice of a fused QKV projection (token stride
    (Hq + 2 Hkv) D); the slice and its ``.contiguous()`` copy must agree at
    the legacy tolerance and both match the oracle.  A backend that requires
    packed q rejects the view by contract (recorded, never misread)."""
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("num_qo_heads must be a multiple of num_kv_heads")  # legacy
    torch.manual_seed(
        seed_of(
            "packed",
            batch_size,
            page_size,
            seq_len,
            num_kv_heads,
            num_qo_heads,
            head_dim,
            causal,
        )
    )
    dev = torch.device(DEVICE)
    nnz = batch_size * seq_len
    pages_per_req = (seq_len + page_size - 1) // page_size
    num_pages = batch_size * pages_per_req
    k_cache = torch.randn(
        num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=dev
    )
    v_cache = torch.randn_like(k_cache)
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device=dev,
    )
    q, _, _ = qkv_packed.split(
        (num_qo_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim),
        dim=-1,
    )
    q = q.view(-1, num_qo_heads, head_dim)
    # a single token (nnz == 1) is contiguous by PyTorch's definition; every
    # other point is the strided fused-QKV slice
    assert q.stride(-1) == 1 and (nnz == 1 or not q.is_contiguous())
    lb = LegacyBatch(
        q=q,
        k=k_cache,
        v=v_cache,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * seq_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_req,
        kv_indices_cpu=torch.arange(num_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (seq_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout="NHD",
    )
    md = lb.metadata()
    attn = plan_batch(lb, md, backend, causal=causal)
    if CAPABILITIES[attn.backend].requires_contiguous_q and not q.is_contiguous():
        with pytest.raises(ValueError, match="requires packed q"):
            attn.run(q, (k_cache, v_cache))
        out_c, lse_c = attn.run(q.contiguous(), (k_cache, v_cache))
        assert_oracle(lb, out_c, lse_c, causal=causal)
        pytest.skip(
            f"[{attn.backend}] rejects the fused-QKV head slice by contract "
            "(requires packed q); the contiguous copy matches the oracle"
        )
    out_packed, lse_packed = attn.run(q, (k_cache, v_cache))
    out_contig, lse_contig = attn.run(q.contiguous(), (k_cache, v_cache))
    torch.testing.assert_close(out_packed, out_contig, rtol=1e-3, atol=2e-3)  # legacy
    assert_oracle(lb, out_packed, lse_packed, causal=causal)
    assert_oracle(lb, out_contig, lse_contig, causal=causal)
