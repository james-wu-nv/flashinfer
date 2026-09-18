"""Legacy -> unified: tests/attention/test_cudnn_prefill_token_indptr.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only, and only if the runner
image ships cudnn-frontend; the unified file is in no default lane
(tests/experimental is excluded by norecursedirs).

The legacy file exercises ``batch_offsets_units="tokens"`` on the RAGGED
cuDNN prefill call (K/V as ``(total_kv, H, D)`` with a token-unit
``kv_indptr``, no page table).  Token-unit offsets are the unified cuDNN
backend's native dialect, so the paged spelling of each fixture is what runs
here: the ragged K/V of every request is laid into its own page-16 pages (a
copy -- the ragged storage has no page structure), a dense block table
addresses them, and the pinned ``backend="cudnn"`` plan runs on that pool.
The legacy reference (the element-offset ragged cuDNN call, i.e. the
historical graph) and the legacy tolerances are kept; the fp32 oracle is
added.  The legacy ``direct`` axis is kept too: the legacy monkeypatch of the
private version gate ``_cudnn_supports_direct_seqlens`` forces the conversion
path for the unified backend's native call exactly as it does for the legacy
call.  Default run = the full legacy grids (128 + 2 + 3 ids).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_cudnn_prefill_token_indptr: ragged K/V re-laid into a page-16 pool
  (copy) so the paged facade can address it; the legacy ``direct=False``
  assertion is BITWISE (conversion path == the identical legacy graph) and
  cannot carry over, since the paged graph is not the ragged reference graph:
  both axes use the legacy direct-path budget (1e-2) instead.  LSE compared
  packed ``(tokens, H)``.
- test_cudnn_prefill_token_indptr_omit_actual_seq_lens: the legacy contract
  (``actual_seq_lens_q/kv`` optional with token-unit indptrs, omitted ==
  passed bitwise) is a native-argument contract: the unified facade always
  derives the lengths from the metadata and passes them, so "omitting" has
  no unified spelling.  The row keeps the fixture: the paged unified spelling
  is compared with the legacy "omitted" native call at the direct budget
  (the facade's derived lengths agree with the native derivation) plus the
  oracle; the bitwise omitted-vs-passed check itself stays native-only.
- test_cudnn_prefill_lse_is_base2: the batch-1 ragged fixture (s_q 37, s_kv
  64) laid into four page-16 pages with an identity table; the base-2 LSE of
  the unified plan is compared against the legacy float reference
  (``logsumexp * log2e``) and the oracle.
"""

import pytest
import torch

from flashinfer.cudnn import cudnn_batch_prefill_with_kv_cache
from flashinfer.cudnn import prefill as cudnn_prefill

from .legacy_unified_helpers import (
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_cudnn_prefill_token_indptr.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_token_indptr",
        ["test_cudnn_prefill_token_indptr"],
        "partial",
        "same grid (B1/4, s_qo 32/87, s_kv 87/512, Hkv 1/4, H8, (Dqk, Dvo) (128,128)/"
        "(192,128), causal, direct), seed 0, the legacy skips inherited; the RAGGED legacy "
        "K/V is laid into a page-16 pool (copy) with a dense table so the paged facade can "
        "run it on pinned cudnn; the legacy reference (element-offset ragged cuDNN call) "
        "at the legacy direct budget (1e-2) for out and packed LSE, plus the oracle; the "
        "legacy direct=False BITWISE assertion does not carry over (paged graph != ragged "
        "reference graph); the legacy monkeypatch of the private gate steers the unified "
        "backend's native call the same way",
    ),
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_token_indptr_omit_actual_seq_lens",
        ["test_cudnn_prefill_token_indptr_omit_actual_seq_lens"],
        "partial",
        "same fixture (B4, s_qo 87, s_kv 512, H8:4, D128, causal, seed 0, direct axis) "
        "re-laid into a page-16 pool; the unified paged spelling (lengths derived from "
        "the metadata) vs the legacy 'omitted actual_seq_lens' native call at the direct "
        "budget (1e-2) plus the oracle; the bitwise omitted-vs-passed contract of the "
        "native token-unit call is native-only (the facade always passes the lengths)",
    ),
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_lse_is_base2",
        ["test_cudnn_prefill_lse_is_base2"],
        "partial",
        "same fixture (B1, s_q 37, s_kv 64, H8, Hkv 1/2/8, D128, bf16, non-causal, seed 0) "
        "laid into four page-16 pages with an identity table (copy); the unified base2 LSE "
        "on pinned cudnn vs the legacy float reference (logsumexp * log2e) at the legacy "
        "budget (1e-2) and vs the oracle (out and LSE)",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


PAGE_SIZE = 16


def _lay_into_pages(kv_indptr_cpu, k_ragged, v_ragged, page_size=PAGE_SIZE):
    """Ragged ``(total_kv, H, D)`` K/V -> NHD page pools ``(pages, P, H, D)``
    plus the dense ``(b, max_pages)`` table, request i owning pages
    ``[pfx_i, pfx_i + ceil(kv_i / P))`` (a copy: the ragged storage has no
    page structure).  Padding columns are 0 (never read: kv_seq_lens bound
    the pages a request touches)."""
    lens = (kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]).tolist()
    pages_per = [(n + page_size - 1) // page_size for n in lens]
    total_pages = sum(pages_per)
    k_pool = torch.zeros(
        total_pages,
        page_size,
        *k_ragged.shape[1:],
        dtype=k_ragged.dtype,
        device=k_ragged.device,
    )
    v_pool = torch.zeros(
        total_pages,
        page_size,
        *v_ragged.shape[1:],
        dtype=v_ragged.dtype,
        device=v_ragged.device,
    )
    table = torch.zeros(len(lens), max(pages_per), dtype=torch.int32)
    page = 0
    for i, (n, npg) in enumerate(zip(lens, pages_per, strict=True)):
        s = int(kv_indptr_cpu[i])
        k_pool.view(total_pages * page_size, *k_ragged.shape[1:])[
            page * page_size : page * page_size + n
        ] = k_ragged[s : s + n]
        v_pool.view(total_pages * page_size, *v_ragged.shape[1:])[
            page * page_size : page * page_size + n
        ] = v_ragged[s : s + n]
        table[i, :npg] = torch.arange(page, page + npg, dtype=torch.int32)
        page += npg
    return k_pool, v_pool, table.to(k_ragged.device)


def _pack_lse(lse_ref, qo_indptr_cpu):
    """The native call's LSE as packed ``(tokens, H)``: already packed, or the
    padded ``(b, max_q, H)`` form gathered per request."""
    if lse_ref.dim() == 2:
        return lse_ref
    rows = []
    for i in range(qo_indptr_cpu.shape[0] - 1):
        n = int(qo_indptr_cpu[i + 1] - qo_indptr_cpu[i])
        rows.append(lse_ref[i, :n])
    return torch.cat(rows)


# ---------------------------------------------------------------------------
# test_cudnn_prefill_token_indptr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo", [32, 87])
@pytest.mark.parametrize("s_kv", [87, 512])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("head_dim_qk,head_dim_vo", [(128, 128), (192, 128)])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr(
    monkeypatch,
    batch_size,
    s_qo,
    s_kv,
    num_kv_heads,
    num_qo_heads,
    head_dim_qk,
    head_dim_vo,
    causal,
    direct,
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")
    if s_qo > s_kv:
        pytest.skip("s_qo > s_kv, skipping test as causal")

    # ---- the legacy fixture, verbatim (ragged) ----
    torch.manual_seed(0)
    device = "cuda:0"
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()
    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_qk,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim_vo,
        device=device,
        dtype=torch.bfloat16,
    )
    workspace_buffer = torch.empty(512 * 1024 * 1024, dtype=torch.int8, device=device)
    scale = float(1.0 / (head_dim_qk**0.5))

    # the legacy reference: the historical element-offset ragged graph
    out_ref, lse_ref = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale=scale,
        workspace_buffer=workspace_buffer,
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        actual_seq_lens_q=actual_seq_lens_q.view(batch_size, 1, 1, 1),
        actual_seq_lens_kv=actual_seq_lens_kv.view(batch_size, 1, 1, 1),
        causal=causal,
        return_lse=True,
        batch_offsets_q=qo_indptr * (num_qo_heads * head_dim_qk),
        batch_offsets_o=qo_indptr * (num_qo_heads * head_dim_vo),
        batch_offsets_k=kv_indptr * (num_kv_heads * head_dim_qk),
        batch_offsets_v=kv_indptr * (num_kv_heads * head_dim_vo),
    )

    if not direct:
        # Force the conversion path even where the direct path is supported
        # (the legacy monkeypatch; the unified backend's native call goes
        # through the same gate).
        monkeypatch.setattr(
            cudnn_prefill,
            "_cudnn_supports_direct_seqlens",
            lambda dtype, *, mixed=False: False,  # the paged dispatch asks mixed=True
        )

    # ---- unified: the ragged K/V laid into page-16 pools, dense table ----
    kv_indptr_cpu = kv_indptr.cpu()
    k_pool, v_pool, table = _lay_into_pages(kv_indptr_cpu, k_cache, v_cache)
    md = dense_metadata(qo_indptr, actual_seq_lens_kv, table, PAGE_SIZE)
    attn = plan_pinned(
        "cudnn",
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        q_dtype=torch.bfloat16,
        kv_layout="NHD",
        causal=causal,
        lse_mode="base2",
    )
    out, lse = attn.run(q, (k_pool, v_pool), sm_scale=scale)

    # the legacy assertion at the legacy direct-path budget (both axes: the
    # paged graph is not the ragged reference graph, so bitwise is out)
    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse, _pack_lse(lse_ref, qo_indptr.cpu()), atol=1e-2, rtol=1e-2
    )
    ref_out, ref_lse = oracle(
        md, q, k_pool, v_pool, causal=causal, kv_layout="NHD", sm_scale=scale
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_cudnn_prefill_token_indptr_omit_actual_seq_lens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direct", [True, False])
def test_cudnn_prefill_token_indptr_omit_actual_seq_lens(monkeypatch, direct):
    """The legacy call omits actual_seq_lens_q/kv (derived from the token-unit
    indptrs natively); the unified facade derives them from the metadata.
    The paged unified spelling of the legacy fixture is compared with the
    legacy 'omitted' native call; the native omitted == passed bitwise
    contract stays native-only."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if direct and not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16):
        pytest.skip("cuDNN backend/frontend too old for direct token-unit seqlens")
    if not direct:
        monkeypatch.setattr(
            cudnn_prefill,
            "_cudnn_supports_direct_seqlens",
            lambda dtype, *, mixed=False: False,
        )

    # ---- the legacy fixture, verbatim (ragged) ----
    torch.manual_seed(0)
    device = "cuda:0"
    batch_size, s_qo, s_kv = 4, 87, 512
    num_qo_heads, num_kv_heads, head_dim = 8, 4, 128
    actual_seq_lens_q = torch.randint(
        1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device
    )
    actual_seq_lens_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_q, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(actual_seq_lens_kv, 0)]).int()
    q = torch.randn(
        int(actual_seq_lens_q.sum()),
        num_qo_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v_cache = torch.randn(
        int(actual_seq_lens_kv.sum()),
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    scale = float(1.0 / (head_dim**0.5))
    # the legacy 'omitted' call (the second call of the legacy test)
    out_without, lse_without = cudnn_batch_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        scale=scale,
        workspace_buffer=torch.empty(
            512 * 1024 * 1024, dtype=torch.int8, device=device
        ),
        max_token_per_sequence=s_qo,
        max_sequence_kv=s_kv,
        causal=True,
        return_lse=True,
        batch_offsets_q=qo_indptr,
        batch_offsets_k=kv_indptr,
        batch_offsets_units="tokens",
    )

    # ---- unified: lengths derived from the metadata, page-16 pool ----
    k_pool, v_pool, table = _lay_into_pages(kv_indptr.cpu(), k_cache, v_cache)
    md = dense_metadata(qo_indptr, actual_seq_lens_kv, table, PAGE_SIZE)
    attn = plan_pinned(
        "cudnn",
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
    )
    out, lse = attn.run(q, (k_pool, v_pool), sm_scale=scale)
    torch.testing.assert_close(out, out_without, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse, _pack_lse(lse_without, qo_indptr.cpu()), atol=1e-2, rtol=1e-2
    )
    ref_out, ref_lse = oracle(
        md, q, k_pool, v_pool, causal=True, kv_layout="NHD", sm_scale=scale
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_cudnn_prefill_lse_is_base2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_cudnn_prefill_lse_is_base2(num_kv_heads):
    """cuDNN returns a base-2 LSE through the unified plan too (lse_mode
    "base2" folds the frontend's natural-log stats); the legacy float
    reference and budget, plus the oracle."""
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")

    from flashinfer.utils import log2e

    torch.manual_seed(0)
    device = "cuda:0"
    s_q, s_kv, num_qo_heads, head_dim = 37, 64, 8, 128
    scale = float(head_dim**-0.5)
    q = torch.randn(s_q, num_qo_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(s_kv, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    # the legacy float reference LSE in base-2 (GQA: kv head broadcast)
    kf = k.float().repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * scale  # [h_qo, s_q, s_kv]
    lse_ref = torch.logsumexp(scores, dim=-1) * log2e  # base-2, [h_qo, s_q]

    # batch 1: the 64 KV tokens are exactly four page-16 pages (identity table)
    kv_indptr_cpu = torch.tensor([0, s_kv], dtype=torch.int32)
    k_pool, v_pool, table = _lay_into_pages(kv_indptr_cpu, k, v)
    qo_indptr = torch.tensor([0, s_q], dtype=torch.int32, device=device)
    md = dense_metadata(
        qo_indptr, torch.tensor([s_kv], dtype=torch.int32), table, PAGE_SIZE
    )
    attn = plan_pinned(
        "cudnn",
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout="NHD",
        causal=False,
        lse_mode="base2",
    )
    out, lse = attn.run(q, (k_pool, v_pool), sm_scale=scale)
    assert lse.shape == (s_q, num_qo_heads) and lse.dtype == torch.float32
    torch.testing.assert_close(lse.transpose(0, 1), lse_ref, atol=1e-2, rtol=1e-2)
    ref_out, ref_lse = oracle(
        md, q, k_pool, v_pool, causal=False, kv_layout="NHD", sm_scale=scale
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
