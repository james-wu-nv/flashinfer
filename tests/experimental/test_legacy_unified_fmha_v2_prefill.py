"""Legacy -> unified: tests/attention/test_fmha_v2_prefill.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane samples it at
1/5 and runs its sm90a branch there; the sm120 branch never executed in CI
(reports/unified-prefill-round4-20260918/ci-status.md).  On B200 every legacy
row skips (FMHA v2 is SM90 / SM12x), so this file is the first record of
these WORKLOADS on an SM100 part.

FMHA v2 (``trtllm_fmha_v2_prefill``) is not a unified backend
(EXPECT_FMHA_V2_CANDIDATE): its paged entry points take a ``(q, 5-D stacked
pool)`` or ``(q, (k_cache, v_cache))`` + ``[B, 2, M]`` table calling
convention, an fp8 q with an independent ``out_dtype``, softmax statistics in
``[max, sum_exp]`` form and a ``chunked_attention_size`` knob.  What this file
converts is the paged WORKLOAD of each legacy function: the legacy fixture
(seed 42, ``run_trtllm_fmha_v2_prefill_case``'s ``Q_PAGED_KV_NHD`` /
``Q_PAGED_KV_HND`` branch) runs on every unified backend that resolves (fa2,
trtllm-gen, cake, cuDNN; the excluded ones are recorded with their reason)
and on ``auto`` (the serving backend is recorded), asserted against the legacy
``attention_ref_torch`` / ``chunked_attention_ref_torch`` reference at the
legacy budget (1e-2) and the fp32 oracle.  The legacy parametrize axes and
values are kept; the non-paged layouts (``PACKED_QKV``, ``CONTIGUOUS_Q_KV``,
``SEPARATE_Q_K_V``) skip with the reason (ragged: out of the paged scope).
Batch 16 is ``slow`` (``FI_PARITY_SLOW=1``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_trtllm_fmha_v2_prefill: fp8 QKV rows (three dtype pairs) assert the
  fp8-q rejection (EXPECT_FP8_Q) and the absence of an output dtype knob
  (EXPECT_OUTPUT_DTYPE); ``save_softmax_stats`` only exists for the
  CONTIGUOUS layout (legacy skip kept); the ``[max, sum_exp]`` statistics form
  has no unified counterpart (the LSE contract is one base-2 number).
  logits_soft_cap 30 resolves on fa2 only; sliding window excludes cuDNN;
  D256 excludes trtllm-gen / cake / cuDNN (capability tables).
- test_trtllm_fmha_v2_prefill_non_interleaved_kv: the pre-expanded [B, 2, M]
  tables are independent page-id mappings (EXPECT_INDEPENDENT_KV_TABLES,
  asserted); the legacy "K and V as block-aligned halves of one allocation"
  IS expressible with the shared table (two 4-D views of the fused buffer),
  so that form runs against the stacked-pool form (legacy 1e-3) and the
  oracle.
- test_trtllm_fmha_v2_prefill_sm120_large_head_dim: D256 / D512 resolve on
  fa2 only (WP-T declared (512, 512) on fa2); PADDING is non-causal; the
  CONTIGUOUS rows skip (ragged).
- test_trtllm_fmha_v2_prefill_skip_softmax: ``skip_softmax_threshold_scale_
  factor`` (approximate softmax, loosened tolerance) has no plan / run knob
  (EXPECT_SKIP_SOFTMAX); every id asserts its absence (the 16k-token
  workload without the knob is plain causal attention already covered).
- test_trtllm_fmha_v2_prefill_attention_sinks: the legacy layout is ragged
  (SEPARATE_Q_K_V) and its reference is the page-1 paged AttentionSink
  wrapper (fa3, H100 only); the conversion lays the fixture out as page-1
  CSR (one page per token, the legacy reference's own form) and runs the
  sink-aware unified path (fa2: trtllm-gen / cake need page >= 16) against
  ``sink_attention_unified`` (the sink reference of the trtllm-gen legacy
  suite) at 1e-2 and the sink-aware oracle.
- chunked attention (two functions): the exact chunked visible set
  (``col <= row and col >= floor(row / C) * C``) is a fa2 custom mask; a
  ``chunked_attention_size`` plan axis (EXPECT_CHUNKED_ATTENTION_KNOB) would
  avoid the O(sum q_i * kv_i) mask bits.
- out-of-scope: the eight ``fmha_v2_prefill_deepseek`` / ``fmha_v2_prefill_
  sm120`` functions (fixed-length BSHD fp8 self-attention API, not paged) and
  ``test_trtllm_fmha_v2_prefill_sm120_chunked_rejected`` (CONTIGUOUS layout).
"""

import inspect
import math

import pytest
import torch

from flashinfer.experimental.paged_attention import CAPABILITIES
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_CHUNKED_ATTENTION_KNOB,
    EXPECT_FMHA_V2_CANDIDATE,
    EXPECT_INDEPENDENT_KV_TABLES,
    EXPECT_OUTPUT_DTYPE,
    EXPECT_SKIP_SOFTMAX,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    csr_metadata_page1,
    dense_metadata,
    fp8_q_rejected,
    gated,
    oracle,
    output_dtype_knob_present,
    resolve_or_skip,
    run_on_backends,
    skip_softmax_knob_present,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_fmha_v2_prefill.py"


LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek_cuda_graph",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_self_attention",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_optional_device_scales",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_validation",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek_validates_seq_len",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_cuda_graph",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_async_enqueue",
        [],
        "out-of-scope",
        "fixed-length BSHD fp8 self-attention API (fmha_v2_prefill_deepseek / "
        "fmha_v2_prefill_sm120, SM12x): not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill",
        ["test_trtllm_fmha_v2_prefill"],
        "partial",
        "same grid and node ids (B1/16 x seq1024 x H4/32:4 x D128/256 x 5 dtype pairs "
        "x 8 layouts x 3 masks x soft cap 0/30; B16 slow); the four paged layout rows "
        "(NHD/HND x page 32/128) in fp16/bf16 run the legacy fixture (seed 42) on "
        "every resolving unified backend and auto (recorded) vs attention_ref_torch "
        "at 1e-2 and the oracle; fp8 pairs assert EXPECT_FP8_Q + EXPECT_OUTPUT_DTYPE; "
        "non-paged layouts skip (ragged); save_softmax_stats keeps the legacy skip; "
        "the FMHA v2 entry point itself is not a unified candidate "
        "(EXPECT_FMHA_V2_CANDIDATE)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_non_interleaved_kv",
        ["test_trtllm_fmha_v2_prefill_non_interleaved_kv"],
        "partial",
        "same grid (NHD/HND x page 32/128 x fp16/bf16 x Hkv 1/4; B4, seq1024, H8, "
        "D128, causal); the stacked pool and the fused two-halves allocation (K = "
        "fused[0], V = fused[1], shared table) run on every resolving backend and "
        "agree at the legacy 1e-3 and with the oracle; the pre-expanded [B, 2, M] "
        "table asserts EXPECT_INDEPENDENT_KV_TABLES",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_sm120_large_head_dim",
        ["test_trtllm_fmha_v2_prefill_sm120_large_head_dim"],
        "partial",
        "same grid (B1/4 x seq1024 x H8:2 x D256/512 x fp16/bf16 x 3 layouts x 4 "
        "masks); the paged NHD rows (page 32/128) run on fa2 (the only backend "
        "declaring D256/D512; trtllm-gen / cake / cuDNN excluded with the reason) and "
        "auto vs attention_ref_torch at 1e-2 and the oracle; PADDING = non-causal; "
        "CONTIGUOUS rows skip (ragged); the legacy itself skips on non-SM12x",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_sm120_chunked_rejected",
        [],
        "out-of-scope",
        "CONTIGUOUS_Q_KV (ragged) request; the chunked knob it rejects is "
        "EXPECT_CHUNKED_ATTENTION_KNOB on the unified side",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_skip_softmax",
        ["test_trtllm_fmha_v2_prefill_skip_softmax"],
        "unsupported-by-design",
        "skip_softmax_threshold_scale_factor (approximate softmax with a loosened "
        "tolerance) has no plan / run spelling (EXPECT_SKIP_SOFTMAX); every id of the "
        "legacy grid asserts its absence; the seq-16384 workload without the knob is "
        "plain causal attention",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_attention_sinks",
        ["test_trtllm_fmha_v2_prefill_attention_sinks"],
        "partial",
        "same grid (B4/16 x seq1024/4096 x H4/32:4 x D128 x fp16/bf16 x 3 masks; "
        "seq4096 slow); the legacy SEPARATE_Q_K_V fixture (seed 42, sinks rand*5) as "
        "page-1 CSR (the legacy reference's own paged form) on the sink-aware unified "
        "path (fa2; trtllm-gen / cake need page >= 16) and auto vs "
        "sink_attention_unified at 1e-2 and the sink-aware oracle (out + LSE)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_chunked_attention",
        ["test_trtllm_fmha_v2_prefill_chunked_attention"],
        "partial",
        "same grid (B1/4 x seq1024/4096 x H4/32:4 x D128 x fp16/bf16 x 3 layouts x "
        "chunk 64/256); the Q_PAGED_KV_NHD rows (page 32) with the exact chunked "
        "visible set as a fa2 custom mask vs chunked_attention_ref_torch at 1e-2 and "
        "the masked oracle; CONTIGUOUS / SEPARATE rows skip (ragged); a "
        "chunked_attention_size plan axis (EXPECT_CHUNKED_ATTENTION_KNOB) is not "
        "wired",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_chunked_prefill_chunked_attention",
        ["test_trtllm_fmha_v2_chunked_prefill_chunked_attention"],
        "partial",
        "same grid (B1/4 x kv1024/4096 x new64/256 x H4/32:4 x D128 x fp16/bf16 x "
        "page 32/128 x chunk 64/256) as a fa2 custom mask over the absolute positions "
        "kv_len - q_len + r vs chunked_attention_ref_torch at 1e-2 and the masked "
        "oracle; knob as above",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def test_fmha_v2_is_not_a_unified_candidate():
    """FMHA v2 (trtllm_fmha_v2_prefill, SM90 / SM12x) has no capability row
    (EXPECT_FMHA_V2_CANDIDATE); an adapter would need cc 9 / 12, the (q, 5-D
    pool) calling convention, the [max, sum_exp] -> LSE normalisation and the
    chunked mask as a plan axis."""
    cap = gated(
        EXPECT_FMHA_V2_CANDIDATE,
        lambda: CAPABILITIES["fmha_v2"],
        match="fmha_v2",
        exc=KeyError,
    )
    if cap is not None:
        pytest.fail("EXPECT_FMHA_V2_CANDIDATE flipped: pin fmha_v2 in the rows above")


# ---------------------------------------------------------------------------
# the legacy paged fixture (run_trtllm_fmha_v2_prefill_case, seed 42)
# ---------------------------------------------------------------------------

_PAGED_LAYOUTS = ("Q_PAGED_KV_NHD", "Q_PAGED_KV_HND")


def _skip_non_paged(input_layout):
    if input_layout not in _PAGED_LAYOUTS:
        pytest.skip(f"{input_layout}: ragged legacy layout, out of the paged scope")


def _legacy_paged_case(
    input_layout,
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    *,
    q_seq_lens=None,
):
    """The Q_PAGED_KV_* branch of the legacy fixture: seq_lens in
    [max/2, max], a randn stacked pool (NHD or HND), a randn packed q and the
    sequential block table; ``q_seq_lens`` (chunked prefill) shortens q."""
    torch.manual_seed(42)
    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=DEVICE,
    )
    if q_seq_lens is not None:
        q_lens = torch.minimum(q_seq_lens(), seq_lens)
    else:
        q_lens = seq_lens
    max_kv_len = int(seq_lens.max())
    cum_seq_lens_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens_kv[1:] = torch.cumsum(seq_lens, dim=0)
    cum_seq_lens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens_q[1:] = torch.cumsum(q_lens, dim=0)
    total_q = int(cum_seq_lens_q[-1])
    is_nhd = input_layout == "Q_PAGED_KV_NHD"
    max_num_blocks = (max_kv_len + page_size - 1) // page_size
    num_pages = batch_size * max_num_blocks
    paged_shape = (
        (num_pages, 2, page_size, num_kv_heads, head_dim)
        if is_nhd
        else (num_pages, 2, num_kv_heads, page_size, head_dim)
    )
    paged_kv_cache = torch.randn(*paged_shape, dtype=dtype, device=DEVICE)
    q = torch.randn(total_q, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    block_tables = torch.zeros(
        batch_size, max_num_blocks, dtype=torch.int32, device=DEVICE
    )
    for i in range(batch_size):
        num_blocks_needed = (int(seq_lens[i]) + page_size - 1) // page_size
        block_tables[i, :num_blocks_needed] = torch.arange(
            i * max_num_blocks, i * max_num_blocks + num_blocks_needed, device=DEVICE
        )
    return dict(
        seq_lens=seq_lens,
        q_lens=q_lens,
        cum_seq_lens_q=cum_seq_lens_q,
        cum_seq_lens_kv=cum_seq_lens_kv,
        paged_kv_cache=paged_kv_cache,
        q=q,
        block_tables=block_tables,
        kv_layout="NHD" if is_nhd else "HND",
        # attention_ref_torch reads an NHD pool
        ref_pool=paged_kv_cache
        if is_nhd
        else paged_kv_cache.transpose(-3, -2).contiguous(),
        sm_scale=1.0 / math.sqrt(head_dim),
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )


def _md(c):
    return dense_metadata(
        c["cum_seq_lens_q"], c["seq_lens"].cpu(), c["block_tables"], c["page_size"]
    )


def _legacy_reference(c, *, causal, window_left, logits_soft_cap):
    from tests.attention.test_fmha_v2_prefill import attention_ref_torch

    return attention_ref_torch(
        (c["q"], c["ref_pool"]),
        seq_lens=c["seq_lens"],
        cum_seq_lens_q=c["cum_seq_lens_q"],
        sm_scale=c["sm_scale"],
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        block_tables=c["block_tables"],
        cum_seq_lens_kv=c["cum_seq_lens_kv"],
    )


def _run_paged_case(c, *, causal, window_left, logits_soft_cap, record_property):
    """Every resolving backend + auto vs the legacy reference (1e-2) and the oracle."""
    md = _md(c)
    k, v = c["paged_kv_cache"][:, 0], c["paged_kv_cache"][:, 1]
    cap = logits_soft_cap if logits_soft_cap > 0 else None
    results = run_on_backends(
        md,
        c["q"],
        (k, v),
        num_qo_heads=c["num_qo_heads"],
        num_kv_heads=c["num_kv_heads"],
        head_dim_qk=c["head_dim"],
        q_dtype=c["dtype"],
        kv_layout=c["kv_layout"],
        causal=causal,
        window_left=window_left,
        logits_soft_cap=cap,
        sm_scale=c["sm_scale"],
        record_property=record_property,
    )
    ref = _legacy_reference(
        c, causal=causal, window_left=window_left, logits_soft_cap=logits_soft_cap
    )
    o_out, o_lse = oracle(
        md,
        c["q"],
        k,
        v,
        causal=causal,
        kv_layout=c["kv_layout"],
        sm_scale=c["sm_scale"],
        window_left=window_left,
        logits_soft_cap=cap,
    )
    for _name, _served, out, lse in results:
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    return results


# ---------------------------------------------------------------------------
# test_trtllm_fmha_v2_prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, pytest.param(16, marks=pytest.mark.slow)])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    ("dtype", "o_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
    ],
)
@pytest.mark.parametrize(
    ("input_layout", "page_size", "save_softmax_stats"),
    [
        ("PACKED_QKV", None, False),
        ("CONTIGUOUS_Q_KV", None, False),
        ("CONTIGUOUS_Q_KV", None, True),
        ("SEPARATE_Q_K_V", None, False),
        ("Q_PAGED_KV_NHD", 32, False),
        ("Q_PAGED_KV_NHD", 128, False),
        ("Q_PAGED_KV_HND", 32, False),
        ("Q_PAGED_KV_HND", 128, False),
    ],
)
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 512, "SLIDING_WINDOW"),
    ],
)
@pytest.mark.parametrize("pos_encoding_mode", [None])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_trtllm_fmha_v2_prefill(
    input_layout: str,
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size,
    dtype: torch.dtype,
    o_dtype: torch.dtype,
    causal: bool,
    mask_mode: str,
    window_left: int,
    logits_soft_cap: float,
    pos_encoding_mode,
    save_softmax_stats: bool,
    record_property,
) -> None:
    _skip_non_paged(input_layout)
    if save_softmax_stats:
        pytest.skip("save_softmax_stats is a CONTIGUOUS_Q_KV-only legacy branch")
    kv_layout = "NHD" if input_layout == "Q_PAGED_KV_NHD" else "HND"
    if dtype == torch.float8_e4m3fn:
        assert output_dtype_knob_present() == EXPECT_OUTPUT_DTYPE
        res = fp8_q_rejected(
            backend="fa2",
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
        )
        if res is not None:
            pytest.fail(
                f"EXPECT_FP8_Q flipped: port the fp8 -> {o_dtype} legacy fixture here"
            )
        return
    c = _legacy_paged_case(
        input_layout,
        batch_size,
        max_seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
    )
    _run_paged_case(
        c,
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        record_property=record_property,
    )


# ---------------------------------------------------------------------------
# test_trtllm_fmha_v2_prefill_non_interleaved_kv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_layout", ["Q_PAGED_KV_NHD", "Q_PAGED_KV_HND"])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_trtllm_fmha_v2_prefill_non_interleaved_kv(
    input_layout: str,
    page_size: int,
    dtype: torch.dtype,
    num_kv_heads: int,
    record_property,
) -> None:
    """The legacy parity of the stacked pool against separate (k_cache,
    v_cache) halves of one allocation with pre-expanded [B, 2, M] tables.
    The halves with the SHARED table are two 4-D views (expressible); the
    [B, 2, M] table is not (EXPECT_INDEPENDENT_KV_TABLES)."""
    c = _legacy_paged_case(
        input_layout, 4, 1024, 8, num_kv_heads, 128, page_size, dtype
    )
    md = _md(c)
    paged = c["paged_kv_cache"]
    fused = torch.empty(2, *paged[:, 0].shape, dtype=dtype, device=DEVICE)
    fused[0].copy_(paged[:, 0])
    fused[1].copy_(paged[:, 1])
    k_cache, v_cache = fused[0], fused[1]  # delta == num_pages in the legacy table
    kv_tables = torch.stack(
        [c["block_tables"], c["block_tables"] + paged.shape[0]], dim=1
    ).int()  # [B, 2, M]
    assert kv_tables.shape == (4, 2, c["block_tables"].shape[1])
    rejected = gated(
        EXPECT_INDEPENDENT_KV_TABLES,
        lambda: PagedAttentionMetadata.dense(
            md.qo_indptr,
            md.kv_seq_lens,
            kv_tables[:, 0].contiguous(),
            v_block_tables=kv_tables[:, 1].contiguous(),
            page_size=page_size,
            max_q_len=md.max_q_len,
            max_kv_len=md.max_kv_len,
        ),
        match="v_block_tables",
        exc=TypeError,
    )
    if rejected is not None:
        pytest.fail("EXPECT_INDEPENDENT_KV_TABLES flipped: port the [B, 2, M] run here")
    common = dict(
        num_qo_heads=8,
        num_kv_heads=num_kv_heads,
        head_dim_qk=128,
        q_dtype=dtype,
        kv_layout=c["kv_layout"],
        causal=True,
        sm_scale=c["sm_scale"],
    )
    stacked = run_on_backends(
        md,
        c["q"],
        (paged[:, 0], paged[:, 1]),
        record_property=record_property,
        **common,
    )
    separate = run_on_backends(
        md, c["q"], (k_cache, v_cache), include_auto=False, **common
    )
    o_out, o_lse = oracle(
        md,
        c["q"],
        paged[:, 0],
        paged[:, 1],
        causal=True,
        kv_layout=c["kv_layout"],
        sm_scale=c["sm_scale"],
    )
    by_name = {name: (out, lse) for name, _served, out, lse in separate}
    for name, _served, out, lse in stacked:
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
        if name in by_name:
            out_sep, lse_sep = by_name[name]
            torch.testing.assert_close(
                out_sep.float(), out.float(), rtol=1e-3, atol=1e-3
            )  # legacy
            torch.testing.assert_close(lse_sep, lse, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# test_trtllm_fmha_v2_prefill_sm120_large_head_dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [2])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("input_layout", "page_size"),
    [
        ("CONTIGUOUS_Q_KV", None),
        ("Q_PAGED_KV_NHD", 32),
        ("Q_PAGED_KV_NHD", 128),
    ],
)
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (False, -1, "PADDING"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 1024, "SLIDING_WINDOW"),
    ],
)
def test_trtllm_fmha_v2_prefill_sm120_large_head_dim(
    input_layout: str,
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size,
    dtype: torch.dtype,
    causal: bool,
    window_left: int,
    mask_mode: str,
    record_property,
) -> None:
    _skip_non_paged(input_layout)
    c = _legacy_paged_case(
        input_layout,
        batch_size,
        max_seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
    )
    results = _run_paged_case(
        c,
        causal=causal,
        window_left=window_left,
        logits_soft_cap=0.0,
        record_property=record_property,
    )
    # D256 / D512 resolve on fa2 only today (capability tables)
    assert {name for name, _s, _o, _l in results} <= {"fa2", "auto"}
    assert all(served == "fa2" for _n, served, _o, _l in results)


# ---------------------------------------------------------------------------
# test_trtllm_fmha_v2_prefill_skip_softmax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [16384])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    ("dtype", "o_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "input_layout", ["CONTIGUOUS_Q_KV", "Q_PAGED_KV_NHD", "Q_PAGED_KV_HND"]
)
@pytest.mark.parametrize(
    (
        "skip_softmax_threshold_scale_factor",
        "rtol",
        "atol",
    ),
    [
        (500.0, 2e-2, 1.2e-1),
        (10000.0, 2e-2, 2e-1),
    ],
)
def test_trtllm_fmha_v2_prefill_skip_softmax(
    input_layout: str,
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    o_dtype: torch.dtype,
    skip_softmax_threshold_scale_factor: float,
    rtol: float,
    atol: float,
) -> None:
    _skip_non_paged(input_layout)
    assert skip_softmax_knob_present() == EXPECT_SKIP_SOFTMAX
    if EXPECT_SKIP_SOFTMAX:
        pytest.fail(
            "EXPECT_SKIP_SOFTMAX flipped: run the legacy 16k-token fixture with "
            f"threshold {skip_softmax_threshold_scale_factor} at ({rtol}, {atol}) here"
        )


# ---------------------------------------------------------------------------
# test_trtllm_fmha_v2_prefill_attention_sinks (page-1 CSR, the legacy
# reference's own paged form)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize(
    "max_seq_len", [1024, pytest.param(4096, marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 512, "SLIDING_WINDOW"),
    ],
)
@pytest.mark.parametrize("pos_encoding_mode", [None])
def test_trtllm_fmha_v2_prefill_attention_sinks(
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    causal: bool,
    window_left: int,
    mask_mode: str,
    pos_encoding_mode,
    record_property,
) -> None:
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    torch.manual_seed(42)
    seq_lens = torch.randint(
        max_seq_len // 2,
        max_seq_len + 1,
        (batch_size,),
        dtype=torch.int32,
        device=DEVICE,
    )
    cum_seq_lens = torch.zeros(batch_size + 1, dtype=torch.int32, device=DEVICE)
    cum_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    total_tokens = int(cum_seq_lens[-1])
    q = torch.randn(total_tokens, num_qo_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=DEVICE)
    sm_scale = 1.0 / math.sqrt(head_dim)
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    ref = sink_attention_unified(
        q,
        k,
        v,
        sink,
        window_left,
        causal,
        sm_scale,
        mode="varlen",
        batch_size=batch_size,
        qo_indptr=cum_seq_lens,
        kv_indptr=cum_seq_lens,
    )
    # one page per token: (pages, 1, Hkv, D) NHD views of the ragged tensors
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=DEVICE)
    md = csr_metadata_page1(cum_seq_lens, cum_seq_lens, kv_indices)
    k1 = k.view(total_tokens, 1, num_kv_heads, head_dim)
    v1 = v.view(total_tokens, 1, num_kv_heads, head_dim)
    results = run_on_backends(
        md,
        q,
        (k1, v1),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=causal,
        window_left=window_left,
        sinks=sink,
        sm_scale=sm_scale,
        record_property=record_property,
    )
    o_out, o_lse = oracle(
        md,
        q,
        k1,
        v1,
        causal=causal,
        kv_layout="NHD",
        sm_scale=sm_scale,
        window_left=window_left,
        sinks=sink,
    )
    for _name, _served, out, lse in results:
        torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# chunked attention as an exact fa2 custom mask
# ---------------------------------------------------------------------------


def _chunked_mask(q_lens, kv_lens, chunk):
    """The exact FMHA v2 chunked visible set, flattened per request in
    request order: for query r of a request (absolute row = kv_len - q_len +
    r), allowed iff col <= row and col >= floor(row / chunk) * chunk."""
    parts = []
    for lq, lkv in zip(q_lens, kv_lens, strict=True):
        lq, lkv = int(lq), int(lkv)
        rows = torch.arange(lq, device=DEVICE).unsqueeze(1) + (lkv - lq)
        cols = torch.arange(lkv, device=DEVICE).unsqueeze(0)
        allowed = (cols <= rows) & (cols >= (rows // chunk) * chunk)
        parts.append(allowed.reshape(-1))
    return torch.cat(parts)


def _fa2_masked(md, q, k, v, *, num_qo_heads, num_kv_heads, head_dim, dtype, mask):
    res = resolve_or_skip(
        "fa2",
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        page_size=md.page_size,
        kv_layout="NHD",
        causal=True,
        need_lse=True,
        custom_mask=True,
    )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="NHD",
        causal=True,
        lse_mode="base2",
        custom_mask=mask,
        backend=res,
    )
    assert attn.backend == "fa2"
    return attn.run(q, (k, v))


def _assert_no_chunk_knob():
    plan_params = inspect.signature(PagedAttention.plan).parameters
    assert ("chunked_attention_size" in plan_params) == EXPECT_CHUNKED_ATTENTION_KNOB


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [1024, 4096])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("input_layout", "page_size"),
    [
        ("CONTIGUOUS_Q_KV", None),
        ("SEPARATE_Q_K_V", None),
        ("Q_PAGED_KV_NHD", 32),
    ],
)
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_trtllm_fmha_v2_prefill_chunked_attention(
    batch_size: int,
    max_seq_len: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    input_layout: str,
    page_size,
    chunked_attention_size: int,
) -> None:
    from tests.attention.test_fmha_v2_prefill import chunked_attention_ref_torch

    _skip_non_paged(input_layout)
    _assert_no_chunk_knob()
    c = _legacy_paged_case(
        input_layout,
        batch_size,
        max_seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
    )
    seq_cpu = c["seq_lens"].cpu()
    mask = _chunked_mask(seq_cpu, seq_cpu, chunked_attention_size)
    md = _md(c)
    k, v = c["paged_kv_cache"][:, 0], c["paged_kv_cache"][:, 1]
    out, lse = _fa2_masked(
        md,
        c["q"],
        k,
        v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        mask=mask,
    )
    output_ref = chunked_attention_ref_torch(
        (c["q"], c["paged_kv_cache"]),
        seq_lens=c["seq_lens"],
        cum_seq_lens_q=c["cum_seq_lens_q"],
        sm_scale=c["sm_scale"],
        chunked_attention_size=chunked_attention_size,
        block_tables=c["block_tables"],
    )
    torch.testing.assert_close(
        out.float(), output_ref.float(), rtol=1e-2, atol=1e-2
    )  # legacy
    o_out, o_lse = oracle(
        md, c["q"], k, v, causal=True, kv_layout="NHD", custom_mask=mask
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    # the chunk boundary really excluded keys: differs from plain causal
    plain_out, _ = oracle(md, c["q"], k, v, causal=True, kv_layout="NHD")
    assert not torch.allclose(o_out, plain_out, **OUT_TOL)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_kv_len", [1024, 4096])
@pytest.mark.parametrize("max_new_tokens", [64, 256])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_trtllm_fmha_v2_chunked_prefill_chunked_attention(
    batch_size: int,
    max_kv_len: int,
    max_new_tokens: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    page_size: int,
    chunked_attention_size: int,
) -> None:
    from tests.attention.test_fmha_v2_prefill import chunked_attention_ref_torch

    _assert_no_chunk_knob()

    def q_seq_lens():
        # the legacy draw order: kv lengths first, then the q lengths
        return torch.randint(
            max(1, max_new_tokens // 2),
            max_new_tokens + 1,
            (batch_size,),
            dtype=torch.int32,
            device=DEVICE,
        )

    c = _legacy_paged_case(
        "Q_PAGED_KV_NHD",
        batch_size,
        max_kv_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        dtype,
        q_seq_lens=q_seq_lens,
    )
    mask = _chunked_mask(c["q_lens"].cpu(), c["seq_lens"].cpu(), chunked_attention_size)
    md = _md(c)
    k, v = c["paged_kv_cache"][:, 0], c["paged_kv_cache"][:, 1]
    out, lse = _fa2_masked(
        md,
        c["q"],
        k,
        v,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        mask=mask,
    )
    output_ref = chunked_attention_ref_torch(
        (c["q"], c["paged_kv_cache"]),
        seq_lens=c["seq_lens"],
        cum_seq_lens_q=c["cum_seq_lens_q"],
        sm_scale=c["sm_scale"],
        chunked_attention_size=chunked_attention_size,
        cum_seq_lens_kv=c["cum_seq_lens_kv"],
        block_tables=c["block_tables"],
    )
    torch.testing.assert_close(
        out.float(), output_ref.float(), rtol=1e-2, atol=1e-2
    )  # legacy
    o_out, o_lse = oracle(
        md, c["q"], k, v, causal=True, kv_layout="NHD", custom_mask=mask
    )
    torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
    torch.testing.assert_close(lse, o_lse, **LSE_TOL)
