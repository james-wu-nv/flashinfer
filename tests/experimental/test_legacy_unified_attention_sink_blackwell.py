"""Legacy -> unified: tests/attention/test_attention_sink_blackwell.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file is collected in the H100 1/5-sample lane and skips there
(SM100 / SM103 only), so it has never executed a kernel in CI; the unified
file is in no default lane (tests/experimental is excluded by norecursedirs).
The B200 runs of this round are among the first complete executions.

Both legacy functions run the trtllm-gen sink kernels on a full-page fixture
(seed 0, ``sink = rand(H) * 5``, arange block table, separate HND K/V pools,
``sm_scale`` 1.0): the context entry (``trtllm_batch_context_with_kv_cache``,
q_len = seq_len) and the decode entry (``trtllm_batch_decode_with_kv_cache``,
q_len 1).  Here both run the same fixture through ``PagedAttention`` in the
dense form on the pinned legacy backend ``trtllm-gen`` and, appended as one
extra parametrize axis (``backend``), on ``cake`` (the separately versioned
FMHA product behind the same front door); the assertions are the legacy
``sink_attention_unified`` reference at the legacy budget (fp16 2e-3 / 1e-3,
bf16 1e-2) and the sink-aware fp32 oracle (out + LSE).  Node ids are the
legacy ids plus the trailing backend component.  Default run = the full
legacy grids x 2 backends (288 ids).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- head_dim 64 rows (both functions): trtllm-gen and cake declare head_dims =
  {(128, 128)} only (_capabilities.py, capability-honesty rule: the legacy
  suite runs D64 natively but the unified conformance matrix has not
  exercised it), so the pinned resolve is rejected ("unsupported head dims
  (64, 64)", asserted behind EXPECT_TRTLLM_HEAD_DIM_64) and the row then
  runs the same fixture on fa2 (the fallback published backend is asserted).
  Flipping the flag turns the D64 rows into positive trtllm-gen / cake rows.
- test_blackwell_trtllm_gen_decode_attention_sink: the legacy decode entry
  maps onto the unified context path with one query token per request (there
  is no separate unified decode entry); the legacy incremental reference and
  budget are kept.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_TRTLLM_HEAD_DIM_64,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    gated,
    oracle,
    plan_pinned,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_attention_sink_blackwell.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_decode_attention_sink",
        ["test_blackwell_trtllm_gen_decode_attention_sink"],
        "partial",
        "the decode entry (trtllm_batch_decode_with_kv_cache, q_len 1) on the unified "
        "context path: same grid (fp16/bf16, B1/4/16, page32, seq 32/128/1024, H32:8/32, "
        "D64/128), seed 0, sm_scale 1.0, dense form, one query token per request, pinned "
        "trtllm-gen plus the appended cake axis; vs the incremental sink reference at the "
        "legacy budget (fp16 1e-3, bf16 1e-2) and the oracle; D64 is capability-excluded "
        "on both (EXPECT_TRTLLM_HEAD_DIM_64: rejection asserted) and runs on fa2 instead",
    ),
    (
        "tests/attention/test_attention_sink_blackwell.py::test_blackwell_trtllm_gen_context_attention_sink",
        ["test_blackwell_trtllm_gen_context_attention_sink"],
        "partial",
        "same grid (fp16/bf16, B1/4/16, page32, seq 32/128/1024, H32:8/32, D64/128), seed "
        "0, sm_scale 1.0, dense form, pinned trtllm-gen plus the appended cake axis; vs the "
        "prefill sink reference at the legacy budget (fp16 2e-3/1e-3, bf16 1e-2) and the "
        "oracle; D64 is capability-excluded on both (EXPECT_TRTLLM_HEAD_DIM_64: rejection "
        "asserted) and runs on fa2 instead",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


def _blackwell_fixture(
    dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim, q_len
):
    """The legacy Blackwell sink fixture: full pages, arange block table,
    separate HND K/V pools, ``sink = rand(H) * 5``; ``q_len`` is ``seq_len``
    for the context entry and 1 for the decode entry (RNG order as legacy)."""
    torch.manual_seed(0)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=DEVICE)
    blocks_per_seq = (seq_lens + page_size - 1) // page_size
    max_num_blocks_per_seq = torch.max(blocks_per_seq).item()
    block_tables = torch.arange(
        batch_size * max_num_blocks_per_seq, dtype=torch.int32, device=DEVICE
    ).reshape(batch_size, max_num_blocks_per_seq)
    num_tokens = seq_len * batch_size
    num_blocks = (num_tokens + page_size - 1) // page_size
    q = torch.randn(
        batch_size * q_len, num_qo_heads, head_dim, dtype=dtype, device=DEVICE
    )
    k_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=DEVICE
    )
    v_cache = torch.randn(
        num_blocks, num_kv_heads, page_size, head_dim, dtype=dtype, device=DEVICE
    )
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32) * q_len
    md = dense_metadata(qo_indptr, seq_lens.cpu(), block_tables, page_size)
    return dict(
        q=q, k_cache=k_cache, v_cache=v_cache, sink=sink, md=md, seq_lens=seq_lens
    )


def _pinned_backend(backend, *, num_qo_heads, num_kv_heads, dtype, head_dim, page_size):
    """The backend a Blackwell row runs on: the pinned one, or fa2 for the
    D64 rows after asserting the trtllm-gen / cake rejection
    (EXPECT_TRTLLM_HEAD_DIM_64)."""
    if head_dim == 64:
        res = gated(
            EXPECT_TRTLLM_HEAD_DIM_64,
            lambda: resolve_paged_attention(
                device=torch.device(DEVICE),
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                q_dtype=dtype,
                page_size=page_size,
                kv_layout="HND",
                causal=True,
                need_lse=True,
                sinks=True,
                backend=backend,
            ),
            match="unsupported head dims \\(64, 64\\)",
        )
        if res is None:
            return "fa2"
    return backend


def _run_blackwell(backend, f, *, num_qo_heads, num_kv_heads, dtype, head_dim):
    attn = plan_pinned(
        backend,
        f["md"],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        q_dtype=dtype,
        kv_layout="HND",
        causal=True,
        lse_mode="base2",
        use_sinks=True,
    )
    return attn.run(f["q"], (f["k_cache"], f["v_cache"]), sm_scale=1.0, sinks=f["sink"])


def _blackwell_tol(dtype, *, context):
    if dtype == torch.float16:
        return dict(atol=2e-3, rtol=1e-3) if context else dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-2, rtol=1e-2)


def _check_oracle(f, out, lse):
    ref_out, ref_lse = oracle(
        f["md"],
        f["q"],
        f["k_cache"],
        f["v_cache"],
        causal=True,
        kv_layout="HND",
        sm_scale=1.0,
        sinks=f["sink"],
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# test_blackwell_trtllm_gen_decode_attention_sink (q_len 1 on the context path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["trtllm-gen", "cake"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_decode_attention_sink(
    backend,
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    import einops
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    backend = _pinned_backend(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
        page_size=page_size,
    )
    f = _blackwell_fixture(
        dtype, batch_size, page_size, seq_len, num_qo_heads, num_kv_heads, head_dim, 1
    )
    out, lse = _run_blackwell(
        backend,
        f,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
    )
    max_num_blocks_per_seq = f["md"].block_tables.shape[1]
    k = einops.rearrange(
        f["k_cache"], "(b n) h p d -> b (n p) h d", n=max_num_blocks_per_seq
    )
    v = einops.rearrange(
        f["v_cache"], "(b n) h p d -> b (n p) h d", n=max_num_blocks_per_seq
    )
    o_ref = sink_attention_unified(
        f["q"], k, v, f["sink"], -1, False, 1.0, mode="incremental"
    )
    torch.testing.assert_close(
        o_ref, out, **_blackwell_tol(dtype, context=False)
    )  # legacy
    _check_oracle(f, out, lse)


# ---------------------------------------------------------------------------
# test_blackwell_trtllm_gen_context_attention_sink
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["trtllm-gen", "cake"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("seq_len", [32, 128, 1024])
@pytest.mark.parametrize("num_qo_heads", [32])
@pytest.mark.parametrize("num_kv_heads", [8, 32])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_blackwell_trtllm_gen_context_attention_sink(
    backend,
    dtype,
    batch_size,
    page_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
):
    import einops
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    backend = _pinned_backend(
        backend,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
        page_size=page_size,
    )
    f = _blackwell_fixture(
        dtype,
        batch_size,
        page_size,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        seq_len,
    )
    out, lse = _run_blackwell(
        backend,
        f,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        dtype=dtype,
        head_dim=head_dim,
    )
    k = einops.rearrange(f["k_cache"], "num_pages h p d -> (num_pages p) h d")
    v = einops.rearrange(f["v_cache"], "num_pages h p d -> (num_pages p) h d")
    o_ref = sink_attention_unified(
        f["q"], k, v, f["sink"], -1, True, 1.0, mode="prefill", batch_size=batch_size
    )
    torch.testing.assert_close(
        o_ref, out, **_blackwell_tol(dtype, context=True)
    )  # legacy
    _check_oracle(f, out, lse)
