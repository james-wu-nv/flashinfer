"""Legacy -> unified: tests/attention/test_xqa_batch_decode.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: the legacy file is in no A10G fixed shard; the H100 lane samples it at
1/5 (XQA runs on SM90 / SM100 / SM12x; the nvfp4 rows are SM12x-only)
(reports/unified-prefill-round4-20260918/ci-status.md).

Scope (round-4 brief): the functions whose reference is the paged prefill
wrapper -- ``test_xqa_batch_decode`` (speculative-decode rows, q_len_per_req
> 1; the q_len 1 rows use the tensor-core decode wrapper and are kept, the
unified prefill path serves them too), ``test_xqa_batch_decode_ragged_q``,
``test_xqa_batch_decode_nvfp4_kv`` and, because it delegates to
``test_xqa_batch_decode`` with a truncating window,
``test_xqa_batch_decode_spec_dec_sliding_window``.  The XQA kernel
(``xqa_batch_decode_with_kv_cache``) is not a unified backend; what runs here
is each row's WORKLOAD: the legacy fixture (seed 0, the decode-test helpers of
the legacy module: query, stacked fp16/bf16/fp8 pool, page table, sinks) in
the dense form on every unified backend that resolves (fa2 first -- the legacy
reference kernel, which also declares fp8 KV with per-tensor scales and
D512 -- then trtllm-gen, cake, cuDNN; excluded ones recorded with the resolve
reason) and on ``auto`` (recorded), asserted against the legacy reference
(the fa2 paged prefill wrapper / tensor-core decode wrapper on the dequantized
pool, or ``sink_attention_unified``) at the legacy budget (1e-2; 1e-1 for an
fp8 KV) and the fp32 oracle on the dequantized pool.  The legacy parametrize
axes and values are kept; ``enable_pdl`` has no unified meaning (its True /
False values are ``slow``, ``None`` is the default row) and the batch-128 /
256 shapes are ``slow`` (``FI_PARITY_SLOW=1``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- the packed uint16 draft mask (``generate_spec_dec_mask`` /
  ``generate_ragged_spec_dec_mask``): the legacy modes are exactly
  bottom-right causal (``"causal"``) and fully visible (``"full"``), which the
  unified ``causal`` flag spells; the packed-mask ENCODING itself has no
  unified spelling (a general draft mask would be a fa2 ``custom_mask``).
- fp8 output (``o_dtype="fp8"`` with its random ``o_scale``): no output dtype
  independent of q (EXPECT_OUTPUT_DTYPE, asserted); the same workload runs
  with the q dtype as output.
- ``"full"`` mask + window 127 (non-causal + sliding window): no unified
  backend declares it (M20), those ids skip with every resolve reason; XQA
  supports it natively.
- fp8 KV rows resolve on fa2 only (declared e4m3 KV with ``k_scale`` /
  ``v_scale``); D512 rows resolve on fa2 only.
- sinks + fp8 KV: the fa2 AttentionSink variant declines the pair at plan
  time ("not verified"); asserted as the clear rejection behind
  EXPECT_FA2_SINKS_FP8_KV (XQA runs it natively).
- sinks + head_dim 512 on fa2: WRONG results (55% of the elements off by up
  to 0.38 vs the sink reference and the oracle) although the capability table
  admits the pair -- a measured library defect, recorded as a non-strict
  xfail behind EXPECT_FA2_SINKS_HEAD_DIM_512 (D128 / D256 sinks and D512
  without sinks are exact).
- nvfp4 KV (``test_xqa_batch_decode_nvfp4_kv``): packed uint8 KV plus block
  scale factors have no spelling (EXPECT_NVFP4_KV, asserted per id); the
  legacy rows are SM12x-only.
- ``test_xqa_batch_decode_mask_mode_deterministic`` does not use the prefill
  wrapper (XQA against an analytic expectation): out-of-scope per the brief;
  it would convert directly (zero q / k, powers-of-two V).
"""

import pytest
import torch

import flashinfer

from flashinfer.prefill import PagedAttention, resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FA2_SINKS_FP8_KV,
    EXPECT_FA2_SINKS_HEAD_DIM_512,
    EXPECT_NVFP4_KV,
    EXPECT_OUTPUT_DTYPE,
    LSE_TOL,
    OUT_TOL,
    check_legacy_map,
    check_legacy_map_complete,
    dense_metadata,
    gated,
    legacy_workspace,
    oracle,
    output_dtype_knob_present,
    run_on_backends,
    xfail_unless,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_xqa_batch_decode.py"

LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode",
        ["test_xqa_batch_decode"],
        "partial",
        "same grid and node ids (13 shapes incl. D512 x window -1/127 x 6 dtype "
        "triples x pdl x sink x kv110 x NHD/HND x causal/full; batch 128/256 and pdl "
        "True/False slow): the legacy fixture (seed 0) on every resolving unified "
        "backend and auto (recorded) vs the legacy reference (tensor-core decode "
        "wrapper for q1, fa2 paged prefill wrapper for the spec-dec rows, "
        "sink_attention_unified for sinks) at 1e-2 / 1e-1 (fp8 KV) and the oracle on "
        "the dequantized pool; fp8 KV via k_scale / v_scale (fa2 only); fp8 output "
        "asserts EXPECT_OUTPUT_DTYPE and runs with the q dtype; full + window 127 has "
        "no unified backend (M20) and skips; sinks + fp8 KV assert the fa2 plan-time "
        "rejection (EXPECT_FA2_SINKS_FP8_KV); sinks + D512 on fa2 are WRONG on B200 "
        "(non-strict xfail, EXPECT_FA2_SINKS_HEAD_DIM_512); the XQA kernel itself "
        "is not a unified candidate",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_spec_dec_sliding_window",
        ["test_xqa_batch_decode_spec_dec_sliding_window"],
        "partial",
        "delegates to test_xqa_batch_decode like the legacy: 4 shapes x window 63/127 "
        "x bf16 / fp8 KV x sink x causal/full x kv 300/30000 (the 30000 rows engage "
        "every backend's long-context path); full + window skips (M20)",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_ragged_q",
        ["test_xqa_batch_decode_ragged_q"],
        "partial",
        "same grid (4 draft-length patterns incl. a zero-length draft x window x bf16 "
        "/ fp8 KV x sink x causal/full x kv 300/30000, NHD): ragged qo_indptr (a "
        "q_len 0 row is legal in the unified contract, M17) on every resolving "
        "backend vs the legacy fa2 prefill-wrapper / sink reference and the oracle; "
        "full + window 127 skips (M20)",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_nvfp4_kv",
        ["test_xqa_batch_decode_nvfp4_kv"],
        "unsupported-by-design",
        "packed nvfp4 KV + block scale factors (stacked / separate sf layouts) have "
        "no unified spelling (EXPECT_NVFP4_KV: run() has no kv_cache_sf, asserted per "
        "id); the legacy rows are SM12x-only and skip on B200",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_mask_mode_deterministic",
        [],
        "out-of-scope",
        "XQA against an analytic uniform-softmax expectation, no prefill-wrapper "
        "reference (brief: out of scope); would convert directly",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


# ---------------------------------------------------------------------------
# the legacy fixture (tests/attention/test_xqa_batch_decode.py helpers, seed 0)
# ---------------------------------------------------------------------------


def _legacy():
    import tests.attention.test_xqa_batch_decode as legacy

    return legacy


def _legacy_problem(
    q_lens,
    seq_lens,
    *,
    page_size,
    num_kv_heads,
    head_grp_size,
    head_dim,
    q_dtype,
    kv_dtype,
    o_dtype,
    kv_layout,
    enable_sink,
):
    """The legacy tensors in the legacy RNG order: query, pool, table, output
    coin (o_scale draw for fp8 output), sinks."""
    lg = _legacy()
    batch_size = int(q_lens.shape[0])
    num_qo_heads = num_kv_heads * head_grp_size
    q, q_scale, ref_q = lg.create_query_tensor(q_lens, num_qo_heads, head_dim, q_dtype)
    q_indptr = lg.generate_cumsum_lens(q_lens)
    kv_cache, k_scale, v_scale, _, _, ref_kv_cache = lg.create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        kv_dtype,
        "bf16" if q_dtype == "fp8" else q_dtype,
        kv_layout,
    )
    page_table, all_page_ids, page_per_seq = lg.create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = lg.generate_cumsum_lens(page_per_seq)
    kv_last_page_len = lg.get_last_page_len(seq_lens, page_size)
    _out, o_scale = lg.create_output(q, o_dtype)  # keeps the legacy RNG stream
    sink = (
        torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
        if enable_sink
        else None
    )
    return dict(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q=q,
        ref_q=ref_q,
        q_dtype=q.dtype,
        kv_dtype=kv_cache.dtype,
        q_lens=q_lens,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        kv_cache=kv_cache,
        ref_kv_cache=ref_kv_cache,
        k_scale=float(k_scale) if isinstance(k_scale, torch.Tensor) else k_scale,
        v_scale=float(v_scale) if isinstance(v_scale, torch.Tensor) else v_scale,
        page_table=page_table,
        all_page_ids=all_page_ids,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        page_size=page_size,
        kv_layout=kv_layout,
        o_scale=o_scale,
        sink=sink,
        sm_scale=float(1.0 / (head_dim**0.5)),
    )


def _legacy_reference(p, *, causal, window_left, q_len_per_req):
    """The legacy reference on the dequantized pool: the tensor-core decode
    wrapper (q_len 1), the fa2 paged prefill wrapper (spec-dec rows) or
    sink_attention_unified (sink rows)."""
    lg = _legacy()
    if p["sink"] is not None:
        k_flat, v_flat, kv_indptr_tokens = lg.flatten_paged_kv(
            p["ref_kv_cache"],
            p["page_table"],
            p["seq_lens"].to(DEVICE),
            p["page_size"],
            p["kv_last_page_len"],
            p["kv_layout"],
        )
        return lg.sink_attention_unified(
            p["ref_q"],
            k_flat,
            v_flat,
            p["sink"],
            window_left,
            causal if q_len_per_req > 1 else True,
            p["sm_scale"],
            mode="varlen",
            batch_size=p["batch_size"],
            qo_indptr=p["q_indptr"],
            kv_indptr=kv_indptr_tokens,
        )
    plan_params = {
        "indptr": p["kv_indptr"],
        "indices": p["all_page_ids"],
        "last_page_len": p["kv_last_page_len"].to(DEVICE),
        "num_qo_heads": p["num_qo_heads"],
        "num_kv_heads": p["num_kv_heads"],
        "head_dim": p["head_dim"],
        "page_size": p["page_size"],
        "pos_encoding_mode": "NONE",
        "kv_data_type": p["ref_kv_cache"].dtype,
        "q_data_type": p["ref_q"].dtype,
        "window_left": window_left,
    }
    if q_len_per_req == 1:
        wrapper_ref = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            legacy_workspace(), p["kv_layout"], use_tensor_cores=True
        )
        wrapper_ref.plan(**plan_params)
        return wrapper_ref.run(p["ref_q"], p["ref_kv_cache"])
    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        legacy_workspace(), p["kv_layout"], backend="fa2"
    )
    plan_params.update(
        {
            "qo_indptr": p["q_indptr"],
            "paged_kv_indptr": plan_params.pop("indptr"),
            "paged_kv_indices": plan_params.pop("indices"),
            "paged_kv_last_page_len": plan_params.pop("last_page_len"),
            "head_dim_qk": plan_params.pop("head_dim"),
            "causal": causal,
            "logits_soft_cap": 0.0,
        }
    )
    wrapper_ref.plan(**plan_params)
    return wrapper_ref.run(p["ref_q"], p["ref_kv_cache"])


def _run_legacy_problem(
    p, *, causal, window_left, q_len_per_req, o_dtype, record_property
):
    if o_dtype == "fp8":
        # the legacy writes an fp8 output scaled by o_scale; unified has no
        # output dtype independent of q: assert, then run with the q dtype
        assert output_dtype_knob_present() == EXPECT_OUTPUT_DTYPE
        if EXPECT_OUTPUT_DTYPE:
            pytest.fail(
                "EXPECT_OUTPUT_DTYPE flipped: port the fp8 output / o_scale here"
            )
    fp8_kv = p["kv_dtype"] == torch.float8_e4m3fn
    md = dense_metadata(p["q_indptr"], p["seq_lens"], p["page_table"], p["page_size"])
    if fp8_kv and p["sink"] is not None and _fa2_resolves(p, causal, window_left):
        # the fa2 AttentionSink variant declines an fp8 KV at plan time
        # (EXPECT_FA2_SINKS_FP8_KV); no other backend takes an fp8 KV.  When
        # fa2 is already excluded at resolve (non-causal + window, M20) the
        # runner below records every reason and skips.
        attn = PagedAttention(torch.device(DEVICE))
        planned = gated(
            EXPECT_FA2_SINKS_FP8_KV,
            lambda: attn.plan(
                md,
                num_qo_heads=p["num_qo_heads"],
                num_kv_heads=p["num_kv_heads"],
                head_dim_qk=p["head_dim"],
                q_dtype=p["q_dtype"],
                kv_dtype=p["kv_dtype"],
                kv_layout=p["kv_layout"],
                causal=causal,
                window_left=window_left,
                lse_mode="base2",
                use_sinks=True,
                backend="fa2",
            ),
            match="attention sinks with an fp8 KV cache",
        )
        if planned is None and not EXPECT_FA2_SINKS_FP8_KV:
            return None
    results = run_on_backends(
        md,
        p["q"],
        (p["kv_cache"][:, 0], p["kv_cache"][:, 1]),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["q_dtype"],
        kv_dtype=p["kv_dtype"],
        kv_layout=p["kv_layout"],
        causal=causal,
        window_left=window_left,
        sinks=p["sink"],
        sm_scale=p["sm_scale"],
        k_scale=p["k_scale"] if fp8_kv else None,
        v_scale=p["v_scale"] if fp8_kv else None,
        record_property=record_property,
    )
    ref = _legacy_reference(
        p, causal=causal, window_left=window_left, q_len_per_req=q_len_per_req
    )
    o_out, o_lse = oracle(
        md,
        p["ref_q"],
        p["ref_kv_cache"][:, 0],
        p["ref_kv_cache"][:, 1],
        causal=causal,
        kv_layout=p["kv_layout"],
        sm_scale=p["sm_scale"],
        window_left=window_left,
        sinks=p["sink"],
    )
    tol = 1e-1 if fp8_kv else 1e-2
    for name, served, out, lse in results:
        if p["sink"] is not None and p["head_dim"] == 512 and served == "fa2":
            # measured defect: the fa2 sink variant at D512 is wrong (helpers)
            ok = torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol)
            record_property(f"fa2_sinks_d512_correct_{name}", ok)
            xfail_unless(
                EXPECT_FA2_SINKS_HEAD_DIM_512,
                ok,
                "fa2 attention sinks at head_dim 512 mismatch the sink reference "
                "(max abs err %.3g); the capability table admits the pair"
                % float((out.float() - ref.float()).abs().max()),
            )
        torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)
        torch.testing.assert_close(out.float(), o_out, **OUT_TOL)
        torch.testing.assert_close(lse, o_lse, **LSE_TOL)
    return results


def _fa2_resolves(p, causal, window_left) -> bool:
    try:
        resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim"],
            q_dtype=p["q_dtype"],
            kv_dtype=p["kv_dtype"],
            page_size=p["page_size"],
            kv_layout=p["kv_layout"],
            causal=causal,
            need_lse=True,
            window_left=window_left,
            sinks=p["sink"] is not None,
            backend="fa2",
        )
    except ValueError:
        return False
    return True


def _sink_causal(spec_dec_mask_mode, q_len_per_req):
    # the legacy: a "full" draft block attends to the whole sequence
    # (non-causal prefill over the paged KV); q_len 1 rows are causal
    return spec_dec_mask_mode == "causal" or q_len_per_req == 1


# ---------------------------------------------------------------------------
# test_xqa_batch_decode
# ---------------------------------------------------------------------------

_XQA_SHAPES = [
    (4, 4, 64, 4, 2, 128),
    (4, 2, 16, 2, 4, 128),
    (4, 3, 32, 2, 6, 128),
    (4, 1, 16, 2, 1, 128),
    (4, 1, 32, 2, 5, 128),
    pytest.param(128, 1, 64, 2, 6, 128, marks=pytest.mark.slow),
    pytest.param(256, 1, 64, 4, 8, 128, marks=pytest.mark.slow),
    # 32 q heads / 2 kv heads (group ratio 16)
    (4, 1, 32, 2, 16, 128),
    (4, 4, 32, 2, 16, 128),
    # head_dim 512 (Gemma-style GQA), decode only (no spec dec)
    (4, 1, 32, 2, 4, 512),
    (4, 1, 32, 2, 5, 512),
    pytest.param(16, 1, 64, 2, 8, 512, marks=pytest.mark.slow),
    (4, 1, 16, 2, 16, 512),
]


@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size,head_dim",
    _XQA_SHAPES,
)
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("fp16", "fp16", "fp16"),
        ("bf16", "fp8", "bf16"),
        ("fp16", "fp8", "fp16"),
        ("bf16", "fp8", "fp8"),
        ("fp16", "fp8", "fp8"),
    ],
)
@pytest.mark.parametrize(
    "enable_pdl",
    [
        pytest.param(True, marks=pytest.mark.slow),
        pytest.param(False, marks=pytest.mark.slow),
        None,
    ],
)
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("max_in_kv_len", [110])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("spec_dec_mask_mode", ["causal", "full"])
def test_xqa_batch_decode(
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    head_dim,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    enable_sink,
    max_in_kv_len,
    kv_layout,
    spec_dec_mask_mode,
    record_property,
):
    """The legacy speculative-decode / decode workload over the paged KV
    through the unified prefill path."""
    if q_len_per_req == 1 and spec_dec_mask_mode == "full":
        pytest.skip("Mask is unused for q_len_per_req == 1")
    lg = _legacy()
    torch.manual_seed(0)
    q_lens, _in_kv_lens, seq_lens = lg.generate_seq_lens_decode(
        batch_size, q_len_per_req, max_in_kv_len
    )
    p = _legacy_problem(
        q_lens,
        seq_lens,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_grp_size=head_grp_size,
        head_dim=head_dim,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
        kv_layout=kv_layout,
        enable_sink=enable_sink,
    )
    _run_legacy_problem(
        p,
        causal=_sink_causal(spec_dec_mask_mode, q_len_per_req),
        window_left=window_left,
        q_len_per_req=q_len_per_req,
        o_dtype=o_dtype,
        record_property=record_property,
    )


@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (4, 2, 32, 2, 4),
        (4, 4, 64, 4, 2),
        (4, 5, 16, 2, 8),
        # 32 q heads / 2 kv heads (group ratio 16)
        (4, 4, 32, 2, 16),
    ],
)
@pytest.mark.parametrize("window_left", [63, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("bf16", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("spec_dec_mask_mode", ["causal", "full"])
@pytest.mark.parametrize(
    "max_in_kv_len",
    [
        300,
        # long context engages the multi-block (split-KV) path
        30000,
    ],
)
def test_xqa_batch_decode_spec_dec_sliding_window(
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    kv_dtype,
    o_dtype,
    enable_sink,
    spec_dec_mask_mode,
    max_in_kv_len,
    record_property,
):
    """Speculative decode with a window that truncates (kv_len >> window)."""
    test_xqa_batch_decode(
        batch_size=batch_size,
        q_len_per_req=q_len_per_req,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_grp_size=head_grp_size,
        head_dim=128,
        window_left=window_left,
        q_dtype=q_dtype,
        o_dtype=o_dtype,
        kv_dtype=kv_dtype,
        enable_pdl=None,
        enable_sink=enable_sink,
        max_in_kv_len=max_in_kv_len,
        kv_layout="NHD",
        spec_dec_mask_mode=spec_dec_mask_mode,
        record_property=record_property,
    )


# ---------------------------------------------------------------------------
# test_xqa_batch_decode_ragged_q
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "q_lens_pattern,page_size,num_kv_heads,head_grp_size",
    [
        # max_q_len * head_grp_size <= 32: single token block per group
        ((1, 3, 2, 4), 32, 2, 4),
        # max_q_len * head_grp_size > 32: multiple token blocks per group,
        # short requests leave whole blocks with zero valid rows
        ((5, 1, 3, 2), 16, 2, 8),
        # 32 q heads / 2 kv heads (group ratio 16)
        ((1, 4, 2, 3), 32, 2, 16),
        # zero-length draft: a request whose drafts were all rejected owns no
        # query rows and must be skipped without touching its neighbors' masks
        ((5, 0, 3, 2), 16, 2, 8),
    ],
)
@pytest.mark.parametrize("window_left", [-1, 127])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("bf16", "bf16", "bf16"),
        ("bf16", "fp8", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_sink", [True, False])
@pytest.mark.parametrize("spec_dec_mask_mode", ["causal", "full"])
@pytest.mark.parametrize(
    "max_in_kv_len",
    [
        300,
        # long context engages the multi-block (split-KV) path
        30000,
    ],
)
def test_xqa_batch_decode_ragged_q(
    q_lens_pattern,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    kv_dtype,
    o_dtype,
    enable_sink,
    spec_dec_mask_mode,
    max_in_kv_len,
    record_property,
):
    """Ragged speculative decode: per-request draft lengths (a zero-length
    draft is a legal q_len 0 row in the unified contract)."""
    torch.manual_seed(0)
    batch_size = len(q_lens_pattern)
    q_lens = torch.tensor(q_lens_pattern, dtype=torch.int32)
    in_kv_lens = torch.randint(0, max_in_kv_len + 1, (batch_size,), dtype=torch.int)
    in_kv_lens[-1] = max_in_kv_len
    seq_lens = q_lens + in_kv_lens
    p = _legacy_problem(
        q_lens,
        seq_lens,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_grp_size=head_grp_size,
        head_dim=128,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        o_dtype=o_dtype,
        kv_layout="NHD",
        enable_sink=enable_sink,
    )
    assert p["q_indptr"].diff().tolist() == list(q_lens_pattern)
    _run_legacy_problem(
        p,
        causal=spec_dec_mask_mode == "causal",
        window_left=window_left,
        q_len_per_req=max(q_lens_pattern),
        o_dtype=o_dtype,
        record_property=record_property,
    )


# ---------------------------------------------------------------------------
# test_xqa_batch_decode_nvfp4_kv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "batch_size,q_len_per_req,page_size,num_kv_heads,head_grp_size",
    [
        (1, 1, 16, 2, 4),
        (1, 1, 32, 2, 4),
        (4, 4, 64, 4, 2),
        (1, 1, 64, 2, 4),
        (1, 1, 64, 2, 8),
        (1, 1, 128, 2, 4),
    ],
)
@pytest.mark.parametrize("window_left", [-1])
@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    [
        ("fp16", "nvfp4", "fp16"),
        ("bf16", "nvfp4", "bf16"),
    ],
)
@pytest.mark.parametrize("enable_pdl", [False])
@pytest.mark.parametrize("enable_sink", [False])
@pytest.mark.parametrize("max_in_kv_len", [300])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("sf_layout", ["stacked", "separate"])
@pytest.mark.parametrize("spec_dec_mask_mode", ["causal", "full"])
def test_xqa_batch_decode_nvfp4_kv(
    batch_size,
    q_len_per_req,
    page_size,
    num_kv_heads,
    head_grp_size,
    window_left,
    q_dtype,
    o_dtype,
    kv_dtype,
    enable_pdl,
    enable_sink,
    max_in_kv_len,
    kv_layout,
    sf_layout,
    spec_dec_mask_mode,
):
    """Packed nvfp4 KV with block scale factors: no unified spelling
    (EXPECT_NVFP4_KV)."""
    import inspect

    from flashinfer.prefill import PagedAttention

    if q_len_per_req == 1 and spec_dec_mask_mode == "full":
        pytest.skip("Mask is unused for q_len_per_req == 1")
    run_params = inspect.signature(PagedAttention.run).parameters
    assert ("kv_cache_sf" in run_params) == EXPECT_NVFP4_KV
    if EXPECT_NVFP4_KV:
        pytest.fail(
            f"EXPECT_NVFP4_KV flipped: port the {sf_layout} scale-factor fixture "
            "(page stride of the SF cache) here"
        )
