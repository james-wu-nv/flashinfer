"""Shared helpers for the legacy -> unified conversion tests.

The per-file conversion modules (``test_legacy_unified_<name>.py``, one per
legacy ``tests/attention/test_<name>.py``) re-run each legacy paged-prefill
test's OWN fixture through the unified ``PagedAttention`` API and assert the
legacy tolerance against the legacy reference and the independent fp32
oracle.  This module holds what every file needs and nothing
backend-specific:

- the ``EXPECT_*`` flags: one per legacy feature the unified API cannot
  express today.  A parity test for such a feature asserts the clear
  rejection while the flag is ``False`` and runs the positive path once the
  sibling capability work flips it to ``True`` -- the manifest row moves
  from ``unsupported-by-design`` to ``equivalent`` without a rewrite.
- ``gated()``: the flip mechanism.
- the lossless legacy CSR -> ``PagedAttentionMetadata.csr`` mapping
  (03 §2.1 of the test review) and the dense form.
- ``slow_case()``: opt-in for the legacy grids' big shapes
  (``FI_PARITY_SLOW=1``); the default subset is documented per file.
- ``reference_long()``: the oracle in query chunks for long sequences.
"""

import os
from typing import Callable, Optional, Sequence, TypeVar

import pytest
import torch

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill

DEVICE = "cuda:0"
OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

# ---------------------------------------------------------------------------
# Legacy features the unified API does not express (2026-09-17, integration
# head 0f0123c5).  Flip to True when the extension lands; the tests below
# each flag then exercise the positive path.  Sibling work package T extends
# e5m2 KV / head_dim 512 on fa2 / pages 128-1024 on trtllm-gen and cake /
# float-KV scales; the rest are proposals in reports/.../wp-s-b.md.
# ---------------------------------------------------------------------------
EXPECT_FA2_E5M2_KV = True  # kv_dtype=float8_e5m2 on fa2 (WP-T, 2e3f80a6)
EXPECT_FA2_HEAD_DIM_512 = True  # (512, 512) on fa2 (WP-T, a466bd67; Gemma-4 shape)
EXPECT_TRTLLM_LARGE_PAGES = (
    True  # page_size 128/256/512/1024 on trtllm-gen, cake (WP-T, a8b1114d)
)
EXPECT_TRTLLM_HEAD_DIM_64 = False  # (64, 64) on trtllm-gen / cake
EXPECT_TRTLLM_HEAD_DIM_256 = False  # (256, 256) on trtllm-gen / cake
EXPECT_TRTLLM_HEAD_DIM_512 = False  # (512, 512) on trtllm-gen / cake
EXPECT_FP8_Q = False  # fp8 (e4m3) query with q_scale (trtllm-gen, cake, cuDNN)
EXPECT_OUTPUT_DTYPE = False  # o_dtype independent of q dtype
EXPECT_NVFP4_KV = False  # packed uint8 KV + block scale factors
EXPECT_DEVICE_SCALES = False  # k_scale / v_scale as device tensors
EXPECT_SKIP_SOFTMAX = False  # trtllm-gen skip_softmax_threshold_scale_factor
EXPECT_INDEPENDENT_KV_TABLES = False  # separate K and V page-id mappings
EXPECT_MULTI_ITEM_SCORING = False  # prefix_len_ptr / token_pos_in_items_ptr
EXPECT_FMHA_V2_CANDIDATE = False  # FMHA v2 as a unified backend
EXPECT_CHUNKED_ATTENTION_KNOB = False  # chunked_attention_size as a plan axis
# Library defects the parity tests reproduce (not API gaps); flip when fixed.
# CR02 (review 2026-09-17, WP-P): BatchAttentionWithAttentionSinkWrapper builds
# its JIT URI from (q dtype, window, backend) only, so the first sink module a
# process builds is reused for every head_dim: D64 after D128 returns NaN,
# D128 after D64 is wrong.  Alone, each head_dim passes.
EXPECT_SINK_JIT_URI_HAS_HEAD_DIM = True  # fixed on the integration head (7c9b2ac8)
# Round 4, WP-C (2026-09-18): axes of the SM100 / SM120 backend-specific legacy
# files (modular cute-dsl, SM120 FMHA / prims, TensorSpeed, XQA) the unified
# API does not express.
EXPECT_HEAD_DIM_32 = False  # (32, 32): no backend declares it (SM120 fp8 fixtures)
EXPECT_MIXED_KV_DTYPE = False  # K and V caches of different dtypes (bf16 K, fp8 V)
EXPECT_ATTENTION_VARIANTS = (
    False  # score-mod / logits-transform variants (sigmoid, ALiBi)
)
# Library behaviour the WP-C conversions reproduce (measured on B200,
# 2026-09-18): fa2 masks the in-page tail past kv_len of a request's last page
# before the PV product, so a non-finite tail is harmless; trtllm-gen, cake and
# cuDNN over-read the tail and 0 x NaN reaches the output (the TensorSpeed and
# modular legacy suites poison exactly that tail).  The input contract now
# states the tail must be finite (PagedAttentionMetadata docstring, ledger
# M23), so the flag stays False by design: the poisoned-tail legacy cases are
# outside the contract and are recorded, not required.
EXPECT_INPAGE_TAIL_IGNORED = False
# fa2 attention sinks with an fp8 KV cache: the AttentionSink JIT variant used
# to decline the pair at plan time ("not verified"); measured on B200 (ledger
# M22, e4m3 / e5m2 at D128 / D256) it matches the sink-aware oracle, the
# decline is gone and the XQA fp8 rows run.
EXPECT_FA2_SINKS_FP8_KV = True
# fa2 attention sinks at head_dim 512 compute WRONG values on B200 (2026-09-18,
# XQA legacy fixture: q_len 1, kv <= 111, H8:2 / 10:2 / 32:2, bf16 and fp16,
# NHD and HND, with and without window 127; 55% of the elements off by up to
# 0.38 against the sink reference and the oracle; D128 / D256 sinks and D512
# without sinks are exact) while the capability table admits the combination.
# Recorded as a non-strict xfail; flip when the sink variant handles D512 (or
# the table excludes the pair).
EXPECT_FA2_SINKS_HEAD_DIM_512 = False

_T = TypeVar("_T")


def gated(
    flag: bool,
    call: Callable[[], _T],
    *,
    match: str,
    exc=ValueError,
) -> Optional[_T]:
    """The manifest flip.

    ``flag`` False (today): ``call()`` must raise ``exc`` whose message matches
    ``match`` -- the clear rejection the ``unsupported-by-design`` row
    documents -- and ``None`` is returned so the caller stops there.
    ``flag`` True (extension landed): ``call()`` must succeed and its result
    is returned for the positive parity path.
    """
    if flag:
        return call()
    with pytest.raises(exc, match=match):
        call()
    return None


def slow_case(what: str) -> None:
    """Opt-in gate for the legacy grids' big shapes (``FI_PARITY_SLOW=1``).

    The default run keeps the documented subset; the full legacy grid is one
    environment variable away and is what the report's full-grid rows ran.
    """
    if os.environ.get("FI_PARITY_SLOW", "0") != "1":
        pytest.skip(f"slow legacy shape ({what}): set FI_PARITY_SLOW=1 to run")


def resolve_or_skip(backend: str, **kw):
    """``resolve_paged_attention`` for a pinned backend, or skip with the
    resolve-time exclusion reason (capability / probe), never silently."""
    try:
        return resolve_paged_attention(
            device=torch.device(kw.pop("device", DEVICE)), backend=backend, **kw
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")


def csr_metadata_from_legacy(
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    page_size: int,
    *,
    device=DEVICE,
) -> PagedAttentionMetadata:
    """Lossless legacy CSR -> unified CSR (03 §2.1).

    ``kv_seq_lens[i] = (pages_i - 1) * page_size + last_page_len[i]`` for a
    request with pages; a request without pages has KV length 0.  The page-id
    stream is handed over unchanged; the host mirrors are the legacy host
    tensors, so construction is zero-sync.
    """
    qo_cpu = qo_indptr.detach().to("cpu", torch.int32)
    kv_indptr_cpu = kv_indptr.detach().to("cpu", torch.int64)
    last_cpu = kv_last_page_len.detach().to("cpu", torch.int64)
    pages = kv_indptr_cpu[1:] - kv_indptr_cpu[:-1]
    kv_lens_cpu = torch.where(
        pages > 0, (pages - 1) * page_size + last_cpu, torch.zeros_like(pages)
    ).to(torch.int32)
    q_lens = qo_cpu[1:] - qo_cpu[:-1]
    dev = torch.device(device)
    return PagedAttentionMetadata.csr(
        qo_cpu.to(dev),
        kv_lens_cpu.to(dev),
        kv_indices.to(dev, torch.int32).contiguous(),
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens_cpu.max()),
        qo_indptr_cpu=qo_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def csr_metadata_page1(
    qo_indptr: torch.Tensor, kv_indptr_tokens: torch.Tensor, kv_indices: torch.Tensor
) -> PagedAttentionMetadata:
    """The attention-sink legacy tests' paged form: page_size 1, one page per
    token, so the token-unit ``kv_indptr`` IS the page indptr and every
    last-page length is 1."""
    b = qo_indptr.shape[0] - 1
    last = torch.ones(b, dtype=torch.int32)
    return csr_metadata_from_legacy(qo_indptr, kv_indptr_tokens, kv_indices, last, 1)


def dense_metadata(
    qo_indptr_cpu: torch.Tensor,
    kv_lens_cpu: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
    *,
    device=DEVICE,
) -> PagedAttentionMetadata:
    """vLLM-style dense form from host lengths and a device block table
    (the table may be wider than the batch needs: capacity columns)."""
    qo_cpu = qo_indptr_cpu.detach().to("cpu", torch.int32)
    kv_cpu = kv_lens_cpu.detach().to("cpu", torch.int32)
    dev = torch.device(device)
    return PagedAttentionMetadata.dense(
        qo_cpu.to(dev),
        kv_cpu.to(dev),
        block_tables,
        page_size=page_size,
        max_q_len=int((qo_cpu[1:] - qo_cpu[:-1]).max()),
        max_kv_len=int(kv_cpu.max()),
        qo_indptr_cpu=qo_cpu,
        kv_seq_lens_cpu=kv_cpu,
    )


def oracle(
    md: PagedAttentionMetadata,
    q: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    *,
    causal: bool,
    kv_layout: str,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    lse_base: str = "2",
    logits_soft_cap: Optional[float] = None,
    custom_mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
):
    """The fp32 oracle on a unified metadata object (either paging form)."""
    return reference_paged_prefill(
        q,
        k_ref,
        v_ref,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        md.block_tables,
        md.page_size,
        causal,
        sm_scale=sm_scale,
        window_left=window_left,
        kv_layout=kv_layout,
        kv_page_indices=md.kv_page_indices,
        lse_base=lse_base,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        sinks=sinks,
    )


def reference_long(
    md: PagedAttentionMetadata,
    q: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    *,
    causal: bool,
    kv_layout: str,
    q_chunk: int = 512,
    sm_scale: Optional[float] = None,
    logits_soft_cap: Optional[float] = None,
):
    """The oracle in query chunks, for sequences whose full score matrix
    would not fit (8192 x 8192 x 64 heads = 17 GiB in fp32).

    A chunk of rows ``[a, b)`` of request ``i`` (query length ``lq``, KV
    length ``L``) sees, under bottom-right causal masking, the keys ``j <=
    L - lq + a + r``; the same rows computed as a request of length ``b - a``
    over the first ``L - lq + b`` keys have exactly that envelope.  Without
    causal masking every chunk sees all ``L`` keys.  No sliding window here;
    the soft cap is per score and passes through.
    """
    qo = md.qo_indptr_cpu
    kv = md.kv_seq_lens_cpu
    outs, lses = [], []
    dev = q.device
    n_pages_prefix = 0
    for i in range(kv.shape[0]):
        s, e = int(qo[i]), int(qo[i + 1])
        lq, L = e - s, int(kv[i])
        n_pages = (L + md.page_size - 1) // md.page_size
        if md.block_tables is not None:
            table = md.block_tables[i : i + 1]
            idx = None
        else:
            table = None
            idx = md.kv_page_indices[n_pages_prefix : n_pages_prefix + n_pages]
        n_pages_prefix += n_pages
        for a in range(0, lq, q_chunk):
            b = min(lq, a + q_chunk)
            kv_len = L - lq + b if causal else L
            o, l = reference_paged_prefill(
                q[s + a : s + b],
                k_ref,
                v_ref,
                torch.tensor([0, b - a], dtype=torch.int32),
                torch.tensor([kv_len], dtype=torch.int32),
                table,
                md.page_size,
                causal,
                sm_scale=sm_scale,
                kv_layout=kv_layout,
                kv_page_indices=idx,
                logits_soft_cap=logits_soft_cap,
            )
            outs.append(o)
            lses.append(l)
    return torch.cat(outs).to(dev), torch.cat(lses).to(dev)


# ---------------------------------------------------------------------------
# The legacy trtllm-gen paged fixture (tests/attention/test_trtllm_gen_attention_
# decode.py helpers under the legacy seed), shared by the trtllm-gen, cake and
# XQA conversion modules.  Round 3 kept it in the trtllm theme file; round 4
# splits that file per legacy source, so the fixture lives here.
# ---------------------------------------------------------------------------

_legacy_ws = None


def legacy_workspace() -> torch.Tensor:
    """The legacy 256 MiB int8 workspace, allocated once per process."""
    global _legacy_ws
    if _legacy_ws is None:
        _legacy_ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
    return _legacy_ws


def legacy_trtllm_problem(
    kv_layout: str,
    batch_size: int,
    page_size: int,
    num_kv_heads: int,
    head_grp_size: int,
    dtype_name: str,
    max_q_len: int,
    max_kv_len: int,
    head_dim: int,
    *,
    seed: int = 0,
) -> dict:
    """``_test_trtllm_batch_prefill``'s tensors in the legacy call order under
    ``torch.manual_seed(seed)``: query, stacked ``(pages, 2, ...)`` pool, the
    dense page table, the CSR mirrors and the sink draw."""
    from tests.attention.test_trtllm_gen_attention_decode import (
        create_kv_cache,
        create_page_table,
        create_query_tensor,
        generate_cumsum_lens,
        generate_seq_lens_prefill,
        get_last_page_len,
    )

    torch.manual_seed(seed)
    num_qo_heads = num_kv_heads * head_grp_size
    q_lens, _, seq_lens = generate_seq_lens_prefill(batch_size, max_q_len, max_kv_len)
    q, _q_scale, ref_q = create_query_tensor(q_lens, num_qo_heads, head_dim, dtype_name)
    q_indptr = generate_cumsum_lens(q_lens)
    kv_cache, _k_scale, _v_scale, ref_kv_cache, _ = create_kv_cache(
        batch_size,
        seq_lens,
        page_size,
        num_kv_heads,
        head_dim,
        dtype_name,
        dtype_name,
        kv_layout,
    )
    page_table, all_page_ids, page_per_seq = create_page_table(
        batch_size, seq_lens, page_size
    )
    kv_indptr = generate_cumsum_lens(page_per_seq)
    kv_last_page_len = get_last_page_len(seq_lens, page_size)
    sink = torch.rand(num_qo_heads, device=DEVICE, dtype=torch.float32) * 5
    return dict(
        kv_layout=kv_layout,
        batch_size=batch_size,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16 if dtype_name == "bf16" else torch.float16,
        q=q,
        ref_q=ref_q,
        q_lens=q_lens,
        seq_lens=seq_lens,
        q_indptr=q_indptr,
        kv_cache=kv_cache,  # (pages, 2, ...) stacked pool
        ref_kv_cache=ref_kv_cache,
        page_table=page_table,
        all_page_ids=all_page_ids,
        kv_indptr=kv_indptr,
        kv_last_page_len=kv_last_page_len,
        sink=sink,
        sm_scale=float(1.0 / (head_dim**0.5)),
    )


def legacy_fa2_paged_reference(p: dict, *, causal: bool, window_left: int = -1):
    """The legacy reference of the no-sink rows: the fa2 paged wrapper on the
    reference pool (the legacy leaves backend='auto', which is fa2 on B200;
    pinned here so the reference is the same kernel on every machine)."""
    import flashinfer

    wrapper_ref = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        legacy_workspace(), p["kv_layout"], backend="fa2"
    )
    wrapper_ref.plan(
        qo_indptr=p["q_indptr"],
        paged_kv_indptr=p["kv_indptr"],
        paged_kv_indices=p["all_page_ids"],
        paged_kv_last_page_len=p["kv_last_page_len"].to(DEVICE),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        page_size=p["page_size"],
        causal=causal,
        pos_encoding_mode="NONE",
        logits_soft_cap=0.0,
        q_data_type=p["ref_q"].dtype,
        kv_data_type=p["ref_kv_cache"].dtype,
        window_left=window_left,
    )
    return wrapper_ref.run(p["ref_q"], p["ref_kv_cache"], return_lse=True)


def legacy_sink_reference(p: dict, *, causal: bool, window_left: int = -1):
    """The legacy reference of the sink rows: ``sink_attention_unified`` on
    the flattened reference pool."""
    from tests.attention.test_trtllm_gen_attention_decode import flatten_paged_kv
    from tests.test_helpers.sink_attention_reference import sink_attention_unified

    k_flat, v_flat, kv_indptr_tokens = flatten_paged_kv(
        p["ref_kv_cache"],
        p["page_table"],
        p["seq_lens"].to(DEVICE),
        p["page_size"],
        p["kv_last_page_len"],
        p["kv_layout"],
    )
    return sink_attention_unified(
        p["ref_q"],
        k_flat,
        v_flat,
        p["sink"],
        window_left,
        causal,
        p["sm_scale"],
        mode="varlen",
        batch_size=p["batch_size"],
        qo_indptr=p["q_indptr"],
        kv_indptr=kv_indptr_tokens,
    )


def assert_legacy_close(out, ref, *, rtol=1e-2, atol=1e-2) -> None:
    """The legacy assertion: ``assert_close`` with the 1e-7 mismatch allowance."""
    from tests.test_helpers.test_helpers import assert_close_with_mismatch_tolerance

    assert_close_with_mismatch_tolerance(
        out.float(),
        ref.float(),
        rtol=rtol,
        atol=atol,
        max_mismatched_elements=int(1e-7 * out.numel()),
    )


def trtllm_resolve_kwargs(p: dict, *, causal: bool, backend: str, **overrides):
    kw = dict(
        device=torch.device(DEVICE),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["dtype"],
        page_size=p["page_size"],
        kv_layout=p["kv_layout"],
        causal=causal,
        need_lse=True,
        backend=backend,
    )
    kw.update(overrides)
    return kw


def plan_legacy_problem(
    p: dict,
    backend: str,
    *,
    causal: bool,
    lse_mode: str = "base2",
    use_sinks: bool = False,
    window_left: int = -1,
):
    """Plan the legacy fixture in the dense form on a pinned backend (skipping
    with the resolve reason when it is capability-excluded), or on ``auto``.
    Returns ``(attn, md)``; ``attn.backend`` is asserted for a pinned name."""
    from flashinfer.prefill import PagedAttention

    md = dense_metadata(p["q_indptr"], p["seq_lens"], p["page_table"], p["page_size"])
    common = dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim"],
        q_dtype=p["dtype"],
        kv_layout=p["kv_layout"],
        causal=causal,
        window_left=window_left,
    )
    if backend == "auto":
        res = "auto"
    else:
        res = resolve_or_skip(
            backend,
            page_size=p["page_size"],
            need_lse=lse_mode != "none",
            kv_input_form="block_tables",
            sinks=use_sinks,
            **common,
        )
    attn = PagedAttention(torch.device(DEVICE))
    attn.plan(md, lse_mode=lse_mode, use_sinks=use_sinks, backend=res, **common)
    if backend != "auto":
        assert attn.backend == backend
    return attn, md


def independent_tables_rejected(p: dict):
    """The legacy interleaved layout (K at page 2p, V at 2p+1, a [B, 2, M]
    table) is two page-id mappings; the metadata takes exactly one.  Returns
    None while EXPECT_INDEPENDENT_KV_TABLES is False (the rejection asserted),
    the metadata once the extension lands."""
    from tests.attention.test_trtllm_gen_attention_decode import (
        prepare_paged_kv_for_kernel,
    )

    (k_i, v_i), table_2, _ = prepare_paged_kv_for_kernel(
        p["kv_cache"], p["page_table"], False
    )
    assert table_2.shape == (p["batch_size"], 2, p["page_table"].shape[1])
    assert not torch.equal(table_2[:, 0], table_2[:, 1])  # K ids != V ids
    md = dense_metadata(p["q_indptr"], p["seq_lens"], p["page_table"], p["page_size"])
    return gated(
        EXPECT_INDEPENDENT_KV_TABLES,
        lambda: PagedAttentionMetadata.dense(
            md.qo_indptr,
            md.kv_seq_lens,
            table_2[:, 0].contiguous(),
            v_block_tables=table_2[:, 1].contiguous(),
            page_size=p["page_size"],
            max_q_len=md.max_q_len,
            max_kv_len=md.max_kv_len,
        ),
        match="v_block_tables",
        exc=TypeError,
    )


def skip_softmax_knob_present() -> bool:
    """Whether ``skip_softmax_threshold_scale_factor`` (approximate softmax) has
    a plan- or run-time spelling; asserted equal to EXPECT_SKIP_SOFTMAX."""
    import inspect

    from flashinfer.prefill import PagedAttention

    run_params = inspect.signature(PagedAttention.run).parameters
    plan_params = inspect.signature(PagedAttention.plan).parameters
    return (
        "skip_softmax_threshold_scale_factor" in run_params
        or "skip_softmax_threshold_scale_factor" in plan_params
    )


def output_dtype_knob_present() -> bool:
    """Whether an output dtype independent of q (``o_dtype`` / ``out_dtype``)
    has a spelling; asserted equal to EXPECT_OUTPUT_DTYPE."""
    import inspect

    from flashinfer.prefill import PagedAttention

    run_params = inspect.signature(PagedAttention.run).parameters
    plan_params = inspect.signature(PagedAttention.plan).parameters
    return "o_dtype" in plan_params or "out_dtype" in run_params


def fp8_q_rejected(
    *,
    backend: str,
    num_qo_heads: int = 2,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    page_size: int = 16,
    kv_layout: str = "HND",
    causal: bool = True,
    window_left: int = -1,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """fp8 (e4m3) q is rejected at resolve on every backend while EXPECT_FP8_Q
    is False (``'unsupported q dtype'``); returns the Resolution once it flips."""
    return gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=kv_dtype,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            need_lse=False,
            backend=backend,
        ),
        match="unsupported q dtype",
    )


def xfail_unless(flag: bool, ok: bool, reason: str) -> None:
    """A measured library behaviour the conversion reproduces: while ``flag``
    is False the row is a non-strict xfail when ``ok`` is False (so the
    outcome is recorded, not hidden); once the flag flips, ``ok`` must hold."""
    if ok:
        return
    if flag:
        pytest.fail(reason)
    pytest.xfail(reason)


# ---------------------------------------------------------------------------
# Multi-backend runner for the backend-specific legacy files (WP-C): the
# workload runs on every pinned unified backend that resolves, the excluded
# ones are recorded with their resolve reason (junit properties), and ``auto``
# records which backend served the workload.
# ---------------------------------------------------------------------------

ALL_BACKENDS = ("fa2", "trtllm-gen", "cake", "cudnn")


def run_on_backends(
    md: PagedAttentionMetadata,
    q: torch.Tensor,
    kv_cache,
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    q_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
    kv_layout: str,
    causal: bool,
    window_left: int = -1,
    lse_mode: str = "base2",
    logits_soft_cap: Optional[float] = None,
    custom_mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    backends: Sequence[str] = ALL_BACKENDS,
    include_auto: bool = True,
    record_property=None,
):
    """Plan and run one legacy workload on every pinned backend in ``backends``
    that resolves, then on ``auto``.  Returns ``[(pinned_name, served_by, out,
    lse)]`` (``pinned_name == "auto"`` for the auto row); skips -- never
    silently -- when no backend resolves, with every reason."""
    from flashinfer.prefill import PagedAttention

    dev = torch.device(DEVICE)
    kv_dtype = kv_dtype if kv_dtype is not None else q_dtype
    resolve_kw = dict(
        device=dev,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        page_size=md.page_size,
        kv_layout=kv_layout,
        causal=causal,
        need_lse=lse_mode != "none",
        window_left=window_left,
        kv_input_form=md.kv_input_form,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask is not None,
        sinks=sinks is not None,
    )
    plan_kw = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        kv_layout=kv_layout,
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        use_sinks=sinks is not None,
    )
    run_kw = dict(
        sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale, sinks=sinks, out=out
    )
    results, excluded = [], {}
    names = list(backends) + (["auto"] if include_auto else [])
    for name in names:
        try:
            res = resolve_paged_attention(backend=name, **resolve_kw)
        except ValueError as e:
            excluded[name] = str(e)
            continue
        attn = PagedAttention(dev)
        try:
            attn.plan(md, backend=res, **plan_kw)
        except ValueError as e:
            # a plan-time (batch-specific) decline of the pinned candidate(s):
            # recorded with its reason like a resolve-time exclusion
            if "cannot plan this batch" not in str(e):
                raise
            excluded[name] = "plan: " + str(e)
            continue
        if name != "auto":
            assert attn.backend == name
        o, lse = attn.run(q, kv_cache, **run_kw)
        results.append((name, attn.backend, o, lse))
    if record_property is not None:
        for name, reason in excluded.items():
            record_property(f"excluded_{name}", reason)
        for name, served, _, _ in results:
            if name == "auto":
                record_property("auto_backend", served)
    if not results:
        detail = "; ".join(f"{k}: {v}" for k, v in excluded.items())
        pytest.skip(f"no unified backend resolves this legacy case ({detail})")
    return results


LEGACY_MAP_STATUSES = frozenset(
    {"equivalent", "partial", "unsupported-by-design", "native-only", "out-of-scope"}
)


def check_legacy_map(legacy_map: Sequence, namespace: dict) -> None:
    """Every unified function a LEGACY_MAP row names exists in the module and
    every row carries one of the five statuses (round-4 PLAN §1).  An
    ``out-of-scope`` row (a legacy function that is not paged prefill:
    ragged / single / decode-only) names no unified function; every other
    status names at least one."""
    seen = set()
    for legacy, unified, status, note in legacy_map:
        assert legacy not in seen, f"duplicate legacy entry {legacy}"
        seen.add(legacy)
        assert status in LEGACY_MAP_STATUSES, (legacy, status)
        assert isinstance(note, str) and note, legacy
        if status == "out-of-scope":
            assert not unified, f"{legacy}: out-of-scope rows name no unified test"
        else:
            assert unified, f"{legacy}: {status} rows name a unified test"
        for name in unified:
            assert callable(namespace.get(name)), f"{legacy}: {name} is not a test here"


def legacy_test_functions(legacy_source: str) -> list:
    """The ``test_*`` function names of a legacy test file (source scan, no
    import: the legacy modules pull in backend-specific dependencies).  A
    conversion module's self-check compares this with its LEGACY_MAP so every
    legacy function has exactly one row."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    tree = ast.parse((root / legacy_source).read_text())
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]


def check_legacy_map_complete(legacy_source: str, legacy_map: Sequence) -> None:
    """One LEGACY_MAP row per legacy ``test_*`` function (PLAN §1: out-of-scope
    functions included), keyed by ``<legacy_source>::<function>``."""
    expected = {f"{legacy_source}::{fn}" for fn in legacy_test_functions(legacy_source)}
    got = {row[0] for row in legacy_map}
    assert got == expected, (
        f"LEGACY_MAP rows missing: {sorted(expected - got)}; "
        f"unexpected: {sorted(got - expected)}"
    )


def plan_pinned(
    backend: str,
    md: PagedAttentionMetadata,
    *,
    attn: Optional[PagedAttention] = None,
    **plan_kw,
) -> PagedAttention:
    """Pin ``backend`` for a plan: ``resolve_or_skip`` on the plan's static
    configuration (skip with the resolve reason when the capability table or
    the probe excludes it), plan a ``PagedAttention`` (a fresh instance unless
    ``attn`` is given, for re-plan scenarios) on the resolution and assert the
    pinned backend was published.  ``plan_kw`` are the ``PagedAttention.plan``
    keywords minus ``metadata`` / ``backend``."""
    res = resolve_or_skip(
        backend,
        num_qo_heads=plan_kw["num_qo_heads"],
        num_kv_heads=plan_kw["num_kv_heads"],
        head_dim_qk=plan_kw["head_dim_qk"],
        head_dim_vo=plan_kw.get("head_dim_vo"),
        q_dtype=plan_kw["q_dtype"],
        kv_dtype=plan_kw.get("kv_dtype"),
        page_size=md.page_size,
        kv_layout=plan_kw.get("kv_layout", "HND"),
        causal=plan_kw.get("causal", True),
        need_lse=plan_kw.get("lse_mode", "none") != "none",
        window_left=plan_kw.get("window_left", -1),
        kv_input_form="block_tables" if md.block_tables is not None else "page_indices",
        logits_soft_cap=plan_kw.get("logits_soft_cap"),
        custom_mask=plan_kw.get("custom_mask") is not None,
        sinks=plan_kw.get("use_sinks", False),
        max_q_len=md.max_q_len,
    )
    if attn is None:
        attn = PagedAttention(torch.device(DEVICE))
    attn.plan(md, backend=res, **plan_kw)
    assert attn.backend == backend
    return attn


# ===========================================================================
# Group A machinery (round 4: one unified file per legacy file).  Appended
# after the shared block so the three conversion branches merge cleanly.
# ===========================================================================

import inspect  # noqa: E402
import itertools  # noqa: E402
import zlib  # noqa: E402

import flashinfer  # noqa: E402
from flashinfer.prefill import GraphCapacity, PagedAttention  # noqa: E402


def check_unified_tests_mapped(legacy_map: Sequence, namespace: dict) -> None:
    """Every ``test_*`` of a conversion module (except the self-check) is named
    by a LEGACY_MAP row: no unified test without a legacy source."""
    referenced = {name for row in legacy_map for name in row[1]}
    module_tests = {
        name
        for name, obj in namespace.items()
        if name.startswith("test_") and callable(obj)
    } - {"test_legacy_map_is_well_formed"}
    assert module_tests <= referenced, (
        f"unified tests without a LEGACY_MAP row: {sorted(module_tests - referenced)}"
    )


BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen", "cake", "auto"]
slow = pytest.mark.slow
MB = 1024 * 1024

# Legacy features of group A the unified API does not express (same flip rule
# as the flags above; the positive branch is written next to each rejection).
EXPECT_FLOAT_KV_SCALES = True  # k_scale / v_scale on a fp16/bf16 KV (WP-T, c690336a)
EXPECT_FULLY_MASKED_ROWS = False  # causal rows with q_len > kv_len: out 0 / LSE -inf
EXPECT_HEAD_DIM_448_256 = False  # the (448, 256) head-dim pair (fa2 CTA-tile probe)
EXPECT_BATCH_INVARIANT = False  # determinism policy (legacy fixed_split_size knobs)
EXPECT_ROPE = False  # fused positional encoding (pos_encoding_mode / rope_*)


def seed_of(*parts) -> int:
    """Deterministic per-row seed from the row's parameters (the legacy tests
    are unseeded; the same point always builds the same fixture)."""
    return zlib.crc32(repr(parts).encode())


def plan_signature_params():
    return inspect.signature(PagedAttention.plan).parameters


def run_signature_params():
    return inspect.signature(PagedAttention.run).parameters


# ---------------------------------------------------------------------------
# legacy node ids and the default / slow parametrization
# ---------------------------------------------------------------------------


def _idval(name, values, val) -> str:
    """pytest's id for one parametrize value: numbers, bools, None and
    strings verbatim; anything else (torch.dtype, a list of pairs) is
    ``<argname><index>``, as pytest generates it."""
    if isinstance(val, (bool, int, float)) or val is None:
        return str(val)
    if isinstance(val, str):
        return val
    return f"{name}{list(values).index(val)}"


def legacy_id(axes: dict, point) -> str:
    """The legacy node id of one grid point.  ``axes`` lists the legacy
    ``@pytest.mark.parametrize`` decorators top-down; pytest joins stacked
    decorators innermost first, hence the reversal."""
    parts = [
        _idval(name, axes[name], val) for name, val in zip(axes, point, strict=True)
    ]
    return "-".join(reversed(parts))


def argnames(axes: dict, *extra: str) -> str:
    return ",".join([*axes, *extra])


def grid(axes: dict, **override):
    return list(itertools.product(*dict(axes, **override).values()))


def param_rows(
    axes: dict,
    default_pred,
    *,
    slow_backends=("fa2",),
    backends=BACKENDS,
    backend_first=False,
):
    """Cross the legacy grid with the unified backends.  A default-subset
    point runs on every backend; every other legacy point stays on the
    legacy backend(s) under ``slow`` (``FI_PARITY_SLOW=1``), so the full
    legacy grid stays runnable without multiplying it by six.  The row id is
    the legacy node id plus ``-<backend>``: a legacy case and its unified
    counterpart differ only by that suffix (``backend_first``: the legacy grid
    had the backend as its innermost axis, so the id starts with it and the
    legacy backends' rows carry exactly the legacy ids)."""
    out = []
    for point in grid(axes):
        default = default_pred(point)
        for backend in backends:
            if not default and backend not in slow_backends:
                continue
            base = legacy_id(axes, point)
            out.append(
                pytest.param(
                    *point,
                    backend,
                    marks=() if default else (slow,),
                    id=f"{backend}-{base}" if backend_first else f"{base}-{backend}",
                )
            )
    return out


def backend_rows(*point, ids=None, backends=BACKENDS):
    """Fixed legacy point(s) crossed with the backends (no slow rows)."""
    return [
        pytest.param(*point, backend, id=f"{ids}-{backend}" if ids else backend)
        for backend in backends
    ]


# ---------------------------------------------------------------------------
# the legacy fixture and its lossless mapping
# ---------------------------------------------------------------------------


class LegacyBatch:
    """A legacy paged-prefill batch: the legacy wrapper's CSR metadata
    (``qo_indptr`` / ``paged_kv_indptr`` / ``paged_kv_indices`` /
    ``paged_kv_last_page_len``) plus the K/V pools as the legacy test holds
    them (combined-pool views or separate pools).  ``metadata()`` performs the
    lossless mapping to the unified canonical form (03 §2.1):

        kv_seq_lens[i] = (pages_i - 1) * page_size + last_page_len[i]
        page_size < 8  -> PagedAttentionMetadata.csr(legacy indices)
        page_size >= 8 -> PagedAttentionMetadata.dense(indices per request)
    """

    def __init__(
        self,
        *,
        q,
        k,
        v,
        q_indptr_cpu,
        kv_indptr_cpu,
        kv_indices_cpu,
        last_page_len_cpu,
        page_size,
        kv_layout,
    ):
        self.q, self.k, self.v = q, k, v
        self.q_indptr_cpu = q_indptr_cpu.to("cpu", torch.int32)
        self.kv_indptr_cpu = kv_indptr_cpu.to("cpu", torch.int32)
        self.kv_indices_cpu = kv_indices_cpu.to("cpu", torch.int32)
        self.last_page_len_cpu = last_page_len_cpu.to("cpu", torch.int32)
        self.page_size = page_size
        self.kv_layout = kv_layout
        pages = self.kv_indptr_cpu.diff()
        self.kv_seq_lens_cpu = torch.where(
            pages > 0,
            (pages - 1) * page_size + self.last_page_len_cpu,
            torch.zeros_like(pages),
        ).to(torch.int32)
        self.q_lens_cpu = self.q_indptr_cpu.diff()

    @property
    def batch_size(self):
        return int(self.kv_seq_lens_cpu.shape[0])

    @property
    def num_qo_heads(self):
        return int(self.q.shape[1])

    @property
    def num_kv_heads(self):
        return int(self.k.shape[2] if self.kv_layout == "NHD" else self.k.shape[1])

    @property
    def head_dim_qk(self):
        return int(self.k.shape[3])

    @property
    def head_dim_vo(self):
        return int(self.v.shape[3])

    @property
    def dtype(self):
        return self.q.dtype

    def plan_kwargs(self):
        return dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim_qk,
            head_dim_vo=self.head_dim_vo,
            q_dtype=self.dtype,
            kv_dtype=self.k.dtype,
            kv_layout=self.kv_layout,
        )

    def resolve_kwargs(self):
        return dict(
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim_qk,
            head_dim_vo=self.head_dim_vo,
            q_dtype=self.dtype,
            kv_dtype=self.k.dtype,
            page_size=self.page_size,
            kv_layout=self.kv_layout,
        )

    # ---- per-request views (the legacy reference's gather) ----
    def page_ids(self, i):
        return self.kv_indices_cpu[
            int(self.kv_indptr_cpu[i]) : int(self.kv_indptr_cpu[i + 1])
        ]

    def request_q(self, i):
        return self.q[int(self.q_indptr_cpu[i]) : int(self.q_indptr_cpu[i + 1])]

    def request_kv(self, i, k=None, v=None):
        """(kv_len, Hkv, D) K and V of request ``i`` gathered from the pools --
        the legacy tests' ``cat(full pages, last page[:last_page_len])``."""
        k = self.k if k is None else k
        v = self.v if v is None else v
        kv_len = int(self.kv_seq_lens_cpu[i])
        ids = self.page_ids(i).to(k.device, torch.long)

        def gather(pool):
            pages = pool[ids]
            if self.kv_layout == "HND":
                pages = pages.permute(0, 2, 1, 3)
            return pages.reshape(-1, pages.shape[-2], pages.shape[-1])[:kv_len]

        return gather(k), gather(v)

    def prefix(self, n_requests):
        """The first ``n_requests`` requests as a batch of their own (the
        legacy batch-invariance fixture); the pools are shared."""
        return LegacyBatch(
            q=self.q[: int(self.q_indptr_cpu[n_requests])],
            k=self.k,
            v=self.v,
            q_indptr_cpu=self.q_indptr_cpu[: n_requests + 1],
            kv_indptr_cpu=self.kv_indptr_cpu[: n_requests + 1],
            kv_indices_cpu=self.kv_indices_cpu[: int(self.kv_indptr_cpu[n_requests])],
            last_page_len_cpu=self.last_page_len_cpu[:n_requests],
            page_size=self.page_size,
            kv_layout=self.kv_layout,
        )

    def truncated(self, kv_lens):
        """The same requests over the first ``kv_lens[i]`` tokens of their
        pages (a warm-up batch for the graph lifecycle)."""
        kv_lens = torch.as_tensor(kv_lens, dtype=torch.int32)
        assert bool((kv_lens >= 1).all()) and bool(
            (kv_lens <= self.kv_seq_lens_cpu).all()
        )
        pages = (kv_lens + self.page_size - 1) // self.page_size
        ids = torch.cat(
            [self.page_ids(i)[: int(pages[i])] for i in range(self.batch_size)]
        )
        indptr = torch.cat([torch.zeros(1, dtype=torch.int32), pages.cumsum(0)]).to(
            torch.int32
        )
        return LegacyBatch(
            q=self.q,
            k=self.k,
            v=self.v,
            q_indptr_cpu=self.q_indptr_cpu,
            kv_indptr_cpu=indptr,
            kv_indices_cpu=ids,
            last_page_len_cpu=((kv_lens - 1) % self.page_size + 1).to(torch.int32),
            page_size=self.page_size,
            kv_layout=self.kv_layout,
        )

    # ---- lossless mapping to the unified form ----
    def metadata(self, form=None, device=DEVICE, table_width=None):
        """``table_width`` pads the dense table (CUDA-graph capacities need
        every batch's table at the captured width)."""
        form = form or ("csr" if self.page_size < 8 else "dense")
        dev = torch.device(device)
        common = dict(
            page_size=self.page_size,
            max_q_len=int(self.q_lens_cpu.max()),
            max_kv_len=max(int(self.kv_seq_lens_cpu.max()), 1),
            qo_indptr_cpu=self.q_indptr_cpu,
            kv_seq_lens_cpu=self.kv_seq_lens_cpu,
        )
        if form == "csr":
            return PagedAttentionMetadata.csr(
                self.q_indptr_cpu.to(dev),
                self.kv_seq_lens_cpu.to(dev),
                self.kv_indices_cpu.to(dev),
                **common,
            )
        assert form == "dense", form
        pages = self.kv_indptr_cpu.diff()
        width = int(pages.max()) if table_width is None else table_width
        assert width >= int(pages.max())
        table = torch.zeros(self.batch_size, width, dtype=torch.int32)
        for i in range(self.batch_size):
            table[i, : int(pages[i])] = self.page_ids(i)
        return PagedAttentionMetadata.dense(
            self.q_indptr_cpu.to(dev),
            self.kv_seq_lens_cpu.to(dev),
            table.to(dev),
            **common,
        )


def legacy_uniform_batch(
    *,
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seed,
    kv_layout="NHD",
    dtype=torch.float16,
    combined=True,
    fp32_source=True,
    scale=1.0,
    device=DEVICE,
):
    """The legacy ``test_batch_prefill_kernels`` fixture: uniform lengths,
    ``paged_kv_indices = arange(total_pages)``, last page ``(kv_len - 1) % P +
    1``; a combined ``(pages, 2, ...)`` pool (K/V are its views) or two pools;
    K/V drawn in fp32 and rounded (``fp32_source``) as the legacy main test
    does, or directly in ``dtype``; ``scale`` divides q and the pools (the
    legacy ``/10`` fixtures)."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device=dev, dtype=dtype
    )
    if scale != 1.0:
        q = q / scale
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = pages_per_seq * batch_size
    if kv_layout == "HND":
        pool_shape = (num_kv_heads, page_size, head_dim)
    else:
        pool_shape = (page_size, num_kv_heads, head_dim)
    src_dtype = torch.float32 if fp32_source else dtype

    def draw(shape):
        t = torch.randn(*shape, dtype=src_dtype, device=dev)
        if scale != 1.0:
            t = t / scale
        return t.to(dtype)

    if combined:
        kv = draw((total_pages, 2, *pool_shape))
        k, v = kv[:, 0], kv[:, 1]  # views, no copy
        assert k.data_ptr() == kv.data_ptr() and not k.is_contiguous()
    else:
        k = draw((total_pages, *pool_shape))
        v = draw((total_pages, *pool_shape))
    return LegacyBatch(
        q=q,
        k=k,
        v=v,
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len,
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32)
        * pages_per_seq,
        kv_indices_cpu=torch.arange(0, total_pages, dtype=torch.int32),
        last_page_len_cpu=torch.full(
            (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout=kv_layout,
    )


# ---------------------------------------------------------------------------
# legacy reference methods and assertions
# ---------------------------------------------------------------------------


def legacy_reference_single_prefill(
    lb,
    *,
    causal,
    window_left=-1,
    logits_soft_cap=0.0,
    backend="auto",
    custom_mask=None,
    pos_encoding_mode="NONE",
    k=None,
    v=None,
):
    """The legacy reference method: ``single_prefill_with_kv_cache`` per
    request on the legacy per-request K/V gather, concatenated.
    ``custom_mask`` is a callable ``i -> mask_i``."""
    outs = []
    for i in range(lb.batch_size):
        ki, vi = lb.request_kv(i, k, v)
        kw = {}
        if custom_mask is not None:
            kw["custom_mask"] = custom_mask(i)
        outs.append(
            flashinfer.prefill.single_prefill_with_kv_cache(
                lb.request_q(i),
                ki,
                vi,
                causal=causal,
                pos_encoding_mode=pos_encoding_mode,
                logits_soft_cap=logits_soft_cap,
                window_left=window_left,
                backend=backend,
                **kw,
            )
        )
    return torch.cat(outs)


def legacy_paged_wrapper(
    lb,
    *,
    backend="fa2",
    causal=True,
    logits_soft_cap=0.0,
    window_left=-1,
    workspace_mb=256,
):
    """The legacy ``BatchPrefillWithPagedKVCacheWrapper`` planned for the
    batch (the legacy tests' own reference where they compare wrapper against
    wrapper); the caller runs it."""
    dev = torch.device(DEVICE)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(workspace_mb * MB, dtype=torch.uint8, device=dev),
        kv_layout=lb.kv_layout,
        backend=backend,
    )
    wrapper.plan(
        lb.q_indptr_cpu.to(dev),
        lb.kv_indptr_cpu.to(dev),
        lb.kv_indices_cpu.to(dev),
        lb.last_page_len_cpu.to(dev),
        lb.num_qo_heads,
        lb.num_kv_heads,
        lb.head_dim_qk,
        lb.page_size,
        causal=causal,
        q_data_type=lb.dtype,
        kv_data_type=lb.k.dtype,
        logits_soft_cap=logits_soft_cap,
        window_left=window_left,
    )
    return wrapper


def assert_legacy_isclose(out, ref, *, rtol, atol, what="output"):
    """The legacy assertion of the group-A suites: ``torch.isclose`` mismatch
    count, one sync (``assert_legacy_close`` above is the group-C form with the
    legacy 1e-7 mismatch allowance)."""
    assert out.shape == ref.shape, (
        f"{what}: shape {tuple(out.shape)} vs {tuple(ref.shape)}"
    )
    close = torch.isclose(out.float(), ref.float(), rtol=rtol, atol=atol)
    bad = int((~close).sum())
    if bad:
        diff = (out.float() - ref.float()).abs()
        raise AssertionError(
            f"{what}: {bad}/{out.numel()} elements outside the legacy tolerance "
            f"rtol={rtol} atol={atol}; max abs diff {diff.max().item():.3e}"
        )


# ---------------------------------------------------------------------------
# the fp32 oracle on a legacy batch (query-row chunked for long requests)
# ---------------------------------------------------------------------------


def oracle_on_batch(
    lb,
    *,
    causal,
    window_left=-1,
    sm_scale=None,
    lse_base="2",
    custom_mask=None,
    logits_soft_cap=None,
    q=None,
    k=None,
    v=None,
    max_rows=2048,
):
    """``reference_paged_prefill`` on the legacy batch.  A request longer than
    ``max_rows`` query rows is split into row chunks, each a pseudo-request
    whose KV is truncated to ``kv_len - q_len + chunk_end`` (bottom-right
    causal alignment and the sliding window depend only on the absolute
    query position, which the truncation preserves), so the (H, lq, lkv)
    score tensor of the 8190x7939 legacy config stays bounded.  Rows carrying
    a custom mask, or non-causal windows, are never chunked."""
    q = lb.q if q is None else q
    k = lb.k if k is None else k
    v = lb.v if v is None else v
    P = lb.page_size
    q_lens, kv_lens, pages = [], [], []
    for i in range(lb.batch_size):
        lq, lkv = int(lb.q_lens_cpu[i]), int(lb.kv_seq_lens_cpu[i])
        ids = lb.page_ids(i)
        chunk = custom_mask is None and lq > max_rows and (causal or window_left < 0)
        if not chunk:
            q_lens.append(lq)
            kv_lens.append(lkv)
            pages.append(ids[: (lkv + P - 1) // P])
            continue
        for a in range(0, lq, max_rows):
            b = min(a + max_rows, lq)
            kv_b = lkv - lq + b if causal else lkv
            q_lens.append(b - a)
            kv_lens.append(kv_b)
            pages.append(ids[: (kv_b + P - 1) // P])
    qo_indptr_cpu = torch.tensor(
        [0] + list(itertools.accumulate(q_lens)), dtype=torch.int32
    )
    kv_lens_cpu = torch.tensor(kv_lens, dtype=torch.int32)
    flat = torch.cat(pages).to(q.device, torch.int32)
    return reference_paged_prefill(
        q.contiguous(),
        k,
        v,
        qo_indptr_cpu,
        kv_lens_cpu,
        None,
        P,
        causal,
        sm_scale=sm_scale,
        window_left=window_left,
        kv_layout=lb.kv_layout,
        kv_page_indices=flat,
        lse_base=lse_base,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
    )


def assert_oracle(lb, out, lse, *, causal, lse_mode="base2", **kw):
    """The independent check: output against ``OUT_TOL``, LSE (when the plan
    asked for one) finite, contract-shaped and within ``LSE_TOL``."""
    ref_out, ref_lse = oracle_on_batch(
        lb, causal=causal, lse_base="e" if lse_mode == "basee" else "2", **kw
    )
    assert torch.isfinite(out).all(), "non-finite output"
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    if lse_mode == "none":
        assert lse is None
    else:
        assert (
            lse.shape == (out.shape[0], lb.num_qo_heads) and lse.dtype == torch.float32
        )
        assert torch.isfinite(lse).all(), "non-finite LSE"
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    return ref_out, ref_lse


# ---------------------------------------------------------------------------
# unified plan / run with explicit backend pinning
# ---------------------------------------------------------------------------


def resolve_batch_or_skip(
    lb,
    md,
    backend,
    *,
    causal=True,
    window_left=-1,
    need_lse=True,
    logits_soft_cap=None,
    custom_mask=False,
):
    """Resolution for the legacy configuration, or a skip that records WHY
    this backend cannot run the row (the report reads it as a capability
    exclusion, never as coverage)."""
    try:
        return resolve_paged_attention(
            device=torch.device(DEVICE),
            **lb.resolve_kwargs(),
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=md.kv_input_form,
            logits_soft_cap=logits_soft_cap,
            custom_mask=custom_mask,
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"[{backend}] capability-excluded: {e}")


def plan_batch(
    lb,
    md,
    backend,
    *,
    causal=True,
    window_left=-1,
    lse_mode="base2",
    logits_soft_cap=None,
    custom_mask=None,
    attn=None,
):
    """Resolve (or skip with the reason), plan on ``attn`` (a fresh eager
    instance by default) and assert the pinned backend was chosen."""
    res = resolve_batch_or_skip(
        lb,
        md,
        backend,
        causal=causal,
        window_left=window_left,
        need_lse=lse_mode != "none",
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask is not None,
    )
    attn = attn if attn is not None else PagedAttention(torch.device(DEVICE))
    attn.plan(
        md,
        **lb.plan_kwargs(),
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        logits_soft_cap=logits_soft_cap,
        custom_mask=custom_mask,
        backend=backend,
    )
    if backend == "auto":
        assert attn.backend in res.backends, attn.explain()
    else:
        assert attn.backend == backend, attn.explain()
    return attn


def run_batch(
    lb,
    md,
    backend,
    *,
    q=None,
    k=None,
    v=None,
    sm_scale=None,
    k_scale=None,
    v_scale=None,
    **plan_kw,
):
    attn = plan_batch(lb, md, backend, **plan_kw)
    out, lse = attn.run(
        lb.q if q is None else q,
        (lb.k if k is None else k, lb.v if v is None else v),
        sm_scale=sm_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    return attn, out, lse


def graph_capacity_for(lb, md):
    """The capacity of exactly this batch in its paging form (dense: the
    table's width rule ``max_kv_len == table_width * page_size``)."""
    common = dict(
        batch_size=lb.batch_size,
        total_q_tokens=int(lb.q_indptr_cpu[-1]),
        max_q_len=int(lb.q_lens_cpu.max()),
        page_size=lb.page_size,
    )
    if md.kv_input_form == "block_tables":
        width = int(md.block_tables.shape[1])
        return GraphCapacity(
            max_kv_len=width * lb.page_size, table_width=width, **common
        )
    return GraphCapacity(
        max_kv_len=max(int(lb.kv_seq_lens_cpu.max()), 1),
        kv_input_form="page_indices",
        flat_capacity=int(md.kv_page_indices.shape[0]),
        **common,
    )


_SHARED_WORKSPACE = None


def shared_workspace(nbytes: int) -> torch.Tensor:
    """One growing caller-owned scratch buffer for the graph-mode rows (the
    tests run sequentially, so sharing it is legal).  ``workspace_requirements``
    asks for several GiB at the largest legacy graph geometries (B128 x
    kv2048 with 32 heads: capacity substitution plans the fa2 split-KV
    scratch for the capacity maxes), where the 128 MiB library default -- and
    the legacy wrapper's 128 MiB, hence the legacy xfail -- overflows."""
    global _SHARED_WORKSPACE
    if _SHARED_WORKSPACE is None or _SHARED_WORKSPACE.numel() < nbytes:
        _SHARED_WORKSPACE = None
        _SHARED_WORKSPACE = torch.empty(
            nbytes, dtype=torch.uint8, device=torch.device(DEVICE)
        )
    return _SHARED_WORKSPACE[:nbytes]


def run_batch_graph(
    lb,
    md,
    backend,
    *,
    q=None,
    k=None,
    v=None,
    sm_scale=None,
    k_scale=None,
    v_scale=None,
    warmup_runs=3,
    causal=True,
    **plan_kw,
):
    """The legacy ``use_cuda_graph=True`` rows through the unified graph
    lifecycle, in the legacy order: a capacity sized for the batch, a
    workspace sized by ``workspace_requirements`` for it, a plan on a warm-up
    batch (the same requests over ``q_len`` tokens if causal, one token
    otherwise -- the legacy warm-up planned one page per request), eager
    warm-up runs on a side stream, capture one ``run()``, ``update()`` to the
    legacy batch, replay.  Returns the captured output buffers."""
    dev = torch.device(DEVICE)
    q = lb.q if q is None else q
    k = lb.k if k is None else k
    v = lb.v if v is None else v
    width = (
        int(md.block_tables.shape[1]) if md.kv_input_form == "block_tables" else None
    )
    warm_lens = lb.q_lens_cpu.clamp(min=1) if causal else torch.ones_like(lb.q_lens_cpu)
    warm = lb.truncated(torch.minimum(warm_lens, lb.kv_seq_lens_cpu))
    warm_md = warm.metadata("dense" if width is not None else "csr", table_width=width)
    feature_kw = {
        key: plan_kw[key]
        for key in ("window_left", "logits_soft_cap")
        if key in plan_kw
    }
    res = resolve_batch_or_skip(lb, md, backend, causal=causal, **feature_kw)
    cap = graph_capacity_for(lb, md)
    # the legacy graph rows were an xfail for the wrapper's 128 MiB workspace
    # overflow; the unified contract sizes the scratch for the capacity
    nbytes = PagedAttention.workspace_requirements(
        cap,
        device=dev,
        **lb.plan_kwargs(),
        causal=causal,
        need_lse=plan_kw.get("lse_mode", "base2") != "none",
        use_cuda_graph=True,
        backend=res,
        **feature_kw,
    )
    attn = PagedAttention(
        dev, graph_capacity=cap, workspace_buffer=shared_workspace(nbytes)
    )
    plan_batch(warm, warm_md, backend, attn=attn, causal=causal, **plan_kw)
    kw = dict(sm_scale=sm_scale, k_scale=k_scale, v_scale=v_scale)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup_runs):
            out, lse = attn.run(q, (k, v), **kw)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(q, (k, v), **kw)
    attn.update(md)
    g.replay()
    torch.cuda.synchronize()
    return attn, out, lse


# ---------------------------------------------------------------------------
# rejection helpers for the support-surface gaps
# ---------------------------------------------------------------------------


def assert_every_backend_excluded(match, *, also=(), **cfg):
    """Every backend is excluded and each reason names the axis (``match``),
    one of the ``also`` axes the configuration violates first, or the
    device's compute capability; ``auto`` raises the aggregated ValueError."""
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        resolve_paged_attention(device=torch.device(DEVICE), backend="auto", **cfg)
    assert match in str(ei.value), str(ei.value)
    accepted = (match, "compute capability", *also)
    for backend in BACKENDS[:-1]:
        with pytest.raises(ValueError) as ei:
            resolve_paged_attention(device=torch.device(DEVICE), backend=backend, **cfg)
        msg = str(ei.value)
        assert any(a in msg for a in accepted), f"{backend}: {msg}"


def assert_nvfp4_unsupported(
    *,
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo,
    page_size,
    q_dtype,
    causal,
    kv_layout="NHD",
    what="",
):
    """The legacy NVFP4 rows: a packed uint8 KV cache is not a declared KV
    dtype (every backend excluded with the dtype -- or, for D512 /
    asymmetric entries, the head-dim -- reason) and ``run()`` has no
    ``kv_cache_sf``.  Flipping ``EXPECT_NVFP4_KV`` marks the row for the
    quantization-descriptor extension."""
    cfg = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim_qk,
        head_dim_vo=head_dim_vo,
        q_dtype=q_dtype,
        kv_dtype=torch.uint8,
        page_size=page_size,
        kv_layout=kv_layout,
        causal=causal,
        # the flat page-id form is legal at every page size (the dense form
        # is refused below MIN_DENSE_PAGE_SIZE before any backend is asked)
        kv_input_form="page_indices",
    )
    with pytest.raises(ValueError, match="no runnable backend") as ei:
        resolve_paged_attention(device=torch.device(DEVICE), backend="auto", **cfg)
    msg = str(ei.value)
    assert (
        "unsupported kv dtype torch.uint8" in msg or "unsupported head dims" in msg
    ), msg
    assert "kv_cache_sf" not in run_signature_params()
    if EXPECT_NVFP4_KV:
        pytest.fail(f"EXPECT_NVFP4_KV is set: port the legacy NVFP4 fixture ({what})")


def assert_rope_kwargs_rejected(lb, md=None, backend="fa2"):
    """The legacy ``pos_encoding_mode="ROPE_LLAMA"`` rows: ``plan()`` has no
    positional-encoding argument (RoPE is the caller's, applied before the
    call), so the legacy kwargs are a TypeError, never a silent NONE."""
    params = plan_signature_params()
    for name in ("pos_encoding_mode", "rope_scale", "rope_theta"):
        assert name not in params, f"plan() grew {name!r}: port the ROPE_LLAMA rows"
    if EXPECT_ROPE:
        pytest.fail(
            "EXPECT_ROPE is set: run the ROPE_LLAMA rows through the fused plan"
        )
    with pytest.raises(TypeError, match="pos_encoding_mode"):
        PagedAttention(torch.device(DEVICE)).plan(
            lb.metadata() if md is None else md,
            **lb.plan_kwargs(),
            causal=True,
            pos_encoding_mode="ROPE_LLAMA",
            backend=backend,
        )


def apply_external_rope(lb, *, q=None, k=None, rope_theta=1e4):
    """The migration adapter for the fused ROPE_LLAMA rows: rotate q at its
    absolute positions ``kv_len - q_len + r`` and the request's K pages at
    positions ``j`` with ``flashinfer.apply_rope_pos_ids`` (Llama
    non-interleaved, theta 1e4, scale 1: the fused kernel's defaults).
    Returns rotated copies; the caller's tensors are untouched.  Requests
    must not share pages (each page is rotated once, in its request's
    position frame)."""
    q = lb.q if q is None else q
    k = lb.k if k is None else k
    dev = q.device
    q_rot = q.clone()
    # keep K's strides: a combined-pool view (page stride 2 * P * H * D) must
    # stay in the same stride family as its V view (trtllm-gen / cake read V
    # with K's strides and reject a mismatch, ledger M16); clone() would
    # compact it
    k_rot = torch.empty_strided(k.shape, k.stride(), dtype=k.dtype, device=k.device)
    k_rot.copy_(k)
    P, Hk, D = lb.page_size, lb.num_kv_heads, lb.head_dim_qk
    for i in range(lb.batch_size):
        s, e = int(lb.q_indptr_cpu[i]), int(lb.q_indptr_cpu[i + 1])
        lq, lkv = e - s, int(lb.kv_seq_lens_cpu[i])
        if lq:
            pos_q = torch.arange(lkv - lq, lkv, dtype=torch.int32, device=dev)
            q_i = q[s:e].contiguous()
            q_rot[s:e] = flashinfer.apply_rope_pos_ids(
                q_i, q_i, pos_q, rope_theta=rope_theta
            )[0]
        ids = lb.page_ids(i).to(dev, torch.long)
        if ids.numel() == 0:
            continue
        pages = k_rot[ids]
        if lb.kv_layout == "HND":
            pages = pages.permute(0, 2, 1, 3)
        rows = pages.reshape(-1, Hk, D).contiguous()
        pos_k = torch.arange(rows.shape[0], dtype=torch.int32, device=dev)
        rows = flashinfer.apply_rope_pos_ids(rows, rows, pos_k, rope_theta=rope_theta)[
            1
        ]
        pages = rows.reshape(ids.numel(), P, Hk, D)
        if lb.kv_layout == "HND":
            pages = pages.permute(0, 2, 1, 3)
        k_rot[ids] = pages
    return q_rot, k_rot
