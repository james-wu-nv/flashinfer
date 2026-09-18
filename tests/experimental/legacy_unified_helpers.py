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
# modular legacy suites poison exactly that tail).  Flip when every backend
# ignores the tail (or the input contract states the tail must be finite).
EXPECT_INPAGE_TAIL_IGNORED = False
# fa2 attention sinks: the AttentionSink JIT variant declines an fp8 KV cache at
# plan time ("not verified", _backends/fa_backend.py) although the capability
# table admits fp8 KV and sinks separately -- the XQA legacy suite runs sinks
# with an fp8 KV natively.  Flip when the adapter verifies the pair.
EXPECT_FA2_SINKS_FP8_KV = False
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
