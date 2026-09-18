"""Shared helpers for the legacy -> unified parity tests, group B.

The group-B parity files (``test_paged_attention_legacy_parity_b_*.py``)
re-run each legacy paged-prefill test's OWN fixture through the unified
``PagedAttention`` API and assert the legacy tolerance against the legacy
reference and the independent fp32 oracle.  This module holds what every
file needs and nothing backend-specific:

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

from flashinfer.prefill import PagedAttentionMetadata, resolve_paged_attention

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
EXPECT_FA2_E5M2_KV = False  # kv_dtype=float8_e5m2 on fa2 (WP-T)
EXPECT_FA2_HEAD_DIM_512 = False  # (512, 512) on fa2 (WP-T; Gemma-4 shape)
EXPECT_TRTLLM_LARGE_PAGES = (
    False  # page_size 128/256/512/1024 on trtllm-gen, cake (WP-T)
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
EXPECT_SINK_JIT_URI_HAS_HEAD_DIM = False

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


def check_legacy_map(legacy_map: Sequence, namespace: dict) -> None:
    """Every unified function a LEGACY_MAP row names exists in the module and
    every row carries one of the four statuses (PLAN §2)."""
    statuses = {"equivalent", "partial", "unsupported-by-design", "native-only"}
    seen = set()
    for legacy, unified, status, note in legacy_map:
        assert legacy not in seen, f"duplicate legacy entry {legacy}"
        seen.add(legacy)
        assert status in statuses, (legacy, status)
        assert isinstance(note, str) and note, legacy
        for name in unified:
            assert callable(namespace.get(name)), f"{legacy}: {name} is not a test here"
