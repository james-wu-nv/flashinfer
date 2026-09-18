# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for the experimental unified ``flashinfer.prefill.PagedAttention``.

``PagedAttention.run(q, kv_cache)`` carries no paging metadata: ``plan()``
owns ``qo_indptr``, ``kv_seq_lens`` and the page table, and it fixes the KV
layout, masking and LSE base.  The legacy ``gqa_paged_prefill`` template
therefore cannot describe it (NHD-only axes, one head dim, no per-request
lengths, always causal, base-2 LSE hard-coded).  This module provides:

- :func:`paged_attention_trace_dispatch` — bound to ``PagedAttention.run``;
  reads the last successful plan through the facade's read-only
  ``_trace_context()`` and returns one stable template per plan identity
  (paging form, Q dtype, KV dtype, KV layout, causal, window, LSE mode,
  logits soft cap, attention sinks, custom mask).
- :class:`_PagedAttentionTraceTemplate` — normalizes the plan-owned inputs
  into the trace kwargs inside ``build_fi_trace_fn`` (the dispatcher's own
  ``**kwargs`` copy never reaches the base builder).
- a pure-torch reference matching ``tests/experimental/paged_attention_reference.py``
  and an ``init`` that builds a valid ``{"plan": ..., "run": ...}`` bundle.

Identity rules: ``op_type="gqa_paged"`` keeps the category; the name prefix
``paged_attention_{dense,csr}[_fp8kv]`` never collides with
``gqa_paged_prefill_*``; the plan's semantic knobs — including the paging
form and the dtypes the prefix spells out — are ``Const`` axes with fixed
integer values, so they enter ``definition_name()``, the reference signature
and the exported ``init``: a consumer rebuilds the traced variant from the
JSON's axes alone.  The resolved backend is NOT part of the identity: the
same mathematical contract must compare fa2, fa3, cuDNN and trtllm-gen.

Encodings (also documented in every definition's description):

    csr          0 = dense block table (``block_tables``), 1 = flat page ids
                 (``kv_page_indices``); also the ``dense`` / ``csr`` prefix
    fp8_kv       0 = K/V in q's dtype, 1 = float8_e4m3fn K/V dequantized by
                 ``k_scale`` / ``v_scale``; also the ``_fp8kv`` prefix
    q_dtype      0 = bfloat16, 1 = float16 queries (and output)
    kv_layout    0 = HND ``(pages, num_kv_heads, page_size, head_dim)``
                 1 = NHD ``(pages, page_size, num_kv_heads, head_dim)``
    causal       0 = every query token sees the whole KV prefix
                 1 = bottom-right aligned causal mask
    window_left  -1 = unlimited, otherwise the sliding-window extent
    lse_mode     0 = no LSE, 1 = base-2 LSE, 2 = natural-log LSE
    logits_soft_cap  0 = off, otherwise the integer cap of
                 ``cap * tanh(score / cap)`` applied to the scaled scores
                 before masking (non-integer caps refuse to trace)
    use_sinks    1 = a ``sinks`` input ``(num_qo_heads,)`` fp32 adds one
                 logit per head to the softmax denominator (no value); the
                 LSE includes it
    use_custom_mask  1 = a plan-owned ``custom_mask`` input (flattened
                 per-request ``(q_len, kv_len)`` bool masks in request
                 order) is ANDed with the causal / window envelope

This module must stay importable without ``flashinfer.experimental``:
``import flashinfer`` binds the template, and the experimental package is
imported lazily by ``PagedAttention`` itself.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Dict

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var

_LAYOUTS = ("HND", "NHD")
_LSE_MODES = ("none", "base2", "basee")
_Q_DTYPES = (torch.bfloat16, torch.float16)
_FP8_KV_DTYPE = torch.float8_e4m3fn


# ── reference ────────────────────────────────────────────────────────────────


@torch.no_grad()
def _paged_attention_reference(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_seq_lens,
    block_tables=None,
    kv_page_indices=None,
    sm_scale=None,
    k_scale=None,
    v_scale=None,
    sinks=None,
    custom_mask=None,
    *,
    kv_layout,
    causal,
    window_left,
    lse_mode,
    csr=None,
    fp8_kv=None,
    q_dtype=None,
    logits_soft_cap=0,
    use_sinks=None,
    use_custom_mask=None,
):
    """FP32 paged attention over packed queries; returns ``(output, lse)``.

    Mirrors ``PagedAttention.run``: ``q`` is packed ``(total_q, Hq, Dqk)``
    with request ``b`` at rows ``qo_indptr[b]:qo_indptr[b+1]``; request ``b``
    reads ``kv_seq_lens[b]`` tokens from its pages — row ``b`` of the dense
    ``block_tables`` or the next ``ceil(len / page_size)`` entries of the flat
    ``kv_page_indices`` (request order) — with the last page partially used;
    a request without query rows (``q_len == 0``) contributes no rows and
    still consumes its pages.  ``kv_layout`` 0/1 = HND/NHD; ``causal`` is
    bottom-right aligned (query
    position ``p`` of a request with ``lq`` queries and ``lkv`` keys sits at
    key position ``lkv - lq + p``); ``window_left >= 0`` additionally hides
    keys more than ``window_left`` positions behind the query; ``lse_mode``
    0/1/2 = none/base-2/natural log.  ``k_scale`` / ``v_scale`` dequantize an
    fp8 KV cache (``real = stored * scale``).  Output dtype follows ``q``;
    the LSE is fp32 ``(total_q, Hq)`` or ``None`` when ``lse_mode == 0``.
    Features: ``logits_soft_cap`` (0 / None = off) applies ``cap * tanh(s /
    cap)`` to the scaled scores before masking; ``custom_mask`` (flattened
    per-request ``(q_len, kv_len)`` bool masks in request order, True = may
    attend) is ANDed with the causal / window envelope; ``sinks``
    ``(num_qo_heads,)`` adds one logit per head to the softmax denominator
    with no value contribution, so ``lse' = logaddexp(lse, sinks[h])`` and
    ``out' = out * exp(lse - lse')``; the returned LSE includes the sink.
    The variant axes are optional: ``csr`` 0/1 selects the paging table (the
    other one is ignored), ``fp8_kv`` 0/1 and ``q_dtype`` 0/1 = bf16/fp16
    are checked against the tensors, and ``use_sinks`` / ``use_custom_mask``
    = 1 require the matching input, so a definition is never evaluated on
    another variant's inputs.
    """
    if csr is not None:
        block_tables, kv_page_indices = (
            (None, kv_page_indices) if csr else (block_tables, None)
        )
    if (block_tables is None) == (kv_page_indices is None):
        raise ValueError("exactly one of block_tables / kv_page_indices is required")
    if fp8_kv is not None and bool(fp8_kv) != (k_cache.dtype == torch.float8_e4m3fn):
        raise ValueError(
            f"fp8_kv={fp8_kv} does not match the K cache dtype {k_cache.dtype}"
        )
    if q_dtype is not None and q.dtype != (torch.bfloat16, torch.float16)[q_dtype]:
        raise ValueError(f"q_dtype={q_dtype} does not match q's dtype {q.dtype}")
    if use_sinks and sinks is None:
        raise ValueError("use_sinks=1 needs the sinks input (num_qo_heads,)")
    if use_custom_mask and custom_mask is None:
        raise ValueError("use_custom_mask=1 needs the custom_mask input")
    cap = float(logits_soft_cap) if logits_soft_cap else None
    if kv_layout == 1:  # NHD -> HND view
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)
    total_q, num_qo_heads, head_dim_qk = q.shape
    num_kv_heads, page_size = k_cache.shape[1], k_cache.shape[2]
    head_dim_vo = v_cache.shape[3]
    group = num_qo_heads // num_kv_heads
    scale = head_dim_qk**-0.5 if sm_scale is None else float(sm_scale)
    offsets = [int(x) for x in qo_indptr.tolist()]
    lengths = [int(x) for x in kv_seq_lens.tolist()]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_vo), dtype=q.dtype, device=q.device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
    )
    page_start = 0
    mask_off = 0
    for b, lkv in enumerate(lengths):
        s, e = offsets[b], offsets[b + 1]
        lq = e - s
        n_pages = (lkv + page_size - 1) // page_size
        if block_tables is not None:
            ids = block_tables[b, :n_pages]
        else:
            ids = kv_page_indices[page_start : page_start + n_pages]
            page_start += n_pages
        if lq == 0 or lkv == 0:
            continue  # nothing to attend; the rows stay zero / -inf
        ids = ids.to(torch.long)
        k = (
            k_cache[ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim_qk)[:, :lkv]
            .to(torch.float32)
        )
        v = (
            v_cache[ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim_vo)[:, :lkv]
            .to(torch.float32)
        )
        if k_scale is not None:
            k = k * float(k_scale)
        if v_scale is not None:
            v = v * float(v_scale)
        k = k.repeat_interleave(group, dim=0)  # (Hq, lkv, Dqk)
        v = v.repeat_interleave(group, dim=0)  # (Hq, lkv, Dvo)
        scores = torch.einsum("qhd,hkd->hqk", q[s:e].to(torch.float32), k) * scale
        if cap is not None:  # soft cap on the scaled scores, before masking
            scores = cap * torch.tanh(scores / cap)
        qpos = torch.arange(lq, device=q.device).unsqueeze(1) + (lkv - lq)
        kpos = torch.arange(lkv, device=q.device).unsqueeze(0)
        allowed = torch.ones((lq, lkv), dtype=torch.bool, device=q.device)
        if causal:
            allowed &= kpos <= qpos
        if window_left >= 0:
            allowed &= kpos >= qpos - window_left
        if custom_mask is not None:  # ANDed with the causal / window envelope
            block = custom_mask[mask_off : mask_off + lq * lkv]
            allowed &= block.view(lq, lkv).to(torch.bool)
            mask_off += lq * lkv
        scores = scores.masked_fill(~allowed.unsqueeze(0), -float("inf"))
        row_lse = torch.logsumexp(scores, dim=-1)  # (Hq, lq)
        rows = torch.einsum("hqk,hkd->qhd", torch.softmax(scores, dim=-1), v)
        if sinks is not None:  # one extra logit per head, no value
            with_sink = torch.logaddexp(row_lse, sinks.to(torch.float32).unsqueeze(1))
            rows = rows * torch.exp(row_lse - with_sink).transpose(0, 1).unsqueeze(-1)
            row_lse = with_sink
        lse[s:e] = row_lse.transpose(0, 1)
        output[s:e] = rows.to(q.dtype)
    if lse_mode == 0:
        return output, None
    if lse_mode == 1:
        lse = lse / math.log(2.0)
    return output, lse


# ── init ─────────────────────────────────────────────────────────────────────


def _paged_attention_init(
    *,
    total_q: int = 8,
    batch_size: int = 2,
    len_indptr: int = 0,
    num_pages: int = 0,
    max_pages: int = 0,
    num_kv_indices: int = 0,
    num_qo_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim_qk: int = 128,
    head_dim_vo: int = 128,
    page_size: int = 16,
    kv_layout: int = 0,
    causal: int = 1,
    window_left: int = -1,
    lse_mode: int = 0,
    csr: int = 0,
    fp8_kv: int = 0,
    q_dtype: int = 0,
    logits_soft_cap: int = 0,
    use_sinks: int = 0,
    use_custom_mask: int = 0,
    num_pages_per_seq: int = 4,
    backend: str = "auto",
    device: str = "cuda",
    seed: int = 0,
    **unused,
):
    """Build ``{"plan": {...}, "run": {...}}`` for ``flashinfer.prefill.PagedAttention``.

    ``attn = PagedAttention(device); attn.plan(**inputs["plan"]);
    attn.run(**inputs["run"])`` is a valid planned execution of the traced
    workload.  The Const axes of the definition (heads, dims, page_size, and
    the encoded ``kv_layout`` 0/1 = HND/NHD, ``causal`` 0/1, ``window_left``
    -1/N, ``lse_mode`` 0/1/2 = none/base2/basee, ``csr`` 0/1 = dense block
    table / flat page ids, ``fp8_kv`` 0/1, ``q_dtype`` 0/1 = bf16/fp16,
    ``logits_soft_cap`` 0 = off / integer cap, ``use_sinks`` 0/1,
    ``use_custom_mask`` 0/1) select the variant; the Var axes size the
    workload and are honoured when given (0 = unspecified):

    - ``total_q`` (>= 1) is split evenly over ``batch_size`` requests (or
      ``len_indptr - 1`` when ``len_indptr`` is given); with ``total_q <
      batch_size`` the last requests have no query rows (vLLM's padded
      ``query_start_loc`` tail), each still reading at least one KV page.
    - dense form: the block table is ``(batch_size, max_pages)`` (default
      width ``num_pages_per_seq``); its columns are capacity, not live pages.
      Each request's live pages are at least its query tokens' pages and at
      most the width; with ``num_pages`` they are sized so the pool holds
      them all (spare pool pages spread over the requests, the remaining
      columns idle), without it every column is live.
    - flat form: the live page-id list has exactly ``num_kv_indices``
      entries, spread over the requests (default ``num_pages_per_seq`` per
      request).
    - ``num_pages`` is the K/V pool size (default: one physical page per
      live page).  Page ids are a random permutation of the pool; a pool
      smaller than the live pages the batch needs makes requests share pages
      (legal: a shared prefix), but a single request never repeats one, so
      the pool must hold the longest request's pages.  Every request's last
      live page is partial where the page size allows.

    Per-request KV lengths never fall below the request's query length, so
    the bundle satisfies the definition's constraints for the causal variant
    too.  Q (and the output) take ``q_dtype``; K/V take q's dtype, or with
    ``fp8_kv`` are quantized per tensor to float8_e4m3fn (scale = amax / 448)
    with the ``k_scale`` / ``v_scale`` dequantization scales in the run
    bundle, and ``plan`` receives the matching ``kv_dtype``.  The feature
    axes reach ``plan`` as ``logits_soft_cap`` (float or None), ``use_sinks``
    and ``custom_mask`` (random per-request masks whose bottom-right
    diagonal stays allowed, so no query row is fully masked) and the run
    bundle carries a random fp32 ``sinks`` vector when ``use_sinks`` is set.
    Requires CUDA (the metadata contract is device-resident).
    """
    del unused
    # The experimental package is imported only when init runs.
    from flashinfer.prefill import PagedAttentionMetadata

    if len_indptr:
        if batch_size and batch_size != len_indptr - 1:
            raise ValueError(
                f"len_indptr ({len_indptr}) must equal batch_size + 1 "
                f"({batch_size + 1})"
            )
        batch_size = len_indptr - 1
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if total_q < 1:
        raise ValueError(f"total_q must be >= 1, got {total_q}")
    torch.manual_seed(seed)
    # even split; total_q < batch_size leaves the last requests without query
    # rows (vLLM's padded query_start_loc tail), which the definition allows
    q_lens = torch.full((batch_size,), total_q // batch_size, dtype=torch.int64)
    q_lens[: total_q % batch_size] += 1
    # kv_len >= q_len and >= 1: a request keeps one KV page even without
    # query rows (kv_len == 0 padding rows are outside the definition)
    min_pages = torch.clamp((q_lens + page_size - 1) // page_size, min=1)

    # pages per request from the Var axes of the traced form
    if csr:
        if num_kv_indices:
            if num_kv_indices < int(min_pages.sum()):
                raise ValueError(
                    f"num_kv_indices ({num_kv_indices}) cannot hold one page per "
                    f"query token: the {batch_size} requests of {total_q} tokens "
                    f"need at least {int(min_pages.sum())} pages"
                )
            pages = min_pages.clone()
            spare = num_kv_indices - int(pages.sum())
            pages += spare // batch_size
            pages[: spare % batch_size] += 1
        else:
            pages = torch.maximum(min_pages, torch.tensor(num_pages_per_seq))
        table_width = int(pages.max())
    else:
        table_width = max_pages or max(num_pages_per_seq, int(min_pages.max()))
        if table_width < int(min_pages.max()):
            raise ValueError(
                f"max_pages ({max_pages}) x page_size ({page_size}) cannot hold "
                f"the longest request's {int(q_lens.max())} query tokens"
            )
        if num_pages:
            # live pages the pool can satisfy; the other columns stay idle
            if num_pages < int(min_pages.max()):
                raise ValueError(
                    f"num_pages ({num_pages}) is smaller than the "
                    f"{int(min_pages.max())} pages the longest request's "
                    f"{int(q_lens.max())} query tokens need"
                )
            pages = min_pages.clone()
            room = table_width - pages
            spare = num_pages - int(pages.sum())  # < 0: requests share pages
            if spare > 0:
                pages += torch.minimum(room, torch.tensor(spare // batch_size))
                spare = num_pages - int(pages.sum())
            for i in range(batch_size):
                if spare <= 0:
                    break
                if pages[i] < table_width:
                    pages[i] += 1
                    spare -= 1
        else:
            pages = torch.full((batch_size,), table_width, dtype=torch.int64)

    # partial last pages (request i leaves (i + 1) % page_size slots unused),
    # never shorter than the request's own query length
    kv_lens = torch.maximum(
        pages * page_size - ((torch.arange(batch_size) + 1) % page_size), q_lens
    )
    assert bool(((kv_lens + page_size - 1) // page_size == pages).all())

    pool_pages = num_pages or int(pages.sum())
    if pool_pages < int(pages.max()):
        raise ValueError(
            f"num_pages ({pool_pages}) is smaller than the {int(pages.max())} "
            "pages the longest request references"
        )
    perm = torch.randperm(pool_pages)  # scattered page ids
    # request-ordered pages (wrapping around a small pool); unused dense slots
    # hold valid pool pages that belong to nobody
    slots = torch.arange(batch_size * table_width).view(batch_size, table_width)
    block_tables_cpu = perm[slots % pool_pages].to(torch.int32)
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages[i])] for i in range(batch_size)]
    )
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0).to(torch.int32)]
    )
    kv_lens_cpu = kv_lens.to(torch.int32)
    common = dict(
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens_cpu.max()),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )
    qo_indptr = qo_indptr_cpu.to(device)
    kv_seq_lens = kv_lens_cpu.to(device)
    if csr:
        metadata = PagedAttentionMetadata.csr(
            qo_indptr, kv_seq_lens, kv_page_indices_cpu.to(device), **common
        )
    else:
        metadata = PagedAttentionMetadata.dense(
            qo_indptr, kv_seq_lens, block_tables_cpu.to(device), **common
        )
    if kv_layout == 0:
        cache_shape = (pool_pages, num_kv_heads, page_size)
    else:
        cache_shape = (pool_pages, page_size, num_kv_heads)
    dtype = (torch.bfloat16, torch.float16)[q_dtype]
    kv_dtype = dtype
    q = torch.randn(total_q, num_qo_heads, head_dim_qk, dtype=dtype, device=device)
    k_cache = torch.randn(*cache_shape, head_dim_qk, dtype=dtype, device=device)
    v_cache = torch.randn(*cache_shape, head_dim_vo, dtype=dtype, device=device)
    run = {"q": q, "kv_cache": (k_cache, v_cache)}
    if fp8_kv:
        kv_dtype = torch.float8_e4m3fn
        k_scale = float(k_cache.abs().amax().item()) / 448.0
        v_scale = float(v_cache.abs().amax().item()) / 448.0
        k_cache = (k_cache.float() / k_scale).to(kv_dtype)
        v_cache = (v_cache.float() / v_scale).to(kv_dtype)
        run = {
            "q": q,
            "kv_cache": (k_cache, v_cache),
            "k_scale": k_scale,
            "v_scale": v_scale,
        }
    plan = {
        "metadata": metadata,
        "num_qo_heads": int(num_qo_heads),
        "num_kv_heads": int(num_kv_heads),
        "head_dim_qk": int(head_dim_qk),
        "head_dim_vo": int(head_dim_vo),
        "q_dtype": dtype,
        "kv_dtype": kv_dtype,
        "kv_layout": ("HND", "NHD")[kv_layout],
        "causal": bool(causal),
        "window_left": int(window_left),
        "lse_mode": ("none", "base2", "basee")[lse_mode],
        "logits_soft_cap": float(logits_soft_cap) if logits_soft_cap else None,
        "use_sinks": bool(use_sinks),
        "backend": backend,
    }
    if use_sinks:
        run["sinks"] = torch.randn(num_qo_heads, dtype=torch.float32, device=device)
    if use_custom_mask:
        # per-request (q_len, kv_len) bool masks flattened in request order;
        # the bottom-right diagonal stays allowed so no query row is empty
        blocks = []
        for lq, lkv in zip(q_lens.tolist(), kv_lens.tolist(), strict=True):
            block = torch.rand(lq, lkv) < 0.5
            rows = torch.arange(lq)
            block[rows, lkv - lq + rows] = True
            blocks.append(block.flatten())
        plan["custom_mask"] = torch.cat(blocks).to(device)
    return {"plan": plan, "run": run}


# ── template ─────────────────────────────────────────────────────────────────


def _bound_trace_context(wrapper: Any) -> Dict[str, Any]:
    """The planned instance's read-only trace context, or a clear diagnostic."""
    if wrapper is None:
        raise ValueError(
            "Tracing PagedAttention.run requires the planned instance: use "
            "flashinfer.fi_trace(attn.run, q=q, kv_cache=(k, v)) (or auto-dump "
            "through attn.run(...)) instead of PagedAttention.run.fi_trace(...)"
        )
    getter = getattr(wrapper, "_trace_context", None)
    if getter is None:
        raise TypeError(
            "Tracing PagedAttention.run: expected a flashinfer.prefill."
            f"PagedAttention instance as self, got {type(wrapper).__name__}"
        )
    return getter()  # ValueError before the first successful plan


def _identity_from_context(ctx: Dict[str, Any]) -> Dict[str, int]:
    # The soft cap is a Const axis, and Const values are integers: the plan
    # normalizes the cap to a positive float (None = off); refuse a cap the
    # definition cannot spell rather than round it.
    cap = ctx.get("logits_soft_cap")
    if cap is not None and float(cap) != int(cap):
        raise ValueError(
            f"Tracing PagedAttention.run: logits_soft_cap {cap!r} is not an "
            "integer; the paged_attention trace definition encodes the cap as an "
            "integer Const axis (cap * tanh(score / cap)); trace a plan with an "
            "integer cap"
        )
    # The definition's constraints require kv_seq_lens >= 1 (its reference
    # has no padding-row convention), while plan() accepts kv_len == 0 padding
    # rows; a trace of such a batch would violate its own constraints.
    kv_cpu = ctx.get("kv_seq_lens_cpu")
    if kv_cpu is not None and kv_cpu.numel() and int(kv_cpu.min()) == 0:
        raise ValueError(
            "Tracing PagedAttention.run: the planned batch contains padding rows "
            "(kv_len == 0), which the paged_attention trace definition excludes "
            "(min(kv_seq_lens) >= 1); trace a batch without padding rows"
        )
    q_dtype, kv_dtype = ctx["q_dtype"], ctx["kv_dtype"]
    if q_dtype not in _Q_DTYPES or kv_dtype not in (q_dtype, _FP8_KV_DTYPE):
        raise ValueError(
            f"Tracing PagedAttention.run: the plan's dtypes (q {q_dtype}, kv "
            f"{kv_dtype}) are not encoded by the paged_attention trace definition "
            "(q bfloat16 / float16, kv the same or float8_e4m3fn)"
        )
    return dict(
        csr=int(ctx["kv_input_form"] == "page_indices"),
        fp8_kv=int(kv_dtype == _FP8_KV_DTYPE),
        q_dtype=_Q_DTYPES.index(q_dtype),
        kv_layout=_LAYOUTS.index(ctx["kv_layout"]),
        causal=int(bool(ctx["causal"])),
        window_left=int(ctx["window_left"]),
        lse_mode=_LSE_MODES.index(ctx["lse_mode"]),
        logits_soft_cap=int(cap) if cap else 0,
        use_sinks=int(bool(ctx.get("use_sinks"))),
        use_custom_mask=int(bool(ctx.get("has_custom_mask"))),
    )


def _require_run_tensors(kwargs: Dict[str, Any]) -> None:
    missing = []
    if not isinstance(kwargs.get("q"), torch.Tensor):
        missing.append("q")
    kv_cache = kwargs.get("kv_cache")
    if not (
        isinstance(kv_cache, (tuple, list))
        and len(kv_cache) == 2
        and all(isinstance(t, torch.Tensor) for t in kv_cache)
    ):
        missing.append("kv_cache=(k_cache, v_cache)")
    if missing:
        raise ValueError(
            "Tracing PagedAttention.run requires the run() tensor argument(s): "
            + ", ".join(missing)
        )


class _PagedAttentionTraceTemplate(TraceTemplate):
    """One plan identity of ``PagedAttention.run``.

    ``build_fi_trace_fn`` normalizes the trace kwargs before the base builder
    sees them: with a bound planned instance (``self`` injected by
    ``flashinfer.fi_trace(attn.run, ...)`` or by auto-dump) the plan-owned
    ``qo_indptr`` / ``kv_seq_lens`` / paging table come from the read-only
    trace context and the plan identity must match this template; without an
    instance an explicit ``metadata=`` (the init round trip) is accepted.  The
    semantic Const values are passed on as int kwargs for the reference.
    """

    def __init__(self, *args, identity: Dict[str, int], **kwargs):
        super().__init__(*args, **kwargs)
        self.identity = dict(identity)

    def normalize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """The trace kwargs the base builder sees for one ``run()`` call.

        Bound instance: the plan-owned tensors come from the trace context,
        and ``q`` / ``out`` / ``lse`` are cut to the batch's live rows
        (``qo_indptr[-1]``, as views) — in CUDA-graph mode they are capacity
        buffers with headroom rows the kernel never reads or writes, and the
        definition's ``total_q == qo_indptr[-1]`` constraint describes the
        live rows only.  The plan-owned ``custom_mask`` comes from the
        context too, and a plan with sinks needs the ``sinks=`` tensor
        ``run()`` takes.  A ``q`` with fewer rows than the plan, or tensors
        whose dtypes differ from the planned ``q_dtype`` / ``kv_dtype`` (the
        schema would then contradict the identity), are refused.
        """
        normalized = dict(kwargs)
        wrapper = normalized.pop("self", None)
        if wrapper is not None:
            ctx = _bound_trace_context(wrapper)
            planned = _identity_from_context(ctx)
            if planned != self.identity:
                raise ValueError(
                    "Tracing PagedAttention.run: the planned instance "
                    f"({planned}) does not match this template "
                    f"({self.identity}); trace through "
                    "flashinfer.fi_trace(attn.run, ...), which dispatches on "
                    "the plan"
                )
            normalized["qo_indptr"] = ctx["qo_indptr"]
            normalized["kv_seq_lens"] = ctx["kv_seq_lens"]
            normalized["block_tables"] = ctx["block_tables"]
            normalized["kv_page_indices"] = ctx["kv_page_indices"]
            normalized["custom_mask"] = ctx.get("custom_mask")
            _require_run_tensors(normalized)
            if self.identity["use_sinks"]:
                sinks = normalized.get("sinks")
                if not (
                    isinstance(sinks, torch.Tensor)
                    and sinks.dtype == torch.float32
                    and tuple(sinks.shape) == (int(ctx["num_qo_heads"]),)
                ):
                    raise ValueError(
                        "Tracing PagedAttention.run: the plan declared use_sinks; "
                        f"pass the fp32 ({ctx['num_qo_heads']},) sinks= tensor "
                        "run() takes"
                    )
            got = (normalized["q"].dtype, *(t.dtype for t in normalized["kv_cache"]))
            want = (ctx["q_dtype"], ctx["kv_dtype"], ctx["kv_dtype"])
            if got != want:
                raise ValueError(
                    "Tracing PagedAttention.run: q / k_cache / v_cache dtypes "
                    f"{got} differ from the plan's {want}; pass the tensors the "
                    "plan was made for"
                )
            live = int(ctx["total_q_tokens"])
            for key in ("q", "out", "lse"):
                t = normalized.get(key)
                if not isinstance(t, torch.Tensor):
                    continue
                if t.dim() == 0 or t.shape[0] < live:
                    raise ValueError(
                        f"Tracing PagedAttention.run: {key} has {tuple(t.shape)} "
                        f"rows but the planned batch has {live} query tokens "
                        "(qo_indptr[-1]); pass the buffer the plan was made for"
                    )
                if t.shape[0] > live:
                    normalized[key] = t[:live]
        else:
            metadata = normalized.pop("metadata", None)
            if metadata is not None:
                normalized.setdefault("qo_indptr", metadata.qo_indptr)
                normalized.setdefault("kv_seq_lens", metadata.kv_seq_lens)
                normalized.setdefault("block_tables", metadata.block_tables)
                normalized.setdefault("kv_page_indices", metadata.kv_page_indices)
            _require_run_tensors(normalized)
        normalized.update(self.identity)  # the reference's variant scalars
        return normalized

    def build_fi_trace_fn(self, fi_api):
        base_fi_trace = super().build_fi_trace_fn(fi_api)
        template = self

        def fi_trace(save_dir=None, name=None, **kwargs):
            return base_fi_trace(
                save_dir=save_dir, name=name, **template.normalize_kwargs(kwargs)
            )

        fi_trace.__doc__ = base_fi_trace.__doc__
        return fi_trace


def _paged_attention_template(
    *,
    csr: int = 0,
    kv_layout: int = 0,
    causal: int = 1,
    window_left: int = -1,
    lse_mode: int = 0,
    fp8_kv: int = 0,
    q_dtype: int = 0,
    logits_soft_cap: int = 0,
    use_sinks: int = 0,
    use_custom_mask: int = 0,
) -> _PagedAttentionTraceTemplate:
    """One stable template object per plan identity.

    Validates the encoding, then hits a positional ``lru_cache`` so every
    spelling of the same identity (defaults or explicit kwargs) returns the
    same object: the decorator's fi-trace cache is keyed by template id and
    lives for the process, so discovery and repeated traces must not create
    new templates.
    """
    for key, value, allowed in (
        ("csr", csr, (0, 1)),
        ("kv_layout", kv_layout, (0, 1)),
        ("causal", causal, (0, 1)),
        ("lse_mode", lse_mode, (0, 1, 2)),
        ("fp8_kv", fp8_kv, (0, 1)),
        ("q_dtype", q_dtype, (0, 1)),
        ("use_sinks", use_sinks, (0, 1)),
        ("use_custom_mask", use_custom_mask, (0, 1)),
    ):
        if type(value) is not int or value not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    if type(window_left) is not int or window_left < -1:
        raise ValueError(f"window_left must be an int >= -1, got {window_left!r}")
    if type(logits_soft_cap) is not int or logits_soft_cap < 0:
        raise ValueError(
            f"logits_soft_cap must be an int >= 0 (0 = off), got {logits_soft_cap!r}"
        )
    return _build_paged_attention_template(
        csr,
        kv_layout,
        causal,
        window_left,
        lse_mode,
        fp8_kv,
        q_dtype,
        logits_soft_cap,
        use_sinks,
        use_custom_mask,
    )


@lru_cache(maxsize=None)
def _build_paged_attention_template(
    csr: int,
    kv_layout: int,
    causal: int,
    window_left: int,
    lse_mode: int,
    fp8_kv: int,
    q_dtype: int,
    logits_soft_cap: int,
    use_sinks: int,
    use_custom_mask: int,
) -> _PagedAttentionTraceTemplate:
    layout_name = _LAYOUTS[kv_layout]
    form = "csr" if csr else "dense"
    if kv_layout == 0:
        cache_dims = ["num_pages", "num_kv_heads", "page_size"]
    else:
        cache_dims = ["num_pages", "page_size", "num_kv_heads"]

    axes: Dict[str, Any] = {
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim_qk": Const(abbrev="dqk"),
        "head_dim_vo": Const(abbrev="dvo"),
        "page_size": Const(abbrev="ps"),
        # The variant the name prefix spells out, as values a consumer can
        # hand to init / the reference (abbrev "" = already in the prefix).
        "csr": Const(
            abbrev="",
            value=csr,
            description="Paging form fixed by plan(): 0 = dense block table "
            "(block_tables), 1 = flat page ids (kv_page_indices); also the "
            "dense / csr name prefix.",
        ),
        "fp8_kv": Const(
            abbrev="",
            value=fp8_kv,
            description="0 = K/V cache in q's dtype, 1 = float8_e4m3fn K/V "
            "dequantized by k_scale / v_scale; also the _fp8kv name prefix.",
        ),
        "q_dtype": Const(
            abbrev="qdtype",
            value=q_dtype,
            description="Query and output dtype fixed by plan(): 0 = bfloat16, "
            "1 = float16.",
        ),
        "kv_layout": Const(
            abbrev="layout",
            value=kv_layout,
            description="Plan-time KV layout: 0 = HND (pages, H, page_size, D), "
            "1 = NHD (pages, page_size, H, D).",
        ),
        "causal": Const(
            abbrev="causal",
            value=causal,
            description="1 = bottom-right aligned causal mask, 0 = none.",
        ),
        "window_left": Const(
            abbrev="wl",
            value=window_left,
            description="Sliding-window extent fixed by plan(); -1 = unlimited.",
        ),
        "lse_mode": Const(
            abbrev="lse",
            value=lse_mode,
            description="LSE base fixed by plan(): 0 = none, 1 = base-2, "
            "2 = natural log.",
        ),
        # Feature knobs fixed by plan(); off = 0 and absent from the name.
        "logits_soft_cap": Const(
            abbrev="cap" if logits_soft_cap else "",
            value=logits_soft_cap,
            description="Logits soft cap fixed by plan(): scores = cap * "
            "tanh(scores / cap) on the scaled scores before masking; 0 = off. "
            "Integer caps only (Const axes are integers).",
        ),
        "use_sinks": Const(
            abbrev="sinks" if use_sinks else "",
            value=use_sinks,
            description="1 = per-head attention sinks: the sinks input adds one "
            "logit per head to the softmax denominator with no value; the LSE "
            "includes it.",
        ),
        "use_custom_mask": Const(
            abbrev="mask" if use_custom_mask else "",
            value=use_custom_mask,
            description="1 = the plan-owned custom_mask input (flattened "
            "per-request (q_len, kv_len) bool masks in request order, True = may "
            "attend) is ANDed with the causal / window envelope.",
        ),
        "total_q": Var(description="Total number of packed query tokens."),
        "batch_size": Var(description="Number of requests in the plan."),
        "len_indptr": Var(description="Length of qo_indptr (batch_size + 1)."),
        "num_pages": Var(description="Physical pages in the KV pool."),
    }
    inputs: Dict[str, Any] = {
        "q": Tensor(
            ["total_q", "num_qo_heads", "head_dim_qk"],
            description="Packed queries; request b occupies rows "
            "qo_indptr[b]:qo_indptr[b+1].",
        ),
        "k_cache": Tensor(
            cache_dims + ["head_dim_qk"],
            param="kv_cache",
            tuple_idx=0,
            description=f"Paged keys in the {layout_name} layout (kv_cache[0]).",
        ),
        "v_cache": Tensor(
            cache_dims + ["head_dim_vo"],
            param="kv_cache",
            tuple_idx=1,
            description=f"Paged values in the {layout_name} layout (kv_cache[1]).",
        ),
        "qo_indptr": Tensor(
            ["len_indptr"],
            dtype="int32",
            optional=True,
            description="Token-unit query prefix sums. Owned by plan(), "
            "filled from the planned instance when tracing.",
        ),
        "kv_seq_lens": Tensor(
            ["batch_size"],
            dtype="int32",
            optional=True,
            description="Valid KV length per request (last page may be "
            "partial). Owned by plan(), filled from the planned instance.",
        ),
    }
    if csr:
        axes["num_kv_indices"] = Var(
            description="Live flat page ids: sum(ceil(kv_seq_lens / page_size))."
        )
        inputs["kv_page_indices"] = Tensor(
            ["num_kv_indices"],
            dtype="int32",
            optional=True,
            description="Flat page ids in request order (sglang-style CSR). "
            "Owned by plan(); the traced tensor is the live prefix.",
        )
    else:
        axes["max_pages"] = Var(description="Dense block-table width (pages).")
        inputs["block_tables"] = Tensor(
            ["batch_size", "max_pages"],
            dtype="int32",
            optional=True,
            description="Dense page table (vLLM-style); row b uses its first "
            "ceil(kv_seq_lens[b] / page_size) entries. Owned by plan().",
        )
    if use_sinks:
        inputs["sinks"] = Tensor(
            ["num_qo_heads"],
            dtype="float32",
            description="Per-head attention-sink logits (run() argument): one "
            "extra softmax logit per head with no value contribution.",
        )
    if use_custom_mask:
        axes["mask_len"] = Var(
            description="Elements of custom_mask: sum over requests of q_len * kv_len."
        )
        inputs["custom_mask"] = Tensor(
            ["mask_len"],
            dtype="bool",
            optional=True,
            description="Flattened per-request (q_len, kv_len) bool masks in "
            "request order (True = may attend), ANDed with the causal / window "
            "envelope. Owned by plan(), filled from the planned instance.",
        )
    inputs.update(
        {
            "sm_scale": Scalar(
                "float32",
                optional=True,
                description="Softmax scale for this call; default 1/sqrt(head_dim_qk).",
            ),
            "k_scale": Scalar(
                "float32",
                optional=True,
                description="Per-tensor dequantization scale of an fp8 K cache "
                "(real = stored * k_scale); fp8 KV only.",
            ),
            "v_scale": Scalar(
                "float32",
                optional=True,
                description="Per-tensor dequantization scale of an fp8 V cache; "
                "fp8 KV only.",
            ),
            "csr": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the csr axis; fixed by plan(), "
                "passed to the reference.",
            ),
            "fp8_kv": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the fp8_kv axis; fixed by plan().",
            ),
            "q_dtype": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the q_dtype axis; fixed by plan().",
            ),
            "kv_layout": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the kv_layout axis; fixed by plan(), "
                "passed to the reference.",
            ),
            "causal": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the causal axis; fixed by plan().",
            ),
            "window_left": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the window_left axis; fixed by plan().",
            ),
            "lse_mode": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the lse_mode axis; fixed by plan().",
            ),
            "logits_soft_cap": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the logits_soft_cap axis (0 = off); "
                "fixed by plan().",
            ),
            "use_sinks": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the use_sinks axis; fixed by plan().",
            ),
            "use_custom_mask": Scalar(
                "int32",
                optional=True,
                description="Same encoding as the use_custom_mask axis; fixed by "
                "plan().",
            ),
        }
    )
    outputs: Dict[str, Any] = {
        "output": Tensor(
            ["total_q", "num_qo_heads", "head_dim_vo"],
            dtype_from="q",
            param="out",
            description="Attention output in q's dtype (optional out= buffer).",
        ),
    }
    if lse_mode:
        outputs["lse"] = Tensor(
            ["total_q", "num_qo_heads"],
            dtype="float32",
            param="lse",
            description=(
                "Packed fp32 log-sum-exp of the attention logits in "
                f"{'base 2' if lse_mode == 1 else 'natural log'} "
                "(optional lse= buffer)."
            ),
        )
    constraints = [
        "num_qo_heads % num_kv_heads == 0",
        "len_indptr == batch_size + 1",
        "total_q == qo_indptr[-1].item()",
        "min(qo_indptr[1:] - qo_indptr[:-1]) >= 0",
        "min(kv_seq_lens) >= 1",
        "causal == 0 or min(kv_seq_lens - (qo_indptr[1:] - qo_indptr[:-1])) >= 0",
        "window_left >= -1",
        "logits_soft_cap >= 0",
    ]
    if use_custom_mask:
        constraints.append(
            "mask_len == ((qo_indptr[1:] - qo_indptr[:-1]) * kv_seq_lens).sum().item()"
        )
    if csr:
        constraints.append(
            "num_kv_indices >= ((kv_seq_lens + page_size - 1) // page_size).sum().item()"
        )
    else:
        constraints.append("max_pages * page_size >= max(kv_seq_lens)")
    tags = [
        "status:experimental",
        "stage:prefill",
        "stage:decode",
        f"form:{form}",
        f"layout:{layout_name}",
        f"mask:{'causal' if causal else 'noncausal'}",
        f"window:{window_left}",
        f"lse:{_LSE_MODES[lse_mode]}",
        f"q:{('bf16', 'fp16')[q_dtype]}",
    ]
    if fp8_kv:
        tags.append("kv:fp8")
    if logits_soft_cap:
        tags.append(f"feature:softcap{logits_soft_cap}")
    if use_sinks:
        tags.append("feature:sinks")
    if use_custom_mask:
        tags.append("feature:custom_mask")
    description = (
        "Experimental unified PagedAttention.run: packed queries over a paged "
        f"KV cache in the {layout_name} layout, paging metadata from plan() in "
        f"the {'flat page-id (CSR)' if csr else 'dense block-table'} form"
        f"{' with an fp8 KV cache (k_scale/v_scale dequantize)' if fp8_kv else ''}. "
        "Const axes encode the plan: csr 0=dense block table/1=flat page ids, "
        "fp8_kv 0=K/V in q's dtype/1=float8_e4m3fn K/V, q_dtype "
        "0=bfloat16/1=float16, kv_layout 0=HND/1=NHD, causal 0/1 "
        "(bottom-right aligned), window_left -1=unlimited, lse_mode "
        "0=none/1=base-2/2=natural log, logits_soft_cap 0=off/integer cap "
        "(cap * tanh(score / cap) before masking), use_sinks 0/1 (a sinks input "
        "adds one logit per head to the softmax denominator; the LSE includes "
        "it), use_custom_mask 0/1 (a plan-owned custom_mask input ANDed with the "
        "causal / window envelope); the reference takes the same values as int "
        "scalars and init rebuilds the variant from them. qo_indptr, "
        "kv_seq_lens, the page table and custom_mask are plan() "
        "inputs, filled from the planned instance by "
        "flashinfer.fi_trace(attn.run, ...). The resolved backend is not part "
        "of this identity."
    )
    return _PagedAttentionTraceTemplate(
        op_type="gqa_paged",
        name_prefix=f"paged_attention_{form}{'_fp8kv' if fp8_kv else ''}",
        description=description,
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        tags=tags,
        reference=_paged_attention_reference,
        init=_paged_attention_init,
        identity=dict(
            csr=csr,
            fp8_kv=fp8_kv,
            q_dtype=q_dtype,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            lse_mode=lse_mode,
            logits_soft_cap=logits_soft_cap,
            use_sinks=use_sinks,
            use_custom_mask=use_custom_mask,
        ),
    )


def paged_attention_trace_dispatch(**kwargs):
    """Select the template for the bound instance's last successful plan.

    ``flashinfer.fi_trace(attn.run, ...)`` and auto-dump inject the instance
    as ``self``; ``PagedAttention.run.fi_trace(...)`` cannot, and raises.
    """
    ctx = _bound_trace_context(kwargs.get("self"))
    return _paged_attention_template(**_identity_from_context(ctx))


# Finite representatives for discovery / consistency tooling (both paging
# forms).  Bound dispatch above synthesizes the exact identity from the plan.
paged_attention_trace_dispatch.templates = [  # type: ignore[attr-defined]
    _paged_attention_template(),
    _paged_attention_template(csr=1),
]

__all__ = ["paged_attention_trace_dispatch"]
