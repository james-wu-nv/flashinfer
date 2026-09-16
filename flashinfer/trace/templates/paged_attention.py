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
  (paging form, KV layout, causal, window, LSE mode, fp8 KV).
- :class:`_PagedAttentionTraceTemplate` — normalizes the plan-owned inputs
  into the trace kwargs inside ``build_fi_trace_fn`` (the dispatcher's own
  ``**kwargs`` copy never reaches the base builder).
- a pure-torch reference matching ``tests/experimental/paged_attention_reference.py``
  and an ``init`` that builds a valid ``{"plan": ..., "run": ...}`` bundle.

Identity rules: ``op_type="gqa_paged"`` keeps the category; the name prefix
``paged_attention_{dense,csr}`` never collides with ``gqa_paged_prefill_*``;
the plan's semantic knobs are ``Const`` axes with fixed integer values, so
they enter ``definition_name()`` and the reference signature.  The resolved
backend is NOT part of the identity: the same mathematical contract must
compare fa2, fa3, cuDNN and trtllm-gen.

Encodings (also documented in every definition's description):

    kv_layout    0 = HND ``(pages, num_kv_heads, page_size, head_dim)``
                 1 = NHD ``(pages, page_size, num_kv_heads, head_dim)``
    causal       0 = every query token sees the whole KV prefix
                 1 = bottom-right aligned causal mask
    window_left  -1 = unlimited, otherwise the sliding-window extent
    lse_mode     0 = no LSE, 1 = base-2 LSE, 2 = natural-log LSE

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
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


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
    *,
    kv_layout,
    causal,
    window_left,
    lse_mode,
):
    """FP32 paged attention over packed queries; returns ``(output, lse)``.

    Mirrors ``PagedAttention.run``: ``q`` is packed ``(total_q, Hq, Dqk)``
    with request ``b`` at rows ``qo_indptr[b]:qo_indptr[b+1]``; request ``b``
    reads ``kv_seq_lens[b]`` tokens from its pages — row ``b`` of the dense
    ``block_tables`` or the next ``ceil(len / page_size)`` entries of the flat
    ``kv_page_indices`` (request order) — with the last page partially used.
    ``kv_layout`` 0/1 = HND/NHD; ``causal`` is bottom-right aligned (query
    position ``p`` of a request with ``lq`` queries and ``lkv`` keys sits at
    key position ``lkv - lq + p``); ``window_left >= 0`` additionally hides
    keys more than ``window_left`` positions behind the query; ``lse_mode``
    0/1/2 = none/base-2/natural log.  ``k_scale`` / ``v_scale`` dequantize an
    fp8 KV cache (``real = stored * scale``).  Output dtype follows ``q``;
    the LSE is fp32 ``(total_q, Hq)`` or ``None`` when ``lse_mode == 0``.
    """
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
        qpos = torch.arange(lq, device=q.device).unsqueeze(1) + (lkv - lq)
        kpos = torch.arange(lkv, device=q.device).unsqueeze(0)
        allowed = torch.ones((lq, lkv), dtype=torch.bool, device=q.device)
        if causal:
            allowed &= kpos <= qpos
        if window_left >= 0:
            allowed &= kpos >= qpos - window_left
        scores = scores.masked_fill(~allowed.unsqueeze(0), -float("inf"))
        lse[s:e] = torch.logsumexp(scores, dim=-1).transpose(0, 1)
        probs = torch.softmax(scores, dim=-1)
        output[s:e] = torch.einsum("hqk,hkd->qhd", probs, v).to(q.dtype)
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
    num_pages_per_seq: int = 4,
    backend: str = "auto",
    device: str = "cuda",
    seed: int = 0,
    **unused,
):
    """Build ``{"plan": {...}, "run": {...}}`` for ``flashinfer.prefill.PagedAttention``.

    ``attn = PagedAttention(device); attn.plan(**inputs["plan"]);
    attn.run(**inputs["run"])`` is a valid planned execution.  Var axes are
    keyword-only; the Const axes of the traced definition (heads, dims,
    page_size, and the encoded ``kv_layout`` 0/1 = HND/NHD, ``causal`` 0/1,
    ``window_left`` -1/N, ``lse_mode`` 0/1/2 = none/base2/basee, ``csr`` 0/1
    = dense block table / flat page ids) are accepted as kwargs so a consumer
    can rebuild exactly the traced variant.  ``total_q`` is split evenly over
    ``batch_size`` requests (``total_q >= batch_size``); every request owns
    ``num_pages_per_seq`` scattered pages and uses a partially filled last
    page.  Q/K/V are bf16.  Requires CUDA (the metadata contract is
    device-resident).
    """
    del len_indptr, num_pages, max_pages, num_kv_indices, unused
    # The experimental package is imported only when init runs.
    from flashinfer.prefill import PagedAttentionMetadata

    if total_q < batch_size:
        raise ValueError(
            f"total_q ({total_q}) must be >= batch_size ({batch_size}): every "
            "request needs at least one query token"
        )
    torch.manual_seed(seed)
    q_lens = torch.full((batch_size,), total_q // batch_size, dtype=torch.int32)
    q_lens[: total_q % batch_size] += 1
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    full_len = num_pages_per_seq * page_size
    # partial last pages, never shorter than the request's own query length
    kv_lens_cpu = torch.maximum(
        torch.full((batch_size,), full_len, dtype=torch.int32)
        - (torch.arange(batch_size, dtype=torch.int32) % page_size),
        q_lens,
    ).to(torch.int32)
    pages = (kv_lens_cpu + page_size - 1) // page_size
    pool_pages = batch_size * num_pages_per_seq
    perm = torch.randperm(pool_pages).to(torch.int32)  # scattered page ids
    block_tables_cpu = perm.view(batch_size, num_pages_per_seq).clone()
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages[i])] for i in range(batch_size)]
    )
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
    dtype = torch.bfloat16
    q = torch.randn(total_q, num_qo_heads, head_dim_qk, dtype=dtype, device=device)
    k_cache = torch.randn(*cache_shape, head_dim_qk, dtype=dtype, device=device)
    v_cache = torch.randn(*cache_shape, head_dim_vo, dtype=dtype, device=device)
    return {
        "plan": {
            "metadata": metadata,
            "num_qo_heads": int(num_qo_heads),
            "num_kv_heads": int(num_kv_heads),
            "head_dim_qk": int(head_dim_qk),
            "head_dim_vo": int(head_dim_vo),
            "q_dtype": dtype,
            "kv_layout": ("HND", "NHD")[kv_layout],
            "causal": bool(causal),
            "window_left": int(window_left),
            "lse_mode": ("none", "base2", "basee")[lse_mode],
            "backend": backend,
        },
        "run": {"q": q, "kv_cache": (k_cache, v_cache)},
    }


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
    return dict(
        csr=int(ctx["kv_input_form"] == "page_indices"),
        kv_layout=_LAYOUTS.index(ctx["kv_layout"]),
        causal=int(bool(ctx["causal"])),
        window_left=int(ctx["window_left"]),
        lse_mode=_LSE_MODES.index(ctx["lse_mode"]),
        fp8_kv=int(ctx["kv_dtype"] in _FP8_DTYPES),
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

    def build_fi_trace_fn(self, fi_api):
        base_fi_trace = super().build_fi_trace_fn(fi_api)
        template = self

        def fi_trace(save_dir=None, name=None, **kwargs):
            normalized = dict(kwargs)
            wrapper = normalized.pop("self", None)
            if wrapper is not None:
                ctx = _bound_trace_context(wrapper)
                planned = _identity_from_context(ctx)
                if planned != template.identity:
                    raise ValueError(
                        "Tracing PagedAttention.run: the planned instance "
                        f"({planned}) does not match this template "
                        f"({template.identity}); trace through "
                        "flashinfer.fi_trace(attn.run, ...), which dispatches on "
                        "the plan"
                    )
                normalized["qo_indptr"] = ctx["qo_indptr"]
                normalized["kv_seq_lens"] = ctx["kv_seq_lens"]
                normalized["block_tables"] = ctx["block_tables"]
                normalized["kv_page_indices"] = ctx["kv_page_indices"]
            else:
                metadata = normalized.pop("metadata", None)
                if metadata is not None:
                    normalized.setdefault("qo_indptr", metadata.qo_indptr)
                    normalized.setdefault("kv_seq_lens", metadata.kv_seq_lens)
                    normalized.setdefault("block_tables", metadata.block_tables)
                    normalized.setdefault("kv_page_indices", metadata.kv_page_indices)
            _require_run_tensors(normalized)
            for key in ("kv_layout", "causal", "window_left", "lse_mode"):
                normalized[key] = template.identity[key]
            return base_fi_trace(save_dir=save_dir, name=name, **normalized)

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
    ):
        if type(value) is not int or value not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got {value!r}")
    if type(window_left) is not int or window_left < -1:
        raise ValueError(f"window_left must be an int >= -1, got {window_left!r}")
    return _build_paged_attention_template(
        csr, kv_layout, causal, window_left, lse_mode, fp8_kv
    )


@lru_cache(maxsize=None)
def _build_paged_attention_template(
    csr: int,
    kv_layout: int,
    causal: int,
    window_left: int,
    lse_mode: int,
    fp8_kv: int,
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
        "min(qo_indptr[1:] - qo_indptr[:-1]) >= 1",
        "min(kv_seq_lens) >= 1",
        "causal == 0 or min(kv_seq_lens - (qo_indptr[1:] - qo_indptr[:-1])) >= 0",
        "window_left >= -1",
    ]
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
    ]
    if fp8_kv:
        tags.append("kv:fp8")
    description = (
        "Experimental unified PagedAttention.run: packed queries over a paged "
        f"KV cache in the {layout_name} layout, paging metadata from plan() in "
        f"the {'flat page-id (CSR)' if csr else 'dense block-table'} form"
        f"{' with an fp8 KV cache (k_scale/v_scale dequantize)' if fp8_kv else ''}. "
        "Const axes encode the plan: kv_layout 0=HND/1=NHD, causal 0/1 "
        "(bottom-right aligned), window_left -1=unlimited, lse_mode "
        "0=none/1=base-2/2=natural log; the reference takes the same values as "
        "int scalars. qo_indptr, kv_seq_lens and the page table are plan() "
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
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            lse_mode=lse_mode,
            fp8_kv=fp8_kv,
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
