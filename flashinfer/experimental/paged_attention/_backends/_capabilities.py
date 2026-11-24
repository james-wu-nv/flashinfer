"""Declarative capability facts for the unified paged-prefill backends.

Mirrors ``flashinfer/mla/_batch_mla/_backends/_capabilities.py``: one frozen
capability value per backend plus a pure rejection-reason check. This is a
description, not a registry: it explains why a backend cannot run a
configuration and never falls back on its own (``_selection.py`` does the
choosing).

Capability honesty rule: ``CAPABILITIES`` declares ONLY what the conformance
matrix and fuzzer actually exercise on hardware. Production entries are wider
(cuDNN head dims beyond 128 / (192, 128), trtllm-gen head dims 256 / 512 and
fp8 KV, compute capability 11, ...); here an admitted config is a
machine-checked config, so under-claiming is the only honest default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


class _BackendPlanUnsupportedError(ValueError):
    """Typed signal for a plan-time (batch-specific) preflight rejection.

    Mirrors the private type of the same name in
    ``flashinfer/mla/_batch_mla/_backends/_capabilities.py`` (also a
    ``ValueError`` subclass); the two are to be unified once the shared
    location is decided.  A backend raises it from ``preflight(meta)`` -- before
    any live-state write -- for a batch it cannot run although the static
    capability check admitted the configuration.  The controller treats it as
    "try the next pinned candidate" and, when the backend was named or no
    candidate is left, raises a plain ``ValueError`` carrying the reason.  Its
    handler names this type, so every other exception (invalid input -- a plain
    ``ValueError`` included --, OOM, JIT/compile failure) propagates unchanged,
    and ``run()`` never falls back.
    """


def _workspace_too_small(need: int, available: int) -> str:
    """Reason for a plan-time rejection of a scratch workspace the backend's
    planner would overflow.  Raised as the typed signal from ``preflight()``,
    before the kernel-side allocator error: ``auto`` moves on to the next
    candidate (the bytes stay in the selection trace), and a named backend, or
    a set in which no candidate fits, surfaces it as a ``ValueError`` with
    this text after the backend's name (the controller adds the name, as it
    does for every preflight reason)."""
    return (
        f"needs {need} bytes of scratch workspace for this batch but the "
        f"workspace buffer holds {available} bytes; size a "
        "caller-owned buffer with PagedAttention.workspace_requirements(...) "
        "and pass it as PagedAttention(workspace_buffer=...)"
    )


# Below this page size a dense (b, max_pages) block table degenerates toward
# (b, max_context): the dense INPUT form requires page_size >= this floor, and
# dense DERIVATION from the flat-indices form is refused below it (backends
# that need the dense table are capability-excluded instead).
MIN_DENSE_PAGE_SIZE = 8


@dataclass(frozen=True)
class PagedAttentionCapabilities:
    """Static description of what one backend can run.

    This is the queryable capability matrix from the proposal (§5.1) — the
    thing that replaces consumer-side support tables (vLLM's scattered gates,
    sglang's whitelists). Declared == machine-checked by the test suites.
    """

    name: str
    # compute-capability majors (minor-insensitive for the prototype)
    cc_majors: frozenset
    q_dtypes: frozenset
    head_dims: frozenset  # of (head_dim_qk, head_dim_vo) pairs
    page_sizes: Optional[frozenset]  # None = any >= the global floor
    kv_layouts: frozenset
    supports_lse: bool
    supports_noncausal: bool
    supports_window: bool
    # True if the backend addresses q as packed THD (token stride == H*D);
    # False if every view the controller admits is addressable (unit inner
    # stride, token-major, 16-byte aligned: required for everyone, see
    # run()).  Measured per backend by
    # tests/experimental/paged_attention/test_paged_attention_strides.py
    # (NATIVE_Q_OUTCOME).  q reaches the library only at run(), after plan()
    # ranked the auto candidates, so the rank cannot route around a True
    # here: an auto candidate must declare False (no backend declares True
    # today).
    requires_contiguous_q: bool
    # sliding window together with non-causal attention (trtllm-gen ships no
    # such context kernel; irrelevant where either axis is already False)
    supports_window_noncausal: bool = True
    # KV-cache dtypes; fp8 entries (e4m3 / e5m2) mean "fp8 KV with a fp16/bf16
    # q", dequantized by the per-tensor k_scale/v_scale of run().  fp8 q is a
    # separate, undeclared axis.
    kv_dtypes: frozenset = frozenset({torch.float16, torch.bfloat16})
    # True if the backend consumes the dense block table (derivation from the
    # flat-indices input form is forbidden below page_size 8 — table blowup)
    needs_dense: bool = False
    # Explicit-False feature axes: a backend that cannot apply a requested
    # feature is EXCLUDED here so `auto` never drops the feature silently.
    #   logits soft cap:  cap * tanh(score / cap) on the scaled scores
    #   custom mask:      per-request flattened boolean mask ANDed into the
    #                     causal/window envelope
    #   sinks:            per-head extra softmax-denominator logit (run-time
    #                     tensor; declared at plan time)
    supports_logits_soft_cap: bool = False
    supports_custom_mask: bool = False
    supports_sinks: bool = False
    # head-dim pairs the sink kernel variant is verified on; None = every
    # declared pair (the sink JIT variant is compiled per head dim, so a
    # backend can be right at one size and wrong at another — see fa2 / M21)
    sinks_head_dims: Optional[frozenset] = None
    # split-KV control: plan(fixed_split_size=...) / plan(disable_split_kv=True),
    # the controls that make a request's output independent of the rest of
    # the batch.  They are arguments of the fa2 planner (PrefillPlan in
    # include/flashinfer/attention/scheduler.cuh); a backend without them is
    # excluded when either is requested.
    supports_split_kv_control: bool = False

    def rejection_reason(
        self,
        *,
        cc_major: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        head_dim_qk: int,
        head_dim_vo: int,
        page_size: int,
        kv_layout: str,
        causal: bool,
        need_lse: bool,
        window_left: int,
        kv_input_form: str,
        logits_soft_cap: Optional[float] = None,
        use_custom_mask: bool = False,
        use_sinks: bool = False,
        split_kv_control: bool = False,
    ) -> Optional[str]:
        """Return None if runnable, else the exclusion reason (for explain())."""
        if cc_major not in self.cc_majors:
            return f"unsupported compute capability sm_{cc_major}x"
        if q_dtype not in self.q_dtypes:
            return f"unsupported q dtype {q_dtype}"
        if kv_dtype not in self.kv_dtypes:
            return f"unsupported kv dtype {kv_dtype}"
        if kv_dtype != q_dtype and not (
            _is_fp8(kv_dtype) and q_dtype in (torch.float16, torch.bfloat16)
        ):
            return f"unsupported q/kv dtype pair ({q_dtype}, {kv_dtype})"
        if (head_dim_qk, head_dim_vo) not in self.head_dims:
            return f"unsupported head dims ({head_dim_qk}, {head_dim_vo})"
        if self.page_sizes is not None and page_size not in self.page_sizes:
            return f"unsupported page_size {page_size} (supported: {sorted(self.page_sizes)})"
        if kv_layout not in self.kv_layouts:
            return f"unsupported kv_layout {kv_layout}"
        if not causal and not self.supports_noncausal:
            return "non-causal attention not supported"
        if need_lse and not self.supports_lse:
            return "LSE output not supported"
        if window_left >= 0 and not self.supports_window:
            return "sliding window (window_left >= 0) not supported"
        if window_left >= 0 and not causal and not self.supports_window_noncausal:
            return "sliding window with non-causal attention not supported"
        if logits_soft_cap is not None and not self.supports_logits_soft_cap:
            return "logits soft cap not supported"
        if use_custom_mask and not self.supports_custom_mask:
            return "custom attention mask not supported"
        if use_sinks and not self.supports_sinks:
            return "attention sinks not supported"
        if (
            use_sinks
            and self.sinks_head_dims is not None
            and (head_dim_qk, head_dim_vo) not in self.sinks_head_dims
        ):
            return (
                f"attention sinks not supported at head dims ({head_dim_qk}, "
                f"{head_dim_vo}) (verified: {sorted(self.sinks_head_dims)})"
            )
        if split_kv_control and not self.supports_split_kv_control:
            return (
                "split-KV control (fixed_split_size / disable_split_kv) not supported"
            )
        if (
            self.needs_dense
            and kv_input_form == "page_indices"
            and page_size < MIN_DENSE_PAGE_SIZE
        ):
            return (
                "needs a dense block table, and deriving one from flat page "
                f"indices at page_size {page_size} < {MIN_DENSE_PAGE_SIZE} "
                "would blow up to (batch, max_context) — CSR-native backends only"
            )
        return None


_F16 = frozenset({torch.float16, torch.bfloat16})
# Both fp8 KV formats the generated fa2 kernels dequantize in-kernel.  e5m2
# measured on B200 (2026-09-17) through the unified API against the oracle on
# the dequantized cache: the four conformance shapes with bf16 and fp16 q,
# head dims 64 / 128 / 256 / 512, NHD, CSR page 1, sliding window 16,
# non-causal, both LSE bases and a uniform-q1 decode batch all match (max
# out err 1.2e-2, LSE err 1.6e-3 -- the same magnitude as e4m3 on the same
# shapes); the legacy wrapper on the legacy fp8 fixture (batch 12, q 7,
# kv 54, page 16, H4:4) matches to 1.5e-5.
_FP8 = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


# Per the capability-honesty rule: these sets mirror exactly what
# tests/experimental/paged_attention/test_paged_attention_{prototype,fuzzer,coverage}.py
# exercise.  Production sets are wider (cuDNN head dims beyond those listed,
# trtllm-gen head dims 256 / 512 and fp8 KV per
# tests/attention/test_trtllm_gen_attention_prefill.py, ...).
CAPABILITIES: Dict[str, PagedAttentionCapabilities] = {
    "fa2": PagedAttentionCapabilities(
        name="fa2",
        # cc 11 (Thor/sm_110) intentionally undeclared: no hardware in the
        # verification pool (capability-honesty rule)
        cc_majors=frozenset({8, 9, 10, 12}),
        q_dtypes=_F16,
        # 512: the Ampere+ large-head path (Gemma-4 full attention; CTA tile
        # 16 / 32 by FA2DetermineCtaTileQ, modelled in fa_backend.py).
        # Measured on B200 (2026-09-17) through the unified API against the
        # oracle: H16:2 bf16 q [17, 5, 1, 33] / kv [300, 97, 1000, 40] in
        # HND and NHD, causal, non-causal, window 64, basee, CSR page 1
        # (max out err 7.6e-3); bf16 q with e4m3 / e5m2 KV (1.0e-2 /
        # 9.6e-3); the legacy fp16 fixture (batch 2, q 17, kv 97, H4:4,
        # NHD; 3.1e-4); fp16 q / e4m3 KV at q = kv = 128 (the CTA-32 shape;
        # 1.6e-3); the Gemma-4 accuracy shapes q 17 and q 1 over KV 10003
        # with e4m3 KV, k_scale = v_scale = 0.02, sm_scale 1 (1.1e-3 /
        # 4.9e-4); q 512 / 300 over kv 2048 / 1024 (1.1e-3).  LSE err
        # <= 4.8e-6 throughout.
        head_dims=frozenset({(64, 64), (128, 128), (256, 256), (512, 512)}),
        # fp8 (e4m3 / e5m2) KV + f16 q, per-tensor scales (in-kernel dequant)
        kv_dtypes=_F16 | _FP8,
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        # M20 (B200, 2026-09-17): the fa2 windowed KV range is trimmed as if
        # causal (include/flashinfer/attention/prefill.cuh), so non-causal +
        # sliding window is WRONG once a request has more than 128 query
        # tokens and a history (q 256 / kv 768 / window 128: rows 0..127 off by
        # up to 0.4; q 128 exact; causal exact).  Declared unsupported until the
        # kernel is fixed; the strict xfail in test_paged_attention_features.py
        # flips then.  fa3 shares the declaration until measured on SM90.
        supports_window_noncausal=False,
        requires_contiguous_q=False,
        # soft cap and custom mask: the generated fa2 kernels
        # (tests/attention/test_batch_prefill_kernels.py); sinks: the
        # AttentionSink JIT variant (tests/attention/test_attention_sink.py)
        supports_logits_soft_cap=True,
        supports_custom_mask=True,
        supports_sinks=True,
        # M21 (B200, 2026-09-18, found by the XQA legacy grid): the fa2
        # AttentionSink variant at head_dim 512 returns wrong values (max abs
        # err 1.4-1.75, LSE off by ~3) while D64/128/256 sinks and D512 without
        # sinks are exact; the legacy sink wrapper is wrong there too.  The
        # pair is excluded until the kernel is fixed.
        sinks_head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        # Both are arguments of the fa2 planner.  fa3 is not declared: its
        # planner (PrefillSM90Plan) takes neither, and no SM90 device is in
        # the verification pool to measure what it does without them.
        supports_split_kv_control=True,
    ),
    "fa3": PagedAttentionCapabilities(
        name="fa3",
        cc_majors=frozenset({9}),
        q_dtypes=_F16,
        # (192,128) is NOT declared: the paged fa kernels require
        # k_page_stride == v_page_stride, which separately-allocated
        # K(192)/V(128) pools violate (needs a stride-matched allocation
        # contract; cudnn covers (192,128) without one).
        head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        # M20 (B200, 2026-09-17): the fa2 windowed KV range is trimmed as if
        # causal (include/flashinfer/attention/prefill.cuh), so non-causal +
        # sliding window is WRONG once a request has more than 128 query
        # tokens and a history (q 256 / kv 768 / window 128: rows 0..127 off by
        # up to 0.4; q 128 exact; causal exact).  Declared unsupported until the
        # kernel is fixed; the strict xfail in test_paged_attention_features.py
        # flips then.  fa3 shares the declaration until measured on SM90.
        supports_window_noncausal=False,
        requires_contiguous_q=False,
        # soft cap: tests/attention/test_hopper.py (cap 30); sinks: the
        # AttentionSink JIT variant; custom mask: the SM90 batch prefill
        # kernels reject MaskMode.CUSTOM (utils.is_fa3_backend_supported)
        supports_logits_soft_cap=True,
        supports_custom_mask=False,
        supports_sinks=True,
    ),
    "cudnn": PagedAttentionCapabilities(
        name="cudnn",
        cc_majors=frozenset({8, 9, 10, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128), (192, 128)}),
        page_sizes=None,
        kv_layouts=frozenset(
            {"HND", "NHD"}
        ),  # NHD = permuted view, stride-driven graph
        supports_lse=True,
        supports_noncausal=True,  # bottom_right mask off + padding mask: verified H100
        # Not declared by this API's cuDNN backend, whose plan() and run()
        # pass no window.  The cuDNN graph path has one (``window_left`` in
        # flashinfer/cudnn/prefill.py), and the legacy wrapper's explicit
        # cuDNN backend passes it.
        supports_window=False,
        # The cuDNN graph declares q with q's own strides and is keyed on
        # them, and the token-unit ragged offsets are scaled by q's token
        # stride (flashinfer/cudnn/prefill.py: the graph's ragged-offset
        # multiplier on the direct path, ``offsets * q.stride(0)`` on the
        # element-offset path), so the views the controller admits (unit
        # inner stride, token-major, 16-byte aligned) are addressed
        # correctly.  Measured on SM100 (2026-11-10/11; cuDNN 9.27 / frontend
        # 1.32 and cuDNN 9.25 / frontend 1.29) through the unified API
        # against the oracle: a fused-QKV head slice, a slice from the middle
        # of a fused buffer, a padded head stride, a storage offset, token
        # stride 2 and a padded token stride; the first four also on the
        # functional entry and replayed from a CUDA graph across re-plans,
        # with and without LSE (max out err 8.5e-3, the packed q's own).  A
        # zero token stride (an expanded q) is not addressed -- the graph
        # refuses it, the element-offset path misreads it -- and the
        # controller refuses it for every backend.  Before the cuDNN graph
        # refactor (#5350 / #5161) the offsets were scaled by
        # num_qo_heads * head_dim and a strided view was silently wrong.
        requires_contiguous_q=False,
        needs_dense=True,
        # None of the three is declared.  The cuDNN graph path has no soft
        # cap and no custom mask; it has a sink token, which this backend
        # does not pass and the legacy paged wrapper refuses on cuDNN.
        supports_logits_soft_cap=False,
        supports_custom_mask=False,
        supports_sinks=False,
    ),
    "trtllm-gen": PagedAttentionCapabilities(
        name="trtllm-gen",
        cc_majors=frozenset({10}),
        q_dtypes=_F16,
        # (64, 64): the legacy sink context suite
        # (tests/attention/test_attention_sink_blackwell.py) passes every D64
        # point on B200, and the unified conversion runs the same fixtures on
        # trtllm-gen against the legacy reference and the oracle.
        head_dims=frozenset({(64, 64), (128, 128)}),
        # Every page size the paged context kernel ships for.  128 .. 1024
        # measured on B200 (2026-09-17) through the unified API against the
        # oracle: GQA 32/8 and 8/2, bf16, q [257, 1, 100, 40] over kv
        # [2P + 37, 300, P, 3P - 1], dense and CSR-derived tables (max out
        # err 4.2e-3, LSE err 2.9e-6); at 8/2 also fp16 NHD, window 127,
        # non-causal, LSE none / basee, a uniform-q1 decode batch at 1024
        # and a CUDA-graph capture / replay at 1024 (6.1e-4).
        page_sizes=frozenset({16, 32, 64, 128, 256, 512, 1024}),
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        # Measured on B200 (2026-09-16) against the oracle with causal=False:
        # the four conformance shapes in bf16 and fp16, decode shape, NHD,
        # CSR page indices, base-e LSE and a q_len > kv_len row all match
        # (max out err 5e-3, LSE err < 1e-4); non-causal + sliding window
        # has no kernel, see supports_window_noncausal.
        supports_noncausal=True,
        supports_window=True,
        supports_window_noncausal=False,
        # The launcher passes q's token and head strides; a fused-QKV head
        # slice, a padded head stride and a storage offset all match the
        # oracle (native probe, B200).  The TMA descriptor needs a unit
        # inner stride, token-major storage and 16-byte alignment; the
        # controller enforces the three for every backend (see run()).
        requires_contiguous_q=False,
        needs_dense=True,
        # sinks: native run-time argument of trtllm_batch_context_with_kv_cache
        # (tests/attention/test_attention_sink_blackwell.py); the kernel has
        # no soft cap ("logits_soft_cap must be 0.0 for trtllm-gen") and no
        # custom mask
        supports_logits_soft_cap=False,
        supports_custom_mask=False,
        supports_sinks=True,
    ),
    # The separately versioned Cake FMHA product behind
    # trtllm_batch_context_with_kv_cache(backend="cake"): same dialect, same
    # envelope as trtllm-gen here.  Measured on B200 (2026-09-16) against the
    # oracle: the four conformance shapes in bf16 and fp16 with and without
    # LSE, decode shape, non-causal, sliding window 16, sinks with and
    # without LSE, NHD -- all match (max out err 5e-3, LSE err < 1e-4); the
    # prototype / fuzzer / cuda-graph / workspace suites run it as a backend.
    # Exact targets only (sm_100 / sm_103; the product raises on others),
    # checked by the selection probe.
    "cake": PagedAttentionCapabilities(
        name="cake",
        cc_majors=frozenset({10}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128)}),
        # 128 .. 1024 measured for cake itself on B200 (2026-09-17), same
        # grid as trtllm-gen above: GQA 32/8 and 8/2 dense and CSR-derived,
        # fp16 NHD, window 127, non-causal, LSE none / basee, q1 decode at
        # 1024, graph replay at 1024 -- max out err 5.5e-3, LSE err 5.0e-5.
        page_sizes=frozenset({16, 32, 64, 128, 256, 512, 1024}),
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        supports_window_noncausal=False,  # shared front door with trtllm-gen
        # measured on B200 (2026-09-16): fused-QKV head slice, head-stride gap
        # and storage offset all match the oracle (the binding forwards the
        # token and head strides, like trtllm-gen)
        requires_contiguous_q=False,
        needs_dense=True,
        supports_logits_soft_cap=False,
        supports_custom_mask=False,
        supports_sinks=True,
    ),
}

__all__ = [
    "CAPABILITIES",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "_BackendPlanUnsupportedError",
    "_workspace_too_small",
]
