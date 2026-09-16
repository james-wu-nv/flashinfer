"""Declarative capability facts for the unified paged-prefill backends.

Mirrors ``flashinfer/mla/_batch_mla/_backends/_capabilities.py``: one frozen
capability value per backend plus a pure rejection-reason check. This is a
description, not a registry: it explains why a backend cannot run a
configuration and never falls back on its own (``_selection.py`` does the
choosing).

Capability honesty rule: ``CAPABILITIES`` declares ONLY what the conformance
matrix and fuzzer actually exercise on hardware. Production entries are wider
(fa2 head_dim 64/256, trtllm large pages with GQA, NHD layouts, ...); here an
admitted config is a machine-checked config, so under-claiming is the only
honest default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


class _BackendPlanUnsupportedError(RuntimeError):
    """Typed signal for a plan-time (batch-specific) preflight rejection.

    Mirrors the private type of the same name in
    ``flashinfer/mla/_batch_mla/_backends/_capabilities.py`` (also a
    ``RuntimeError`` subclass); the two are to be unified once the shared
    location is decided.  A backend raises it from ``preflight(meta)`` -- before
    any live-state write -- for a batch it cannot run although the static
    capability check admitted the configuration.  The controller treats it as
    "try the next pinned candidate"; every other exception (invalid input, OOM,
    JIT/compile failure) propagates unchanged, and ``run()`` never falls back.
    """


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
    # False if any view with q.stride(-1) == 1 is addressable (the controller
    # requires the unit inner stride for everyone).  Measured per backend by
    # tests/experimental/test_paged_attention_strides.py (NATIVE_Q_OUTCOME).
    requires_contiguous_q: bool
    # KV-cache dtypes; fp8 entries mean "fp8 KV with a fp16/bf16 q" (per-tensor
    # k_scale/v_scale at run()). fp8 q is a separate, undeclared axis.
    kv_dtypes: frozenset = frozenset({torch.float16, torch.bfloat16})
    # True if the backend consumes the dense block table (derivation from the
    # flat-indices input form is forbidden below page_size 8 — table blowup)
    needs_dense: bool = False
    # native LSE format, normalized by the backend:
    #   "base2_tokens_h"   — already the contract
    #   "base2_padded_bsh" — base-2 padded (b, max_q, h); backend gathers
    lse_native: str = "base2_tokens_h"

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
_FP8 = frozenset({torch.float8_e4m3fn})


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


# Per the capability-honesty rule: these sets mirror exactly what
# tests/experimental/test_paged_attention_{prototype,fuzzer}.py exercise.
# Production sets are wider (fa2 64/256 head dims, trtllm pages up to 1024
# with GQA per tests/attention/test_trtllm_gen_attention_prefill.py, NHD...).
CAPABILITIES: Dict[str, PagedAttentionCapabilities] = {
    "fa2": PagedAttentionCapabilities(
        name="fa2",
        # cc 11 (Thor/sm_110) intentionally undeclared: no hardware in the
        # verification pool (capability-honesty rule)
        cc_majors=frozenset({8, 9, 10, 12}),
        q_dtypes=_F16,
        head_dims=frozenset({(64, 64), (128, 128), (256, 256)}),
        kv_dtypes=_F16 | _FP8,  # fp8 KV + f16 q, per-tensor scales (in-kernel dequant)
        page_sizes=None,
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        supports_noncausal=True,
        supports_window=True,
        requires_contiguous_q=False,
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
        requires_contiguous_q=False,
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
        supports_window=False,  # no sliding window in the cuDNN SDPA graph path
        # cuDNN builds its graph from q.stride() but scales the token-unit
        # ragged offsets by num_qo_heads*head_dim, so only packed THD is
        # addressed correctly: a fused-QKV head slice or a padded head stride
        # is silently wrong (native probe, B200); a storage offset is fine.
        requires_contiguous_q=True,
        needs_dense=True,
        lse_native="base2_padded_bsh",
    ),
    "trtllm-gen": PagedAttentionCapabilities(
        name="trtllm-gen",
        cc_majors=frozenset({10}),
        q_dtypes=_F16,
        head_dims=frozenset({(128, 128)}),
        # 128+ pages are supported by the kernel with GQA (repo tests cover
        # up to 1024) — kept out until this suite exercises them.
        page_sizes=frozenset({16, 32, 64}),
        kv_layouts=frozenset({"HND", "NHD"}),
        supports_lse=True,
        # vLLM routes non-causal away from trtllm; unverified here, so False.
        supports_noncausal=False,
        supports_window=True,
        # The launcher passes q's token and head strides; a fused-QKV head
        # slice, a padded head stride and a storage offset all match the
        # oracle (native probe, B200).  Only the unit inner stride is a TMA
        # requirement, and the controller enforces that for every backend.
        requires_contiguous_q=False,
        needs_dense=True,
    ),
}

__all__ = [
    "CAPABILITIES",
    "MIN_DENSE_PAGE_SIZE",
    "PagedAttentionCapabilities",
    "_BackendPlanUnsupportedError",
]
