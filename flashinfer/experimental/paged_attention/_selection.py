"""Static backend selection (proposal §5.3, level 1).

Tensor-free: callable at engine init, before the KV pool exists and before
any CUDA graph is captured. Evaluates every declared capability so the
returned ``Resolution`` carries a reason for each excluded backend — the
"explain" answer consumer-side tables cannot give.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

from ._backends._capabilities import CAPABILITIES
from ._contracts import (
    Resolution,
    _expect,
    _expect_page_size,
    _expect_window_left,
    _normalize_logits_soft_cap,
    resolve_config_key,
)


def _probe_fa(backend: str, device: Optional[torch.device]) -> Optional[str]:
    if backend == "fa3":
        from ...utils import is_sm90a_supported, version_at_least

        if device is not None:
            if not is_sm90a_supported(device):
                return "fa3 requires SM90a (Hopper) and CUDA >= 12.3"
        elif not version_at_least(torch.version.cuda, "12.3"):
            # explicit-cc_major resolution: the major is already gated by the
            # capability table; only the toolkit level can be checked here
            return "fa3 requires CUDA >= 12.3"
    return None


def _probe_cudnn(device: Optional[torch.device]) -> Optional[str]:
    from ...cudnn import prefill as cudnn_prefill

    if not cudnn_prefill.CUDNN_AVAILABLE:
        return "cudnn-frontend python package not importable"
    return None


def _probe_trtllm(device: Optional[torch.device]) -> Optional[str]:
    # Cubin availability is a real capability question (proposal: it should be
    # a library answer, not an engine-side HTTP probe).  The prototype defers
    # to first-run download; a production probe would consult the local cubin
    # cache / FLASHINFER_NO_DOWNLOAD for ``device``.
    return None


def _probe_cake(device: Optional[torch.device]) -> Optional[str]:
    # The Cake FMHA product ships exact targets only (sm_100a / sm_103a) and
    # raises for any other Blackwell part; the capability table can only
    # gate the major, so the minor is checked here when a device is known.
    if device is not None:
        from ...utils import get_compute_capability

        cc = get_compute_capability(device)
        if cc not in ((10, 0), (10, 3)):
            return f"cake ships sm_100a / sm_103a cubins only, not sm_{cc[0]}{cc[1]}"
    return None


# Environment probes: things the static capability table cannot know
# (installed packages, toolkit level). Run only for capability-admitted
# backends so explain() stays cheap.  Every probe answers for the TARGET
# device it is given (``None`` = resolved from an explicit cc_major without a
# device: only device-independent facts can be checked).
PROBES: Dict[str, Callable[[Optional[torch.device]], Optional[str]]] = {
    "fa2": lambda device: _probe_fa("fa2", device),
    "fa3": lambda device: _probe_fa("fa3", device),
    "cudnn": _probe_cudnn,
    "trtllm-gen": _probe_trtllm,
    "cake": _probe_cake,
}


def _bind_device(
    device: Optional[torch.device], cc_major: Optional[int]
) -> Tuple[Optional[torch.device], int, Optional[int], Optional[int]]:
    """Resolve the device binding ``(device, cc_major, cc_minor, device_index)``.

    - ``device`` given: read its compute capability (an explicit ``cc_major``
      must agree with it) and pin the Resolution to that device.
    - only ``cc_major`` given: pin the compute-capability major alone; the
      Resolution is NOT bound to a device (``cc_minor``/``device_index`` are
      ``None``) and device-dependent probes are skipped.
    - neither given: the current CUDA device.
    """
    if device is None and cc_major is not None:
        return None, cc_major, None, None
    dev = torch.device(device) if device is not None else torch.device("cuda")
    _expect(
        dev.type == "cuda",
        f"resolve_paged_attention(device=...) must be a CUDA device, got {dev}",
    )
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(dev)
    if cc_major is not None:
        _expect(
            cc_major == props.major,
            f"cc_major={cc_major} contradicts device {dev} (sm_{props.major}"
            f"{props.minor}); pass one or the other",
        )
    return dev, int(props.major), int(props.minor), int(dev.index)


# Static preference order per compute-capability major, bucketed by the
# caller's ``max_q_len`` hint (proposal §5.2: seeded from the benchmark suite;
# it only has to beat consumer tables that rot).  Each entry is
# ``(bound, order)``: the order applies to a hint ``<= bound``; entries are
# ascending and the last one (``bound=None``) is the default, used with no hint
# or a hint above every bound.  ``auto`` stays a static selection: the hint is
# the only shape fact it reads, and plan() never reorders.
#
# sm_100 numbers (B200, bf16, GQA 32/8, head_dim 128, page 16, causal, dense
# table, CUDA-graph ``run`` medians in µs; full sweep in
# reports/unified-prefill-implementation-20260916/wp-k.md).  trtllm-gen and
# cake run the paged CONTEXT kernel for every batch, whose per-request tile
# is sized for prefill, so at decode / speculative query lengths it reads the
# whole KV of a request for a handful of tokens and fa2's split-KV decode
# schedule wins:
#   B=32, kv=4096:   q=1   fa2  92   trtllm-gen  513  cudnn  143  (fa2 5.6x)
#                    q=8   fa2 247   trtllm-gen  588  cudnn  515  (2.4x)
#                    q=16  fa2 248   trtllm-gen  618  cudnn  516  (2.5x)
#                    q=64  fa2 491   trtllm-gen  667  cudnn  518  (1.4x)
#                    q=256 fa2 OOM   trtllm-gen  697  cudnn  670  (crossover)
#   B=1,  kv=1024:   q=16  fa2  12   trtllm-gen   18                (1.5x)
#                    q=64  fa2  20   trtllm-gen   18                (0.9x)
#   trtllm-gen / fa2 over the 12 (batch, kv_len) cells: q<=16 min 1.00
#   (B=8, kv=1024, q=16 tie) median 2.5-5.3 max 9.5; q=64 min 0.47 median
#   1.31; q>=256 always < 1 (0.25-0.84).  The bound below is the largest
#   q_len at which fa2 is at least as fast as trtllm-gen on every measured
#   cell; sweep in wp-k-sweep.csv.
HEURISTIC_ORDER: Dict[int, Tuple[Tuple[Optional[int], Tuple[str, ...]], ...]] = {
    10: (
        # decode / speculative: fa2 first, then the context kernels
        (16, ("fa2", "trtllm-gen", "cake", "cudnn")),
        # default (chunked / full prefill, or no hint): cake right after
        # trtllm-gen — same dialect and envelope, so it is the natural next
        # candidate when trtllm-gen declines a batch
        (None, ("trtllm-gen", "cake", "cudnn", "fa2")),
    ),
    9: ((None, ("fa3", "fa2", "cudnn")),),
    8: ((None, ("fa2", "cudnn")),),
    12: ((None, ("fa2", "cudnn")),),
}


def heuristic_order(cc_major: int, max_q_len: Optional[int] = None) -> Tuple[str, ...]:
    """The static preference order for ``cc_major`` under the ``max_q_len``
    hint (``None`` = no hint: the default bucket); empty for an unknown major."""
    for bound, order in HEURISTIC_ORDER.get(cc_major, ()):
        if bound is None or (max_q_len is not None and max_q_len <= bound):
            return order
    return ()


def _expect_max_q_len_hint(max_q_len: Optional[int]) -> None:
    _expect(
        max_q_len is None
        or (
            isinstance(max_q_len, int)
            and not isinstance(max_q_len, bool)
            and max_q_len >= 1
        ),
        "max_q_len must be None (no hint) or a positive host int: the largest "
        "number of query tokens per request the batches planned with this "
        f"Resolution will have, got {max_q_len!r}",
    )


def resolve_paged_attention(
    *,
    device: Optional[torch.device] = None,
    cc_major: Optional[int] = None,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: Optional[int] = None,
    q_dtype: torch.dtype,
    kv_dtype: Optional[torch.dtype] = None,
    page_size: int,
    kv_layout: str = "HND",
    causal: bool = True,
    need_lse: bool = False,
    window_left: int = -1,
    kv_input_form: str = "block_tables",
    logits_soft_cap: Optional[float] = None,
    custom_mask: bool = False,
    sinks: bool = False,
    max_q_len: Optional[int] = None,
    backend: str = "auto",
) -> Resolution:
    """Static backend resolution — no plan state, no tensors.

    Raises ``ValueError`` when nothing can run, with per-backend reasons.
    The result is pinned to ``device`` (or to the current CUDA device when
    neither ``device`` nor ``cc_major`` is given); with only ``cc_major`` it
    is pinned to that compute-capability major and to no device.

    ``logits_soft_cap`` / ``custom_mask`` / ``sinks`` declare the features the
    plans will use; a backend that cannot apply one is excluded (with the
    reason) rather than dropping it silently.

    ``max_q_len`` is an optional shape hint: the batches planned with this
    Resolution have at most this many query tokens per request.  It selects
    the candidate ORDER (``HEURISTIC_ORDER`` bucket) — never the candidate
    set — and is pinned in ``config``, so plan() rejects a batch above it.
    """
    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    if kv_dtype is None:
        kv_dtype = q_dtype
    dev, cc_major, cc_minor, device_index = _bind_device(device, cc_major)
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) and num_kv_heads ({num_kv_heads}) "
            "must be positive integers"
        )
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads}) for GQA/MQA"
        )
    if kv_input_form not in ("block_tables", "page_indices"):
        raise ValueError(
            f"kv_input_form must be 'block_tables' or 'page_indices', got "
            f"{kv_input_form!r}"
        )
    _expect_window_left(window_left)
    _expect_page_size(page_size, kv_input_form)
    _expect_max_q_len_hint(max_q_len)
    logits_soft_cap = _normalize_logits_soft_cap(logits_soft_cap)
    custom_mask, sinks = bool(custom_mask), bool(sinks)

    order = heuristic_order(cc_major, max_q_len)
    if backend != "auto":
        if backend not in CAPABILITIES:
            raise ValueError(
                f"unknown backend {backend!r}; known: {sorted(CAPABILITIES)} or 'auto'"
            )
        evaluate: Tuple[str, ...] = (backend,)
    else:
        # Evaluate EVERY known backend so explain() is complete: candidates
        # are ordered by the heuristic; everything else carries its reason.
        evaluate = tuple(order) + tuple(n for n in CAPABILITIES if n not in order)

    candidates, excluded = [], {}
    for name in evaluate:
        cap = CAPABILITIES[name]
        reason = cap.rejection_reason(
            cc_major=cc_major,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            head_dim_qk=head_dim_qk,
            head_dim_vo=head_dim_vo,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            need_lse=need_lse,
            window_left=window_left,
            kv_input_form=kv_input_form,
            logits_soft_cap=logits_soft_cap,
            use_custom_mask=custom_mask,
            use_sinks=sinks,
        )
        if reason is None:
            reason = PROBES[name](dev)
        if reason is None:
            candidates.append(name)
        else:
            excluded[name] = reason

    if not candidates:
        detail = "; ".join(f"{k}: {v}" for k, v in excluded.items())
        raise ValueError(f"no runnable backend for this configuration ({detail})")
    return Resolution(
        backends=tuple(candidates),
        excluded=excluded,
        kv_layout=kv_layout,
        config=resolve_config_key(
            num_qo_heads,
            num_kv_heads,
            head_dim_qk,
            head_dim_vo,
            q_dtype,
            kv_dtype,
            page_size,
            kv_layout,
            causal,
            need_lse,
            window_left,
            kv_input_form,
            logits_soft_cap,
            custom_mask,
            sinks,
            max_q_len,
            cc_major,
            cc_minor,
            device_index,
        ),
    )


__all__ = ["HEURISTIC_ORDER", "PROBES", "heuristic_order", "resolve_paged_attention"]
