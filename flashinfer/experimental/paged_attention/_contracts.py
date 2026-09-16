"""Contract values shared by the controller, selection, and backends.

Mirrors ``flashinfer/mla/_batch_mla/_contracts.py``: the immutable values that
fix what a plan means (``PlanMetadata``), what a resolution promised
(``Resolution``), and the loud-error helpers every layer uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

import torch

from ._backends._capabilities import MIN_DENSE_PAGE_SIZE

LSE_MODES = ("none", "base2", "basee")
LN2 = math.log(2.0)


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _expect_lse_mode(lse_mode: str) -> None:
    _expect(
        lse_mode in LSE_MODES,
        f"lse_mode must be one of {LSE_MODES} (MLA vocabulary: none / base-2 / "
        f"natural log), got {lse_mode!r}",
    )


def _expect_window_left(window_left: int) -> None:
    # trtllm's launcher special-cases exactly -1 (unlimited); other negatives
    # become a force-enabled negative window there while fa/cudnn read them
    # as unlimited — backend-divergent, so reject anything below -1 loudly.
    _expect(
        isinstance(window_left, int) and window_left >= -1,
        f"window_left must be >= -1 (-1 = unlimited), got {window_left!r}",
    )


def _normalize_logits_soft_cap(logits_soft_cap: Optional[float]) -> Optional[float]:
    """``None`` / ``0`` mean "no soft cap" (the legacy convention); anything
    else must be a positive finite host float and is returned as ``float``."""
    if logits_soft_cap is None:
        return None
    _expect(
        isinstance(logits_soft_cap, (int, float))
        and not isinstance(logits_soft_cap, bool)
        and math.isfinite(logits_soft_cap)
        and logits_soft_cap >= 0,
        "logits_soft_cap must be None (off) or a non-negative finite host float "
        f"(cap * tanh(score / cap)), got {logits_soft_cap!r}",
    )
    return float(logits_soft_cap) if logits_soft_cap > 0 else None


def _expect_custom_mask(
    custom_mask: torch.Tensor, device: torch.device, numel: int
) -> None:
    """The flattened per-request boolean mask contract (legacy mask layout)."""
    _expect(
        isinstance(custom_mask, torch.Tensor),
        f"custom_mask must be a torch.Tensor, got {type(custom_mask).__name__}",
    )
    _expect(
        custom_mask.dtype == torch.bool,
        f"custom_mask must be a bool tensor (True = may attend), got {custom_mask.dtype}",
    )
    _expect(
        custom_mask.device == device,
        f"custom_mask lives on {custom_mask.device} but this instance is bound to {device}",
    )
    _expect(
        custom_mask.dim() == 1 and custom_mask.is_contiguous(),
        "custom_mask must be a contiguous 1-D tensor: the per-request "
        "(q_len_i, kv_len_i) masks flattened row-major and concatenated in "
        f"request order, got shape {tuple(custom_mask.shape)}",
    )
    _expect(
        custom_mask.numel() == numel,
        f"custom_mask has {custom_mask.numel()} elements but the batch needs "
        f"sum(q_len_i * kv_len_i) = {numel} (flattened per-request masks in "
        "request order)",
    )


def _expect_page_size(page_size: int, kv_input_form: str) -> None:
    _expect(
        isinstance(page_size, int) and page_size >= 1,
        f"page_size must be a positive host int, got {page_size!r}",
    )
    if kv_input_form == "block_tables":
        _expect(
            page_size >= MIN_DENSE_PAGE_SIZE,
            f"page_size {page_size} < {MIN_DENSE_PAGE_SIZE} with the dense "
            "block_tables form: a dense table at token-granular page sizes "
            "degenerates to (batch, max_context) — pass the flat "
            "kv_page_indices form instead (any page_size >= 1)",
        )


def resolve_config_key(
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
    cc_major,
    cc_minor,
    device_index,
) -> Tuple:
    """The observational config a Resolution is pinned to (drift detection).

    ``logits_soft_cap`` (normalized value or None), ``custom_mask`` and
    ``sinks`` (booleans) are the feature axes: pinning covers them, so a
    plan() that requests a feature the Resolution was not resolved for is a
    drift error rather than a silent capability change.

    The last element is the device binding ``(cc_major, cc_minor,
    device_index)``: a Resolution resolved on one device must not be handed
    to a controller bound to another device or compute capability, because
    the probes (fa3's SM90a check, cubin availability) answered for the
    device they were given.  ``cc_minor``/``device_index`` are ``None`` when
    the caller resolved from an explicit ``cc_major`` without a device; such a
    Resolution is pinned to the compute-capability major only.
    """
    return (
        num_qo_heads,
        num_kv_heads,
        head_dim_qk,
        head_dim_vo,
        str(q_dtype),
        str(kv_dtype),
        page_size,
        kv_layout,
        causal,
        need_lse,
        window_left,
        kv_input_form,
        logits_soft_cap,
        bool(custom_mask),
        bool(sinks),
        (cc_major, cc_minor, device_index),
    )


def _expect_pinned_device(resolved: Tuple, want: Tuple) -> None:
    """Reject a Resolution whose device binding does not cover ``want``.

    ``resolved`` / ``want`` are the trailing device triples of two config
    keys.  A ``None`` in the resolved triple means "not pinned on this axis"
    (explicit-``cc_major`` resolution) and matches anything.
    """
    r_major, r_minor, r_index = resolved
    w_major, w_minor, w_index = want
    _expect(
        r_major == w_major and (r_minor is None or r_minor == w_minor),
        "the pinned Resolution was resolved for compute capability "
        f"sm_{r_major}{'x' if r_minor is None else r_minor} but this instance runs on "
        f"sm_{w_major}{w_minor} — the probes answered for another GPU; re-run "
        "resolve_paged_attention(device=...) on the target device",
    )
    _expect(
        r_index is None or r_index == w_index,
        f"the pinned Resolution was resolved on cuda:{r_index} but this instance "
        f"is bound to cuda:{w_index} — resolve once per device (or resolve from "
        "an explicit cc_major to pin the compute capability only)",
    )


@dataclass(frozen=True)
class Resolution:
    """Init-time resolution result (proposal §5.3, level 1).

    ``backends`` is the pinned, ordered candidate set: every member passed
    the static capability check and the environment probes for the declared
    configuration on the resolved device, and is observationally identical
    at the contract level (same dtypes, same LSE availability), so a later
    plan-time choice within this set cannot surprise the engine.  Membership
    is support *evidence*, not a guarantee: a candidate may still decline a
    specific batch at plan time with a typed unsupported signal, in which case
    plan() continues to the next member (see ``PagedAttentionController``).
    Pass the whole Resolution to ``plan(backend=...)`` to enforce the
    pinning: plan() verifies its arguments and its device match ``config``
    and chooses only within ``backends``.
    """

    backends: Tuple[str, ...]
    excluded: Dict[str, str] = field(default_factory=dict)
    kv_layout: str = "HND"
    # the resolve-time observational config, used by plan() to detect drift;
    # its last element is the device binding (cc_major, cc_minor, device_index)
    config: Tuple = ()

    @property
    def chosen(self) -> str:
        return self.backends[0]

    @property
    def device_binding(self) -> Tuple:
        """``(cc_major, cc_minor, device_index)`` this Resolution answers for."""
        return self.config[-1] if self.config else (None, None, None)

    def explain(self) -> str:
        cc_major, cc_minor, index = self.device_binding
        if cc_major is not None:
            where = (
                f"sm_{cc_major}{cc_minor}, cuda:{index}"
                if cc_minor is not None
                else f"sm_{cc_major}x (compute-capability major only; not "
                "pinned to a device)"
            )
            lines = [f"resolved for: {where}"]
        else:
            lines = []
        lines.append(f"candidates (preference order): {list(self.backends)}")
        for name, reason in self.excluded.items():
            lines.append(f"excluded {name}: {reason}")
        return "\n".join(lines)


@dataclass(frozen=True, eq=False)
class PagedAttentionMetadata:
    """The canonical per-batch metadata for paged attention (one object).

    Build it once per scheduler step with :meth:`dense` (vLLM-style block
    table) or :meth:`csr` (sglang-style flat page ids); the two constructors
    make "exactly one paging form" structural.  Construction validates shapes
    and values against the host mirrors — pass the mirrors the engine already
    owns and construction is zero-sync; otherwise it performs ONE documented
    D2H here, and every later ``plan()`` on this object is sync-free.  The
    derived forms are computed from the mirrors on the host and reach the
    device through one pinned upload (see ``_planning.derive``).

    The object also owns the backend-neutral derived forms (CSR page indptr,
    cumulative KV lengths, dense table) lazily and per need set, so several
    plans over the same batch (windowed / full layers, causal / non-causal)
    derive once and only the forms the chosen backend reads are computed.

    Padding rows: ``kv_seq_lens[i] == 0`` is legal and marks a padding row
    (vLLM's CUDA-graph padding fills ``seq_lens`` with 0 and the table row
    with its null block; sglang's fill value 1 is an ordinary live row).  The
    query length of a padding row must still be >= 1.  The library guarantees
    no page of that row is read, a finite output row, and unchanged results
    for every other row; the row's output values and LSE are unspecified.
    Every backend handles it natively (measured on B200: trtllm-gen
    ``seq_lens=0``, cuDNN ``actual_seq_lens_kv=0``, the FA kernels a CSR row
    with zero pages all give a zero output row and LSE -inf).

    Identity semantics: two objects compare by identity, not by tensor
    contents (they are meant to be built once per step and reused).
    """

    qo_indptr: torch.Tensor  # (b+1,) int32 device, token-unit prefix sums
    kv_seq_lens: torch.Tensor  # (b,) int32 device, per-request valid KV lengths
    page_size: int
    max_q_len: int
    max_kv_len: int
    block_tables: Optional[torch.Tensor] = None  # (b, max_pages) dense form
    kv_page_indices: Optional[torch.Tensor] = None  # flat CSR page ids
    qo_indptr_cpu: Optional[torch.Tensor] = None
    kv_seq_lens_cpu: Optional[torch.Tensor] = None
    # the validated host arrays (``_planning.HostArrays``): one pinned staging
    # tensor computed at construction, uploaded once on the first plan that
    # needs a device form; lifetime = this object
    _host: Any = field(default=None, repr=False)
    _derived: Dict[Tuple[FrozenSet[str], int], Any] = field(
        default_factory=dict, repr=False
    )

    # ---- constructors ----
    @classmethod
    def dense(
        cls,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        *,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> "PagedAttentionMetadata":
        """vLLM-style: a dense ``(batch, max_pages_per_seq)`` block table (page_size >= 8)."""
        return cls(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            block_tables=block_tables,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )

    @classmethod
    def csr(
        cls,
        qo_indptr: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_page_indices: torch.Tensor,
        *,
        page_size: int,
        max_q_len: int,
        max_kv_len: int,
        qo_indptr_cpu: Optional[torch.Tensor] = None,
        kv_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> "PagedAttentionMetadata":
        """sglang-style: flat CSR page ids in request order (any page_size >= 1).

        Page-unit indptr and last-page lengths are NOT accepted: they derive
        from ``kv_seq_lens`` + ``page_size``; a second copy would be a second
        truth.
        """
        return cls(
            qo_indptr=qo_indptr,
            kv_seq_lens=kv_seq_lens,
            page_size=page_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            kv_page_indices=kv_page_indices,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_seq_lens_cpu,
        )

    def __post_init__(self):
        from ._planning import validate_structure, validate_values

        _expect(
            (self.block_tables is None) != (self.kv_page_indices is None),
            "pass EXACTLY ONE paging form: PagedAttentionMetadata.dense(block_tables) "
            "or .csr(kv_page_indices)",
        )
        _expect(
            isinstance(self.kv_seq_lens, torch.Tensor) and self.kv_seq_lens.is_cuda,
            "kv_seq_lens must be a CUDA tensor",
        )
        validate_structure(
            self.kv_seq_lens.device,
            self.qo_indptr,
            self.kv_seq_lens,
            self.block_tables,
            self.kv_page_indices,
            self.page_size,
            self.max_q_len,
            self.max_kv_len,
            self.kv_input_form,
        )
        # Value-level validation is unconditional — it is what makes the
        # reject-or-correct property hold.  Zero-sync iff the caller hands us
        # the host mirrors it already owns; otherwise ONE documented D2H here
        # (both arrays packed into one transfer when both are missing).
        if self.qo_indptr_cpu is None and self.kv_seq_lens_cpu is None:
            b = self.kv_seq_lens.shape[0]
            packed = torch.cat([self.qo_indptr, self.kv_seq_lens]).cpu()
            object.__setattr__(self, "qo_indptr_cpu", packed[: b + 1])
            object.__setattr__(self, "kv_seq_lens_cpu", packed[b + 1 :])
        elif self.qo_indptr_cpu is None:
            object.__setattr__(self, "qo_indptr_cpu", self.qo_indptr.cpu())
        elif self.kv_seq_lens_cpu is None:
            object.__setattr__(self, "kv_seq_lens_cpu", self.kv_seq_lens.cpu())
        # the causal envelope depends on plan(causal=...): see
        # validate_causal_envelope()
        host = validate_values(
            self.qo_indptr,
            self.kv_seq_lens,
            self.block_tables,
            self.kv_page_indices,
            self.page_size,
            self.max_q_len,
            self.max_kv_len,
            self.qo_indptr_cpu,
            self.kv_seq_lens_cpu,
        )
        object.__setattr__(self, "_host", host)

    # ---- derived facts ----
    @property
    def kv_input_form(self) -> str:
        return "block_tables" if self.block_tables is not None else "page_indices"

    @property
    def device(self) -> torch.device:
        return self.kv_seq_lens.device

    @property
    def batch_size(self) -> int:
        return int(self.kv_seq_lens.shape[0])

    @property
    def total_q_tokens(self) -> int:
        return int(self._host.numpy("qo_indptr")[-1])

    def validate_causal_envelope(self) -> None:
        """``q_len_i <= kv_len_i`` for every request (host arrays, zero sync)."""
        from ._planning import validate_causal_envelope

        validate_causal_envelope(self._host)

    def derived(self, *, needs, max_kv_len: Optional[int] = None):
        """The derived forms in ``needs`` (``_planning.DERIVED_FORMS`` names),
        computed once per (object, need set, width) and cached: several plans
        over the same batch that pick the same backend derive once, and a
        backend never pays for a form it does not read.

        ``max_kv_len`` widens a dense table derived from flat page ids to
        ``ceil(max_kv_len / page_size)`` columns (CUDA-graph mode derives at
        the capacity so the table fits its reserved storage); ``None`` uses
        the batch's own max, and a value below it is rejected.
        """
        from ._planning import derive, normalize_needs

        width_max = self.max_kv_len if max_kv_len is None else max_kv_len
        _expect(
            width_max >= self.max_kv_len,
            f"derived(max_kv_len={max_kv_len}) is narrower than the batch's "
            f"max_kv_len {self.max_kv_len}",
        )
        key = (normalize_needs(needs), width_max)
        d = self._derived.get(key)
        if d is None:
            d = derive(
                self.block_tables,
                self.kv_page_indices,
                width_max,
                needs=key[0],
                host=self._host,
                device=self.device,
            )
            self._derived[key] = d
        return d


@dataclass(frozen=True)
class PlanMetadata:
    """Everything a backend may read about one planned batch.

    Built by the controller after validation; backends receive it together
    with the derived forms (``_planning.Derived``) and must not reach past it
    into caller state. ``block_tables`` is the dense table — the caller's own
    (or its copy in reserved graph storage) in the dense input form, derived
    in the flat form — and is ``None`` only in the flat form when the chosen
    backend does not read a dense table. In CUDA-graph mode ``max_q_len`` /
    ``max_kv_len`` are the capacity's values, not the batch's own.
    """

    qo_indptr: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_tables: Optional[torch.Tensor]
    kv_input_form: str
    page_size: int
    max_q_len: int
    max_kv_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    causal: bool
    window_left: int
    kv_layout: str
    lse_mode: str  # "none" | "base2" | "basee"
    batch_size: int
    qo_indptr_cpu: torch.Tensor
    kv_seq_lens_cpu: torch.Tensor
    # ---- feature axes (declared at plan time; capability-checked) ----
    # softmax logits soft cap: cap * tanh(score / cap) on the scaled scores
    # (None = off; normalized by the controller)
    logits_soft_cap: Optional[float] = None
    # per-request flattened boolean mask, concatenated in request order
    # (sum(q_len_i * kv_len_i),), True = may attend; ANDed into the
    # causal/window envelope
    custom_mask: Optional[torch.Tensor] = None
    # attention sinks will be passed to run(): the backend must plan the
    # sink-aware kernel variant
    use_sinks: bool = False

    @property
    def total_q_tokens(self) -> int:
        return int(self.qo_indptr_cpu[-1])

    @property
    def need_lse(self) -> bool:
        return self.lse_mode != "none"

    @property
    def mask_numel(self) -> int:
        """Length of the flattened custom mask this batch requires (host)."""
        q_lens = self.qo_indptr_cpu.diff().to(torch.int64)
        return int((q_lens * self.kv_seq_lens_cpu.to(torch.int64)).sum())


__all__ = [
    "LN2",
    "LSE_MODES",
    "PagedAttentionMetadata",
    "PlanMetadata",
    "Resolution",
    "_expect_custom_mask",
    "_expect_pinned_device",
    "_normalize_logits_soft_cap",
    "resolve_config_key",
]
