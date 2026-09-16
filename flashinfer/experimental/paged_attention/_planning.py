"""Validation and canonical-to-derived metadata (proposal §3.2).

Mirrors ``flashinfer/mla/_batch_mla/_planning.py``: everything here is
backend-neutral. Structural checks are host-only; value checks read the host
mirrors the contract guarantees; derivation computes only the forms the chosen
backend declared it needs, as pure device ops with no sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

import torch

from ._contracts import _expect, _expect_page_size

# Names of the derived forms a backend may request (``Derived.needs``).  Each
# backend declares the set it reads (``DERIVED_NEEDS`` on its class) and the
# controller derives exactly that set: trtllm-gen consumes the canonical form
# plus cumulative KV lengths, cuDNN per-request query lengths, the generated
# FA kernels CSR page metadata.  A form the caller already supplied
# (``block_tables`` for the dense input, ``kv_page_indices`` for CSR) is
# returned as-is; the cross-form conversions are the only device work.
FORM_Q_SEQ_LENS = "q_seq_lens"
FORM_CUM_KV_SEQ_LENS = "cum_kv_seq_lens"
FORM_KV_PAGE_INDPTR = "kv_page_indptr"
FORM_KV_PAGE_INDICES = "kv_page_indices"
FORM_BLOCK_TABLES = "block_tables"
DERIVED_FORMS: FrozenSet[str] = frozenset(
    {
        FORM_Q_SEQ_LENS,
        FORM_CUM_KV_SEQ_LENS,
        FORM_KV_PAGE_INDPTR,
        FORM_KV_PAGE_INDICES,
        FORM_BLOCK_TABLES,
    }
)


def normalize_needs(needs: Iterable[str]) -> FrozenSet[str]:
    needs = frozenset(needs)
    unknown = needs - DERIVED_FORMS
    if unknown:
        raise ValueError(
            f"unknown derived form(s) {sorted(unknown)}; known: {sorted(DERIVED_FORMS)}"
        )
    return needs


@dataclass
class Derived:
    """The derived forms one plan requested (``needs``); everything else is
    ``None``.  Backends read fields through :meth:`require` so an unrequested
    form fails loudly at the use site instead of surfacing as an attribute
    error deep inside a kernel wrapper."""

    needs: FrozenSet[str]
    q_seq_lens: Optional[torch.Tensor] = None  # (b,)  device - diff(qo_indptr)
    cum_kv_seq_lens: Optional[torch.Tensor] = (
        None  # (b+1,) device - trtllm cu_seq_len_kv
    )
    kv_page_indptr: Optional[torch.Tensor] = None  # (b+1,) device CSR page-unit indptr
    kv_page_indices: Optional[torch.Tensor] = None  # flat page ids (CSR-compacted
    # prefix up to kv_page_indptr[-1]; any tail is untouched scratch never read
    # by kernels, which bound reads by the indptr)
    block_tables: Optional[torch.Tensor] = None  # (b, width) dense - given or derived

    def require(self, name: str) -> torch.Tensor:
        value = getattr(self, name)
        if value is None:
            raise AssertionError(
                f"derived form {name!r} was not requested for this plan "
                f"(needs={sorted(self.needs)}); add it to the backend's DERIVED_NEEDS"
            )
        return value


def validate_structure(
    device: torch.device,
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_q_len,
    max_kv_len,
    kv_input_form,
) -> None:
    """Closed-set structural validation. Cheap (host-only, shape-derived)."""
    checks = [
        ("qo_indptr", qo_indptr, 1),
        ("kv_seq_lens", kv_seq_lens, 1),
    ]
    if block_tables is not None:
        checks.append(("block_tables", block_tables, 2))
    if kv_page_indices is not None:
        checks.append(("kv_page_indices", kv_page_indices, 1))
    for name, t, dim in checks:
        _expect(isinstance(t, torch.Tensor), f"{name} must be a torch.Tensor")
        _expect(
            t.is_cuda and t.device == device,
            f"{name} must be on CUDA device {device}, got {t.device}",
        )
        _expect(
            t.dtype == torch.int32,
            f"{name} must be int32, got {t.dtype} — torch cumsum/arange "
            "default to int64; build with dtype=torch.int32 or .int()",
        )
        _expect(t.dim() == dim, f"{name} must be {dim}-D, got shape {tuple(t.shape)}")

    b = kv_seq_lens.shape[0]
    _expect(b >= 1, "batch size must be >= 1")
    _expect(
        qo_indptr.shape[0] == b + 1,
        f"qo_indptr must have shape (batch_size+1,) = ({b + 1},), got "
        f"{tuple(qo_indptr.shape)} — it is a token-unit prefix sum "
        "(qo_indptr[0] = 0)",
    )
    if block_tables is not None:
        _expect(
            block_tables.shape[0] == b,
            f"block_tables must have shape (batch_size, max_pages) with "
            f"batch_size={b}, got {tuple(block_tables.shape)}",
        )
    _expect_page_size(page_size, kv_input_form)
    for nm, v in (("max_q_len", max_q_len), ("max_kv_len", max_kv_len)):
        _expect(
            isinstance(v, int) and v >= 1,
            f"{nm} must be a positive host int (required; it kills the "
            f"hidden device sync), got {v!r}",
        )
    if block_tables is not None:
        capacity = block_tables.shape[1] * page_size
        _expect(
            max_kv_len <= capacity,
            f"max_kv_len ({max_kv_len}) exceeds block_tables capacity "
            f"({block_tables.shape[1]} pages x page_size {page_size} = "
            f"{capacity})",
        )


def validate_values(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_q_len,
    max_kv_len,
    qo_indptr_cpu,
    kv_seq_lens_cpu,
    causal,
) -> None:
    """Value-level validation against host mirrors. Always runs.

    These checks are what turns "silently wrong" into "loud error" for
    value corruption: an under-claimed max, an indptr that does not sum
    to the token count, or KV lens exceeding the table capacity would
    otherwise reach a kernel that trusts them as layout/scheduling truth.
    """
    _expect(
        qo_indptr_cpu.device.type == "cpu"
        and tuple(qo_indptr_cpu.shape) == tuple(qo_indptr.shape),
        "qo_indptr_cpu must be a CPU mirror with the same shape as qo_indptr",
    )
    _expect(
        kv_seq_lens_cpu.device.type == "cpu"
        and tuple(kv_seq_lens_cpu.shape) == tuple(kv_seq_lens.shape),
        "kv_seq_lens_cpu must be a CPU mirror with the same shape as kv_seq_lens",
    )
    d = qo_indptr_cpu.diff()
    if not bool((d > 0).all()):
        bad = int((d <= 0).nonzero()[0])
        raise ValueError(
            f"qo_indptr must be strictly increasing (q_len >= 1); entry "
            f"{bad}->{bad + 1} is {int(qo_indptr_cpu[bad])}->"
            f"{int(qo_indptr_cpu[bad + 1])} — zero-length requests are "
            "outside the v1 envelope; filter them before plan()"
        )
    _expect(int(qo_indptr_cpu[0]) == 0, "qo_indptr[0] must be 0")
    _expect(
        int(d.max()) <= max_q_len,
        f"max_q_len ({max_q_len}) is smaller than the actual longest "
        f"query ({int(d.max())}) — this would silently corrupt scheduling "
        "or graph shapes downstream",
    )
    if not bool((kv_seq_lens_cpu >= 1).all()):
        bad = int((kv_seq_lens_cpu < 1).nonzero()[0])
        raise ValueError(
            f"kv_seq_lens must be >= 1 (request {bad} has "
            f"{int(kv_seq_lens_cpu[bad])}) — zero-length KV rows are "
            "outside the v1 envelope; filter empty requests before plan()"
        )
    if causal:
        validate_causal_envelope(qo_indptr_cpu, kv_seq_lens_cpu)
    _expect(
        int(kv_seq_lens_cpu.max()) <= max_kv_len,
        f"max_kv_len ({max_kv_len}) is smaller than the actual longest "
        f"KV ({int(kv_seq_lens_cpu.max())})",
    )
    if block_tables is not None:
        capacity = block_tables.shape[1] * page_size
        if not bool((kv_seq_lens_cpu <= capacity).all()):
            bad = int((kv_seq_lens_cpu > capacity).nonzero()[0])
            raise ValueError(
                f"kv_seq_lens[{bad}] = {int(kv_seq_lens_cpu[bad])} exceeds "
                f"block_tables capacity ({block_tables.shape[1]} pages x "
                f"page_size {page_size} = {capacity}) — widen block_tables "
                "or fix the length"
            )
    else:
        total_pages = int(torch.sum((kv_seq_lens_cpu + page_size - 1) // page_size))
        _expect(
            kv_page_indices.shape[0] >= total_pages,
            f"kv_page_indices has {kv_page_indices.shape[0]} entries but "
            f"kv_seq_lens require {total_pages} pages at page_size "
            f"{page_size} — the flat page-id list must cover "
            "sum(ceil(kv_len/page_size)) entries in request order",
        )


def validate_causal_envelope(qo_indptr_cpu, kv_seq_lens_cpu) -> None:
    """Causal masking requires q_len_i <= kv_len_i (host mirrors, zero sync)."""
    d = qo_indptr_cpu.diff()
    if not bool((d <= kv_seq_lens_cpu).all()):
        bad = int((d > kv_seq_lens_cpu).nonzero()[0])
        raise ValueError(
            f"causal masking requires q_len_i <= kv_len_i for every "
            f"request; request {bad} has q_len {int(d[bad])} > kv_len "
            f"{int(kv_seq_lens_cpu[bad])} (fully-masked rows have "
            "backend-divergent LSE semantics and are outside the v1 "
            "envelope)"
        )


def derive(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_kv_len,
    *,
    needs: Iterable[str],
) -> Derived:
    """Canonical -> the derived forms in ``needs`` (see ``DERIVED_FORMS``).

    Pure device ops, zero sync; nothing outside ``needs`` is computed (ledger
    M7: the dense input used to pay the CSR compaction for every backend).  All
    output shapes are static functions of the input shapes and host ints:

    - dense given -> flat indices by capacity scatter (a boolean masked-select
      would sync to size its result; scatter does not);
    - flat indices given -> dense (when a candidate needs it) by a gather of
      width ceil(max_kv_len / page_size), with each row's tail CLAMPED TO THE
      REQUEST'S OWN LAST PAGE.  The per-row clamp is load-bearing: cuDNN
      gathers K/V pages by table width before masking, so a tail that
      pointed into the over-allocated (possibly uninitialized) region of
      kv_page_indices produced NaN outputs / out-of-pool reads (found
      empirically by the NaN-page probe; see the fuzzer's
      csr_overallocated_nan_tail mutation).
    """
    needs = normalize_needs(needs)
    dev = kv_seq_lens.device
    b = kv_seq_lens.shape[0]
    out = Derived(needs=needs)

    def zero_prefixed_cumsum(values):
        zero = torch.zeros(1, dtype=torch.int32, device=dev)
        return torch.cat([zero, torch.cumsum(values, 0, dtype=torch.int32)])

    if FORM_Q_SEQ_LENS in needs:
        out.q_seq_lens = qo_indptr.diff()
    if FORM_CUM_KV_SEQ_LENS in needs:
        out.cum_kv_seq_lens = zero_prefixed_cumsum(kv_seq_lens)

    # the page-unit indptr feeds both cross-form conversions
    compact_flat = FORM_KV_PAGE_INDICES in needs and block_tables is not None
    gather_dense = FORM_BLOCK_TABLES in needs and block_tables is None
    pages = kv_page_indptr = None
    if FORM_KV_PAGE_INDPTR in needs or compact_flat or gather_dense:
        pages = (kv_seq_lens + page_size - 1) // page_size  # (b,)
        kv_page_indptr = zero_prefixed_cumsum(pages)
    if FORM_KV_PAGE_INDPTR in needs:
        out.kv_page_indptr = kv_page_indptr

    if FORM_KV_PAGE_INDICES in needs:
        if block_tables is None:
            out.kv_page_indices = kv_page_indices
        else:
            width = block_tables.shape[1]
            capacity = b * width
            col = torch.arange(width, device=dev, dtype=torch.int64)
            valid = col.unsqueeze(0) < pages.unsqueeze(1)  # (b, width)
            # compact destination of each (row, col) lane; invalid lanes all
            # target a dummy tail slot (duplicate writes there are benign)
            dst = kv_page_indptr[:-1].to(torch.int64).unsqueeze(1) + col.unsqueeze(0)
            dst = torch.where(valid, dst, capacity)
            buf = torch.empty(capacity + 1, dtype=torch.int32, device=dev)
            buf.scatter_(0, dst.reshape(-1), block_tables.reshape(-1))
            out.kv_page_indices = buf[:capacity]

    if FORM_BLOCK_TABLES in needs:
        if block_tables is not None:
            out.block_tables = block_tables
        else:
            width = (max_kv_len + page_size - 1) // page_size  # host int, no sync
            col = torch.arange(width, device=dev, dtype=torch.int64)
            src = kv_page_indptr[:-1].to(torch.int64).unsqueeze(1) + col.unsqueeze(0)
            # clamp each row's tail to the request's OWN last live page (kv_len
            # >= 1 is validated, so every row owns at least one).  Load-bearing:
            # cuDNN gathers K/V pages by table width before masking, so a tail
            # pointing into a neighbouring request or the over-allocated
            # (possibly uninitialized) region of kv_page_indices produced NaN
            # outputs / out-of-pool reads (fuzzer: csr_overallocated_nan_tail).
            row_last = kv_page_indptr[1:].to(torch.int64).unsqueeze(1) - 1
            src = torch.minimum(src, row_last)
            out.block_tables = (
                kv_page_indices.to(torch.int64)
                .gather(0, src.reshape(-1))
                .reshape(b, width)
                .to(torch.int32)
            )
    return out


__all__ = [
    "DERIVED_FORMS",
    "FORM_BLOCK_TABLES",
    "FORM_CUM_KV_SEQ_LENS",
    "FORM_KV_PAGE_INDICES",
    "FORM_KV_PAGE_INDPTR",
    "FORM_Q_SEQ_LENS",
    "Derived",
    "derive",
    "normalize_needs",
    "validate_causal_envelope",
    "validate_structure",
    "validate_values",
]
