"""Validation and canonical-to-derived metadata (proposal §3.2).

Mirrors ``flashinfer/mla/_batch_mla/_planning.py``: everything here is
backend-neutral. Structural checks are host-only; value checks read the host
mirrors the contract guarantees; derivation computes only the forms the chosen
backend declared it needs, on the host from the mirrors, and reaches the device
through one pinned upload (no sync).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

import numpy as np
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
# pinned int32 host arrays for a CPU-side scheduler (the generated FA planner)
FORM_HOST_ARRAYS = "host_arrays"
DERIVED_FORMS: FrozenSet[str] = frozenset(
    {
        FORM_Q_SEQ_LENS,
        FORM_CUM_KV_SEQ_LENS,
        FORM_KV_PAGE_INDPTR,
        FORM_KV_PAGE_INDICES,
        FORM_BLOCK_TABLES,
        FORM_HOST_ARRAYS,
    }
)
_DEVICE_FORMS = frozenset({FORM_Q_SEQ_LENS, FORM_CUM_KV_SEQ_LENS, FORM_KV_PAGE_INDPTR})


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
    # FORM_HOST_ARRAYS: pinned int32 CPU views into one staging tensor (the
    # same tensor the device forms above were uploaded from), so a wrapper's
    # non_blocking upload of them is a real asynchronous copy
    qo_indptr_host: Optional[torch.Tensor] = None  # (b+1,)
    kv_seq_lens_host: Optional[torch.Tensor] = None  # (b,)
    kv_page_indptr_host: Optional[torch.Tensor] = None  # (b+1,)
    kv_last_page_len_host: Optional[torch.Tensor] = None  # (b,)

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


class _HostStaging:
    """All derived host arrays of one batch in ONE pinned int32 tensor.

    Layout (b = batch size): qo_indptr (b+1) | q_seq_lens (b) |
    cum_kv_seq_lens (b+1) | kv_page_indptr (b+1) | pages (b) |
    kv_last_page_len (b) | kv_seq_lens (b).  Filled with numpy on the mirrors
    (small-array torch CPU ops cost 2-5 us each; numpy about 1.5 us), then
    uploaded with one ``copy_(non_blocking=True)`` when a device form is
    requested; the device forms are slices of that one device buffer.

    Lifetime: the staging tensor lives as long as the ``Derived`` that
    references it, i.e. per metadata object (v1).  PyTorch's caching host
    allocator makes that safe and cheap: a pinned block freed while an
    asynchronous copy from it is in flight is only reused after the copy's
    stream event completes, and same-size re-allocations are served from the
    cache (about 1 us).  Follow-up: pool one staging slot pair per
    PagedAttention instance so the graph-mode update path can upload straight
    into its reserved storage.
    """

    def __init__(
        self, qo_indptr_cpu: torch.Tensor, kv_seq_lens_cpu: torch.Tensor, page_size: int
    ):
        b = kv_seq_lens_cpu.shape[0]
        self.b = b
        o_qo, o_q, o_ck, o_pi, o_pg, o_ll, o_kv, total = (
            0,
            b + 1,
            2 * b + 1,
            3 * b + 2,
            4 * b + 3,
            5 * b + 3,
            6 * b + 3,
            7 * b + 3,
        )
        self.slices = dict(
            qo_indptr=slice(o_qo, o_q),
            q_seq_lens=slice(o_q, o_ck),
            cum_kv_seq_lens=slice(o_ck, o_pi),
            kv_page_indptr=slice(o_pi, o_pg),
            pages=slice(o_pg, o_ll),
            kv_last_page_len=slice(o_ll, o_kv),
            kv_seq_lens=slice(o_kv, total),
        )
        self.host = torch.empty(total, dtype=torch.int32, pin_memory=True)
        st = self.host.numpy()
        qo = qo_indptr_cpu.numpy()
        kv = kv_seq_lens_cpu.numpy()
        st[o_qo:o_q] = qo
        np.subtract(qo[1:], qo[:-1], out=st[o_q:o_ck])
        st[o_ck] = 0
        np.cumsum(kv, out=st[o_ck + 1 : o_pi])
        pages = st[o_pg:o_ll]
        np.floor_divide(kv + (page_size - 1), page_size, out=pages)
        st[o_pi] = 0
        np.cumsum(pages, out=st[o_pi + 1 : o_pg])
        # last page occupancy; a kv_len of 0 (no pages) yields page_size,
        # which the FA kernels never read (get_length() returns 0 first)
        np.subtract(kv, (pages - 1) * page_size, out=st[o_ll:o_kv])
        st[o_kv:total] = kv
        self.device: Optional[torch.Tensor] = None

    def upload(self, device: torch.device) -> None:
        """ONE non_blocking H2D of the whole staging into a device buffer."""
        if self.device is None:
            self.device = torch.empty(
                self.host.shape[0], dtype=torch.int32, device=device
            )
            self.device.copy_(self.host, non_blocking=True)

    def on_device(self, name: str) -> torch.Tensor:
        assert self.device is not None
        return self.device[self.slices[name]]

    def on_host(self, name: str) -> torch.Tensor:
        return self.host[self.slices[name]]


def derive(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    kv_page_indices,
    page_size,
    max_kv_len,
    *,
    needs: Iterable[str],
    qo_indptr_cpu: torch.Tensor,
    kv_seq_lens_cpu: torch.Tensor,
) -> Derived:
    """Canonical -> the derived forms in ``needs`` (see ``DERIVED_FORMS``).

    Zero sync; nothing outside ``needs`` is computed (ledger M7: the dense
    input used to pay the CSR compaction for every backend).  The length
    arithmetic runs on the host mirrors (``_HostStaging``) and reaches the
    device through one pinned upload, so a plan that needs only cumulative KV
    lengths or query lengths costs one memcpy and no kernel (ledger M5).  The
    only device work is the two cross-form conversions; their output shapes
    are static functions of the input shapes and host ints:

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

    The host mirrors are trusted to match the device tensors (documented
    caller contract, same as validation).
    """
    needs = normalize_needs(needs)
    dev = kv_seq_lens.device
    b = kv_seq_lens.shape[0]
    out = Derived(needs=needs)
    staging = _HostStaging(qo_indptr_cpu, kv_seq_lens_cpu, page_size)

    # the page-unit indptr feeds both cross-form conversions
    compact_flat = FORM_KV_PAGE_INDICES in needs and block_tables is not None
    gather_dense = FORM_BLOCK_TABLES in needs and block_tables is None
    if (needs & _DEVICE_FORMS) or compact_flat or gather_dense:
        staging.upload(dev)
    if FORM_Q_SEQ_LENS in needs:
        out.q_seq_lens = staging.on_device("q_seq_lens")
    if FORM_CUM_KV_SEQ_LENS in needs:
        out.cum_kv_seq_lens = staging.on_device("cum_kv_seq_lens")
    if FORM_KV_PAGE_INDPTR in needs:
        out.kv_page_indptr = staging.on_device("kv_page_indptr")
    if FORM_HOST_ARRAYS in needs:
        out.qo_indptr_host = staging.on_host("qo_indptr")
        out.kv_seq_lens_host = staging.on_host("kv_seq_lens")
        out.kv_page_indptr_host = staging.on_host("kv_page_indptr")
        out.kv_last_page_len_host = staging.on_host("kv_last_page_len")

    if FORM_KV_PAGE_INDICES in needs:
        if block_tables is None:
            out.kv_page_indices = kv_page_indices
        else:
            kv_page_indptr = staging.on_device("kv_page_indptr")
            pages = staging.on_device("pages")
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
            kv_page_indptr = staging.on_device("kv_page_indptr")
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
    "FORM_HOST_ARRAYS",
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
