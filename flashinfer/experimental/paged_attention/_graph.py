"""CUDA-graph re-plan protocol: reserved metadata storage + transactional staging.

Mirrors the reserved-buffer protocol of ``BatchPrefillWithPagedKVCacheWrapper``
(``use_cuda_graph=True``) and the transactional staging of the Batch MLA
backends (``flashinfer/mla/_batch_mla/_backends/_fa_common.py``).

A captured graph bakes in device pointers. In graph mode the controller
therefore never hands a backend the caller's tensors or a fresh derivation:
it owns one set of reserved buffers sized by a ``GraphCapacity`` — given
explicitly at construction or inferred from the FIRST plan (the capture
shapes) — and every later ``plan()`` copies the new batch into them. Shapes a
captured kernel depends on — batch size, table width, host maxes, total query
tokens — are checked against the capacity before anything is written, and a
plan that fails midway restores every buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from ._backends._capabilities import MIN_DENSE_PAGE_SIZE
from ._contracts import PagedAttentionMetadata, _expect, _expect_page_size
from ._planning import Derived


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _expect_positive_int(name: str, value) -> None:
    _expect(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1,
        f"GraphCapacity.{name} must be a positive host int, got {value!r}",
    )


@dataclass(frozen=True)
class GraphCapacity:
    """The capture shapes of one CUDA-graph bucket.

    A captured graph bakes in the reserved buffers' pointers, its launch grids
    and the host integers its kernels were planned with.  ``GraphCapacity``
    names those shapes so an engine can size a bucket at construction::

        cap = GraphCapacity(batch_size=256, total_q_tokens=1024, max_q_len=4,
                            max_kv_len=131072, page_size=16, table_width=8192)
        attn = PagedAttention(device, graph_capacity=cap)

    That is the dense ``block_tables`` form; the flat ``kv_page_indices`` form
    passes ``kv_input_form="page_indices"`` and ``flat_capacity=`` instead of
    ``table_width=``.  ``PagedAttention(use_cuda_graph=True)`` infers the
    capacity from the first plan instead (:meth:`from_metadata`).

    Rules every later batch (``plan()`` / ``update()``) is checked against:

    - ``batch_size``, ``kv_input_form`` and ``page_size`` hold exactly (each
      backend's grid and paging dialect follow from them).
    - ``total_q_tokens``, ``max_q_len`` and ``max_kv_len`` are upper bounds.
      The kernels are planned with the capacity values and keep reading them
      after capture (capacity substitution: in graph mode a backend sees the
      capacity maxes, never a batch's own), so a batch may use less.  ``q``,
      ``out`` and ``lse`` may carry up to ``total_q_tokens`` rows; rows past
      the batch are neither read nor written.
    - dense form: ``table_width`` is the caller's block-table width and must
      equal ``ceil(max_kv_len / page_size)`` — cuDNN plans its paged gather
      from both and rejects any other pairing — so pass
      ``max_kv_len = table_width * page_size`` when the table is wider than
      the longest context.  ``flat_capacity`` is ``batch_size * table_width``.
    - flat form: ``flat_capacity`` bounds the length of ``kv_page_indices``.
      A dense table is reserved at ``ceil(max_kv_len / page_size)`` only once
      a backend that needs one is chosen — never below
      ``MIN_DENSE_PAGE_SIZE``, where it would be ``(batch, max_context)``.
    """

    batch_size: int
    total_q_tokens: int
    max_q_len: int
    max_kv_len: int
    page_size: int
    kv_input_form: str = "block_tables"
    table_width: Optional[int] = None  # dense form: the block table width
    flat_capacity: Optional[int] = None  # flat form: kv_page_indices length

    def __post_init__(self):
        for name in ("batch_size", "total_q_tokens", "max_q_len", "max_kv_len"):
            _expect_positive_int(name, getattr(self, name))
        _expect(
            self.kv_input_form in ("block_tables", "page_indices"),
            "GraphCapacity.kv_input_form must be 'block_tables' or "
            f"'page_indices', got {self.kv_input_form!r}",
        )
        _expect_page_size(self.page_size, self.kv_input_form)
        _expect(
            self.total_q_tokens >= self.batch_size,
            f"GraphCapacity.total_q_tokens ({self.total_q_tokens}) must cover at "
            f"least one query token per request (batch_size {self.batch_size})",
        )
        _expect(
            self.max_q_len <= self.total_q_tokens,
            f"GraphCapacity.max_q_len ({self.max_q_len}) exceeds total_q_tokens "
            f"({self.total_q_tokens})",
        )
        if self.kv_input_form == "block_tables":
            _expect(
                self.table_width is not None,
                "GraphCapacity: the dense block_tables form requires table_width "
                "(the block table's width in pages)",
            )
            _expect_positive_int("table_width", self.table_width)
            _expect(
                self.table_width * self.page_size >= self.max_kv_len,
                f"GraphCapacity: table_width {self.table_width} x page_size "
                f"{self.page_size} = {self.table_width * self.page_size} is too "
                f"narrow for max_kv_len {self.max_kv_len}",
            )
            pages = _ceil_div(self.max_kv_len, self.page_size)
            _expect(
                self.table_width == pages,
                f"GraphCapacity: table_width {self.table_width} must equal "
                f"ceil(max_kv_len / page_size) = {pages}: cuDNN plans its paged "
                "gather from both and rejects any other pairing "
                "(CUDNN_STATUS_BAD_PARAM); pass max_kv_len = table_width * "
                f"page_size = {self.table_width * self.page_size}",
            )
            flat = self.batch_size * self.table_width
            if self.flat_capacity is None:
                object.__setattr__(self, "flat_capacity", flat)
            else:
                _expect(
                    self.flat_capacity == flat,
                    f"GraphCapacity: in the dense form flat_capacity is derived as "
                    f"batch_size x table_width = {flat}; leave it unset",
                )
        else:
            _expect(
                self.table_width is None,
                "GraphCapacity: the flat page_indices form derives its dense table "
                "at ceil(max_kv_len / page_size) when a backend needs one; do not "
                "pass table_width",
            )
            _expect(
                self.flat_capacity is not None,
                "GraphCapacity: the flat page_indices form requires flat_capacity "
                "(the reserved length of kv_page_indices)",
            )
            _expect_positive_int("flat_capacity", self.flat_capacity)
            need = max(self.batch_size, self.dense_table_width)
            _expect(
                self.flat_capacity >= need,
                f"GraphCapacity.flat_capacity ({self.flat_capacity}) cannot hold "
                f"one page per request and one request of max_kv_len (needs >= {need})",
            )

    @property
    def dense_table_width(self) -> int:
        """Width of the dense table a backend that needs one reads."""
        if self.table_width is not None:
            return self.table_width
        return _ceil_div(self.max_kv_len, self.page_size)

    @classmethod
    def from_metadata(cls, metadata: PagedAttentionMetadata) -> "GraphCapacity":
        """The capacity ``use_cuda_graph=True`` infers from the first batch.

        Dense form: ``max_kv_len`` is widened to ``table_width * page_size``,
        so the table's whole width stays usable by later batches and the
        width rule above holds whatever the first batch's longest context.
        """
        common = dict(
            batch_size=metadata.batch_size,
            total_q_tokens=metadata.total_q_tokens,
            max_q_len=metadata.max_q_len,
            page_size=metadata.page_size,
            kv_input_form=metadata.kv_input_form,
        )
        if metadata.block_tables is not None:
            width = int(metadata.block_tables.shape[1])
            return cls(
                **common, max_kv_len=width * metadata.page_size, table_width=width
            )
        return cls(
            **common,
            max_kv_len=metadata.max_kv_len,
            flat_capacity=int(metadata.kv_page_indices.shape[0]),
        )


class GraphBuffers:
    """Reserved device storage for one PagedAttention instance in graph mode."""

    def __init__(self, capacity: GraphCapacity, device: torch.device):
        self.capacity = capacity
        self._device = device
        b = capacity.batch_size
        i32 = dict(dtype=torch.int32, device=device)
        # the caller-facing canonical metadata, mirrored into stable storage
        self.qo_indptr = torch.zeros(b + 1, **i32)
        self.kv_seq_lens = torch.zeros(b, **i32)
        # flat page ids: given (flat form) or derived from the dense table
        self.kv_page_indices = torch.zeros(capacity.flat_capacity, **i32)
        # backend-neutral derived forms
        self.q_seq_lens = torch.zeros(b, **i32)
        self.cum_kv_seq_lens = torch.zeros(b + 1, **i32)
        self.kv_page_indptr = torch.zeros(b + 1, **i32)
        # The dense table.  Dense form: the caller's own, mirrored.  Flat form:
        # reserved by reserve_dense_table() only once a backend that needs it
        # is chosen — at token-granular page sizes it would be (b, max_context)
        # (128 MiB per bucket for sglang's page_size=1 at 128K context).
        self.block_tables: Optional[torch.Tensor] = (
            torch.zeros(b, capacity.table_width, **i32)
            if capacity.kv_input_form == "block_tables"
            else None
        )

    def reserve_dense_table(self) -> torch.Tensor:
        """Flat form: reserve the derived dense table for a backend that needs it."""
        if self.block_tables is None:
            cap = self.capacity
            _expect(
                cap.page_size >= MIN_DENSE_PAGE_SIZE,
                f"a dense block table cannot be reserved at page_size {cap.page_size} "
                f"< {MIN_DENSE_PAGE_SIZE}: it would be (batch, max_context); backends "
                "that need one are capability-excluded there",
            )
            self.block_tables = torch.zeros(
                cap.batch_size,
                cap.dense_table_width,
                dtype=torch.int32,
                device=self._device,
            )
        return self.block_tables

    # ---- checks ----
    def preflight(self, metadata: PagedAttentionMetadata) -> None:
        """Reject before writing anything a captured kernel would misread.

        Exact: batch size, paging form, page size, dense table width (the
        table is mirrored whole).  Bounded: total query tokens, host maxes,
        flat page-id length — the kernels were planned with the capacity
        values (see ``GraphCapacity``), so a batch may use less.
        """
        cap = self.capacity
        exact = (
            ("batch_size", metadata.batch_size, cap.batch_size),
            ("kv_input_form", metadata.kv_input_form, cap.kv_input_form),
            ("page_size", metadata.page_size, cap.page_size),
        )
        for name, got, want in exact:
            _expect(
                got == want,
                f"CUDA graph re-plan: {name} {got!r} differs from the captured "
                f"{want!r}; a captured graph cannot change it — use one "
                "PagedAttention graph instance per graph bucket",
            )
        bounded = (
            ("total_q_tokens", metadata.total_q_tokens, cap.total_q_tokens),
            ("max_q_len", metadata.max_q_len, cap.max_q_len),
            ("max_kv_len", metadata.max_kv_len, cap.max_kv_len),
        )
        for name, got, limit in bounded:
            _expect(
                got <= limit,
                f"CUDA graph re-plan: {name} {got} exceeds this instance's "
                f"capacity {limit} (the captured kernels were planned with the "
                "capacity value) — use a bucket with a larger GraphCapacity",
            )
        if metadata.block_tables is not None:
            _expect(
                int(metadata.block_tables.shape[1]) == cap.table_width,
                f"CUDA graph re-plan: block_tables width {metadata.block_tables.shape[1]} "
                f"differs from the captured {cap.table_width}",
            )
        else:
            _expect(
                int(metadata.kv_page_indices.shape[0]) <= cap.flat_capacity,
                f"CUDA graph re-plan: kv_page_indices has "
                f"{metadata.kv_page_indices.shape[0]} entries, reserved capacity is "
                f"{cap.flat_capacity}",
            )

    # ---- staging ----
    def targets(
        self, metadata: PagedAttentionMetadata, fresh: Derived
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """(reserved destination, source) pairs for one re-plan."""
        pairs = [
            (self.qo_indptr, metadata.qo_indptr),
            (self.kv_seq_lens, metadata.kv_seq_lens),
            (self.q_seq_lens, fresh.q_seq_lens),
            (self.cum_kv_seq_lens, fresh.cum_kv_seq_lens),
            (self.kv_page_indptr, fresh.kv_page_indptr),
        ]
        if metadata.block_tables is not None:
            pairs.append((self.block_tables, metadata.block_tables))
            pairs.append((self.kv_page_indices, fresh.kv_page_indices))  # b*W exactly
        else:
            n = int(metadata.kv_page_indices.shape[0])
            pairs.append((self.kv_page_indices[:n], metadata.kv_page_indices))
            if fresh.block_tables is not None:
                assert self.block_tables is not None  # reserve_dense_table() ran
                pairs.append((self.block_tables, fresh.block_tables))
        return pairs

    def derived_view(self, *, needs_dense: bool) -> Derived:
        return Derived(
            q_seq_lens=self.q_seq_lens,
            cum_kv_seq_lens=self.cum_kv_seq_lens,
            kv_page_indptr=self.kv_page_indptr,
            kv_page_indices=self.kv_page_indices,
            block_tables=self.block_tables if needs_dense else None,
        )


class Transaction:
    """Snapshot/restore for a set of (destination, source) copies plus any
    follow-on work; ``commit()`` drops the snapshots, leaving the context
    without committing restores every destination.

    The staging copies run inside ``__enter__``.  Python does not call
    ``__exit__`` when ``__enter__`` itself raises, so a copy that fails midway
    (a source of the wrong shape or dtype is the synchronous case) is rolled
    back right there: every destination written so far is restored from its
    snapshot before the error propagates.  Asynchronous CUDA execution
    failures are outside this guarantee, as for every sync-free protocol.
    """

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._pairs = pairs
        self._snapshots: Optional[List[torch.Tensor]] = None
        self._committed = False

    def __enter__(self) -> "Transaction":
        snapshots = [dst.clone() for dst, _ in self._pairs]
        written = 0
        try:
            for dst, src in self._pairs:
                dst.copy_(src, non_blocking=True)
                written += 1
        except Exception:
            # restore every destination touched, including the one whose copy
            # raised (a synchronous failure leaves it unwritten; restoring it
            # from its own snapshot is harmless either way)
            touched = self._pairs[: written + 1]
            for (dst, _), snap in zip(touched, snapshots[: len(touched)], strict=True):
                dst.copy_(snap)
            raise
        self._snapshots = snapshots
        return self

    def commit(self) -> None:
        self._committed = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._committed and self._snapshots is not None:
            for (dst, _), snap in zip(self._pairs, self._snapshots, strict=True):
                dst.copy_(snap)
        self._snapshots = None


__all__ = ["GraphBuffers", "GraphCapacity", "Transaction"]
