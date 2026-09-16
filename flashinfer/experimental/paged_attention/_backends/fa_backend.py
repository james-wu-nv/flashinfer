"""fa2/fa3 backend: the existing BatchPrefillWithPagedKVCacheWrapper.

Dialect: CSR page metadata + host arrays for the split-KV scheduler
(computed from the mirrors the contract layer guarantees — zero-sync).
The generated FA wrapper holds its own plan state, so this backend keeps one
wrapper instance as stable storage and re-plans it in place.
"""

from __future__ import annotations

import torch

from .._contracts import LN2, PlanMetadata
from .._planning import Derived
from ._capabilities import _BackendPlanUnsupportedError


class _FaBackend:
    def __init__(
        self, device, kv_layout, workspace, backend: str = "fa2", graph_capacity=None
    ):
        from ....prefill import BatchPrefillWithPagedKVCacheWrapper

        self.name = backend
        self._device = device
        if graph_capacity is None:
            self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace, kv_layout, backend=backend
            )
        else:
            # The wrapper's own CUDA-graph protocol: it copies each plan's CSR
            # metadata into these reserved buffers, so the captured kernel
            # keeps reading valid pointers across re-plans.
            b = graph_capacity.batch_size
            i32 = dict(dtype=torch.int32, device=device)
            self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout,
                use_cuda_graph=True,
                qo_indptr_buf=torch.zeros(b + 1, **i32),
                paged_kv_indptr_buf=torch.zeros(b + 1, **i32),
                paged_kv_indices_buf=torch.zeros(graph_capacity.flat_capacity, **i32),
                paged_kv_last_page_len_buf=torch.zeros(b, **i32),
                backend=backend,
            )
            # The wrapper fixes its row budget from the first plan it sees and
            # afterwards accepts any total_num_rows <= that budget (its CPU
            # scheduler plans for the budget; the kernel reads the live row
            # count from the int workspace).  Seeding it with the capacity
            # lets the first batch be smaller than the bucket; this poke goes
            # away when the backend calls the FA module directly.
            self._wrapper._max_total_num_rows = graph_capacity.total_q_tokens
        self._lse_mode = "none"
        self._total_q_tokens = 0
        self._head_dim_vo = 0

    def preflight(self, meta: PlanMetadata) -> None:
        """Batch-specific checks; typed unsupported only, no allocation."""
        if self.name == "fa3":
            from ....utils import is_sm90a_supported

            if not is_sm90a_supported(self._device):
                raise _BackendPlanUnsupportedError(
                    f"fa3 needs SM90a and CUDA >= 12.3; {self._device} does not qualify"
                )

    def plan(self, meta: PlanMetadata, derived: Derived) -> None:
        # The kernel walks kv_page_indices as a raw int32 pointer bounded by
        # the page indptr; a strided view would be silently misread.  In graph
        # mode this is the reserved (contiguous) buffer, so only the eager CSR
        # form can fail here — reject before the wrapper's plan moves state.
        idx = derived.kv_page_indices
        if not idx.is_contiguous():
            raise ValueError(
                f"{self.name} requires a contiguous kv_page_indices, got strides "
                f"{tuple(idx.stride())}: the kernel walks the flat page-id list as a "
                "packed int32 array — pass kv_page_indices.contiguous()"
            )
        qo_host = meta.qo_indptr_cpu.to(torch.int32)
        kv_lens_host = meta.kv_seq_lens_cpu.to(torch.int32)
        page = meta.page_size
        pages_host = (kv_lens_host + page - 1) // page
        kv_indptr_host = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32),
                torch.cumsum(pages_host, 0, dtype=torch.int32),
            ]
        )
        last_len_host = ((kv_lens_host - 1) % page + 1).to(torch.int32)
        self._wrapper.plan(
            qo_host,
            kv_indptr_host,
            derived.kv_page_indices,
            last_len_host,
            meta.num_qo_heads,
            meta.num_kv_heads,
            meta.head_dim_qk,
            page,
            head_dim_vo=meta.head_dim_vo,
            causal=meta.causal,
            window_left=meta.window_left,
            q_data_type=meta.q_dtype,
            kv_data_type=meta.kv_dtype,
        )
        self._lse_mode = meta.lse_mode
        self._total_q_tokens = meta.total_q_tokens
        self._head_dim_vo = meta.head_dim_vo

    def run(
        self,
        q,
        k_cache,
        v_cache,
        *,
        out=None,
        lse=None,
        sm_scale: float,
        k_scale=None,
        v_scale=None,
    ):
        # The generated-FA wrapper reads sm_scale from plan-time state and its
        # run() has no override, while the kernel takes it as a launch arg.
        # Setting it here keeps sm_scale a per-run (per-layer) value; this
        # poke goes away when the backend calls the FA module directly.
        self._wrapper._sm_scale = sm_scale
        need_lse = self._lse_mode != "none"
        rows, n = q.shape[0], self._total_q_tokens
        if rows != n:
            # Graph mode: q/out/lse are the capacity-sized capture buffers, and
            # the wrapper insists on q.shape[0] == qo_indptr[-1].  Hand it the
            # batch's row prefix — the same storage, so a captured graph keeps
            # reading the same pointers — and return the caller's buffers whole.
            if out is None:
                out = torch.empty(
                    rows, q.shape[1], self._head_dim_vo, dtype=q.dtype, device=q.device
                )
            if lse is None and need_lse:
                lse = torch.empty(
                    rows, q.shape[1], dtype=torch.float32, device=q.device
                )
            q_v, out_v = q[:n], out[:n]
            lse_v = lse[:n] if lse is not None else None
        else:
            q_v, out_v, lse_v = q, out, lse
        r = self._wrapper.run(
            q_v,
            (k_cache, v_cache),
            k_scale=k_scale,  # fp8 KV: folded into the softmax scale by the wrapper
            v_scale=v_scale,  # fp8 KV: applied to the output by the kernel path
            out=out_v,
            lse=lse_v,
            return_lse=need_lse,
        )
        if not need_lse:
            return (out if out is not None else r), None
        out_t, lse_t = r
        if self._lse_mode == "basee":
            lse_t.mul_(LN2)  # FA kernels emit base-2 (exp2 softmax); one fold
        return (out if out is not None else out_t), (lse if lse is not None else lse_t)


__all__ = ["_FaBackend"]
