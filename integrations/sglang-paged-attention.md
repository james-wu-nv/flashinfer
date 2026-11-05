# sglang: paged prefill via the unified API

Target: `python/sglang/srt/layers/attention/flashinfer_backend.py`
(clone @ e7f74473).
Scope: the **full-sequence paged prefill paths** (no radix-cache hit /
multimodal / deterministic mode). Explicitly OUT of the v1 diff:
- the radix-extend cascade, BOTH halves. The paged (prefix) half runs
  `causal=False` over `kv = prefix_lens` (`forward_extend`, `:1428`), and
  no-prefix requests have `prefix_len == 0`, i.e. zero-length KV rows. Those
  are accepted as padding rows now (output and LSE unspecified, possibly unwritten), but the
  cascade merge (`_safe_merge_state`, `:1437`) consumes the prefix half's
  LSE, which the contract does not define for them; it waits for the ragged
  follow-up anyway. The CUDA-graph fill value 1
  (`get_cuda_graph_seq_len_fill_value`, `:1286`) is an ordinary live row.
- custom-mask paths (target-verify / multi-item). `plan(custom_mask=...)`
  exists on fa2, but sglang runs these under CUDA graphs and a custom mask in
  graph mode is rejected by the library until the packed-mask storage joins
  `GraphCapacity`; they pin fa2 exactly as today.

The library contract is described in
`docs/design_docs/paged_attention_unified_lifecycle.md`; the signatures below
are those of `flashinfer/_paged_attention.py` at the integration head.

## 1. Init: unified instance + one resolution (token-CSR form)

```python
# FlashInferAttnBackend.__init__ (replacing the fa2 pin at :307 and the
# paged-prefill wrapper construction at :490-503)
+from flashinfer.prefill import (
+    GraphCapacity, PagedAttention, PagedAttentionMetadata,
+    resolve_paged_attention,
+)
+
+self._prefill_resolution = resolve_paged_attention(
+    device=self.device,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type,
+    page_size=1,                      # token-granular slots, as today
+    kv_layout="NHD",
+    causal=True,
+    need_lse=True,                    # merge_state consumes LSE
+    kv_input_form="page_indices",     # the flat kv_indices sglang builds
+    window_left=self.sliding_window_size if swa_layer_group else -1,
+)
+# workspace_buffer: hand it the workspace sglang already shares with its
+# legacy wrappers (:438-444); omitted, instances share a 128 MiB per-device
+# library pool
+self.prefill_attn = PagedAttention(self.device, workspace_buffer=self.workspace_buffer)
```

`backend="fa2"` pinning becomes unnecessary: at `page_size=1` the
dense-needing backends (cuDNN, trtllm-gen, cake) are capability-excluded
automatically and `self._prefill_resolution.explain()` says why; on
architectures where more CSR-native backends appear, sglang inherits them
with zero code change. `window_left` is part of the resolution's config key,
so an SWA layer group holds its own `Resolution` and `PagedAttention`.

## 2. Per-batch: plan directly from what the indices updater builds

`call_begin_forward` (`:1690-1800`) already receives `seq_lens_cpu` from the
updater (it builds `global_override_indptr_cpu` from it, `:1747-1751`), so
the host mirror of the KV lengths exists at the plan site; the query-side
mirror is `qo_indptr`'s CPU origin (`extend_seq_lens_cpu`). Both are
copy-forwards of host data, not syncs.

```python
# FlashInferIndicesUpdaterPrefill.call_begin_forward
 kv_indices = ...  # UNCHANGED: translator gather from req_to_token (pool-owned)
-kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)   # DELETED (:1712)
-global_override_indptr_cpu[...] = torch.cumsum(seq_lens_cpu, dim=0) # DELETED (:1747-1751)
-wrapper.begin_forward(kv_indptr, kv_indices, self.kv_last_page_len[:bs], ...)
+md = PagedAttentionMetadata.csr(
+    qo_indptr,                                 # same preallocated buffer slice
+    paged_kernel_lens,                         # the masking truth, directly
+    kv_indices,                                # over-allocated tail is fine
+    page_size=1,
+    max_q_len=max_extend_len,                  # host ints sglang carries
+    max_kv_len=max_kv_len,
+    qo_indptr_cpu=qo_indptr_cpu,               # host origin of qo_indptr
+    kv_seq_lens_cpu=seq_lens_cpu[:bs],         # already passed to this call
+)
+self.prefill_attn.plan(
+    md,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type,
+    kv_layout="NHD",
+    causal=True,
+    window_left=window_left,                   # -1, or the SWA group's size
+    lse_mode="base2",
+    backend=self._prefill_resolution,
+)
```

Deleted wholesale:
- the `kv_indptr` page-unit cumsum and the `kv_last_page_len` argument
  (derived internally from `kv_seq_lens`; one truth, not two);
- the `global_override_indptr_cpu` host cumsum (the metadata object derives
  every host array from the mirrors in one numpy pass and uploads them once
  from pinned memory, so the fa2 planner sees no pageable host-to-device
  copy).

The plan is sync-free with the mirrors supplied; without them the metadata
constructor performs one packed device-to-host copy.

## 3. forward_extend: run

```python
-o = prefill_wrapper_paged.forward(q, kv_pool, causal=True, sm_scale=layer.scaling,
-                                  window_left=..., logits_soft_cap=...,
-                                  k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
+o, lse = self.prefill_attn.run(
+    q.view(-1, layer.tp_q_head_num, layer.head_dim),
+    (k_pool, v_pool),
+    sm_scale=layer.scaling,                    # per layer; one plan serves all layers
+    k_scale=layer.k_scale_float if kv_is_fp8 else None,
+    v_scale=layer.v_scale_float if kv_is_fp8 else None,
+)
```

`logits_soft_cap` moves to `plan()` (it selects a compiled kernel variant on
fa2/fa3 and is a rejectable capability axis); `sm_scale` and the fp8 KV
scales stay per-call host floats. For the prefix-cache cascade, the paged
half's `(o, lse)` would feed `_safe_merge_state` against the ragged half's
output exactly as today (same base-2 LSE convention), once the zero-row LSE
question above is settled.

## 4. CUDA-graph replay: `update()` replaces `fast_prefill_plan`

`fast_prefill_plan` (`:177-289`) is CUDA-graph replay machinery: asserted
fa2-only, it writes the wrapper's pinned capture buffers for EAGLE
draft-extend. The unified API's graph lifecycle covers that role:

```python
# capture, per graph bucket (batch size bs, num_draft_tokens per request)
cap = GraphCapacity(
    batch_size=bs, total_q_tokens=bs * num_draft_tokens,
    max_q_len=num_draft_tokens, max_kv_len=max_context_len,
    page_size=1, kv_input_form="page_indices",
    flat_capacity=bs * max_context_len,        # reserved length of kv_indices
)
attn = PagedAttention(self.device, graph_capacity=cap, workspace_buffer=self.workspace_buffer)
attn.plan(md_capture, ..., lse_mode="base2", backend=self._prefill_resolution)
with torch.cuda.graph(g):
    attn.run(q_buf, (k_pool, v_pool), out=out_buf, lse=lse_buf, sm_scale=layer.scaling)

# per replay
attn.update(md_step)      # only the batch changes; rolled back on failure
g.replay()
```

At `page_size=1` no dense table is reserved for the flat form (dense-reading
backends are excluded), so a bucket costs the CSR buffers only. The fill
value 1 rows are live rows; a row with `kv_len == 0` is a padding row.
`update()` is rejected during capture and requires the same batch size,
paging form and page size as the capacity. This mapping has not been run
against the live engine.

## What stays (v1 scoping, honest)

- the entire radix-extend cascade (both halves; see Scope above),
- decode wrappers plus `fast_decode_plan` (decode follow-up),
- custom-mask paths (target-verify / multi-item): graph mode plus custom mask
  is a library follow-up,
- the fa2 `AttentionSink` and multi-item paths that pin fa2 today: sinks are
  a plan-time axis (`plan(use_sinks=True)`, `run(sinks=...)`) on fa2/fa3,
  trtllm-gen and cake, so they can move once their graph buckets are
  expressed as `GraphCapacity` instances,
- SWA: the paged-only SWA half maps to `window_left` (its own Resolution,
  since window is in the config key); the ragged-extend SWA half remains on
  the custom-mask fa2 path.
