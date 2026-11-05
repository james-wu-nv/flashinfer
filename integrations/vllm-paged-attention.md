# vLLM: paged prefill via the unified API

Target: `vllm/v1/attention/backends/flashinfer.py` (clone @ 3bb78262).
Scope: the **prefill half** only. Decode routing, cascade, DCP and the
quantized-KV paths keep their current code (they share
`use_trtllm_attention` with decode and wait for the decode follow-up).

The library contract this sketch relies on is described in
`docs/design_docs/paged_attention_unified_lifecycle.md`; the signatures
below are those of `flashinfer/_paged_attention.py` at the integration head.

## 1. Init: one pinned resolution, one instance per graph bucket

```python
# FlashInferMetadataBuilder.__init__ (after the q dtype / hyperparameter
# decisions, ~:893-895 read window_left, logits_soft_cap, has_sinks)
+from flashinfer.prefill import (
+    GraphCapacity, PagedAttention, PagedAttentionMetadata,
+    resolve_paged_attention,
+)
+
+# The Resolution's config key covers window_left, causal, need_lse and the
+# three feature flags, so a model with a sliding window or attention sinks
+# resolves its own variant; hold one Resolution per (KV-cache group, window).
+self._prefill_resolution = resolve_paged_attention(
+    device=self.device,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type_prefill,      # bf16/fp16 envelope
+    kv_dtype=self.kv_cache_dtype,          # fp8 e4m3 KV is declared for fa2 only
+    page_size=self.page_size,
+    kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
+    causal=True,
+    window_left=self.window_left,          # uniform per batch, as today
+    need_lse=False,                        # DCP / cascade stay on the current path
+    logits_soft_cap=self.logits_soft_cap,  # rejectable axis: fa2/fa3 only
+    sinks=self.has_sinks,                  # rejectable axis: fa2/fa3/trtllm-gen/cake
+)
+logger.info_once(self._prefill_resolution.explain())   # per-backend reasons
+
+# Eager instance, constructed ONCE (per-build construction would redo backend
+# setup every scheduler step).  Pass the workspace vLLM already shares with
+# its legacy wrappers; without it every instance on the device shares one
+# 128 MiB library pool, which the fa2 split-KV planner can overflow on a
+# single long prefill (ledger M14).
+self._prefill_attn = PagedAttention(
+    self.device, workspace_buffer=self._get_workspace_buffer()
+)
```

Deletes on the prefill side: the `prefill_use_trtllm` predicate
(`use_trtllm_attention(...)` at `:1333-1345`), the `page_size >= 128` force
(`:1329-1332`), and the init-time capability guesswork this resolution
replaces. `explain()` reports every exclusion reason, replacing the logged
"reverting to FlashInfer" strings. `need_lse` is part of the config key: the
day DCP or cascade move over and need the LSE, re-resolve with
`need_lse=True` rather than reusing this object.

## 2. build(): one metadata object and one plan replace both prefill builds

```python
# in build(), prefill branch (replacing the TRTLLMPrefill construction
# :1540-1575 AND the FIPrefill CSR construction :1577-1661 for the prefill slice)
+md = PagedAttentionMetadata.dense(          # once per step; validates once
+    qo_indptr[prefill_start:] - qo_indptr[prefill_start],
+    seq_lens[prefill_start:],
+    block_table_tensor[prefill_start:],      # the persistent, wide table: fine
+    page_size=page_size,
+    max_q_len=max_q_len_prefill,             # host ints vLLM already has
+    max_kv_len=max_seq_len,
+    qo_indptr_cpu=qo_indptr_prefill_cpu,     # mirrors vLLM already owns
+    kv_seq_lens_cpu=seq_lens_cpu[prefill_start:],
+)
+self._prefill_attn.plan(
+    md,
+    num_qo_heads=self.num_qo_heads,
+    num_kv_heads=self.num_kv_heads,
+    head_dim_qk=self.head_dim,
+    q_dtype=self.q_data_type_prefill,
+    kv_dtype=self.kv_cache_dtype,
+    kv_layout=get_flashinfer_layout_string(self.kv_cache_layout),
+    causal=causal,
+    window_left=self.window_left,
+    lse_mode="none",
+    logits_soft_cap=self.logits_soft_cap,
+    use_sinks=self.has_sinks,
+    backend=self._prefill_resolution,
+)
```

Deletes: `_compute_flashinfer_kv_metadata` for the prefill slice (the numpy
indptr plus the Triton `_copy_page_indices_kernel` CSR expansion,
`:1248-1290`), the trtllm `cum_seq_lens_kv` GPU cumsum (`:1548-1562`), and
both prefill dataclasses' metadata duplication. The per-build `q_data_type`
mutation **stays**: it un-quantizes q for the FlashInfer-native *decode* path
too, so it can only go with the decode follow-up.

Notes that came out of the review and the library work:

- **Block table width.** Pass the persistent capacity-width table with the
  actual `max_seq_len`. cuDNN needs a table exactly
  `ceil(max_kv_len / page_size)` wide and takes that view internally;
  trtllm-gen and cake walk the table as a packed `(batch, width)` array and
  **decline a narrow view** (a `block_table_tensor[:, :w]` slice) at plan
  time, so never pre-trim the table for them.
- **Host mirrors.** Value validation is unconditional. With the mirrors the
  plan is sync-free; without them `PagedAttentionMetadata` performs one
  packed device-to-host copy at construction. The all-trtllm async path that
  today skips `seq_lens_cpu` retrieval (`needs_seq_lens_cpu`, `:1410-1415`)
  would therefore pay one D2H per step through the unified API.
- **Zero rows.** vLLM's graph padding (`seq_lens[num_reqs:].fill_(0)` in
  `vllm/v1/worker/gpu_model_runner.py:2202`) is legal input: a `kv_len == 0`
  row is a padding row whose page is never read and whose output row and LSE
  are unspecified (trtllm-gen leaves them unwritten; vLLM never reads padded
  rows). The padding rows' `q_len == 0`
  (`query_start_loc.np[num_reqs + 1:].fill(...)`, `:2084`) is **not** yet
  accepted (ledger M17), so the eager prefill path above is unaffected but a
  captured decode-shaped bucket would have to give each padding row one query
  token until M17 lands.
- **Fused-QKV queries.** `q = qkv[:, :num_qo_heads]` is accepted without a
  copy by fa2, fa3 and trtllm-gen (only `q.stride(-1) == 1` is required);
  cuDNN and cake require packed storage and reject the slice, so `auto` on a
  cuDNN-only configuration needs a packed copy.

## 3. forward(): one run replaces the trtllm / FlashInfer fork

```python
-# trtllm_batch_context_with_kv_cache(...)  /  prefill_wrapper.run(...)
-# (the prefill_use_trtllm fork at :2148 ff.)
+out, _ = attn_metadata.prefill_attn.run(
+    prefill_query,                            # the head slice is fine (see above)
+    (kv_cache_k, kv_cache_v),
+    out=output[num_decode_tokens:],
+    sm_scale=self.scale,                      # per layer; one plan serves all layers
+    k_scale=layer._k_scale_float if kv_is_fp8 else None,
+    v_scale=layer._v_scale_float if kv_is_fp8 else None,
+    sinks=self.sinks if self.has_sinks else None,   # fp32 (num_heads,) tensor
+)
```

## 4. CUDA graphs: one instance per bucket, `update()` per step

```python
# at graph-capture time, per cudagraph bucket (batch_size, total tokens)
cap = GraphCapacity(
    batch_size=bucket_batch_size,
    total_q_tokens=bucket_num_tokens,
    max_q_len=max_q_len_in_bucket,
    max_kv_len=block_table_width * page_size,   # cuDNN: table_width * page_size
    page_size=page_size,
    table_width=block_table_width,
)
attn = PagedAttention(self.device, graph_capacity=cap,
                      workspace_buffer=self._get_workspace_buffer())
attn.plan(md_capture, ..., backend=self._prefill_resolution)   # freezes backend + semantics
with torch.cuda.graph(g):
    attn.run(q_buf, (k_cache, v_cache), out=out_buf, sinks=..., sm_scale=self.scale)

# per step
attn.update(md_step)      # stages into reserved storage; must fit `cap`
g.replay()
```

Rules: `batch_size`, paging form and `page_size` must match the capacity
exactly; `total_q_tokens`, `max_q_len`, `max_kv_len` and the table width are
bounds (`table_width` exact). Backends are planned with the capacity maxes
(capacity substitution), so pass the step's actual maxes. `update()` rolls
back on failure and raises during capture. `sm_scale` is a launch scalar
baked into the captured graph. Until M17 lands, every padded row needs
`q_len >= 1`.

## What stays (v1 scoping, honest)

- decode routing plus `use_trtllm_attention` (shared with decode), and the
  artifactory HTTP probe behind the decode gates;
- **DCP**: build() rewrites `seq_lens_cpu` to DCP-local lengths while the GPU
  `seq_lens` stays global (`:1418-1440`), and the unified API trusts the
  mirrors to match the device tensors, so DCP needs a device-side local-lens
  step first. The LSE contract (packed fp32, declared base, every backend)
  makes the eventual migration attractive;
- cascade (fa2 `MultiLevelCascadeAttentionWrapper` hardwired);
- fp8-Q, nvfp4 and the shared `q_data_type` un-quantize mutation (fp8 e4m3
  KV with a bf16/fp16 q is inside the envelope on fa2 only);
- spec-decode reorder policy;
- graph-mode buckets whose padding rows have `q_len == 0` (M17).

Models with `logits_soft_cap` or attention sinks are **in** scope now: both
are plan-time axes, and a backend that cannot apply one is excluded at
resolve with a reason instead of dropping it silently.
