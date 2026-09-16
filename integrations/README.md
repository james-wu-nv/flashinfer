# Engine integration sketches for the unified paged-prefill API

Two reviewable diffs showing what vLLM and sglang delete and gain by moving
their paged-prefill path to `flashinfer.prefill.PagedAttention` (PR 4015 and
the follow-up work described in
`docs/design_docs/paged_attention_unified_lifecycle.md`).

**Verification status, read this first.** These diffs have NOT been run
against the live engines. What IS machine-checked, on real GPUs, is the
data flow they rely on: `tests/experimental/test_paged_attention_engine_shapes.py`
replicates each engine's exact metadata pipeline and drives the unified API
under a zero-sync guard against an independent fp32 oracle, and
`tests/experimental/test_paged_attention_coverage.py` replays vLLM-shaped and
sglang-shaped mixed batches (TC04), strided inputs (TC05) and the CUDA-graph
cross-feature matrix (TC07). The diffs were audited against the pinned engine
clones (vLLM @ 3bb78262, sglang @ e7f74473); every claim below survived that
audit (several earlier claims did not and were removed).

## Friction found (the honest list; this is the point of the exercise)

Library-side, fixed during the exercise:
- **CSR to dense derivation tail reads (was silent-NaN class)**: cuDNN gathers
  K/V pages by dense-table *width* before masking, so derived rows' tail
  columns are dereferenced. The derivation now clamps each row's tail to
  the request's own last page (fuzzer regression:
  `csr_overallocated_nan_tail`, found by a NaN-page probe).
- **`window_left < -1` was backend-divergent** (trtllm's launcher
  special-cases exactly -1; -2 became a force-enabled negative window, i.e.
  garbage on SM100 only). Now rejected at plan/resolve.
- **Query and page-table ABI**: fa2 and trtllm-gen silently misread a
  non-unit inner q stride, cuDNN misreads any q that is not packed THD, and
  trtllm-gen misreads a narrow view of a wider block table. `run()` now
  requires `q.stride(-1) == 1` for every backend and packed q where the
  backend needs it; trtllm-gen declines a non-contiguous table at plan;
  cuDNN takes its width-exact view internally, so a capacity-width table is
  fine for every backend.
- **cuDNN graph-cache key** omitted the page table's shape and strides and
  the decorator order let a failed build poison the cache
  (`flashinfer/cudnn/prefill.py`); both fixed with native regression tests.
- **Graph lifecycle**: a re-plan that changed a semantic argument replayed
  stale kernels, a copy that failed midway left reserved buffers
  inconsistent, and a failed first plan locked the capacity. Now the first
  graph-mode plan freezes the semantic contract, staging is transactional,
  and the capacity is published only with a successful plan
  (`GraphCapacity`, `update()`).
- **Zero rows**: `kv_len == 0` is a legal padding row (vLLM's graph padding);
  sglang's fill value 1 stays a live row.
- **Plan cost**: with host mirrors the eager plan on trtllm-gen or cuDNN
  issues zero kernel launches and one pinned host-to-device copy; the fa2
  planner no longer receives pageable host arrays.
- fa2/fa3 paged `(192,128)` requires `k_page_stride == v_page_stride`;
  separately-allocated K/V pools violate it, so it is not declared for the fa
  family (cudnn covers it).

Engine-side, to carry in the real PRs:
- **vLLM**: value validation is unconditional and consumes the host mirrors,
  so the all-trtllm async-mode path that today *skips* `seq_lens_cpu`
  retrieval would pay one packed device-to-host copy per step (the metadata
  constructor fetches missing mirrors itself). DCP rewrites `seq_lens_cpu` to
  DCP-local lengths while the GPU `seq_lens` stays global; the
  mirror-consistency contract needs a device-side local-lens step, so DCP
  stays out of the v1 diff. The `q_data_type` un-quantize mutation is shared
  with decode and must stay until the decode follow-up. Padding rows with
  `q_len == 0` (vLLM's decode graphs) are still rejected (ledger M17), so a
  captured bucket must give each padding row one query token for now.
  `logits_soft_cap` and sinks models are in scope: both are rejectable
  plan-time axes.
- **sglang**: the prefill plan site already receives `seq_lens_cpu`
  (`call_begin_forward`), so the KV mirror is a copy-forward; the query
  mirror comes from `extend_seq_lens_cpu`. The radix-extend cascade's paged
  half is OUT of the v1 envelope (it runs `causal=False` with
  `kv = prefix_lens`, which contains ZERO rows for no-prefix requests;
  accepted as padding rows, but their LSE is unspecified and the merge needs
  it). `fast_prefill_plan` maps to `PagedAttention.update()` under an
  explicit `GraphCapacity`; the mapping is written but has not been run
  against the engine. Custom-mask paths stay on fa2 because graph mode plus
  custom mask is a library follow-up.

Both engines: pass the workspace the engine already shares with its legacy
wrappers. The library's shared 128 MiB default has been observed to overflow
the fa2 split-KV planner on a single 2048-token, 32-head prefill (ledger
M14); the sibling work package WP-I is adding `workspace_requirements()`.
