# Unified paged attention: contract and lifecycle

This document describes the experimental `flashinfer.prefill.PagedAttention`
facade and the private `flashinfer.experimental.paged_attention` package behind
it: the public surface, the three-stage CUDA-graph lifecycle, backend
selection, per-backend metadata derivation, the input, zero-row and LSE
contracts, the feature axes, workspace ownership, the benchmark tooling,
and the correspondence to the Batch MLA package it mirrors
(`docs/design_docs/batch_mla_backend_architecture.md`). Everything stated
here is taken from the code; where a report and the code disagree, the code
wins, and the "Known limitations" section carries the ledger ids of the open
items. The user-facing reference (status, opt-in, limitations, the backend
selection rule and its measurements) is
`flashinfer/experimental/paged_attention/README.md`; the API is not listed in
`docs/api/` before graduation.

## Summary

`PagedAttention` runs paged attention over the existing fa2/fa3, cuDNN,
trtllm-gen and Cake kernels behind one contract. There is no new kernel. The
design separates four concerns:

- The facade (`flashinfer/_paged_attention.py`, re-exported from
  `flashinfer.prefill`) owns the public signatures and the experimental
  marker, and hands off lazily to the package.
- The contract and planning modules own what a batch means
  (`PagedAttentionMetadata`), what a plan means (`PlanMetadata`), what a
  resolution promised (`Resolution`), validation, and the canonical-to-derived
  conversions.
- The controller owns the plan/run lifecycle: level-1 pinning against a
  `Resolution`, the level-2 candidate walk in the auto policy's order,
  capacity substitution, the frozen contract of a captured graph,
  transactional staging and publication.
- Each backend module owns one complete implementation behind three calls:
  `preflight(meta)`, `plan(meta, derived)` and `run(...)`.

The plan/run flow is:

```text
flashinfer.prefill.resolve_paged_attention()        (engine init, tensor-free)
  -> Resolution: the candidate set + exclusion reasons, pinned to a device

flashinfer.prefill.PagedAttention(device, graph_capacity=..., workspace_buffer=...)
  -> PagedAttentionMetadata.dense(...) / .csr(...)   (once per scheduler step)
  -> plan(metadata, **semantic kwargs, backend=Resolution | name | "auto")
       -> validate; pin against the Resolution or resolve now
       -> [graph mode] capacity preflight; substitute capacity maxes
       -> rank the pinned candidates for this batch (_auto_policy.py;
          graph mode: until the first successful plan, which freezes the backend)
       -> for each candidate in that order: derive its declared forms, preflight()
       -> [graph mode] check the frozen contract; stage into reserved storage
       -> chosen backend.plan(meta, derived)
       -> publish
  -> run(q, (k_cache, v_cache), out=, lse=, sm_scale=, k_scale=, v_scale=, sinks=)
       -> runtime validation only, then backend.run()
  -> [graph mode] plan(metadata, same arguments) per step, then graph.replay()
```

## Motivation

vLLM and SGLang each carry their own paged-prefill routing, metadata
translation and CUDA-graph re-plan machinery on top of the legacy
`BatchPrefillWithPagedKVCacheWrapper`, `cudnn_batch_prefill_with_kv_cache` and
`trtllm_batch_context_with_kv_cache`. The failure modes that motivated this
package are recorded in the module docstring of `flashinfer/_paged_attention.py`
and in the source reports: a layout guessed wrong, a second copy of the paging
truth diverging from the first, a captured graph silently replaying kernels
planned for a different semantic configuration, a metadata copy that failed
halfway and left reserved buffers inconsistent, a page table addressed by the
wrong stride, and a plan path that paid hundreds of microseconds and several
device-to-host copies per step.

The package answers each with one mechanism, described below, and tests it
against an independent fp32 oracle (`tests/experimental/paged_attention/paged_attention_reference.py`)
with randomized valid and corrupted inputs
(`tests/experimental/paged_attention/test_paged_attention_fuzzer.py`). The property the package
is built around is reject-or-correct: every call either raises an actionable
error or returns results matching the oracle.

## Design properties

- One canonical metadata form: token-unit `qo_indptr`, per-request
  `kv_seq_lens`, exactly one paging form (dense `block_tables` or flat
  `kv_page_indices`), and required host maxes. Everything any backend wants is
  derived internally, and only for the backend that reads it.
- Closed input set with loud errors. Combinations outside the contract raise
  `ValueError` with the fix in the message.
- Two-level selection: a tensor-free `resolve_paged_attention()` at engine
  init fixes the candidate set; each `plan()` ranks that set from host facts
  of the batch and walks it, and only a typed unsupported signal from a
  candidate's `preflight()` moves on to the next candidate.
- Capacity-based CUDA-graph lifecycle: construct with a `GraphCapacity`, let
  the first plan freeze the semantic contract, then `plan()` again per step.
- Transactional publication: a failed plan leaves the previous plan runnable
  and, in graph mode, restores every reserved buffer it touched.
- Sync-free planning with host mirrors: one numpy pass validates and derives
  on the host, one pinned upload reaches the device, no implicit
  device-to-host copy after metadata construction.
- One output contract: packed fp32 LSE `(total_q_tokens, num_qo_heads)` in
  the base declared at plan time, identical for every backend.

## Interface boundaries

This design is specific to paged attention over a `(k_cache, v_cache)` pair.
It does not define:

- A repository-wide backend interface or registry. The three-call backend
  protocol in `flashinfer/experimental/paged_attention/_backends/__init__.py`
  is private to this package.
- A functional entry point. Planning is required before running.
- Autotuning. `backend="auto"` ranks the candidates with a fixed rule over
  host facts of the batch (`_auto_policy.py`); nothing is timed.
- KV-cache append, cascade or DCP orchestration.
- Public exposure of the package modules or the concrete backend classes.

## Public surface

`flashinfer/prefill.py` imports `PagedAttention` and `resolve_paged_attention`
eagerly from `flashinfer/_paged_attention.py` and exposes the value types
`PagedAttentionMetadata`, `GraphCapacity`, `Resolution` and
`PagedAttentionCapabilities` through a module `__getattr__`, so importing
`flashinfer` never loads the experimental package
(`tests/experimental/paged_attention/test_paged_attention_contract_cpu.py::test_importing_flashinfer_does_not_load_the_experimental_package`).
`resolve_paged_attention`, `PagedAttention.plan` and `run` carry
`@flashinfer_experimental_api`; calling one is the opt-in and emits an
`ExperimentalWarning` once.

### `resolve_paged_attention`

```python
resolve_paged_attention(
    *, device=None, cc_major=None,
    num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo=None,
    q_dtype, kv_dtype=None, page_size, kv_layout="HND",
    causal=True, need_lse=False, window_left=-1,
    kv_input_form="block_tables",
    logits_soft_cap=None, custom_mask=False, sinks=False,
    max_q_len=None,
    backend="auto",
) -> Resolution
```

Tensor-free. Returns the candidate set, in resolve-time order, and a reason
for every excluded backend; raises `ValueError` when nothing can run. The
result is pinned to `device` (compute capability major, minor and device
index) or, with only `cc_major`, to that major and no device
(`_selection._bind_device`). `max_q_len` is an optional shape hint, the
largest number of query tokens per request the batches planned with this
`Resolution` have; see "Level 1" under "Backend selection".

### `PagedAttentionMetadata`

```python
PagedAttentionMetadata.dense(qo_indptr, kv_seq_lens, block_tables, *,
                             page_size, max_q_len, max_kv_len,
                             qo_indptr_cpu=None, kv_seq_lens_cpu=None)
PagedAttentionMetadata.csr(qo_indptr, kv_seq_lens, kv_page_indices, *,
                           page_size, max_q_len, max_kv_len,
                           qo_indptr_cpu=None, kv_seq_lens_cpu=None)
```

Built once per scheduler step. The two constructors make "exactly one paging
form" structural. Construction runs the structural and value validation
(`_planning.validate_structure`, `validate_values`) and builds the
`HostArrays` staging tensor. With both mirrors supplied construction is
sync-free; otherwise it performs one device-to-host copy here (both arrays
packed into one transfer when both are missing) and every later `plan()` on
the object is sync-free (`_contracts.PagedAttentionMetadata.__post_init__`).
The object caches its derived forms per (need set, table width), so several
plans over one batch derive once.

### `PagedAttention`

```python
PagedAttention(device=None, *, graph_capacity=None, use_cuda_graph=False,
               workspace_buffer=None)
.plan(metadata, *, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo=None,
      q_dtype, kv_dtype=None, kv_layout="HND", causal=True, window_left=-1,
      lse_mode="none", logits_soft_cap=None, custom_mask=None,
      use_sinks=False, backend="auto") -> PagedAttention
.run(q, kv_cache, *, out=None, lse=None, sm_scale=None,
     k_scale=None, v_scale=None, sinks=None) -> (out, lse)
.explain() -> str
.backend -> Optional[str]
.device -> torch.device
PagedAttention.workspace_requirements(capacity, *, device=None, ...) -> int
```

`plan()` returns `self`. `run()` returns `(out, lse)` with `lse` `None` when
`lse_mode="none"`. `workspace_requirements()` is a static method; see
"Workspace".

### `GraphCapacity`

```python
GraphCapacity(batch_size, total_q_tokens, max_q_len, max_kv_len, page_size,
              kv_input_form="block_tables", table_width=None, flat_capacity=None)
GraphCapacity.from_metadata(metadata)
```

The capture shapes of one CUDA-graph bucket. Validation in `__post_init__`
(`_graph.py`): positive ints; `max_q_len <= total_q_tokens` (a bucket may
hold fewer query tokens than requests, because `q_len == 0` rows are legal);
dense form requires `table_width` with
`table_width == ceil(max_kv_len / page_size)` and derives
`flat_capacity = batch_size * table_width`; flat form requires
`flat_capacity >= max(batch_size, ceil(max_kv_len / page_size))` and forbids
`table_width`.

## Public usage

Eager, engine-shaped:

```python
import torch
from flashinfer.prefill import (
    PagedAttention, PagedAttentionMetadata, resolve_paged_attention,
)

device = torch.device("cuda")
res = resolve_paged_attention(
    device=device, num_qo_heads=32, num_kv_heads=8, head_dim_qk=128,
    q_dtype=torch.bfloat16, page_size=16, kv_layout="HND",
    causal=True, need_lse=True,
)
attn = PagedAttention(device, workspace_buffer=engine_workspace)

for step in scheduler:
    md = PagedAttentionMetadata.dense(
        qo_indptr, kv_seq_lens, block_tables,
        page_size=16, max_q_len=max_q_len, max_kv_len=max_kv_len,
        qo_indptr_cpu=qo_indptr_cpu, kv_seq_lens_cpu=kv_seq_lens_cpu,
    )
    attn.plan(md, num_qo_heads=32, num_kv_heads=8, head_dim_qk=128,
              q_dtype=torch.bfloat16, kv_layout="HND", causal=True,
              lse_mode="base2", backend=res)
    for layer in model:
        out, lse = attn.run(q, (k_cache, v_cache), sm_scale=layer.scale)
```

One CUDA-graph bucket:

```python
from flashinfer.prefill import GraphCapacity

cap = GraphCapacity(batch_size=256, total_q_tokens=1024, max_q_len=4,
                    max_kv_len=8192 * 16, page_size=16, table_width=8192)
attn = PagedAttention(device, graph_capacity=cap, workspace_buffer=engine_workspace)
attn.plan(md0, num_qo_heads=32, num_kv_heads=8, head_dim_qk=128,
          q_dtype=torch.bfloat16, kv_layout="HND", causal=True,
          lse_mode="base2", backend=res)          # freezes backend + semantics
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    attn.run(q_buf, (k_cache, v_cache), out=out_buf, lse=lse_buf)
for step in scheduler:
    attn.plan(md_step, num_qo_heads=32, num_kv_heads=8, head_dim_qk=128,
              q_dtype=torch.bfloat16, kv_layout="HND", causal=True,
              lse_mode="base2", backend=res)      # only the batch changes
    g.replay()
```

`q_buf`, `out_buf` and `lse_buf` are sized to `cap.total_q_tokens` rows. Rows
past the batch are not read; `out` and `lse` rows past it are unspecified and
may be written by a backend's whole-buffer post-processing. The `Resolution`
passed to
`plan()` must have been resolved with `need_lse=True` when `lse_mode` is not
`"none"`, because `need_lse` is part of the pinned config key.

## Layering and ownership

### Facade

`flashinfer/_paged_attention.py` owns the public signatures, the
experimental marker and the lazy handoff to the package. It contains no
backend logic.

### Controller

`_controller.py` (`PagedAttentionController`) owns:

- argument validation and normalization (`window_left >= -1`, `lse_mode`,
  `logits_soft_cap`, the custom-mask contract, the causal envelope);
- level-1 pinning against a `Resolution` (semantic key and device binding)
  or an inline resolve;
- level 2: the host-only `_PlanFacts` of the batch, the call into
  `_auto_policy.plan_order()`, the candidate walk in that order, and the
  published selection order and trace;
- graph mode: capacity preflight, capacity substitution, reserved-table
  reservation, the frozen contract, the staging transaction;
- publication of the plan state, `run()` validation and dispatch, and
  `explain()`.

### Contracts and planning

`_contracts.py` owns `PagedAttentionMetadata`, `PlanMetadata`, `Resolution`,
the config key (`resolve_config_key`) and the shared `_expect_*` helpers.
`_planning.py` owns the derived-form names (`DERIVED_FORMS`), `HostArrays`,
`Derived`, structural and value validation, the causal-envelope check and
`derive()`.

### Graph storage

`_graph.py` owns `GraphCapacity`, `GraphBuffers` (the reserved device storage
of one instance) and `Transaction` (snapshot/restore around the staging
copies and the backend's plan).

### Selection and capabilities

`_selection.py` owns `resolve_paged_attention()`, the environment probes and
`HEURISTIC_ORDER`, the resolve-time order of the candidate set.
`_auto_policy.py` owns the plan-time ranking of that set (`plan_order`, the
per-architecture `ordered_*_backends` over `_PlanFacts`) and the list of auto
candidates; it never decides support. The same module owns the measured
selection behind `backend="autotune"` (the workload key, the tuner runners
and `select_planned`), as the MLA package's `_auto_policy.py` does. `_backends/_capabilities.py` owns the
declarative `PagedAttentionCapabilities` table, its pure
`rejection_reason()` and the typed `_BackendPlanUnsupportedError`. The
capability table only holds facts that drive selection; adapter details such
as which derived forms a backend reads live on the backend class
(`DERIVED_NEEDS`) and are exposed through `_backends.derived_needs(name)`.

### Backends

`_backends/fa_backend.py` (`_FaBackend`, names `fa2` and `fa3`),
`_backends/cudnn_backend.py` (`_CudnnBackend`) and
`_backends/trtllm_gen_backend.py` (`_TrtllmGenBackend`, names `trtllm-gen`
and `cake`). Each owns its native dialect, its plan state, its LSE
normalization and its ABI checks. Nothing outside a backend knows its
dialect.

How each backend launches:

- fa2 / fa3 keep one `BatchPrefillWithPagedKVCacheWrapper` as stable storage
  and plan it in place through its `plan()`. `plan()` then stages the
  positional argument list of the kernel module's `paged_run` from the
  wrapper's state (`_FaBackend._stage_paged_run`), and `run()` fills the six
  entries only a call knows (`q`, the two caches, `out`, `lse`, the softmax
  scale) and launches the module itself. The list changes only with the
  plan, whereas the wrapper's `run()` rebuilds it from the wrapper's
  attributes on every call, behind a second API-decorator layer and a repeat
  of the checks `PagedAttention.run()` has made. The wrapper's `run()` stays
  the path of a sink plan (the `AttentionSink` variant wrapper owns its
  `run()`) and of every plan the staging declines because that `run()` would
  leave its plain path: a batch-invariant wrapper, a custom JIT module, NVFP4
  or per-(token, head) scaled KV, an fp8 `q`, multi-item scoring and the SM90
  fp8-KV verify kernel.
  `tests/experimental/paged_attention/test_paged_attention_fa_prepared.py`
  runs one planned wrapper through its own `run()` and through the staged
  list and compares what reaches `paged_run`, so an upstream change to that
  positional list fails the test.
- cuDNN stages a `_CudnnPrefillPlan` at `plan()`, and `run()` executes a
  prepared cuDNN graph through it, rebinding only Q, K, V, O and the stats.
  `cudnn_batch_prefill_with_kv_cache` remains the fallback where no plan can
  be staged (a cuDNN below the direct token-offset path, 9.24 for the
  explicit lengths this backend passes) and the packed-LSE probe. What the
  plan declares to cuDNN and which page table it binds differ between eager
  and CUDA-graph mode; see "Page-table ABI per backend".
- trtllm-gen / cake call `trtllm_batch_context_with_kv_cache` with the
  canonical tensors and the cumulative KV lengths.

## Package structure

```text
flashinfer/
|-- prefill.py                     re-exports; lazy value types
|-- _paged_attention.py            facade: resolve_paged_attention, PagedAttention
`-- experimental/paged_attention/
    |-- __init__.py
    |-- _contracts.py              PagedAttentionMetadata, PlanMetadata, Resolution
    |-- _planning.py               validation, HostArrays, Derived, derive()
    |-- _selection.py              resolve_paged_attention(), probes, HEURISTIC_ORDER
    |-- _auto_policy.py            plan-time ranking of the auto candidates
    |-- _controller.py             plan()/run() lifecycle
    |-- _graph.py                  GraphCapacity, GraphBuffers, Transaction
    `-- _backends/
        |-- __init__.py            factory, derived_needs()
        |-- _capabilities.py       CAPABILITIES, _BackendPlanUnsupportedError
        |-- fa_backend.py
        |-- cudnn_backend.py
        `-- trtllm_gen_backend.py
```

## Plan lifecycle

### Eager mode

`plan()` runs these phases (`_controller.PagedAttentionController.plan`):

1. Validate the metadata object and its device, normalize `head_dim_vo`,
   `kv_dtype`, `logits_soft_cap`, check the head counts and head dims,
   `kv_layout`, `window_left`, `lse_mode`, the custom mask, and the causal
   envelope when `causal=True`. The head counts and the head dims get one
   check each (`_expect_heads`: positive host ints, `num_qo_heads` divisible
   by `num_kv_heads`; `_expect_head_dims`: positive host ints), shared with
   `resolve_paged_attention()` and applied whatever the `backend` argument
   is: the auto policy divides by the head counts, the backends size launch
   grids from them and put the head dims into a JIT module name, and a
   pinned `Resolution` is not re-resolved (its config key compares with
   `==`, which a float or a numpy integer of the same value passes). A host
   int is a Python `int`: a `bool`, a numpy integer and a float are
   rejected with `ValueError`. The same holds for `page_size`, `max_q_len`
   and `max_kv_len`, which are checked when the metadata is constructed.
2. Pin: with a `Resolution`, compare the semantic part of the config key and
   the device binding; with a name or `"auto"`, resolve now on the instance's
   device.
3. Rank the pinned candidates with the auto policy (a single candidate is not
   ranked), then walk them in that order. For each: derive exactly the forms
   its `DERIVED_NEEDS` declares (cached on the metadata object), build a
   `PlanMetadata`, construct or reuse the backend object, and call
   `preflight(meta)`. A `_BackendPlanUnsupportedError` records the reason and
   moves on; any other exception propagates; an explicit backend name turns
   the typed signal into a plain `ValueError`.
4. Call the chosen backend's `plan(meta, derived)`.
5. Publish the backend, the metadata, the derived forms, the resolution, the
   walked order and the selection trace together.

A failed re-plan leaves the previous plan runnable, with one caveat the
controller documents: the generated-FA backend re-plans its legacy wrapper
in place, so a failure inside that wrapper's plan can leave backend-internal
state ahead of the published metadata. The argument list of the backend's
direct launch (see "Backends" under "Layering and ownership") is staged from
the wrapper after its plan has returned and is published together with it,
so `run()` never launches one plan's arguments against another plan's
schedule.

### Graph mode: three stages

Graph mode is entered with `graph_capacity=GraphCapacity(...)` (recommended)
or `use_cuda_graph=True` (capacity inferred from the first plan).

**Stage 1, construct.** With an explicit capacity the controller allocates
`GraphBuffers` at construction: `qo_indptr (b+1)`, `kv_seq_lens (b)`,
`kv_page_indices (flat_capacity)`, `q_seq_lens (b)`, `cum_kv_seq_lens (b+1)`,
`kv_page_indptr (b+1)`, and in the dense form the mirrored table
`(b, table_width)`. In the flat form the dense table is reserved later, only
once a backend that reads one is chosen (`GraphBuffers.reserve_dense_table`),
and never below `MIN_DENSE_PAGE_SIZE` (8), where it would degenerate to
`(batch, max_context)`. With `use_cuda_graph=True`, the buffers are built
into a local during the first plan from `GraphCapacity.from_metadata` and
published only on success, so a failed first plan installs no capacity
(`_controller.plan`, the `gb` local). In the inferred dense form
`max_kv_len` is widened to `table_width * page_size` so the table's whole
width stays usable.

**Stage 2, first plan freezes.** Before any reserved buffer is written,
`GraphBuffers.preflight(metadata)` checks the batch against the capacity:
exact for `batch_size`, `kv_input_form`, `page_size` and the dense table
width; upper bounds for `total_q_tokens`, `max_q_len`, `max_kv_len` and the
flat page-id length. Then the candidates are ranked, on the capacity's bounds
and the first batch's host lengths (see "Level 2: the auto policy"), and
walked, and the controller builds the frozen contract of the chosen plan:
the backend name, whether a dense table is present, and every `PlanMetadata`
field that is not a per-batch value
(`_PER_BATCH_FIELDS` = `qo_indptr`, `kv_seq_lens`, `block_tables`,
`custom_mask`, `qo_indptr_cpu`, `kv_seq_lens_cpu`, `batch_size`, `max_q_len`,
`max_kv_len`). The set is computed from `dataclasses.fields(PlanMetadata)`,
so a semantic keyword added later is frozen automatically. A later graph-mode
`plan()` whose contract differs raises `ValueError` naming the field, before
the transaction is entered; the fix is a new instance and a recapture.

**Capacity substitution.** In graph mode backends are planned with the
capacity's `max_q_len` and `max_kv_len`, not the batch's own, and the dense
table derived from flat page ids is sized to the reserved one
(`metadata.derived(needs=..., max_kv_len=gb.capacity.max_kv_len)`). The
captured kernels keep reading those values; a batch's own maxes are only
validated to be at most the capacity's. Engines therefore pass the actual
per-step maxes and need not know about upper bounds.

**Staging and rollback.** The new batch is copied into the reserved storage
inside a `Transaction` built from `GraphBuffers.targets(metadata, fresh)`,
which pairs each reserved destination with its source and skips forms the
chosen backend did not request. `Transaction.__enter__` snapshots every
destination, runs the copies, and on a failure restores every destination it
touched before re-raising (Python does not call `__exit__` when `__enter__`
raises). The backend's `plan(meta, derived)` runs inside the `with` block; a
failure there also restores every buffer. Asynchronous CUDA execution
failures are outside this guarantee, as for every sync-free protocol.

Backends in graph mode only ever see the reserved storage
(`GraphBuffers.derived_view`) and are constructed with the capacity so they
can size their own state up front: the generated-FA wrapper is built with
`use_cuda_graph=True` and reserved CSR buffers sized by the capacity, and its
row budget is seeded with `capacity.total_q_tokens`
(`_FaBackend._graph_bufs`, `_make_wrapper`); cuDNN's LSE gather indices, on
the fallback path, keep the capacity's row count; trtllm-gen's multi-CTA KV
counter buffer is sized at plan from `(batch, heads, SM count)` and stays
stable because the batch size is fixed.

Candidate backend objects are cached per `(name, kv_layout)` once
constructed, except during a first graph-mode plan whose capacity is not yet
published: a backend built against an unpublished capacity is cached only at
publication, so a failed first plan leaves neither a capacity nor a backend
sized to it.

**Stage 3, the per-step re-plan.** Each step calls `plan()` again with the
new metadata and the arguments of the first plan; there is no separate
update call, as in the MLA wrapper. The re-plan stages into the reserved
storage, preflights against the capacity and rolls back on failure. Nothing
is ranked: the controller narrows the candidate set to the frozen backend.
With a pinned `Resolution` as `backend=` nothing is resolved per step
either; a backend string is resolved again on every plan (about 0.02 ms on
the host). `plan()` raises `RuntimeError` when the current stream is
capturing. After the first graph-mode plan it must run on that plan's
stream, the stream the graph is replayed on, so that the staging copies
precede the replay; on another stream it raises `ValueError`.

**Warm-up before capture.** Every backend needs one eager `run()` after the
plan and before capture for its lazy work (module load, cuDNN graph build).
The cuDNN graph is keyed on `q`'s strides, so each new stride layout of `q`
costs one cuDNN graph build inside `run()`: warm up with the layout that
will be used.
Cake additionally binds `q`, `k_cache` and `v_cache` through TMA descriptors
keyed by storage address, shape and strides, created eagerly and pinned when
capture first replays them; a q/k/v binding that never ran eagerly is refused
inside the capture with `RuntimeError("... prewarm each Cake FMHA
tensor/layout binding before CUDA Graph capture")` (ledger M18). The rule for
cake is therefore: run the exact `q`, `k_cache` and `v_cache` tensors you
will capture once eagerly first; `out` and `lse` are plain pointers and may
differ. The refused capture is clean: no device fault, the instance and the
stream stay usable, and an eager run on those tensors followed by a new
capture succeeds.

**`run()` under a captured graph.** `q`, `out` and `lse` are the capture
buffers. The controller accepts `total_q_tokens <= q.shape[0] <=
capacity.total_q_tokens` and checks `out` and `lse` against `q.shape[0]`
rows. The FA backend launches with the batch's row prefix of the same
storage (`q[:n]`, `out[:n]`, `lse[:n]`), so a captured graph keeps reading
the same pointers. The prefix goes to the launch on either path: the direct
launch of the kernel module from the argument list staged at `plan()`, and
the wrapper's `run()` (sink plans and plans the staging declines), which
insists on `q.shape[0] == qo_indptr[-1]`. The backend's post-processing (the
`v_scale` multiply, the LSE base fold) covers the caller's whole buffers,
because a capture fixes the prefix at the first batch's row count. cuDNN and
trtllm-gen read the row count from `qo_indptr`. `sm_scale`, `k_scale` and
`v_scale` are launch scalars: a captured graph keeps the values it was
captured with
(`tests/experimental/paged_attention/test_paged_attention_coverage.py::test_tc02_graph_bakes_the_captured_sm_scale`).

## Backend selection

### Level 1: `resolve_paged_attention()`

`_selection.resolve_paged_attention` binds the device, validates the static
arguments, normalizes `logits_soft_cap`, and evaluates backends. With
`backend="auto"` it evaluates every declared backend, ordered by
`HEURISTIC_ORDER[cc_major]` first and the remaining names after, so
`Resolution.explain()` carries a reason for each excluded backend. `cake` is
not an auto candidate on any architecture: under `auto` it is excluded with
that reason and runs only when named. With an explicit name it evaluates only
that backend. For each backend it asks the
capability table (`rejection_reason`) and, if admitted, the environment probe
(`PROBES[name](device)`):

- `fa3`: `is_sm90a_supported(device)`; with only `cc_major`, CUDA >= 12.3.
- `cudnn`: the cudnn-frontend package must be importable.
- `trtllm-gen`: no probe; cubin availability is deferred to the first run.
- `cake`: exact compute capability `(10, 0)` or `(10, 3)` when a device is
  known.

The result is a frozen `Resolution(backends, excluded, kv_layout, config)`.
`config` is `resolve_config_key(...)`: the semantic arguments including the
three feature flags and the `max_q_len` hint, and as its last element the
device binding `(cc_major, cc_minor, device_index)`.

`HEURISTIC_ORDER` (`_selection.py`):

| cc major | preference order |
| --- | --- |
| 10 | trtllm-gen, cudnn, fa2 (fa2, trtllm-gen, cudnn with a `max_q_len` hint <= 16) |
| 9 | fa3, fa2, cudnn |
| 8 | fa2, cudnn |
| 12 | fa2, cudnn |

This is only the order of `Resolution.backends` before a batch exists (what
`chosen` and `Resolution.explain()` show). Resolution fixes the candidate
set; `plan()` ranks that set per batch ("Level 2: the auto policy"). The
`max_q_len` hint picks the bucket here and is a pinned promise — `plan()`
rejects a batch, or a graph capacity, above it — but it does not decide
which backend a plan tries first.

### Capability axes

`PagedAttentionCapabilities` (`_backends/_capabilities.py`) declares per
backend: `cc_majors`, `q_dtypes`, `kv_dtypes`, `head_dims` (pairs),
`page_sizes` (`None` = any at or above the global floor), `kv_layouts`,
`supports_lse`, `supports_noncausal`, `supports_window`,
`supports_window_noncausal`, `sinks_head_dims`, `requires_contiguous_q`, `needs_dense`,
`supports_logits_soft_cap`, `supports_custom_mask`,
`supports_sinks`. The rule is capability honesty: an admitted configuration
is one the conformance matrix and fuzzer exercise on hardware.

| backend | cc | head dims | page sizes | layouts | noncausal | window | window + noncausal | contiguous q | kv dtypes | dense table | soft cap | custom mask | sinks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fa2 | 8, 9, 10, 12 | 64, 128, 256, 512 | any | HND, NHD | yes | yes | no | no | f16/bf16, fp8 e4m3 / e5m2 | no | yes | yes | yes |
| fa3 | 9 | 64, 128, 256 | any | HND, NHD | yes | yes | no | no | f16/bf16 | no | yes | no | yes |
| cudnn | 8, 9, 10, 12 | 128, (192, 128) | any | HND, NHD | yes | no | n/a | no | f16/bf16 | yes | no | no | no |
| trtllm-gen | 10 | 64, 128 | 16, 32, 64, 128, 256, 512, 1024 | HND, NHD | yes | yes | no | no | f16/bf16 | yes | no | no | yes |
| cake | 10 | 128 | 16, 32, 64, 128, 256, 512, 1024 | HND, NHD | yes | yes | no | no | f16/bf16 | yes | no | no | yes |

No backend declares `requires_contiguous_q`. `q` reaches the library only at
`run()`, after `plan()` has ranked the candidates, so the ranking cannot
route around a backend that needs a packed `q`: an auto candidate must
declare `False`.

`rejection_reason()` checks, in order: compute capability, q dtype, kv dtype,
the q/kv pair (only an fp8 KV with an f16/bf16 q may differ), head dims, page
size, layout, non-causal, LSE, window, window with non-causal, soft cap,
custom mask, sinks, and finally whether a dense-reading backend would have
to derive its table from flat page ids below `MIN_DENSE_PAGE_SIZE`.

### Level 2: the auto policy

`_auto_policy.py` orders the resolved candidates for one `plan()`. It only
ranks: the capability table and probes own the set, and each backend's
`preflight()` owns eligibility for the batch. A single candidate (an explicit
backend name, or a `Resolution` with one member) is not ranked. `cake` is in
no order, because it is not an auto candidate; a member of a caller's own
`Resolution` that the policy does not rank keeps its place after the ranked
ones (`plan_order`). Nothing is timed.

What it reads (`_PlanFacts`, host values only, built by the controller):

| fact | eager | CUDA-graph mode |
| --- | --- | --- |
| `batch_size` | the batch | the batch (equal to the capacity's) |
| `total_q_tokens`, `max_q_len`, `max_kv_len` | the metadata's values | the capacity's values |
| per-request Q / KV lengths | the metadata's host mirrors | the host mirrors of the batch being planned (a frozen instance does not rank) |
| heads, head dims, page size, layout, dtypes, causal, window, soft cap, custom mask, sinks | `plan()` arguments | `plan()` arguments |

The mirrors are read on every call and never fetched from the device; a
device tensor in their place is treated as absent. A
`PagedAttentionMetadata` always carries both mirrors.

#### The SM100 rule

Only exact SM100, compute capability `(10, 0)`, has a measured rule. Below,
`max_q` and `max_kv` are the `max_q_len` and `max_kv_len` facts: declared
upper bounds, which a caller may set above the batch's longest lengths. The
rule puts one backend first:

1. cuDNN, inside the legacy wrapper's qualified cuDNN region at head dim
   128. The region duplicates the head-dim-128 part of
   `flashinfer.prefill._blackwell_paged_auto_cudnn` and the feature
   conditions around its call. On eager facts the two are equal at head dim
   128, and a CPU test compares them over a grid of facts, which is where a
   change of the legacy predicate shows
   (`tests/experimental/paged_attention/test_paged_attention_auto_policy.py::test_sm100_cudnn_region_against_the_legacy_predicate`).
   The legacy predicate's head-dim-256 domain is not copied (known
   limitation D1).

   Every admission needs all of:
   - causal, with no sliding window, soft cap, custom mask or sinks;
   - HND layout, bf16 `q` and KV, head dim 128, page size 16, 8 to 64 query
     heads, GQA group 1, 2, 4, 8 or 16;
   - `flashinfer.cudnn.prefill._cudnn_supports_paged_auto()`, the gate the
     legacy wrapper asks: the cudnn-frontend FROST runtime with the paged
     packed-split selector, on cuDNN 9.24 or newer. cudnn-frontend 1.32 is
     the package's minimum requirement; the helper does not compare the
     frontend's version. Where the gate does not hold the region is empty
     and cuDNN stays last.

   With Mi for 2^20 and `work` for `num_qo_heads * sum(q_i * kv_i)` over the
   requests, two parts admit:
   - The bounded domain, the legacy wrapper's original one: at most four
     requests whose bounds are a prefix, `64 <= max_q <= 1024`,
     `2048 <= max_kv <= 32768` and `max_kv >= 4 * max_q`, or a full prefill,
     `1024 <= max_q == max_kv <= 8192`, with `work >= 8 Mi`. Equal maxima do
     not show that a batch is a full prefill, so the full-prefill case also
     needs every request's Q length nonzero and equal to its KV length and a
     longest query of at least 1024 tokens.
   - The length-based extension, for any number of requests:
     `work >= 16 Mi` and every `0 < q_i <= kv_i`; then either every
     `q_i == kv_i` with the longest query in 1024..16384, or every
     `q_i > 128` with the longest query at most 8192 and the longest KV in
     2048..32768. One request of at most 128 query tokens keeps a batch that
     is not all full prefill out of the extension, whatever work the other
     requests carry.

   Which values the tests read:
   - The two bound tests of the bounded domain read the declared bounds
     `max_q` / `max_kv`. So does a cheap work bound that every admission
     passes before the gate is asked,
     `total_q_tokens * num_qo_heads * max_kv >= 8 Mi`.
   - The work, the `q_i == kv_i` proof and every test of the extension read
     the per-request lengths of the host mirrors; "longest" in the extension
     is the batch's own. A batch that qualifies for the extension is
     therefore admitted under any bounds declared above its lengths: a
     bucketed `max_kv_len`, or a table sized for a longer context, does not
     put it outside the region. A batch the extension does not admit (less
     than 16 Mi of work, for instance, or a request of at most 128 query
     tokens in a batch that is not all full prefill) needs the bounded
     domain and with it the declared bounds: under a `max_kv` above 32768 it
     is outside the region whatever its lengths.
   - Two host values rule the extension out before a length is read: a
     `max_q` of at most 128, and a `total_q_tokens` that allows neither more
     than 128 tokens for every request nor a 1024-token request next to
     non-empty ones (`total_q_tokens <= 128 * batch_size` and
     `total_q_tokens < 1023 + batch_size`). A plan outside the bounded
     domain is then refused without reading its lengths, a decode batch
     under any declared `max_q`.
   - Facts without host lengths (a device tensor in place of a mirror; a
     `PagedAttentionMetadata` never produces them) admit through the prefix
     test only, for at most four requests and with the cheap bound as the
     only work test. Full prefill and the extension are never assumed from
     the bounds.

   CUDA-graph facts mix two sources:
   - The bounds, `total_q_tokens` included, are the capacity's: the upper
     bounds of a bucket, not one batch's maxima. An inferred capacity's KV
     bound is the table's whole width in tokens, which exceeds a
     full-prefill length that is not a multiple of the page size, so equal
     bounds cannot identify full prefill there. The full-prefill bound test
     in CUDA-graph mode is `1024 <= max_q <= max_kv <= 8192`, still with
     the per-request proof; a capacity that also meets the prefix bounds
     needs no proof.
   - The per-request lengths are those of the batch being planned. A
     graph-mode instance ranks until a plan succeeds, so they are the first
     planned batch's, and that batch alone decides the work, the
     full-prefill proof and the extension. The capacity does not limit the
     extension: a first batch that qualifies freezes cuDNN, planned with the
     capacity's bounds, however far those are above it, and later batches
     are not checked against the region.

   Given the same bounds and lengths, CUDA-graph facts admit everything the
   legacy predicate admits (the same test checks it) and differ from it in
   two places. The legacy predicate requires the query lengths to sum to
   its declared total; a capacity's total is an upper bound, so CUDA-graph
   facts drop that clause. And the wider full-prefill bound test admits a
   full-prefill batch planned under a KV bound above its query bound, which
   the legacy wrapper admits when it derives its bounds from the batch.
   That widening matters only in a band: at most four requests of full
   prefill with `8 Mi <= work < 16 Mi` (8 to 15 query heads; for example, at
   8 heads, a single request of 1024 to 1448 tokens) under a capacity with
   `max_q < max_kv <= 8192`. From 16 Mi the extension admits such a batch by
   its lengths under any capacity, so a 4096-token full prefill in a table
   sized for a longer context is inside the region; a batch of the band
   under a capacity whose KV bound is above 8192 is inside only if the
   capacity meets the prefix bounds.
2. Otherwise trtllm-gen, when the query rows fill enough of the context
   kernel's 128-row tiles: `rows >= W` and `rows / ntg >= F`, with
   `rows = num_qo_heads * total_q_tokens / 128` and
   `ntg = batch_size * num_qo_heads * ceil(max_q / 128)`, the context
   kernel's CTA grid. `W` is 64 at head dim 128 and above and 96 below. `F`
   is 0.375 in CUDA-graph mode and 0.75 eager: between the two the context
   kernel is a few percent ahead, less than its host cost on an eager
   multi-request batch.

   Eager only, and only above 148 rows, trtllm-gen also needs
   `kv_eff >= 1024` or `rows * kv_eff >= 2^19`. `kv_eff` is the most keys one
   query row attends: the batch's longest KV length, capped at
   `window_left + 1` under a sliding window. The length comes from the host
   mirror; the declared `max_kv_len` may be a loose bound (a bucket, the
   table's capacity) and stands in only when the facts carry no host
   lengths. Where a row attends few keys the kernels take tens of
   microseconds and the launch decides, so fa2, the legacy wrapper's choice
   outside the cuDNN region, keeps the batch. Up to 148 rows trtllm-gen keeps
   it: fa2's planner splits the KV range while its CTA grid fits the 148 SMs
   of a B200 and pays a merge kernel that short KV does not earn back. A
   graph replay has no launch cost, and trtllm-gen leads in every measured
   short-KV case there, so CUDA-graph mode has no such exception.
3. Otherwise fa2.

The remaining candidates follow in a fixed order: the other one of
trtllm-gen and fa2 next, and cuDNN last when the batch is outside its
region. Inside the region the order is cuDNN, then the choice of rules 2 and
3, then the other one.

One exception inside the region is fitted here, for eager plans only:
trtllm-gen goes ahead of cuDNN (the order is then trtllm-gen, cuDNN, fa2)
for a small kernel, which is `kv_eff <= 4096`, at most 256 rows in the
longest request (`num_qo_heads * its query length / 128`, the batch's own
lengths when they are on the host) and `rows * kv_eff <= 2^22`. Among the
291 eager shapes of the comparison lists inside the region, the 83 of that
kind are the only cell where trtllm-gen is consistently ahead: its `run()`
takes 0.70 to 1.00 of cuDNN's named through this API (geometric mean 0.86),
and `plan()` costs about 0.05 ms against 0.3 to 0.6 ms. With longer KV
cuDNN's split wins by up to 2.6x; with a longer request or more work the two
are within about 10% either way and cuDNN, the legacy wrapper's choice,
keeps the batch. In CUDA-graph mode the same cell is 0.70 to 1.48 (mean
0.95, 17 of 83 more than 1% slower), so graph plans keep cuDNN first in the
whole region.

#### Other architectures

The other orders are fixed per compute-capability major. SM90 and the other
SM10x parts read `max_q_len` (the metadata's in eager mode, the capacity's
in CUDA-graph mode); SM8x and SM12x read no shape fact:

| architecture | order |
| --- | --- |
| SM90 | fa3, fa2, cudnn; fa2 first at `max_q_len <= 64`, the legacy wrapper's short-query rule. The legacy wrapper's cuDNN admission on H100 is not copied (known limitation D2) |
| SM10x other than `(10, 0)` | trtllm-gen, cudnn, fa2; fa2 first at `max_q_len <= 16`, the `HEURISTIC_ORDER` buckets evaluated on the plan's `max_q_len` |
| SM8x, SM12x | fa2, cudnn |

#### Measurement basis

The SM100 rule is fitted and checked on GB200 only (cudnn-frontend 1.32,
cuDNN 9.27; the fit of rule 2 on 2026-11-10, the check below on
2026-11-18, after the legacy wrapper widened its cuDNN admission). The check compares unified `backend="auto"` with
`BatchPrefillWithPagedKVCacheWrapper(backend="auto")` on identical inputs.
The timed unit is one eager `run()`, or one replay of a graph that captured
one `run()`; `plan()` is outside the timed region. Timing is in paired
blocks. A shape passes when no block is more than 1% slower than the legacy
time, is a confirmed regression when every block is, and is inconclusive
otherwise.

| shapes | mode | geomean speedup | pass | inconclusive | confirmed regression |
| --- | --- | --- | --- | --- | --- |
| main list, 219 | CUDA graph | 1.49x | 163 | 47 | 9 |
| main list, 219 | eager | 1.41x | 205 | 14 | 0 |
| all eight lists, 661 | CUDA graph | 1.39x | 494 | 151 | 16 |
| all eight lists, 661 | eager | 1.20x | 575 | 85 | 1 |

- The CUDA-graph regressions all run cuDNN under both APIs, so the order is
  not their cause; they take at most 6.8% and at most 21 us longer. They
  are an effect of the measurement, not of the API: with the legacy wrapper
  in both measured slots (34 same-backend shapes, 24 independent processes
  each, 2026-11-10) the second slot is between 0.967 and 1.025 of the first
  per shape and about 3% of the shapes of a run are flagged the same way;
  legacy against unified gave 0.975 to 1.017, mean 1.002, and 2%. The
  eager regression is one head-dim-64 shape (B3, q 128, kv 2048, 64/8) on
  which the rule picks trtllm-gen and the legacy wrapper fa2: 10% longer in
  this run, 4% shorter in the run before.
- Five lists sit on the rule's thresholds on purpose (the sliding-window,
  short-KV, row-bound, 128-to-160-row and threshold-fit lists, 212 shapes).
  One holds 176 shapes inside the legacy wrapper's widened cuDNN admission
  (1.07x eager, 1.05x CUDA graph: cuDNN under both APIs except the eager
  small kernels, whose 83 shapes over all lists take 0.67 to 0.97 of the
  legacy time, geometric mean 0.83, and all pass) and one 54 shapes at head
  dim 256 (fa2 under both APIs, 1.00x). The main list is the representative
  one.
- Fitted on GB200: the row and fill thresholds of rule 2, the eager
  short-KV exception and the eager small-kernel exception inside the cuDNN
  region. The tables are next to the constants in `_auto_policy.py`.
- Not refitted: the cuDNN region, which is a copy of the legacy wrapper's
  head-dim-128 predicate, the bounded domain and the length-based extension
  alike. Only its CUDA-graph form is this package's own: the wider
  full-prefill bound test and the dropped total clause.
- Not measured through this API: the SM90, other SM10x, SM8x and SM12x
  orders. CPU tests pin them; see "Known limitations", which also records
  that the SM90 order does not take the legacy wrapper's H100 cuDNN
  admission (D2).

#### When a ranking applies

Eager instances rank on every `plan()`, so the backend can change between
batches. A graph-mode instance ranks on the capacity's bounds and the
planned batch's host lengths until a plan succeeds: the first successful
plan freezes its backend, and later plans keep it without consulting the
policy.

The row and fill rule reads only the capacity. The cuDNN region also reads
the first plan's per-request lengths (the exact work, the Q == KV proof and
every test of its length-based extension), so two instances with one
capacity can freeze different backends, and a first batch inside the
extension puts cuDNN first under any capacity that holds it. A capacity of
one request with 128 query tokens over 16384 keys (32 query and 8 KV heads)
freezes cuDNN when the first batch fills it and fa2 when the first batch is
a 4-token warm-up over 64 keys. Plan the first batch with representative
lengths, or pin the backend: pass its name (`backend="fa2"`), or a
`Resolution` resolved with that name.

The policy ranks from `plan()` facts and cannot see what arrives at `run()`.
The `run()` tensor requirements are those of the backend planned for the
batch (see "`run()` tensors"), so under eager `auto` they can change from
batch to batch. fa3, trtllm-gen and cake read the V cache with the K cache's
strides and reject a `(k_cache, v_cache)` pair whose strides differ; fa2 and
cuDNN walk each pool by its own strides. Two separately allocated contiguous
pools of one shape have equal strides and pass. An engine whose pools differ
in strides (for example V as a view of a wider buffer next to a packed K)
gets that `ValueError` for the batches `auto` plans on trtllm-gen (SM10x) or
fa3 (SM90) and not for the others: give both pools one layout, or pin a
backend. The layout of `q` is not such a requirement: the controller checks
one rule for every backend (unit inner stride, token-major storage, 16-byte
alignment; see "`run()` tensors"), and every backend takes any view that
meets it.

### Level 2: the candidate walk

In `plan()`, the controller walks the candidates in the policy's order. A
candidate may decline the specific batch from `preflight(meta)` with
`_BackendPlanUnsupportedError`, a private `ValueError` subclass defined in
`_capabilities.py`. The MLA package's private type of the same name has the
same base class; there the typed error itself reaches the caller, here the
controller converts it (below). Preflight is allocation-free and runs before
any live-state write, so declining is free of side effects. The walk
finishes before any reserved buffer is written in graph mode. The
controller's handler names the typed class, so every other exception
propagates unchanged, a plain `ValueError` for invalid input included;
`run()` never falls back. With an explicit backend name the typed signal
becomes a plain `ValueError` `"backend 'x' cannot plan this batch: ..."`. If
every candidate declines, the controller raises `ValueError`
`"no pinned candidate can plan this batch"` listing each reason.

Current preflight rules:

- `fa2`/`fa3` (`_FaBackend.preflight`): `fa3` requires SM90a and CUDA >= 12.3
  on the instance's device; `fa2` declines a batch whose split-KV planner
  allocation does not fit the scratch workspace (see "Workspace"); with
  `use_sinks=True`, a soft cap (the AttentionSink variant has no soft-cap
  hook) and a custom mask are declined as unverified (an fp8 KV cache with
  sinks is verified — ledger M22 — with `k_scale` folded into the launch
  scale and `v_scale` applied to the whole output buffer). A custom mask in
  graph mode raises `ValueError` (not the typed signal, since no other
  backend takes custom masks): the wrapper's packed-mask storage is not part
  of the capacity yet.
- `cudnn` (`_CudnnBackend.preflight`): the cudnn-frontend package must be
  importable.
- `trtllm-gen`/`cake` (`_TrtllmGenBackend.preflight`): the page size must be
  one the paged context kernel ships for; non-causal with a sliding window has
  no kernel; a non-contiguous dense block table is declined (see the input
  contract); an LSE plan whose softmax-stats scratch does not fit the
  workspace is declined (see "Workspace"); `cake` declines a batch with a
  `kv_len == 0` request (ledger M19).

### Level 2: measured selection (`autotune`)

`backend="autotune"` keeps the candidate set of `auto` and replaces the
ranking by a measurement. It follows `_BatchMLAPagedAttentionAutotuneBackend`
of the MLA wrapper: the plan prepares every eligible candidate, the first
`run()` selects one from the tuner cache or by profiling the current
workload, the selection lasts until the next plan, and a workload that was
never tuned is an error outside `flashinfer.autotune(True)`.
`resolve_paged_attention(backend="autotune")` returns the `auto` set with
`Resolution.autotune=True`; the flag is a selection policy, not
configuration, so it is not part of `config`.

In `plan()`, with more than one candidate and no frozen backend, the walk
does not stop at the first acceptance: every candidate that accepts the batch
is planned, in the auto policy's order. In graph mode they share one staging
transaction, and the union of their derived forms is staged once into the
reserved storage. The set is kept as the open selection, with its workload
key. `backend` reports `autotune` for the whole plan, before and after the
selection, and `explain()` names the selected candidate. A plan that ends with one accepted candidate has nothing to
select and behaves as `auto`.

The workload key (`autotune_workload`) identifies the planned workload; no
dimension is bucketed. It holds the resolved configuration with the compute
capability, the mode, the bounds the kernels are planned with (the capacity
in graph mode), a SHA-256 digest of the batch's metadata tensors
(`metadata_digest`: the query offsets, the KV lengths and the page ids, as
MLA digests its metadata snapshot), the workspace size, the device and the
names of the prepared candidates. The lengths come from the host arrays; a
device page table is read back, so an autotune plan synchronizes.

Every `run()` of an autotune plan requires `out`. The first one settles the
selection after its argument checks (`_settle_autotune` calling
`select_planned`):

1. Under CUDA graph capture it raises `RuntimeError`: the selection must be
   prepared before capture.
2. The cache is searched with the workload key and the run signature (the
   shapes, strides, dtypes, devices and alignment of q, K, V, `out`, `lse`
   and sinks, and the host scales). A hit selects without launching anything
   extra.
3. On a miss outside `flashinfer.autotune(True)` it raises `RuntimeError`
   asking for a warm-up.
4. On a miss inside the context each candidate is launched once outside the
   tuner, so a real failure raises instead of being scored as a slow tactic,
   and `AutoTuner.choose_one` profiles them with one runner per candidate.
   Profiling writes private `out` / `lse` buffers; the caller's buffers are
   written once, by the selected backend. If the tuner recorded no result
   (the op is in `skip_ops`, or profiling failed) it raises `RuntimeError`.

The selected candidate becomes the active backend until the next plan, and
in graph mode it replaces the provisional head in the frozen contract, so
a re-plan plans it alone. A graph-mode re-plan is refused while the
selection is open. An error in the first `run()` leaves the selection open.

### `explain()`

`PagedAttention.explain()` raises `ValueError` before the first successful
plan. Afterwards it prints the chosen backend, the candidates that plan could
choose from in the order it walked them, the plan-time trace (one line per
candidate tried: `name [preflight]: accepted` or the decline reason) and the
`Resolution.explain()` text (the device the resolution answers for, the
candidates in resolve-time order, and each exclusion reason). The trace and the order are also available as
`PagedAttentionController.selection_trace` and `.selection_order`.

The walked order is the auto policy's order for an eager plan and for a
graph-mode plan made before a backend is frozen. Once a graph-mode backend
is frozen nothing is ranked and the order is the frozen backend alone: the
controller narrows the `Resolution` to the frozen backend before the walk
(`_narrowed_to`), so the trace holds only that backend's entry, no other
member is constructed or asked, and the `Resolution.explain()` text lists
the other members among its exclusions ("not the backend the first
graph-mode plan froze"). A re-plan with a `Resolution` that does not contain
the frozen backend raises `ValueError`. The order the freezing plan ranked
is not kept.

## Planning contract

### Derived forms and per-backend needs

`_planning.DERIVED_FORMS` names what a backend may request: `q_seq_lens`,
`cum_kv_seq_lens`, `kv_page_indptr`, `kv_page_indices`, `block_tables`,
`host_arrays`. Each backend class declares `DERIVED_NEEDS`:

| backend | `DERIVED_NEEDS` | native dialect |
| --- | --- | --- |
| fa2 / fa3 | `kv_page_indices`, `host_arrays` | CSR page ids on device; `qo_indptr`, page indptr, last-page lengths and KV lengths as pinned host arrays for the wrapper's CPU split-KV scheduler |
| cudnn | `q_seq_lens`, `block_tables` | token-unit `batch_offsets_q`, `actual_seq_lens_{q,kv}` as `(b, 1, 1, 1)`, a dense table of exactly the declared width (eager: built by the backend from the plan's table; graph mode: the reserved one, which has the capacity's width) |
| trtllm-gen / cake | `cum_kv_seq_lens`, `block_tables` | the canonical form natively plus cumulative KV lengths |

`Derived.require(name)` raises at the use site if a backend reads a form it
did not declare.

### Host staging

`HostArrays` (`_planning.py`) holds every host-side length array of one
batch in one pinned int32 tensor with the layout
`qo_indptr (b+1) | q_seq_lens (b) | cum_kv_seq_lens (b+1) | kv_page_indptr (b+1) | pages (b) | kv_last_page_len (b) | kv_seq_lens (b)`.
It is filled with numpy from the mirrors at metadata construction, read by
value validation and the causal-envelope check, and uploaded with one
`copy_(non_blocking=True)` the first time a plan needs a device form; the
device forms are slices of that one buffer. The tensor is pinned when CUDA is
available and pageable otherwise (host-only tests). Its lifetime is the
metadata object's.

`derive()` performs no host sync and computes nothing outside `needs`. The
only device work is the two cross-form conversions:

- dense given, flat needed (fa backends): a capacity scatter into a
  `b * width` buffer (a masked select would sync to size its result);
- flat given, dense needed (cuDNN, trtllm-gen, cake): a gather of width
  `ceil(max_kv_len / page_size)` with each row's tail clamped to the request's
  own last page. The clamp is load-bearing: cuDNN gathers K/V pages by table
  width before masking, so a tail pointing into over-allocated or
  uninitialized page ids produced NaN outputs (fuzzer mutation
  `csr_overallocated_nan_tail`). An all-padding batch gathers nothing.

The mirrors are trusted to match the device tensors; validating equality
would cost the sync this path removes. With the mirrors supplied, the eager
plan on trtllm-gen with a dense table issues zero kernel launches and one
pinned host-to-device copy (`benchmarks/bench_paged_attention_plan.py`
counts the launches and copies of a plan). The eager cuDNN plan adds the
five small kernels that build its own page table (see "Page-table ABI per
backend"). No warm plan uploads from pageable memory or
copies from the device to the host
(`tests/experimental/paged_attention/test_paged_attention_prototype.py::test_warm_plan_uploads_from_pinned_memory_only`).

### Validation

`validate_structure` (host-only, shape-derived): every metadata tensor is an
int32 CUDA tensor on the metadata's device with the right rank; `qo_indptr`
has `batch_size + 1` entries; `block_tables` has `batch_size` rows; the page
size is a positive host int and respects the dense floor; `max_q_len` and
`max_kv_len` are positive host ints (a Python `int`; a `bool` and a numpy
integer are rejected); `max_kv_len` fits the table capacity.

`validate_values` (always runs, from the mirrors): `qo_indptr` is
non-decreasing (`q_len == 0` is a padding row) and the batch holds at least
one query token; `qo_indptr[0] == 0`; the longest query is at most `max_q_len`;
`kv_seq_lens >= 0`; the longest KV is at most `max_kv_len` and the table
capacity; the flat page-id list covers `sum(ceil(kv_len / page_size))`
entries. `validate_causal_envelope` (only with `causal=True`) requires
`q_len <= kv_len` for every request with `kv_len > 0`.

## Input contract

### Metadata

Dense form: `block_tables (b, max_pages)`, `page_size >= 8`. Flat form:
`kv_page_indices` in request order, any `page_size >= 1`; the page-unit
indptr and last-page lengths are not accepted, because they derive from
`kv_seq_lens` and `page_size` and a second copy would be a second truth.
`block_tables` values (page ids) are trusted to be in-pool.

### `run()` tensors

The controller checks (`_controller.run`):

- `q`: 3-D `(rows, num_qo_heads, head_dim_qk)`, planned dtype,
  `q.stride(-1) == 1` for every backend (the FA and trtllm-gen bindings pass
  only the token and head strides, so a non-unit inner stride is silently
  misread; cuDNN rejects it). Eager: `rows == total_q_tokens`; graph mode:
  bounded as above. `q` must also be stored token-major and 16-byte
  aligned, for every backend: head stride `>= head_dim_qk`, token stride
  `>= (num_qo_heads - 1) * head stride + head_dim_qk`, both strides
  multiples of 16 bytes (8 elements of an f16/bf16 `q`), and the data
  pointer on a 16-byte boundary. Anything else raises `ValueError`: a
  `(H, T, D)` buffer viewed as `(T, H, D)` and an expanded (zero-stride)
  dim are misread by trtllm-gen, and an unaligned offset or stride is
  refused at launch by trtllm-gen and cuDNN and faults on the device in
  fa2. The alignment rule was measured for bf16, head dim 128 on SM100.
  Strides that are positive but overlap (below the two bounds) are refused
  as well, although fa2, cuDNN and trtllm-gen read them correctly when
  measured. Every backend accepts any view that meets the rule (a
  fused-QKV head slice, a padded head or token stride, an aligned storage
  offset), measured natively in
  `tests/experimental/paged_attention/test_paged_attention_strides.py`. cuDNN
  declares `q` to its graph with `q`'s own strides and scales the token-unit
  ragged offsets by `q`'s token stride, so a head slice of a fused QKV
  buffer is addressed correctly; its graph is keyed on those strides, so
  each new stride layout costs one graph build inside `run()`. The
  controller keeps a guard for a backend that declares
  `requires_contiguous_q` (a `ValueError` naming the backends that take a
  strided `q`); no backend declares it.
- `kv_cache`: a `(k_cache, v_cache)` pair, each 4-D in the planned layout
  (`HND`: `(pages, H, page_size, D)`; `NHD`: `(pages, page_size, H, D)`), with
  the planned `kv_dtype` and a unit stride along the head dim. A transposed
  pair gets a hint naming the layout. fa3, trtllm-gen and cake read V with
  K's strides and reject a pair whose strides differ; fa2 and cuDNN walk each
  pool by its own strides.
- `out`: contiguous `(q.shape[0], num_qo_heads, head_dim_vo)` in q's dtype.
- `lse`: contiguous fp32 `(q.shape[0], num_qo_heads)`; requires
  `lse_mode != "none"`.
- Every tensor on the instance's device.
- `sm_scale`: positive finite host float, default `1 / sqrt(head_dim_qk)`.
- `k_scale`, `v_scale`: positive finite host floats, with any `kv_dtype`
  (dequantization scales of an fp8 cache, multipliers of a float cache; see
  "K / V scales" under Feature axes).
- `sinks`: required iff the plan declared `use_sinks=True`; contiguous fp32
  `(num_qo_heads,)` on q's device.

Nothing is copied to satisfy a backend; an unsupported layout is rejected
with the constraint named.

### Page-table ABI per backend

- **cuDNN** (`_CudnnBackend.plan`): cuDNN requires the page dimension of the
  table it binds to equal `ceil(declared KV bound / page_size)` exactly, or
  it fails at finalize. The prefill graph cache key in
  `flashinfer/cudnn/prefill.py` includes the table's shape and strides, the
  q/k/v strides and dtypes and the softmax scale, and the cache decorator sits
  outside the build step so a failed build inserts nothing. What the plan
  declares and which table it binds depend on the mode:
  - Eager: the plan declares a shape class, not the batch's exact maxes
    (`_declared_bounds`). cuDNN finalizes one execution plan per declared
    `(max_q, max_kv)`, about 5 ms each, and the longest KV of a serving loop
    grows every step, so exact maxes would build a plan on every step. KV
    rounds up to the next power of two. A longest query above 64 tokens,
    with a GQA group that divides 128, rounds up to whole
    `128 // group`-token query tiles, at least 128; up to 64 tokens the Q
    bound stays exact, because the `max_q_len == 1` LSE layout and cuDNN's
    single-token paths depend on the exact value. These are the formulas
    the legacy wrapper applies where its `auto` selects cuDNN on compute
    capability `(10, 0)` or `(10, 3)`; this backend applies them on every
    architecture and for a named cuDNN backend as well (known limitation
    D5). The per-request lengths stay the actual ones and mask the live
    tokens.

    The class reaches past the table columns the caller filled for the
    batch, and one cuDNN engine loads pages through the columns past a
    request's own pages: it fetches whole 128-token KV tiles and multiplies
    the masked probabilities with what it fetched, so a pool page holding
    NaN reaches the output as `0 * NaN`. The backend therefore binds a table
    of its own (`_own_pages_table`): a new contiguous
    `(batch, ceil(class / page_size))` tensor that holds each request's own
    pages followed by its last live page again. It is built on the device at
    `plan()` without a host sync, at the cost of five small kernels and
    about 0.1 ms of host time per eager plan. The plan's table, the
    caller's or the one derived from flat page ids, is only read: it needs a
    unit stride along the page dimension and at least
    `ceil(max_kv_len / page_size)` columns, and its width and row stride are
    otherwise free, so every source reaches one cuDNN graph per (batch,
    class).
  - CUDA-graph mode: the plan declares the capacity's bounds, because a
    captured graph serves every later batch of its bucket. It binds the
    controller's reserved table, which is exactly
    `ceil(capacity.max_kv_len / page_size)` columns wide in both paging
    forms, without a copy, so a capture keeps its pointer and row stride. In the
    dense form the columns the caller left past a request's own pages
    therefore do reach cuDNN, and where the capacity's KV bound is not a
    whole number of 128-token tiles the same engine also loads pool page 0
    for the rest of the last tile; see "Known limitations".
- **trtllm-gen / cake** (`_TrtllmGenBackend.preflight`): the launcher takes
  the row stride from `block_tables.size(-1)` and the kernel walks a raw int32
  pointer, so a narrow view of a wider table is read as a packed array and is
  silently wrong. A non-contiguous table is declined with the typed signal;
  under `auto` the walk moves on to the next candidate, an explicit backend
  surfaces a `ValueError` with the `.contiguous()` hint. Extra columns past
  `max_kv_len` are fine. The check
  runs on the table the backend is handed: the caller's in eager mode, the
  reserved contiguous one in graph mode (the caller's table is copied into
  it at staging), so a narrow view is declined eagerly and accepted under a
  graph (measured on B200, 2026-09-21).
- **fa2 / fa3** (`_FaBackend.plan`): the flat `kv_page_indices` must be
  contiguous (the kernel walks it as a packed int32 array); only the eager
  flat form can fail this, since the reserved graph buffer is contiguous.

### Zero-row contract

`kv_seq_lens[i] == 0` is legal and marks a padding row (vLLM's CUDA-graph
padding fills `seq_lens` with 0 and the table row with its null block;
SGLang's fill value 1 is an ordinary live row). The library guarantees that no page of that row
is read and that every other row is unchanged; the row's output and LSE are
unspecified by contract and may be left unwritten, so an engine must never
read a padding row. fa2/fa3 and cuDNN write a zero output row and an LSE of
`-inf`, trtllm-gen leaves the rows untouched (measured with a NaN-poisoned
output allocation), cake declines the batch; measured on
B200 with the native calls. Implementation: value validation
rejects only negative lengths; the causal envelope exempts padding rows; the
derived last-page length of a padding row is `page_size` by the
`kv - (pages - 1) * page_size` formula (the FA kernel never reads it, since
its indptr span is empty); the flat-to-dense gather clamps a page-less row to
a live in-pool id that no kernel reads for that row. Padding rows can be
toggled live/padding across graph re-plans
(`tests/experimental/paged_attention/test_paged_attention_cuda_graph.py::test_replan_toggles_padding_rows`).

`q_len == 0` rows (`qo_indptr[i] == qo_indptr[i + 1]`, vLLM's padded
`query_start_loc` tail) are legal as well (ledger M17). Such a row owns no
query token and no output row, so there is nothing unspecified about it; the
contract is that every other row is unchanged. Measured on B200 with the
native calls: fa2 (CSR planner), cuDNN (`actual_seq_lens_q = 0`
with a repeated `batch_offsets_q`), trtllm-gen and cake (a repeated
`cum_seq_lens_q` value) all accept a q_len 0 row in the middle or at the tail,
with `kv_len > 0` or `kv_len == 0`, and leave the other rows exact; cake's
`kv_len == 0` hang (M19) is the only exception and its preflight declines
that batch. `qo_indptr` must be non-decreasing and the batch must hold at
least one query token. One backend-side consequence: cuDNN's `max_q_len == 1`
LSE shortcut (the padded `(b, 1, h)` stats read as the packed `(b, h)`
buffer) holds only while every request has exactly one token, so it is taken
in eager mode only and only when `total_q_tokens == batch_size`; graph-mode
decode buckets use the gather path
(`tests/experimental/paged_attention/test_paged_attention_cuda_graph.py::test_decode_bucket_replan_with_zero_q_tail_rows`).

## Output and LSE contract

`lse_mode` is one of `"none"`, `"base2"`, `"basee"` (`_contracts.LSE_MODES`).
The returned LSE is packed fp32 `(total_q_tokens, num_qo_heads)` in the
declared base for every backend. With `use_sinks=True` the LSE includes the
sink logit.

| backend | native LSE | fold to `base2` | fold to `basee` |
| --- | --- | --- | --- |
| fa2 / fa3 | base 2, packed | none | `lse.mul_(ln 2)` in `_FaBackend.run` |
| trtllm-gen / cake | base 2, packed | none | `lse.mul_(ln 2)` in `_TrtllmGenBackend.run` |
| cudnn | natural log, padded `(b, max_q, h)` or packed | library `lse_base="2"` | library `lse_base="e"` (free) |

cuDNN writes the packed `(tokens, h)` stats straight into the caller's buffer
through token-unit `batch_offsets_stats` when a one-time per-device feature
probe (`_packed_lse_supported`, a graph build and execute on a toy problem,
no host sync) succeeds (`lse_path == "direct"`). The probe runs from cuDNN
9.25; on 9.24 a plan is staged and the LSE is gathered. At
`max_q_len == 1` the padded `(b, 1, h)` layout is byte-identical to the
packed one and is used as a view (`"view"`), because cuDNN 9.25 writes no
stats there when a ragged stats offset is set. Otherwise the backend gathers
padded native stats with plan-time precomputed indices (`"gather"`); the
padded buffer has the Q bound declared to cuDNN (a shape class in eager
mode), and in graph mode the indices keep the capacity's row count.

## Feature axes

All of these are `plan()` arguments, capability-checked at resolve and plan
time, and frozen by the first graph-mode plan:

- `causal` (default `True`): also enforces the causal envelope. Every backend
  declares non-causal support; trtllm-gen's was measured on B200
  (`_capabilities.py` comment; `test(paged_attention): trtllm-gen non-causal
  capability by measurement`).
- `window_left` (default `-1` = unlimited; values below `-1` are rejected
  because backends disagree on their meaning): plan-time because it selects a
  compiled kernel variant on the FA backends. This API's cuDNN backend
  declares no window; the cuDNN graph path has one, and the legacy wrapper's
  explicit cuDNN backend passes it (known limitation D3). Non-causal plus
  window therefore has no backend here: trtllm-gen and cake ship no such
  kernel, and fa2 and fa3 declare it unsupported (ledger M20).
- `logits_soft_cap` (`None` or `0` = off; otherwise a positive finite float):
  `cap * tanh(score / cap)` on the scaled scores. fa2 and fa3 only, through
  the wrapper's soft-cap kernel variant.
- `custom_mask`: a contiguous 1-D bool tensor on the instance's device, the
  per-request `(q_len_i, kv_len_i)` masks flattened row-major and
  concatenated in request order (`sum(q_len_i * kv_len_i)` elements, `True` =
  may attend). fa2 only. The FA backend ANDs it with the causal / window
  envelope (`_envelope_mask`) because `MaskMode.CUSTOM` replaces the kernel's
  causal mask, so a mask never widens the envelope. Not available in graph
  mode.
  *Packed masks (design note, WP-T).* The legacy wrapper also takes a
  pre-packed `packed_custom_mask`: a uint8 tensor produced by
  `segment_packbits(mask, mask_indptr, bitorder="little")`, i.e. each
  request's row-major `(q_len_i, kv_len_i)` mask packed separately, eight
  positions per byte, least significant bit first, every request padded to a
  byte boundary, addressed by a `mask_indptr` the wrapper derives from the
  page metadata. The unified API does not accept it, for three reasons. (1)
  The contract ANDs the mask with the causal / window envelope; for a packed
  input the FA adapter would have to unpack it on the device, AND, and hand
  the bool mask to the wrapper, which packs it again with the same
  `segment_packbits` kernel — the caller saves nothing and the plan gains
  two passes. (2) A pass-through that skips the repack is correct only with
  `causal=False, window_left=-1` and would put the legacy encoding (bit
  order, per-request byte alignment, a second indptr) into the public
  contract as an adapter detail, which this design forbids. (3) Custom masks
  are not in the CUDA-graph capacity yet; a second representation before the
  mask storage joins `GraphCapacity` would have to be revisited then. A
  caller holding a legacy packed mask unpacks it once on the device
  (`(packed.unsqueeze(-1) >> torch.arange(8)) & 1`, per request, trimmed to
  `q_len_i * kv_len_i`) and passes the bool mask. Revisit together with the
  graph-mode mask storage; the pass-through variant is the candidate if a
  consumer shows the repack on its plan path.
- `use_sinks` / `run(sinks=)`: per-head attention sinks, one extra softmax
  denominator logit per head with no value contribution. fa2 and fa3 select
  the `AttentionSink` JIT variant wrapper at plan time (the default wrapper
  forwards `sinks` only on its trtllm-gen path, and its standard fa2 / fa3
  adapter refuses a sink tensor with `NotImplementedError`); trtllm-gen and
  cake pass the tensor natively. The pair mirrors the MLA wrapper's
  `use_sinks` / `sinks`.
- fp8 KV cache: `kv_dtype=torch.float8_e4m3fn` or `torch.float8_e5m2` with
  an f16/bf16 q, declared for fa2 only (e5m2 measured on B200 against the
  oracle on the dequantized cache, same error magnitude as e4m3; see the
  `_capabilities.py` comment). `k_scale` / `v_scale` are per-tensor host
  floats (`dequant = fp8_value * scale`); omitted scales mean no scaling.
  fp8 q and nvfp4 are undeclared axes (the legacy wrapper's cuDNN backend
  takes an fp8 q; known limitation D4).
- K / V scales on a fp16 / bf16 cache: the same `run(k_scale=, v_scale=)`
  multiply K and V of a float cache too (the legacy wrapper's
  `test_kv_scale_forwarding_*` semantics), on every backend and with one
  rule: `k_scale` folds into the softmax scale the kernel launches with, so
  the soft cap, the sinks and the returned LSE see the scaled logits exactly
  as `K * k_scale` would give, and `v_scale` multiplies the output (the
  whole caller buffer, so a captured graph scales every row the capacity may
  hold). fa2/fa3 fold in the adapter (the AttentionSink variant's `run()`
  has no `k_scale` argument), trtllm-gen and cake pass `bmm1 = sm_scale *
  k_scale` / `bmm2 = v_scale` natively, and cuDNN folds `k_scale` into
  `attn_scale` and multiplies the output because its f16/bf16 SDPA graph has
  no descale tensors (passing them is silently ignored; measured on B200).
  Measured on B200 for fa2, cuDNN, trtllm-gen and cake with k 0.5 / v 2.0 in
  fp16 and bf16 against the oracle on the scaled K / V (max out err 1.0e-2,
  bf16; 1.3e-3, fp16).

`sm_scale` is a `run()` argument so one plan serves layers with different
scales.

## Workspace

Every backend runs on one plain scratch buffer, the role of the legacy
wrappers' `float_workspace_buffer` (`_controller.py`, top of file). By
default all instances on a device share one lazily allocated library-owned
pool of `_WORKSPACE_BYTES = 512 MiB` (the upper end of what the engines
allocate for the same kernels), so one instance per graph bucket costs no
workspace per bucket. A caller may pass `workspace_buffer=`: a contiguous
1-D `uint8` or `int8` tensor on the instance's device, viewed as `uint8`. The
generated-FA wrapper receives it as its float workspace; cuDNN views it as
`int8`; trtllm-gen and cake use it as their softmax-stats scratch and keep a
separate small zero-initialized multi-CTA counter buffer of their own.
Instances sharing a workspace must not run concurrently on different streams;
pass a private buffer where that isolation is needed.

The fa2 split-KV planner is the only consumer whose need grows with the
geometry (ledger M14). trtllm-gen and cake need the softmax-stats scratch of
an LSE plan, and cuDNN sizes its own graph workspace.
`PagedAttention.workspace_requirements(capacity, ...)` returns a conservative
upper bound in bytes for every plan within a `GraphCapacity`: the maximum
over the backends that resolve for the configuration, or over the one named
or the pinned `Resolution`. It is tensor-free and sync-free, for sizing a
caller-owned buffer at engine init.

A backend whose scratch does not fit the buffer declines the batch in its
`preflight()`, before any kernel-side allocator error. fa2 replays its
planner's arithmetic on the host mirrors (`workspace_need`, a lower bound on
the planner's allocation, so the decline is never spurious); trtllm-gen and
cake compute the softmax-stats bytes. The decline is the typed signal: under
`auto` the walk moves on to the next candidate and the byte counts stay in
the selection trace, and a named backend, or a set in which no candidate
fits, raises `ValueError` naming the required and the available bytes. For
a named backend the message reads `"backend 'fa2' cannot plan this batch:
needs N bytes of scratch workspace for this batch but the workspace buffer
holds M bytes; ..."` (`_workspace_too_small` gives the reason, the
controller adds the backend's name as for every preflight reason).

## Benchmark and API records

### Benchmark

`benchmarks/bench_paged_attention.py` (a standalone script; the API is
experimental, so it is not a routine of the shared benchmark CLI) times one set of inputs for every
requested backend (`fa2`, `fa3`, `cudnn`, `trtllm-gen`, `cake`, `auto`),
gates each on the fp32 oracle before timing, and emits one CSV row per
(backend, `api_variant`, phase): `plan` (metadata construction plus `plan()`,
synchronized host wall; a graph-mode re-plan when CUDA graphs are on), `run`
(warmed `run()` with preallocated buffers, GPU time: by default one `run()`
on the case's own buffers captured once and its replay timed with CUDA events
and an L2 flush before every replay — bindings fixed, so cake's M18 rule holds
for every backend, instead of the timing helper's rotating clones), `step` (one plan
plus N eager runs). Rows for unsupported, erroring or incorrect candidates
are kept with a `status` column; the typed preflight signal, alone or chained
under the candidate walk's `ValueError`, is classified as `unsupported`.
`auto` rows record the first candidate of the resolve-time order
(`static_backend`) and the backend the plan ran (`resolved_backend`); the two
differ when the plan-time policy ranks another candidate first or the first
one declines. `--pa_legacy` adds rows through the legacy public API of the
same kernel; its cuDNN provider hands the native API the width-exact column
view of the case's wider block table with the actual `max_kv_len` (the
unified cuDNN backend declares other bounds: its capacity's in CUDA-graph
mode, here the table's full width, and a shape class with a table of its
own in an eager plan).
`benchmarks/bench_paged_attention_plan.py` is the micro-benchmark for
metadata construction, eager plan and graph re-plan cost, with launch and
memcpy counts from the profiler.

### Trace

`PagedAttention.run` has no fi_trace template. Trace templates are optional
for an experimental API and part of the graduation checklist
(`flashinfer/experimental/README.md`), together with the
`tests/trace/example.py` entry.

### Records of the inner fa wrapper

The API log and the dump belong to the unified entry,
`PagedAttention.run`. For a plain fa2 / fa3 plan, `run()` launches the
kernel module from the argument list staged at `plan()` and does not enter
`BatchPrefillWithPagedKVCacheWrapper.run()`, so a unified fa run does not
produce that method's own records:

- its `FLASHINFER_LOGLEVEL` entry and its level-10 dump;
- the `gqa_paged_prefill_*` trace dump under `FLASHINFER_TRACE_DUMP`;
- Trace Apply under that definition name.

A consumer that keyed on those records uses the `PagedAttention.run` entry;
there is no unified trace definition yet. A sink plan and a plan the staging
declines run through the wrapper's `run()` and produce its records.

## Correspondence to the Batch MLA package

| Batch MLA (`flashinfer/mla/_batch_mla/`) | Paged attention (`flashinfer/experimental/paged_attention/`) |
| --- | --- |
| `flashinfer.mla` facade, `_core.py` | `flashinfer.prefill` re-exports, `flashinfer/_paged_attention.py` |
| `_wrapper.py` `BatchMLAPagedAttentionWrapper` | `_controller.py` `PagedAttentionController` behind `PagedAttention` |
| `MLAPlanMetadata.csr()` / `.dense()` / `.dual()` | `PagedAttentionMetadata.dense()` / `.csr()` (no dual form: one truth) |
| `MLAInputContract` | `PlanMetadata` plus the frozen contract dict in graph mode |
| `_MLAPlanArguments` and its lazy resolver | `PlanMetadata`, `Derived`, `HostArrays` (`_planning.py`) |
| `_auto_policy.py`: `backend="auto"` is resolved in `plan()`; per-architecture `ordered_*_backends` over the plan request; `_BatchMLAPagedAttentionAutoBackend.plan_from_wrapper` walks that order and moves on only on `_BackendPlanUnsupportedError`; a graph re-plan keeps the previous backend; the module also owns the `autotune` policy | `_auto_policy.py` only ranks (`plan_order`, per-architecture `ordered_*_backends` over `_PlanFacts`); `resolve_paged_attention()` fixes the candidate set at init (`cake` excluded under `auto`) and the controller walks it in `plan()`, moving on only on `_BackendPlanUnsupportedError`; a graph-mode instance keeps the first successful plan's backend; the module also owns the measured selection of `backend="autotune"` (`select_planned`), which the controller calls at the first `run()` |
| `_backends/_capabilities.py`, `_BackendPlanUnsupportedError` (a `ValueError` subclass that reaches the caller) | same module name, same-named private type with the same base class; the controller converts it to a plain `ValueError` that names the backend or lists every candidate's reason (to be unified) |
| `_fa_common.py` + `fa2_backend.py` + `fa3_backend.py` | `fa_backend.py` (one class, backend name parameter) |
| `cutlass_backend.py`, `trtllm_gen_backend.py`, ... | `cudnn_backend.py`, `trtllm_gen_backend.py` (trtllm-gen and cake) |
| reserved metadata buffers, `supports_cuda_graph_replan` per backend | `GraphCapacity`, `GraphBuffers`, `Transaction` in `_graph.py`; every backend re-plans through reserved storage |
| no public plan-update API: a graph re-plan is `plan()` again | the same |
| MLA trace template in `trace/templates/attention.py` | none yet: optional for an experimental API, required for graduation |

Differences worth knowing: paged attention has no deprecated argument forms
and no compatibility adapters; its `plan()` accepts every backend's semantic
arguments in one signature and excludes backends by capability instead of
raising per backend; its graph lifecycle is uniform across backends
because the controller, not the backend, owns the reserved storage; and its
auto policy only orders candidates, with the walk in the controller and the
order and trace of a plan readable through `explain()`, where the MLA
wrapper exposes neither.

## Known limitations

Rows `M` are ledger items. Rows `D1` to `D5` are the known differences
between this API and the legacy wrapper
(`BatchPrefillWithPagedKVCacheWrapper`) around cuDNN: where `auto` selects
it, and what the cuDNN backend takes and declares. They are stated here once
and referred to by id elsewhere.

| id | limitation | state |
| --- | --- | --- |
| M14 | The fa2 split-KV planner carves scratch that grows with the geometry (heads, head dim, SM count, and in graph mode the capacity's query tokens), so a bucket can need more than the buffer in use, the shared 512 MiB default included. | `PagedAttention.workspace_requirements()` bounds the need for sizing a caller-owned buffer. fa2 declines a batch that does not fit at `plan()` with the typed signal: `auto` moves on to the next candidate, a named backend raises `ValueError` naming the bytes. |
| M15 | `flashinfer/cudnn/decode.py` keeps the older graph-cache key and decorator order that the prefill path fixed. | Out of this package; recorded. |
| M16 | trtllm-gen and cake compute silently wrong results when the V cache's page stride differs from the K cache's (independently allocated pools); cuDNN and fa2 are correct. | Rejected at `run()` in the trtllm-gen backend (a `ValueError` naming the stride constraint); `tests/experimental/paged_attention/test_paged_attention_coverage.py::test_tc05_strided_inputs[trtllm-gen-kv_independent_strides]` pins it. |
| M18 | Cake creates its TMA descriptors for `q`, `k_cache` and `v_cache` (storage address, shape, strides) eagerly and pins them at capture; a `run()` whose q/k/v binding was never run eagerly raises `RuntimeError("... prewarm each Cake FMHA tensor/layout binding before CUDA Graph capture")` inside the capture. The eager descriptor pool is bounded (`kMaxReusableSlots = kMaxPinnedSlots = 4096` in `csrc/cake_fmha/jit/*_jit_binding.cu`), so a capture over thousands of distinct binding sets — the timing helper's cold-L2 rotating clones for an input far below L2 — cannot be prewarmed into it. | Documented rule, not fixable from `plan()` (it sees no tensors): run the exact `q`, `k_cache`, `v_cache` once eagerly before capturing them; `out` and `lse` may be fresh. The failed capture is clean (no device fault, instance reusable). The benchmark therefore captures one `run()` on fixed buffers and flushes L2 between replays. `tests/experimental/paged_attention/test_paged_attention_cuda_graph.py::test_cake_capture_requires_an_eager_run_on_the_captured_tensors`. |
| M20 | The fa2 kernel trims the windowed KV range as if the mask were causal (`include/flashinfer/attention/prefill.cuh`), so non-causal + sliding window is wrong once a request has more than 128 query tokens and a history (B200: q 256 / kv 768 / window 128 → rows 0..127 off by up to 0.4, LSE by 0.86; q 128, causal, window −1 and no history are exact). The legacy wrapper has the same defect (`tests/attention/test_attention_sink.py` chunk-prefill xfail). | fa2 and fa3 declare `supports_window_noncausal=False`, so the combination has no runnable backend and `resolve_paged_attention` names the reason for each; `tests/experimental/paged_attention/test_paged_attention_features.py::test_fa2_kernel_noncausal_sliding_window_defect` is a strict xfail on the kernel through the legacy wrapper — a kernel fix turns it into XPASS and asks for the capability flip. |
| M21 | The fa2 attention-sink kernel variant is wrong at head_dim 512 (B200: max abs err 1.4–1.75, LSE off by ~3; D64/128/256 sinks and D512 without sinks are exact; the legacy `BatchAttentionWithAttentionSinkWrapper` shows the same). Found by the XQA legacy grid once WP-T declared D512. | fa2 declares `sinks_head_dims = {64, 128, 256}`, so `use_sinks=True` at (512, 512) is excluded at resolve with the reason; `tests/experimental/paged_attention/test_paged_attention_features.py::test_fa2_kernel_sinks_head_dim_512_defect` is a strict xfail on the kernel through the legacy sink wrapper — a kernel fix turns it into XPASS and asks for the capability flip; the XQA conversion's D512 + sink rows skip with the resolve reason. |
| M22 | `_FaBackend.preflight` declined `use_sinks=True` with an fp8 KV cache as "not verified" although the capability table admits both; the XQA legacy suite runs the pair natively. Measured on B200 (e4m3 / e5m2, D128 / D256): the sink variant with `sm_scale * k_scale` and `out * v_scale` matches the sink-aware oracle to 4e-3. | Decline removed; `tests/experimental/paged_attention/test_paged_attention_features.py::test_attention_sinks_with_fp8_kv` pins the pair against the oracle. |
| M23 | trtllm-gen, cake and cuDNN read the slots of a request's last page past `kv_len` (the masked probabilities are 0, but `0 * NaN` reaches the output); fa2 zeroes the tail before the PV product. The TensorSpeed and modular legacy suites poison exactly that tail and expect it ignored. | Input contract (`PagedAttentionMetadata` docstring, "In-page tail"): the tail must be finite, any finite content is exact. Not a backend defect — the legacy TensorSpeed kernel masks it, the shared native kernels do not. Measured on B200 with a NaN tail: fa2 is exact, trtllm-gen, cake and cuDNN are not; the contract therefore requires finite tails (no test pins the per-backend outcome). |
| — | Only exact SM100 `(10, 0)` has a measured `auto` rule, fitted on GB200 with cudnn-frontend 1.32 / cuDNN 9.27. Its cuDNN region follows the library's gate, which holds from cuDNN 9.24, so cuDNN also leads there on cuDNN 9.24 to 9.26, for which this API has no timings. SM90 and the other SM10x parts take fixed orders keyed on `max_q_len`, SM8x and SM12x one fixed order; those orders are pinned by CPU tests and not measured through this API. On H100 the SM90 order differs from the legacy wrapper's `auto` (D2). | Needs a measured run per architecture; `explain()` shows the order a plan walked. |
| — | The GB200 comparison flags a few CUDA-graph shapes that run the same backend under both APIs as confirmed regressions in each run (9 of the 219 main-list shapes, at most 6.8% and 21 us). The comparison flags as many when both measured slots run the legacy wrapper (about 3% of same-backend shapes per run; per-shape slot ratio 0.967 to 1.025 over 24 processes), so the flags measure the harness, not the API. | Read same-backend CUDA-graph rows against that A/A floor; see "Measurement basis". |
| — | `custom_mask` with graph mode raises `ValueError`: the FA wrapper's packed-mask storage is not part of `GraphCapacity`. | Follow-up. |
| — | The generated-FA backend re-plans its wrapper in place; a failure inside that plan can leave backend-internal state ahead of the published metadata. | Same caveat as the MLA generated backends. |
| — | fa3 rows are compiled-checked only; the verification pool has no SM90 device. fa3 shares fa2's staged launch list, which has not been executed on SM90 either. | Needs one H100 run. |
| — | Capacity substitution plans kernels for the capacity maxes, and a graph-mode instance ranks `auto` on those bounds and the host lengths of the first batch it plans successfully, and keeps that backend. A short batch replayed in a prefill-sized capacity stays on the capacity's choice; the performance effect on short batches is unmeasured. So is cuDNN frozen by a first batch inside the length-based extension of its SM100 region under a capacity far above that batch: the cuDNN plan declares the capacity's bounds, not a shape class of the batch. | Benchmark item. Plan the first batch with representative lengths, or pin the backend. |
| — | cuDNN in CUDA-graph mode binds the controller's reserved table, the caller's columns included. One cuDNN engine loads pages through the columns past a request's own pages, up to the end of the request's last 128-token KV tile, and, where the capacity's KV bound is not a whole number of 128-token tiles, pool page 0 for the rest of the last tile; a NaN or Inf in such a page reaches the output as `0 * NaN`. An eager plan binds a table that names only each request's own pages, so no page the batch does not own reaches the result (`tests/experimental/paged_attention/test_paged_attention_cudnn_bounds.py::test_pages_the_batch_does_not_own_do_not_reach_the_result`). | In graph mode keep the pages those columns name finite, and either choose a capacity KV bound of whole 128-token tiles or keep pool page 0 finite. |
| D1 | Head dim 256 on SM100. The legacy wrapper's `auto` also selects cuDNN in a head-dim-256 GQA prefix domain (`flashinfer.prefill._blackwell_paged_gqa_auto_cudnn`: bf16, HND, page size 16, at most four requests of 64 to 1024 query tokens, each over 2048 to 32768 keys and at least four times its query length, GQA groups of 2, 4, 8 or 16, from 16 Mi of work), behind `_cudnn_supports_paged_gqa_auto(256)`, which needs cudnn-frontend 1.33. This API has no cuDNN candidate at head dim 256 (the capability table declares 128 and (192, 128) for cuDNN), so `auto` plans fa2 there. | Not copied: the gate is False on cudnn-frontend 1.32, the stack this API is measured on, and cuDNN at head dim 256 is not verified through this API. `_in_sm100_cudnn_region` answers False at head dim 256. Pinned by `tests/experimental/paged_attention/test_paged_attention_auto_policy.py::test_sm100_head_dim_256_is_not_copied`. |
| D2 | H100 at head dim 256 (SM90). On "NVIDIA H100 80GB HBM3" (compute capability `(9, 0)`, 132 SMs) the legacy wrapper's `auto` selects cuDNN where its native planner selects fa3 (`flashinfer.prefill._hopper_paged_auto_cudnn`): head dim 256, HND, page size 16, fp16 or bf16 with one dtype for Q, KV and output, causal or non-causal, 1 to 32 requests, 16 to 48 query heads, a longest query of 65 to 16384 tokens and a longest KV of at most 32768 that equal the declared maxes, and at least 2^24 query-key pairs across the query heads. This API keeps the fixed SM90 order (fa3 first, fa2 first up to 64 query tokens) and has no cuDNN candidate at head dim 256. | Not copied: the verification pool has no SM90 device. The legacy admission is tied to that device name and SM count; it selects cuDNN on no other Hopper device. Pinned by `tests/experimental/paged_attention/test_paged_attention_auto_policy.py::test_sm90_head_dim_256_is_not_copied` (candidate set and order only). |
| D3 | Sliding windows on cuDNN. The cuDNN graph path has a sliding window, and the legacy wrapper's explicit `backend="cudnn"` passes it, non-causal plans included. This API's cuDNN backend declares none (`supports_window=False`): a windowed plan excludes cuDNN at resolve, and non-causal with a window has no backend (M20). `auto` is unaffected: the legacy wrapper's `auto` selects cuDNN only without a window as well. | Not declared: this backend's `plan()` and `run()` pass no window, and windowed cuDNN is not measured through this API. The exclusion is pinned by `tests/experimental/paged_attention/test_paged_attention_contract_cpu.py::test_capability_rejection_reason_matrix` (row `cudnn-window0`). |
| D4 | fp8 query on cuDNN. The legacy wrapper's explicit `backend="cudnn"` takes an fp8 e4m3 `q` with an fp8 KV cache, a `q_scale` and a bf16 output; its test runs on compute-capability major 10 from cuDNN 9.24. This API declares fp16 / bf16 `q` for every backend, an fp8 KV cache for fa2 only, no `q_scale`, and an output in q's dtype. | Undeclared axis; `tests/experimental/paged_attention/test_legacy_unified_cudnn_prefill.py::test_cudnn_prefill_fp8` records the legacy test as outside the API and fails once the resolver accepts an fp8 `q` on cudnn; `tests/experimental/paged_attention/test_paged_attention_contract_cpu.py::test_capability_rejection_reason_matrix` (row `cudnn-fp8-q`) pins the capability row. |
| D5 | Bucketed cuDNN declarations. An eager cuDNN plan of this API declares a shape class (`_CudnnBackend._declared_bounds`) on every architecture, also when cuDNN is named. The legacy wrapper buckets only where its `auto` selects cuDNN on compute capability `(10, 0)` or `(10, 3)`, and declares the plan's own maxes on Hopper and for an explicit `backend="cudnn"`. | The classes are measured on SM100 only; on the other architectures they are unmeasured. |
