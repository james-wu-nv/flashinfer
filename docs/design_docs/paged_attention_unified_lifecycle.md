# Unified paged attention: contract and lifecycle

This document describes the experimental `flashinfer.prefill.PagedAttention`
facade and the private `flashinfer.experimental.paged_attention` package behind
it: the public surface, the three-stage CUDA-graph lifecycle, backend
selection, per-backend metadata derivation, the input, zero-row and LSE
contracts, the feature axes, workspace ownership, the benchmark and trace
tooling, and the correspondence to the Batch MLA package it mirrors
(`docs/design_docs/batch_mla_backend_architecture.md`). Everything stated
here is taken from the code; where a report and the code disagree, the code
wins, and the "Known limitations" section carries the ledger ids of the open
items.

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
- The controller owns the plan/update/run lifecycle: level-1 pinning against a
  `Resolution`, the level-2 candidate walk, capacity substitution, the
  frozen contract of a captured graph, transactional staging and publication.
- Each backend module owns one complete implementation behind three calls:
  `preflight(meta)`, `plan(meta, derived)` and `run(...)`.

The plan/run flow is:

```text
flashinfer.prefill.resolve_paged_attention()        (engine init, tensor-free)
  -> Resolution: ordered candidates + exclusion reasons, pinned to a device

flashinfer.prefill.PagedAttention(device, graph_capacity=..., workspace_buffer=...)
  -> PagedAttentionMetadata.dense(...) / .csr(...)   (once per scheduler step)
  -> plan(metadata, **semantic kwargs, backend=Resolution | name | "auto")
       -> validate; pin against the Resolution or resolve now
       -> [graph mode] capacity preflight; substitute capacity maxes
       -> for each pinned candidate: derive its declared forms, preflight()
       -> [graph mode] check the frozen contract; stage into reserved storage
       -> chosen backend.plan(meta, derived)
       -> publish
  -> run(q, (k_cache, v_cache), out=, lse=, sm_scale=, k_scale=, v_scale=, sinks=)
       -> runtime validation only, then backend.run()
  -> [graph mode] update(metadata) per step, then graph.replay()
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
against an independent fp32 oracle (`tests/experimental/paged_attention_reference.py`)
with randomized valid and corrupted inputs
(`tests/experimental/test_paged_attention_fuzzer.py`). The property the package
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
  init, then a plan-time walk within the pinned candidate set in which only a
  typed unsupported signal from a candidate's `preflight()` moves on.
- Capacity-based CUDA-graph lifecycle: construct with a `GraphCapacity`, let
  the first plan freeze the semantic contract, then `update()` per step.
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
- Autotuning. `backend="auto"` is a static per-architecture preference order
  seeded by hand (`HEURISTIC_ORDER` in `_selection.py`).
- KV-cache append, cascade or DCP orchestration.
- Public exposure of the package modules or the concrete backend classes.

## Public surface

`flashinfer/prefill.py` imports `PagedAttention` and `resolve_paged_attention`
eagerly from `flashinfer/_paged_attention.py` and exposes the value types
`PagedAttentionMetadata`, `GraphCapacity`, `Resolution` and
`PagedAttentionCapabilities` through a module `__getattr__`, so importing
`flashinfer` never loads the experimental package
(`tests/experimental/test_paged_attention_contract_cpu.py::test_importing_flashinfer_does_not_load_the_experimental_package`).
`resolve_paged_attention`, `PagedAttention.plan`, `update` and `run` carry
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
    backend="auto",
) -> Resolution
```

Tensor-free. Returns the ordered candidate set and a reason for every
excluded backend; raises `ValueError` when nothing can run. The result is
pinned to `device` (compute capability major, minor and device index) or, with
only `cc_major`, to that major and no device (`_selection._bind_device`).

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
.update(metadata) -> PagedAttention
.run(q, kv_cache, *, out=None, lse=None, sm_scale=None,
     k_scale=None, v_scale=None, sinks=None) -> (out, lse)
.explain() -> str
.backend -> Optional[str]
.device -> torch.device
```

`plan()` returns `self`. `run()` returns `(out, lse)` with `lse` `None` when
`lse_mode="none"`.

### `GraphCapacity`

```python
GraphCapacity(batch_size, total_q_tokens, max_q_len, max_kv_len, page_size,
              kv_input_form="block_tables", table_width=None, flat_capacity=None)
GraphCapacity.from_metadata(metadata)
```

The capture shapes of one CUDA-graph bucket. Validation in `__post_init__`
(`_graph.py`): positive ints; `total_q_tokens >= batch_size`;
`max_q_len <= total_q_tokens`; dense form requires `table_width` with
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
    attn.update(md_step)                            # only the batch changes
    g.replay()
```

`q_buf`, `out_buf` and `lse_buf` are sized to `cap.total_q_tokens` rows; rows
past the batch are neither read nor written. The `Resolution` passed to
`plan()` must have been resolved with `need_lse=True` when `lse_mode` is not
`"none"`, because `need_lse` is part of the pinned config key.

## Layering and ownership

### Facade

`flashinfer/_paged_attention.py` owns the public signatures, the
experimental marker, the trace hook on `run()` and the lazy handoff to the
package. It contains no backend logic.

### Controller

`_controller.py` (`PagedAttentionController`) owns:

- argument validation and normalization (`window_left >= -1`, `lse_mode`,
  `logits_soft_cap`, the custom-mask contract, the causal envelope);
- level-1 pinning against a `Resolution` (semantic key and device binding)
  or an inline resolve;
- the level-2 candidate walk and the plan-time selection trace;
- graph mode: capacity preflight, capacity substitution, reserved-table
  reservation, the frozen contract, the staging transaction;
- publication of the plan state, `update()`, `run()` validation and dispatch,
  `explain()`, and the read-only `trace_context()`.

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
`HEURISTIC_ORDER`. `_backends/_capabilities.py` owns the declarative
`PagedAttentionCapabilities` table, its pure `rejection_reason()` and the
typed `_BackendPlanUnsupportedError`. The capability table only holds facts
that drive selection; adapter details such as which derived forms a backend
reads live on the backend class (`DERIVED_NEEDS`) and are exposed through
`_backends.derived_needs(name)`.

### Backends

`_backends/fa_backend.py` (`_FaBackend`, names `fa2` and `fa3`),
`_backends/cudnn_backend.py` (`_CudnnBackend`) and
`_backends/trtllm_gen_backend.py` (`_TrtllmGenBackend`, names `trtllm-gen`
and `cake`). Each owns its native dialect, its plan state, its LSE
normalization and its ABI checks. Nothing outside a backend knows its
dialect.

## Package structure

```text
flashinfer/
|-- prefill.py                     re-exports; lazy value types
|-- _paged_attention.py            facade: resolve_paged_attention, PagedAttention
|-- trace/templates/paged_attention.py
`-- experimental/paged_attention/
    |-- __init__.py
    |-- _contracts.py              PagedAttentionMetadata, PlanMetadata, Resolution
    |-- _planning.py               validation, HostArrays, Derived, derive()
    |-- _selection.py              resolve_paged_attention(), probes, HEURISTIC_ORDER
    |-- _controller.py             plan()/update()/run() lifecycle
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
   `kv_dtype`, `logits_soft_cap`, check `kv_layout`, `window_left`,
   `lse_mode`, the custom mask, and the causal envelope when `causal=True`.
2. Pin: with a `Resolution`, compare the semantic part of the config key and
   the device binding; with a name or `"auto"`, resolve now on the instance's
   device.
3. Walk the pinned candidates in order. For each: derive exactly the forms
   its `DERIVED_NEEDS` declares (cached on the metadata object), build a
   `PlanMetadata`, construct or reuse the backend object, and call
   `preflight(meta)`. A `_BackendPlanUnsupportedError` records the reason and
   moves on; any other exception propagates; an explicit backend name turns
   the typed signal into a `ValueError`.
4. Call the chosen backend's `plan(meta, derived)`.
5. Publish the backend, the metadata, the derived forms, the resolution and
   the selection trace together.

A failed re-plan leaves the previous plan runnable, with one caveat the
controller documents: the generated-FA backend re-plans its legacy wrapper
in place, so a failure inside that wrapper's plan can leave backend-internal
state ahead of the published metadata.

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
flat page-id length. Then the candidate walk runs, and the controller builds
the frozen contract of the chosen plan: the backend name, whether a dense
table is present, and every `PlanMetadata` field that is not a per-batch
value (`_PER_BATCH_FIELDS` = `qo_indptr`, `kv_seq_lens`, `block_tables`,
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

**Stage 3, `update()`.** `update(metadata)` re-issues `plan()` with the
keyword arguments the first graph-mode plan froze, including the chosen
backend name as an explicit `backend=`, so it stages into the reserved
storage, preflights against the capacity and rolls back on failure exactly as
a re-plan does; there is no second staging path. It raises `RuntimeError`
when the instance is not in graph mode, when no plan has succeeded yet, or
when the current stream is capturing. `plan()` in graph mode is also rejected
during capture. `update()` has no stream binding; call it on the stream the
graph is replayed on.

**Warm-up before capture.** Every backend needs one eager `run()` after the
plan and before capture for its lazy work (module load, cuDNN graph build).
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
rows. The FA backend hands its wrapper the batch's row prefix of the same
storage (`q[:n]`, `out[:n]`, `lse[:n]`), because the legacy wrapper insists
on `q.shape[0] == qo_indptr[-1]`; cuDNN and trtllm-gen read the row count
from `qo_indptr`. `sm_scale`, `k_scale` and `v_scale` are launch scalars: a
captured graph keeps the values it was captured with
(`tests/experimental/test_paged_attention_coverage.py::test_tc02_graph_bakes_the_captured_sm_scale`).

## Backend selection

### Level 1: `resolve_paged_attention()`

`_selection.resolve_paged_attention` binds the device, validates the static
arguments, normalizes `logits_soft_cap`, and evaluates backends. With
`backend="auto"` it evaluates every declared backend, ordered by
`HEURISTIC_ORDER[cc_major]` first and the remaining names after, so
`Resolution.explain()` carries a reason for each excluded backend. With an
explicit name it evaluates only that backend. For each backend it asks the
capability table (`rejection_reason`) and, if admitted, the environment probe
(`PROBES[name](device)`):

- `fa3`: `is_sm90a_supported(device)`; with only `cc_major`, CUDA >= 12.3.
- `cudnn`: the cudnn-frontend package must be importable.
- `trtllm-gen`: no probe; cubin availability is deferred to the first run.
- `cake`: exact compute capability `(10, 0)` or `(10, 3)` when a device is
  known.

The result is a frozen `Resolution(backends, excluded, kv_layout, config)`.
`config` is `resolve_config_key(...)`: the semantic arguments including the
three feature flags, and as its last element the device binding
`(cc_major, cc_minor, device_index)`.

`HEURISTIC_ORDER` (`_selection.py`):

| cc major | preference order |
| --- | --- |
| 10 | trtllm-gen, cake, cudnn, fa2 |
| 9 | fa3, fa2, cudnn |
| 8 | fa2, cudnn |
| 12 | fa2, cudnn |

It is a static placeholder, to be seeded from the benchmark suite.

### Capability axes

`PagedAttentionCapabilities` (`_backends/_capabilities.py`) declares per
backend: `cc_majors`, `q_dtypes`, `kv_dtypes`, `head_dims` (pairs),
`page_sizes` (`None` = any at or above the global floor), `kv_layouts`,
`supports_lse`, `supports_noncausal`, `supports_window`,
`supports_window_noncausal`, `requires_contiguous_q`, `needs_dense`,
`supports_logits_soft_cap`, `supports_custom_mask`,
`supports_sinks`. The rule is capability honesty: an admitted configuration
is one the conformance matrix and fuzzer exercise on hardware.

| backend | cc | head dims | page sizes | layouts | noncausal | window | window + noncausal | contiguous q | kv dtypes | dense table | soft cap | custom mask | sinks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fa2 | 8, 9, 10, 12 | 64, 128, 256, 512 | any | HND, NHD | yes | yes | yes | no | f16/bf16, fp8 e4m3 / e5m2 | no | yes | yes | yes |
| fa3 | 9 | 64, 128, 256 | any | HND, NHD | yes | yes | yes | no | f16/bf16 | no | yes | no | yes |
| cudnn | 8, 9, 10, 12 | 128, (192, 128) | any | HND, NHD | yes | no | n/a | yes | f16/bf16 | yes | no | no | no |
| trtllm-gen | 10 | 128 | 16, 32, 64, 128, 256, 512, 1024 | HND, NHD | yes | yes | no | no | f16/bf16 | yes | no | no | yes |
| cake | 10 | 128 | 16, 32, 64, 128, 256, 512, 1024 | HND, NHD | yes | yes | no | no | f16/bf16 | yes | no | no | yes |

`rejection_reason()` checks, in order: compute capability, q dtype, kv dtype,
the q/kv pair (only an fp8 KV with an f16/bf16 q may differ), head dims, page
size, layout, non-causal, LSE, window, window with non-causal, soft cap,
custom mask, sinks, and finally whether a dense-reading backend would have
to derive its table from flat page ids below `MIN_DENSE_PAGE_SIZE`.

### Level 2: the candidate walk

In `plan()`, the controller walks `resolution.backends` in order. A candidate
may decline the specific batch from `preflight(meta)` with
`_BackendPlanUnsupportedError` (a `RuntimeError` subclass defined in
`_capabilities.py`, the twin of the MLA package's private type). Preflight is
allocation-free and runs before any live-state write, so declining is free of
side effects. The walk finishes before any reserved buffer is written in
graph mode. Every other exception propagates unchanged; `run()` never falls
back. With an explicit backend name the typed signal becomes a `ValueError`
`"backend 'x' cannot plan this batch: ..."`. If every candidate declines, the
controller raises `ValueError` `"no pinned candidate can plan this batch"`
listing each reason.

Current preflight rules:

- `fa2`/`fa3` (`_FaBackend.preflight`): `fa3` requires SM90a and CUDA >= 12.3
  on the instance's device; with `use_sinks=True`, a soft cap (the
  AttentionSink variant has no soft-cap hook), a custom mask and an fp8 KV
  cache are declined as unverified. A custom mask in graph mode raises
  `ValueError` (not the typed signal, since no other backend takes custom
  masks): the wrapper's packed-mask storage is not part of the capacity yet.
- `cudnn` (`_CudnnBackend.preflight`): the cudnn-frontend package must be
  importable.
- `trtllm-gen`/`cake` (`_TrtllmGenBackend.preflight`): the page size must be
  one the paged context kernel ships for; non-causal with a sliding window has
  no kernel; a non-contiguous dense block table is declined (see the input
  contract).

### `explain()`

`PagedAttention.explain()` raises `ValueError` before the first successful
plan. Afterwards it prints the chosen backend, the plan-time trace (one line
per candidate tried: `name [preflight]: accepted` or the decline reason) and
the `Resolution.explain()` text (the device the resolution answers for, the
candidates in preference order, and each exclusion reason). The trace is also
available as `PagedAttentionController.selection_trace`.

## Planning contract

### Derived forms and per-backend needs

`_planning.DERIVED_FORMS` names what a backend may request: `q_seq_lens`,
`cum_kv_seq_lens`, `kv_page_indptr`, `kv_page_indices`, `block_tables`,
`host_arrays`. Each backend class declares `DERIVED_NEEDS`:

| backend | `DERIVED_NEEDS` | native dialect |
| --- | --- | --- |
| fa2 / fa3 | `kv_page_indices`, `host_arrays` | CSR page ids on device; `qo_indptr`, page indptr, last-page lengths and KV lengths as pinned host arrays for the wrapper's CPU split-KV scheduler |
| cudnn | `q_seq_lens`, `block_tables` | token-unit `batch_offsets_q`, `actual_seq_lens_{q,kv}` as `(b, 1, 1, 1)`, a width-exact dense table |
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
plan on trtllm-gen or cuDNN with a dense table issues zero kernel launches
and one pinned host-to-device copy (`wp-c.md` §2.2 measurement; pinned by
`tests/experimental/test_paged_attention_prototype.py::test_warm_plan_uploads_from_pinned_memory_only`).

### Validation

`validate_structure` (host-only, shape-derived): every metadata tensor is an
int32 CUDA tensor on the metadata's device with the right rank; `qo_indptr`
has `batch_size + 1` entries; `block_tables` has `batch_size` rows; the page
size respects the dense floor; `max_q_len` and `max_kv_len` are positive host
ints; `max_kv_len` fits the table capacity.

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
  bounded as above. Backends with `requires_contiguous_q` (cudnn, cake) also
  require `q.is_contiguous()`: cuDNN builds its graph from `q.stride()` but
  scales the token-unit ragged offsets by `num_qo_heads * head_dim`, so a
  head slice of a fused QKV buffer is misaddressed. fa2, fa3 and trtllm-gen
  accept any unit-inner-stride view (a fused-QKV head slice, a padded head
  stride, a storage offset), measured natively in
  `tests/experimental/test_paged_attention_strides.py`.
- `kv_cache`: a `(k_cache, v_cache)` pair, each 4-D in the planned layout
  (`HND`: `(pages, H, page_size, D)`; `NHD`: `(pages, page_size, H, D)`), with
  the planned `kv_dtype`. A transposed pair gets a hint naming the layout.
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

- **cuDNN** (`_CudnnBackend.plan`): the table's page dimension must equal
  `ceil(max_kv_len / page_size)` exactly, or cuDNN fails at finalize. Engines
  hand over capacity-width tables, so the backend takes the internal narrow
  view `block_tables[:, :width]`, which keeps the caller's row stride (the
  cuDNN graph is stride-driven). It requires a unit stride along the page
  dimension and at least `width` columns. The prefill graph cache key in
  `flashinfer/cudnn/prefill.py` includes the table's shape and strides, the
  q/k/v strides and dtypes and the softmax scale, and the cache decorator sits
  outside the build step so a failed build inserts nothing. In graph mode the
  view is of the reserved buffer.
- **trtllm-gen / cake** (`_TrtllmGenBackend.preflight`): the launcher takes
  the row stride from `block_tables.size(-1)` and the kernel walks a raw int32
  pointer, so a narrow view of a wider table is read as a packed array and is
  silently wrong. A non-contiguous table is declined with the typed signal;
  under `auto` the walk moves to a backend that walks the table by its
  strides, an explicit backend surfaces a `ValueError` with the
  `.contiguous()` hint. Extra columns past `max_kv_len` are fine. In graph
  mode the caller's table is copied into the contiguous reserved buffer, so a
  narrow view is accepted there.
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
B200 with the native calls (`wp-c.md` §3). Implementation: value validation
rejects only negative lengths; the causal envelope exempts padding rows; the
derived last-page length of a padding row is `page_size` by the
`kv - (pages - 1) * page_size` formula (the FA kernel never reads it, since
its indptr span is empty); the flat-to-dense gather clamps a page-less row to
a live in-pool id that no kernel reads for that row. Padding rows can be
toggled live/padding across graph re-plans
(`tests/experimental/test_paged_attention_cuda_graph.py::test_replan_toggles_padding_rows`).

`q_len == 0` rows (`qo_indptr[i] == qo_indptr[i + 1]`, vLLM's padded
`query_start_loc` tail) are legal as well (ledger M17). Such a row owns no
query token and no output row, so there is nothing unspecified about it; the
contract is that every other row is unchanged. Measured on B200 with the
native calls (`wp-n.md`): fa2 (CSR planner), cuDNN (`actual_seq_lens_q = 0`
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
(`tests/experimental/test_paged_attention_cuda_graph.py::test_decode_bucket_replan_with_zero_q_tail_rows`).

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
no host sync) succeeds (`lse_path == "direct"`). At `max_q_len == 1` the padded
`(b, 1, h)` layout is byte-identical to the packed one and is used as a view
(`"view"`), because cuDNN 9.25 writes no stats there when a ragged stats
offset is set. Otherwise the backend gathers padded native stats with
plan-time precomputed indices (`"gather"`); in graph mode those indices keep
the capacity's row count.

## Feature axes

All of these are `plan()` arguments, capability-checked at resolve and plan
time, and frozen by the first graph-mode plan:

- `causal` (default `True`): also enforces the causal envelope. Every backend
  declares non-causal support; trtllm-gen's was measured on B200
  (`_capabilities.py` comment; `test(paged_attention): trtllm-gen non-causal
  capability by measurement`).
- `window_left` (default `-1` = unlimited; values below `-1` are rejected
  because backends disagree on their meaning): plan-time because it selects a
  compiled kernel variant on the FA backends. cuDNN declares no window;
  trtllm-gen and cake decline non-causal plus window.
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
  forwards `sinks` only on its trtllm-gen path); trtllm-gen and cake pass the
  tensor natively. The pair mirrors the MLA wrapper's `use_sinks` / `sinks`.
- fp8 KV cache: `kv_dtype=torch.float8_e4m3fn` or `torch.float8_e5m2` with
  an f16/bf16 q, declared for fa2 only (e5m2 measured on B200 against the
  oracle on the dequantized cache, same error magnitude as e4m3; see the
  `_capabilities.py` comment). `k_scale` / `v_scale` are per-tensor host
  floats (`dequant = fp8_value * scale`); omitted scales mean no scaling.
  fp8 q and nvfp4 are undeclared axes.
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
pool of `_WORKSPACE_BYTES = 128 MiB`, so one instance per graph bucket costs
no workspace per bucket. A caller may pass `workspace_buffer=`: a contiguous
1-D `uint8` or `int8` tensor on the instance's device, viewed as `uint8`. The
generated-FA wrapper receives it as its float workspace; cuDNN views it as
`int8`; trtllm-gen and cake use it as their softmax-stats scratch and keep a
separate small zero-initialized multi-CTA counter buffer of their own.
Instances sharing a workspace must not run concurrently on different streams;
pass a private buffer where that isolation is needed.

The 128 MiB default has been observed to overflow the fa2 split-KV planner
on a single 2048-token request with 32 heads (ledger M14, found by the
benchmark smoke run). The sibling work package WP-I (branch
`next/workspace-sizing`) is adding `PagedAttention.workspace_requirements()` and revisiting the
default; this section describes the code at the integration head only.

## Benchmark and trace

### Benchmark

`benchmarks/flashinfer_benchmark.py --routine PagedAttention`
(`benchmarks/routines/paged_attention.py`) times one set of inputs for every
requested backend (`fa2`, `fa3`, `cudnn`, `trtllm-gen`, `cake`, `auto`),
gates each on the fp32 oracle before timing, and emits one CSV row per
(backend, `api_variant`, phase): `plan` (metadata construction plus `plan()`,
synchronized host wall; a graph-mode re-plan when CUDA graphs are on), `run`
(warmed `run()` with preallocated buffers, GPU time: by default one `run()`
on the case's own buffers captured once and its replay timed with CUDA events
and an L2 flush before every replay — bindings fixed, so cake's M18 rule holds
for every backend, instead of the timing helper's rotating clones), `update`
and `update_replay` (CUDA graphs on: a second instance built with an explicit
`GraphCapacity` from the case, one `run()` captured, then per step a fresh
metadata plus `update()`, alone and followed by the replay; synchronized host
wall, the replayed buffers re-checked against the oracle), `step` (one plan
plus N eager runs). Rows for unsupported, erroring or incorrect candidates
are kept with a `status` column; the typed preflight signal, alone or chained
under the candidate walk's `ValueError`, is classified as `unsupported`.
`auto` rows record the resolved backend. `--pa_legacy` adds rows through the
legacy public API of the same kernel; its cuDNN provider hands the native API
the width-exact column view of the case's wider block table (the view the
unified cuDNN backend takes), and its `update` rows time the per-step re-arm
of the captured legacy call (the graph-mode wrapper's `plan()` for fa2/fa3,
the device-metadata derivation into fixed buffers for cuDNN and
trtllm-gen/cake).
`benchmarks/bench_paged_attention_plan.py` is the micro-benchmark for
metadata construction, eager plan and graph re-plan cost, with launch and
memcpy counts from the profiler.

### Trace

`PagedAttention.run` carries a flashinfer-bench trace template
(`flashinfer/trace/templates/paged_attention.py`, bound through
`@flashinfer_experimental_api(trace=paged_attention_trace_dispatch)`).
`flashinfer.fi_trace(attn.run, q=q, kv_cache=(k, v))` on a planned instance,
or `FLASHINFER_TRACE_DUMP=1` during `run()`, exports a definition whose
identity is the plan's paging form (`paged_attention_dense` /
`paged_attention_csr`, with a `_fp8kv` schema variant), KV layout, causal
flag, sliding window and LSE mode as integer `Const` axes plus the geometry
(heads, head dims, page size). The resolved backend is provenance, not
identity, so fa2, cuDNN and trtllm-gen traces of one plan compare against the
same definition. The dispatcher reads the controller's read-only
`trace_context()`: no device-to-host copy, no launch; the tensors are the
ones the planned kernels read (the reserved storage in graph mode), and the
flat form traces the live prefix of the plan's own page-id list even when
the chosen backend read a derived dense table. A plan that uses
`logits_soft_cap`, a custom mask or sinks refuses to trace, because the
definition does not encode them yet; so does a planned batch that contains
padding rows (`kv_len == 0`), because the definition constrains
`min(kv_seq_lens) >= 1` and its reference has no padding-row convention.
Tracing before the first successful plan or through the unbound
`PagedAttention.run.fi_trace(...)` raises.

## Correspondence to the Batch MLA package

| Batch MLA (`flashinfer/mla/_batch_mla/`) | Paged attention (`flashinfer/experimental/paged_attention/`) |
| --- | --- |
| `flashinfer.mla` facade, `_core.py` | `flashinfer.prefill` re-exports, `flashinfer/_paged_attention.py` |
| `_wrapper.py` `BatchMLAPagedAttentionWrapper` | `_controller.py` `PagedAttentionController` behind `PagedAttention` |
| `MLAPlanMetadata.csr()` / `.dense()` / `.dual()` | `PagedAttentionMetadata.dense()` / `.csr()` (no dual form: one truth) |
| `MLAInputContract` | `PlanMetadata` plus the frozen contract dict in graph mode |
| `_MLAPlanArguments` and its lazy resolver | `PlanMetadata`, `Derived`, `HostArrays` (`_planning.py`) |
| backend fixed at construction; `determine_mla_backend()` picks fa2/fa3 | `resolve_paged_attention()` at init, candidate walk within the pinned set at plan |
| `_backends/_capabilities.py`, `_BackendPlanUnsupportedError` | same module name and twin exception type (to be unified) |
| `_fa_common.py` + `fa2_backend.py` + `fa3_backend.py` | `fa_backend.py` (one class, backend name parameter) |
| `cutlass_backend.py`, `trtllm_gen_backend.py`, ... | `cudnn_backend.py`, `trtllm_gen_backend.py` (trtllm-gen and cake) |
| reserved metadata buffers, `supports_cuda_graph_replan` per backend | `GraphCapacity`, `GraphBuffers`, `Transaction` in `_graph.py`; every backend re-plans through reserved storage |
| no public plan-update API | `PagedAttention.update(metadata)` |
| MLA trace template in `trace/templates/attention.py` | `trace/templates/paged_attention.py` |

Differences worth knowing: paged attention has no deprecated argument forms
and no compatibility adapters; its `plan()` accepts every backend's semantic
arguments in one signature and excludes backends by capability instead of
raising per backend; and its graph lifecycle is uniform across backends
because the controller, not the backend, owns the reserved storage.

## Known limitations

| id | limitation | state |
| --- | --- | --- |
| M14 | The 128 MiB shared default workspace overflows the fa2 split-KV planner on a single 2048-token request with 32 heads. | WP-I adds `workspace_requirements()`; pass the engine's buffer meanwhile. |
| M15 | `flashinfer/cudnn/decode.py` keeps the older graph-cache key and decorator order that the prefill path fixed. | Out of this package; recorded. |
| M16 | trtllm-gen and cake compute silently wrong results when the V cache's page stride differs from the K cache's (independently allocated pools); cuDNN and fa2 are correct. | Rejected at `run()` in the trtllm-gen backend (a `ValueError` naming the stride constraint); `tests/experimental/test_paged_attention_coverage.py::test_tc05_strided_inputs[trtllm-gen-kv_independent_strides]` pins it. |
| M18 | Cake creates its TMA descriptors for `q`, `k_cache` and `v_cache` (storage address, shape, strides) eagerly and pins them at capture; a `run()` whose q/k/v binding was never run eagerly raises `RuntimeError("... prewarm each Cake FMHA tensor/layout binding before CUDA Graph capture")` inside the capture. The eager descriptor pool is bounded (`kMaxReusableSlots = kMaxPinnedSlots = 4096` in `csrc/cake_fmha/jit/*_jit_binding.cu`), so a capture over thousands of distinct binding sets — the timing helper's cold-L2 rotating clones for an input far below L2 — cannot be prewarmed into it. | Documented rule, not fixable from `plan()` (it sees no tensors): run the exact `q`, `k_cache`, `v_cache` once eagerly before capturing them; `out` and `lse` may be fresh. The failed capture is clean (no device fault, instance reusable). The benchmark therefore captures one `run()` on fixed buffers and flushes L2 between replays. `tests/experimental/test_paged_attention_cuda_graph.py::test_cake_capture_requires_an_eager_run_on_the_captured_tensors`. |
| M20 | The fa2 kernel trims the windowed KV range as if the mask were causal (`include/flashinfer/attention/prefill.cuh`), so non-causal + sliding window is wrong once a request has more than 128 query tokens and a history (B200: q 256 / kv 768 / window 128 → rows 0..127 off by up to 0.4, LSE by 0.86; q 128, causal, window −1 and no history are exact). The legacy wrapper has the same defect (`tests/attention/test_attention_sink.py` chunk-prefill xfail). | fa2 and fa3 declare `supports_window_noncausal=False`, so the combination has no runnable backend and `resolve_paged_attention` names the reason for each; `tests/experimental/test_paged_attention_features.py::test_fa2_kernel_noncausal_sliding_window_defect` is a strict xfail on the kernel through the legacy wrapper — a kernel fix turns it into XPASS and asks for the capability flip. |
| — | `auto` is a static order. On SM100 it picks trtllm-gen for a decode-shaped batch (B=32, q=1, kv=4096) that fa2 runs about five times faster in the benchmark smoke run. | Shape-bucket order table is wave-2 work; the benchmark's `resolved_backend` column shows the regret. |
| — | `custom_mask` with graph mode raises `ValueError`: the FA wrapper's packed-mask storage is not part of `GraphCapacity`. | Follow-up. |
| — | The generated-FA backend re-plans its wrapper in place; a failure inside that plan can leave backend-internal state ahead of the published metadata. | Same caveat as the MLA generated backends. |
| — | `update()` has no stream binding. | Call it on the replay stream. |
| — | `cake` keeps `requires_contiguous_q=True`; only the trtllm-gen kernel was probed for strided q. | Probe pending. |
| — | fa3 rows are compiled-checked only; the verification pool has no SM90 device. | Needs one H100 run. |
| — | The trace definition constrains `min(kv_seq_lens) >= 1`, older than the zero-row contract; a batch with padding rows refuses to trace rather than export a self-violating workload. | Give the reference a padding-row convention, relax the constraint and regenerate the fixture. |
| — | Capacity substitution plans kernels for the capacity maxes; its performance effect on short batches is unmeasured. | Benchmark item. |
