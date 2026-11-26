# Experimental unified paged attention (`PagedAttention`)

`flashinfer.prefill.PagedAttention` is the experimental successor to
`BatchPrefillWithPagedKVCacheWrapper` for paged attention over the existing
fa2/fa3, cuDNN, trtllm-gen and (on SM100/SM103) Cake kernels. This package is
its implementation; the public names are thin entry points in
`flashinfer/_paged_attention.py`, re-exported from `flashinfer.prefill`.

- **Owner:** @james-wu-nv
- **Tracking issue:** [#5007](https://github.com/flashinfer-ai/flashinfer/issues/5007) (graduation plan)
- **Design doc:** [`docs/design_docs/paged_attention_unified_lifecycle.md`](../../../docs/design_docs/paged_attention_unified_lifecycle.md)
  (contract, lifecycle, known limitations)

## Status and opt-in

- **What is experimental.** The API: `PagedAttention`,
  `PagedAttentionMetadata`, `GraphCapacity` and `resolve_paged_attention` may
  change or be removed without deprecation. The package contains no kernels;
  it plans and launches the kernels FlashInfer already ships.
- **Opt-in.** Calling any of these is the opt-in (an `ExperimentalWarning` is
  emitted once). No environment variable gates it.
- **Automatic routing.** No stable API routes to this one. Inside it,
  `backend="auto"` and `backend="autotune"` choose among fa2, fa3, trtllm-gen
  and cuDNN; `cake` runs only when named.
- **Packaging.** Nothing is registered in `flashinfer/aot.py`, and the API is
  not listed in `docs/api/` before graduation; this README is its reference.
- **Relation to the legacy wrapper.** `BatchPrefillWithPagedKVCacheWrapper`
  remains fully supported and receives bug fixes; its deprecation is planned
  for after this API graduates. The unified API is the only path on which
  `backend="auto"` can select the trtllm-gen kernel (the wrapper's own `auto`
  chooses between fa2 / fa3 and cuDNN, the latter inside its SM100 domains
  and, at head dimension 256, on H100 80GB).

## Usage

```python
import torch
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

device = torch.device("cuda")
config = dict(
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim_qk=128,
    q_dtype=torch.bfloat16,
    kv_layout="HND",
    causal=True,
)

# Engine init: which backends can run this configuration, and why the others cannot.
resolution = resolve_paged_attention(
    device=device, page_size=16, need_lse=True, **config
)
print(resolution.explain())

# One scheduler step: 4 requests of 128 query tokens over 512 cached tokens each.
batch, q_len, kv_len, page_size = 4, 128, 512, 16
pages = kv_len // page_size
qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * q_len
kv_seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
block_tables = torch.arange(batch * pages, dtype=torch.int32, device=device).view(
    batch, pages
)
metadata = PagedAttentionMetadata.dense(
    qo_indptr,
    kv_seq_lens,
    block_tables,
    page_size=page_size,
    max_q_len=q_len,
    max_kv_len=kv_len,
)

# caller-owned scratch, as the legacy wrappers' float_workspace_buffer
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
attn = PagedAttention(workspace)
attn.plan(metadata, lse_mode="base2", backend=resolution, **config)

q = torch.randn(batch * q_len, 32, 128, dtype=torch.bfloat16, device=device)
k_cache = torch.randn(
    batch * pages, 8, page_size, 128, dtype=torch.bfloat16, device=device
)  # HND: (pages, kv_heads, page_size, head_dim)
v_cache = torch.randn_like(k_cache)
out, lse = attn.run(q, (k_cache, v_cache))  # one call per layer
print(attn.explain())
```

`examples/experimental/paged_attention.py` is the runnable example: it
resolves at engine init, runs one batch on every runnable backend and
compares them, runs a decode-shaped batch through `auto`, and shows a
rejected call. The design doc's "Public usage" section has the engine-shaped
loop and the CUDA-graph form, and [`integrations/`](integrations/README.md)
holds the vLLM and SGLang integration sketches. The docstrings of
`PagedAttention` (`plan`, `run`, `explain`, `backend`,
`workspace_requirements`), `PagedAttentionMetadata` (`dense`, `csr`),
`GraphCapacity` (`from_metadata`) and `resolve_paged_attention` are the
argument reference.

## Contract

- **Metadata.** One `PagedAttentionMetadata` object per scheduler step:
  token-unit `qo_indptr`, per-request `kv_seq_lens`, a dense block table via
  `.dense(...)` or flat page ids via `.csr(...)`, required host maxes,
  optional CPU mirrors for a zero-sync plan. A `kv_seq_lens` entry of 0 marks
  a padding row, and a request with `q_len == 0` is legal too.
- **Resolve.** `resolve_paged_attention` answers at engine init which
  backends can run a configuration and why the others cannot.
- **Plan.** `plan()` declares the LSE base (`lse_mode`) and the feature axes
  (`window_left`, `logits_soft_cap`, `custom_mask`, `use_sinks`,
  `fixed_split_size` / `disable_split_kv`), each capability-checked so
  `backend="auto"` excludes a backend that cannot apply one instead of
  dropping it.
- **Run.** `run()` takes the per-layer `sm_scale`, the `sinks` tensor when
  planned, and the per-tensor `k_scale` / `v_scale` (dequantization scales of
  an fp8 e4m3 / e5m2 KV cache, plain multipliers of a fp16 / bf16 one;
  `k_scale` folds into the softmax scale, `v_scale` into the output, on every
  backend).
- **Explain.** `explain()` returns the chosen backend, the candidates in the
  order the last plan walked them, every candidate tried at plan time and the
  resolve-time exclusion reasons.

### CUDA graphs

CUDA graphs follow a three-stage lifecycle:

1. `PagedAttention(workspace, graph_capacity=GraphCapacity(...))` reserves the metadata
   storage of one graph bucket at construction (`use_cuda_graph=True` infers
   the capacity from the first plan).
2. The first graph-mode `plan()` freezes the backend and every semantic
   argument, and a later plan that changes one is rejected (a re-plan chooses
   from the frozen backend alone, so `explain()` then lists only that backend
   in the order and the trace).
3. Calling `plan()` again with the same arguments then stages each step's
   batch into the reserved storage so the captured `run()` replays it.

A batch must fit the capacity: batch size, paging form and page size exactly,
total query tokens and host maxes at most the capacity's, which is what the
kernels are planned with. Run once eagerly after the plan and before
capturing, with `q` in the stride layout the capture will use (each new `q`
stride layout costs one cuDNN graph build inside `run()`); with the Cake
backend that eager `run()` must use the very `q` / `k_cache` / `v_cache`
tensors the capture will bind (its TMA descriptors are created eagerly and
pinned at capture; a new binding inside capture raises a `RuntimeError`
asking for the prewarm).

### Workspace

The kernel workspace is caller-owned and required, as in the legacy wrappers
and the MLA wrapper: `PagedAttention(workspace_buffer, ...)` takes a
contiguous 1-D `uint8` buffer and runs on its device; the library allocates
none. One buffer may back several instances (one per graph bucket costs no
workspace per bucket), but instances sharing a buffer must not run
concurrently on different streams.
`PagedAttention.workspace_requirements()` returns, for a
`GraphCapacity` and a model configuration, a conservative bound on the bytes
the resolvable backends' planners carve out of that buffer (the fa2 split-KV
scratch dominates), so an engine can size the buffer before capture. A
backend whose planner would overflow the buffer declines the batch at
`plan()`: `auto` moves on to a candidate that fits, and a named backend, or a
candidate set in which none fits, raises a `ValueError` naming the required
bytes (`backend 'fa2' cannot plan this batch: needs N bytes of scratch
workspace ...`).

### Split-KV control

`plan(fixed_split_size=S)` and `plan(disable_split_kv=True)` are the legacy
wrapper's arguments of the same names. The fa2 planner splits a request's KV
range across kernel blocks and merges the partial results; by default it
chooses the chunk size from the whole batch, so the last bits of a request's
output depend on the other requests. `fixed_split_size` fixes the chunk at
`S` pages; `disable_split_kv` never splits.

- Only fa2 takes them. Pass the same values to `resolve_paged_attention`:
  every other backend is then excluded with the reason, and `auto` plans on
  fa2.
- The two are mutually exclusive, and `fixed_split_size` must be a positive
  Python int.
- On a CUDA-graph instance `disable_split_kv` works and is frozen by the
  first plan; `fixed_split_size` is rejected (whether a batch is split would
  depend on its KV lengths, and a captured graph launches what the first
  plan decided).
- A `fixed_split_size` plan can need more scratch workspace than the default
  bound; `workspace_requirements()` takes the same two arguments.

Measured on GB200 through this API (bf16, head dim 128, page size 16;
request 0 of a batch against the same request in a prefix of the batch):

| batch and prefix | query tile (batch / prefix) | default | `disable_split_kv` | `fixed_split_size` 8 / 64 |
| --- | --- | --- | --- | --- |
| 4 x (q 128, kv 4096), first 2 | 128 / 128 | differs | bitwise equal | bitwise equal |
| q 64 x 3 over kv 4096 / 300 / 9000, first 1 | 128 / 128 | differs | bitwise equal | bitwise equal |
| q 40 / 700 / 3 over kv 3000 / 700 / 8000, first 2 | 128 / 128 | differs | bitwise equal | bitwise equal |
| q 8 / 200 over kv 2048, first 1 | 128 / 64 | differs | differs | differs |
| q 200 and 15 x q 1 over kv 2048, first 1 | 64 / 128 | differs | differs | differs |

The differences are in the last bits (at most 2e-3 here; every run is within
the oracle tolerance). So the controls make a request's output independent
of the batch as long as the planner gives the batches the same query tile.
See "Limitations" for what decides the tile.

### Tensor layout and argument types

`run()` takes `q` as any view that is dense along `head_dim`, stored
token-major and 16-byte aligned (head stride at least `head_dim_qk`, token
stride at least `(num_qo_heads - 1) * head stride + head_dim_qk`, both
strides multiples of 16 bytes, data pointer 16-byte aligned), for example the
head slice of a fused QKV buffer; any other layout raises a `ValueError` on
every backend. Head counts, head dimensions, `page_size`, `max_q_len` and
`max_kv_len` must be Python ints (a `bool` or a numpy integer is rejected).

## Limitations

Non-causal attention with a sliding window has no runnable backend in this
API today: trtllm-gen and cake ship no such kernel, this API's cuDNN backend
declares no sliding window (the legacy wrapper's explicit cuDNN backend runs
one; design doc, known limitation D3), and the fa2 kernel computes it wrong
for requests longer than 128 query tokens with a history (design doc, known
limitation M20), so fa2/fa3 declare it unsupported and
`resolve_paged_attention` reports the reason per backend.

The slots of a request's last page past its `kv_seq_lens` entry are never
attended but must hold finite values: the FA kernels zero that tail before
the PV product, while trtllm-gen, cake and cuDNN load the whole page and a
NaN or Inf there reaches the output as `0 * NaN` (design doc, known
limitation M23). Any finite content — stale tokens, zeros — gives the exact
result; a KV pool allocated with `torch.empty` and never written past
`kv_len` is not covered by that guarantee until its storage is initialised.

`fixed_split_size` and `disable_split_kv` do not fix the fa2 query tile, so
they give bitwise batch invariance only across batches planned with the same
tile (design doc, known limitation M24). An eager plan derives the tile from
the batch's average `q_len * (num_qo_heads // num_kv_heads)`: 128 above 64
packed rows (64 at head dim 256), 64 above 16, otherwise 16 or 64 by the
device's shared memory; head dim 512 uses 16 and 32. A CUDA-graph instance
derives it from its capacity, the same for every re-plan. The stable
wrapper's `batch_invariant=True` class is a separate mechanism and is not
reachable through this API.

cuDNN finalizes one execution plan per declared shape (about 5 ms), so an
eager cuDNN plan declares a shape class instead of the batch's exact maxes,
on every architecture (the legacy wrapper does so only where its `auto`
selects cuDNN on Blackwell; design doc, known limitation D5):
the KV bound rounds up to the next power of two, and a query bound above 64
tokens rounds up to whole `128 // (num_qo_heads // num_kv_heads)`-token
tiles, at least 128, when that GQA group divides 128. The plans of a serving
loop whose KV grows every step then share one cuDNN plan per class: a new one
is built only when the KV bound crosses a power of two. The per-request
lengths stay the actual ones. Such a plan binds a page table of its own at
the class width, built on the device at `plan()` from each request's own
pages, so the columns of the caller's table past a request's pages do not
reach cuDNN. In CUDA-graph mode the backend declares the capacity's bounds
and binds the instance's reserved table, the caller's columns included,
without a copy. The pages those columns name must then hold finite values,
and so must pool page 0 unless the capacity's KV bound is a whole number of
128-token tiles (design doc, known limitations).

## Backend selection and the `max_q_len` hint

`backend="auto"` is a fixed rule over host-side facts of the batch, not
autotuning: nothing is timed (`backend="autotune"`, below, measures
instead). It works on two levels.

`resolve_paged_attention` fixes the candidate
*set*: those of fa2, fa3, trtllm-gen and cuDNN that the capability table and
the environment probes admit for the configuration, with a reason for every
backend left out. `cake` is never an auto candidate; it runs only when
named (`backend="cake"`).

`plan()` ranks that set for the batch and takes the first candidate that
accepts it, moving on only when a candidate declines the batch. The rank
reads host values only: the batch size, the total query tokens, the declared
`max_q_len` and `max_kv_len`, the per-request lengths from the metadata's
host mirrors, the head counts and dimensions, dtypes, layout, page size, the
feature flags and whether the instance is in CUDA-graph mode. No device
tensor is read. `max_q_len` and `max_kv_len` are the maxes the metadata
declares, in CUDA-graph mode the capacity's (a capacity inferred from a dense
block table takes the table's whole width in tokens as its `max_kv_len`):
upper bounds, not the batch's longest lengths. An eager instance ranks on
every `plan()`, so `PagedAttention.backend` can change from one batch to
the next. A CUDA-graph instance ranks on the bounds of its capacity and the
per-request lengths of the batch being planned, until its first successful
plan freezes the backend: later plans keep it. Plan the
first batch of a bucket with representative lengths, or pin the backend. A
backend name, or a `Resolution` resolved with one, pins the backend:
nothing is ranked, and a batch that backend cannot plan raises a
`ValueError`.

On SM100 (compute capability 10.0) the rule is:

1.  cuDNN first inside the region in which the legacy wrapper's `auto`
    selects it at head dimension 128. Every admission needs plain causal
    attention (no sliding window, soft cap, custom mask or sinks) with bf16 Q
    and KV, head dimension 128, page size 16, HND layout and 8 to 64 query
    heads in a GQA group of 1, 2, 4, 8 or 16. With `work` for
    `num_qo_heads * q_len * kv_len` summed over the requests and Mi for
    `2**20`, two parts admit:

      - The bounded domain: at most four requests with at least 8 Mi of work,
        under declared maxes that are either a prefix, `max_q_len` of 64 to
        1024 with `max_kv_len` of 2048 to 32768 and at least four times
        `max_q_len`, or a full prefill, `max_q_len == max_kv_len` of 1024
        to 8192 (in CUDA-graph mode `1024 <= max_q_len <= max_kv_len <= 8192`) with every request's query non-empty and as long as its KV and
        the longest at least 1024 tokens.
      - The length-based extension, for any number of requests: at least 16 Mi
        of work and every request with `0 < q_len <= kv_len`; then either
        every `q_len == kv_len` with a longest query of 1024 to 16384 tokens,
        or every `q_len` above 128 with a longest query of at most 8192 and a
        longest KV of 2048 to 32768 tokens.

    The prefix and full-prefill tests read the declared maxes. The work, the
    proof that every query is as long as its KV and every test of the
    extension read the per-request lengths on the host mirrors, and "longest"
    in the extension is the batch's own. A batch that qualifies for the
    extension is therefore admitted under any maxes declared above its
    lengths, a full prefill under a `max_kv_len` sized for a longer
    context for instance. A batch the extension does not admit (less than
    16 Mi of work, for instance, or a request of at most 128 query tokens in
    a batch that is not all full prefill) needs the bounded domain, which
    reads the declared maxes: under a `max_kv_len` above 32768 it is
    outside the region whatever its lengths.

    In CUDA-graph mode the declared maxes are the capacity's and the lengths
    are those of the first planned batch. That batch alone decides the
    extension: one that qualifies freezes cuDNN, planned with the capacity's
    bounds, however far those are above it. The wider full-prefill test of
    that mode matters only for at most four requests of full prefill with at
    least 8 Mi and less than 16 Mi of work (8 to 15 query heads) under a
    capacity whose `max_kv_len` is above its `max_q_len`; from 16 Mi the
    extension admits such a batch under any capacity.

    cuDNN leads there only where the library's gate for this selection
    holds, the one the legacy wrapper asks: the cudnn-frontend FROST runtime
    with its paged packed-split selector, on cuDNN 9.24 or newer
    (cudnn-frontend 1.32 is the package's minimum requirement). Otherwise the
    region is empty and cuDNN stays the last candidate. The legacy wrapper's
    head-dimension-256 domain on SM100 is not copied (design doc, known
    limitation D1).

    One exception is this API's own, in eager mode only: inside the region
    trtllm-gen goes ahead of cuDNN for a small kernel, which is a batch whose
    longest KV has at most 4096 tokens, whose longest request has at most
    256 rows (`num_qo_heads * its query length / 128`) and whose `rows * longest KV` (`rows` as in rule 2) is at most `2**22`. On the 83
    measured shapes of that kind trtllm-gen's `run()` takes 0.67 to 0.97
    of the legacy wrapper's cuDNN time (geometric mean 0.83), and its
    `plan()` is cheaper too. CUDA-graph plans keep cuDNN first in the whole
    region.

2.  Otherwise trtllm-gen first when the packed query rows fill its 128-row
    tiles. With `rows = num_qo_heads * total_q_tokens / 128` and `fill = rows / (batch_size * num_qo_heads * ceil(max_q_len / 128))` it needs
    `rows >= 64` (96 below head dimension 128) and `fill >= 0.375` in
    CUDA-graph mode or `fill >= 0.75` eager. Eager and above 148 rows it
    also needs a query row to attend at least 1024 keys (the batch's actual
    longest KV length, from the host mirror, or `window_left + 1` if that is
    smaller) or `rows * keys >= 2**19`: on a shorter KV the kernels take
    tens of microseconds, the host cost of the launch decides, and fa2's is
    lower.

3.  Otherwise fa2 first.

The remaining candidates follow as fallbacks: the other one of fa2 and
trtllm-gen, then cuDNN outside its region. The predicate of the region and
the thresholds are in
`flashinfer/experimental/paged_attention/_auto_policy.py`; the design doc
describes the policy under "Level 2: the auto policy".

The other architectures have fixed orders, selected by the major of the
compute capability: SM90 `fa3, fa2, cudnn`, with fa2 first when
`max_q_len` is at most 64 (the legacy wrapper's short-query rule; its
cuDNN admission on H100 at head dimension 256 is not copied, design doc,
known limitation D2); the other SM10x parts `trtllm-gen, cudnn, fa2`,
with fa2 first when `max_q_len` is at most 16; SM8x and SM12x
`fa2, cudnn`, whatever the shape. `max_q_len` is the declared bound
here too (the capacity's in CUDA-graph mode). Only the SM100 rule is fitted
to measurements of this API.

The `max_q_len` hint of `resolve_paged_attention`
is a promise about the batches planned with the Resolution: they have at most
this many query tokens per request (1 for plain decode, the draft length for
speculative decoding; leave it out for prefill or mixed batches). It does two
things. `plan()` checks it: a batch (in CUDA-graph mode: a capacity) with a
larger `max_q_len` is rejected with a `ValueError`, so hold one Resolution
per query-length bucket, as an engine's decode / prefill split already does. And
it orders the resolve-time candidate list (`Resolution.backends`,
`Resolution.chosen` and what `Resolution.explain()` prints), which is the
estimate available before a batch exists: on compute-capability major 10 a
hint `<= 16` lists `fa2, trtllm-gen, cudnn` and a larger hint, or none,
`trtllm-gen, cudnn, fa2`; the other majors have one order. The hint does
not decide the backend of a plan: `plan()` ranks the candidates from the
batch itself, so a hinted and an unhinted Resolution plan the same batch on
the same backend. It never changes the candidate set or the exclusions. It is
pinned in the Resolution (`explain()` prints it; Resolutions of one model
configuration with different hints have different `config` keys).

### Measured selection: `backend="autotune"`

`backend="autotune"` selects within the same candidate set as `"auto"`
by measurement. It follows `backend="autotune"` of
`BatchMLAPagedAttentionWrapper`: tune the planned workload inside
`flashinfer.autotune(True)`, serve the cached result outside it.
`resolve_paged_attention` with
`backend="autotune"` returns the `"auto"` set with
`Resolution.autotune` set, so `plan(backend=resolution)` carries the mode.

```python
attn = flashinfer.prefill.PagedAttention(workspace)
with flashinfer.autotune(True, cache="paged_attention.json"):
    attn.plan(metadata, backend="autotune", **config)  # prepares every candidate
    attn.run(q, (k_cache, v_cache), out=out)           # profiles, keeps the winner

attn.plan(metadata, backend="autotune", **config)
attn.run(q, (k_cache, v_cache), out=out)               # cached result, no profiling
```

`plan()` prepares every candidate that accepts the batch.
`PagedAttention.backend` is `"autotune"` for the
whole plan; `explain()` lists the candidates and, after the first
`run()`, the selected one (`chosen: autotune (selected <name>)`).
`run()` requires
`out=` under this mode. The first `run()` selects one candidate and keeps
it until the next plan:

- A result cached for the planned workload and this call's tensor layouts
  and options is used (this process, or a file loaded with
  `flashinfer.autotune(cache=...)`).
- Otherwise, inside `flashinfer.autotune(True)`, each candidate is
  launched once and profiled by the autotuner on the caller's q and KV cache
  with private output buffers; the winner is cached.
- Otherwise it raises `RuntimeError`: warm up inside the context first.
  A context that skips the operation (`skip_ops`) raises as well.

The cache identity is the planned workload, not a shape bucket: the resolved
configuration, the mode (eager or CUDA graph), the bounds the kernels are
planned with, a digest of the batch's metadata (the per-request query and KV
lengths and the page ids), the workspace size, the device and the prepared
candidates. A batch with other lengths or other pages is another workload.
Computing the digest reads a device page table back, so an autotune plan
synchronizes.

The selecting `run()` cannot be under CUDA graph capture. In CUDA-graph
mode it is the eager warm-up before capture: a re-plan is refused until it
has run, and the selected backend is the one the instance freezes, so later
plans re-plan that backend alone. An eager instance prepares
every candidate on each `plan()`, which costs more host time than an
`"auto"` plan. A named backend never enters the tuner.

### Measurements

The SM100 rule is measured against
`BatchPrefillWithPagedKVCacheWrapper(backend="auto")` on GB200
(cudnn-frontend 1.32 / cuDNN 9.27, 2026-11-18; bf16, `run()` medians, every
shape eager and as a CUDA-graph replay). A shape passes when no paired
measurement block is more than 1% slower than the legacy time and is a
regression when every block is more than 1% slower:

| shapes          | mode       | geomean speedup | pass | inconclusive | regression |
| --------------- | ---------- | --------------- | ---- | ------------ | ---------- |
| 219 (main list) | CUDA graph | 1.49x           | 163  | 47           | 9          |
| 219 (main list) | eager      | 1.41x           | 205  | 14           | 0          |
| 661 (all lists) | CUDA graph | 1.39x           | 494  | 151          | 16         |
| 661 (all lists) | eager      | 1.20x           | 575  | 85           | 1          |

The CUDA-graph regressions run cuDNN under both APIs and take at most 6.8%
and 21 µs longer in that run. They are an effect of the measurement:
comparing the legacy wrapper with itself flags as many same-backend shapes
(about 3% per run, measured 2026-11-10), and over 24 independent processes
per shape the unified time was 0.975 to 1.017 of the legacy time (mean
1.002). The eager one is a head-dimension-64 shape on which the legacy
wrapper runs fa2 and the rule trtllm-gen; it takes 10% longer in this run
and 4% less in the run before.
The main list holds engine benchmark and CI configurations, the shapes the
legacy wrapper's first cuDNN region was admitted on and length-regime
probes. Of the other 442 shapes, 212 are placed on the thresholds of the
rule, 176 inside the legacy wrapper's widened cuDNN admission (1.07x eager,
1.05x CUDA graph: cuDNN under both APIs except the eager small kernels) and
54 at head dimension 256 (fa2 under both APIs, 1.00x), so the main list is
the representative one.

Examples from the measurement the thresholds were fitted on (2026-11-10; 32 query / 8 KV heads, head_dim 128, page
size 16, causal, no LSE; a dense block table exactly `kv_len / 16` columns
wide, so a CUDA-graph capacity equals the shape): `run()` medians in µs of
each backend named through this API, and the backend `auto` ran. The
benchmark script below builds a table three columns wider; with CUDA graphs
on and the dense form, its `auto` ranks on that width as the KV bound. A
wider KV bound can put a shape outside the cuDNN region only through the two
bound tests: the last shape (a full prefill of 8192 tokens) is admitted by
its lengths under either bound.

| batch | q_len | kv_len | mode       | fa2  | trtllm-gen | cuDNN | auto       |
| ----- | ------ | ------- | ---------- | ---- | ---------- | ----- | ---------- |
| 32    | 16     | 4096    | CUDA graph | 224  | 392        | 469   | fa2        |
| 32    | 16     | 4096    | eager      | 154  | 440        | 514   | fa2        |
| 32    | 64     | 4096    | CUDA graph | 446  | 396        | 472   | trtllm-gen |
| 32    | 64     | 4096    | eager      | 435  | 444        | 517   | fa2        |
| 8     | 96     | 2048    | CUDA graph | 122  | 66         | 76    | trtllm-gen |
| 8     | 96     | 2048    | eager      | 130  | 106        | 150   | trtllm-gen |
| 8     | 128    | 512     | CUDA graph | 50   | 28         | 46    | trtllm-gen |
| 8     | 128    | 512     | eager      | 55   | 66         | 88    | fa2        |
| 1     | 128    | 16384   | CUDA graph | 115  | 193        | 52    | cudnn      |
| 1     | 128    | 16384   | eager      | 133  | 230        | 100   | cudnn      |
| 1     | 8192   | 8192    | CUDA graph | 1690 | 443        | 365   | cudnn      |
| 1     | 8192   | 8192    | eager      | 1588 | 492        | 424   | cudnn      |

## Benchmark

`benchmarks/experimental/paged_attention/bench_paged_attention.py` is a standalone script (the API is
experimental, so it is not a routine of the shared benchmark CLI):

```bash
python benchmarks/experimental/paged_attention/bench_paged_attention.py \
    --backends fa2 cudnn trtllm-gen auto --batch_size 4 --s_qo 128 \
    --s_kv 1024 --num_qo_heads 32 --num_kv_heads 8 --head_dim_qk 128 \
    --page_size 16 --causal --kv_layout HND --lse_mode basee --refcheck
```

Every requested backend sees the same inputs (packed Q, a paged K/V pool with
a shuffled page mapping, exact per-request lengths) and is checked against the
fp32 oracle before it is timed (always; `--pa_skip_refcheck`
opts out for shapes the oracle cannot hold). The script reports one CSV row
per phase:
`plan` (metadata construction plus `plan()`, synchronized host wall time),
`run` (a warmed `run()` with preallocated outputs; GPU time under a CUDA
graph by default, eager with `--no_cuda_graph`) and `step` (one plan
followed by N `run()` calls, N from `--pa_layers`). Use `--s_qo 1` for
decode, `--kv_input_form csr` for flat page indices,
`--random_actual_seq_len` for variable lengths, `--pa_max_q_len_hint N`
to resolve with the `max_q_len` hint (it bounds the batches and orders the
resolve-time list that `static_backend` reports; the `auto` rows run what
`plan()` ranks first for the batch) and `--pa_legacy` to add
rows for the same inputs through the legacy public API of the same kernel
(`api_variant=legacy`). Rows for unsupported,
failing or incorrect backends are kept with a `status` column and a
`refcheck_passed` column; `static_backend` records the first candidate in
resolve-time order and `resolved_backend` the backend the plan ran. Under
`auto` the two differ when `plan()` ranks another candidate first for the
batch or when a candidate declines it (`auto` is a fixed rule, not
autotuning). `--backends autotune` warms each instance up inside
`flashinfer.autotune(True)` and then times the selected backend; its eager
`plan` rows include preparing every candidate. The benchmark generates
fp16/bf16 Q and KV; the CSV columns are `OUTPUT_COLUMNS` in the script.
`--testlist benchmarks/experimental/paged_attention/bench_paged_attention_testlist.txt` runs a list of
such command lines into one CSV (`--output_path`).

## API logging and trace

`PagedAttention.run` has no fi_trace template: trace templates are optional
for an experimental API and required for graduation, so
`flashinfer.fi_trace(attn.run, ...)` raises a `ValueError`. API logging
(`FLASHINFER_LOGLEVEL`) covers `resolve_paged_attention`, `plan()` and
`run()` like any other API.

The fa2/fa3 backends launch the kernel module from an argument list staged
at `plan()` and do not enter `BatchPrefillWithPagedKVCacheWrapper.run()`,
so `run()` of such a plan produces the unified API-log entry and dump only.
The wrapper's `run()` log entry and level-10 dump, its `gqa_paged_prefill`
trace definition and Trace Apply under that definition name apply only to
the plans that go through the wrapper's `run()`: a plan with attention
sinks, and a plan whose launch cannot be staged.

## Package layout

| File | Role |
| --- | --- |
| `_contracts.py` | `PagedAttentionMetadata`, `PlanMetadata`, `Resolution`, loud-error helpers |
| `_planning.py` | Structural and value validation, canonical to derived forms |
| `_selection.py` | Tensor-free `resolve_paged_attention()`: the candidate set and its resolve-time order |
| `_auto_policy.py` | Plan-time ranking of the auto candidates for one batch; ranks only, never decides support |
| `_controller.py` | `plan()` / `run()` lifecycle, the candidate walk, transactional publication |
| `_graph.py` | `GraphCapacity`, reserved storage and staging for CUDA graphs |
| `integrations/` | vLLM and SGLang integration sketches (documents, not code) |
| `_backends/` | One module per backend (`fa_backend.py`, `cudnn_backend.py`, `trtllm_gen_backend.py`; `cake` is a variant of the trtllm-gen one) and the declarative capabilities |

The layout mirrors `flashinfer/mla/_batch_mla/`. Core imports this package
only through deferred imports in `flashinfer/_paged_attention.py`.

## Validation

```bash
python -m pytest tests/experimental/paged_attention -q
python examples/experimental/paged_attention.py
python benchmarks/experimental/paged_attention/bench_paged_attention.py \
    --testlist benchmarks/experimental/paged_attention/bench_paged_attention_testlist.txt --output_path pa.csv
```

`tests/experimental/paged_attention/` holds the suites of this API
(`test_paged_attention_*.py`: contract, selection, auto policy, autotune,
CUDA graphs, feature axes, the split-KV control, strides, workspace, the
benchmark script and a fuzzer of valid and corrupted inputs) and the legacy wrapper's suites
converted to it (`test_legacy_unified_*.py`). Results are checked against the
fp32 oracle in `paged_attention_reference.py`.
