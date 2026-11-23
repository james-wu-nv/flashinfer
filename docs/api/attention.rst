.. _apiattention:

FlashInfer Attention Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Experimental Task-Scheduled Attention
=====================================

The experimental Blackwell task-scheduled FMHA context, FMHA decode,
block-sparse FMHA, and MLA decode APIs are imported from
``flashinfer.attention.prims_ts``. Scheduling, tile selection, and split-KV
reduction are automatic implementation details; there are no public tuning
knobs.

See the `PrimTS guide index <https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/prims_ts/README.md>`_
for the public entry points, supported contracts, and examples. Current accuracy
and performance signoff is on SM100a/B200; SM103a/B300 is architecture-gated
but not yet signoff-qualified.

Calling these APIs is an explicit opt-in and emits an
``ExperimentalWarning`` once per decorated function. They provide no API
compatibility guarantee; generated stable API reference entries are deferred
until graduation. Logging and existing ``fi_trace`` bindings remain available.

QToken-KvBlock-Sparse-Attention
--------------------------------

QToken-KvBlock-Sparse-Attention consumes per-query
``indexer_block_ids[total_q, block_topk]`` and a dense physical
``block_table``. Packed prefill uses ``[total_q, Hq, D]`` with
``qo_indptr``; fixed MTP decode uses ``[B, Nq, G, Hq, D]``.
``kv_block_size`` is the semantic sparse K/V atom and currently supports
only four tokens. The wrapper plans capacity outside CUDA Graph capture and
runs live route metadata on the hot path.


flashinfer.decode
=================

.. currentmodule:: flashinfer.decode

Single Request Decoding
-----------------------

.. autosummary::
    :toctree: ../generated

    single_decode_with_kv_cache
    single_decode_with_kv_cache_with_jit_module

Batch Decoding
--------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_decode_with_kv_cache
    trtllm_batch_decode_with_kv_cache
    xqa_batch_decode_with_kv_cache

DCP Speculative Decode Workspace
--------------------------------

The native Cake FMHA DCP speculative route of
:func:`flashinfer.decode.trtllm_batch_decode_with_kv_cache` uses caller-owned
scratch buffers so a prewarmed invocation can be captured in a CUDA Graph.
It is also reachable through
:func:`flashinfer.cake_fmha.cake_batch_decode_with_kv_cache`; the non-null
``causal_seqlens_kv_global`` argument is the explicit add-on selection key.
On SM103, the same Cake entrypoint accepts a device ``request_order`` tensor
for BF16-query, FP8-E4M3 paged decode with head dimension 256.  Precompute an
optional immutable length-aware schedule with
:func:`flashinfer.plan_cake_fmha_request_ordered_paged_decode` before graph
capture.  Page-table rows must be padded to
``4 * ceil(max_seq_len / 256)`` entries, and the exact tensor/workspace binding
must be invoked once eagerly to initialize its TMA descriptors before capture.
After that prewarm, changing only the order tensor contents does not require
recapture.

.. currentmodule:: flashinfer

.. autosummary::
    :toctree: ../generated

    get_dcp_spec_workspace_size_bytes
    get_dcp_spec_counter_bytes
    plan_cake_fmha_request_ordered_paged_decode
    CakeFmhaRequestOrderedDecodePlan

.. currentmodule:: flashinfer.decode

.. autoclass:: BatchDecodeWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    :members:

    .. automethod:: __init__


XQA
---

.. currentmodule:: flashinfer.xqa

.. autosummary::
    :toctree: ../generated

    xqa
    xqa_mla

Experimental: PagedAttention
============================

:class:`flashinfer.prefill.PagedAttention` is the experimental successor to
:class:`~flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper` for paged
attention over the existing fa2/fa3, cuDNN, trtllm-gen and (on SM100/SM103)
Cake kernels. The contract and lifecycle are described in the
`unified paged attention design doc <https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/paged_attention_unified_lifecycle.md>`_.
One :class:`~flashinfer.prefill.PagedAttentionMetadata` object per scheduler
step (token-unit ``qo_indptr``, per-request ``kv_seq_lens``, a dense block
table via ``.dense(...)`` or flat page ids via ``.csr(...)``, required host
maxes, optional CPU mirrors for a zero-sync plan; a ``kv_seq_lens`` entry of 0
marks a padding row, and a request with ``q_len == 0`` is legal too);
:func:`~flashinfer.prefill.resolve_paged_attention` answers at engine init
which backends can run a configuration and why the others cannot; ``plan()``
declares the LSE base (``lse_mode``) and the feature axes (``window_left``,
``logits_soft_cap``, ``custom_mask``, ``use_sinks``), each capability-checked
so ``backend="auto"`` excludes a backend that cannot apply one instead of
dropping it; ``run()`` takes the per-layer ``sm_scale``, the ``sinks`` tensor
when planned, and the per-tensor ``k_scale`` / ``v_scale`` (dequantization
scales of an fp8 e4m3 / e5m2 KV cache, plain multipliers of a fp16 / bf16
one; ``k_scale`` folds into the softmax scale, ``v_scale`` into the output,
on every backend). ``explain()`` prints the chosen backend, the candidates in
the order the last plan walked them, every candidate tried at plan time and
the resolve-time exclusion reasons. CUDA graphs follow a
three-stage lifecycle:
``PagedAttention(graph_capacity=GraphCapacity(...))`` reserves the metadata
storage of one graph bucket at construction (``use_cuda_graph=True`` infers
the capacity from the first plan); the first graph-mode ``plan()`` freezes the
backend and every semantic argument, and a later plan that changes one is
rejected (a re-plan chooses from the frozen backend alone, so ``explain()``
then lists only that backend in the order and the trace);
calling ``plan()`` again with the same arguments then stages each step's
batch into the reserved storage so the captured ``run()`` replays it. A batch must fit the
capacity: batch size, paging form and page size exactly, total query tokens
and host maxes at most the capacity's, which is what the kernels are planned
with. Run once eagerly after the plan and before capturing, with ``q`` in
the stride layout the capture will use (each new ``q`` stride layout costs
one cuDNN graph build inside ``run()``); with the Cake
backend that eager ``run()`` must use the very ``q`` / ``k_cache`` /
``v_cache`` tensors the capture will bind (its TMA descriptors are created
eagerly and pinned at capture; a new binding inside capture raises a
``RuntimeError`` asking for the prewarm). The kernel workspace is shared per device (a 512 MiB pool) or
caller-supplied via ``workspace_buffer=``, so one instance per graph bucket
costs no workspace per bucket; ``PagedAttention.workspace_requirements()``
returns, for a :class:`~flashinfer.prefill.GraphCapacity` and a model
configuration, a conservative bound on the bytes the resolvable backends'
planners carve out of that buffer (the fa2 split-KV scratch dominates), so an
engine can size the buffer before capture. A backend whose planner would
overflow the buffer declines the batch at ``plan()``: ``auto`` moves on to a
candidate that fits, and a named backend, or a candidate set in which none
fits, raises a ``ValueError`` naming the required bytes (``backend 'fa2'
cannot plan this batch: needs N bytes of scratch workspace ...``). ``run()``
takes ``q`` as any view that is dense along ``head_dim``, stored token-major
and 16-byte aligned (head stride at least ``head_dim_qk``, token stride at
least ``(num_qo_heads - 1) * head stride + head_dim_qk``, both strides
multiples of 16 bytes, data pointer 16-byte aligned), for example the head
slice of a fused QKV buffer; any other layout raises a ``ValueError`` on
every backend. Head counts, head dimensions, ``page_size``, ``max_q_len``
and ``max_kv_len`` must be Python ints (a ``bool`` or a numpy integer is
rejected). Calling
any of these is the opt-in (an
``ExperimentalWarning`` is emitted once); see the tracking issue
`#5007 <https://github.com/flashinfer-ai/flashinfer/issues/5007>`_ for the
graduation plan.

Non-causal attention with a sliding window has no runnable backend in this
API today: trtllm-gen and cake ship no such kernel, this API's cuDNN backend
declares no sliding window (the legacy wrapper's explicit cuDNN backend runs
one; design doc, known limitation D3), and the fa2 kernel computes it wrong
for requests longer than 128 query tokens with a history (design doc, known
limitation M20), so fa2/fa3 declare it unsupported and
``resolve_paged_attention`` reports the reason per backend.

The slots of a request's last page past its ``kv_seq_lens`` entry are never
attended but must hold finite values: the FA kernels zero that tail before
the PV product, while trtllm-gen, cake and cuDNN load the whole page and a
NaN or Inf there reaches the output as ``0 * NaN`` (design doc, known
limitation M23). Any finite content — stale tokens, zeros — gives the exact
result; a KV pool allocated with ``torch.empty`` and never written past
``kv_len`` is not covered by that guarantee until its storage is initialised.

cuDNN finalizes one execution plan per declared shape (about 5 ms), so an
eager cuDNN plan declares a shape class instead of the batch's exact maxes,
on every architecture (the legacy wrapper does so only where its ``auto``
selects cuDNN on Blackwell; design doc, known limitation D5):
the KV bound rounds up to the next power of two, and a query bound above 64
tokens rounds up to whole ``128 // (num_qo_heads // num_kv_heads)``-token
tiles, at least 128, when that GQA group divides 128. The plans of a serving
loop whose KV grows every step then share one cuDNN plan per class: a new one
is built only when the KV bound crosses a power of two. The per-request
lengths stay the actual ones. Such a plan binds a page table of its own at
the class width, built on the device at ``plan()`` from each request's own
pages, so the columns of the caller's table past a request's pages do not
reach cuDNN. In CUDA-graph mode the backend declares the capacity's bounds
and binds the instance's reserved table, the caller's columns included,
without a copy. The pages those columns name must then hold finite values,
and so must pool page 0 unless the capacity's KV bound is a whole number of
128-token tiles (design doc, known limitations).

Backend selection and the ``max_q_len`` hint
--------------------------------------------

``backend="auto"`` is a fixed rule over host-side facts of the batch, not
autotuning: nothing is timed (``backend="autotune"``, below, measures
instead). It works on two levels.

:func:`~flashinfer.prefill.resolve_paged_attention` fixes the candidate
*set*: those of fa2, fa3, trtllm-gen and cuDNN that the capability table and
the environment probes admit for the configuration, with a reason for every
backend left out. ``cake`` is never an auto candidate; it runs only when
named (``backend="cake"``).

``plan()`` ranks that set for the batch and takes the first candidate that
accepts it, moving on only when a candidate declines the batch. The rank
reads host values only: the batch size, the total query tokens, the declared
``max_q_len`` and ``max_kv_len``, the per-request lengths from the metadata's
host mirrors, the head counts and dimensions, dtypes, layout, page size, the
feature flags and whether the instance is in CUDA-graph mode. No device
tensor is read. ``max_q_len`` and ``max_kv_len`` are the maxes the metadata
declares, in CUDA-graph mode the capacity's (a capacity inferred from a dense
block table takes the table's whole width in tokens as its ``max_kv_len``):
upper bounds, not the batch's longest lengths. An eager instance ranks on
every ``plan()``, so ``PagedAttention.backend`` can change from one batch to
the next. A CUDA-graph instance ranks on the bounds of its capacity and the
per-request lengths of the batch being planned, until its first successful
plan freezes the backend: later plans keep it. Plan the
first batch of a bucket with representative lengths, or pin the backend. A
backend name, or a ``Resolution`` resolved with one, pins the backend:
nothing is ranked, and a batch that backend cannot plan raises a
``ValueError``.

On SM100 (compute capability 10.0) the rule is:

1. cuDNN first inside the region in which the legacy wrapper's ``auto``
   selects it at head dimension 128. Every admission needs plain causal
   attention (no sliding window, soft cap, custom mask or sinks) with bf16 Q
   and KV, head dimension 128, page size 16, HND layout and 8 to 64 query
   heads in a GQA group of 1, 2, 4, 8 or 16. With ``work`` for
   ``num_qo_heads * q_len * kv_len`` summed over the requests and Mi for
   ``2**20``, two parts admit:

   - The bounded domain: at most four requests with at least 8 Mi of work,
     under declared maxes that are either a prefix, ``max_q_len`` of 64 to
     1024 with ``max_kv_len`` of 2048 to 32768 and at least four times
     ``max_q_len``, or a full prefill, ``max_q_len == max_kv_len`` of 1024
     to 8192 (in CUDA-graph mode ``1024 <= max_q_len <= max_kv_len <=
     8192``) with every request's query non-empty and as long as its KV and
     the longest at least 1024 tokens.
   - The length-based extension, for any number of requests: at least 16 Mi
     of work and every request with ``0 < q_len <= kv_len``; then either
     every ``q_len == kv_len`` with a longest query of 1024 to 16384 tokens,
     or every ``q_len`` above 128 with a longest query of at most 8192 and a
     longest KV of 2048 to 32768 tokens.

   The prefix and full-prefill tests read the declared maxes. The work, the
   proof that every query is as long as its KV and every test of the
   extension read the per-request lengths on the host mirrors, and "longest"
   in the extension is the batch's own. A batch that qualifies for the
   extension is therefore admitted under any maxes declared above its
   lengths, a full prefill under a ``max_kv_len`` sized for a longer
   context for instance. A batch the extension does not admit (less than
   16 Mi of work, for instance, or a request of at most 128 query tokens in
   a batch that is not all full prefill) needs the bounded domain, which
   reads the declared maxes: under a ``max_kv_len`` above 32768 it is
   outside the region whatever its lengths.

   In CUDA-graph mode the declared maxes are the capacity's and the lengths
   are those of the first planned batch. That batch alone decides the
   extension: one that qualifies freezes cuDNN, planned with the capacity's
   bounds, however far those are above it. The wider full-prefill test of
   that mode matters only for at most four requests of full prefill with at
   least 8 Mi and less than 16 Mi of work (8 to 15 query heads) under a
   capacity whose ``max_kv_len`` is above its ``max_q_len``; from 16 Mi the
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
   256 rows (``num_qo_heads * its query length / 128``) and whose ``rows *
   longest KV`` (``rows`` as in rule 2) is at most ``2**22``. On the 83
   measured shapes of that kind trtllm-gen's ``run()`` takes 0.67 to 0.97
   of the legacy wrapper's cuDNN time (geometric mean 0.83), and its
   ``plan()`` is cheaper too. CUDA-graph plans keep cuDNN first in the whole
   region.

2. Otherwise trtllm-gen first when the packed query rows fill its 128-row
   tiles. With ``rows = num_qo_heads * total_q_tokens / 128`` and ``fill =
   rows / (batch_size * num_qo_heads * ceil(max_q_len / 128))`` it needs
   ``rows >= 64`` (96 below head dimension 128) and ``fill >= 0.375`` in
   CUDA-graph mode or ``fill >= 0.75`` eager. Eager and above 148 rows it
   also needs a query row to attend at least 1024 keys (the batch's actual
   longest KV length, from the host mirror, or ``window_left + 1`` if that is
   smaller) or ``rows * keys >= 2**19``: on a shorter KV the kernels take
   tens of microseconds, the host cost of the launch decides, and fa2's is
   lower.
3. Otherwise fa2 first.

The remaining candidates follow as fallbacks: the other one of fa2 and
trtllm-gen, then cuDNN outside its region. The predicate of the region and
the thresholds are in
``flashinfer/experimental/paged_attention/_auto_policy.py``; the design doc
describes the policy under "Level 2: the auto policy".

The other architectures have fixed orders, selected by the major of the
compute capability: SM90 ``fa3, fa2, cudnn``, with fa2 first when
``max_q_len`` is at most 64 (the legacy wrapper's short-query rule; its
cuDNN admission on H100 at head dimension 256 is not copied, design doc,
known limitation D2); the other SM10x parts ``trtllm-gen, cudnn, fa2``,
with fa2 first when ``max_q_len`` is at most 16; SM8x and SM12x
``fa2, cudnn``, whatever the shape. ``max_q_len`` is the declared bound
here too (the capacity's in CUDA-graph mode). Only the SM100 rule is fitted
to measurements of this API.

The ``max_q_len`` hint of :func:`~flashinfer.prefill.resolve_paged_attention`
is a promise about the batches planned with the Resolution: they have at most
this many query tokens per request (1 for plain decode, the draft length for
speculative decoding; leave it out for prefill or mixed batches). It does two
things. ``plan()`` checks it: a batch (in CUDA-graph mode: a capacity) with a
larger ``max_q_len`` is rejected with a ``ValueError``, so hold one Resolution
per query-length bucket, as an engine's decode / prefill split already does. And
it orders the resolve-time candidate list (``Resolution.backends``,
``Resolution.chosen`` and what ``Resolution.explain()`` prints), which is the
estimate available before a batch exists: on compute-capability major 10 a
hint ``<= 16`` lists ``fa2, trtllm-gen, cudnn`` and a larger hint, or none,
``trtllm-gen, cudnn, fa2``; the other majors have one order. The hint does
not decide the backend of a plan: ``plan()`` ranks the candidates from the
batch itself, so a hinted and an unhinted Resolution plan the same batch on
the same backend. It never changes the candidate set or the exclusions. It is
pinned in the Resolution (``explain()`` prints it; Resolutions of one model
configuration with different hints have different ``config`` keys).

Measured selection: ``backend="autotune"``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``backend="autotune"`` selects within the same candidate set as ``"auto"``
by measurement. It follows ``backend="autotune"`` of
``BatchMLAPagedAttentionWrapper``: tune the planned workload inside
``flashinfer.autotune(True)``, serve the cached result outside it.
:func:`~flashinfer.prefill.resolve_paged_attention` with
``backend="autotune"`` returns the ``"auto"`` set with
``Resolution.autotune`` set, so ``plan(backend=resolution)`` carries the mode.

.. code-block:: python

    attn = flashinfer.prefill.PagedAttention(device)
    with flashinfer.autotune(True, cache="paged_attention.json"):
        attn.plan(metadata, backend="autotune", **config)  # prepares every candidate
        attn.run(q, (k_cache, v_cache), out=out)           # profiles, keeps the winner

    attn.plan(metadata, backend="autotune", **config)
    attn.run(q, (k_cache, v_cache), out=out)               # cached result, no profiling

``plan()`` prepares every candidate that accepts the batch.
:attr:`~flashinfer.prefill.PagedAttention.backend` is ``"autotune"`` for the
whole plan; ``explain()`` lists the candidates and, after the first
``run()``, the selected one (``chosen: autotune (selected <name>)``).
``run()`` requires
``out=`` under this mode. The first ``run()`` selects one candidate and keeps
it until the next plan:

- A result cached for the planned workload and this call's tensor layouts
  and options is used (this process, or a file loaded with
  ``flashinfer.autotune(cache=...)``).
- Otherwise, inside ``flashinfer.autotune(True)``, each candidate is
  launched once and profiled by the autotuner on the caller's q and KV cache
  with private output buffers; the winner is cached.
- Otherwise it raises ``RuntimeError``: warm up inside the context first.
  A context that skips the operation (``skip_ops``) raises as well.

The cache identity is the planned workload, not a shape bucket: the resolved
configuration, the mode (eager or CUDA graph), the bounds the kernels are
planned with, a digest of the batch's metadata (the per-request query and KV
lengths and the page ids), the workspace size, the device and the prepared
candidates. A batch with other lengths or other pages is another workload.
Computing the digest reads a device page table back, so an autotune plan
synchronizes.

The selecting ``run()`` cannot be under CUDA graph capture. In CUDA-graph
mode it is the eager warm-up before capture: a re-plan is refused until it
has run, and the selected backend is the one the instance freezes, so later
plans re-plan that backend alone. An eager instance prepares
every candidate on each ``plan()``, which costs more host time than an
``"auto"`` plan. A named backend never enters the tuner.

The SM100 rule is measured against
``BatchPrefillWithPagedKVCacheWrapper(backend="auto")`` on GB200
(cudnn-frontend 1.32 / cuDNN 9.27, 2026-11-18; bf16, ``run()`` medians, every
shape eager and as a CUDA-graph replay). A shape passes when no paired
measurement block is more than 1% slower than the legacy time and is a
regression when every block is more than 1% slower:

================  ==========  ===============  ====  ============  ==========
shapes            mode        geomean speedup  pass  inconclusive  regression
================  ==========  ===============  ====  ============  ==========
219 (main list)   CUDA graph  1.49x            163   47            9
219 (main list)   eager       1.41x            205   14            0
661 (all lists)   CUDA graph  1.39x            494   151           16
661 (all lists)   eager       1.20x            575   85            1
================  ==========  ===============  ====  ============  ==========

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
size 16, causal, no LSE; a dense block table exactly ``kv_len / 16`` columns
wide, so a CUDA-graph capacity equals the shape): ``run()`` medians in µs of
each backend named through this API, and the backend ``auto`` ran. The
benchmark script below builds a table three columns wider; with CUDA graphs
on and the dense form, its ``auto`` ranks on that width as the KV bound. A
wider KV bound can put a shape outside the cuDNN region only through the two
bound tests: the last shape (a full prefill of 8192 tokens) is admitted by
its lengths under either bound.

=====  =====  ======  ==========  ====  ==========  =====  ==========
batch  q_len  kv_len  mode        fa2   trtllm-gen  cuDNN  auto
=====  =====  ======  ==========  ====  ==========  =====  ==========
32     16     4096    CUDA graph  224   392         469    fa2
32     16     4096    eager       154   440         514    fa2
32     64     4096    CUDA graph  446   396         472    trtllm-gen
32     64     4096    eager       435   444         517    fa2
8      96     2048    CUDA graph  122   66          76     trtllm-gen
8      96     2048    eager       130   106         150    trtllm-gen
8      128    512     CUDA graph  50    28          46     trtllm-gen
8      128    512     eager       55    66          88     fa2
1      128    16384   CUDA graph  115   193         52     cudnn
1      128    16384   eager       133   230         100    cudnn
1      8192   8192    CUDA graph  1690  443         365    cudnn
1      8192   8192    eager       1588  492         424    cudnn
=====  =====  ======  ==========  ====  ==========  =====  ==========

Benchmarking the unified API
----------------------------

``benchmarks/bench_paged_attention.py`` is a standalone script (the API is
experimental, so it is not a routine of the shared benchmark CLI)::

    python benchmarks/bench_paged_attention.py \
        --backends fa2 cudnn trtllm-gen auto --batch_size 4 --s_qo 128 \
        --s_kv 1024 --num_qo_heads 32 --num_kv_heads 8 --head_dim_qk 128 \
        --page_size 16 --causal --kv_layout HND --lse_mode basee --refcheck

Every requested backend sees the same inputs (packed Q, a paged K/V pool with
a shuffled page mapping, exact per-request lengths) and is checked against the
fp32 oracle before it is timed (always; ``--pa_skip_refcheck``
opts out for shapes the oracle cannot hold). The script reports one CSV row
per phase:
``plan`` (metadata construction plus ``plan()``, synchronized host wall time),
``run`` (a warmed ``run()`` with preallocated outputs; GPU time under a CUDA
graph by default, eager with ``--no_cuda_graph``) and ``step`` (one plan
followed by N ``run()`` calls, N from ``--pa_layers``). Use ``--s_qo 1`` for
decode, ``--kv_input_form csr`` for flat page indices,
``--random_actual_seq_len`` for variable lengths, ``--pa_max_q_len_hint N``
to resolve with the ``max_q_len`` hint (it bounds the batches and orders the
resolve-time list that ``static_backend`` reports; the ``auto`` rows run what
``plan()`` ranks first for the batch) and ``--pa_legacy`` to add
rows for the same inputs through the legacy public API of the same kernel
(``api_variant=legacy``). Rows for unsupported,
failing or incorrect backends are kept with a ``status`` column and a
``refcheck_passed`` column; ``static_backend`` records the first candidate in
resolve-time order and ``resolved_backend`` the backend the plan ran. Under
``auto`` the two differ when ``plan()`` ranks another candidate first for the
batch or when a candidate declines it (``auto`` is a fixed rule, not
autotuning). ``--backends autotune`` warms each instance up inside
``flashinfer.autotune(True)`` and then times the selected backend; its eager
``plan`` rows include preparing every candidate. The benchmark generates fp16/bf16 Q and KV; see
``benchmarks/README.md`` for the columns.

Tracing the unified API
-----------------------

``run()`` carries a `flashinfer-bench <https://github.com/flashinfer-ai/flashinfer-bench>`_
trace template. After ``plan()``, ``flashinfer.fi_trace(attn.run, q=q,
kv_cache=(k, v))`` returns (and with ``save_dir=`` writes) the definition of
the planned contract, and ``FLASHINFER_TRACE_DUMP=1`` exports it automatically
on the first ``run()`` of each definition. The trace is read-only and
sync-free: it takes the plan-owned ``qo_indptr``, ``kv_seq_lens`` and the
page table (the dense block table, or the live prefix of the flat page ids)
from the last successful ``plan()``, so it describes what ``run()`` executes
even after a rejected re-plan, and in CUDA-graph mode it points at the
reserved storage a captured ``run()`` reads. The identity of a definition is
the plan's paging form (``paged_attention_dense`` / ``paged_attention_csr``,
``_fp8kv`` for an fp8 KV cache), Q and KV dtypes, KV layout, causal flag,
sliding window, LSE mode and geometry, encoded as integer Const axes so the
exported reference can be called with the same values and the exported
``init`` rebuilds the traced variant from the JSON alone: ``csr`` 0/1 = dense
block table / flat page ids, ``fp8_kv`` 0/1 = K/V in q's dtype /
float8_e4m3fn (the init then plans with that ``kv_dtype`` and puts
``k_scale`` / ``v_scale`` in its run bundle), ``q_dtype`` 0/1 =
bfloat16/float16, ``kv_layout`` 0/1 = HND/NHD, ``causal`` 0/1,
``window_left`` -1 = unlimited, ``lse_mode`` 0/1/2 = none/base-2/natural
log. The resolved backend is not part of the identity, so fa2, fa3, cuDNN and
trtllm-gen traces of one plan compare against the same definition. Tracing
tensors whose dtypes differ from the plan's is refused. The plan's feature
knobs are Const axes too: ``logits_soft_cap`` (the integer cap of ``cap *
tanh(score / cap)``, applied to the scaled scores before masking; 0 = off; a
non-integer cap refuses to trace), ``use_sinks`` (a ``sinks`` input of
``(num_qo_heads,)`` fp32 logits, one per head in the softmax denominator with
no value; the LSE includes it; pass ``sinks=`` to the trace as to ``run()``)
and ``use_custom_mask`` (the plan-owned flattened bool ``custom_mask`` input
with a ``mask_len`` axis, ANDed with the causal / window envelope). Each
enters the name only when on (``_cap30``, ``_sinks1``, ``_mask1``), the
reference applies them, and ``init`` rebuilds them. Tracing requires the planned
instance: ``PagedAttention.run.fi_trace(...)`` and a trace before ``plan()``
raise instead of guessing; a batch with padding rows (``kv_len == 0``)
refuses to trace, which the
definition's ``min(kv_seq_lens) >= 1`` constraint excludes. Requests without
query rows (``q_len == 0``, vLLM's padded ``query_start_loc`` tail) are part
of the definition: its constraint is ``min(q_len) >= 0``, the reference
skips them, and the exported ``init`` produces them when ``total_q <
batch_size``.
The fa2/fa3 backends launch the kernel module from an argument list staged
at ``plan()`` and do not enter ``BatchPrefillWithPagedKVCacheWrapper.run()``,
so ``run()`` of such a plan produces the unified API-log entry, dump and
definition only. The wrapper's ``run()`` log entry and level-10 dump, its
nested ``gqa_paged_prefill`` definition and Trace Apply under that definition
name apply only to the plans that go through the wrapper's ``run()``: a
plan with attention sinks, and a plan whose launch cannot be staged.

.. currentmodule:: flashinfer.prefill

.. autosummary::
    :toctree: ../generated

    resolve_paged_attention

.. autoclass:: PagedAttention
    :members: plan, run, explain, backend, workspace_requirements

    .. automethod:: __init__

.. autoclass:: PagedAttentionMetadata
    :members: dense, csr

.. autoclass:: GraphCapacity
    :members: from_metadata

flashinfer.prefill
==================

Attention kernels for prefill & append attention in both single request and batch serving setting.

.. currentmodule:: flashinfer.prefill

Single Request Prefill/Append Attention
---------------------------------------

.. autosummary::
    :toctree: ../generated

    single_prefill_with_kv_cache
    single_prefill_with_kv_cache_return_lse
    single_prefill_with_kv_cache_with_jit_module

Batch Prefill/Append Attention
------------------------------

.. autosummary::
    :toctree: ../generated

    cudnn_batch_prefill_with_kv_cache
    trtllm_batch_context_with_kv_cache
    trtllm_ragged_attention_deepseek
    fmha_v2_prefill_deepseek
    trtllm_fmha_v2_prefill
    fmha_v2_prefill_sm120

.. autoclass:: BatchPrefillWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__

.. note::

    :class:`BatchPrefillWithPagedKVCacheWrapper` is **superseded** by the
    experimental :class:`~flashinfer.prefill.PagedAttention` (above) and is
    scheduled for deprecation once that API graduates (tracking:
    `#5007 <https://github.com/flashinfer-ai/flashinfer/issues/5007>`_). It
    remains fully supported and receives bug fixes; new integrations should
    start from the unified API, which is the only path on which ``backend="auto"``
    can select the trtllm-gen kernel (the wrapper's own ``auto`` chooses
    between fa2 / fa3 and cuDNN, the latter inside its SM100 domains and,
    at head dimension 256, on H100 80GB).

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, forward, forward_return_lse

    .. automethod:: __init__


Causal + Bidirectional Ranges Prefill
-------------------------------------

.. currentmodule:: flashinfer.attention

A batch-prefill wrapper whose fa2 attention variant owns the whole mask:
causal, plus an inclusive per-query key span attended in both directions. The
spans are handed to :meth:`BatchPrefillWithCausalBidirectionalRangesWrapper.run`
as a compact ``int32 [total_q, 2]`` tensor and no mask is materialized, so
nothing scales with ``qo_len * kv_len``. The JIT module is specialized in the
constructor, and the inherited options the variant makes meaningless are
rejected rather than ignored.

.. autoclass:: BatchPrefillWithCausalBidirectionalRangesWrapper
    :members:

    .. automethod:: __init__


Unified BatchAttention
----------------------

.. currentmodule:: flashinfer.attention

The ``BatchAttention`` class provides a holistic attention wrapper that automatically dispatches
between paged-prefill and paged-decode based on per-request sequence lengths. It is the
recommended entry point for serving stacks that batch mixed prefill/decode requests in a
single kernel launch.

.. autoclass:: BatchAttention
    :members:

    .. automethod:: __init__

.. autoclass:: BatchAttentionWithAttentionSinkWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchDecodeWithAttentionSinkWrapper

    .. automethod:: __init__


SM120 NVFP4 Attention
---------------------

.. currentmodule:: flashinfer.nvfp4_attention_sm120

.. autosummary::
    :toctree: ../generated

    nvfp4_attention_sm120_quantize_qkv
    nvfp4_attention_sm120_fwd


flashinfer.mla
==============

MLA (Multi-head Latent Attention) is an attention mechanism proposed in DeepSeek series of models (
`DeepSeek-V2 <https://arxiv.org/abs/2405.04434>`_, `DeepSeek-V3 <https://arxiv.org/abs/2412.19437>`_,
and `DeepSeek-R1 <https://arxiv.org/abs/2501.12948>`_).

.. currentmodule:: flashinfer.mla

Sliding-window MLA (SM90)
-------------------------

.. autofunction:: batch_decode_with_kv_cache_mla_swa
.. autofunction:: get_mla_swa_workspace_size_bytes

PageAttention for MLA
---------------------

.. autosummary::
    :toctree: ../generated

    trtllm_batch_decode_with_kv_cache_mla
    trtllm_prefill_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    nvfp4_quantize_pack_sparse_mla_cache
    nvfp4_quantize_append_sparse_mla_cache
    dsv41_fp4_quantize_pack_sparse_mla_cache
    dsv41_fp4_quantize_append_sparse_mla_cache
    dsv41_fp8_quantize_pack_sparse_mla_cache
    dsv41_fp8_quantize_append_sparse_mla_cache
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla
    supported_sparse_mla_sm120_configs
    SparseMLASm120DecodeConfig
    SparseMLASm120Wrapper
    cake_sparse_mla_sm120_dsv4_nvfp4_decode
    cake_sparse_mla_sm120_dsv4_nvfp4_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_prefill
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_head_tiles
    cake_sparse_mla_sm120_dsv4_nvfp4_plan_splits
    cake_sparse_mla_sm120_dsv4_nvfp4_scratch_bytes
    cake_dsv4_nvfp4_rope_quantize_insert
    cake_dsv4_nvfp4_kv_rope_quantize_insert

.. note::

    Experimental ``SparseMLASm120Wrapper.run_mega`` for FP8 caches with ``backend="sparse"``
    fuses query RoPE, sparse attention, inverse output RoPE, and per-32-value
    FP8 E4M3 output quantization. Inputs are BF16 ``q[T,H,512]``, int32
    ``positions[T]``, and FP32 ``cos_sin[num_positions,64]``. Output buffers
    are float8_e4m3fn ``output[T,H,512]`` and uint8 ``output_scales[T,H,16]``
    (UE8M0). Positions, the RoPE table, and output buffers must be contiguous.

    RoPE rotates adjacent pairs in the final 64 features. Each RoPE table
    row contains 32 cosines followed by 32 sines. Q uses forward rotation;
    output uses inverse rotation at the same query position. This follows
    FlashMLA's
    `RoPE parameters <https://github.com/deepseek-ai/FlashMLA/blob/395cb3afbb331fef7b1b21e9340861b9436b42d8/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h#L27-L31>`_.
    Positions must be in bounds and input/output storage must not overlap.
    Q normalization and sparse-index construction are caller responsibilities.
    Cached keys must already have RoPE applied. Output uses ``[T,H,512]``.
    Warm up before CUDA graph capture. Other attention keywords follow
    ``run`` semantics, including sparse lengths, the second cache and LSE.

.. note::

    With ``backend="cake"`` on SM120/SM121 (``kv_cache_format="nvfp4"``), every
    call runs the Cake split decode kernel or the single-launch prefill kernel
    as chosen by ``cake_sparse_mla_sm120_dsv4_nvfp4_select_kernel`` from the
    token, head and candidate counts and the SM count (DeepSeek-V4 sparse MLA
    SM120 tracker: flashinfer#4254).

.. note::

    The NVFP4 cache that route reads is written per layer by the fused
    SM120/SM121 writers ``cake_dsv4_nvfp4_rope_quantize_insert`` (sliding-window
    pool: GPT-J RoPE of the query and latent KV, head-padded ``q_out`` or, with
    ``q_inplace=True`` and no head padding, the query rotated in place, NVFP4
    quantization and paged insert in one launch) and
    ``cake_dsv4_nvfp4_kv_rope_quantize_insert`` (compressed pool with
    ``compress_ratio`` 1 or 2, speculative context with ratio 1); both produce
    the bytes of ``nvfp4_quantize_append_sparse_mla_cache`` applied to the
    BF16-rounded roped rows.

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

.. note::

    ``kv_cache_format="nvfp4"`` (the 384-byte-per-token DeepSeek-V4 NVFP4 sparse
    cache: 448 NoPE values as E2M1 with one E4M3 scale per 16 values, 64 BF16
    RoPE values, ``page_size * 352`` data bytes followed by ``page_size * 32``
    scale bytes per page) is consumed by ``backend="sparse"`` on SM120 / SM121 and
    by ``backend="cake"`` on SM100 / SM103 (B200 / GB300). The CAKE route takes a
    BF16 query, two independent tables (``sparse_indices`` over ``swa_kv_cache``
    and ``extra_sparse_indices`` over ``compressed_kv_cache`` with their own
    ``*_topk_lens``; ``-1`` entries are masked), ``sinks``, a caller-owned
    ``workspace_buffer`` sized by
    :func:`flashinfer.mla.cake_dsv4.get_cake_dsv4_workspace_bytes` and is
    CUDA-Graph safe. Build the cache with
    :func:`nvfp4_quantize_pack_sparse_mla_cache` /
    :func:`nvfp4_quantize_append_sparse_mla_cache` (one implementation for all
    four architectures). ``backend="auto"`` keeps selecting TRTLLM-GEN on
    SM100 / SM103; pass ``backend="cake"`` explicitly.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__
