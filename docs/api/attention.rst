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

.. currentmodule:: flashinfer.attention.prims_ts

FMHA Context/Prefill
--------------------

.. autosummary::
    :toctree: ../generated

    batch_prefill
    batch_prefill_with_paged_kv_cache

.. autoclass:: BatchPrefillTSWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BatchPrefillPagedTSWrapper
    :members:

    .. automethod:: __init__

FMHA Decode
-----------

.. autosummary::
    :toctree: ../generated

    batch_decode_with_paged_kv_cache
    get_prims_ts_batch_decode_workspace_size
    prepare_prims_ts_batch_decode_with_kv_cache
    prims_ts_batch_decode_with_kv_cache

.. autoclass:: PrimsTSBatchDecodePlan
    :members:

.. autoclass:: BatchDecodePagedTSWrapper
    :members:

    .. automethod:: __init__

QToken-KvBlock-Sparse-Attention
--------------------------------

QToken-KvBlock-Sparse-Attention consumes per-query
``indexer_block_ids[total_q, block_topk]`` and a dense physical
``block_table``. Packed prefill uses ``[total_q, Hq, D]`` with
``qo_indptr``; fixed MTP decode uses ``[B, Nq, G, Hq, D]``.
``kv_block_size`` is the semantic sparse K/V atom and currently supports
only four tokens. The wrapper plans capacity outside CUDA Graph capture and
runs live route metadata on the hot path.

.. autosummary::
    :toctree: ../generated

    QTokenKvBlockSparsePagedTSWrapper
    get_q_token_kv_block_sparse_workspace_size
    q_token_kv_block_sparse_attention_with_paged_kv_cache
    validate_q_token_kv_block_sparse_group_size
    suggest_q_token_kv_block_sparse_group_size
    make_q_token_kv_block_sparse_qo_indptr

.. autoclass:: QTokenKvBlockSparsePagedTSWrapper
    :members:

Block-Sparse FMHA
-----------------

.. autosummary::
    :toctree: ../generated

    block_sparse_attention
    block_sparse_attention_with_paged_kv_cache

.. autoclass:: BlockSparseTSWrapper
    :members:

    .. automethod:: __init__

.. autoclass:: BlockSparsePagedTSWrapper
    :members:

    .. automethod:: __init__

MLA Decode
----------

.. autosummary::
    :toctree: ../generated

    batch_mla_decode_with_paged_kv_cache
    get_prims_ts_batch_mla_decode_workspace_size
    prims_ts_batch_mla_decode_with_kv_cache

.. autoclass:: BatchMLADecodePagedTSWrapper
    :members:

    .. automethod:: __init__


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
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. autoclass:: BatchDecodeMlaWithPagedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

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
:func:`resolve_paged_attention` answers at engine init
which backends can run a configuration and why the others cannot; ``plan()``
declares the LSE base (``lse_mode``) and the feature axes (``window_left``,
``logits_soft_cap``, ``custom_mask``, ``use_sinks``), each capability-checked
so ``backend="auto"`` excludes a backend that cannot apply one instead of
dropping it; ``run()`` takes the per-layer ``sm_scale``, the ``sinks`` tensor
when planned, and the per-tensor ``k_scale`` / ``v_scale`` (dequantization
scales of an fp8 e4m3 / e5m2 KV cache, plain multipliers of a fp16 / bf16
one; ``k_scale`` folds into the softmax scale, ``v_scale`` into the output,
on every backend). ``explain()`` prints the chosen backend, every candidate tried
at plan time and the resolve-time exclusion reasons. CUDA graphs follow a
three-stage lifecycle:
``PagedAttention(graph_capacity=GraphCapacity(...))`` reserves the metadata
storage of one graph bucket at construction (``use_cuda_graph=True`` infers
the capacity from the first plan); the first graph-mode ``plan()`` freezes the
backend and every semantic argument, and a later plan that changes one is
rejected; ``update(metadata)`` then re-plans each step's batch into the
reserved storage so the captured ``run()`` replays it. A batch must fit the
capacity: batch size, paging form and page size exactly, total query tokens
and host maxes at most the capacity's, which is what the kernels are planned
with. Run once eagerly after the plan and before capturing; with the Cake
backend that eager ``run()`` must use the very ``q`` / ``k_cache`` /
``v_cache`` tensors the capture will bind (its TMA descriptors are created
eagerly and pinned at capture; a new binding inside capture raises a
``RuntimeError`` asking for the prewarm). The kernel workspace is shared per device (a 512 MiB pool) or
caller-supplied via ``workspace_buffer=``, so one instance per graph bucket
costs no workspace per bucket; ``PagedAttention.workspace_requirements()``
returns, for a :class:`~flashinfer.prefill.GraphCapacity` and a model
configuration, a conservative bound on the bytes the resolvable backends'
planners carve out of that buffer (the fa2 split-KV scratch dominates), so an
engine can size the buffer before capture, and ``plan()`` rejects a buffer a
batch would overflow with a ``ValueError`` naming the required bytes. Calling
any of these is the opt-in (an
``ExperimentalWarning`` is emitted once); see the tracking issue
`#5007 <https://github.com/flashinfer-ai/flashinfer/issues/5007>`_ for the
graduation plan.

Non-causal attention with a sliding window has no runnable backend today:
trtllm-gen and cake ship no such kernel, cuDNN has no sliding window, and
the fa2 kernel computes it wrong for requests longer than 128 query tokens
with a history (design doc, known limitation M20), so fa2/fa3 declare it
unsupported and ``resolve_paged_attention`` reports the reason per backend.

Backend selection and the ``max_q_len`` hint
--------------------------------------------

``backend="auto"`` is a **static** selection, not autotuning: nothing is
timed, ``resolve_paged_attention`` ranks the runnable backends by a small
per-architecture table (``HEURISTIC_ORDER``), and ``plan()`` walks that
order, moving on only when a candidate declines the batch. The one shape
fact the table reads is the optional ``max_q_len`` hint of
:func:`resolve_paged_attention`: "the batches planned with this Resolution
have at most this many query tokens per request" (1 for plain decode, the
draft length for speculative decoding; leave it out for prefill or mixed
batches). The hint selects the candidate *order*, never the candidate set or
the exclusions, so it cannot change what a plan computes. It is pinned in the
Resolution (``explain()`` prints it; Resolutions of one model configuration
with different hints have different ``config`` keys), and a ``plan()`` whose
batch (in CUDA-graph mode: whose capacity) has a larger ``max_q_len`` is
rejected with a ``ValueError`` — hold one Resolution per query-length bucket,
as an engine's decode / prefill split already does.

The table is seeded from measurements. On sm_100 (B200, bf16, 32 query / 8
KV heads, head_dim 128, page size 16, causal, dense block table; CUDA-graph
``run`` medians of the ``PagedAttention`` benchmark routine, in µs) the
trtllm-gen and cake backends run the paged *context* kernel for every batch,
which at decode and speculative query lengths reads a request's whole KV for
a handful of tokens; fa2's split-KV schedule is up to 5x faster there and the
context kernel is ahead from 64 (short kv) to 256 (kv 16K) query tokens per request:

==========  ==========  =====  =======  ==========  ========================
batch       kv_len      q_len  fa2      trtllm-gen  auto without / with hint
==========  ==========  =====  =======  ==========  ========================
32          4096        1      92       513         trtllm-gen / fa2
32          4096        8      247      588         trtllm-gen / fa2
32          4096        16     248      618         trtllm-gen / fa2
32          4096        64     491      667         trtllm-gen / trtllm-gen
8           4096        256    487      168         trtllm-gen / trtllm-gen
1           16384       1      22       209         trtllm-gen / fa2
==========  ==========  =====  =======  ==========  ========================

Hence on sm_100 a hint ``<= 16`` ranks fa2 first (``fa2, trtllm-gen, cake,
cudnn``) and a larger hint, or none, keeps the default order (``trtllm-gen,
cake, cudnn, fa2``); sm_80 / sm_90 / sm_120 have a single order. Numbers were
taken on a shared, unlocked GPU and are the basis of the table, not a
performance claim; the full sweep is in the WP-K report. Without a hint the
selection is exactly what it was before the hint existed.

Benchmarking the unified API
----------------------------

The shared benchmark CLI provides a ``PagedAttention`` routine::

    python benchmarks/flashinfer_benchmark.py --routine PagedAttention \
        --backends fa2 cudnn trtllm-gen auto --batch_size 4 --s_qo 128 \
        --s_kv 1024 --num_qo_heads 32 --num_kv_heads 8 --head_dim_qk 128 \
        --page_size 16 --causal --kv_layout HND --lse_mode basee --refcheck

Every requested backend sees the same inputs (packed Q, a paged K/V pool with
a shuffled page mapping, exact per-request lengths) and is checked against the
fp32 oracle before it is timed (always for this routine; ``--pa_skip_refcheck``
opts out for shapes the oracle cannot hold). The routine reports one CSV row
per phase:
``plan`` (metadata construction plus ``plan()``, synchronized host wall time),
``run`` (a warmed ``run()`` with preallocated outputs; GPU time under a CUDA
graph by default, eager with ``--no_cuda_graph``) and ``step`` (one plan
followed by N ``run()`` calls, N from ``--pa_layers``). Use ``--s_qo 1`` for
decode, ``--kv_input_form csr`` for flat page indices,
``--random_actual_seq_len`` for variable lengths, ``--pa_max_q_len_hint N``
to resolve with the ``max_q_len`` hint (the ``auto`` rows then follow the
hinted order) and ``--pa_legacy`` to add
rows for the same inputs through the legacy public API of the same kernel
(``api_variant=legacy``). Rows for unsupported,
failing or incorrect backends are kept with a ``status`` column and a
``refcheck_passed`` column; ``static_backend`` records the facade's static
choice and ``resolved_backend`` the backend the plan actually ran, which
differ when the static choice declined the batch at plan time (``auto`` is a
static selection, not autotuning). The benchmark generates fp16/bf16 Q and KV; see
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
the plan's paging form (``paged_attention_dense`` / ``paged_attention_csr``),
KV layout, causal flag, sliding window, LSE mode and geometry, encoded as
integer Const axes so the exported reference can be called with the same
values: ``kv_layout`` 0/1 = HND/NHD, ``causal`` 0/1, ``window_left`` -1 =
unlimited, ``lse_mode`` 0/1/2 = none/base-2/natural log. The resolved backend
is not part of the identity, so fa2, fa3, cuDNN and trtllm-gen traces of one
plan compare against the same definition. Tracing requires the planned
instance: ``PagedAttention.run.fi_trace(...)`` and a trace before ``plan()``
raise instead of guessing; a plan that uses ``logits_soft_cap``, a custom
mask or attention sinks refuses to trace until the definition encodes them,
and so does a batch with padding rows (``kv_len == 0``), which the
definition's ``min(kv_seq_lens) >= 1`` constraint excludes.
With the fa2/fa3 backends auto-dump also emits the nested legacy
``gqa_paged_prefill`` definition of the wrapper they run on.

.. currentmodule:: flashinfer.prefill

.. autosummary::
    :toctree: ../generated

    resolve_paged_attention

.. autoclass:: PagedAttention
    :members: plan, update, run, explain, backend, workspace_requirements

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
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

    .. automethod:: __init__

.. note::

    :class:`BatchPrefillWithPagedKVCacheWrapper` is **superseded** by the
    experimental :class:`~flashinfer.prefill.PagedAttention` (above) and is
    scheduled for deprecation once that API graduates (tracking:
    `#5007 <https://github.com/flashinfer-ai/flashinfer/issues/5007>`_). It
    remains fully supported and receives bug fixes; new integrations should
    start from the unified API, which is the only path on which ``backend="auto"``
    can select the cuDNN and trtllm-gen kernels.

.. autoclass:: BatchPrefillWithRaggedKVCacheWrapper
    :members:
    :exclude-members: begin_forward, end_forward, forward, forward_return_lse

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

PageAttention for MLA
---------------------

.. autosummary::
    :toctree: ../generated

    trtllm_batch_decode_with_kv_cache_mla
    trtllm_prefill_with_kv_cache_mla
    trtllm_batch_decode_sparse_mla_dsv4
    nvfp4_quantize_pack_sparse_mla_cache
    nvfp4_quantize_append_sparse_mla_cache
    convert_compressed_page_aligned_sparse_indices_to_hca_metadata
    DSV4HCAMetadata
    xqa_batch_decode_with_kv_cache_mla
    supported_sparse_mla_sm120_configs
    SparseMLASm120DecodeConfig
    SparseMLASm120Wrapper

.. note::

    With ``backend="cute-dsl"``, pass ``hca_swa_indices`` as absolute rows into
    the flattened SWA cache and ``hca_compressed_block_tables`` as physical
    compressed-cache page IDs. The SWA table has shape ``[B * Q, 128]`` and may
    express ring rotation or wraparound. Combined tables whose compressed
    segment is a canonical page expansion can opt into compatibility conversion
    with ``hca_sparse_indices_format="compressed-page-aligned"``. SWA entries
    remain arbitrary absolute rows. Precompute that conversion before a CUDA
    Graph or a latency-sensitive loop.

.. autoclass:: BatchMLAPagedAttentionWrapper
    :members:

    .. automethod:: __init__
