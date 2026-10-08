"""Legacy -> unified: tests/attention/test_batch_decode_kernels.py

Only one function of the legacy file uses the paged PREFILL wrapper:
``test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache`` runs the
tensor-core decode wrapper in CUDA-graph mode on a spec-decode verify batch
(every request carries ``q_len_per_req`` tokens) and compares it with the fa2
``BatchPrefillWithPagedKVCacheWrapper`` eagerly, then re-plans with different
KV lengths between replays.  The unified row below carries the legacy name,
builds the same fixture (bf16 combined pool as ``K = kv[:, 0]`` / ``V = kv[:,
1]`` views, page 16, D128, capture KV 129 per request, the two legacy replay
KV-length sets down to ``kv_len == q_len_per_req``), runs it through the
unified graph lifecycle (``GraphCapacity`` -> plan -> warm-up -> capture ->
``update()`` -> replay, twice more) and compares each replay with the same
legacy prefill-wrapper reference at the legacy rtol/atol 1e-2 and with the
fp32 oracle.  The parametrize axes keep the legacy names and values; the row
id is the legacy node id plus ``-<backend>``.

CI: legacy file in A10G fixed shard part1 (full) and H100 1/5 sampling.  The
unified file is not collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- the legacy subject is the decode wrapper's tensor-core path with its
  ``plan(q_len_per_req=)`` contract; the unified API has no q_len_per_req
  (query lengths come from ``qo_indptr``) and dispatches the multi-token
  batch as prefill, so the row is ``partial``: same batches, same reference,
  same lifecycle shape, different entry point.
- every other function of the legacy file drives the decode-only API
  (``BatchDecodeWithPagedKVCacheWrapper`` / ``CUDAGraphBatchDecode...`` /
  ``single_decode_with_kv_cache``) and is out of scope for this conversion.
"""

import pytest
import torch

from .legacy_unified_helpers import (
    DEVICE,
    LegacyBatch,
    argnames,
    assert_oracle,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    graph_capacity_for,
    legacy_paged_wrapper,
    param_rows,
    plan_batch,
    seed_of,
)
from flashinfer.prefill import PagedAttention

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_batch_decode_kernels.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_with_fast_plan",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_tuple_paged_kv_cache",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_cuda_graph_batch_decode_with_paged_kv_cache",
        [],
        "out-of-scope",
        "CUDAGraphBatchDecodeWithPagedKVCacheWrapper: decode-only API.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_nvfp4",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.  "
        "(NVFP4 KV is also undeclared on every unified backend.)",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_nvfp4_large_head",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_rejects_unequal_kv_strides_nvfp4_contract",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_single_decode_torch_compile_cuda_graph",
        [],
        "out-of-scope",
        "single_decode_with_kv_cache under torch.compile, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache",
        ["test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache"],
        "partial",
        "same 4 legacy axes (B4/8, q_len_per_req 2/3, Hkv2/8, group 4/8), bf16 combined "
        "pool, page16 D128, capture KV 129 and the two legacy replay sets (down to "
        "kv_len == q_len_per_req); the unified graph lifecycle (capacity -> plan -> "
        "capture -> update -> replay, twice re-planned) vs the legacy fa2 "
        "prefill-wrapper reference at 1e-2 (the legacy reference) + oracle; the legacy "
        "subject was the tensor-core decode wrapper's plan(q_len_per_req) contract, "
        "which the unified API expresses through qo_indptr.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_tensor_core_decode_rejects_mismatched_q_len",
        [],
        "out-of-scope",
        "decode wrapper plan(q_len_per_req) contract, no unified counterpart (query "
        "lengths come from qo_indptr).",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_paged_decode_extreme_negative_logits",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
]

MULTI_TOKEN_AXES = dict(
    batch_size=[4, 8],
    q_len_per_req=[2, 3],
    num_kv_heads=[2, 8],
    gqa_group_size=[4, 8],
)


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


def _multi_token_batch(q, kv_data, kv_lens, *, q_len_per_req, page_size):
    """The legacy ``build_kv_layout``: pages per request from ``kv_lens``,
    page ids ``arange`` (a prefix of the pool), last page ``(kv - 1) % P + 1``."""
    batch_size = len(kv_lens)
    pages_per = [(length + page_size - 1) // page_size for length in kv_lens]
    indptr = torch.tensor(
        [0, *torch.tensor(pages_per).cumsum(0).tolist()], dtype=torch.int32
    )
    return LegacyBatch(
        q=q,
        k=kv_data[:, 0],
        v=kv_data[:, 1],
        q_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * q_len_per_req,
        kv_indptr_cpu=indptr,
        kv_indices_cpu=torch.arange(int(indptr[-1]), dtype=torch.int32),
        last_page_len_cpu=torch.tensor(
            [(length - 1) % page_size + 1 for length in kv_lens], dtype=torch.int32
        ),
        page_size=page_size,
        kv_layout="NHD",
    )


@pytest.mark.parametrize(
    argnames(MULTI_TOKEN_AXES, "backend"), param_rows(MULTI_TOKEN_AXES, lambda p: True)
)
def test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache(
    batch_size, q_len_per_req, num_kv_heads, gqa_group_size, backend
):
    """Spec-decode verify shape: every request carries q_len_per_req tokens.
    The unified graph lifecycle must produce prefill-wrapper-identical
    results (legacy 1e-2) at capture and stay correct when ``update()``
    re-plans it with different KV lengths between replays."""
    page_size, head_dim, dtype = 16, 128, torch.bfloat16
    num_qo_heads = num_kv_heads * gqa_group_size
    capture_kv_lens = [129] * batch_size
    # tightest legal kv under the plan contract kv_len >= q_len_per_req
    replay_kv_lens_sets = [
        [[33, 17, 65, 129][i % 4] for i in range(batch_size)],
        [[q_len_per_req, 128, 64, 100][i % 4] for i in range(batch_size)],
    ]
    torch.manual_seed(
        seed_of("multi-token", batch_size, q_len_per_req, num_kv_heads, gqa_group_size)
    )
    dev = torch.device(DEVICE)
    total_pages = batch_size * ((129 + page_size - 1) // page_size)
    kv_data = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev
    )
    q = torch.randn(
        batch_size * q_len_per_req, num_qo_heads, head_dim, dtype=dtype, device=dev
    )
    lb = _multi_token_batch(
        q, kv_data, capture_kv_lens, q_len_per_req=q_len_per_req, page_size=page_size
    )
    width = int(lb.kv_indptr_cpu.diff().max())
    md = lb.metadata(table_width=width)

    def reference(batch):
        wrapper = legacy_paged_wrapper(
            batch, backend="fa2", causal=True, workspace_mb=128
        )
        return wrapper.run(batch.q, kv_data)  # the legacy reference (output only)

    attn = PagedAttention(dev, graph_capacity=graph_capacity_for(lb, md))
    plan_batch(lb, md, backend, attn=attn, causal=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out, lse = attn.run(q, (lb.k, lb.v))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(q, (lb.k, lb.v), out=out, lse=lse)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, reference(lb), rtol=1e-2, atol=1e-2)  # legacy
    assert_oracle(lb, out, lse, causal=True)

    for kv_lens in replay_kv_lens_sets:
        new = _multi_token_batch(
            q, kv_data, kv_lens, q_len_per_req=q_len_per_req, page_size=page_size
        )
        attn.update(new.metadata(table_width=width))
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, reference(new), rtol=1e-2, atol=1e-2)  # legacy
        assert_oracle(new, out, lse, causal=True)
