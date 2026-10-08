"""Legacy -> unified: tests/attention/test_workspace_size.py

The legacy prefill tests size the fa2 ``BatchPrefillWithPagedKVCacheWrapper``
float / int workspaces with ``workspace_size()``, then plan on buffers of
exactly those sizes; the unaligned test expects the 16-byte-alignment
rejection.  Here the same geometry (batch 3, q 64, kv 1024, page 16, H16:4,
D128, fp16, NHD, non-causal) goes through ``PagedAttention`` pinned to fa2:
the unified sizing contract is ``PagedAttention.workspace_requirements()``
(one byte bound for the single caller scratch buffer; the int workspace is
instance-owned), and a buffer of exactly that size plans AND runs against the
fp32 oracle (the legacy only planned).  ``fixed_split_size`` is not a unified
plan() argument; the alignment rejection surfaces at plan(), not at
construction.

The decode-wrapper tests are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import GraphCapacity, PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_workspace_size.py"
LEGACY_MAP = [
    (
        "tests/attention/test_workspace_size.py::test_batch_decode_workspace_size_plans_with_exact_buffers",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper.workspace_size: decode API",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers",
        ["test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers"],
        "partial",
        "fixed_split_size is not a plan() argument (TypeError asserted); a buffer of "
        "exactly workspace_requirements() bytes plans and runs on fa2 against the fp32 "
        "oracle; no separate int workspace",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers",
        ["test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers"],
        "partial",
        "graph mode: a buffer of exactly workspace_requirements(use_cuda_graph=True) "
        "bytes plans, captures and replays on fa2 against the fp32 oracle; no separate "
        "int workspace",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_decode_workspace_size_rejects_unaligned_workspace_buffer",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper.reset_workspace_buffer: decode API",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer",
        ["test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer"],
        "partial",
        "an unaligned caller workspace is rejected at plan() on fa2 with the legacy "
        "'float_workspace_buffer must be 16-byte aligned' (not at construction); the int "
        "workspace is instance-owned",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="workspace sizing tests require CUDA"
)

# the legacy _run_batch_prefill_workspace_size_plan geometry
BATCH_SIZE, QO_LEN, KV_LEN, PAGE_SIZE = 3, 64, 1024, 16
NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM = 16, 4, 128
PLAN_KWARGS = dict(
    num_qo_heads=NUM_QO_HEADS,
    num_kv_heads=NUM_KV_HEADS,
    head_dim_qk=HEAD_DIM,
    q_dtype=torch.float16,
    kv_layout="NHD",
    causal=False,
)


def _metadata(device):
    """The legacy paged-KV inputs (pages ``arange``, full pages) as a dense
    block table."""
    pages_per_seq = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    qo_indptr_cpu = torch.arange(BATCH_SIZE + 1, dtype=torch.int32) * QO_LEN
    kv_lens_cpu = torch.full((BATCH_SIZE,), KV_LEN, dtype=torch.int32)
    block_tables = torch.arange(
        BATCH_SIZE * pages_per_seq, dtype=torch.int32, device=device
    ).reshape(BATCH_SIZE, pages_per_seq)
    return PagedAttentionMetadata.dense(
        qo_indptr_cpu.to(device),
        kv_lens_cpu.to(device),
        block_tables,
        page_size=PAGE_SIZE,
        max_q_len=QO_LEN,
        max_kv_len=KV_LEN,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _qkv(device):
    """q and NHD K / V pools for the run (the legacy test only planned)."""
    torch.manual_seed(0)
    num_pages = BATCH_SIZE * ((KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE)
    q = torch.randn(
        BATCH_SIZE * QO_LEN, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    kv_shape = (num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)
    k = torch.randn(kv_shape, dtype=torch.float16, device=device)
    v = torch.randn(kv_shape, dtype=torch.float16, device=device)
    return q, k, v


def _assert_oracle(md, q, k, v, out, lse):
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k,
        v,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        md.block_tables,
        PAGE_SIZE,
        False,
        kv_layout="NHD",
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers():
    device = torch.device("cuda:0")
    md = _metadata(device)
    with pytest.raises(TypeError, match="fixed_split_size"):
        PagedAttention(device).plan(
            md, **PLAN_KWARGS, fixed_split_size=16, disable_split_kv=False
        )

    nbytes = PagedAttention.workspace_requirements(
        GraphCapacity.from_metadata(md),
        device=device,
        **PLAN_KWARGS,
        use_cuda_graph=False,
        backend="fa2",
    )
    assert nbytes > 0
    attn = PagedAttention(
        device, workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=device)
    )
    attn.plan(md, **PLAN_KWARGS, lse_mode="base2", backend="fa2")
    assert attn.backend == "fa2"

    q, k, v = _qkv(device)
    out, lse = attn.run(q, (k, v))
    _assert_oracle(md, q, k, v, out, lse)


def test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers():
    device = torch.device("cuda:0")
    md = _metadata(device)
    capacity = GraphCapacity.from_metadata(md)
    nbytes = PagedAttention.workspace_requirements(
        capacity, device=device, **PLAN_KWARGS, use_cuda_graph=True, backend="fa2"
    )
    assert nbytes > 0
    attn = PagedAttention(
        device,
        graph_capacity=capacity,
        workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=device),
    )
    attn.plan(md, **PLAN_KWARGS, lse_mode="base2", backend="fa2")
    assert attn.backend == "fa2"

    q, k, v = _qkv(device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        attn.run(q, (k, v))
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = attn.run(q, (k, v))
    graph.replay()
    torch.cuda.synchronize()
    _assert_oracle(md, q, k, v, out, lse)


def test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer():
    device = torch.device("cuda:0")
    workspace = torch.empty(32 * 1024 * 1024 + 1, dtype=torch.uint8, device=device)[1:]
    assert workspace.data_ptr() % 16 != 0
    # the constructor takes the buffer; the fa2 plan rejects it
    attn = PagedAttention(device, workspace_buffer=workspace)
    with pytest.raises(
        ValueError, match="float_workspace_buffer must be 16-byte aligned"
    ):
        attn.plan(_metadata(device), **PLAN_KWARGS, backend="fa2")
