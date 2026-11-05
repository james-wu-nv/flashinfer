"""Legacy -> unified: tests/attention/test_workspace_size.py

The legacy prefill tests size the fa2 ``BatchPrefillWithPagedKVCacheWrapper``
float / int workspaces with ``workspace_size()`` and plan on buffers of exactly
those sizes (no kernel runs); the unaligned test expects the 16-byte-alignment
rejection.  The unified sizing contract is
``PagedAttention.workspace_requirements()``: one byte bound for the single
caller scratch buffer (the int workspace is instance-owned).  Same geometry
(batch 3, q 64, kv 1024, page 16, H16:4, D128, fp16, NHD, non-causal).

- fixed split: ``fixed_split_size`` is not a ``plan()`` /
  ``workspace_requirements()`` argument, an expected failure naming the API.
- CUDA graph: once per backend (fa2, fa3, cudnn, trtllm-gen; skipped with the
  resolver's reason where it cannot run), a buffer of exactly the reported size
  plans, captures and replays; the legacy only planned, so the replayed output
  and LSE are checked against the fp32 oracle.
- unaligned buffer: once per backend, plan() must reject it with the legacy
  "16-byte aligned" message; an expected failure (with the reason) where the
  backend accepts it.

The decode-wrapper tests are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import (
    GraphCapacity,
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

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
        "unsupported-by-design",
        "xfail: plan() / workspace_requirements() have no fixed_split_size (API limit)",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers",
        ["test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers"],
        "partial",
        "graph mode on fa2 / fa3 / cudnn / trtllm-gen: a buffer of exactly "
        "workspace_requirements(use_cuda_graph=True, backend=...) bytes plans, "
        "captures and replays, checked against the fp32 oracle; no separate int "
        "workspace",
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
        "an unaligned caller workspace must be rejected at plan() with the legacy "
        "'16-byte aligned' message, on fa2 / fa3 / cudnn / trtllm-gen (xfail where the "
        "backend accepts it); the int workspace is instance-owned",
    ),
]

BACKENDS = ["fa2", "fa3", "cudnn", "trtllm-gen"]
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


def _skip_unless_runnable(backend):
    try:
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=NUM_QO_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim_qk=HEAD_DIM,
            q_dtype=torch.float16,
            page_size=PAGE_SIZE,
            kv_layout="NHD",
            causal=False,
            need_lse=True,
            kv_input_form="page_indices",
            backend=backend,
        )
    except ValueError as e:
        pytest.skip(f"{backend} not runnable here: {e}")


def _metadata():
    """The legacy paged-KV inputs (pages ``arange``, full pages) as CSR."""
    pages_per_seq = (KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    qo_indptr = torch.arange(BATCH_SIZE + 1, dtype=torch.int32, device="cuda") * QO_LEN
    paged_kv_indices = torch.arange(
        BATCH_SIZE * pages_per_seq, dtype=torch.int32, device="cuda"
    )
    return PagedAttentionMetadata.csr(
        qo_indptr,
        torch.full((BATCH_SIZE,), KV_LEN, dtype=torch.int32, device="cuda"),
        paged_kv_indices,
        page_size=PAGE_SIZE,
        max_q_len=QO_LEN,
        max_kv_len=KV_LEN,
    )


def test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers():
    pytest.xfail(
        "API limit: PagedAttention.plan() / workspace_requirements() have no "
        "fixed_split_size (the legacy sizes and plans a fixed split of 16 pages)"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers(backend):
    _skip_unless_runnable(backend)
    device = torch.device("cuda:0")
    metadata = _metadata()
    capacity = GraphCapacity.from_metadata(metadata)
    nbytes = PagedAttention.workspace_requirements(
        capacity, device=device, **PLAN_KWARGS, use_cuda_graph=True, backend=backend
    )
    attn = PagedAttention(
        device,
        graph_capacity=capacity,
        workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=device),
    )
    attn.plan(metadata, **PLAN_KWARGS, lse_mode="base2", backend=backend)
    assert attn.backend == backend

    torch.manual_seed(0)
    num_pages = BATCH_SIZE * ((KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE)
    q = torch.randn(
        BATCH_SIZE * QO_LEN, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    k = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    v = torch.randn_like(k)

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

    # fp32 oracle (the legacy test only plans)
    ref_out, ref_lse = reference_paged_prefill(
        q,
        k,
        v,
        metadata.qo_indptr.cpu(),
        metadata.kv_seq_lens.cpu(),
        None,
        PAGE_SIZE,
        False,
        kv_layout="NHD",
        kv_page_indices=metadata.kv_page_indices,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer(backend):
    _skip_unless_runnable(backend)
    device = torch.device("cuda:0")
    workspace = torch.empty(32 * 1024 * 1024 + 1, dtype=torch.uint8, device=device)[1:]
    assert workspace.data_ptr() % 16 != 0
    attn = PagedAttention(device, workspace_buffer=workspace)
    try:
        attn.plan(_metadata(), **PLAN_KWARGS, backend=backend)
    except ValueError as e:
        assert "16-byte aligned" in str(e), str(e)
        return
    pytest.xfail(
        f"API gap: {backend} plans on a workspace buffer that is not 16-byte "
        "aligned (the alignment check lives in the fa2 / fa3 wrappers only)"
    )
