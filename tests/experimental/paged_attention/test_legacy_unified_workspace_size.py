"""Legacy -> unified: tests/attention/test_workspace_size.py

The legacy prefill tests size the fa2 ``BatchPrefillWithPagedKVCacheWrapper``
float / int workspaces with ``workspace_size()`` and plan on buffers of exactly
those sizes (no kernel runs); the unaligned test expects the 16-byte-alignment
rejection.  The unified sizing contract is
``PagedAttention.workspace_requirements()``: one byte bound for the single
caller scratch buffer (the int workspace is instance-owned).  Same geometry
(batch 3, q 64, kv 1024, page 16, H16:4, D128, fp16, NHD, non-causal).

- fixed split: ``plan(fixed_split_size=16)`` on fa2, the one backend that
  declares the control, eager as in the legacy test: a buffer of exactly
  ``workspace_requirements(fixed_split_size=16, use_cuda_graph=False)`` bytes
  (non-zero: 16 pages split the 64-page KV range) plans and runs.
- CUDA graph: once per backend (fa2, fa3, cudnn, trtllm-gen; skipped with the
  resolver's reason where it cannot run), a buffer of exactly the reported size
  plans, captures and replays.
- CUDA graph without a split: ``plan(disable_split_kv=True)`` on fa2;
  ``workspace_requirements()`` reports 0 bytes and a zero-byte buffer plans,
  captures and replays.
- the legacy only planned, so each of these checks the output and LSE against
  the fp32 oracle.
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
        "partial",
        "fa2, eager: a buffer of exactly workspace_requirements(fixed_split_size=16, "
        "use_cuda_graph=False) bytes plans and runs with plan(fixed_split_size=16), "
        "checked against the fp32 oracle; no separate int workspace",
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
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_no_split_cuda_graph_with_exact_buffers",
        [
            "test_batch_prefill_workspace_size_plans_no_split_cuda_graph_with_exact_buffers"
        ],
        "partial",
        "fa2 graph mode with plan(disable_split_kv=True): workspace_requirements() "
        "is 0 and a zero-byte buffer plans, captures and replays, checked against "
        "the fp32 oracle; no separate int workspace",
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


def _skip_unless_runnable(backend, **control):
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
            **control,
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


def _qkv(device):
    torch.manual_seed(0)
    num_pages = BATCH_SIZE * ((KV_LEN + PAGE_SIZE - 1) // PAGE_SIZE)
    q = torch.randn(
        BATCH_SIZE * QO_LEN, NUM_QO_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    k = torch.randn(
        num_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device=device
    )
    return q, k, torch.randn_like(k)


def _capture_and_replay(attn, q, k, v):
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
    return out, lse


def _assert_matches_oracle(metadata, q, k, v, out, lse):
    """fp32 oracle (the legacy tests only plan)."""
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


def test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers():
    control = dict(fixed_split_size=16)  # the legacy split, in pages
    _skip_unless_runnable("fa2", **control)
    device = torch.device("cuda:0")
    metadata = _metadata()
    nbytes = PagedAttention.workspace_requirements(
        GraphCapacity.from_metadata(metadata),
        device=device,
        **PLAN_KWARGS,
        use_cuda_graph=False,
        backend="fa2",
        **control,
    )
    assert nbytes > 0  # the legacy assertion: a fixed split needs float scratch
    attn = PagedAttention(torch.empty(nbytes, dtype=torch.uint8, device=device))
    attn.plan(metadata, **PLAN_KWARGS, lse_mode="base2", backend="fa2", **control)
    assert attn.backend == "fa2"
    q, k, v = _qkv(device)
    out, lse = attn.run(q, (k, v))
    _assert_matches_oracle(metadata, q, k, v, out, lse)


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
        torch.empty(nbytes, dtype=torch.uint8, device=device), graph_capacity=capacity
    )
    attn.plan(metadata, **PLAN_KWARGS, lse_mode="base2", backend=backend)
    assert attn.backend == backend
    q, k, v = _qkv(device)
    out, lse = _capture_and_replay(attn, q, k, v)
    _assert_matches_oracle(metadata, q, k, v, out, lse)


def test_batch_prefill_workspace_size_plans_no_split_cuda_graph_with_exact_buffers():
    control = dict(disable_split_kv=True)
    _skip_unless_runnable("fa2", **control)
    device = torch.device("cuda:0")
    metadata = _metadata()
    capacity = GraphCapacity.from_metadata(metadata)
    nbytes = PagedAttention.workspace_requirements(
        capacity,
        device=device,
        **PLAN_KWARGS,
        use_cuda_graph=True,
        backend="fa2",
        **control,
    )
    assert nbytes == 0  # an unsplit plan takes no float scratch
    attn = PagedAttention(
        torch.empty(nbytes, dtype=torch.uint8, device=device), graph_capacity=capacity
    )
    attn.plan(metadata, **PLAN_KWARGS, lse_mode="base2", backend="fa2", **control)
    assert attn.backend == "fa2"
    q, k, v = _qkv(device)
    out, lse = _capture_and_replay(attn, q, k, v)
    _assert_matches_oracle(metadata, q, k, v, out, lse)


@pytest.mark.parametrize("backend", BACKENDS)
def test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer(backend):
    _skip_unless_runnable(backend)
    device = torch.device("cuda:0")
    workspace = torch.empty(32 * 1024 * 1024 + 1, dtype=torch.uint8, device=device)[1:]
    assert workspace.data_ptr() % 16 != 0
    attn = PagedAttention(workspace)
    try:
        attn.plan(_metadata(), **PLAN_KWARGS, backend=backend)
    except ValueError as e:
        assert "16-byte aligned" in str(e), str(e)
        return
    pytest.xfail(
        f"API gap: {backend} plans on a workspace buffer that is not 16-byte "
        "aligned (the alignment check lives in the fa2 / fa3 wrappers only)"
    )
