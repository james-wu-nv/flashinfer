"""Legacy -> unified: tests/attention/test_workspace_size.py

Every test function below carries the name of the legacy test it converts and
runs the legacy geometry (B3 q64 kv1024 page16 H16:4 D128, fp16 NHD pools)
through flashinfer.prefill.PagedAttention.  The legacy tests size the fa2
wrapper's float / int workspaces with ``workspace_size()`` and plan on
buffers of exactly those sizes; the unified sizing contract is
``PagedAttention.workspace_requirements()`` (one conservative byte bound for
one scratch buffer per ``GraphCapacity``, eager or graph mode), so each row
sizes a buffer with it, plans AND runs the legacy geometry against the fp32
oracle (the legacy tests only planned).  The rows carry the legacy function
names; the added ``backend`` axis records the sizing behaviour per backend.

CI: legacy file only in the H100 1/5 sampling lane.  The unified file is not
collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- ``fixed_split_size`` / ``disable_split_kv`` are not plan() arguments
  (TypeError, asserted); the unified bound does not expose the legacy exact
  float / int split (the int workspace is instance-owned).  ``partial``.
- graph mode: ``workspace_requirements(use_cuda_graph=True)`` is exact for fa2
  (a buffer of that size captures and replays; 16 bytes less is rejected at
  plan() naming the bytes); other backends reject-or-run.  ``partial``.
- alignment: the unified constructor validates dtype / device / shape only;
  the 16-byte-alignment rejection surfaces at plan() from the fa2 wrapper
  ("float_workspace_buffer must be 16-byte aligned"), not at construction as
  in legacy; cuDNN, trtllm-gen and cake accept an unaligned buffer and match
  the oracle (recorded as a skip with the reason).  ``partial``.
- decode-wrapper functions of the legacy file are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    MB,
    assert_oracle,
    backend_rows,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    graph_capacity_for,
    legacy_uniform_batch,
    plan_batch,
    plan_signature_params,
    resolve_batch_or_skip,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_workspace_size.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_workspace_size.py::test_batch_decode_workspace_size_plans_with_exact_buffers",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper.workspace_size: decode-only API.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers",
        ["test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers"],
        "partial",
        "fixed_split_size is not expressible (TypeError asserted); the unified sizing "
        "contract is workspace_requirements() (one conservative byte bound, not an exact "
        "float/int split) and a buffer of exactly that size plans and runs the legacy "
        "geometry (B3 q64 kv1024 page16 H16:4 D128) against the oracle, per backend.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers",
        ["test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers"],
        "partial",
        "graph mode: workspace_requirements(use_cuda_graph=True) for the legacy geometry "
        "captures and replays on a buffer of exactly that size (against the oracle); 16 "
        "bytes less is rejected at plan naming the bytes on fa2 (exact bound), other "
        "backends reject-or-run; the legacy int-workspace half is instance-owned in the "
        "unified API.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_decode_workspace_size_rejects_unaligned_workspace_buffer",
        [],
        "out-of-scope",
        "BatchDecodeWithPagedKVCacheWrapper.reset_workspace_buffer: decode-only API.",
    ),
    (
        "tests/attention/test_workspace_size.py::test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer",
        ["test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer"],
        "partial",
        "an unaligned caller workspace is rejected at plan() on fa2 with the legacy "
        "wrapper's 'float_workspace_buffer must be 16-byte aligned' (not at "
        "construction: the unified constructor validates dtype/device/shape only); cudnn "
        "/ trtllm-gen / cake accept it and match the oracle (skip with reason).",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


def _workspace_fixture(seed):
    """The legacy ``_run_batch_prefill_workspace_size_plan`` geometry with
    fp16 NHD pools (the legacy test plans only; here the buffer must also run)."""
    return legacy_uniform_batch(
        batch_size=3,
        kv_len=1024,
        qo_len=64,
        page_size=16,
        num_qo_heads=16,
        num_kv_heads=4,
        head_dim=128,
        combined=False,
        seed=seed,
    )


def _requirements(lb, md, *, use_cuda_graph, backend):
    return PagedAttention.workspace_requirements(
        graph_capacity_for(lb, md),
        device=torch.device(DEVICE),
        **lb.plan_kwargs(),
        causal=True,
        need_lse=True,
        use_cuda_graph=use_cuda_graph,
        backend=backend,
    )


@pytest.mark.parametrize("backend", backend_rows())
def test_batch_prefill_workspace_size_plans_fixed_split_with_exact_buffers(backend):
    """fixed_split_size is not a unified argument; the eager bound of
    ``workspace_requirements`` for the legacy geometry is positive and a
    buffer of exactly that size plans AND runs against the oracle."""
    params = plan_signature_params()
    assert "fixed_split_size" not in params and "disable_split_kv" not in params
    lb = _workspace_fixture(seed_of("ws-eager"))
    md = lb.metadata()
    with pytest.raises(TypeError, match="fixed_split_size"):
        PagedAttention(torch.device(DEVICE)).plan(
            md, **lb.plan_kwargs(), fixed_split_size=16, disable_split_kv=False
        )
    resolve_batch_or_skip(lb, md, backend)
    nbytes = _requirements(lb, md, use_cuda_graph=False, backend=backend)
    assert nbytes > 0  # legacy: float_workspace_size > 0 with a fixed split
    dev = torch.device(DEVICE)
    attn = PagedAttention(
        dev, workspace_buffer=torch.empty(nbytes, dtype=torch.uint8, device=dev)
    )
    plan_batch(lb, md, backend, attn=attn)
    out, lse = attn.run(lb.q, (lb.k, lb.v))
    assert_oracle(lb, out, lse, causal=True)


@pytest.mark.parametrize("backend", backend_rows())
def test_batch_prefill_workspace_size_plans_cuda_graph_with_exact_buffers(backend):
    """The graph-mode bound for the legacy geometry captures and replays on a
    buffer of exactly that size (against the oracle); 16 bytes less is
    rejected at plan naming the bytes (fa2: exact bound) or planned and run
    correctly by a backend whose bound is conservative."""
    lb = _workspace_fixture(seed_of("ws-graph"))
    width = int(lb.kv_indptr_cpu.diff().max())
    md = lb.metadata(table_width=width)
    resolve_batch_or_skip(lb, md, backend)
    nbytes = _requirements(lb, md, use_cuda_graph=True, backend=backend)
    assert nbytes > 0
    dev = torch.device(DEVICE)
    ws = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    attn = PagedAttention(
        dev, graph_capacity=graph_capacity_for(lb, md), workspace_buffer=ws
    )
    plan_batch(lb, md, backend, attn=attn)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    attn.update(md)
    g.replay()
    torch.cuda.synchronize()
    assert_oracle(lb, out, lse, causal=True)
    short = PagedAttention(
        dev,
        graph_capacity=graph_capacity_for(lb, md),
        workspace_buffer=ws[: nbytes - 16],
    )
    try:
        short.plan(
            md, **lb.plan_kwargs(), causal=True, lse_mode="base2", backend=backend
        )
    except ValueError as e:
        assert f"needs {nbytes} bytes" in str(e), e
        return
    if backend in ("fa2", "auto") and short.backend == "fa2":
        raise AssertionError("fa2's graph-mode bound is exact: 16 bytes less must fail")
    out2, lse2 = short.run(lb.q, (lb.k, lb.v))
    torch.cuda.synchronize()
    assert_oracle(lb, out2, lse2, causal=True)


@pytest.mark.parametrize("backend", backend_rows())
def test_batch_prefill_workspace_size_rejects_unaligned_workspace_buffer(backend):
    """A caller buffer whose data pointer is not 16-byte aligned.  The
    unified constructor accepts any contiguous 1-D byte tensor; the rejection
    surfaces at plan() from the fa2 wrapper ('float_workspace_buffer must be
    16-byte aligned').  Other backends: reject with a ValueError or run
    correctly, never misread."""
    lb = _workspace_fixture(seed_of("ws-unaligned"))
    md = lb.metadata()
    resolve_batch_or_skip(lb, md, backend)
    dev = torch.device(DEVICE)
    ws = torch.empty(32 * MB + 1, dtype=torch.uint8, device=dev)[1:]
    assert ws.data_ptr() % 16 != 0
    attn = PagedAttention(dev, workspace_buffer=ws)
    try:
        plan_batch(lb, md, backend, attn=attn)
        out, lse = attn.run(lb.q, (lb.k, lb.v))
    except ValueError as e:
        assert "aligned" in str(e), e
        if backend in ("fa2", "auto") and attn.backend in (None, "fa2"):
            assert "float_workspace_buffer must be 16-byte aligned" in str(e)
        return
    torch.cuda.synchronize()
    assert_oracle(lb, out, lse, causal=True)
    pytest.skip(
        f"[{attn.backend}] accepts the unaligned workspace and matches the oracle; "
        "the legacy alignment rejection is fa2-specific"
    )
