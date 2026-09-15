"""Workspace ownership of PagedAttention (experimental).

Problem under test: every ``PagedAttention`` instance owns its workspace and
there is no way to hand it a shared one.

- ``_controller._shared_workspace()`` allocates a private 128 MB scratch
  buffer per instance, unconditionally, on the first ``plan()``
  (``make_backend(..., self._shared_workspace(), ...)``).
- The trtllm-gen backend additionally allocates its own zero-initialized
  128 MB buffer, so a trtllm-gen instance holds 256 MB of which the shared
  half is never touched.
- The class docstring prescribes "one PagedAttention(use_cuda_graph=True)
  instance per graph bucket".  A serving engine has tens of graph buckets, so
  the workspace cost scales as ``num_buckets x 128 MB`` (or ``x 256 MB`` on
  trtllm-gen) — where the legacy wrappers share ONE caller-owned workspace
  across every instance.

Contract these tests assert (the controller implements it as one per-device
shared scratch buffer plus an optional caller-supplied one):

1. Adding one more graph-bucket instance must not cost another workspace.
2. trtllm-gen runs on the same shared scratch (its 128 MB buffer is ordinary
   softmax-stats/scratch); the zero-initialized part it really needs is a
   KB-sized counter buffer the backend owns, so run() allocates nothing.
3. A caller-supplied ``workspace_buffer`` is what every backend runs on, and a
   buffer on the wrong device / of the wrong shape is rejected loudly.
"""

import gc

import pytest
import torch

from flashinfer.prefill import PagedAttention

from .test_paged_attention_prototype import (
    BACKENDS,
    _resolve_or_skip,
    make_metadata,
    make_problem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MB = 1024 * 1024
# What _controller._shared_workspace() and the trtllm-gen backend each reserve.
WORKSPACE_MB = 128
# Generous ceiling for the legitimate per-bucket state of one more instance:
# reserved int32 metadata (KBs) plus the generated-FA wrapper's 8 MB int
# workspace.  Half a workspace is far above that and far below one workspace,
# so the assertion is insensitive to allocator rounding and only trips on a
# duplicated 128 MB buffer.
PER_BUCKET_LIMIT_MB = WORKSPACE_MB // 2
# Engine-shaped bucket count kept small so the test itself stays cheap; the
# waste is linear in it, so 4 is enough to show the slope.
NUM_BUCKETS = 4

_SHAPE = dict(
    batch_size=4,
    max_q=32,
    max_kv=256,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)


def _plan(attn, p, backend):
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        q_dtype=p["dtype"],
        causal=True,
        lse_mode="base2",
        backend=backend,
    )


def _allocated_mb(dev) -> float:
    torch.cuda.synchronize(dev)
    gc.collect()
    return torch.cuda.memory_allocated(dev) / MB


def _large_buffers(attn):
    """White-box: (label, tensor) for every >= 64 MB buffer an instance holds.

    Only used to make the failure message name the culprit; the assertions
    themselves are black-box (device memory growth)."""
    found = []
    try:
        impl = attn._impl
        if impl._workspace is not None:
            found.append(("controller._workspace", impl._workspace))
        for key, be in impl._backends.items():
            ws = getattr(be, "_workspace", None)
            if ws is not None:
                found.append((f"backend{key}._workspace", ws))
            wrapper = getattr(be, "_wrapper", None)
            fws = getattr(wrapper, "_float_workspace_buffer", None)
            if fws is not None:
                found.append((f"backend{key}._wrapper._float_workspace_buffer", fws))
    except AttributeError:  # internals moved; the black-box assert still stands
        pass
    return [(lbl, t) for lbl, t in found if t.numel() * t.element_size() >= 64 * MB]


def _describe(instances) -> str:
    ptrs = {}
    for i, attn in enumerate(instances):
        for lbl, t in _large_buffers(attn):
            ptrs.setdefault(t.data_ptr(), []).append(
                f"bucket{i}.{lbl} ({t.numel() * t.element_size() // MB} MB)"
            )
    lines = [f"{len(ptrs)} distinct >=64 MB buffers across {len(instances)} instances:"]
    for holders in ptrs.values():
        lines.append("  " + ", ".join(holders))
    return "\n".join(lines)


@pytest.mark.parametrize("backend", BACKENDS)
def test_workspace_not_duplicated_per_graph_bucket(backend):
    """One instance per graph bucket must not mean one 128 MB workspace per bucket."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, backend)
    dev = torch.device(p["device"])

    # Warm-up instance: pays every one-time cost (JIT module load, cuDNN
    # handle, first derivation) so the measured slope is per-bucket state only.
    warm = PagedAttention(dev, use_cuda_graph=True)
    _plan(warm, p, backend)

    before = _allocated_mb(dev)
    buckets = []
    for _ in range(NUM_BUCKETS):
        attn = PagedAttention(dev, use_cuda_graph=True)
        _plan(attn, p, backend)
        buckets.append(attn)
    per_bucket = (_allocated_mb(dev) - before) / NUM_BUCKETS

    assert per_bucket < PER_BUCKET_LIMIT_MB, (
        f"[{backend}] each additional PagedAttention(use_cuda_graph=True) "
        f"instance costs {per_bucket:.1f} MB of device memory (limit "
        f"{PER_BUCKET_LIMIT_MB} MB): every graph bucket allocates its own "
        f"{WORKSPACE_MB} MB workspace instead of sharing one.\n"
        + _describe([warm, *buckets])
    )


def test_trtllm_gen_runs_on_the_shared_workspace():
    """trtllm-gen's 128 MB buffer is ordinary scratch: it must be the shared
    workspace, not a second private allocation next to it.  The only thing the
    kernel needs zero-initialized is its KB-sized multi-CTA KV counter buffer."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, "trtllm-gen")
    dev = torch.device(p["device"])

    before = _allocated_mb(dev)
    attn = PagedAttention(dev)
    _plan(attn, p, "trtllm-gen")
    growth = _allocated_mb(dev) - before

    # at most the one shared workspace (first user on this device pays it);
    # anything past 1.5x of it means a second workspace was allocated alongside
    limit = WORKSPACE_MB * 3 // 2
    assert growth < limit, (
        f"[trtllm-gen] planning one instance allocated {growth:.1f} MB "
        f"(limit {limit} MB): a second {WORKSPACE_MB} MB workspace was allocated "
        "next to the shared one.\n" + _describe([attn])
    )
    held = {t.data_ptr() for _, t in _large_buffers(attn)}
    assert len(held) <= 1, "[trtllm-gen] holds two distinct >=64 MB buffers\n" + (
        _describe([attn])
    )


def test_trtllm_gen_run_allocates_nothing():
    """With out=/lse= supplied, run() must not allocate: the zero-initialized
    multi-CTA KV counter buffer is owned by the backend and passed explicitly,
    instead of a fresh ``torch.zeros`` per call (which the one-shot function
    does when no counter buffer is given)."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, "trtllm-gen")
    dev = torch.device(p["device"])
    attn = PagedAttention(dev)
    _plan(attn, p, "trtllm-gen")
    out = torch.empty(
        p["q"].shape[0],
        p["num_qo_heads"],
        p["head_dim_vo"],
        dtype=p["dtype"],
        device=dev,
    )
    lse = torch.empty(
        p["q"].shape[0], p["num_qo_heads"], dtype=torch.float32, device=dev
    )
    attn.run(p["q"], (p["k_cache"], p["v_cache"]), out=out, lse=lse)  # warm
    torch.cuda.synchronize(dev)

    allocs_before = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
    attn.run(p["q"], (p["k_cache"], p["v_cache"]), out=out, lse=lse)
    torch.cuda.synchronize(dev)
    allocs = torch.cuda.memory_stats(dev)["allocation.all.allocated"] - allocs_before
    assert allocs == 0, (
        f"[trtllm-gen] run() with out=/lse= performed {allocs} device allocation(s); "
        "expected none (per-call counter buffer?)"
    )


@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen"])
def test_caller_supplied_workspace_is_used(backend):
    """The engine's existing workspace (shared with legacy wrappers) is honored:
    planning allocates no workspace of its own, and the scratch backend runs
    on the supplied buffer."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, backend)
    dev = torch.device(p["device"])
    ws = torch.empty(WORKSPACE_MB * MB, dtype=torch.uint8, device=dev)

    warm = PagedAttention(dev, workspace_buffer=ws)  # one-time costs (JIT etc.)
    _plan(warm, p, backend)

    before = _allocated_mb(dev)
    attn = PagedAttention(dev, workspace_buffer=ws)
    _plan(attn, p, backend)
    growth = _allocated_mb(dev) - before
    assert growth < PER_BUCKET_LIMIT_MB, (
        f"[{backend}] planning with a caller-supplied workspace still allocated "
        f"{growth:.1f} MB\n" + _describe([attn])
    )
    held = {t.data_ptr() for _, t in _large_buffers(attn)}
    assert held == {ws.data_ptr()}, (
        f"[{backend}] backend does not run on the supplied workspace\n"
        + _describe([attn])
    )
    # and the plan is actually runnable on it
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize(dev)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()


def test_workspace_buffer_validation():
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="1-D"):
        PagedAttention(
            dev, workspace_buffer=torch.empty(4, 4, dtype=torch.uint8, device=dev)
        )
    with pytest.raises(ValueError, match="uint8"):
        PagedAttention(
            dev, workspace_buffer=torch.empty(16, dtype=torch.float32, device=dev)
        )
    with pytest.raises(ValueError, match="lives on"):
        PagedAttention(dev, workspace_buffer=torch.empty(16, dtype=torch.uint8))
    with pytest.raises(ValueError, match="torch.Tensor"):
        PagedAttention(dev, workspace_buffer=WORKSPACE_MB * MB)
