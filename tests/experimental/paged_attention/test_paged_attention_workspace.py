"""Workspace ownership of PagedAttention (experimental).

``PagedAttention`` runs on one caller-owned scratch workspace, the first
constructor argument, as the legacy wrappers and the MLA wrapper take their
``float_workspace_buffer``; the library allocates none.  A serving engine
holds one instance per graph bucket, so the contract is:

1. Every instance built over one buffer runs on it: one more graph-bucket
   instance costs no workspace.
2. trtllm-gen runs on the same buffer (ordinary softmax-stats/scratch); the
   zero-initialized part it really needs is a KB-sized counter buffer the
   backend owns, so run() allocates nothing.
3. The instance runs on the buffer's device; a buffer of the wrong type,
   shape or dtype is rejected loudly, and so is a missing one.

Sizing (ledger M14): the fa2 split-KV planner carves ``tmp_v`` / ``tmp_s`` out
of this buffer; before it reserved per query row (#5177) the legacy 128 MiB
buffer overflowed on one request of q = kv = 2048 with 32/8 heads in graph
mode (129 MiB; now 32.25 MiB).  Contract:

4. That shape plans and runs on a 128 MiB workspace and matches the oracle.
5. ``PagedAttention.workspace_requirements()`` is >= the bytes the fa2 planner
   actually allocates (observed through the legacy wrapper's
   ``workspace_size()``, the planner's own counting mode), and the per-batch
   need the plan-time check computes equals the planner's.
6. A backend whose need for a batch exceeds the buffer declines it at
   ``plan()``, before any kernel-side error, with the typed plan-time
   rejection: ``backend="auto"`` moves on to the next candidate and keeps the
   required bytes in the selection trace; a named backend, or a candidate set
   in which nothing fits, gets a ``ValueError`` naming the required bytes.
   The tests of the walk itself force its order and run on the host with the
   backends' real ``preflight()``; one runs the fallback's kernels.
"""

import dataclasses
import gc
import re

import pytest
import torch

from flashinfer.experimental.paged_attention import _controller, _selection
from flashinfer.experimental.paged_attention._backends import (
    CAPABILITIES,
    _BackendPlanUnsupportedError,
)
from flashinfer.experimental.paged_attention._backends._capabilities import (
    _workspace_too_small,
)
from flashinfer.experimental.paged_attention._backends.fa_backend import _FaBackend
from flashinfer.experimental.paged_attention._backends.trtllm_gen_backend import (
    _TrtllmGenBackend,
)
from flashinfer.experimental.paged_attention._controller import (
    PagedAttentionController,
)
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    GraphCapacity,
    PagedAttention,
    resolve_paged_attention,
)

from .paged_attention_reference import (
    SCRATCH_WORKSPACE_BYTES,
    reference_paged_prefill,
    scratch_workspace,
)
from .test_paged_attention_contract_cpu import _host_metadata
from .test_paged_attention_prototype import (
    BACKENDS,
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
    quantize_kv,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MB = 1024 * 1024
# The tests' workspace (scratch_workspace(): one buffer per device).
WORKSPACE_MB = SCRATCH_WORKSPACE_BYTES // MB
# Generous ceiling for the legitimate per-bucket state of one more instance:
# reserved int32 metadata (KBs) plus the generated-FA wrapper's 8 MB int
# workspace.  Half a workspace is far above that and far below one workspace,
# so the assertion is insensitive to allocator rounding and only trips on a
# duplicated workspace.
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


@needs_cuda
@pytest.mark.parametrize("backend", BACKENDS)
def test_workspace_not_duplicated_per_graph_bucket(backend):
    """One instance per graph bucket over one caller buffer must not mean one
    workspace per bucket."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, backend)
    dev = torch.device(p["device"])

    # Warm-up instance: pays every one-time cost (JIT module load, cuDNN
    # handle, first derivation) so the measured slope is per-bucket state only.
    warm = PagedAttention(scratch_workspace(dev), use_cuda_graph=True)
    _plan(warm, p, backend)

    before = _allocated_mb(dev)
    buckets = []
    for _ in range(NUM_BUCKETS):
        attn = PagedAttention(scratch_workspace(dev), use_cuda_graph=True)
        _plan(attn, p, backend)
        buckets.append(attn)
    per_bucket = (_allocated_mb(dev) - before) / NUM_BUCKETS

    assert per_bucket < PER_BUCKET_LIMIT_MB, (
        f"[{backend}] each additional PagedAttention(workspace, use_cuda_graph=True) "
        f"instance costs {per_bucket:.1f} MB of device memory (limit "
        f"{PER_BUCKET_LIMIT_MB} MB): every graph bucket allocates a workspace "
        "of its own instead of running on the caller's.\n" + _describe([warm, *buckets])
    )


@needs_cuda
def test_trtllm_gen_run_allocates_nothing():
    """With out=/lse= supplied, run() must not allocate: the zero-initialized
    multi-CTA KV counter buffer is owned by the backend and passed explicitly,
    instead of a fresh ``torch.zeros`` per call (which the one-shot function
    does when no counter buffer is given)."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, "trtllm-gen")
    dev = torch.device(p["device"])
    attn = PagedAttention(scratch_workspace(dev))
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


@needs_cuda
@pytest.mark.parametrize("backend", ["fa2", "fa3", "cudnn", "trtllm-gen"])
def test_caller_supplied_workspace_is_used(backend):
    """The engine's existing workspace (shared with legacy wrappers) is honored:
    planning allocates no workspace of its own, and the scratch backend runs
    on the supplied buffer."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, backend)
    dev = torch.device(p["device"])
    ws = torch.empty(WORKSPACE_MB * MB, dtype=torch.uint8, device=dev)

    warm = PagedAttention(ws)  # one-time costs (JIT etc.)
    _plan(warm, p, backend)

    before = _allocated_mb(dev)
    attn = PagedAttention(ws)
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


@needs_cuda
def test_workspace_buffer_validation():
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="1-D"):
        PagedAttention(torch.empty(4, 4, dtype=torch.uint8, device=dev))
    with pytest.raises(ValueError, match="uint8"):
        PagedAttention(torch.empty(16, dtype=torch.float32, device=dev))
    with pytest.raises(ValueError, match="torch.Tensor"):
        PagedAttention(WORKSPACE_MB * MB)
    with pytest.raises(ValueError, match="torch.Tensor"):
        PagedAttention(dev)  # a device is not a workspace
    with pytest.raises(TypeError, match="workspace_buffer"):
        PagedAttention()  # the library allocates none
    # the instance runs on the buffer's device
    assert PagedAttention(torch.empty(16, dtype=torch.uint8, device=dev)).device == dev
    assert PagedAttention(torch.empty(16, dtype=torch.uint8)).device.type == "cpu"


# ---------------------------------------------------------------------------
# Sizing (M14)
# ---------------------------------------------------------------------------


def _exact_problem(
    q_lens,
    kv_lens,
    *,
    num_qo_heads,
    num_kv_heads,
    head_dim=128,
    page_size=16,
    dtype=torch.bfloat16,
    kv_dtype=None,
    input_form="block_tables",
    seed=0,
    device="cuda:0",
):
    """Problem dict (as ``make_problem``) with EXACT per-request lengths."""
    g = torch.Generator().manual_seed(seed)
    q_lens = torch.tensor(list(q_lens), dtype=torch.int32)
    kv_lens = torch.tensor(list(kv_lens), dtype=torch.int32)
    b = q_lens.shape[0]
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages = (kv_lens + page_size - 1) // page_size
    width = max(int(pages.max()), 1)
    pool = int(pages.sum()) + 4
    perm = torch.randperm(pool, generator=g, dtype=torch.int32)
    table = torch.zeros(b, width, dtype=torch.int32)
    off = 0
    for i in range(b):
        n = int(pages[i])
        table[i, :n] = perm[off : off + n]
        off += n
    total = int(qo_indptr_cpu[-1])
    q = torch.randn(total, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(pool, num_kv_heads, page_size, head_dim, dtype=dtype, device=device)
    v = torch.randn_like(k)
    k_ref, v_ref, k_scale, v_scale = k, v, None, None
    if kv_dtype is not None and kv_dtype != dtype:
        k, v, k_ref, v_ref, k_scale, v_scale = quantize_kv(k, v, kv_dtype)
    return dict(
        q=q,
        k_cache=k,
        v_cache=v,
        k_ref=k_ref,
        v_ref=v_ref,
        kv_dtype=kv_dtype if kv_dtype is not None else dtype,
        k_scale=k_scale,
        v_scale=v_scale,
        qo_indptr=qo_indptr_cpu.to(device),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens=kv_lens.to(device),
        kv_seq_lens_cpu=kv_lens,
        block_tables=table.to(device),
        kv_page_indices=torch.cat([table[i, : int(pages[i])] for i in range(b)]).to(
            device
        ),
        kv_layout="HND",
        input_form=input_form,
        page_size=page_size,
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        dtype=dtype,
        device=device,
    )


def _capacity(p, *, total_q_tokens=None):
    """The GraphCapacity of problem ``p``: the dense table's full width, or
    the flat page-id list's length in the flat form."""
    common = dict(
        batch_size=int(p["kv_seq_lens_cpu"].shape[0]),
        total_q_tokens=total_q_tokens or int(p["qo_indptr_cpu"][-1]),
        max_q_len=p["max_q_len"],
        page_size=p["page_size"],
    )
    if p.get("input_form") == "page_indices":
        return GraphCapacity(
            **common,
            max_kv_len=p["max_kv_len"],
            kv_input_form="page_indices",
            flat_capacity=int(p["kv_page_indices"].shape[0]),
        )
    width = int(p["block_tables"].shape[1])
    return GraphCapacity(**common, max_kv_len=width * p["page_size"], table_width=width)


def _plan_kwargs(p, *, window_left=-1, lse_mode="base2"):
    return dict(
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        causal=True,
        window_left=window_left,
        lse_mode=lse_mode,
    )


# The M14 shape: one request, q = kv = 2048, 32 query / 8 KV heads, head_dim 128.
M14 = dict(q_lens=[2048], kv_lens=[2048], num_qo_heads=32, num_kv_heads=8)


def _fa2_planner_float_bytes(p, *, graph, total_q_tokens=None, window_left=-1):
    """What the fa2 planner allocates from the float workspace for ``p``:
    the legacy wrapper's ``workspace_size()`` (PrefillPlanWorkspaceSize, the
    planner in counting mode), in the same mode and with the same row budget
    the facade's fa backend plans with."""
    dev = torch.device(p["device"])
    page = p["page_size"]
    kv = p["kv_seq_lens_cpu"]
    pages = (kv + page - 1) // page
    kv_indptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(pages, 0, dtype=torch.int32)]
    )
    last = torch.where(pages > 0, kv - (pages - 1) * page, torch.tensor(page))
    b = kv.shape[0]
    kw = {}
    if graph:
        i32 = dict(dtype=torch.int32, device=dev)
        kw = dict(
            use_cuda_graph=True,
            qo_indptr_buf=torch.zeros(b + 1, **i32),
            paged_kv_indptr_buf=torch.zeros(b + 1, **i32),
            paged_kv_indices_buf=torch.zeros(max(int(kv_indptr[-1]), 1), **i32),
            paged_kv_last_page_len_buf=torch.zeros(b, **i32),
        )
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(16, dtype=torch.uint8, device=dev), "HND", backend="fa2", **kw
    )
    if graph:
        # the facade seeds the wrapper's row budget with the capacity
        wrapper._max_total_num_rows = total_q_tokens or int(p["qo_indptr_cpu"][-1])
    float_bytes, _int_bytes = wrapper.workspace_size(
        p["qo_indptr_cpu"],
        kv_indptr,
        torch.arange(int(kv_indptr[-1]), dtype=torch.int32),
        last.to(torch.int32),
        p["num_qo_heads"],
        p["num_kv_heads"],
        p["head_dim_qk"],
        page,
        head_dim_vo=p["head_dim_vo"],
        causal=True,
        window_left=window_left,
        q_data_type=p["dtype"],
        kv_data_type=p.get("kv_dtype"),
        seq_lens=kv,
    )
    return int(float_bytes)


def _fa2_need_via_plan(p, *, graph, total_q_tokens=None, window_left=-1):
    """The per-batch need the plan-time check computes, read back from the
    ValueError a 16-byte buffer provokes (0 when the plan goes through: the
    planner touches no float scratch and a 16-byte buffer is enough)."""
    dev = torch.device(p["device"])
    tiny = torch.empty(16, dtype=torch.uint8, device=dev)
    attn = PagedAttention(
        tiny,
        graph_capacity=_capacity(p, total_q_tokens=total_q_tokens) if graph else None,
    )
    try:
        attn.plan(
            make_metadata(p), backend="fa2", **_plan_kwargs(p, window_left=window_left)
        )
    except ValueError as exc:
        m = re.search(r"needs (\d+) bytes of scratch workspace", str(exc))
        assert m, f"unexpected ValueError: {exc}"
        assert attn.backend is None, "a rejected plan must publish nothing"
        return int(m.group(1))
    return 0


def _requirements(p, *, graph, total_q_tokens=None, window_left=-1, backend="fa2"):
    return PagedAttention.workspace_requirements(
        _capacity(p, total_q_tokens=total_q_tokens),
        device=torch.device(p["device"]),
        use_cuda_graph=graph,
        backend=backend,
        **{
            k: v
            for k, v in _plan_kwargs(p, window_left=window_left).items()
            if k not in ("lse_mode",)
        },
    )


@needs_cuda
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_m14_shape_runs_on_a_128_mib_workspace(graph):
    """The shape that overflowed the legacy wrappers' documented 128 MiB
    before #5177 (fa2 graph mode now needs 32.25 MiB) plans and runs on a
    buffer of that size and matches the fp32 oracle."""
    p = _exact_problem(seed=14, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    attn = PagedAttention(
        scratch_workspace(dev, 128 * MB),
        graph_capacity=_capacity(p) if graph else None,
    )
    attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize(dev)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# (name, problem kwargs, graph mode, capacity total tokens, window_left)
_SIZING_SHAPES = [
    ("m14-graph", M14, True, None, -1),
    ("m14-eager", M14, False, None, -1),
    (
        "decode-b64-graph",
        dict(q_lens=[1] * 64, kv_lens=[4096] * 64, num_qo_heads=32, num_kv_heads=8),
        True,
        None,
        -1,
    ),
    (
        "ragged-b8-graph-cap2048",
        dict(
            q_lens=[5, 100, 7, 300, 1, 64, 33, 900],
            kv_lens=[5, 400, 7, 300, 2000, 64, 33, 900],
            num_qo_heads=32,
            num_kv_heads=8,
        ),
        True,
        2048,
        -1,
    ),
    (
        "ragged-b8-eager",
        dict(
            q_lens=[5, 100, 7, 300, 1, 64, 33, 900],
            kv_lens=[5, 400, 7, 300, 2000, 64, 33, 900],
            num_qo_heads=32,
            num_kv_heads=8,
        ),
        False,
        None,
        -1,
    ),
    (
        "mqa-32-1-eager",
        dict(q_lens=[4] * 64, kv_lens=[8192] * 64, num_qo_heads=32, num_kv_heads=1),
        False,
        None,
        -1,
    ),
    (
        "mqa-32-1-graph",
        dict(q_lens=[4] * 64, kv_lens=[8192] * 64, num_qo_heads=32, num_kv_heads=1),
        True,
        None,
        -1,
    ),
    (
        "hd256-eager",
        dict(
            q_lens=[512, 512],
            kv_lens=[1024, 2048],
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=256,
        ),
        False,
        None,
        -1,
    ),
    (
        "hd64-page1-eager",
        dict(
            q_lens=[512, 512],
            kv_lens=[1024, 2048],
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            page_size=1,
            input_form="page_indices",
        ),
        False,
        None,
        -1,
    ),
    (
        "window-eager",
        dict(q_lens=[512] * 4, kv_lens=[4096] * 4, num_qo_heads=32, num_kv_heads=8),
        False,
        None,
        128,
    ),
    (
        "fp8kv-decode-graph",
        dict(
            q_lens=[1] * 32,
            kv_lens=[2048] * 32,
            num_qo_heads=32,
            num_kv_heads=8,
            kv_dtype=torch.float8_e4m3fn,
        ),
        True,
        None,
        -1,
    ),
    (
        "padding-rows-eager",
        dict(
            q_lens=[5, 1, 7, 1], kv_lens=[37, 0, 64, 0], num_qo_heads=8, num_kv_heads=2
        ),
        False,
        None,
        -1,
    ),
    # head_dim 512 (Gemma-4 full attention): CTA tile 16 / 32 by
    # FA2DetermineCtaTileQ's VO-split branch, modelled in fa_backend.py
    (
        "hd512-gemma-eager",
        dict(
            q_lens=[17, 300],
            kv_lens=[10003, 1024],
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=512,
        ),
        False,
        None,
        -1,
    ),
    (
        "hd512-gemma-graph",
        dict(
            q_lens=[17, 300],
            kv_lens=[10003, 1024],
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=512,
        ),
        True,
        None,
        -1,
    ),
    (
        "hd512-fp8kv-decode-graph",
        dict(
            q_lens=[1] * 8,
            kv_lens=[10003] * 8,
            num_qo_heads=16,
            num_kv_heads=2,
            head_dim=512,
            kv_dtype=torch.float8_e4m3fn,
        ),
        True,
        None,
        -1,
    ),
]


@needs_cuda
@pytest.mark.parametrize(
    "shape,graph,total,window_left",
    [(s, g, t, w) for _, s, g, t, w in _SIZING_SHAPES],
    ids=[n for n, *_ in _SIZING_SHAPES],
)
def test_workspace_requirements_covers_fa2_planner(shape, graph, total, window_left):
    """The bound covers what the fa2 planner allocates, and the per-batch need
    the plan-time check computes IS the planner's number (so the check never
    rejects a batch the planner would accept, nor lets an overflow through)."""
    p = _exact_problem(seed=1, **shape)
    _resolve_or_skip(p, "fa2", window_left=window_left)
    planner = _fa2_planner_float_bytes(
        p, graph=graph, total_q_tokens=total, window_left=window_left
    )
    bound = _requirements(p, graph=graph, total_q_tokens=total, window_left=window_left)
    need = _fa2_need_via_plan(
        p, graph=graph, total_q_tokens=total, window_left=window_left
    )
    assert need == planner, f"plan-time need {need} != planner {planner}"
    assert bound >= planner, f"bound {bound} < planner {planner}"
    if graph:
        # graph mode pads to the capacity: the bound is the planner's number
        assert bound == planner


@needs_cuda
def test_workspace_requirements_bound_is_runnable():
    """A buffer of exactly the bound plans and runs the M14 shape (graph mode,
    where the planner's need is the bound); one byte less is rejected."""
    p = _exact_problem(seed=3, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    bound = _requirements(p, graph=True)
    # 32 heads x ceil(64 tiles x 128 packed rows / gqa 4) x (128 + 1) x 4 B
    assert bound == 33816576
    ws = torch.empty(bound, dtype=torch.uint8, device=dev)
    attn = PagedAttention(ws, graph_capacity=_capacity(p))
    attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize(dev)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()

    short = PagedAttention(ws[: bound - 16], graph_capacity=_capacity(p))
    with pytest.raises(ValueError, match=rf"needs {bound} bytes"):
        short.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))


@needs_cuda
def test_too_small_workspace_rejected_at_plan_fa2():
    """A 32 MiB buffer is a quarter MiB short for the M14 shape in graph
    mode: plan() says so (required and available bytes), before the planner's
    kernel-side 'Buffer overflow' error, and publishes nothing."""
    p = _exact_problem(seed=5, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    ws = torch.empty(32 * MB, dtype=torch.uint8, device=dev)
    attn = PagedAttention(ws, graph_capacity=_capacity(p))
    with pytest.raises(ValueError) as info:
        attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    msg = str(info.value)
    assert "33816576 bytes" in msg and f"holds {32 * MB} bytes" in msg
    assert "workspace_requirements" in msg
    assert "Buffer overflow" not in msg
    # a named backend never falls back: its name and the whole rejection text
    # reach the caller
    assert msg == (
        "backend 'fa2' cannot plan this batch: "
        + _workspace_too_small(33816576, 32 * MB)
    )
    assert attn.backend is None


@needs_cuda
def test_too_small_workspace_rejected_at_plan_trtllm_gen():
    """trtllm-gen's softmax-stats carve-out (LSE plans) is checked the same way."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, "trtllm-gen")
    dev = torch.device(p["device"])
    need = _requirements(p, graph=False, backend="trtllm-gen")
    # 8 B x heads x batch x round_up(max_q_len, 256) + the 1 MiB guard
    assert need == 8 * p["num_qo_heads"] * 4 * 256 + MB
    attn = PagedAttention(torch.empty(MB, dtype=torch.uint8, device=dev))
    with pytest.raises(ValueError, match=rf"needs {need} bytes") as info:
        attn.plan(make_metadata(p), backend="trtllm-gen", **_plan_kwargs(p))
    assert str(info.value) == (
        "backend 'trtllm-gen' cannot plan this batch: " + _workspace_too_small(need, MB)
    )
    assert attn.backend is None
    # without an LSE output nothing is carved out: the same buffer plans
    attn.plan(
        make_metadata(p), backend="trtllm-gen", **_plan_kwargs(p, lse_mode="none")
    )
    assert attn.backend == "trtllm-gen"


def _lead_with(monkeypatch, *leading):
    """Replace the auto policy: the ``leading`` candidates are walked first,
    in that order, then the rest of the set.  The tests below are about the
    walk, not about what the policy ranks first for a shape."""
    monkeypatch.setattr(
        _controller,
        "plan_order",
        lambda candidates, cc, facts: (
            *(name for name in leading if name in candidates),
            *(name for name in candidates if name not in leading),
        ),
    )


@needs_cuda
def test_too_small_workspace_moves_auto_to_the_next_candidate(monkeypatch):
    """The 32 MiB buffer fa2 cannot plan the M14 graph bucket on is enough
    for another candidate: with fa2 leading the walk, auto records fa2's
    rejection (bytes and sizing hint), plans the next candidate that accepts
    and runs it on that buffer."""
    p = _exact_problem(seed=6, **M14)
    res = _resolve_or_skip(p, "auto")
    if "fa2" not in res.backends or len(res.backends) < 2:
        pytest.skip("needs fa2 and one more runnable candidate")
    _lead_with(monkeypatch, "fa2")
    dev = torch.device(p["device"])
    ws = torch.empty(32 * MB, dtype=torch.uint8, device=dev)
    attn = PagedAttention(ws, graph_capacity=_capacity(p))
    attn.plan(make_metadata(p), backend="auto", **_plan_kwargs(p))
    trace = attn._impl.selection_trace
    assert trace[0] == (
        "fa2",
        "preflight",
        _workspace_too_small(33816576, 32 * MB),
    )
    assert trace[-1] == (attn.backend, "preflight", "accepted")
    assert attn.backend != "fa2"
    assert "fa2 [preflight]: needs 33816576 bytes" in attn.explain()
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize(dev)
    ref_out, ref_lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"],
        p["page_size"],
        True,
    )
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


# ---------------------------------------------------------------------------
# The same contract on the host: the real controller walk and the real fa2 /
# trtllm-gen preflight(), no kernel and no device.
# ---------------------------------------------------------------------------

_M14_PLAN = dict(
    num_qo_heads=32,
    num_kv_heads=8,
    head_dim_qk=128,
    q_dtype=torch.bfloat16,
    lse_mode="base2",
)
_M14_CAPACITY = GraphCapacity(
    batch_size=1,
    total_q_tokens=2048,
    max_q_len=2048,
    max_kv_len=2048,
    page_size=16,
    table_width=128,
)
# what each backend carves out for the M14 graph bucket with an LSE output
_M14_NEED = {
    "fa2": 33816576,
    # 8 B x 32 heads x 1 request x 2048 tokens + the 1 MiB guard
    "trtllm-gen": 8 * 32 * 2048 + MB,
}


def _host_controller(monkeypatch, workspace_bytes, *, first):
    """A graph-mode controller for the M14 bucket on the host, as SM100 sees
    it (candidates fa2, trtllm-gen and cudnn), walked ``first``, then the
    other of fa2 / trtllm-gen, then cuDNN.  fa2 and trtllm-gen are the real
    backends with their real preflight() and a plan() that does nothing; the
    walk must not get as far as cuDNN."""

    def make_backend(name, device, kv_layout, workspace, *, graph_capacity=None):
        if name == "fa2":
            # the constructor reads the device and builds the wrapper;
            # preflight() reads only these (148 SMs: a B200)
            backend = object.__new__(_FaBackend)
            backend.name, backend._workspace = name, workspace
            backend._graph_capacity = graph_capacity
            backend._sm_count, backend._smem_optin = 148, None
        else:
            assert name == "trtllm-gen", f"the walk reached {name}"
            backend = _TrtllmGenBackend(device, kv_layout, workspace)
        backend.plan = lambda meta, derived: None
        return backend

    monkeypatch.setattr(
        _selection, "PROBES", {name: (lambda device: None) for name in CAPABILITIES}
    )
    monkeypatch.setattr(
        _controller,
        "resolve_paged_attention",
        lambda *, device, **kwargs: resolve_paged_attention(cc_major=10, **kwargs),
    )
    monkeypatch.setattr(
        PagedAttentionController, "_device_binding", lambda self: (10, 0, 0)
    )
    monkeypatch.setattr(_controller, "make_backend", make_backend)
    monkeypatch.setattr(_controller, "derived_needs", lambda name: frozenset())
    monkeypatch.setattr(_controller, "_expect_not_capturing", lambda what: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: "stream")
    _lead_with(monkeypatch, first, *(n for n in _M14_NEED if n != first))
    return PagedAttentionController(
        torch.empty(workspace_bytes, dtype=torch.uint8),
        graph_capacity=_M14_CAPACITY,
    )


def test_workspace_overflow_is_a_typed_rejection_auto_walks_past(monkeypatch):
    """backend="auto", fa2 leading on a 32 MiB buffer: fa2's overflow is one
    more typed preflight rejection, so the walk goes on to trtllm-gen (1.5 MiB
    of softmax stats) and the trace keeps fa2's bytes."""
    attn = _host_controller(monkeypatch, 32 * MB, first="fa2")
    attn.plan(_host_metadata([2048], [2048]), **_M14_PLAN)
    assert attn.backend == "trtllm-gen"
    assert attn.selection_trace == (
        ("fa2", "preflight", _workspace_too_small(33816576, 32 * MB)),
        ("trtllm-gen", "preflight", "accepted"),
    )
    # a buffer of exactly fa2's need: no rejection, the walk ends at fa2
    attn = _host_controller(monkeypatch, _M14_NEED["fa2"], first="fa2")
    attn.plan(_host_metadata([2048], [2048]), **_M14_PLAN)
    assert attn.selection_trace == (("fa2", "preflight", "accepted"),)


def test_workspace_overflow_of_every_candidate_is_a_value_error(monkeypatch):
    """No candidate of the set fits a 1 MiB buffer (trtllm-gen's rejection is
    typed too, or the walk would stop on it): a ValueError that names each
    one's bytes, and nothing is published."""
    attn = _host_controller(monkeypatch, MB, first="fa2")
    pinned = dataclasses.replace(
        resolve_paged_attention(
            cc_major=10,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
            page_size=16,
            need_lse=True,
        ),
        backends=("fa2", "trtllm-gen"),
    )
    with pytest.raises(ValueError, match="no pinned candidate") as info:
        attn.plan(_host_metadata([2048], [2048]), backend=pinned, **_M14_PLAN)
    for name, need in _M14_NEED.items():
        assert f"{name}: {_workspace_too_small(need, MB)}" in str(info.value)
    assert attn.backend is None and attn.selection_trace == ()


@pytest.mark.parametrize("backend", sorted(_M14_NEED))
def test_workspace_overflow_of_a_named_backend_is_a_value_error(monkeypatch, backend):
    """A named backend never falls back: the typed rejection becomes a plain
    ValueError whose text is the rejection's (bytes needed and held, the
    workspace_requirements hint), and nothing is published."""
    attn = _host_controller(monkeypatch, MB, first=backend)
    with pytest.raises(ValueError) as info:
        attn.plan(_host_metadata([2048], [2048]), backend=backend, **_M14_PLAN)
    rejection = _workspace_too_small(_M14_NEED[backend], MB)
    assert str(info.value) == f"backend {backend!r} cannot plan this batch: {rejection}"
    assert "workspace_requirements" in rejection
    assert type(info.value) is ValueError
    assert type(info.value.__cause__) is _BackendPlanUnsupportedError
    assert str(info.value.__cause__) == rejection
    assert attn.backend is None


@needs_cuda
def test_workspace_requirements_auto_is_max_over_candidates():
    p = _exact_problem(seed=9, **M14)
    dev = torch.device(p["device"])
    res = _resolve_or_skip(p, "auto")
    auto = _requirements(p, graph=True, backend="auto")
    per_backend = {
        name: _requirements(p, graph=True, backend=name) for name in res.backends
    }
    assert auto == max(per_backend.values()), per_backend
    assert _requirements(p, graph=True, backend=res) == auto
    # eager-only bound does not depend on the token count, graph does
    eager = _requirements(p, graph=False, backend="fa2")
    assert eager == _requirements(p, graph=False, total_q_tokens=8192, backend="fa2")
    assert _requirements(p, graph=True, total_q_tokens=8192, backend="fa2") > auto
    with pytest.raises(ValueError, match="GraphCapacity"):
        PagedAttention.workspace_requirements(
            dict(batch_size=1),
            device=dev,
            num_qo_heads=32,
            num_kv_heads=8,
            head_dim_qk=128,
            q_dtype=torch.bfloat16,
        )
