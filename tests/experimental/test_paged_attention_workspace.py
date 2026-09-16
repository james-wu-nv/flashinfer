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

Sizing (ledger M14): the fa2 split-KV planner carves ``tmp_v`` / ``tmp_s`` out
of this buffer and the legacy 128 MiB default overflowed on one request of
q = kv = 2048 with 32/8 heads in graph mode (129 MiB).  Contract:

4. That shape plans and runs on the default workspace and matches the oracle.
5. ``PagedAttention.workspace_requirements()`` is >= the bytes the fa2 planner
   actually allocates (observed through the legacy wrapper's
   ``workspace_size()``, the planner's own counting mode), and the per-batch
   need the plan-time check computes equals the planner's.
6. A buffer smaller than a batch's need is rejected at ``plan()`` with a
   ``ValueError`` naming the required bytes, before any kernel-side error.
"""

import gc
import re

import pytest
import torch

from flashinfer.experimental.paged_attention._controller import _WORKSPACE_BYTES
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    GraphCapacity,
    PagedAttention,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    BACKENDS,
    LSE_TOL,
    OUT_TOL,
    _resolve_or_skip,
    make_metadata,
    make_problem,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MB = 1024 * 1024
# What _controller._shared_workspace() reserves per device (the trtllm-gen
# backend runs on the same buffer).
WORKSPACE_MB = _WORKSPACE_BYTES // MB
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
        k_scale = float(k.abs().amax().item()) / 448.0
        v_scale = float(v.abs().amax().item()) / 448.0
        k = (k.float() / k_scale).to(kv_dtype)
        v = (v.float() / v_scale).to(kv_dtype)
        k_ref, v_ref = k.float() * k_scale, v.float() * v_scale
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
        dev,
        graph_capacity=_capacity(p, total_q_tokens=total_q_tokens) if graph else None,
        workspace_buffer=tiny,
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


@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_m14_shape_runs_on_default_workspace(graph):
    """The shape that overflowed the 128 MiB default (fa2 graph mode needs
    129 MiB) plans and runs on the default shared workspace and matches the
    fp32 oracle."""
    p = _exact_problem(seed=14, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    attn = PagedAttention(dev, graph_capacity=_capacity(p) if graph else None)
    attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    assert attn._impl._workspace.numel() == _WORKSPACE_BYTES
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
]


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


def test_workspace_requirements_bound_is_runnable():
    """A buffer of exactly the bound plans and runs the M14 shape (graph mode,
    where the planner's need is the bound); one byte less is rejected."""
    p = _exact_problem(seed=3, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    bound = _requirements(p, graph=True)
    assert bound == 135266304  # 32 heads x 64 tiles x 128 rows x (128 + 1) x 4 B
    ws = torch.empty(bound, dtype=torch.uint8, device=dev)
    attn = PagedAttention(dev, graph_capacity=_capacity(p), workspace_buffer=ws)
    attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.cuda.synchronize(dev)
    assert torch.isfinite(out).all() and torch.isfinite(lse).all()

    short = PagedAttention(
        dev, graph_capacity=_capacity(p), workspace_buffer=ws[: bound - 16]
    )
    with pytest.raises(ValueError, match=rf"needs {bound} bytes"):
        short.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))


def test_too_small_workspace_rejected_at_plan_fa2():
    """The legacy 128 MiB default is one MiB short for the M14 shape in graph
    mode: plan() says so (required and available bytes), before the planner's
    kernel-side 'Buffer overflow' error, and publishes nothing."""
    p = _exact_problem(seed=5, **M14)
    _resolve_or_skip(p, "fa2")
    dev = torch.device(p["device"])
    ws = torch.empty(128 * MB, dtype=torch.uint8, device=dev)
    attn = PagedAttention(dev, graph_capacity=_capacity(p), workspace_buffer=ws)
    with pytest.raises(ValueError) as info:
        attn.plan(make_metadata(p), backend="fa2", **_plan_kwargs(p))
    msg = str(info.value)
    assert "135266304 bytes" in msg and f"holds {128 * MB} bytes" in msg
    assert "workspace_requirements" in msg
    assert "Buffer overflow" not in msg
    assert attn.backend is None


def test_too_small_workspace_rejected_at_plan_trtllm_gen():
    """trtllm-gen's softmax-stats carve-out (LSE plans) is checked the same way."""
    p = make_problem(seed=7, **_SHAPE)
    _resolve_or_skip(p, "trtllm-gen")
    dev = torch.device(p["device"])
    need = _requirements(p, graph=False, backend="trtllm-gen")
    # 8 B x heads x batch x round_up(max_q_len, 256) + the 1 MiB guard
    assert need == 8 * p["num_qo_heads"] * 4 * 256 + MB
    attn = PagedAttention(
        dev, workspace_buffer=torch.empty(MB, dtype=torch.uint8, device=dev)
    )
    with pytest.raises(ValueError, match=rf"needs {need} bytes"):
        attn.plan(make_metadata(p), backend="trtllm-gen", **_plan_kwargs(p))
    assert attn.backend is None
    # without an LSE output nothing is carved out: the same buffer plans
    attn.plan(
        make_metadata(p), backend="trtllm-gen", **_plan_kwargs(p, lse_mode="none")
    )
    assert attn.backend == "trtllm-gen"


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
