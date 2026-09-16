"""Micro-benchmark: PagedAttention metadata construction and plan() cost.

For every (shape, backend, input form, host mirrors present/absent) this
script measures, on the experimental ``flashinfer.prefill.PagedAttention``:

  (a) ``PagedAttentionMetadata`` construction,
  (b) eager ``plan()`` on a fresh metadata object,
  (c) graph-mode re-plan (``PagedAttention(use_cuda_graph=True)``, every plan
      after the first one, which fixes the capture shapes),

as synchronized wall time (median of ``--repeat`` runs; the wall includes the
``torch.cuda.synchronize()`` that waits for whatever the call enqueued), plus
the GPU-side activity one warm call enqueues, taken from ``torch.profiler``:
kernel launches and memcpys by kind.  ``Memcpy HtoD (Pageable -> Device)`` is
reported on its own because a non_blocking upload from pageable memory is a
blocking staging copy on the host - the failure mode this benchmark exists to
catch.

Optionally (``--with-run``) it also counts the GPU activity of one ``run()``
with ``lse_mode="base2"`` and a caller-provided ``lse`` buffer, which is where
the cuDNN backend's padded-to-packed LSE normalisation shows up.

Nothing here is an end-to-end serving number: the KV pool is a stand-in and
the kernels run once, only to make the launch counts complete.

Usage (inside the test container)::

    python benchmarks/bench_paged_attention_plan.py \
        --backends fa2,cudnn,trtllm-gen --forms dense,csr --mirrors with,without \
        --shapes chunked,ragged --repeat 30 --json /tmp/plan_cost.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from typing import Callable, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile

from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

# (per-request q lens, per-request kv lens, page size); heads/dims are shared
SHAPES = {
    "short": ([128], [128], 16),
    "chunked": ([128] * 4, [2048] * 4, 16),
    "ragged": ([40, 17, 9], [300, 210, 64], 64),
    "decode64": ([1] * 64, [1024] * 64, 16),
}
HEADS = (32, 8)
HEAD_DIM = 128
DTYPE = torch.bfloat16

MEMCPY_KINDS = (
    "Memcpy HtoD (Pageable -> Device)",
    "Memcpy HtoD (Pinned -> Device)",
    "Memcpy DtoH (Device -> Pageable)",
    "Memcpy DtoH (Device -> Pinned)",
    "Memcpy DtoD (Device -> Device)",
)


# ----------------------------------------------------------------------------
# problem construction
# ----------------------------------------------------------------------------


class Problem:
    def __init__(self, name: str, q_lens, kv_lens, page: int, device: torch.device):
        g = torch.Generator().manual_seed(0)
        self.name = name
        self.device = device
        self.page = page
        self.batch = len(q_lens)
        self.max_q = max(q_lens)
        self.max_kv = max(kv_lens)
        q_lens_t = torch.tensor(q_lens, dtype=torch.int32)
        kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
        self.qo_indptr_cpu = torch.zeros(self.batch + 1, dtype=torch.int32)
        torch.cumsum(q_lens_t, 0, dtype=torch.int32, out=self.qo_indptr_cpu[1:])
        self.qo_indptr_cpu = self.qo_indptr_cpu.pin_memory()
        self.kv_seq_lens_cpu = kv_lens_t.pin_memory()
        self.qo_indptr = self.qo_indptr_cpu.to(device)
        self.kv_seq_lens = self.kv_seq_lens_cpu.to(device)
        pages = (kv_lens_t + page - 1) // page
        # exactly ceil(max_kv / page) columns: cuDNN rejects wider tables
        # today (the narrow-view fix is a separate work package), and the
        # plan cost does not depend on the tail
        width = int(pages.max())
        pool = self.batch * width + 8
        perm = torch.randperm(pool, generator=g, dtype=torch.int32)
        table = torch.zeros(self.batch, width, dtype=torch.int32)
        flat = []
        off = 0
        for i in range(self.batch):
            n = int(pages[i])
            table[i, :n] = perm[off : off + n]
            flat.append(perm[off : off + n])
            # unused tail lanes carry a stale in-pool id, as a persistent
            # engine table would
            table[i, n:] = perm[-1]
            off += n
        self.block_tables = table.to(device)
        # sglang-style flat page ids with an over-allocated tail
        flat_t = torch.cat(flat + [torch.zeros(256, dtype=torch.int32)])
        self.kv_page_indices = flat_t.to(device)
        self.total_q = int(self.qo_indptr_cpu[-1])
        hq, hk = HEADS
        self.q = torch.randn(self.total_q, hq, HEAD_DIM, dtype=DTYPE, device=device)
        self.k = torch.randn(pool, hk, page, HEAD_DIM, dtype=DTYPE, device=device)
        self.v = torch.randn_like(self.k)
        self.out = torch.empty_like(self.q)
        self.lse = torch.empty(self.total_q, hq, dtype=torch.float32, device=device)

    def metadata(self, form: str, mirrors: bool) -> PagedAttentionMetadata:
        common = dict(
            page_size=self.page,
            max_q_len=self.max_q,
            max_kv_len=self.max_kv,
        )
        if mirrors:
            common.update(
                qo_indptr_cpu=self.qo_indptr_cpu, kv_seq_lens_cpu=self.kv_seq_lens_cpu
            )
        if form == "dense":
            return PagedAttentionMetadata.dense(
                self.qo_indptr, self.kv_seq_lens, self.block_tables, **common
            )
        return PagedAttentionMetadata.csr(
            self.qo_indptr, self.kv_seq_lens, self.kv_page_indices, **common
        )


# ----------------------------------------------------------------------------
# measurement helpers
# ----------------------------------------------------------------------------


def wall_us(fns: List[Callable[[], object]], warmup: int) -> Dict[str, float]:
    """Synchronized wall time of each call in ``fns`` (one fresh input per
    call); the first ``warmup`` calls are discarded."""
    values = []
    for i, fn in enumerate(fns):
        torch.cuda.synchronize()
        t = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        if i >= warmup:
            values.append((time.perf_counter_ns() - t) / 1000)
    return dict(
        median=statistics.median(values),
        min=min(values),
        max=max(values),
        n=len(values),
    )


def gpu_activity(fn: Callable[[], object]) -> Dict[str, int]:
    """GPU-side events one call enqueues: kernels and memcpys by kind."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    counts = {k: 0 for k in MEMCPY_KINDS}
    counts.update(kernels=0, memset=0, other_memcpy=0)
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = e.name
        if name in counts:
            counts[name] += 1
        elif name.startswith("Memcpy"):
            counts["other_memcpy"] += 1
        elif name.startswith("Memset"):
            counts["memset"] += 1
        else:
            counts["kernels"] += 1
    return counts


def fmt_activity(c: Dict[str, int]) -> str:
    parts = [f"k={c['kernels']}"]
    short = {
        "Memcpy HtoD (Pageable -> Device)": "H2D-pageable",
        "Memcpy HtoD (Pinned -> Device)": "H2D-pinned",
        "Memcpy DtoH (Device -> Pageable)": "D2H-pageable",
        "Memcpy DtoH (Device -> Pinned)": "D2H-pinned",
        "Memcpy DtoD (Device -> Device)": "D2D",
    }
    for k, s in short.items():
        if c[k]:
            parts.append(f"{s}={c[k]}")
    if c["memset"]:
        parts.append(f"memset={c['memset']}")
    if c["other_memcpy"]:
        parts.append(f"memcpy?={c['other_memcpy']}")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# one (shape, backend, form, mirrors) cell
# ----------------------------------------------------------------------------


def bench_cell(
    prob: Problem,
    backend: str,
    form: str,
    mirrors: bool,
    *,
    repeat: int,
    warmup: int,
    lse_mode: str,
    with_run: bool,
    workspace: torch.Tensor,
) -> Dict[str, object]:
    dev = prob.device
    hq, hk = HEADS
    spec = dict(
        num_qo_heads=hq,
        num_kv_heads=hk,
        head_dim_qk=HEAD_DIM,
        q_dtype=DTYPE,
        kv_layout="HND",
        causal=True,
        lse_mode=lse_mode,
    )
    row: Dict[str, object] = dict(
        shape=prob.name, backend=backend, form=form, mirrors=mirrors
    )
    try:
        resolution = resolve_paged_attention(
            device=dev,
            page_size=prob.page,
            backend=backend,
            need_lse=lse_mode != "none",
            kv_input_form="block_tables" if form == "dense" else "page_indices",
            **{k: v for k, v in spec.items() if k != "lse_mode"},
        )
    except ValueError as e:
        row["status"] = f"unsupported: {e}"
        return row

    n = repeat + warmup
    # (a) metadata construction
    row["meta_us"] = wall_us([lambda: prob.metadata(form, mirrors)] * n, warmup)
    row["meta_gpu"] = gpu_activity(lambda: prob.metadata(form, mirrors))

    # (b) eager plan on a fresh metadata object each time (derivation is
    # cached per object, so reusing one object would measure nothing)
    attn = PagedAttention(dev, workspace_buffer=workspace)
    mds = [prob.metadata(form, mirrors) for _ in range(n)]
    plan_calls = [
        (lambda md=md: attn.plan(md, backend=resolution, **spec)) for md in mds
    ]
    row["plan_us"] = wall_us(plan_calls, warmup)
    md = prob.metadata(form, mirrors)
    row["plan_gpu"] = gpu_activity(lambda: attn.plan(md, backend=resolution, **spec))
    row["chosen"] = attn.backend

    # (c) graph-mode re-plan: the first plan fixes the capture shapes and is
    # excluded; every later plan stages into the reserved storage
    attn_g = PagedAttention(dev, use_cuda_graph=True, workspace_buffer=workspace)
    attn_g.plan(prob.metadata(form, mirrors), backend=resolution, **spec)
    mds_g = [prob.metadata(form, mirrors) for _ in range(n)]
    replan_calls = [
        (lambda md=md: attn_g.plan(md, backend=resolution, **spec)) for md in mds_g
    ]
    row["replan_us"] = wall_us(replan_calls, warmup)
    md = prob.metadata(form, mirrors)
    row["replan_gpu"] = gpu_activity(
        lambda: attn_g.plan(md, backend=resolution, **spec)
    )

    if with_run:
        scale = 1.0 / math.sqrt(HEAD_DIM)
        kwargs = dict(out=prob.out, sm_scale=scale)
        if lse_mode != "none":
            kwargs["lse"] = prob.lse
        run = lambda: attn.run(prob.q, (prob.k, prob.v), **kwargs)  # noqa: E731
        for _ in range(3):
            run()
        row["run_gpu"] = gpu_activity(run)
        row["run_us"] = wall_us([run] * n, warmup)
    row["status"] = "ok"
    return row


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------


def print_tables(rows: List[Dict[str, object]], with_run: bool) -> None:
    def us(v: Optional[Dict[str, float]]) -> str:
        return f"{v['median']:.1f}" if v else "-"

    print("\n### timing (median synchronized wall, us)\n")
    hdr = "| shape | backend | form | mirrors | metadata | plan (eager) | re-plan (graph) |"
    if with_run:
        hdr += " run |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        if r.get("status") != "ok":
            print(
                f"| {r['shape']} | {r['backend']} | {r['form']} | "
                f"{'yes' if r['mirrors'] else 'no'} | {r.get('status')} | | |"
                + (" |" if with_run else "")
            )
            continue
        line = (
            f"| {r['shape']} | {r['backend']} | {r['form']} | "
            f"{'yes' if r['mirrors'] else 'no'} | {us(r['meta_us'])} | "
            f"{us(r['plan_us'])} | {us(r['replan_us'])} |"
        )
        if with_run:
            line += f" {us(r.get('run_us'))} |"
        print(line)

    print("\n### GPU activity of one warm call (k = kernel launches)\n")
    hdr = "| shape | backend | form | mirrors | metadata | plan (eager) | re-plan (graph) |"
    if with_run:
        hdr += " run |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        if r.get("status") != "ok":
            continue
        line = (
            f"| {r['shape']} | {r['backend']} | {r['form']} | "
            f"{'yes' if r['mirrors'] else 'no'} | {fmt_activity(r['meta_gpu'])} | "
            f"{fmt_activity(r['plan_gpu'])} | {fmt_activity(r['replan_gpu'])} |"
        )
        if with_run:
            line += f" {fmt_activity(r['run_gpu'])} |"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backends", default="fa2,cudnn,trtllm-gen")
    ap.add_argument("--forms", default="dense,csr")
    ap.add_argument("--mirrors", default="with,without")
    ap.add_argument("--shapes", default="chunked", help=",".join(SHAPES))
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lse-mode", default="none", choices=["none", "base2", "basee"])
    ap.add_argument(
        "--with-run",
        action="store_true",
        help="also time one run() and count its GPU activity",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None, help="write the raw rows here")
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.cuda.set_device(dev)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev)
    rows: List[Dict[str, object]] = []
    for shape in args.shapes.split(","):
        prob = Problem(shape, *SHAPES[shape], dev)
        for backend in args.backends.split(","):
            for form in args.forms.split(","):
                for mirrors in args.mirrors.split(","):
                    row = bench_cell(
                        prob,
                        backend,
                        form,
                        mirrors == "with",
                        repeat=args.repeat,
                        warmup=args.warmup,
                        lse_mode=args.lse_mode,
                        with_run=args.with_run,
                        workspace=workspace,
                    )
                    rows.append(row)
                    print(
                        f"done {shape} {backend} {form} mirrors={mirrors}: "
                        f"{row.get('status')}",
                        flush=True,
                    )
    print(
        f"\ntorch {torch.__version__}, {torch.cuda.get_device_name(dev)}, "
        f"repeat={args.repeat}, lse_mode={args.lse_mode}"
    )
    print_tables(rows, args.with_run)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        print(f"\nraw rows written to {args.json}")


if __name__ == "__main__":
    main()
