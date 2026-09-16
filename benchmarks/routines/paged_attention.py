"""``PagedAttention`` routine for the shared benchmark CLI.

Times the experimental :class:`flashinfer.prefill.PagedAttention` facade on
ONE set of inputs for every requested backend.  Each candidate is gated on
the fp32 oracle (``tests/experimental/paged_attention_reference.py``) before
it is timed, and every (backend, phase) pair yields one CSV row:

``plan``
    Build the per-step :class:`PagedAttentionMetadata` (with host mirrors, as
    an engine that owns them would) and call ``PagedAttention.plan``.
    Synchronized host wall time.  With CUDA graphs on (the default) the
    instance is constructed with ``use_cuda_graph=True``, so this is the
    graph-mode re-plan into reserved storage; with ``--no_cuda_graph`` it is
    the eager plan.
``run``
    A warmed ``PagedAttention.run`` with preallocated ``out``/``lse``.  GPU
    time from ``bench_gpu_time`` (CUDA graph replay by default, eager with
    ``--no_cuda_graph``), cold L2.
``step``
    One ``plan`` followed by N ``run`` calls (N from ``--pa_layers``), eager,
    synchronized host wall time — the per-scheduler-step cost an engine pays
    for a model with N attention layers sharing one plan.

Rows for unsupported / erroring / incorrect candidates are KEPT with a
``status`` column (never dropped, never NaN-only); ``auto`` rows record the
resolved backend.  ``auto`` is the facade's static selection, not autotuning.

``--pa_legacy`` adds ``api_variant=legacy`` rows: the SAME inputs through the
legacy public API of the same kernel, with the same phases, so
``facade overhead = unified / legacy - 1`` can be read per phase:

- fa2/fa3: ``BatchPrefillWithPagedKVCacheWrapper``.  Its ``plan`` is fed the
  page-unit CSR metadata an engine holds — derived on the device from the
  canonical tensors for the CSR form (sglang-style; the wrapper then copies
  indptr/last-page lengths to the host itself, a D2H sync the unified plan
  with host mirrors does not pay) or on the host from the host block table
  for the dense form (vLLM-style; pageable H2D instead).
- cudnn: ``cudnn_batch_prefill_with_kv_cache`` (dense table only).
- trtllm-gen: ``trtllm_batch_context_with_kv_cache`` (dense table only).

Legacy rows keep each API's NATIVE LSE contract (fa/trtllm-gen: packed
base-2; cuDNN: padded ``(batch, max_q, heads)`` in the requested base); the
oracle comparison normalizes outside the timed region, so with an LSE
requested the unified/legacy ratio includes the facade's normalization.
"""

from collections import defaultdict
import importlib.util
import math
import pathlib
import time
import traceback

import numpy as np
import torch

import flashinfer
from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    PagedAttention,
    PagedAttentionMetadata,
    cudnn_batch_prefill_with_kv_cache,
    resolve_paged_attention,
    trtllm_batch_context_with_kv_cache,
)
from flashinfer.testing.utils import (
    attention_tb_per_sec_with_actual_seq_lens,
    attention_tflops_per_sec_with_actual_seq_lens,
    bench_gpu_time,
)

from .attention import sample_actual_seq_lens
from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    get_device,
    is_close_stats,
    print_perf_metrics,
)

PAGED_ATTENTION_BACKENDS = ("fa2", "fa3", "cudnn", "trtllm-gen", "auto")
LN2 = math.log(2.0)
# Same tolerances as tests/experimental/test_paged_attention_prototype.py.
OUT_TOL = dict(rtol=2e-2, atol=2e-2)
LSE_TOL = dict(rtol=2e-2, atol=3e-2)
# The dense table is wider than any request needs and the pool has spare
# pages: a backend that reads past a row's live prefix hits real (finite)
# pages that belong to nobody, so the oracle comparison catches it.
TABLE_SLACK_COLUMNS = 3
POOL_SLACK_PAGES = 8
WORKSPACE_BYTES = 128 * 1024 * 1024


def _load_reference_oracle():
    """Load ``reference_paged_prefill`` from ``tests/experimental`` by path.

    ``benchmarks/`` is not a package that can import ``tests``; loading the
    oracle by file keeps ONE copy of the math (the one the experimental test
    suite and fuzzer use) and keeps ``flashinfer_benchmark.py --help`` free of
    any dependency on it.
    """
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "tests"
        / "experimental"
        / "paged_attention_reference.py"
    )
    if not path.is_file():
        raise FileNotFoundError(f"--refcheck needs the fp32 oracle at {path}")
    spec = importlib.util.spec_from_file_location("paged_attention_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_paged_prefill


def _cupti_available():
    try:
        return importlib.util.find_spec("cupti") is not None
    except (ImportError, ValueError):
        return False


def _build_case(args, device, q_dtype, kv_dtype, head_dim_vo):
    """Q/K/V, exact per-request lengths and a shuffled page mapping."""
    batch_size, page_size = args.batch_size, args.page_size
    num_qo_heads, num_kv_heads = args.num_qo_heads, args.num_kv_heads
    head_dim_qk = args.head_dim_qk

    # Lengths follow the other attention routines: fixed at --s_qo/--s_kv, or
    # sampled in [1, max] with --random_actual_seq_len (kv >= q per request;
    # kv == q when s_qo == s_kv).
    q_lens = sample_actual_seq_lens(
        args.s_qo, batch_size, None, args.random_actual_seq_len
    ).flatten()
    if args.s_qo == args.s_kv:
        kv_lens = q_lens.clone()
    else:
        kv_lens = torch.maximum(
            sample_actual_seq_lens(
                args.s_kv, batch_size, None, args.random_actual_seq_len
            ).flatten(),
            q_lens,
        )
    q_lens = q_lens.to(torch.int32)
    kv_lens = kv_lens.to(torch.int32)
    qo_indptr_cpu = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, 0, dtype=torch.int32)]
    )
    pages_per_seq = (kv_lens + page_size - 1) // page_size
    width = int(pages_per_seq.max()) + TABLE_SLACK_COLUMNS
    pool_pages = int(pages_per_seq.sum()) + POOL_SLACK_PAGES

    # Every request gets a random, non-contiguous set of physical pages; the
    # unused table slots hold valid-but-foreign page ids (poison), never zeros.
    perm = torch.randperm(pool_pages, dtype=torch.int32)
    block_tables_cpu = torch.randint(
        0, pool_pages, (batch_size, width), dtype=torch.int32
    )
    offset = 0
    for i in range(batch_size):
        n = int(pages_per_seq[i])
        block_tables_cpu[i, :n] = perm[offset : offset + n]
        offset += n
    kv_page_indices_cpu = torch.cat(
        [block_tables_cpu[i, : int(pages_per_seq[i])] for i in range(batch_size)]
    ).to(torch.int32)

    total_q = int(qo_indptr_cpu[-1])
    q = torch.randn(total_q, num_qo_heads, head_dim_qk, dtype=q_dtype, device=device)
    if args.kv_layout == "HND":
        k_shape = (pool_pages, num_kv_heads, page_size, head_dim_qk)
        v_shape = (pool_pages, num_kv_heads, page_size, head_dim_vo)
    else:
        k_shape = (pool_pages, page_size, num_kv_heads, head_dim_qk)
        v_shape = (pool_pages, page_size, num_kv_heads, head_dim_vo)
    k_cache = torch.randn(*k_shape, dtype=kv_dtype, device=device)
    v_cache = torch.randn(*v_shape, dtype=kv_dtype, device=device)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        q_lens_cpu=q_lens,
        kv_seq_lens_cpu=kv_lens,
        qo_indptr_cpu=qo_indptr_cpu,
        qo_indptr=qo_indptr_cpu.to(device),
        kv_seq_lens=kv_lens.to(device),
        block_tables_cpu=block_tables_cpu,
        block_tables=block_tables_cpu.to(device),
        kv_page_indices_cpu=kv_page_indices_cpu,
        kv_page_indices=kv_page_indices_cpu.to(device),
        max_q_len=int(q_lens.max()),
        max_kv_len=int(kv_lens.max()),
        total_q=total_q,
        batch_size=batch_size,
        page_size=page_size,
        table_width=width,
        pool_pages=pool_pages,
    )


def _make_metadata(case, kv_input_form):
    """Fresh per-step metadata with host mirrors (zero-sync construction)."""
    common = dict(
        page_size=case["page_size"],
        max_q_len=case["max_q_len"],
        max_kv_len=case["max_kv_len"],
        qo_indptr_cpu=case["qo_indptr_cpu"],
        kv_seq_lens_cpu=case["kv_seq_lens_cpu"],
    )
    if kv_input_form == "dense":
        return PagedAttentionMetadata.dense(
            case["qo_indptr"], case["kv_seq_lens"], case["block_tables"], **common
        )
    return PagedAttentionMetadata.csr(
        case["qo_indptr"], case["kv_seq_lens"], case["kv_page_indices"], **common
    )


def _error_status(exc):
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"error: {type(exc).__name__}: {first_line[:240]}"


def _check_against_oracle(out, lse, ref_out, ref_lse, lse_mode):
    """None when the candidate matches the oracle, else a description."""
    problems = []
    if out is None:
        return "no output returned"
    out_f = out.float()
    if not torch.isfinite(out_f).all():
        problems.append("output has non-finite values")
    num_diff, num_total, pct = is_close_stats(out_f, ref_out, **OUT_TOL)
    if num_diff > 0:
        max_abs_err = (out_f - ref_out).abs().max().item()
        problems.append(
            f"out {num_diff}/{num_total} ({pct:.2f}%) elements differ, "
            f"max_abs_err={max_abs_err:.4g}"
        )
    if lse_mode != "none":
        if lse is None:
            problems.append("lse requested but not returned")
        else:
            lse_f = lse.float()
            num_diff, num_total, pct = is_close_stats(lse_f, ref_lse, **LSE_TOL)
            if num_diff > 0:
                max_abs_err = (lse_f - ref_lse).abs().max().item()
                problems.append(
                    f"lse {num_diff}/{num_total} ({pct:.2f}%) elements differ, "
                    f"max_abs_err={max_abs_err:.4g}"
                )
    elif lse is not None:
        problems.append("lse returned although lse_mode is 'none'")
    return "; ".join(problems) or None


def _wall_ms(fn, dry_run_iters, num_iters):
    """Synchronized host wall time per call, in milliseconds."""
    for _ in range(dry_run_iters):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(num_iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


class _PhaseRows:
    """The CSV rows of one candidate: plan, run and one step row per N."""

    def __init__(self, args, case, backend, api_variant, timing_metric, head_dim_vo):
        self.label = f"{backend}/{api_variant}"
        self.rows = []
        for phase, layers in [("plan", ""), ("run", "")] + [
            ("step", n) for n in args.pa_layers
        ]:
            row = defaultdict(str)
            row["routine"] = args.routine
            row["backend"] = backend
            row["resolved_backend"] = backend
            row["api_variant"] = api_variant
            row["phase"] = phase
            row["layers"] = layers
            row["status"] = ""
            row["timing_metric"] = timing_metric if phase == "run" else "host_wall"
            row["page_size"] = args.page_size
            row["batch_size"] = args.batch_size
            row["s_qo"] = args.s_qo
            row["s_kv"] = args.s_kv
            row["num_qo_heads"] = args.num_qo_heads
            row["num_kv_heads"] = args.num_kv_heads
            row["head_dim_qk"] = args.head_dim_qk
            row["head_dim_vo"] = head_dim_vo
            row["causal"] = args.causal
            row["q_dtype"] = args.q_dtype
            row["kv_dtype"] = args.kv_dtype
            row["out_dtype"] = args.q_dtype
            row["kv_layout"] = args.kv_layout
            row["kv_input_form"] = args.kv_input_form
            row["lse_mode"] = args.lse_mode
            row["window_left"] = args.window_left
            row["avg_actual_seq_len"] = (
                int(case["q_lens_cpu"].sum()) // case["batch_size"]
            )
            row["random_actual_seq_len"] = args.random_actual_seq_len
            row["case_tag"] = args.case_tag
            self.rows.append(row)

    def set_resolved(self, resolved_backend):
        for row in self.rows:
            row["resolved_backend"] = resolved_backend
        if self.rows and self.rows[0]["backend"] != resolved_backend:
            self.label = (
                f"{self.rows[0]['backend']}({resolved_backend})/"
                f"{self.rows[0]['api_variant']}"
            )

    def set_all(self, status):
        for row in self.rows:
            row["status"] = status
        print(f"[INFO] {self.label}: {status}")


def _time_candidate(args, phases, case, plan_once, run_once, recheck, use_cuda_graph):
    """Fill the timing of every phase row; a failing phase keeps its own status.

    ``recheck`` (optional, zero-arg) re-validates the buffers the timed run
    wrote against the oracle.
    """
    q, k_cache, v_cache = case["q"], case["k_cache"], case["v_cache"]
    out, lse = case["out"], case["lse"]
    q_lens, kv_lens = case["q_lens_cpu"], case["kv_seq_lens_cpu"]
    q_dtype = q.dtype
    for row in phases.rows:
        phase, layers = row["phase"], row["layers"]
        try:
            if phase == "plan":
                samples = _wall_ms(plan_once, args.dry_run_iters, args.num_iters)
            elif phase == "run":
                samples = bench_gpu_time(
                    fn=run_once,
                    dry_run_iters=args.dry_run_iters,
                    repeat_iters=args.num_iters,
                    sleep_after_run=False,
                    enable_cupti=args.use_cupti,
                    use_cuda_graph=use_cuda_graph,
                    cold_l2_cache=True,
                    input_args=(q, k_cache, v_cache, out, lse),
                )
                if recheck is not None:
                    # The timed calls wrote into the original buffers (the first
                    # rotation slot); a replay/eager result that drifted from
                    # the oracle marks the row instead of publishing its time.
                    torch.cuda.synchronize()
                    problem = recheck()
                    if problem is not None:
                        row["status"] = f"incorrect (timed run): {problem}"
                        print(f"[ERROR] {phases.label} {phase}: {row['status']}")
                        continue
            else:

                def step():
                    plan_once()
                    for _ in range(layers):
                        run_once(q, k_cache, v_cache, out, lse)

                samples = _wall_ms(step, args.dry_run_iters, args.num_iters)
        except Exception as exc:  # noqa: BLE001 - every failure becomes a row
            row["status"] = _error_status(exc)
            print(f"[ERROR] {phases.label} {phase}: {row['status']}")
            if args.verbose >= 2:
                traceback.print_exc()
            continue

        median_time = float(np.median(samples))
        std_time = float(np.std(samples))
        row["median_time"] = median_time
        row["std_time"] = std_time
        if not row["status"]:
            row["status"] = "ok"
        if phase == "run":
            tflops = attention_tflops_per_sec_with_actual_seq_lens(
                q_lens,
                kv_lens,
                args.head_dim_qk,
                row["head_dim_vo"],
                args.num_qo_heads,
                args.causal,
                median_time,
            )
            tb_per_sec = attention_tb_per_sec_with_actual_seq_lens(
                q_lens,
                kv_lens,
                args.head_dim_qk,
                row["head_dim_vo"],
                args.num_qo_heads,
                args.num_kv_heads,
                median_time,
                q_dtype=q_dtype,
                kv_dtype=k_cache.dtype,
                o_dtype=q_dtype,
            )
            row["tflops"] = tflops
            row["tb_per_sec"] = tb_per_sec
            print_perf_metrics(
                f"{phases.label} run", median_time, std_time, tflops, tb_per_sec
            )
        else:
            name = f"{phases.label} {phase}" + (f"[{layers}]" if layers != "" else "")
            print(
                f"[PERF] {name.ljust(15)}:: median time {median_time:.3f} ms; "
                f"std {std_time:.3f} ms (synchronized host wall)"
            )


def _bench_unified(args, case, backend, ctx):
    """Rows for one requested backend through the unified facade."""
    phases = _PhaseRows(
        args, case, backend, "unified", ctx["timing_metric"], ctx["head_dim_vo"]
    )
    if backend not in PAGED_ATTENTION_BACKENDS:
        phases.set_all(
            f"unsupported: {backend!r} is not a PagedAttention backend "
            f"(choose from {', '.join(PAGED_ATTENTION_BACKENDS)})"
        )
        return phases, None
    if ctx["reference_error"] is not None:
        phases.set_all(ctx["reference_error"])
        return phases, None

    try:
        resolution = resolve_paged_attention(
            device=ctx["device"],
            page_size=args.page_size,
            causal=args.causal,
            window_left=args.window_left,
            need_lse=args.lse_mode != "none",
            kv_input_form=ctx["kv_input_form_api"],
            backend=backend,
            **ctx["static_kwargs"],
        )
    except ValueError as exc:
        # On this base resolve() raises ValueError both for capability
        # rejections and for invalid arguments; the message carries the
        # per-backend reason.
        reason = str(exc).splitlines()[0]
        prefix = "no runnable backend for this configuration ("
        if reason.startswith(prefix) and reason.endswith(")"):
            reason = reason[len(prefix) : -1]
        phases.set_all(f"unsupported: {reason[:240]}")
        return phases, None
    resolved = resolution.chosen
    phases.set_resolved(resolved)
    if args.verbose >= 1 and backend == "auto":
        print(f"[INFO] auto resolution:\n{resolution.explain()}")

    attn = PagedAttention(
        ctx["device"],
        use_cuda_graph=ctx["use_cuda_graph"],
        workspace_buffer=ctx["workspace"],
    )
    plan_kwargs = dict(ctx["static_kwargs"])
    plan_kwargs.update(
        causal=args.causal, window_left=args.window_left, lse_mode=args.lse_mode
    )
    scale = ctx["scale"]

    def plan_once():
        attn.plan(
            _make_metadata(case, args.kv_input_form), backend=resolution, **plan_kwargs
        )

    def run_once(q_, k_, v_, out_, lse_):
        return attn.run(q_, (k_, v_), out=out_, lse=lse_, sm_scale=scale)

    try:
        ctx["workspace"].zero_()
        plan_once()
        got_out, got_lse = run_once(
            case["q"], case["k_cache"], case["v_cache"], case["out"], case["lse"]
        )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a row
        phases.set_all(_error_status(exc))
        if args.verbose >= 2:
            traceback.print_exc()
        return phases, None

    recheck = None
    if ctx["reference"] is not None:
        ref_out, ref_lse = ctx["reference"]

        def recheck():
            return _check_against_oracle(
                case["out"], case["lse"], ref_out, ref_lse, args.lse_mode
            )

        problem = _check_against_oracle(
            got_out, got_lse, ref_out, ref_lse, args.lse_mode
        )
        if problem is not None:
            phases.set_all(f"incorrect: {problem}")
            print(f"[ERROR] {phases.label}: output mismatch against the fp32 oracle")
            if not args.allow_output_mismatch:
                return phases, resolved
        elif args.verbose >= 1:
            print(f"[INFO] {phases.label}: matches the fp32 oracle")

    _time_candidate(
        args, phases, case, plan_once, run_once, recheck, ctx["use_cuda_graph"]
    )
    return phases, resolved


# --------------------------- legacy same-kernel rows ---------------------------


class _LegacyFaProvider:
    """fa2/fa3 through ``BatchPrefillWithPagedKVCacheWrapper``.

    Receives the canonical device tensors an engine holds and derives the
    wrapper's page-unit CSR dialect the way the engines do: on the device for
    the CSR form (sglang; the wrapper's plan then copies indptr and last-page
    lengths to the host itself), on the host for the dense form (vLLM; the
    wrapper uploads the host arrays).  ``sm_scale`` is plan-time in this API.
    """

    def __init__(self, name, args, case, ctx):
        self.name = name
        self.args, self.case, self.ctx = args, case, ctx
        device = ctx["device"]
        batch_size = case["batch_size"]
        if ctx["use_cuda_graph"]:
            # the wrapper's own reserved-buffer protocol (what an engine sets
            # up per graph bucket)
            i32 = dict(dtype=torch.int32, device=device)
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
                ctx["workspace"],
                args.kv_layout,
                use_cuda_graph=True,
                qo_indptr_buf=torch.zeros(batch_size + 1, **i32),
                paged_kv_indptr_buf=torch.zeros(batch_size + 1, **i32),
                paged_kv_indices_buf=torch.zeros(
                    batch_size * case["table_width"], **i32
                ),
                paged_kv_last_page_len_buf=torch.zeros(batch_size, **i32),
                backend=name,
            )
        else:
            self.wrapper = BatchPrefillWithPagedKVCacheWrapper(
                ctx["workspace"], args.kv_layout, backend=name
            )
        self.last_lse = None

    def plan(self):
        args, case = self.args, self.case
        page_size = case["page_size"]
        if args.kv_input_form == "csr":
            kv_seq_lens = case["kv_seq_lens"]
            pages = (kv_seq_lens + page_size - 1) // page_size
            zero = torch.zeros(1, dtype=torch.int32, device=kv_seq_lens.device)
            kv_indptr = torch.cat([zero, torch.cumsum(pages, 0, dtype=torch.int32)])
            last_page_len = (kv_seq_lens - 1) % page_size + 1
            qo_indptr, kv_page_indices = case["qo_indptr"], case["kv_page_indices"]
        else:
            kv_lens_cpu = case["kv_seq_lens_cpu"]
            pages = (kv_lens_cpu + page_size - 1) // page_size
            kv_indptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    torch.cumsum(pages, 0, dtype=torch.int32),
                ]
            )
            last_page_len = (kv_lens_cpu - 1) % page_size + 1
            live = torch.arange(case["table_width"]).unsqueeze(0) < pages.unsqueeze(1)
            kv_page_indices = case["block_tables_cpu"][live]
            qo_indptr = case["qo_indptr_cpu"]
        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_page_indices,
            last_page_len,
            args.num_qo_heads,
            args.num_kv_heads,
            args.head_dim_qk,
            page_size,
            head_dim_vo=self.ctx["head_dim_vo"],
            causal=args.causal,
            sm_scale=self.ctx["scale"],
            window_left=args.window_left,
            q_data_type=case["q"].dtype,
            kv_data_type=case["k_cache"].dtype,
        )

    def run(self, q, k_cache, v_cache, out, lse):
        need_lse = self.args.lse_mode != "none"
        result = self.wrapper.run(
            q, (k_cache, v_cache), out=out, lse=lse, return_lse=need_lse
        )
        if need_lse:
            out, self.last_lse = result
            return out, self.last_lse
        return result, None


class _LegacyCudnnProvider:
    """cuDNN through ``cudnn_batch_prefill_with_kv_cache`` (dense table).

    The API takes per-request lengths as ``(b, 1, 1, 1)`` device tensors and
    token-unit batch offsets; the LSE is padded ``(b, max_q, h)`` in the
    requested base.
    """

    name = "cudnn"

    def __init__(self, args, case, ctx):
        self.args, self.case, self.ctx = args, case, ctx
        self.workspace = ctx["workspace"].view(torch.int8)
        self.native_lse = (
            torch.empty(
                case["batch_size"],
                case["max_q_len"],
                args.num_qo_heads,
                dtype=torch.float32,
                device=ctx["device"],
            )
            if args.lse_mode != "none"
            else None
        )
        self.q_lens4 = self.kv_lens4 = None
        self.last_lse = None

    def plan(self):
        batch_size = self.case["batch_size"]
        self.q_lens4 = self.case["qo_indptr"].diff().view(batch_size, 1, 1, 1)
        self.kv_lens4 = self.case["kv_seq_lens"].view(batch_size, 1, 1, 1)

    def run(self, q, k_cache, v_cache, out, lse):
        args, case = self.args, self.case
        if args.kv_layout == "NHD":
            k_cache = k_cache.permute(0, 2, 1, 3)
            v_cache = v_cache.permute(0, 2, 1, 3)
        out, self.last_lse = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            self.ctx["scale"],
            self.workspace,
            max_token_per_sequence=case["max_q_len"],
            max_sequence_kv=case["max_kv_len"],
            actual_seq_lens_q=self.q_lens4,
            actual_seq_lens_kv=self.kv_lens4,
            block_tables=case["block_tables"],
            causal=args.causal,
            return_lse=args.lse_mode != "none",
            lse_base="e" if args.lse_mode == "basee" else "2",
            batch_offsets_q=case["qo_indptr"],
            batch_offsets_units="tokens",
            out=out,
            lse=self.native_lse,
        )
        return out, self.last_lse


class _LegacyTrtllmProvider:
    """trtllm-gen through ``trtllm_batch_context_with_kv_cache`` (dense table).

    The unified metadata is this API's native dialect; the only per-step
    derivation is the cumulative KV length vector.
    """

    name = "trtllm-gen"

    def __init__(self, args, case, ctx):
        self.args, self.case, self.ctx = args, case, ctx
        self.cum_kv_seq_lens = None
        self.last_lse = None

    def plan(self):
        kv_seq_lens = self.case["kv_seq_lens"]
        zero = torch.zeros(1, dtype=torch.int32, device=kv_seq_lens.device)
        self.cum_kv_seq_lens = torch.cat(
            [zero, torch.cumsum(kv_seq_lens, 0, dtype=torch.int32)]
        )

    def run(self, q, k_cache, v_cache, out, lse):
        args, case = self.args, self.case
        need_lse = args.lse_mode != "none"
        result = trtllm_batch_context_with_kv_cache(
            q,
            (k_cache, v_cache),
            self.ctx["workspace"],
            case["block_tables"],
            case["kv_seq_lens"],
            case["max_q_len"],
            case["max_kv_len"],
            self.ctx["scale"],
            1.0,
            case["batch_size"],
            case["qo_indptr"],
            self.cum_kv_seq_lens,
            window_left=args.window_left,
            out=out,
            kv_layout=args.kv_layout,
            causal=args.causal,
            lse=lse,
            return_lse=need_lse,
        )
        if need_lse:
            out, self.last_lse = result
            return out, self.last_lse
        return result, None


def _legacy_lse_to_contract(provider, lse_native, case, lse_mode):
    """Native legacy LSE -> packed (total_q, heads) in the requested base."""
    if lse_native is None or lse_mode == "none":
        return lse_native
    if provider.name == "cudnn":
        q_lens = case["q_lens_cpu"]
        return torch.cat(
            [lse_native[i, : int(q_lens[i])] for i in range(case["batch_size"])]
        )
    # fa2/fa3/trtllm-gen emit base-2 natively
    return lse_native * LN2 if lse_mode == "basee" else lse_native


def _bench_legacy(args, case, backend, ctx):
    """Rows for the same inputs through the legacy public API of ``backend``."""
    phases = _PhaseRows(
        args, case, backend, "legacy", ctx["timing_metric"], ctx["head_dim_vo"]
    )
    if backend in ("cudnn", "trtllm-gen") and args.kv_input_form != "dense":
        phases.set_all(
            f"unsupported: the legacy {backend} API takes a dense block table; "
            "use --kv_input_form dense for this comparison"
        )
        return phases
    if ctx["reference_error"] is not None:
        phases.set_all(ctx["reference_error"])
        return phases

    def run_once(q_, k_, v_, out_, lse_):
        return provider.run(q_, k_, v_, out_, lse_)

    try:
        if backend in ("fa2", "fa3"):
            provider = _LegacyFaProvider(backend, args, case, ctx)
        elif backend == "cudnn":
            provider = _LegacyCudnnProvider(args, case, ctx)
        else:
            provider = _LegacyTrtllmProvider(args, case, ctx)
        ctx["workspace"].zero_()
        provider.plan()
        got_out, got_lse = run_once(
            case["q"], case["k_cache"], case["v_cache"], case["out"], case["lse"]
        )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - every failure becomes a row
        phases.set_all(_error_status(exc))
        if args.verbose >= 2:
            traceback.print_exc()
        return phases

    recheck = None
    if ctx["reference"] is not None:
        ref_out, ref_lse = ctx["reference"]

        def check(out_, lse_native):
            return _check_against_oracle(
                out_,
                _legacy_lse_to_contract(provider, lse_native, case, args.lse_mode),
                ref_out,
                ref_lse,
                args.lse_mode,
            )

        def recheck():
            return check(case["out"], provider.last_lse)

        problem = check(got_out, got_lse)
        if problem is not None:
            phases.set_all(f"incorrect: {problem}")
            print(f"[ERROR] {phases.label}: output mismatch against the fp32 oracle")
            if not args.allow_output_mismatch:
                return phases
        elif args.verbose >= 1:
            print(f"[INFO] {phases.label}: matches the fp32 oracle")

    _time_candidate(
        args, phases, case, provider.plan, run_once, recheck, ctx["use_cuda_graph"]
    )
    return phases


def testPagedAttention(args):
    """
    Benchmark ``flashinfer.prefill.PagedAttention`` (experimental unified paged
    attention) for the requested backends on one shared set of inputs.

    This test:
    1. Builds packed Q, a paged K/V pool with a shuffled page mapping, exact
       per-request lengths and the dense or CSR ``PagedAttentionMetadata``.
    2. Resolves and plans each requested backend (``fa2 fa3 cudnn trtllm-gen
       auto``); unsupported ones are recorded, not dropped.
    3. With ``--refcheck``, compares output (and requested LSE) against the
       fp32 oracle before any timing.
    4. Times the ``plan``, ``run`` and ``step`` phases (see module docstring).

    Args:
        args: Parsed command line arguments containing test configuration

    Returns:
        list: One result dictionary per (backend, phase) row
    """
    if args.verbose >= 1:
        print("[INFO] Running testPagedAttention")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )
    res = []

    q_dtype = dtype_str_to_torch_dtype(args.q_dtype)
    kv_dtype = dtype_str_to_torch_dtype(args.kv_dtype)
    if q_dtype not in (torch.float16, torch.bfloat16) or kv_dtype != q_dtype:
        print(
            "[ERROR] PagedAttention benchmark generates fp16/bf16 Q and KV of one "
            f"dtype; got q_dtype={args.q_dtype}, kv_dtype={args.kv_dtype}. Exiting."
        )
        return res
    if (
        args.out_dtype is not None
        and dtype_str_to_torch_dtype(args.out_dtype) != q_dtype
    ):
        print("[WARNING] PagedAttention writes out in q_dtype; ignoring --out_dtype.")
    if args.head_dim_qk is None:
        print("[ERROR] --head_dim_qk is required for PagedAttention. Exiting.")
        return res
    head_dim_vo = args.head_dim_vo if args.head_dim_vo is not None else args.head_dim_qk
    if args.page_size <= 0:
        print("[ERROR] --page_size is required for PagedAttention. Exiting.")
        return res
    if args.s_qo > args.s_kv:
        print("[ERROR] s_qo > s_kv is not supported. Exiting.")
        return res
    if getattr(args, "autotune", False):
        print(
            "[ERROR] PagedAttention has no autotuner; 'auto' is a static "
            "selection. Drop --autotune. Exiting."
        )
        return res
    if args.enable_pdl:
        print("[WARNING] PagedAttention does not expose PDL; ignoring --enable_pdl.")

    case = _build_case(args, device, q_dtype, kv_dtype, head_dim_vo)
    case["out"] = torch.empty(
        case["total_q"], args.num_qo_heads, head_dim_vo, dtype=q_dtype, device=device
    )
    case["lse"] = (
        torch.empty(
            case["total_q"], args.num_qo_heads, dtype=torch.float32, device=device
        )
        if args.lse_mode != "none"
        else None
    )
    if args.verbose >= 1:
        print(
            f"[VERBOSE] total_q={case['total_q']} max_q_len={case['max_q_len']} "
            f"max_kv_len={case['max_kv_len']} pool_pages={case['pool_pages']} "
            f"table_width={case['table_width']}"
        )
    if args.verbose >= 2:
        print(f"[VVERBOSE] q_lens={case['q_lens_cpu'].tolist()}")
        print(f"[VVERBOSE] kv_lens={case['kv_seq_lens_cpu'].tolist()}")

    use_cuda_graph = not args.no_cuda_graph
    if args.use_cupti and _cupti_available():
        timing_metric = "cupti"
    elif use_cuda_graph:
        timing_metric = "cuda_graph_events"
    else:
        timing_metric = "cuda_events"

    ctx = dict(
        device=device,
        head_dim_vo=head_dim_vo,
        use_cuda_graph=use_cuda_graph,
        timing_metric=timing_metric,
        # one caller-owned scratch workspace, shared by every candidate
        workspace=torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=device),
        scale=float(1.0 / math.sqrt(args.head_dim_qk)),
        kv_input_form_api="block_tables"
        if args.kv_input_form == "dense"
        else "page_indices",
        static_kwargs=dict(
            num_qo_heads=args.num_qo_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim_qk=args.head_dim_qk,
            head_dim_vo=head_dim_vo,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            kv_layout=args.kv_layout,
        ),
        reference=None,
        reference_error=None,
    )
    if args.refcheck:
        try:
            reference_paged_prefill = _load_reference_oracle()
            with torch.no_grad():
                ref_out, ref_lse = reference_paged_prefill(
                    case["q"],
                    case["k_cache"],
                    case["v_cache"],
                    case["qo_indptr_cpu"],
                    case["kv_seq_lens_cpu"],
                    case["block_tables"] if args.kv_input_form == "dense" else None,
                    args.page_size,
                    args.causal,
                    sm_scale=ctx["scale"],
                    window_left=args.window_left,
                    kv_layout=args.kv_layout,
                    kv_page_indices=case["kv_page_indices"]
                    if args.kv_input_form == "csr"
                    else None,
                    lse_base="e" if args.lse_mode == "basee" else "2",
                )
            torch.cuda.synchronize()
            ctx["reference"] = (ref_out, ref_lse)
        except Exception as exc:  # noqa: BLE001 - recorded on every row
            ctx["reference_error"] = "reference failed: " + _error_status(exc)
            print(f"[ERROR] fp32 oracle failed; no candidate is timed: {exc}")

    legacy_targets = []
    for backend in args.backends:
        phases, resolved = _bench_unified(args, case, backend, ctx)
        res.extend(phases.rows)
        if resolved is not None and resolved not in legacy_targets:
            legacy_targets.append(resolved)
    if args.pa_legacy:
        # one legacy row set per concrete kernel that ran through the facade
        # (`auto` contributes the backend it resolved to; duplicates collapse)
        if not legacy_targets:
            print("[INFO] --pa_legacy: no unified candidate planned; no legacy rows")
        for backend in legacy_targets:
            res.extend(_bench_legacy(args, case, backend, ctx).rows)
    return res
