"""Regressions for ``benchmarks/bench_paged_attention.py`` (the standalone
PagedAttention benchmark), driven in-process through the script's parser.

Each test pins a review finding on the routine itself and fails on the
routine as it was before the fix:

- R5 / R03 / C07: the ``--pa_legacy`` cuDNN provider handed the native API the
  case's over-wide dense page table (CUDNN_STATUS_BAD_PARAM on every dense
  case); it must pass the width-exact view and match the oracle.
- R6 / R04: under the default CUDA-graph timing the helper's cold-L2 rotating
  clones were captured without an eager run, which cake refuses (M18); the
  run rows must time on every backend, cake included.
- The phase rows are ``plan`` / ``run`` / ``step`` in both modes; under CUDA
  graphs the ``plan`` row is the graph-mode re-plan.
- Every row keeps its truthful ``status`` and the legacy rows carry the same
  ``static_backend`` / ``resolved_backend`` columns as the unified rows.

The routine gates every candidate on the fp32 oracle, so ``status == "ok"``
on a row also means the timed kernel matched the reference.
"""

import pathlib
import shlex
import sys

import pytest
import torch

BENCHMARKS_DIR = pathlib.Path(__file__).resolve().parents[3] / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

bench_paged_attention = pytest.importorskip("bench_paged_attention")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the benchmark routine needs a GPU"
)

COMMON = (
    "--num_qo_heads 8 --num_kv_heads 2 --head_dim_qk 128 "
    "--head_dim_vo 128 --page_size 16 --causal --kv_layout HND --refcheck "
    "--pa_layers 1 --num_iters 2 --dry_run_iters 1"
)


def _rows(line):
    args = bench_paged_attention.parse_args(shlex.split(line))
    return bench_paged_attention.run_paged_attention(args)


def _select(rows, api_variant, phase=None):
    return [
        r
        for r in rows
        if r["api_variant"] == api_variant and (phase is None or r["phase"] == phase)
    ]


def _skip_unless_runnable(rows, backend):
    """Skip when the device has no such kernel (an ``unsupported`` unified row)."""
    plan = [r for r in _select(rows, "unified", "plan") if r["backend"] == backend]
    assert plan, f"no unified plan row for {backend}"
    if str(plan[0]["status"]).startswith("unsupported"):
        pytest.skip(f"{backend} is not runnable here: {plan[0]['status']}")


def _assert_all_ok(rows, label):
    bad = [
        (r["backend"], r["api_variant"], r["phase"], r["status"], r["refcheck_passed"])
        for r in rows
        if r["status"] != "ok" or r["refcheck_passed"] is not True
    ]
    assert not bad, f"{label}: rows not ok / not oracle-checked: {bad}"


def test_legacy_cudnn_dense_rows_match_the_oracle():
    """R5: the width-exact page-table view makes every legacy cuDNN row ok."""
    rows = _rows(
        f"{COMMON} --backends cudnn --batch_size 2 --s_qo 4 --s_kv 64 "
        "--lse_mode none --pa_legacy --no_cuda_graph"
    )
    _skip_unless_runnable(rows, "cudnn")
    legacy = _select(rows, "legacy")
    assert {r["phase"] for r in legacy} == {"plan", "run", "step"}
    _assert_all_ok(rows, "cudnn dense, eager")


def test_legacy_cudnn_dense_rows_under_cuda_graphs_with_lse():
    """R5, graph mode with an LSE: the strided view is a legal cuDNN graph."""
    rows = _rows(
        f"{COMMON} --backends cudnn --batch_size 2 --s_qo 4 --s_kv 64 "
        "--lse_mode base2 --pa_legacy"
    )
    _skip_unless_runnable(rows, "cudnn")
    assert _select(rows, "legacy", "run")
    _assert_all_ok(rows, "cudnn dense, graph, base2")


def test_cake_run_rows_time_under_cuda_graphs():
    """R6: the run phase captures fixed bindings, so cake's M18 rule holds
    for the unified and the legacy row on an input far below L2."""
    rows = _rows(
        f"{COMMON} --backends cake --batch_size 2 --s_qo 4 --s_kv 32 "
        "--lse_mode none --pa_legacy"
    )
    _skip_unless_runnable(rows, "cake")
    for variant in ("unified", "legacy"):
        (run,) = _select(rows, variant, "run")
        assert run["status"] == "ok", (variant, run["status"])
        assert run["timing_metric"] in ("cuda_graph_events", "cupti")
        assert run["cold_l2_cache"] is True
    _assert_all_ok(rows, "cake, graph")


def test_phase_rows_are_the_same_in_both_modes():
    """plan / run / step, with or without CUDA graphs: the graph-mode
    re-plan is the ``plan`` row of a graph-mode instance."""
    for mode, flag in (("graph", ""), ("eager", " --no_cuda_graph")):
        rows = _rows(
            f"{COMMON} --backends fa2 --batch_size 2 --s_qo 4 --s_kv 64 "
            f"--lse_mode basee --pa_legacy{flag}"
        )
        _skip_unless_runnable(rows, "fa2")
        for variant in ("unified", "legacy"):
            phases = [r["phase"] for r in _select(rows, variant)]
            assert phases == ["plan", "run", "step"], phases
            (row,) = _select(rows, variant, "plan")
            assert row["timing_metric"] == "host_wall"
        _assert_all_ok(rows, f"fa2 {mode} mode")


def test_rows_for_the_csr_form():
    """The flat form plans, re-plans and runs under CUDA graphs; the legacy
    cuDNN rows say truthfully why they cannot."""
    rows = _rows(
        "--backends fa2 cudnn --batch_size 3 --s_qo 5 "
        "--s_kv 100 --num_qo_heads 8 --num_kv_heads 2 --head_dim_qk 128 "
        "--page_size 16 --causal --kv_layout NHD --kv_input_form csr "
        "--lse_mode basee --refcheck --random_actual_seq_len --pa_legacy "
        "--pa_layers 1 --num_iters 2 --dry_run_iters 1"
    )
    _skip_unless_runnable(rows, "fa2")
    fa2 = [r for r in rows if r["backend"] == "fa2"]
    assert {r["phase"] for r in _select(fa2, "unified")} == {"plan", "run", "step"}
    _assert_all_ok(fa2, "fa2 csr")
    cudnn_legacy = [r for r in _select(rows, "legacy") if r["backend"] == "cudnn"]
    assert cudnn_legacy
    for r in cudnn_legacy:
        assert str(r["status"]).startswith("unsupported: the legacy cudnn API"), r[
            "status"
        ]


def test_legacy_rows_carry_the_backend_columns():
    """Legacy rows name the kernel in backend, static_backend and
    resolved_backend alike; unified auto rows name the resolved kernel."""
    rows = _rows(
        f"{COMMON} --backends auto --batch_size 2 --s_qo 4 --s_kv 64 "
        "--lse_mode none --pa_legacy"
    )
    _skip_unless_runnable(rows, "auto")
    unified = _select(rows, "unified")
    resolved = {r["resolved_backend"] for r in unified}
    assert len(resolved) == 1 and "auto" not in resolved, resolved
    assert all(r["backend"] == "auto" for r in unified)
    legacy = _select(rows, "legacy")
    assert legacy, "auto resolved to a kernel, so a legacy row set is due"
    (kernel,) = resolved
    for r in legacy:
        assert (r["backend"], r["static_backend"], r["resolved_backend"]) == (
            kernel,
            kernel,
            kernel,
        )
    _assert_all_ok(rows, "auto + legacy")
