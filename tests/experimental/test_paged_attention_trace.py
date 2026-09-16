"""fi_trace for the experimental ``PagedAttention.run`` (unified paged attention).

The stable trace lane (``tests/trace/``) filters experimental registrations,
so this file exercises the template directly and through the bound entry:

- ``flashinfer.fi_trace(attn.run, ...)`` on planned instances over the
  dense/CSR x HND/NHD x none/base2/basee x causal/window matrix, on fa2 and
  on cuDNN / trtllm-gen where they resolve — the definition must not change
  with the backend;
- the exported ``reference`` / ``init`` sources execute in a fresh namespace,
  the reference matches ``paged_attention_reference.py`` and the rebuilt
  instance re-traces to the same definition;
- auto-dump through the bound ``self`` writes a complete JSON (no unknown
  dtypes, no missing Const values, public fi_api tag);
- identity encodes the plan (form, layout, causal, window, LSE mode);
- tracing before plan / without an instance raises; a failed re-plan still
  traces the last successful plan; in graph mode the traced metadata is the
  reserved storage; tracing is sync-free and read-only.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import flashinfer.trace.template as trace_template_mod
from flashinfer.fi_trace import fi_trace
from flashinfer.prefill import PagedAttention, resolve_paged_attention
from flashinfer.trace.template import (
    Const,
    Scalar,
    Tensor,
    _render_init_source,
    _render_reference_source,
)
from flashinfer.trace.templates.paged_attention import (
    _paged_attention_init,
    _paged_attention_reference,
    _paged_attention_template,
    paged_attention_trace_dispatch,
)

from .paged_attention_reference import reference_paged_prefill
from .test_paged_attention_prototype import (
    LSE_TOL,
    OUT_TOL,
    make_metadata,
    make_problem,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FI_API_TAG = "fi_api:flashinfer.prefill.PagedAttention.run"
LSE_MODES = ("none", "base2", "basee")
LAYOUTS = ("HND", "NHD")
FORMS = {"dense": "block_tables", "csr": "page_indices"}
MASKS = {"causal": (True, -1), "window": (True, 32)}
_SHAPE = dict(
    batch_size=4,
    max_q=16,
    max_kv=160,
    num_qo_heads=8,
    num_kv_heads=2,
    head_dim_qk=128,
    page_size=16,
    dtype=torch.bfloat16,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _problem(seed, *, form="dense", layout="HND", **overrides):
    return make_problem(
        seed, **dict(_SHAPE, **overrides), input_form=FORMS[form], kv_layout=layout
    )


def _runnable(p, backend, *, causal, window_left, lse_mode):
    try:
        resolve_paged_attention(
            device=torch.device(p["device"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim_qk=p["head_dim_qk"],
            head_dim_vo=p["head_dim_vo"],
            q_dtype=p["dtype"],
            kv_dtype=p.get("kv_dtype"),
            page_size=p["page_size"],
            kv_layout=p["kv_layout"],
            causal=causal,
            need_lse=lse_mode != "none",
            window_left=window_left,
            kv_input_form=p["input_form"],
            backend=backend,
        )
    except ValueError:
        return False
    return True


def _plan(p, backend, *, causal=True, window_left=-1, lse_mode="base2", graph=False):
    attn = PagedAttention(torch.device(p["device"]), use_cuda_graph=graph)
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_dtype=p.get("kv_dtype"),
        kv_layout=p["kv_layout"],
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
        backend=backend,
    )
    return attn


def _trace(attn, p, **extra):
    """Bound trace under the sync-debug guard: tracing must never sync."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        return fi_trace(
            attn.run, q=p["q"], kv_cache=(p["k_cache"], p["v_cache"]), **extra
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")


def _assert_complete(defn):
    json.dumps(defn)
    missing = [
        k for k, a in defn["axes"].items() if a["type"] == "const" and "value" not in a
    ]
    assert not missing, f"Const axes without values: {missing}"
    unknown = [
        k
        for section in ("inputs", "outputs")
        for k, v in defn[section].items()
        if v.get("dtype") == "unknown"
    ]
    assert not unknown, f"unknown dtypes: {unknown}"
    assert defn["op_type"] == "gqa_paged"
    assert defn["tags"][0] == FI_API_TAG
    assert defn["outputs"]["output"]["param"] == "out"
    assert "reference" in defn and "init" in defn


def _exec(source):
    namespace = {}
    exec(source, namespace)  # noqa: S102
    return namespace


def _oracle(p, *, causal, window_left, lse_mode):
    out, lse = reference_paged_prefill(
        p["q"],
        p["k_ref"],
        p["v_ref"],
        p["qo_indptr_cpu"],
        p["kv_seq_lens_cpu"],
        p["block_tables"] if p["input_form"] == "block_tables" else None,
        p["page_size"],
        causal,
        window_left=window_left,
        kv_layout=p["kv_layout"],
        kv_page_indices=p["kv_page_indices"],
        lse_base="e" if lse_mode == "basee" else "2",
    )
    return out, (None if lse_mode == "none" else lse)


# ── bound traces over the matrix; identity independent of the backend ────────


@cuda_only
@pytest.mark.parametrize("mask", list(MASKS))
@pytest.mark.parametrize("lse_mode", LSE_MODES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("form", list(FORMS))
def test_bound_trace_matrix(form, layout, lse_mode, mask):
    causal, window_left = MASKS[mask]
    p = _problem(11, form=form, layout=layout)
    definitions = {}
    for backend in ("fa2", "cudnn", "trtllm-gen"):
        if not _runnable(
            p, backend, causal=causal, window_left=window_left, lse_mode=lse_mode
        ):
            continue
        attn = _plan(
            p, backend, causal=causal, window_left=window_left, lse_mode=lse_mode
        )
        definitions[backend] = _trace(attn, p)
    assert "fa2" in definitions, "fa2 must be runnable for the matrix"
    defn = definitions["fa2"]
    _assert_complete(defn)

    prefix = f"paged_attention_{form}_"
    assert defn["name"].startswith(prefix)
    axes = {k: v.get("value") for k, v in defn["axes"].items()}
    assert axes["num_qo_heads"] == 8 and axes["num_kv_heads"] == 2
    assert axes["head_dim_qk"] == 128 and axes["head_dim_vo"] == 128
    assert axes["page_size"] == 16
    assert axes["kv_layout"] == LAYOUTS.index(layout)
    assert axes["causal"] == int(causal)
    assert axes["window_left"] == window_left
    assert axes["lse_mode"] == LSE_MODES.index(lse_mode)
    assert defn["name"].endswith(
        f"_layout{LAYOUTS.index(layout)}_causal{int(causal)}_wl{window_left}"
        f"_lse{LSE_MODES.index(lse_mode)}"
    )

    inputs = defn["inputs"]
    cache_shape = (
        ["num_pages", "num_kv_heads", "page_size"]
        if layout == "HND"
        else ["num_pages", "page_size", "num_kv_heads"]
    )
    assert inputs["k_cache"]["shape"] == cache_shape + ["head_dim_qk"]
    assert inputs["v_cache"]["shape"] == cache_shape + ["head_dim_vo"]
    assert inputs["k_cache"]["dtype"] == "bfloat16"
    assert inputs["qo_indptr"]["dtype"] == "int32" and inputs["qo_indptr"]["optional"]
    assert inputs["kv_seq_lens"]["dtype"] == "int32"
    if form == "dense":
        assert inputs["block_tables"]["shape"] == ["batch_size", "max_pages"]
        assert "kv_page_indices" not in inputs and "max_pages" in defn["axes"]
    else:
        assert inputs["kv_page_indices"]["shape"] == ["num_kv_indices"]
        assert "block_tables" not in inputs and "num_kv_indices" in defn["axes"]
    for name in ("sm_scale", "k_scale", "v_scale"):
        assert inputs[name] == {
            "shape": None,
            "dtype": "float32",
            "optional": True,
            "description": inputs[name]["description"],
        }
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert defn["outputs"]["output"]["shape"] == [
        "total_q",
        "num_qo_heads",
        "head_dim_vo",
    ]
    if lse_mode == "none":
        assert "lse" not in defn["outputs"]
    else:
        assert defn["outputs"]["lse"] == {
            "shape": ["total_q", "num_qo_heads"],
            "dtype": "float32",
            "param": "lse",
            "description": defn["outputs"]["lse"]["description"],
        }
    assert f"form:{form}" in defn["tags"] and f"layout:{layout}" in defn["tags"]
    assert f"lse:{lse_mode}" in defn["tags"]
    assert not any(t.startswith("backend:") for t in defn["tags"])

    # the mathematical identity does not depend on the resolved backend
    for backend, other in definitions.items():
        assert other == defn, f"definition differs on {backend}"


@cuda_only
def test_fp8_kv_cache_is_a_separate_schema():
    p = _problem(13, kv_dtype=torch.float8_e4m3fn)
    if not _runnable(p, "fa2", causal=True, window_left=-1, lse_mode="base2"):
        pytest.skip("fa2 fp8 KV not runnable here")
    attn = _plan(p, "fa2")
    defn = _trace(attn, p, k_scale=p["k_scale"], v_scale=p["v_scale"])
    _assert_complete(defn)
    assert defn["name"].startswith("paged_attention_dense_fp8kv_")
    assert defn["inputs"]["k_cache"]["dtype"] == "float8_e4m3fn"
    assert defn["inputs"]["q"]["dtype"] == "bfloat16"
    assert defn["outputs"]["output"]["dtype"] == "bfloat16"
    assert "kv:fp8" in defn["tags"]
    bf16 = _trace(_plan(_problem(13), "fa2"), _problem(13))
    assert bf16["name"] != defn["name"]


# ── reference == oracle, also from the exported source ───────────────────────


def _cpu_case(seed, *, head_dim_qk=128, head_dim_vo=128, page_size=8):
    g = torch.Generator().manual_seed(seed)
    q_lens = torch.tensor([5, 1, 7], dtype=torch.int32)
    kv_lens = torch.tensor([21, 9, 7], dtype=torch.int32)  # partial last pages
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), q_lens.cumsum(0).int()])
    pages = (kv_lens + page_size - 1) // page_size
    width, pool = int(pages.max()), int(pages.sum()) + 3
    perm = torch.randperm(pool, generator=g).int()
    block_tables = torch.zeros(len(kv_lens), width, dtype=torch.int32)
    off = 0
    for i, n in enumerate(pages.tolist()):
        block_tables[i, :n] = perm[off : off + n]
        off += n
    flat = torch.cat([block_tables[i, :n] for i, n in enumerate(pages.tolist())])
    q = torch.randn(int(qo_indptr[-1]), 8, head_dim_qk, generator=g)
    k = torch.randn(pool, 2, page_size, head_dim_qk, generator=g)
    v = torch.randn(pool, 2, page_size, head_dim_vo, generator=g)
    return dict(
        q=q,
        k_hnd=k,
        v_hnd=v,
        qo_indptr=qo_indptr,
        kv_seq_lens=kv_lens,
        block_tables=block_tables,
        kv_page_indices=flat,
        page_size=page_size,
    )


@pytest.mark.parametrize("dims", [(128, 128), (192, 128)], ids=["d128", "d192_128"])
@pytest.mark.parametrize(
    "causal,window_left",
    [(1, -1), (0, -1), (1, 4), (0, 4)],
    ids=["causal", "full", "causal_w4", "full_w4"],
)
@pytest.mark.parametrize("lse_mode", [0, 1, 2])
@pytest.mark.parametrize("kv_layout", [0, 1])
@pytest.mark.parametrize("csr", [0, 1])
def test_reference_matches_oracle_cpu(
    csr, kv_layout, lse_mode, causal, window_left, dims
):
    """The template reference and its exported source agree with the oracle
    (bottom-right causal, window, GQA, Dqk != Dvo, partial pages, LSE base)."""
    c = _cpu_case(17, head_dim_qk=dims[0], head_dim_vo=dims[1])
    if kv_layout:
        k = c["k_hnd"].permute(0, 2, 1, 3).contiguous()
        v = c["v_hnd"].permute(0, 2, 1, 3).contiguous()
    else:
        k, v = c["k_hnd"], c["v_hnd"]
    kwargs = dict(
        q=c["q"],
        k_cache=k,
        v_cache=v,
        qo_indptr=c["qo_indptr"],
        kv_seq_lens=c["kv_seq_lens"],
        block_tables=None if csr else c["block_tables"],
        kv_page_indices=c["kv_page_indices"] if csr else None,
        sm_scale=0.1,
        kv_layout=kv_layout,
        causal=causal,
        window_left=window_left,
        lse_mode=lse_mode,
    )
    ref_out, ref_lse = reference_paged_prefill(
        c["q"],
        k,
        v,
        c["qo_indptr"],
        c["kv_seq_lens"],
        None if csr else c["block_tables"],
        c["page_size"],
        bool(causal),
        sm_scale=0.1,
        window_left=window_left,
        kv_layout=LAYOUTS[kv_layout],
        kv_page_indices=c["kv_page_indices"] if csr else None,
        lse_base="2" if lse_mode == 1 else "e",
    )
    exported = _exec(_render_reference_source(_paged_attention_reference))
    for fn in (_paged_attention_reference, exported["_paged_attention_reference"]):
        out, lse = fn(**kwargs)
        assert out.dtype == c["q"].dtype
        torch.testing.assert_close(out, ref_out, atol=1e-5, rtol=1e-5)
        if lse_mode == 0:
            assert lse is None
        else:
            assert lse.dtype == torch.float32
            torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-5)


# ── exported init / reference round trip on the GPU ──────────────────────────


@cuda_only
@pytest.mark.parametrize("lse_mode", ["none", "basee"])
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("form", list(FORMS))
def test_exported_init_and_reference_rebuild_the_traced_plan(form, layout, lse_mode):
    """init from the JSON builds a valid planned instance of the traced
    variant; its run() matches the JSON's reference; re-tracing it yields the
    same definition."""
    p = _problem(19, form=form, layout=layout)
    attn = _plan(p, "fa2", lse_mode=lse_mode)
    defn = _trace(attn, p)
    const = {k: v["value"] for k, v in defn["axes"].items() if v["type"] == "const"}

    init_fn = _exec(defn["init"])["_paged_attention_init"]
    inputs = init_fn(
        total_q=40,
        batch_size=3,
        csr=int(form == "csr"),
        backend="fa2",
        device=p["device"],
        **const,
    )
    assert set(inputs) == {"plan", "run"}
    assert inputs["plan"]["kv_layout"] == layout
    assert inputs["plan"]["lse_mode"] == lse_mode
    rebuilt = PagedAttention(torch.device(p["device"]))
    rebuilt.plan(**inputs["plan"])
    out, lse = rebuilt.run(**inputs["run"])
    assert out.shape == (40, 8, 128)
    assert (lse is None) == (lse_mode == "none")

    ctx = rebuilt._trace_context()
    ref_fn = _exec(defn["reference"])["_paged_attention_reference"]
    ref_out, ref_lse = ref_fn(
        inputs["run"]["q"],
        *inputs["run"]["kv_cache"],
        ctx["qo_indptr"],
        ctx["kv_seq_lens"],
        block_tables=ctx["block_tables"],
        kv_page_indices=ctx["kv_page_indices"],
        kv_layout=const["kv_layout"],
        causal=const["causal"],
        window_left=const["window_left"],
        lse_mode=const["lse_mode"],
    )
    torch.testing.assert_close(out.float(), ref_out.float(), **OUT_TOL)
    if lse_mode != "none":
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
    # partial last pages and scattered pages really are in the bundle
    assert int((ctx["kv_seq_lens_cpu"] % ctx["page_size"] != 0).sum()) >= 1

    again = fi_trace(rebuilt.run, **inputs["run"])
    assert again["name"] == defn["name"]
    assert again["axes"] == defn["axes"]
    assert again["inputs"] == defn["inputs"] and again["outputs"] == defn["outputs"]


@cuda_only
def test_direct_template_accepts_init_bundle_without_instance():
    """A template's own fi_trace works on init's flattened bundle (metadata=)."""
    tpl = _paged_attention_template(csr=1, kv_layout=1, lse_mode=1)
    inputs = _paged_attention_init(
        total_q=8, batch_size=2, csr=1, kv_layout=1, lse_mode=1, device="cuda"
    )
    defn = tpl.build_fi_trace_fn(FI_API_TAG[len("fi_api:") :])(
        **inputs["plan"], **inputs["run"]
    )
    _assert_complete(defn)
    assert defn["name"].startswith("paged_attention_csr_")
    assert defn["inputs"]["kv_page_indices"]["dtype"] == "int32"


# ── auto-dump ────────────────────────────────────────────────────────────────


@cuda_only
def test_auto_dump_writes_complete_json(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(trace_template_mod, "_DUMPED_NAMES", set())
    p = _problem(23, form="csr", layout="NHD")
    attn = _plan(p, "fa2", lse_mode="basee")
    attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    expected = fi_trace(attn.run, q=p["q"], kv_cache=(p["k_cache"], p["v_cache"]))
    path = tmp_path / f"{expected['name']}.json"
    assert path.exists(), sorted(f.name for f in tmp_path.iterdir())
    doc = json.loads(path.read_text())
    _assert_complete(doc)
    assert doc == expected
    assert expected["name"] in trace_template_mod._DUMPED_NAMES
    # dedup: a second run does not rewrite the file
    stamp = path.stat().st_mtime_ns
    attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    assert path.stat().st_mtime_ns == stamp


# ── identity ─────────────────────────────────────────────────────────────────


@cuda_only
def test_definition_name_encodes_plan_semantics():
    """Same tensor shapes, different plan semantics -> different definitions;
    the same semantics on another backend -> the same definition name."""
    names = {}
    for form in FORMS:
        for layout in LAYOUTS:
            p = _problem(29, form=form, layout=layout)
            for lse_mode in LSE_MODES:
                for causal, window_left in ((True, -1), (False, -1), (True, 32)):
                    attn = _plan(
                        p,
                        "fa2",
                        causal=causal,
                        window_left=window_left,
                        lse_mode=lse_mode,
                    )
                    key = (form, layout, lse_mode, causal, window_left)
                    names[key] = _trace(attn, p)["name"]
    assert len(set(names.values())) == len(names)
    p = _problem(29)
    for backend in ("cudnn", "trtllm-gen"):
        if _runnable(p, backend, causal=True, window_left=-1, lse_mode="base2"):
            attn = _plan(p, backend)
            assert _trace(attn, p)["name"] == names[("dense", "HND", "base2", True, -1)]


@cuda_only
def test_dispatch_is_cached_and_publishes_representatives():
    p = _problem(31)
    attn = _plan(p, "fa2")
    tpl = paged_attention_trace_dispatch(self=attn, q=p["q"])
    assert tpl is paged_attention_trace_dispatch(self=attn)
    assert tpl is _paged_attention_template(lse_mode=1)
    assert tpl.identity == dict(
        csr=0, kv_layout=0, causal=1, window_left=-1, lse_mode=1, fp8_kv=0
    )
    labels = [t.name_prefix for t in paged_attention_trace_dispatch.templates]
    assert labels == ["paged_attention_dense", "paged_attention_csr"]
    assert all(
        t.init is _paged_attention_init
        for t in paged_attention_trace_dispatch.templates
    )
    assert hasattr(PagedAttention.run, "fi_trace") and hasattr(
        PagedAttention.run, "fi_init"
    )


# ── diagnostics ──────────────────────────────────────────────────────────────


@cuda_only
def test_trace_before_plan_raises():
    p = _problem(37)
    attn = PagedAttention(torch.device(p["device"]))
    with pytest.raises(ValueError, match="before plan"):
        fi_trace(attn.run, q=p["q"], kv_cache=(p["k_cache"], p["v_cache"]))
    with pytest.raises(ValueError, match="before plan"):
        attn._trace_context()


@cuda_only
def test_unbound_trace_raises_instead_of_guessing():
    p = _problem(37)
    with pytest.raises(ValueError, match=r"flashinfer\.fi_trace\(attn\.run"):
        PagedAttention.run.fi_trace(q=p["q"], kv_cache=(p["k_cache"], p["v_cache"]))
    attn = _plan(p, "fa2")
    with pytest.raises(ValueError, match="requires the run\\(\\) tensor"):
        fi_trace(attn.run, q=p["q"])
    # a template of another identity refuses the bound plan
    other = _paged_attention_template(csr=1, lse_mode=1)
    with pytest.raises(ValueError, match="does not match this template"):
        other.build_fi_trace_fn("x")(
            self=attn, q=p["q"], kv_cache=(p["k_cache"], p["v_cache"])
        )


def test_template_factory_rejects_bad_identity():
    with pytest.raises(ValueError, match="lse_mode"):
        _paged_attention_template(lse_mode=3)
    with pytest.raises(ValueError, match="window_left"):
        _paged_attention_template(window_left=-2)
    with pytest.raises(ValueError, match="kv_layout"):
        _paged_attention_template(kv_layout=True)


# ── failed re-plan, graph storage, read-only ─────────────────────────────────


def _permuted_pages(p, seed):
    """Same batch (lengths, shapes) with the pool pages renamed: a re-plan
    that fits the graph capacity and changes the page mapping only."""
    g = torch.Generator().manual_seed(seed)
    pool = p["k_cache"].shape[0]
    rename = torch.randperm(pool, generator=g).to(torch.int32)
    bt = rename[p["block_tables"].cpu().long()]
    flat = rename[p["kv_page_indices"].cpu().long()]
    dev = torch.device(p["device"])
    return dict(p, block_tables=bt.to(dev), kv_page_indices=flat.to(dev))


@cuda_only
@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_failed_replan_still_traces_last_successful_plan(graph):
    p1 = _problem(41)
    attn = _plan(p1, "fa2", graph=graph)
    before = _trace(attn, p1)
    ctx1 = attn._trace_context()
    kv_before = ctx1["kv_seq_lens"].clone()

    bad = dict(p1)
    kv_bad = p1["kv_seq_lens_cpu"].clone()
    i = int(p1["qo_indptr_cpu"].diff().argmax())
    kv_bad[i] = max(1, int(p1["qo_indptr_cpu"].diff()[i]) - 1)  # q_len > kv_len
    bad["kv_seq_lens_cpu"] = kv_bad
    bad["kv_seq_lens"] = kv_bad.to(p1["device"])
    with pytest.raises(ValueError, match="causal masking requires"):
        _plan_into(attn, bad)
    if graph:
        smaller = make_problem(42, **dict(_SHAPE, batch_size=3))
        with pytest.raises(ValueError, match="CUDA graph re-plan: batch_size"):
            _plan_into(attn, smaller)

    ctx = attn._trace_context()
    assert ctx["kv_seq_lens"].data_ptr() == ctx1["kv_seq_lens"].data_ptr()
    assert torch.equal(ctx["kv_seq_lens"], kv_before)
    assert torch.equal(ctx["kv_seq_lens_cpu"], p1["kv_seq_lens_cpu"])
    assert _trace(attn, p1) == before


def _plan_into(attn, p, backend="fa2"):
    attn.plan(
        make_metadata(p),
        num_qo_heads=p["num_qo_heads"],
        num_kv_heads=p["num_kv_heads"],
        head_dim_qk=p["head_dim_qk"],
        head_dim_vo=p["head_dim_vo"],
        q_dtype=p["dtype"],
        kv_layout=p["kv_layout"],
        causal=True,
        lse_mode="base2",
        backend=backend,
    )


@cuda_only
@pytest.mark.parametrize("form", list(FORMS))
def test_graph_mode_trace_reads_reserved_storage(form):
    p1 = _problem(43, form=form)
    attn = _plan(p1, "fa2", graph=True)
    gb = attn._impl._graph
    assert gb is not None
    ctx = attn._trace_context()
    assert ctx["graph_capacity"] is gb.capacity
    assert ctx["qo_indptr"].data_ptr() == gb.qo_indptr.data_ptr()
    assert ctx["kv_seq_lens"].data_ptr() == gb.kv_seq_lens.data_ptr()
    live = int(((p1["kv_seq_lens_cpu"] + 15) // 16).sum())
    if form == "dense":
        assert ctx["kv_page_indices"] is None
        assert ctx["block_tables"].data_ptr() == gb.block_tables.data_ptr()
        assert torch.equal(ctx["block_tables"], p1["block_tables"])
    else:
        assert ctx["block_tables"] is None
        assert ctx["kv_page_indices"].data_ptr() == gb.kv_page_indices.data_ptr()
        assert ctx["kv_page_indices"].numel() == live
        assert torch.equal(ctx["kv_page_indices"], p1["kv_page_indices"][:live])
    defn1 = _trace(attn, p1)

    # re-plan into the same capacity: same storage, new contents, same identity
    p2 = _permuted_pages(p1, seed=44)
    _plan_into(attn, p2)
    ctx2 = attn._trace_context()
    table = "block_tables" if form == "dense" else "kv_page_indices"
    assert ctx2[table].data_ptr() == ctx[table].data_ptr()
    expected = p2[table] if form == "dense" else p2[table][:live]
    assert torch.equal(ctx2[table], expected)
    assert not torch.equal(
        ctx2[table], p1[table] if form == "dense" else p1[table][:live]
    )
    assert _trace(attn, p2) == defn1


@cuda_only
def test_trace_is_sync_free_and_read_only():
    p = _problem(47, form="csr")
    attn = _plan(p, "fa2")
    impl = attn._impl
    published = (impl._meta, impl._derived, impl._active, impl._backend_name)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        ctx = attn._trace_context()
        defn = fi_trace(attn.run, q=p["q"], kv_cache=(p["k_cache"], p["v_cache"]))
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert (impl._meta, impl._derived, impl._active, impl._backend_name) == published
    assert ctx["backend"] == "fa2" and defn["name"].startswith("paged_attention_csr_")
    assert ctx["kv_page_indices"].numel() == int(
        ((p["kv_seq_lens_cpu"] + 15) // 16).sum()
    )


# ── schema consistency (the stable lane filters experimental templates) ──────


def _run_params():
    return {n for n in inspect.signature(PagedAttention.run).parameters if n != "self"}


@pytest.mark.parametrize(
    "tpl",
    [
        _paged_attention_template(),
        _paged_attention_template(csr=1),
        _paged_attention_template(kv_layout=1, causal=0, lse_mode=2),
        _paged_attention_template(csr=1, window_left=64, lse_mode=1, fp8_kv=1),
    ],
    ids=["dense", "csr", "nhd_noncausal_basee", "csr_window_fp8"],
)
def test_template_schema_is_consistent_with_run(tpl):
    params = _run_params()
    for key, d in tpl.inputs.items():
        if getattr(d, "optional", False):
            continue
        assert (d.param or key) in params, f"{key} -> {d.param or key} not in run()"
    tensor_dims = {
        n for d in tpl.inputs.values() if isinstance(d, Tensor) for n in d.dim_names
    }
    scalar_keys = {k for k, d in tpl.inputs.items() if isinstance(d, Scalar)}
    for name, marker in tpl.axes.items():
        if isinstance(marker, Const) and marker.value is None:
            assert name in tensor_dims, f"Const axis {name} has no tensor source"
        if isinstance(marker, Const) and marker.value is not None:
            assert name in scalar_keys, (
                f"semantic axis {name} is not a reference scalar"
            )
    allowed = set(tpl.axes) | set(tpl.inputs) | {"max", "min"}
    for constraint in tpl.constraints:
        names = {
            n.id
            for n in ast.walk(ast.parse(constraint, mode="eval"))
            if isinstance(n, ast.Name)
        }
        assert names <= allowed, f"{constraint!r} references {names - allowed}"
    for out in tpl.outputs.values():
        assert out.param in ("out", "lse")
    assert tpl.op_type == "gqa_paged"
    assert not tpl.name_prefix.startswith("gqa_paged")
    # the reference signature carries the plan semantics as keyword-only ints
    ref = inspect.signature(_paged_attention_reference)
    for name in ("kv_layout", "causal", "window_left", "lse_mode"):
        assert ref.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert compile(_render_init_source(_paged_attention_init), "init", "exec")


def test_import_flashinfer_does_not_load_the_experimental_package():
    script = (
        "import sys\n"
        "import flashinfer, flashinfer.prefill\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('flashinfer.experimental'))\n"
        "assert 'flashinfer.experimental.paged_attention' not in sys.modules, loaded\n"
        "assert 'flashinfer.trace.templates.paged_attention' in sys.modules\n"
        "assert hasattr(flashinfer.prefill.PagedAttention.run, 'fi_trace')\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("ok")
