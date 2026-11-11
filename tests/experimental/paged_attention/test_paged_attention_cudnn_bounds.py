"""Declared bounds of the unified cuDNN backend (``backend="cudnn"`` pinned).

cuDNN builds one execution plan per declared shape, and a serving loop's
longest KV grows on every step.  An eager ``plan()`` therefore declares a
shape class instead of the batch's exact maxes
(``_CudnnBackend._declared_bounds``): the KV bound is the next power of two,
the Q bound follows the query-tile rule from 64 tokens up.  The per-request
lengths stay the actual ones.  The table cuDNN binds is the backend's own
(``_CudnnBackend._own_pages_table``): the class's columns, each request's
own pages followed by its last page again, in one layout whatever the caller
handed over -- a dense table of any width or row stride, or flat page ids.
Pinned here, against the fp32 paged-attention oracle and without timing
thresholds:

1. A serving loop on one eager instance (one request, and a ragged batch of
   three) prepares one cuDNN graph per KV class, not one per step: through a
   dense table of the class width, a dense table narrower than the class and
   flat page ids.
2. The Q classes at their boundaries, and that a rounded bound writes
   nothing past the caller's packed ``out`` / ``lse``.
3. The KV class does not depend on the table: narrower than the class, of
   its width, wider, a row-stride view, derived from flat page ids.  One
   instance alternates between those sources and between two classes; a
   dense caller and a flat-id caller of one class reach one cuDNN graph.
4. The bound table: every eager plan hands cuDNN each request's own pages
   only, so no pool page the batch does not own can reach the result --
   neither what a dense caller left in the columns past a request's pages
   nor what a flat-id caller left past its page ids (ids of NaN-filled pages
   here), nor pool page 0.  A request without KV, and one whose device
   length is past its CPU mirror.  The table is built on the device without
   a host sync, a failed plan leaves the published one's table alone, and
   its plan-time cost is pinned as kernel launches.  CUDA-graph mode binds
   its reserved table, the caller's columns included, and promises nothing
   about them.
5. The three LSE layouts, NHD, non-causal and run-time scales with a rounded
   bound.
6. CUDA-graph mode declares its capacity, not a class.
7. The functional fallback declares the same class and reaches the same
   cuDNN graph.

The declared values are the ones run() hands the functional entry, checked
against the staged plan's metadata where a plan is staged; the backend's
private state is read through ``attn._impl._active``, as the neighbouring
cuDNN tests do.
"""

import pytest
import torch

import flashinfer.cudnn as fi_cudnn
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.experimental.paged_attention._backends import cudnn_backend
from flashinfer.prefill import GraphCapacity

from .test_paged_attention_coverage import (
    assert_canaries,
    assert_matches,
    canary_buffers,
    plan,
    reference,
    run,
)
from .test_paged_attention_cudnn_prepared import (
    DEVICE,
    PAGE,
    _RUN_SCALES,
    _answer_packed_lse_probe,
    _assert_rows_match,
    _assert_staged,
    _capture_buffers,
    _count_preparations,
    _forbid_functional_entry,
    _isolate_graph_cache,
    _load,
    _no_staged_plan,
    _plan_cudnn,
    _problem,
    _replay,
    _require_prepared,
    _update,
    _warm_and_capture,
)
from .test_paged_attention_prototype import LSE_TOL, OUT_TOL, _gpu_activity

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        not cudnn_prefill.CUDNN_AVAILABLE,
        reason="cudnn-frontend python package not available",
    ),
]

# Three requests with a query of 64 tokens or more next to shorter ones: at
# (8, 2) heads the Q class is 128 (32-token query tiles), and the KV class
# is 256, 16 page columns.
_ROUNDED = ([70, 3, 64], [200, 90, 64])
_ROUNDED_BOUNDS = (128, 256)

# What a caller hands over, as _problem() overrides for a batch whose KV
# class has ``columns`` page columns: a dense table of the class width, a
# dense table of the batch's own width (narrower than the class unless the
# longest request ends in the class's last page), or flat page ids, from
# which the controller derives a table of the batch's own width.
_SOURCES = ["class_width", "own_width", "flat"]


def _source(kind, columns):
    return {
        "class_width": dict(table_width=columns),
        "own_width": dict(),
        "flat": dict(input_form="page_indices"),
    }[kind]


def _declared(backend):
    """(max_q, max_kv, table columns) of the published plan: what run() hands
    the functional entry, and the same values in the staged plan's metadata
    where a plan is staged."""
    declared = (*backend._declared, backend._block_tables.shape[1])
    if backend._cudnn_plan is not None:
        metadata = backend._cudnn_plan.metadata
        assert declared == (
            metadata.max_token_per_sequence,
            metadata.max_sequence_kv,
            metadata.block_tables.shape[1],
        )
        assert metadata.block_tables.data_ptr() == backend._block_tables.data_ptr()
    return declared


def _page_ids(p):
    """The page ids in the form the caller hands over."""
    return p["kv_page_indices" if p["input_form"] == "page_indices" else "block_tables"]


def _own_pages(p):
    return [-(-int(kv) // p["page_size"]) for kv in p["kv_seq_lens_cpu"]]


def _own_pages_table(p, columns):
    """``columns`` columns per request: its own pages followed by its last
    page again (a request without KV repeats the first column of its row)."""
    given = p["block_tables"]
    table = given.new_empty(given.shape[0], columns)
    for i, n in enumerate(_own_pages(p)):
        table[i, :n] = given[i, :n]
        table[i, n:] = given[i, max(n - 1, 0)]
    return table


def _binds_own_pages_table(backend, p):
    """The table cuDNN binds is the backend's own: a contiguous table of the
    declared width in storage apart from the plan's dense table (the
    caller's, or the one derived from flat page ids), naming each request's
    own pages followed by its last page again.  A request without KV owns no
    page; its row is compared in the dense form only, where it repeats the
    first column of the caller's row."""
    bound, source = backend._block_tables, backend._meta.block_tables
    columns = -(-backend._declared[1] // p["page_size"])
    dense = p["input_form"] == "block_tables"
    rows = [i for i, n in enumerate(_own_pages(p)) if n or dense]
    return (
        tuple(bound.shape) == (source.shape[0], columns)
        and bound.is_contiguous()
        and bound.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        and torch.equal(bound[rows], _own_pages_table(p, columns)[rows])
    )


def _nan_tail(p):
    """``p`` with every pool page no request owns filled with NaN, pool page
    0 among them, and the page ids the caller left past each request's own
    pages naming one of them: valid ids the caller never meant this batch to
    read.  In the dense form those are the table columns past a request's
    pages; in the flat form, entries the list is over-allocated by.  Returns
    the problem and that page id."""
    table, flat = p["block_tables"].clone(), p["kv_page_indices"].clone()
    pages = _own_pages(p)
    live = set()
    for row, n in zip(table.tolist(), pages, strict=True):
        live.update(row[:n])
    dead = [i for i in range(p["k_cache"].shape[0]) if i not in live]
    k, v = p["k_cache"].clone(), p["v_cache"].clone()
    if 0 in live:
        # page 0 trades its id with an unowned page: the content moves along
        k[dead[0]], v[dead[0]] = k[0], v[0]
        for i, n in enumerate(pages):
            table[i, :n][table[i, :n] == 0] = dead[0]
        flat[flat == 0] = dead[0]
        dead[0] = 0
    k[dead] = float("nan")
    v[dead] = float("nan")
    nan_page = dead[-1]
    assert nan_page != 0
    for i, n in enumerate(pages):
        table[i, n:] = nan_page
    flat = torch.cat([flat, flat.new_full((2 * table.shape[1],), nan_page)])
    return (
        dict(
            p,
            block_tables=table,
            kv_page_indices=flat,
            k_cache=k,
            v_cache=v,
            k_ref=k,
            v_ref=v,
        ),
        nan_page,
    )


# ---------------------------------------------------------------------------
# 1: a serving loop
# ---------------------------------------------------------------------------

# One eager instance re-planned on every step, each request's KV growing by
# its query length: (q_lens, the KV lengths of step 0, _problem() overrides,
# the declared Q bound, the KV class of each step).
_GROWS_TO_256 = [128] * 4 + [256] * 2
_GROWS_TO_2048 = [1024] * 2 + [2048] * 6
_SERVING = {
    # A dense table of the widest class's width.  The longest KV stays
    # within 128 for four steps, ending in the class's last page on the
    # fourth, and crosses into the 256 class on the fifth.
    "one": ([16], [80], dict(table_width=16, pool=24), 16, _GROWS_TO_256),
    "ragged": (
        [16, 8, 4],
        [80, 40, 100],
        dict(table_width=16, pool=40),
        16,
        _GROWS_TO_256,
    ),
    # The same loops on flat page ids.  The table the controller derives has
    # the batch's own columns, five to ten at page size 16: the class's
    # eight on the fourth step alone.
    "one_flat": (
        [16],
        [80],
        dict(input_form="page_indices", pool=24),
        16,
        _GROWS_TO_256,
    ),
    "ragged_flat": (
        [16, 8, 4],
        [80, 40, 100],
        dict(input_form="page_indices", pool=40),
        16,
        _GROWS_TO_256,
    ),
    # Flat page ids at page size 8, the smallest a dense table is derived
    # at: the 256 class has 32 columns.
    "one_flat_page_8": (
        [16],
        [80],
        dict(input_form="page_indices", page=8, pool=28),
        16,
        _GROWS_TO_256,
    ),
    "ragged_flat_page_8": (
        [16, 8, 4],
        [80, 40, 100],
        dict(input_form="page_indices", page=8, pool=53),
        16,
        _GROWS_TO_256,
    ),
    # A dense table of 100 columns, 1600 tokens: from 928 tokens across 1024
    # up to the table's capacity.  The 2048 class has 128 columns, more than
    # the table, on six of the eight steps.
    "one_narrow": (
        [96],
        [928],
        dict(table_width=100, pool=124),
        128,
        _GROWS_TO_2048,
    ),
    "ragged_narrow": (
        [96, 8, 4],
        [928, 40, 100],
        dict(table_width=100, pool=124),
        128,
        _GROWS_TO_2048,
    ),
}


@pytest.mark.parametrize("lse_mode", ["none", "base2"])
@pytest.mark.parametrize("loop", list(_SERVING))
def test_serving_loop_prepares_one_graph_per_kv_class(monkeypatch, loop, lse_mode):
    """One eager instance re-planned on every step with fresh metadata
    tensors, other page ids and a longer KV.  Every step matches its oracle,
    declares the class of its longest KV and binds a table of each request's
    own pages at the class width, whatever the caller handed over; a cuDNN
    graph is prepared on the first step and once more when the longest KV
    crosses into the next power-of-two class, never in between."""
    q_lens, kv_lens, overrides, graph_q, kv_classes = _SERVING[loop]
    steps = [
        _problem(
            q_lens,
            [kv + step * q for q, kv in zip(q_lens, kv_lens, strict=True)],
            seed=200 + step,
            **overrides,
        )
        for step in range(len(kv_classes))
    ]
    longest = [p["max_kv_len"] for p in steps]
    assert [1 << (kv - 1).bit_length() for kv in longest] == kv_classes
    assert len(set(longest)) == len(steps)
    assert kv_classes[0] in longest  # one step ends in its class's last page
    if loop.endswith("narrow"):  # up to the table's capacity, short of the class
        assert longest[-1] == overrides["table_width"] * PAGE < kv_classes[-1]

    attn, backend = _plan_cudnn(steps[0], lse_mode=lse_mode)
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    previous = None
    for step, (p, kv_class) in enumerate(zip(steps, kv_classes, strict=True)):
        _plan_cudnn(p, lse_mode=lse_mode, attn=attn)
        if previous is not None:
            assert not torch.equal(_page_ids(p), _page_ids(previous))
        previous = p
        for _ in range(2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode=lse_mode)
        assert _declared(backend) == (graph_q, kv_class, kv_class // p["page_size"])
        assert _binds_own_pages_table(backend, p)
        if staged:
            assert len(preparations) == len(set(kv_classes[: step + 1]))


# ---------------------------------------------------------------------------
# 2: the Q classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_q,heads,graph_q",
    [
        (63, (8, 2), 63),  # below 64: exact
        (64, (8, 2), 64),
        (65, (8, 2), 128),
        (128, (8, 2), 128),
        (129, (8, 1), 144),  # group 8: 16-token query tiles
        (129, (4, 4), 256),  # group 1: 128-token query tiles
        (129, (6, 2), 129),  # group 3 does not divide 128: exact
    ],
)
def test_q_class_boundaries(max_q, heads, graph_q):
    """The declared Q bound next to a shorter request.  q / out / lse stay
    packed at the real token count: with a bound above the longest query
    nothing is written around the caller's buffers."""
    p = _problem([max_q, 5], [max_q + 40, 30], seed=210, heads=heads)
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    assert _declared(backend)[0] == graph_q
    out_big, out_view, lse_big, lse_view = canary_buffers(p)
    for _ in range(2):
        out, lse = attn.run(
            p["q"], (p["k_cache"], p["v_cache"]), out=out_view, lse=lse_view
        )
        assert out is out_view and lse is lse_view
        assert_matches(out, lse, p, lse_mode="base2")
        assert_canaries(out_big, out_view, 64)
        assert_canaries(lse_big, lse_view, 64)


def test_q_class_follows_the_callers_bound():
    """The class is taken from the bound the caller declares, which may be
    looser than the batch's longest request."""
    p = _problem([40, 9], [100, 30], seed=211)
    attn, backend = _plan_cudnn(dict(p, max_q_len=65), lse_mode="base2")
    assert _declared(backend)[0] == 128
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")


# ---------------------------------------------------------------------------
# 3: the KV class does not depend on the table
# ---------------------------------------------------------------------------

# max_kv_len 100, the class of 128 with eight page columns, through what a
# caller can hand over: (_problem() overrides, the columns of the plan's
# dense table).
_TABLES_OF_ONE_CLASS = {
    "one_column_short": (dict(table_width=7), 7),
    "class_width": (dict(table_width=8), 8),
    "wider": (dict(table_width=11), 11),
    # seven columns of the wider table, in its row stride of eleven
    "row_stride_view": (dict(table_width=11), 7),
    # the controller derives the batch's own seven columns
    "flat": (dict(input_form="page_indices"), 7),
}


@pytest.mark.parametrize("table", list(_TABLES_OF_ONE_CLASS))
def test_kv_class_does_not_depend_on_the_table(table):
    """Whatever the plan's dense table has -- fewer columns than the class,
    as many, more, another row stride, the columns derived from flat page
    ids -- the plan declares the class and binds eight columns of each
    request's own pages."""
    overrides, source_columns = _TABLES_OF_ONE_CLASS[table]
    p = _problem([7, 16, 3], [64, 30, 100], seed=220, **overrides)
    if table == "row_stride_view":
        p = dict(p, block_tables=p["block_tables"][:, :source_columns])
        assert p["block_tables"].stride(0) == 11
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    assert backend._meta.block_tables.shape[1] == source_columns
    assert _declared(backend) == (16, 128, 8)
    assert _binds_own_pages_table(backend, p)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")


def test_callers_own_class_bound_binds_the_own_pages_table():
    """A caller that declares the power of two itself declares the class: no
    column is added to its bound, and the plan binds each request's own
    pages like every other eager plan."""
    p = _problem([7, 16, 3], [64, 30, 100], seed=221, table_width=11)
    attn, backend = _plan_cudnn(dict(p, max_kv_len=128), lse_mode="base2")
    assert _declared(backend) == (16, 128, 8)
    assert _binds_own_pages_table(backend, p)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")


@pytest.mark.parametrize("max_kv", [90, 120, 128])
def test_table_derived_from_flat_page_ids(max_kv):
    """The flat form: the controller derives a table of the batch's own
    width, six or eight columns here, each row's tail already repeating the
    request's last page.  The backend binds neither that table nor a view of
    it, but the class's eight columns built from it."""
    p = _problem([7, 16, 3], [64, 30, max_kv], seed=222, input_form="page_indices")
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    assert backend._meta.block_tables.shape[1] == -(-max_kv // PAGE)
    assert _declared(backend) == (16, 128, 8)
    assert _binds_own_pages_table(backend, p)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")


def test_replans_across_table_sources_and_kv_classes(monkeypatch):
    """One instance alternating between what callers hand over -- flat page
    ids, a wide dense table, one of the batch's own width -- and between the
    KV classes of 128 and 256, with other lengths and page ids on every
    plan, so a binding left over from the previous plan's table would read
    the wrong pages.  Each plan matches its oracle and binds its own pages
    in its class's layout: two layouts, hence two prepared cuDNN graphs,
    both kept in the instance history.  The 256 class has sixteen columns,
    more than either table it is declared through."""
    same = dict(pool=30)
    flat = dict(input_form="page_indices", **same)
    flat_a = _problem([7, 16, 3], [64, 30, 90], seed=230, **flat)
    wide = _problem([16, 2, 9], [90, 41, 17], seed=231, table_width=11, **same)
    # ends in the class's last page
    flat_b = _problem([4, 9, 16], [33, 120, 60], seed=232, **flat)
    own = _problem([16, 16, 1], [16, 90, 77], seed=233, **same)
    longer = _problem([16, 3, 7], [150, 30, 64], seed=234, table_width=11, **same)
    longer_flat = _problem([9, 16, 2], [40, 200, 77], seed=235, **flat)
    assert own["block_tables"].shape[1] == 6

    attn, backend = _plan_cudnn(flat_a, lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    steps = (
        (flat_a, 128, 1),
        (wide, 128, 1),
        (longer, 256, 2),
        (flat_b, 128, 2),
        (own, 128, 2),
        (longer_flat, 256, 2),
        (wide, 128, 2),
        (longer, 256, 2),
        (flat_a, 128, 2),
    )
    for p, kv_class, prepared_so_far in steps:
        _plan_cudnn(p, lse_mode="base2", attn=attn)
        for _ in range(2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode="base2")
        assert _declared(backend) == (16, kv_class, kv_class // PAGE)
        assert _binds_own_pages_table(backend, p)
        if staged:
            assert len(preparations) == prepared_so_far


@pytest.mark.parametrize("dense", [t for t in _TABLES_OF_ONE_CLASS if t != "flat"])
def test_dense_and_flat_callers_of_one_class_share_one_cudnn_graph(monkeypatch, dense):
    """Two instances on one process, as two engines or two layers would be:
    one planned from a dense table, one from flat page ids, with other
    lengths and page ids but the same batch size, K/V pool and classes.  The
    layout of the bound table is the class's alone, so both ask the graph
    cache for the same key and get the same cuDNN graph object, and on a
    private graph cache the second caller builds nothing."""
    _require_prepared()
    overrides, source_columns = _TABLES_OF_ONE_CLASS[dense]
    of_dense = _problem([7, 16, 3], [64, 30, 100], seed=240, pool=30, **overrides)
    if dense == "row_stride_view":
        table = of_dense["block_tables"][:, :source_columns]
        of_dense = dict(of_dense, block_tables=table)
    of_flat = _problem(
        [16, 2, 9], [77, 41, 17], seed=241, pool=30, input_form="page_indices"
    )
    # before the private cache: the one-time probe fetches a toy graph
    _answer_packed_lse_probe()
    _isolate_graph_cache(monkeypatch)
    fetched = _count_preparations(monkeypatch)
    builds = []
    for p in (of_dense, of_flat):
        before = cudnn_prefill._prefill_graph_builds
        attn, backend = _plan_cudnn(p, lse_mode="base2")
        assert _declared(backend) == (16, 128, 8)
        for _ in range(2):
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode="base2")
        builds.append(cudnn_prefill._prefill_graph_builds - before)
    graph_of_dense, graph_of_flat = fetched
    assert graph_of_flat.key == graph_of_dense.key
    assert graph_of_flat.graph is graph_of_dense.graph
    assert builds[0] >= 1  # the private cache started empty
    assert builds[1] == 0


# ---------------------------------------------------------------------------
# 4: the bound table
# ---------------------------------------------------------------------------

# (_problem() overrides, q_lens, kv_lens, the columns of the KV class).
# ``tiles`` is a shape where cuDNN's backend engine was measured to fetch
# the pages of a request's last 128-token KV tile, past its last page,
# through the table, and pool page 0 for what a bound that is not whole
# tiles leaves of the last one.  In ``last_page`` the longest request ends
# in the class's last page, so the class adds no column to the caller's
# bound and only the shorter requests have a tail.
_TAIL_SHAPES = {
    "below_one_tile": (dict(), [5, 17, 1, 9], [40, 17, 60, 33], 4),
    "small_one": (dict(), [17], [70], 8),
    "small_ragged": (dict(), [5, 17, 1, 9], [40, 17, 70, 33], 8),
    "tiles_one": (dict(heads=(8, 1)), [129], [700], 64),
    "tiles_ragged": (dict(heads=(8, 1)), [129, 30, 64], [700, 30, 400], 64),
    "tiles_last_page": (dict(heads=(8, 1)), [129, 30, 64], [1020, 30, 400], 64),
    "tiles_last_token": (dict(heads=(8, 1)), [129, 30], [1024, 100], 64),
    # large enough for the frontend's own prefill engines where the stack
    # has them
    "large_ragged": (dict(), [300, 64, 1], [3000, 1100, 2048], 256),
    "padding_row": (dict(), [9, 0, 20], [100, 0, 37], 8),
}


@pytest.mark.parametrize("lse_mode", ["base2", "none"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("shape", list(_TAIL_SHAPES))
def test_pages_the_batch_does_not_own_do_not_reach_the_result(
    shape, source, causal, lse_mode
):
    """Every pool page no request owns holds NaN, and the caller's page ids
    past each request's own pages name one: the columns of a dense table
    that the caller never initialised for this batch, or the tail of an
    over-allocated flat list.  The output and the LSE are finite and match
    the oracle for every eager plan -- one request and ragged batches,
    through a table of the class width, a narrower one and flat page ids,
    whether or not the class adds a column."""
    kwargs, q_lens, kv_lens, columns = _TAIL_SHAPES[shape]
    p, _ = _nan_tail(
        _problem(q_lens, kv_lens, seed=250, **_source(source, columns), **kwargs)
    )
    attn, backend = _plan_cudnn(p, causal=causal, lse_mode=lse_mode)
    assert _declared(backend)[1:] == (columns * PAGE, columns)
    for _ in range(2):
        out, lse = run(attn, p)
        assert_matches(out, lse, p, causal=causal, lse_mode=lse_mode)


@pytest.mark.parametrize("source", _SOURCES)
def test_nan_tail_through_every_step_of_a_class(monkeypatch, source):
    """One eager instance, a long request growing through the KV class of
    1024 next to a short one whose tail names NaN-filled pages: before the
    class's last page, on its first and last token, with other page ids on
    every step.  Each step is finite and matches its oracle through a table
    of the requests' own pages, and the steps share one cuDNN graph."""
    kv_steps = [1000, 1008, 1009, 1024]
    steps = [
        _nan_tail(
            _problem(
                [129, 30],
                [kv, 100],
                seed=255 + i,
                heads=(8, 1),
                pool=80,
                **_source(source, 64),
            )
        )[0]
        for i, kv in enumerate(kv_steps)
    ]
    assert [_own_pages(p)[0] for p in steps] == [63, 63, 64, 64]
    attn, backend = _plan_cudnn(steps[0], lse_mode="base2")
    staged = _assert_staged(backend)
    preparations = _count_preparations(monkeypatch)
    if staged:
        _forbid_functional_entry(monkeypatch)
    for p in steps:
        _plan_cudnn(p, lse_mode="base2", attn=attn)
        assert _declared(backend) == (144, 1024, 64)
        assert _binds_own_pages_table(backend, p)
        out, lse = run(attn, p)
        assert_matches(out, lse, p, lse_mode="base2")
    if staged:
        assert len(preparations) == 1


@pytest.mark.parametrize("table", ["own_width", "class_width", "wider", "flat"])
@pytest.mark.parametrize("shape", list(_TAIL_SHAPES))
def test_bound_table_repeats_each_requests_last_page(shape, table):
    """Why the tests above hold whatever engine cuDNN selects: the table an
    eager plan binds is the backend's own, the class's columns with each
    request's own pages followed by its last page again (a request without
    KV repeats its first column), contiguous whatever the caller's table
    has.  What the caller handed over is left as it was."""
    kwargs, q_lens, kv_lens, columns = _TAIL_SHAPES[shape]
    overrides = (
        dict(table_width=columns + 3) if table == "wider" else _source(table, columns)
    )
    p, nan_page = _nan_tail(_problem(q_lens, kv_lens, seed=251, **overrides, **kwargs))
    given = _page_ids(p).clone()
    _, backend = _plan_cudnn(p, lse_mode="base2")
    bound = backend._block_tables
    assert tuple(bound.shape) == (len(q_lens), columns)
    assert _binds_own_pages_table(backend, p)
    if all(_own_pages(p)):
        assert not (bound == nan_page).any()
    assert torch.equal(_page_ids(p), given)


def _without_request(p, row):
    """The batch ``p`` without request ``row``, as a dense-form problem for
    the oracle, and the rows of ``p``'s packed query the other requests
    own."""
    qo = p["qo_indptr_cpu"]
    others = [i for i in range(qo.shape[0] - 1) if i != row]
    rows = torch.cat([torch.arange(int(qo[i]), int(qo[i + 1])) for i in others])
    q_lens = qo.diff()[others]
    compact = dict(
        p,
        input_form="block_tables",
        q=p["q"][rows.to(DEVICE)],
        qo_indptr_cpu=torch.cat([q_lens.new_zeros(1), q_lens.cumsum(0)]),
        kv_seq_lens_cpu=p["kv_seq_lens_cpu"][others],
        block_tables=p["block_tables"][others],
    )
    return compact, rows.to(DEVICE)


@pytest.mark.parametrize("queries", [0, 5], ids=["no_query", "with_query"])
@pytest.mark.parametrize("row", [0, 1, 2], ids=["leading", "middle", "trailing"])
@pytest.mark.parametrize("source", _SOURCES)
def test_request_without_kv_under_the_class(source, row, queries):
    """A padding row (kv_len 0), without query tokens as vLLM pads a batch
    and with some, at each position of a batch of three under the KV class
    of 128.  It owns no page, so its row of the bound table names whatever
    the caller's row starts with -- a NaN-filled page in the dense form
    here.  The other requests are finite and match the oracle of the batch
    without it; the row's own output is unspecified."""
    q_lens, kv_lens = [9, 16, 20], [100, 70, 90]
    q_lens[row], kv_lens[row] = queries, 0
    p, _ = _nan_tail(_problem(q_lens, kv_lens, seed=266, **_source(source, 8)))
    attn, backend = _plan_cudnn(p, lse_mode="base2")
    assert _declared(backend)[1:] == (128, 8)
    assert _binds_own_pages_table(backend, p)
    others, rows = _without_request(p, row)
    ref_out, ref_lse = reference(others)
    for _ in range(2):
        out, lse = run(attn, p)
        assert out.shape[0] == lse.shape[0] == sum(q_lens)
        assert torch.isfinite(out[rows]).all() and torch.isfinite(lse[rows]).all()
        torch.testing.assert_close(out[rows].float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse[rows], ref_lse, **LSE_TOL)


@pytest.mark.parametrize("device_kv", [120, 1100], ids=["one_page", "past_the_class"])
@pytest.mark.parametrize("source", ["own_width", "flat"])
def test_device_length_past_its_mirror_stays_within_the_source_table(source, device_kv):
    """Outside the contract: plan() validates the CPU mirror of the KV
    lengths and trusts the device tensor to match it.  A device length that
    names more pages than its mirror -- one more, the class's last column,
    or more than the class has -- must still not index past a source table
    narrower than the class, dense or derived from flat page ids: an
    out-of-range gather is a device-side assert, after which every CUDA call
    of the process fails.  The plan binds the request's source columns
    followed by the last of them again (what such a request computes is
    unspecified, and it is not run here), and the instance then serves the
    batch under its true lengths."""
    p = _problem([7, 16, 3], [64, 30, 100], seed=267, pool=30, **_source(source, 8))
    kv_seq_lens = p["kv_seq_lens"].clone()
    kv_seq_lens[2] = device_kv
    attn, backend = _plan_cudnn(dict(p, kv_seq_lens=kv_seq_lens), lse_mode="base2")
    torch.cuda.synchronize()  # where a device-side assert would surface
    assert backend._meta.block_tables.shape[1] == 7
    assert _declared(backend) == (16, 128, 8)
    assert _binds_own_pages_table(backend, p)
    _plan_cudnn(p, lse_mode="base2", attn=attn)
    out, lse = run(attn, p)
    assert_matches(out, lse, p, lse_mode="base2")


def _plan_events(attn, p):
    """GPU events (kernels and memcpys) of one warm eager plan(), without
    LSE: the padded-stats layout precomputes gather indices on the device."""
    return _gpu_activity(lambda: plan(attn, p, "cudnn", lse_mode="none"))


def test_plan_time_kernels_of_the_bound_table():
    """What the bound table costs a warm eager plan of a dense caller, as
    launch counts: next to the metadata upload, the five small kernels that
    build it (two for each request's last page, a column range, the clamp,
    the gather) and no other, through a table narrower than the class as
    through a wider one.  (From flat page ids the controller's derivation of
    the dense table launches its own kernels before these.)"""
    narrow = _problem([7, 16, 3], [64, 30, 100], seed=264, table_width=7, pool=30)
    wide = _problem([7, 16, 3], [64, 30, 100], seed=265, table_width=9, pool=30)
    attn, backend = _plan_cudnn(narrow, lse_mode="none")
    for p in (wide, narrow):  # warm: first-plan allocations of both shapes
        _plan_cudnn(p, lse_mode="none", attn=attn)
        run(attn, p)
    for p in (narrow, wide):
        events = _plan_events(attn, p)
        if not events:  # not even the metadata upload
            pytest.skip("the profiler reports no GPU activity here")
        built = [e for e in events if not e.startswith(("Memcpy", "Memset"))]
        assert 0 < len(built) <= 5, built
        assert _binds_own_pages_table(backend, p)


@pytest.mark.parametrize("source", _SOURCES)
def test_bound_table_is_built_without_a_host_sync(source):
    """The bound table comes from the device KV lengths and the plan's
    device table: a warm eager re-plan does not synchronize (the serving
    loop re-plans on every step)."""
    form = dict(pool=30, **_source(source, 8))
    a = _problem([7, 16, 3], [64, 30, 100], seed=262, **form)
    b = _problem([16, 2, 9], [77, 110, 17], seed=263, **form)
    attn, backend = _plan_cudnn(a, lse_mode="base2")
    run(attn, a)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        plan(attn, b, "cudnn", lse_mode="base2")
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert _binds_own_pages_table(backend, b)
    out, lse = run(attn, b)
    assert_matches(out, lse, b, lse_mode="base2")


@pytest.mark.parametrize("source", _SOURCES)
def test_failed_replan_keeps_the_previous_plans_table(monkeypatch, source):
    """A re-plan that fails after its table was built publishes nothing: the
    earlier plan still binds its own table, with its own content, and runs
    correctly."""
    form = dict(pool=30, **_source(source, 8))
    a = _problem([7, 16, 3], [64, 30, 100], seed=260, **form)
    b = _problem([16, 2, 9], [77, 110, 17], seed=261, **form)
    attn, backend = _plan_cudnn(a, lse_mode="base2")
    table = backend._block_tables
    kept = table.clone()
    built = []

    def failing(meta, derived, block_tables, path, declared):
        built.append((declared, block_tables))
        raise RuntimeError("injected staging failure")

    with monkeypatch.context() as m:
        m.setattr(backend, "_stage_cudnn_plan", failing)
        with pytest.raises(RuntimeError, match="injected staging failure"):
            plan(attn, b, "cudnn", lse_mode="base2")
    # the failing plan got as far as a table of its own, in a's layout
    assert len(built) == 1
    declared, b_table = built[0]
    assert declared == (16, 128) and tuple(b_table.shape) == tuple(table.shape)
    assert b_table.stride() == table.stride()
    assert torch.equal(b_table, _own_pages_table(b, 8))
    assert not torch.equal(b_table, kept)
    assert backend._block_tables is table and torch.equal(table, kept)
    out, lse = run(attn, a)
    assert_matches(out, lse, a, lse_mode="base2")


# ---------------------------------------------------------------------------
# 5: LSE layouts, layouts, masks and scales with a rounded bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lse_mode", ["base2", "basee"])
@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize("layout", ["direct", "gather", "view"])
def test_lse_layouts_with_a_rounded_bound(monkeypatch, layout, source, lse_mode):
    """``direct``: ragged stats written into the caller's packed buffer, with
    a Q bound above the longest query.  ``gather``: the same batch with the
    packed-stats capability reported absent, so cuDNN writes padded stats --
    padded to the declared Q bound, not to the longest query -- and the
    backend gathers them.  ``view``: single-token requests keep the exact Q
    bound of 1 (their padded stats are the packed ones) under a rounded KV
    bound.  Each through a table of the class width, a narrower one and flat
    page ids."""
    dev = torch.device(DEVICE)
    form = _source(source, 16)
    if layout == "gather":
        monkeypatch.setitem(cudnn_backend._PACKED_LSE_SUPPORTED, dev, False)
    if layout == "view":
        p = _problem([1, 1, 1, 1], [200, 33, 90, 150], seed=270, **form)
        bounds = (1, 256)
    else:
        p = _problem(*_ROUNDED, seed=271, **form)
        bounds = _ROUNDED_BOUNDS
    attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
    assert _declared(backend) == (*bounds, 16)
    if _assert_staged(backend):
        _forbid_functional_entry(monkeypatch)
    if layout == "direct":
        # needs this cuDNN's ragged stats; without them the backend gathers
        assert backend._lse_path in ("direct", "gather")
    else:
        assert backend._lse_path == layout
    if backend._lse_path == "gather":
        assert tuple(backend._native_lse.shape) == (3, bounds[0], p["num_qo_heads"])
    lse_buf = torch.full(
        (p["q"].shape[0], p["num_qo_heads"]),
        float("nan"),
        dtype=torch.float32,
        device=dev,
    )
    for _ in range(2):
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), lse=lse_buf)
        assert lse is lse_buf
        assert_matches(out, lse, p, lse_mode=lse_mode)


@pytest.mark.parametrize("source", _SOURCES)
@pytest.mark.parametrize(
    "kv_layout,causal,scales",
    [("NHD", True, None), ("HND", False, None), ("NHD", False, _RUN_SCALES)],
    ids=["nhd", "noncausal", "nhd_noncausal_scales"],
)
def test_rounded_bounds_with_nhd_noncausal_and_run_time_scales(
    monkeypatch, kv_layout, causal, scales, source
):
    """Both bounds rounded with an NHD cache, without the causal mask, and
    with sm_scale / k_scale / v_scale given to run(), through a table of the
    class width, a narrower one and flat page ids."""
    scales = scales or {}
    p = _problem(*_ROUNDED, seed=272, kv_layout=kv_layout, **_source(source, 16))
    attn, backend = _plan_cudnn(p, causal=causal, lse_mode="base2")
    assert _declared(backend) == (*_ROUNDED_BOUNDS, 16)
    if _assert_staged(backend):
        _forbid_functional_entry(monkeypatch)
    for _ in range(2):
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]), **scales)
        _assert_rows_match(out, lse, p, lse_mode="base2", causal=causal, **scales)


# ---------------------------------------------------------------------------
# 6: CUDA-graph mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lse_mode", ["none", "base2"])
def test_cuda_graph_mode_declares_the_capacity(monkeypatch, lse_mode):
    """A bucket whose capacity is not a class: max_q_len 100 (an eager plan
    would declare 128) and max_kv_len 250 in a 16-column table (an eager
    plan would declare 256).  The captured graph must serve every later
    batch of the bucket, so plan() and update() declare the capacity
    whatever the batch, through the reserved table itself; one capture
    replays batches with other lengths and page ids."""
    capacity = GraphCapacity(
        batch_size=3,
        total_q_tokens=200,
        max_q_len=100,
        max_kv_len=250,
        page_size=PAGE,
        table_width=16,
    )
    batches = [
        _problem(q_lens, kv_lens, seed=280 + i, table_width=16, pool=40)
        for i, (q_lens, kv_lens) in enumerate(
            [
                ([100, 30, 64], [250, 90, 64]),
                ([20, 100, 5], [100, 249, 31]),
                ([64, 64, 64], [64, 128, 250]),
            ]
        )
    ]
    p1 = batches[0]
    eager, eager_backend = _plan_cudnn(p1, lse_mode=lse_mode)
    attn, backend = _plan_cudnn(p1, lse_mode=lse_mode, graph_capacity=capacity)
    staged = _assert_staged(backend)
    assert _declared(eager_backend) == (128, 256, 16)
    assert _declared(backend) == (100, 250, 16)
    reserved = backend._block_tables
    assert reserved.data_ptr() == attn._impl._graph.block_tables.data_ptr()

    buffers = _capture_buffers(p1, lse_mode)
    q, k, v, out, lse = buffers
    graph = _warm_and_capture(attn, q, (k, v), out, lse)
    preparations = _count_preparations(monkeypatch)
    _replay(graph, buffers)
    _assert_rows_match(out, lse, p1, lse_mode=lse_mode)
    for p in batches[1:]:
        assert not torch.equal(p["block_tables"], p1["block_tables"])
        _load(buffers, p)
        _update(attn, p)
        assert backend._block_tables is reserved
        assert torch.equal(reserved, p["block_tables"])  # the caller's columns
        assert _declared(backend) == (100, 250, 16)
        _replay(graph, buffers)
        _assert_rows_match(out, lse, p, lse_mode=lse_mode)
    if staged:
        assert not preparations


# ---------------------------------------------------------------------------
# 7: the functional fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lse_mode", ["none", "base2", "basee"])
@pytest.mark.parametrize("source", ["own_width", "flat"])
@pytest.mark.parametrize("packed_lse", [True, False], ids=["direct", "gather"])
def test_functional_fallback_declares_the_same_class(
    monkeypatch, packed_lse, source, lse_mode
):
    """Staging patched off: run() hands the functional entry the class the
    staged plan declares -- the bounds, the backend's table of the class
    width and, on the gather path, stats padded to the declared Q bound --
    so both paths fetch one cuDNN graph and both match the oracle.  The
    caller's table is narrower than the class, or derived from flat page
    ids."""
    _require_prepared()
    dev = torch.device(DEVICE)
    if not packed_lse:
        monkeypatch.setitem(cudnn_backend._PACKED_LSE_SUPPORTED, dev, False)
    p = _problem(*_ROUNDED, seed=290, **_source(source, 16))

    def fetch(functional):
        with monkeypatch.context() as m:
            if functional:
                _no_staged_plan(m)
            # plan before counting: the one-time packed-LSE probe of this
            # process fetches a toy graph through the entry
            attn, backend = _plan_cudnn(p, lse_mode=lse_mode)
            assert (backend._cudnn_plan is None) == functional
            fetched = _count_preparations(m)
            declared = []
            real = fi_cudnn.cudnn_batch_prefill_with_kv_cache

            def entry(*args, **kwargs):
                declared.append(
                    (
                        kwargs["max_token_per_sequence"],
                        kwargs["max_sequence_kv"],
                        kwargs["block_tables"].shape[1],
                    )
                )
                return real(*args, **kwargs)

            m.setattr(fi_cudnn, "cudnn_batch_prefill_with_kv_cache", entry)
            out, lse = run(attn, p)
            assert_matches(out, lse, p, lse_mode=lse_mode)
            assert len(fetched) == 1
            assert _declared(backend) == (*_ROUNDED_BOUNDS, 16)
            assert declared == ([(*_ROUNDED_BOUNDS, 16)] if functional else [])
            return fetched[0], out, lse

    prepared, out, lse = fetch(False)
    functional, out_functional, lse_functional = fetch(True)
    assert functional.key == prepared.key
    assert functional.graph is prepared.graph
    assert torch.equal(out_functional, out)
    if lse_mode != "none":
        assert torch.equal(lse_functional, lse)
