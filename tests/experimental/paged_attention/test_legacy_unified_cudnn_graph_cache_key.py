"""Legacy -> unified: tests/attention/test_cudnn_graph_cache_key.py

The legacy prefill tests are regressions of the process-global cuDNN graph
cache behind ``cudnn_batch_prefill_with_kv_cache``: a key missing the attn
scale, the page table's strides or its width replays a stale graph.
``PagedAttention`` pinned to cudnn drives the same cache, so each test runs
the legacy fixture (same seed, same tensors, so the node ids equal the legacy
ids) through it and a stale replay shows up as a wrong answer against the
legacy torch reference (at the legacy tolerance) and the fp32 paged-attention
oracle (output and LSE).

Partial rows:
- scale: the legacy call is ragged; its K/V storage is viewed as page-16
  pages with an identity block table (no copy).
- table width / failed build: the facade views a wider table to the exact
  width ``ceil(max_kv / page)`` before binding it, so the legacy trigger (an
  over-wide table rejected by cuDNN's finalize) does not exist here: the
  wider table is asserted correct instead, and the failed-build case uses a
  table the cudnn plan rejects (page dimension not unit-stride).  The private
  ``_sdpa_prefill_key_fn`` distinctness check stays native-only.

The legacy decode tests (``cudnn_batch_decode_with_kv_cache``) are out of
scope: ``PagedAttention`` is the paged-prefill API and has no decode route.
"""

import math

import pytest
import torch

from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata
from flashinfer.utils import get_compute_capability
from tests.attention.test_cudnn_graph_cache_key import _paged_problem

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cudnn_graph_cache_key.py"
LEGACY_MAP = [
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_scale_in_graph_cache_key",
        [],
        "out-of-scope",
        "ragged legacy fixture (no page table); not paged",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_scale_in_graph_cache_key",
        [],
        "out-of-scope",
        "decode entry (cudnn_batch_decode_with_kv_cache); PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_block_table_strides_in_graph_cache_key",
        ["test_cudnn_prefill_block_table_strides_in_graph_cache_key"],
        "equivalent",
        "same fixture (column view of a capacity table vs its contiguous copy, both "
        "orders) on cudnn; legacy torch reference (2e-2) plus the fp32 oracle (output "
        "and LSE)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_table_width_in_graph_cache_key",
        ["test_cudnn_prefill_table_width_in_graph_cache_key"],
        "partial",
        "width-64 and width-67 problems both correct on cudnn in one process; the "
        "width-64 problem on a capacity-67 table is viewed to the exact width, so the "
        "legacy rejection becomes a correct result; the private key-fn check is "
        "native-only",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_prefill_failed_build_leaves_no_cache_entry",
        ["test_cudnn_prefill_failed_build_leaves_no_cache_entry"],
        "partial",
        "the over-wide table no longer fails a cuDNN build through the facade; a table "
        "with a non-unit-stride page dimension is rejected at plan twice with the same "
        "message, the previous plan stays correct and a valid re-plan is correct "
        "(legacy reference 2e-2 plus the fp32 oracle)",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_block_table_strides_in_graph_cache_key",
        [],
        "out-of-scope",
        "decode entry (cudnn_batch_decode_with_kv_cache); PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_table_width_in_graph_cache_key",
        [],
        "out-of-scope",
        "decode entry (cudnn_batch_decode_with_kv_cache); PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_cudnn_graph_cache_key.py::test_cudnn_decode_failed_build_leaves_no_cache_entry",
        [],
        "out-of-scope",
        "decode entry (cudnn_batch_decode_with_kv_cache); PagedAttention has no decode route",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


def _skip_if_unsupported(device):
    # the legacy gate
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    major, _ = get_compute_capability(torch.device(device))
    if major < 8:
        pytest.skip("cuDNN SDPA requires SM80+")


def _metadata(qo_indptr, kv_lens, block_tables, page_size):
    """Dense metadata with host mirrors; max lengths from the batch."""
    qo_indptr_cpu = qo_indptr.cpu()
    kv_lens_cpu = kv_lens.cpu()
    return PagedAttentionMetadata.dense(
        qo_indptr,
        kv_lens,
        block_tables,
        page_size=page_size,
        max_q_len=int((qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).max()),
        max_kv_len=int(kv_lens_cpu.max()),
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )


def _oracle(md, q, k_cache, v_cache, kv_layout="HND", sm_scale=None):
    return reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        md.qo_indptr_cpu,
        md.kv_seq_lens_cpu,
        md.block_tables,
        md.page_size,
        True,
        sm_scale=sm_scale,
        kv_layout=kv_layout,
    )


def _plan_cudnn(md, num_heads, head_dim, kv_layout="HND", attn=None):
    attn = attn or PagedAttention(torch.device("cuda:0"))
    attn.plan(
        md,
        num_qo_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim_qk=head_dim,
        q_dtype=torch.bfloat16,
        kv_layout=kv_layout,
        causal=True,
        lse_mode="base2",
        backend="cudnn",
    )
    assert attn.backend == "cudnn"
    return attn


@pytest.mark.parametrize("order", ["view_then_contiguous", "contiguous_then_view"])
def test_cudnn_prefill_block_table_strides_in_graph_cache_key(order):
    device = "cuda:0"
    _skip_if_unsupported(device)

    # the legacy fixture, verbatim
    torch.manual_seed(0)
    batch_size, num_heads, head_dim, page_size = 3, 4, 128, 16
    kv_len, q_len = 120, 32
    width = (kv_len + page_size - 1) // page_size
    pool_pages = batch_size * width + 8
    q_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    q = torch.randn(
        batch_size * q_len, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_cache = torch.randn(
        pool_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    # a capacity-width table (3 spare columns) whose live prefix is a random
    # page permutation; the narrow view keeps its row stride width + 3
    perm = torch.randperm(pool_pages, dtype=torch.int32, device=device)
    wide = torch.zeros(batch_size, width + 3, dtype=torch.int32, device=device)
    wide[:, :width] = perm[: batch_size * width].view(batch_size, width)
    view = wide[:, :width]
    packed = view.contiguous()
    assert view.stride(0) == width + 3 and packed.stride(0) == width

    # the legacy torch reference
    outs = []
    for i in range(batch_size):
        pages = packed[i].to(torch.int64)
        k_i = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads, -1, head_dim)[:, :kv_len]
            .float()
        )
        v_i = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads, -1, head_dim)[:, :kv_len]
            .float()
        )
        q_i = q[i * q_len : (i + 1) * q_len].float()
        scores = torch.einsum("qhd,hkd->hqk", q_i, k_i) / math.sqrt(head_dim)
        qpos = torch.arange(q_len, device=device).unsqueeze(1)
        kpos = torch.arange(kv_len, device=device).unsqueeze(0)
        allowed = kpos <= (kv_len - q_len) + qpos
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        outs.append(torch.einsum("hqk,hkd->qhd", torch.softmax(scores, -1), v_i))
    ref = torch.cat(outs)

    tables = (view, packed) if order == "view_then_contiguous" else (packed, view)
    for table in tables:
        strides = tuple(table.stride())
        md = _metadata(qo_indptr, kv_lens, table, page_size)
        attn = _plan_cudnn(md, num_heads, head_dim)
        out, lse = attn.run(q, (k_cache, v_cache))
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, st=strides: (
                f"block_tables strides {st}: stale-stride graph replay?\n{m}"
            ),
        )
        ref_out, ref_lse = _oracle(md, q, k_cache, v_cache)
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_cudnn_prefill_table_width_in_graph_cache_key():
    device = "cuda:0"
    _skip_if_unsupported(device)
    torch.manual_seed(0)
    p64 = _paged_problem(device, batch_size=2, width=64)
    p67 = _paged_problem(device, batch_size=2, width=67)
    # the width-64 problem on a capacity-67 table: the legacy native call
    # rejects it, the facade views it to the exact width
    wide = torch.zeros(2, 67, dtype=torch.int32, device=device)
    wide[:, :64] = p64["block_tables"]
    cases = [
        (p64, p64["block_tables"], "width-64 exact"),
        (p67, p67["block_tables"], "width-67 exact"),
        (p64, wide, "width-64 problem on a capacity-67 table"),
        (p64, p64["block_tables"], "width-64 exact again"),
    ]
    for p, table, what in cases:
        md = _metadata(p["qo_indptr"], p["kv_lens"], table, 16)
        attn = _plan_cudnn(md, 4, p["head_dim"])
        out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
        torch.testing.assert_close(
            out.float(),
            p["reference"](),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, w=what: f"{w}: stale-width graph replay?\n{m}",
        )
        ref_out, ref_lse = _oracle(md, p["q"], p["k_cache"], p["v_cache"])
        torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)


def test_cudnn_prefill_failed_build_leaves_no_cache_entry():
    device = "cuda:0"
    _skip_if_unsupported(device)
    torch.manual_seed(1)
    p = _paged_problem(device, batch_size=3, width=8)
    ref = p["reference"]()
    md = _metadata(p["qo_indptr"], p["kv_lens"], p["block_tables"], 16)
    attn = _plan_cudnn(md, 4, p["head_dim"])
    out, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)

    # the same (3, 8) table values with a page-dimension stride of 3: the
    # cudnn plan rejects it, twice with the same message
    transposed = p["block_tables"].t().contiguous().t()
    assert transposed.stride(1) == 3 and torch.equal(transposed, p["block_tables"])
    bad_md = _metadata(p["qo_indptr"], p["kv_lens"], transposed, 16)
    messages = []
    for _ in range(2):
        with pytest.raises(ValueError, match="unit-stride along the page") as info:
            _plan_cudnn(bad_md, 4, p["head_dim"], attn=attn)
        messages.append(str(info.value))
    assert messages[0] == messages[1]

    # the previous plan is still published and correct, and so is a re-plan
    assert attn.backend == "cudnn"
    out, _ = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    attn = _plan_cudnn(md, 4, p["head_dim"], attn=attn)
    out, lse = attn.run(p["q"], (p["k_cache"], p["v_cache"]))
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    ref_out, ref_lse = _oracle(md, p["q"], p["k_cache"], p["v_cache"])
    torch.testing.assert_close(out.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
