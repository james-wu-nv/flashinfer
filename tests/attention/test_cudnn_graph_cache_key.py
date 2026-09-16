"""Regression tests: the cuDNN graph-cache keys must include everything the
built graph bakes in — attn scale, the page table's shape and strides — and
a build that fails must leave nothing in the cache.

The cuDNN SDPA graph bakes ``attn_scale`` in as a compile-time constant, but
``_sdpa_prefill_key_fn`` did not include it in the process-global graph-cache
key.  A second call with identical shapes and a *different* ``scale``
therefore silently replayed the first call's graph and computed attention
with the stale scale (observed on H100: output error vs the correct
reference 2.56, vs the stale-scale reference 0.0037 — an exact stale
replay).  Same-shape different-scale calls are routine in serving (per-layer
logit scaling, muP, model switching), so this is a silent-wrong-results bug,
not a perf detail.

This test runs the same shape twice with two scales and checks each result
against an independent torch reference; without the key fix the second
iteration fails.
"""

import math

import pytest
import torch

from flashinfer.cudnn import (
    cudnn_batch_decode_with_kv_cache,
    cudnn_batch_prefill_with_kv_cache,
)
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.utils import get_compute_capability


def _skip_if_unsupported(device):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    major, _ = get_compute_capability(torch.device(device))
    if major < 8:
        pytest.skip("cuDNN SDPA requires SM80+")


def _reference(q, k, v, q_lens, kv_lens, scale, causal):
    outs = []
    qo_off = 0
    kv_off = 0
    for lq, lkv in zip(q_lens.tolist(), kv_lens.tolist(), strict=True):
        q_i = q[qo_off : qo_off + lq].float()  # (lq, H, D)
        k_i = k[kv_off : kv_off + lkv].float()
        v_i = v[kv_off : kv_off + lkv].float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if causal:
            qpos = torch.arange(lq, device=q.device).unsqueeze(1)
            kpos = torch.arange(lkv, device=q.device).unsqueeze(0)
            allowed = kpos <= (lkv - lq) + qpos
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v_i))
        qo_off += lq
        kv_off += lkv
    return torch.cat(outs)


def test_cudnn_prefill_scale_in_graph_cache_key():
    device = "cuda:0"
    _skip_if_unsupported(device)

    torch.manual_seed(0)
    batch_size, num_qo_heads, num_kv_heads, head_dim = 2, 4, 4, 128
    q_lens = torch.tensor([32, 32], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([48, 48], dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(kv_lens, 0)]).int()

    q = torch.randn(
        int(q_lens.sum()), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        int(kv_lens.sum()), num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Same shapes twice, different scales: without `scale` in the cache key
    # the second iteration silently reuses the first graph (stale scale).
    for scale_mult in (1.0, 3.0):
        scale = scale_mult / math.sqrt(head_dim)
        out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            workspace,
            max_token_per_sequence=32,
            max_sequence_kv=48,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            causal=True,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
        )
        ref = _reference(q, k, v, q_lens, kv_lens, scale, causal=True)
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, s=scale, sm=scale_mult: (
                f"scale={s} (mult {sm}): stale-scale graph replay?\n{m}"
            ),
        )


def test_cudnn_decode_scale_in_graph_cache_key():
    """Decode analog: the decode key omitted scale (and keyed only on
    (max_sequence_kv, q.shape, k_cache.shape)); same shapes with a different
    scale silently replayed the stale graph."""
    device = "cuda:0"
    _skip_if_unsupported(device)

    torch.manual_seed(0)
    batch_size, num_heads, head_dim, page_size = 2, 4, 128, 16
    kv_len = 48
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq

    q = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_cache = torch.randn(
        total_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch_size, pages_per_seq
    )
    seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), kv_len, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def decode_reference(scale):
        outs = []
        for i in range(batch_size):
            pages = block_tables[i].to(torch.int64)
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
            scores = torch.einsum("hd,hkd->hk", q[i].float(), k_i) * scale
            p = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("hk,hkd->hd", p, v_i))
        return torch.stack(outs)

    for scale_mult in (1.0, 3.0):
        scale = scale_mult / math.sqrt(head_dim)
        out = cudnn_batch_decode_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_sequence_kv=kv_len,
            actual_seq_lens_kv=seq_lens_kv,
            block_tables=block_tables,
        )
        torch.testing.assert_close(
            out.float(),
            decode_reference(scale),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, s=scale, sm=scale_mult: (
                f"decode scale={s} (mult {sm}): stale-scale graph replay?\n{m}"
            ),
        )


@pytest.mark.parametrize("order", ["view_then_contiguous", "contiguous_then_view"])
def test_cudnn_prefill_block_table_strides_in_graph_cache_key(order):
    """The paged graph binds block_tables with tensor_like(), so its strides
    are baked in, but the key held only ``block_tables is not None``.  A graph
    built for a column view of a wider table (row stride > width) replayed on
    a contiguous table of the same shape — or the reverse — read the wrong
    pages (71.7% wrong elements, found by the PagedAttention fuzzer)."""
    device = "cuda:0"
    _skip_if_unsupported(device)

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
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def reference():
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
        return torch.cat(outs)

    ref = reference()
    tables = (view, packed) if order == "view_then_contiguous" else (packed, view)
    for table in tables:
        strides = tuple(table.stride())
        out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k_cache,
            v_cache,
            1.0 / math.sqrt(head_dim),
            workspace,
            max_token_per_sequence=q_len,
            max_sequence_kv=kv_len,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            block_tables=table,
            causal=True,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_units="tokens",
        )
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, st=strides: (
                f"block_tables strides {st}: stale-stride graph replay?\n{m}"
            ),
        )


def _paged_problem(
    device, *, batch_size, width, page_size=16, num_heads=4, head_dim=128
):
    """Exact-width paged problem: every request has kv_len == width * page_size
    (so ceil(kv/page) == width) and q_len 32; random page permutation."""
    kv_len, q_len = width * page_size, 32
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
    perm = torch.randperm(pool_pages, dtype=torch.int32, device=device)
    block_tables = perm[: batch_size * width].view(batch_size, width).contiguous()

    def reference():
        outs = []
        for i in range(batch_size):
            pages = block_tables[i].to(torch.int64)
            k_i = k_cache[pages].permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)
            v_i = v_cache[pages].permute(1, 0, 2, 3).reshape(num_heads, -1, head_dim)
            q_i = q[i * q_len : (i + 1) * q_len].float()
            scores = torch.einsum("qhd,hkd->hqk", q_i, k_i.float()) / math.sqrt(
                head_dim
            )
            qpos = torch.arange(q_len, device=device).unsqueeze(1)
            kpos = torch.arange(kv_len, device=device).unsqueeze(0)
            allowed = kpos <= (kv_len - q_len) + qpos
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
            outs.append(
                torch.einsum("hqk,hkd->qhd", torch.softmax(scores, -1), v_i.float())
            )
        return torch.cat(outs)

    return dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        q_lens=q_lens,
        kv_lens=kv_lens,
        qo_indptr=qo_indptr,
        block_tables=block_tables,
        q_len=q_len,
        kv_len=kv_len,
        head_dim=head_dim,
        batch_size=batch_size,
        reference=reference,
    )


def _paged_call(p, workspace, *, block_tables=None, max_sequence_kv=None):
    b = p["batch_size"]
    return cudnn_batch_prefill_with_kv_cache(
        p["q"],
        p["k_cache"],
        p["v_cache"],
        1.0 / math.sqrt(p["head_dim"]),
        workspace,
        max_token_per_sequence=p["q_len"],
        max_sequence_kv=p["kv_len"] if max_sequence_kv is None else max_sequence_kv,
        actual_seq_lens_q=p["q_lens"].view(b, 1, 1, 1),
        actual_seq_lens_kv=p["kv_lens"].view(b, 1, 1, 1),
        block_tables=p["block_tables"] if block_tables is None else block_tables,
        causal=True,
        return_lse=True,
        batch_offsets_q=p["qo_indptr"],
        batch_offsets_units="tokens",
    )


def test_cudnn_prefill_table_width_in_graph_cache_key():
    """Tables of width 64 and 67 (each exact for its own max_kv) build two
    distinct cache entries and are both correct; a width-67 table handed to
    the width-64 max_kv must not replay the width-64 graph (the old key, which
    ignored the table's shape, returned "ok" with a mismatched descriptor) —
    cuDNN's finalize rejects it instead.  Found by the PagedAttention
    benchmark smoke run."""
    device = "cuda:0"
    _skip_if_unsupported(device)
    torch.manual_seed(0)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    p64 = _paged_problem(device, batch_size=2, width=64)
    p67 = _paged_problem(device, batch_size=2, width=67)
    keys = set()
    for p in (p64, p67):
        b = p["batch_size"]
        keys.add(
            cudnn_prefill._sdpa_prefill_key_fn(
                p["q"],
                p["k_cache"],
                p["v_cache"],
                1.0 / math.sqrt(p["head_dim"]),
                max_token_seq_q=p["q_len"],
                max_sequence_kv=p["kv_len"],
                actual_seq_lens_q=p["q_lens"].view(b, 1, 1, 1),
                actual_seq_lens_kv=p["kv_lens"].view(b, 1, 1, 1),
                block_tables=p["block_tables"],
                batch_offsets_q=p["qo_indptr"],
                bottom_right_causal_mask=True,
                return_lse=True,
            )
        )
        out, _ = _paged_call(p, workspace)
        torch.testing.assert_close(out.float(), p["reference"](), atol=2e-2, rtol=2e-2)
    assert len(keys) == 2, "width-64 and width-67 tables must not share a cache entry"
    # width 67 with the width-64 problem's max_kv: reject, never replay
    with pytest.raises(RuntimeError, match="page table"):
        _paged_call(p64, workspace, block_tables=p67["block_tables"])
        torch.cuda.synchronize()


def test_cudnn_prefill_failed_build_leaves_no_cache_entry():
    """A build that fails at finalize (over-wide page table -> BAD_PARAM) must
    not leave a half-built graph in the cache: retrying the same call must
    raise the same cuDNN error (not "attn_scale with tensor and value cannot
    be set at the same time" from re-building the stale object), and a valid
    same-process call must stay correct.  Found by the PagedAttention
    benchmark smoke run."""
    device = "cuda:0"
    _skip_if_unsupported(device)
    torch.manual_seed(1)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    p = _paged_problem(device, batch_size=3, width=8)
    ref = p["reference"]()
    out, _ = _paged_call(p, workspace)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    wide = torch.zeros(p["batch_size"], 8 + 3, dtype=torch.int32, device=device)
    wide[:, :8] = p["block_tables"]
    for _ in range(2):  # the second attempt must fail the same way
        with pytest.raises(RuntimeError, match="page table"):
            _paged_call(p, workspace, block_tables=wide)
            torch.cuda.synchronize()
    out, _ = _paged_call(p, workspace)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
