"""Legacy -> unified: tests/attention/test_shared_prefix_kernels.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention: the fp16
NHD pool filled with ``append_paged_kv_cache`` (shared pages first, then the
per-request unique pages), the decode / append stages, the legacy CSR of the
two cascade levels mapped losslessly into (a) ONE paged attention per request
over ``shared pages ++ unique pages`` (the same physical shared pages appear in
every request's page list) and (b) the two-level composition: one
``PagedAttention`` run over the shared pages, one over the unique pages, merged
with ``flashinfer.merge_state`` on the base-2 LSE.  Both are compared with the
legacy reference, ``MultiLevelCascadeAttentionWrapper`` (2 levels), at the
legacy rtol/atol 1e-3, and with the fp32 oracle of the one-shot batch.  The
parametrize axes keep the legacy names and values; the row id is the legacy
node id plus ``-<backend>``; the full 192-point grid is under ``slow``.

CI: legacy file only in the H100 1/5 sampling lane.  The unified file is not
collected by default CI (``norecursedirs``).

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- the cascade (multi-level) inference is not a PagedAttention feature: the
  unified API has no merge entry point and no level structure; the
  composition above is external (two plans + ``merge_state``), so the row is
  ``partial`` (numerics of every legacy point covered, the cascade operator
  itself not).  Page 1 rows resolve on the CSR-native backends only; D256
  rows skip on cudnn / trtllm-gen / cake.
- ``test_merge_state_in_place_with_mask`` is a merge-operator contract with
  no unified counterpart (``native-only``); the legacy operator test stays.
"""

import inspect

import pytest
import torch

import flashinfer
from flashinfer.prefill import PagedAttention

from .legacy_unified_helpers import (
    DEVICE,
    LegacyBatch,
    argnames,
    assert_oracle,
    check_legacy_map,
    check_legacy_map_complete,
    check_unified_tests_mapped,
    param_rows,
    run_batch,
    seed_of,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_shared_prefix_kernels.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_shared_prefix_kernels.py::test_batch_attention_with_shared_prefix_paged_kv_cache",
        ["test_batch_attention_with_shared_prefix_paged_kv_cache"],
        "partial",
        "same 8 legacy axes (decode/append, B12/17, unique37/17, shared128/512/2048, "
        "H8/16, non-causal, D128/256, page1/16) and the append_paged_kv_cache pool; "
        "one-shot paged attention over shared ++ unique pages AND the two-level "
        "composition (two PagedAttention runs + merge_state on base-2 LSE), both vs "
        "MultiLevelCascadeAttentionWrapper at 1e-3 + oracle; the cascade operator itself "
        "is an external composition, not a PagedAttention feature.",
    ),
    (
        "tests/attention/test_shared_prefix_kernels.py::test_merge_state_in_place_with_mask",
        ["test_merge_state_in_place_with_mask"],
        "native-only",
        "merge-operator contract (seed 0, 50 tries); PagedAttention has no merge entry "
        "point and the legacy operator test is retained.",
    ),
]

SHARED_PREFIX_AXES = dict(
    stage=["decode", "append"],
    batch_size=[12, 17],
    unique_kv_len=[37, 17],
    shared_kv_len=[128, 512, 2048],
    num_heads=[8, 16],
    causal=[False],
    head_dim=[128, 256],
    page_size=[1, 16],
)


def _shared_default(p):
    stage, b, unique, shared, heads, causal, hd, page = p
    return (b, unique, shared, heads) in {
        (12, 37, 128, 8),
        (17, 17, 512, 16),
        (12, 17, 2048, 8),
    }


def _ceil_div(a, b):
    return (a + b - 1) // b


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)
    check_unified_tests_mapped(LEGACY_MAP, globals())


def _shared_prefix_fixture(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    head_dim,
    page_size,
    seed,
):
    """Verbatim legacy construction: q, k/v_shared, k/v_unique, and the pool
    filled with ``append_paged_kv_cache`` (shared pages first)."""
    torch.manual_seed(seed)
    dev = torch.device(DEVICE)
    kv_layout = "NHD"
    if stage == "append":
        q = torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
        q_indptr = torch.arange(0, batch_size + 1).to(dev).int() * unique_kv_len
    else:
        q = torch.randn(batch_size, num_heads, head_dim).to(dev).half()
        q_indptr = torch.arange(0, batch_size + 1).to(dev).int()
    k_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(dev).half()
    v_shared = torch.randn(shared_kv_len, num_heads, head_dim).to(dev).half()
    k_unique = (
        torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
    )
    v_unique = (
        torch.randn(batch_size * unique_kv_len, num_heads, head_dim).to(dev).half()
    )
    n_shared_pages = _ceil_div(shared_kv_len, page_size)
    n_unique_pages = _ceil_div(unique_kv_len, page_size)
    kv_data = (
        torch.zeros(
            n_shared_pages + batch_size * n_unique_pages,
            2,
            page_size,
            num_heads,
            head_dim,
        )
        .to(dev)
        .half()
    )
    shared_kv_indices = torch.arange(0, n_shared_pages).to(dev).int()
    shared_append_indptr = torch.arange(0, 2).to(dev).int() * shared_kv_len
    shared_kv_indptr = torch.arange(0, 2).to(dev).int() * n_shared_pages
    shared_last_page_len = torch.full(
        (1,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(dev)
    flashinfer.append_paged_kv_cache(
        k_shared,
        v_shared,
        *flashinfer.get_batch_indices_positions(
            shared_append_indptr,
            flashinfer.get_seq_lens(shared_kv_indptr, shared_last_page_len, page_size),
            k_shared.shape[0],
        ),
        kv_data,
        shared_kv_indices,
        shared_kv_indptr,
        shared_last_page_len,
        kv_layout,
    )
    unique_kv_indices = (
        torch.arange(0, batch_size * n_unique_pages).to(dev).int() + n_shared_pages
    )
    unique_append_indptr = torch.arange(0, batch_size + 1).to(dev).int() * unique_kv_len
    unique_kv_indptr = torch.arange(0, batch_size + 1).to(dev).int() * n_unique_pages
    unique_last_page_len = torch.full(
        (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(dev)
    flashinfer.append_paged_kv_cache(
        k_unique,
        v_unique,
        *flashinfer.get_batch_indices_positions(
            unique_append_indptr,
            flashinfer.get_seq_lens(unique_kv_indptr, unique_last_page_len, page_size),
            k_unique.shape[0],
        ),
        kv_data,
        unique_kv_indices,
        unique_kv_indptr,
        unique_last_page_len,
        kv_layout,
    )
    return dict(
        q=q,
        q_indptr=q_indptr,
        kv_data=kv_data,
        shared=(shared_kv_indptr, shared_kv_indices, shared_last_page_len),
        unique=(unique_kv_indptr, unique_kv_indices, unique_last_page_len),
        n_shared_pages=n_shared_pages,
        n_unique_pages=n_unique_pages,
        kv_layout=kv_layout,
    )


def _legacy_cascade_reference(f, *, num_heads, head_dim, page_size, stage, causal):
    """The legacy reference: ``MultiLevelCascadeAttentionWrapper`` (2 levels)."""
    dev = torch.device(DEVICE)
    wrapper = flashinfer.MultiLevelCascadeAttentionWrapper(
        2, torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(dev), f["kv_layout"]
    )
    qo_indptr_top = torch.tensor([0, f["q"].shape[0]], dtype=torch.int32).to(dev)
    kw = {} if stage == "decode" else dict(causal=causal)
    wrapper.plan(
        [qo_indptr_top, f["q_indptr"]],
        [f["shared"][0], f["unique"][0]],
        [f["shared"][1], f["unique"][1]],
        [f["shared"][2], f["unique"][2]],
        num_heads,
        num_heads,
        head_dim,
        page_size,
        **kw,
    )
    return wrapper.run(f["q"], f["kv_data"])


def _shared_prefix_batches(f, *, batch_size, shared_kv_len, unique_kv_len, page_size):
    """Three legacy-shaped batches over the SAME pool: the one-shot
    (shared pages ++ unique pages per request), the shared-only and the
    unique-only segment (for the composed check)."""
    n_s, n_u = f["n_shared_pages"], f["n_unique_pages"]
    q_indptr_cpu = f["q_indptr"].cpu()
    k, v = f["kv_data"][:, 0], f["kv_data"][:, 1]
    shared_ids = torch.arange(0, n_s, dtype=torch.int32)
    full_ids, full_indptr = [], [0]
    for i in range(batch_size):
        full_ids.append(shared_ids)
        full_ids.append(
            torch.arange(n_s + i * n_u, n_s + (i + 1) * n_u, dtype=torch.int32)
        )
        full_indptr.append(full_indptr[-1] + n_s + n_u)
    common = dict(
        q=f["q"],
        k=k,
        v=v,
        q_indptr_cpu=q_indptr_cpu,
        page_size=page_size,
        kv_layout=f["kv_layout"],
    )
    one_shot = LegacyBatch(
        kv_indptr_cpu=torch.tensor(full_indptr, dtype=torch.int32),
        kv_indices_cpu=torch.cat(full_ids),
        last_page_len_cpu=torch.full(
            (batch_size,), (unique_kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        **common,
    )
    shared_only = LegacyBatch(
        kv_indptr_cpu=torch.arange(0, batch_size + 1, dtype=torch.int32) * n_s,
        kv_indices_cpu=shared_ids.repeat(batch_size),
        last_page_len_cpu=torch.full(
            (batch_size,), (shared_kv_len - 1) % page_size + 1, dtype=torch.int32
        ),
        **common,
    )
    unique_only = LegacyBatch(
        kv_indptr_cpu=f["unique"][0].cpu(),
        kv_indices_cpu=f["unique"][1].cpu(),
        last_page_len_cpu=f["unique"][2].cpu(),
        **common,
    )
    return one_shot, shared_only, unique_only


@pytest.mark.parametrize(
    argnames(SHARED_PREFIX_AXES, "backend"),
    param_rows(SHARED_PREFIX_AXES, _shared_default),
)
def test_batch_attention_with_shared_prefix_paged_kv_cache(
    stage,
    batch_size,
    unique_kv_len,
    shared_kv_len,
    num_heads,
    causal,
    head_dim,
    page_size,
    backend,
):
    """(a) ONE paged attention over ``shared pages ++ unique pages`` per
    request and (b) the legacy two-level path composed from the unified API
    (two ``PagedAttention`` runs, ``lse_mode="base2"`` as ``merge_state``
    consumes, merged with ``flashinfer.merge_state``), both vs the legacy
    ``MultiLevelCascadeAttentionWrapper`` at 1e-3 and vs the oracle."""
    f = _shared_prefix_fixture(
        stage,
        batch_size,
        unique_kv_len,
        shared_kv_len,
        num_heads,
        head_dim,
        page_size,
        seed=seed_of(
            "shared",
            stage,
            batch_size,
            unique_kv_len,
            shared_kv_len,
            num_heads,
            head_dim,
            page_size,
        ),
    )
    one_shot, shared_only, unique_only = _shared_prefix_batches(
        f,
        batch_size=batch_size,
        shared_kv_len=shared_kv_len,
        unique_kv_len=unique_kv_len,
        page_size=page_size,
    )
    ref = _legacy_cascade_reference(
        f,
        num_heads=num_heads,
        head_dim=head_dim,
        page_size=page_size,
        stage=stage,
        causal=causal,
    )
    # (a) one shot
    _, out, lse = run_batch(one_shot, one_shot.metadata(), backend, causal=causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(one_shot, out, lse, causal=causal)
    # (b) composed: shared level (non-causal) + unique level, merged
    _, o_s, s_s = run_batch(shared_only, shared_only.metadata(), backend, causal=False)
    _, o_u, s_u = run_batch(unique_only, unique_only.metadata(), backend, causal=causal)
    out_c, lse_c = flashinfer.merge_state(o_s, s_s, o_u, s_u)
    torch.testing.assert_close(out_c, ref, rtol=1e-3, atol=1e-3)  # legacy
    assert_oracle(one_shot, out_c, lse_c, causal=causal)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("num_tries", [50])
def test_merge_state_in_place_with_mask(seed, num_tries):
    """A merge-operator contract: the unified API has no merge entry point
    (composition uses the cascade operators, see the test above) and the
    legacy operator test stays as is."""
    assert not hasattr(PagedAttention, "merge")
    assert callable(flashinfer.merge_state_in_place) and callable(
        flashinfer.merge_state
    )
    params = inspect.signature(PagedAttention.run).parameters
    assert "lse" in params and "sinks" in params and "merge" not in params
