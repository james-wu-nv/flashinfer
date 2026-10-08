"""Legacy -> unified: tests/attention/test_batch_invariant_fa2.py

``test_batch_prefill_tensor_cores`` runs the legacy fixture (fp16 combined
pool ``/ 10`` as ``K = kv[:, 0]`` / ``V = kv[:, 1]`` views, HND and NHD) through
``PagedAttention`` pinned to fa2, the backend the legacy wrapper picks on this
GPU, at batch B and at its first ``invariant_bs`` requests.  Same grid and
tensors, so the node ids equal the legacy ids.

The legacy test pins batch invariance with ``plan(fixed_split_size=,
disable_split_kv=)``; ``plan()`` has neither knob, and the split-KV schedule
is the backend's own (no batch-invariance contract).  Each case asserts the
knobs are rejected, then runs both batches without them and checks each
against the fp32 paged-attention oracle (output and LSE); the legacy bitwise
comparison is an xfail when the two batches differ.  The ``ROPE_LLAMA`` rows
assert that ``plan()`` rejects the fused positional encoding and stop.

The legacy decode test (``BatchDecodeWithPagedKVCacheWrapper``) is out of
scope: ``PagedAttention`` is the paged-prefill API and has no decode route.
"""

import pytest
import torch

from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_invariant_fa2.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_decode_tensor_cores",
        [],
        "out-of-scope",
        "decode entry (BatchDecodeWithPagedKVCacheWrapper, use_tensor_cores); "
        "PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_batch_invariant_fa2.py::test_batch_prefill_tensor_cores",
        ["test_batch_prefill_tensor_cores"],
        "unsupported-by-design",
        "same grid and tensors on fa2; plan() rejects fixed_split_size / "
        "disable_split_kv and ROPE_LLAMA (asserted); both batches checked against "
        "the fp32 oracle; no batch-invariance contract, so the legacy bitwise "
        "check is an xfail when the batches differ",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)
LSE_TOL = dict(atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("invariant_bs", [2])
@pytest.mark.parametrize("kv_len", [4096, 5000])
@pytest.mark.parametrize("qo_len", [128, 256])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    qo_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    # the legacy fixture, verbatim (same RNG order: q, kv_data)
    num_qo_heads = num_kv_heads * group_size
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
        if kv_layout == "HND"
        else torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
    )
    k_cache, v_cache = kv_data[:, 0], kv_data[:, 1]
    kv_indices = torch.arange(total_num_pages, dtype=torch.int32, device="cuda:0")

    def plan_and_run(bs, **legacy_kwargs):
        qo_indptr_cpu = torch.arange(bs + 1, dtype=torch.int32) * qo_len
        kv_lens_cpu = torch.full((bs,), kv_len, dtype=torch.int32)
        md = PagedAttentionMetadata.csr(
            qo_indptr_cpu.to("cuda:0"),
            kv_lens_cpu.to("cuda:0"),
            kv_indices[: bs * num_pages_per_seq],
            page_size=page_size,
            max_q_len=qo_len,
            max_kv_len=kv_len,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_seq_lens_cpu=kv_lens_cpu,
        )
        attn = PagedAttention(torch.device("cuda:0"))
        attn.plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.float16,
            kv_layout=kv_layout,
            causal=False,
            lse_mode="base2",
            backend="fa2",
            **legacy_kwargs,
        )
        assert attn.backend == "fa2"
        o, lse = attn.run(q[: bs * qo_len], (k_cache, v_cache))

        # fp32 oracle: output and base-2 LSE
        ref_out, ref_lse = reference_paged_prefill(
            q[: bs * qo_len],
            k_cache,
            v_cache,
            qo_indptr_cpu,
            kv_lens_cpu,
            None,
            page_size,
            False,
            kv_layout=kv_layout,
            kv_page_indices=md.kv_page_indices,
        )
        torch.testing.assert_close(o.float(), ref_out, **OUT_TOL)
        torch.testing.assert_close(lse, ref_lse, **LSE_TOL)
        return o, lse

    # the legacy batch-invariance knobs are not plan() arguments
    with pytest.raises(TypeError, match="fixed_split_size"):
        plan_and_run(
            batch_size,
            fixed_split_size=fixed_split_size if not disable_split_kv else None,
            disable_split_kv=disable_split_kv,
        )
    # nor is the fused positional encoding
    if pos_encoding_mode == "ROPE_LLAMA":
        with pytest.raises(TypeError, match="pos_encoding_mode"):
            plan_and_run(batch_size, pos_encoding_mode=pos_encoding_mode)
        return

    o_tensor_cores, lse_tensor_cores = plan_and_run(batch_size)
    o_tensor_cores_invariant, lse_tensor_cores_invariant = plan_and_run(invariant_bs)

    # legacy assertion: bitwise equal on the invariant prefix
    n = invariant_bs * qo_len
    if not (
        torch.equal(o_tensor_cores[:n], o_tensor_cores_invariant)
        and torch.equal(lse_tensor_cores[:n], lse_tensor_cores_invariant)
    ):
        pytest.xfail(
            "no batch-invariance contract: the prefix batch differs bitwise; "
            "legacy pinned it with fixed_split_size / disable_split_kv"
        )
