"""Legacy -> unified: tests/attention/test_cudnn_prefill_paged_cu_seqlens.py

The legacy test drives one paged cuDNN call (token-unit ``batch_offsets_q``)
through both cuDNN sequence-length paths by monkeypatching the private gate
``flashinfer.cudnn.prefill._cudnn_supports_direct_seqlens``: forced off (the
element-offset conversion path) and forced on for the mixed form (the direct
``cu_seq_len_q`` + ``seq_len_kv`` path), and checks the two agree.
``PagedAttention`` pinned to cudnn issues that same paged call, so the same
monkeypatch steers it: one plan + run under each gate on the legacy fixture
(same grid, seed and tensors, so the node ids equal the legacy ids).  Each
case checks the legacy assertion (direct vs conversion) at the legacy
tolerance, and both outputs against the fp32 paged-attention oracle.
"""

import pytest
import torch

from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.prefill import PagedAttention, PagedAttentionMetadata

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py"
LEGACY_MAP = [
    (
        "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py::test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy",
        ["test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy"],
        "equivalent",
        "same grid, seed and tensors on cudnn; the legacy gate monkeypatch steers the "
        "unified call to each path; direct vs conversion at the legacy tolerance plus "
        "the fp32 oracle on both outputs",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(64, 512), (17, 200)])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
def test_cudnn_paged_prefill_cu_seqlens_direct_matches_legacy(
    monkeypatch, batch_size, s_qo, s_kv, page_size, num_kv_heads, causal
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    if not cudnn_prefill._cudnn_supports_direct_seqlens(torch.bfloat16, mixed=True):
        pytest.skip("cuDNN backend/frontend too old for mixed-form paged seqlens")

    device = "cuda:0"
    num_qo_heads, head_dim = 8, 128

    # the legacy fixture (_make_paged_inputs), verbatim
    torch.manual_seed(1)
    seq_q = torch.randint(1, s_qo + 1, (batch_size,), dtype=torch.int32, device=device)
    seq_kv = torch.randint(
        s_qo, s_kv + 1, (batch_size,), dtype=torch.int32, device=device
    )
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(seq_q, 0)]).int()
    q = torch.randn(
        int(seq_q.sum()), num_qo_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_cache = torch.randn(
        total_num_pages,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    strides = (
        2 * page_size * num_kv_heads * head_dim,
        head_dim,
        num_kv_heads * head_dim,
        1,
    )
    k_cache = kv_cache[:, 0].as_strided(kv_cache[:, 0].shape, strides)
    v_cache = kv_cache[:, 1].as_strided(kv_cache[:, 1].shape, strides)
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    scale = float(head_dim**-0.5)

    qo_indptr_cpu = qo_indptr.cpu()
    kv_lens_cpu = seq_kv.cpu()
    md = PagedAttentionMetadata.dense(
        qo_indptr,
        seq_kv,
        block_tables,
        page_size=page_size,
        max_q_len=s_qo,
        max_kv_len=s_kv,
        qo_indptr_cpu=qo_indptr_cpu,
        kv_seq_lens_cpu=kv_lens_cpu,
    )

    def run():
        # plan and run under the active gate (run() issues the cuDNN call)
        attn = PagedAttention(torch.device(device))
        attn.plan(
            md,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.bfloat16,
            kv_layout="HND",
            causal=causal,
            lse_mode="none",  # legacy return_lse=False
            backend="cudnn",
        )
        assert attn.backend == "cudnn"
        return attn.run(q, (k_cache, v_cache), sm_scale=scale)[0]

    # Legacy paged path: force the gate off -> element-offset conversion.
    monkeypatch.setattr(
        cudnn_prefill,
        "_cudnn_supports_direct_seqlens",
        lambda dtype, *, mixed=False: False,
    )
    out_legacy = run()

    # Direct paged path: force the gate on, but only for the mixed-form request.
    mixed_calls = []

    def _gate_direct(dtype, *, mixed=False):
        mixed_calls.append(mixed)
        return mixed

    monkeypatch.setattr(cudnn_prefill, "_cudnn_supports_direct_seqlens", _gate_direct)
    out_direct = run()

    # legacy assertion at the legacy tolerance
    assert True in mixed_calls, (
        "paged dispatch did not consult _cudnn_supports_direct_seqlens with "
        "mixed=True, so the direct mixed-form path was not exercised"
    )
    torch.testing.assert_close(out_direct, out_legacy, atol=1e-2, rtol=1e-2)

    # fp32 oracle on both paths
    ref_out, _ = reference_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr_cpu,
        kv_lens_cpu,
        block_tables,
        page_size,
        causal,
        sm_scale=scale,
        kv_layout="HND",
    )
    torch.testing.assert_close(out_direct.float(), ref_out, **OUT_TOL)
    torch.testing.assert_close(out_legacy.float(), ref_out, **OUT_TOL)
