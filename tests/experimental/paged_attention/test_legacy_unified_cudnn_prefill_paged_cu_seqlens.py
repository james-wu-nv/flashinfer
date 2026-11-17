"""Legacy -> unified: tests/attention/test_cudnn_prefill_paged_cu_seqlens.py

The legacy test drives one paged cuDNN call (token-unit ``batch_offsets_q``)
through both cuDNN offset paths, the direct token-offset path and the
element-offset conversion taken below the library's gate, and checks that the
two agree.  It has two halves: with explicit per-request Q lengths the gate is
``_cudnn_supports_paged_token_offsets(explicit_q=True, explicit_kv=True)``
(cuDNN 9.24 for f16 / bf16); with the Q lengths omitted cuDNN masks by the
cumulative offsets (the mixed form, cuDNN 9.25).

``PagedAttention`` pinned to cudnn issues the explicit-Q call: its backend
always passes per-request Q and KV lengths next to the token offsets.  The
backend asks that helper before it stages a plan and the functional entry asks
it when it resolves a paged call, both as an attribute of
``flashinfer.cudnn.prefill``.  The legacy test patches the version gate under
the helper; this conversion patches the helper itself, where both read it: one
plan + run with it answering False and one with it answering True, on the
legacy fixture (same grid, seed and tensors, so the node ids equal the legacy
ids), with no LSE and the legacy 256 MiB workspace.

The two paths give bitwise-equal outputs on this grid (SM100, cuDNN 9.25 and
9.27), so the comparison passes just as well when both runs take one path.
Each run's path is therefore observed:

- the gate was asked, and only for explicit Q and KV lengths;
- every cuDNN metadata resolved during the run took the path the gate named
  (``_token_offsets``);
- with the gate off nothing is staged, so run() went through the functional
  entry, which converts the offsets to element units; with it on, plan()
  staged the token-offset metadata run() executes.

Each case then checks the legacy assertion (direct vs conversion) at the
legacy tolerance and both outputs against the fp32 paged-attention oracle.
The cumulative-Q half is an expected failure that names the limit:
``PagedAttention`` cannot omit the Q lengths.  It first runs the case and
fails, instead of xfailing, if the gate is asked for anything but explicit Q
and KV lengths.  A cuDNN test, so only cudnn runs; a GPU where it cannot is
skipped with the resolver's reason.
"""

import pytest
import torch

from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.prefill import (
    PagedAttention,
    PagedAttentionMetadata,
    resolve_paged_attention,
)

from .paged_attention_reference import reference_paged_prefill

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py"
LEGACY_MAP = [
    (
        "tests/attention/test_cudnn_prefill_paged_cu_seqlens.py::test_cudnn_paged_prefill_token_offsets_match_legacy",
        ["test_cudnn_paged_prefill_token_offsets_match_legacy"],
        "partial",
        "explicit-q half: same grid, seed and tensors on cudnn (no LSE, the legacy "
        "256 MiB workspace); the library's paged token-offset gate steers the "
        "unified call to each path and each run's path is observed (resolved "
        "metadata, staged plan); direct vs conversion at the legacy tolerance plus "
        "the fp32 oracle on both outputs. cumulative-q half xfail: not expressible, "
        "the cudnn backend always passes explicit per-request Q lengths",
    ),
]

OUT_TOL = dict(atol=2e-2, rtol=2e-2)

CUMULATIVE_Q_XFAIL = (
    "API limit: PagedAttention cannot omit the per-request Q lengths (its cudnn "
    "backend always passes them next to the token offsets), so the cumulative-Q "
    "form of the cuDNN call is never issued"
)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("s_qo,s_kv", [(64, 512), (17, 200)])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(
    "explicit_q", [False, True], ids=["cumulative-q", "explicit-q"]
)
def test_cudnn_paged_prefill_token_offsets_match_legacy(
    monkeypatch, batch_size, s_qo, s_kv, page_size, num_kv_heads, causal, explicit_q
):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    # the legacy gate, read before the helper is patched below
    if not cudnn_prefill._cudnn_supports_paged_token_offsets(
        torch.bfloat16, explicit_q=explicit_q, explicit_kv=True
    ):
        pytest.skip("cuDNN backend/frontend does not support these paged token offsets")

    device = "cuda:0"
    num_qo_heads, head_dim = 8, 128
    try:
        resolve_paged_attention(
            device=torch.device(device),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=torch.bfloat16,
            page_size=page_size,
            kv_layout="HND",
            causal=causal,
            backend="cudnn",
        )
    except ValueError as e:
        pytest.skip(f"cudnn not runnable here: {e}")

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
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

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

    # What one plan + run asks the gate, and the path every cuDNN metadata
    # resolved during it took (True: direct token offsets).
    asked, resolved = [], []
    real_resolve = cudnn_prefill._PrefillMetadata.resolve_from_plan

    def resolve_from_plan(self, *args, **kwargs):
        metadata = real_resolve(self, *args, **kwargs)
        resolved.append(metadata._token_offsets)
        return metadata

    monkeypatch.setattr(
        cudnn_prefill._PrefillMetadata, "resolve_from_plan", resolve_from_plan
    )

    def run(direct):
        # plan and run with the gate answering ``direct`` (run() issues the
        # cuDNN call); returns the output and the plan the backend staged,
        # its private state, read as the neighbouring cuDNN tests read it
        del asked[:], resolved[:]

        def gate(dtype, *, explicit_q, explicit_kv):
            asked.append((explicit_q, explicit_kv))
            return direct

        monkeypatch.setattr(cudnn_prefill, "_cudnn_supports_paged_token_offsets", gate)
        attn = PagedAttention(torch.device(device), workspace_buffer=ws)
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
        out = attn.run(q, (k_cache, v_cache), sm_scale=scale)[0]
        assert set(asked) == {(True, True)}, (
            "the paged cuDNN call did not consult "
            "_cudnn_supports_paged_token_offsets for explicit Q and KV lengths "
            f"only (asked {asked}), so the gate did not choose its path"
        )
        return out, attn._impl._active._cudnn_plan

    if not explicit_q:
        # The legacy half omits the Q lengths.  This API has no such call:
        # on this case too the gate is asked for explicit Q and KV lengths
        # only (run() asserts it), so the limit is observed before it is
        # reported.
        run(direct=True)
        pytest.xfail(CUMULATIVE_Q_XFAIL)

    # Legacy paged path: the gate off -> element-offset conversion, issued by
    # the functional entry since nothing is staged.
    out_legacy, staged = run(direct=False)
    assert staged is None
    assert resolved and not any(resolved)

    # Direct paged path: the gate on -> plan() stages token-offset metadata.
    out_direct, staged = run(direct=True)
    assert staged is not None and staged.metadata._token_offsets
    assert resolved and all(resolved)

    # legacy assertion at the legacy tolerance
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
