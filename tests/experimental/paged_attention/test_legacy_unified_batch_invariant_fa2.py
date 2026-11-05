"""Legacy -> unified: tests/attention/test_batch_invariant_fa2.py

``test_batch_prefill_tensor_cores`` checks that fa2 is batch invariant under
the planner knobs ``fixed_split_size`` / ``disable_split_kv``: the first
``invariant_bs`` requests must give bitwise the same output and LSE as the
full batch.  ``PagedAttention.plan()`` has neither knob, so the test cannot be
expressed: every case is an expected failure naming the missing API (the
``ROPE_LLAMA`` rows also need the fused positional encoding).  The grid is the
legacy grid, so the node ids equal the legacy ids.

The legacy decode test (``BatchDecodeWithPagedKVCacheWrapper``) is out of
scope: ``PagedAttention`` is the paged-prefill API and has no decode route.
"""

import pytest

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
        "xfail: plan() has no fixed_split_size / disable_split_kv, the knobs this "
        "test pins (API limit); ROPE_LLAMA rows also need the fused RoPE",
    ),
]


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
    reason = (
        "API limit: PagedAttention.plan() has no fixed_split_size / "
        "disable_split_kv, the batch-invariance knobs this test pins"
    )
    if pos_encoding_mode != "NONE":
        reason += "; no fused RoPE (pos_encoding_mode) either"
    pytest.xfail(reason)
