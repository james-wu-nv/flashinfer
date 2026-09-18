"""Legacy -> unified: tests/attention/test_hopper_fp8_sliding_window.py

Every test function below carries the name of the legacy test it converts and
runs the legacy fixture through flashinfer.prefill.PagedAttention.

CI: legacy file runs in the H100 1/5-sample lane only (SM90a-gated); the
unified file is in no default lane (tests/experimental is excluded by
norecursedirs).

The legacy paged function is a hang regression for the SM90 fp8 consumer
with a sliding window: fp8 (e4m3) QUERY and K/V with per-head scale tensors,
fp16 output, ``window_left`` 128 / 256 / 511 on seq 257 / 512 / 1024, MSE
budget against the fp16 fa3 wrapper.  The sliding window is a unified axis
(fa3 declares supports_window=True for fp16 / bf16 q), the fp8 q dtype is
not: every backend declares q_dtypes = {fp16, bf16}
(flashinfer/experimental/paged_attention/_backends/_capabilities.py) and fp8
is a KV-only axis with host-float per-tensor scales, so resolve() rejects
q_dtype float8 for every backend (EXPECT_FP8_Q; per-head scale tensors also
need EXPECT_DEVICE_SCALES).  Each legacy id asserts that rejection on the
legacy static configuration (GQA 32:8, D128, page 32, causal, the legacy
window); the positive branch waits for the quantization descriptor.

H100 command (repo root): same rejection (fa3 excluded by q dtype instead
of compute capability):

    python -m pytest -q -ra -o faulthandler_timeout=300 \\
        tests/experimental/test_legacy_unified_hopper_fp8_sliding_window.py

Cannot-cover / partial notes (kept next to the LEGACY_MAP rows):
- test_fp8_paged_prefill_sliding_window: fp8 q, see above
  (unsupported-by-design).
- test_fp8_ragged_prefill_sliding_window: ragged wrapper -> out-of-scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention

from .legacy_unified_helpers import (
    DEVICE,
    EXPECT_FP8_Q,
    check_legacy_map,
    check_legacy_map_complete,
    gated,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LEGACY_SOURCE = "tests/attention/test_hopper_fp8_sliding_window.py"
LEGACY_MAP = [
    # (legacy nodeid, [unified test function names in this file], status, note)
    (
        "tests/attention/test_hopper_fp8_sliding_window.py::test_fp8_ragged_prefill_sliding_window",
        [],
        "out-of-scope",
        "ragged KV wrapper, not paged prefill",
    ),
    (
        "tests/attention/test_hopper_fp8_sliding_window.py::test_fp8_paged_prefill_sliding_window",
        ["test_fp8_paged_prefill_sliding_window"],
        "unsupported-by-design",
        "same grid (seq 257/512/1024, window 128/256/511; GQA 32:8, D128, page 32, "
        "causal, e4m3): fp8 QUERY + K/V with per-head scale tensors and a fp16 output on "
        "the SM90 fp8 kernel; the window is a unified axis for fp16/bf16 q, the fp8 q "
        "dtype is the gap (EXPECT_FP8_Q; per-head scale tensors also EXPECT_DEVICE_SCALES): "
        "resolve() rejects q_dtype float8 for every backend, asserted per legacy id; the "
        "legacy hang-regression timeout has no unified counterpart until then",
    ),
]


def test_legacy_map_is_well_formed():
    check_legacy_map(LEGACY_MAP, globals())
    check_legacy_map_complete(LEGACY_SOURCE, LEGACY_MAP)


HQ, HKV, D = 32, 8, 128  # the legacy GQA shape
SEQ_LENS = [257, 512, 1024]
WINDOWS = [128, 256, 511]


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("window_left", WINDOWS)
def test_fp8_paged_prefill_sliding_window(seq_len, window_left):
    page_size = 32
    res = gated(
        EXPECT_FP8_Q,
        lambda: resolve_paged_attention(
            device=torch.device(DEVICE),
            num_qo_heads=HQ,
            num_kv_heads=HKV,
            head_dim_qk=D,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=page_size,
            kv_layout="NHD",
            causal=True,
            need_lse=False,
            window_left=window_left,
            kv_input_form="page_indices",
            max_q_len=seq_len,
            backend="auto",
        ),
        match="unsupported q dtype",
    )
    if res is None:
        return
    pytest.fail(
        "EXPECT_FP8_Q flipped: port the legacy per_head_symmetric_quant sliding-window "
        "fixture (tests/attention/test_hopper_fp8_sliding_window.py) here"
    )
