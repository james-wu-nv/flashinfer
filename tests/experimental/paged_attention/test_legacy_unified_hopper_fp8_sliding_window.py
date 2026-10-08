"""Legacy -> unified: tests/attention/test_hopper_fp8_sliding_window.py

The paged legacy function is a hang regression for the SM90 fp8 fa3 consumer
with a sliding window: an fp8 (e4m3) QUERY and K/V with per-head scale
tensors, an fp16 output, ``window_left`` 128 / 256 / 511 on seq 257 / 512 /
1024, checked by an MSE budget against the fp16 fa3 wrapper.  The window is a
unified axis, the fp8 q is not: every backend admits fp16 / bf16 q only (fp8
is a KV-only axis with host-float per-tensor scales).  So each legacy id pins
fa3, the backend the legacy calls, on the legacy static configuration (GQA
32:8, D128, page 32, NHD, causal, the legacy window) and asserts the clean
"unsupported q dtype" rejection.  The legacy SM90a gate is kept, so on other
parts these rows skip like the legacy.

The ragged function is out of scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention
from flashinfer.utils import is_sm90a_supported

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_hopper_fp8_sliding_window.py"
LEGACY_MAP = [
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
        "same grid and ids; fp8 q with per-head scale tensors and an fp16 output on "
        "fa3 has no unified spelling (the window does): pinned fa3 rejects q_dtype "
        "float8 (SM90a-gated like the legacy)",
    ),
]

HQ, HKV, D = 32, 8, 128  # the legacy GQA shape
SEQ_LENS = [257, 512, 1024]
WINDOWS = [128, 256, 511]


@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("window_left", WINDOWS)
def test_fp8_paged_prefill_sliding_window(seq_len, window_left):
    device = torch.device("cuda:0")
    if not is_sm90a_supported(device):
        pytest.skip("SM90A is not supported")

    with pytest.raises(ValueError, match="fa3: unsupported q dtype"):
        resolve_paged_attention(
            device=device,
            num_qo_heads=HQ,
            num_kv_heads=HKV,
            head_dim_qk=D,
            q_dtype=torch.float8_e4m3fn,
            kv_dtype=torch.float8_e4m3fn,
            page_size=32,
            kv_layout="NHD",
            causal=True,
            window_left=window_left,
            kv_input_form="page_indices",
            backend="fa3",
        )
