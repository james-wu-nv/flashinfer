"""Legacy -> unified: tests/attention/test_hopper_fp8_sliding_window.py

The paged legacy function is a hang regression for the SM90 fp8 fa3 consumer
with a sliding window: an fp8 (e4m3) QUERY and K/V with per-head scale
tensors, an fp16 output, ``window_left`` 128 / 256 / 511 on seq 257 / 512 /
1024, checked by an MSE budget against the fp16 fa3 wrapper.  The window is
a ``PagedAttention`` axis, the fp8 q is not: fa3 declares fp16 / bf16 q only,
the output is written in q's dtype, and ``run()`` takes no ``q_scale`` and no
per-head scale tensors.  So every legacy id is an expected failure naming
that API limit, after the legacy SM90a skip (so on other parts these rows
skip like the legacy).

The ragged function is out of scope.
"""

import pytest
import torch

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
        "same grid and ids; xfail after the legacy SM90a skip: fp8 q with per-head "
        "scale tensors and an fp16 output has no PagedAttention spelling (the window "
        "has; API limit)",
    ),
]

FP8_Q_XFAIL = (
    "API limit: PagedAttention has no fp8-q path: fa3 declares fp16 / bf16 q only, "
    "run() writes the output in q's dtype (the legacy writes fp16 from fp8 q) and "
    "takes no q_scale and no per-head scale tensors (host-float per-tensor "
    "k_scale / v_scale only)"
)


@pytest.mark.parametrize("seq_len", [257, 512, 1024])
@pytest.mark.parametrize("window_left", [128, 256, 511])
def test_fp8_paged_prefill_sliding_window(seq_len, window_left):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is not supported")
    pytest.xfail(FP8_Q_XFAIL)
