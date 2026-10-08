"""Legacy -> unified: tests/attention/test_fmha_v2_prefill.py

The paged legacy functions call the FMHA v2 kernel (``trtllm_fmha_v2_prefill``
with a ``Q_PAGED_KV_NHD`` / ``Q_PAGED_KV_HND`` layout).  FMHA v2 is not a
``PagedAttention`` backend, so each converted id pins it by name on the legacy
static configuration and asserts the clean "unknown backend" rejection.  The
legacy grids, ids and skip conditions (SM90a / SM12x gates included) are kept,
so on other parts these rows skip like the legacy; the non-paged layout rows
of a mixed grid skip as out of scope.  Several of these functions also need a
knob the unified API does not have (fp8 q with an output dtype, the
non-interleaved ``[B, 2, M]`` K / V tables, skip-softmax, chunked attention);
the notes below record them.

The fixed-length ``fmha_v2_prefill_deepseek`` / ``fmha_v2_prefill_sm120``
functions and the ragged-layout functions are out of scope.
"""

import pytest
import torch

from flashinfer.prefill import resolve_paged_attention
from flashinfer.utils import is_sm12x_supported, is_sm90a_supported

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_fmha_v2_prefill.py"
LEGACY_MAP = [
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek",
        [],
        "out-of-scope",
        "fmha_v2_prefill_deepseek: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek_cuda_graph",
        [],
        "out-of-scope",
        "fmha_v2_prefill_deepseek: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_self_attention",
        [],
        "out-of-scope",
        "fmha_v2_prefill_sm120: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_optional_device_scales",
        [],
        "out-of-scope",
        "fmha_v2_prefill_sm120: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_validation",
        [],
        "out-of-scope",
        "fmha_v2_prefill_sm120: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_deepseek_validates_seq_len",
        [],
        "out-of-scope",
        "fmha_v2_prefill_deepseek: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_cuda_graph",
        [],
        "out-of-scope",
        "fmha_v2_prefill_sm120: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_fmha_v2_prefill_sm120_async_enqueue",
        [],
        "out-of-scope",
        "fmha_v2_prefill_sm120: fixed-length BSHD self-attention, not paged",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill",
        ["test_trtllm_fmha_v2_prefill"],
        "unsupported-by-design",
        "same grid, ids and legacy skips; the paged rows pin FMHA v2, which is not "
        "a unified backend (rejected); fp8 q with an output dtype and the "
        "[max, sum_exp] softmax stats have no unified spelling either; non-paged "
        "layout rows skip (out of scope)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_non_interleaved_kv",
        ["test_trtllm_fmha_v2_prefill_non_interleaved_kv"],
        "unsupported-by-design",
        "same grid, ids and legacy gate; pins FMHA v2 (rejected); the "
        "pre-expanded [B, 2, M] independent K / V page tables have no unified "
        "spelling (one shared table)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_sm120_large_head_dim",
        ["test_trtllm_fmha_v2_prefill_sm120_large_head_dim"],
        "unsupported-by-design",
        "same grid, ids and SM12x gate; the paged NHD rows pin FMHA v2 (rejected); "
        "CONTIGUOUS_Q_KV rows skip (out of scope)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_sm120_chunked_rejected",
        [],
        "out-of-scope",
        "CONTIGUOUS_Q_KV (non-paged) request",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_skip_softmax",
        ["test_trtllm_fmha_v2_prefill_skip_softmax"],
        "unsupported-by-design",
        "same grid, ids and legacy skips; the paged rows pin FMHA v2 (rejected); "
        "skip_softmax_threshold_scale_factor has no unified knob; CONTIGUOUS_Q_KV "
        "rows skip (out of scope)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_attention_sinks",
        [],
        "out-of-scope",
        "SEPARATE_Q_K_V (ragged) layout only, not paged prefill",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_prefill_chunked_attention",
        ["test_trtllm_fmha_v2_prefill_chunked_attention"],
        "unsupported-by-design",
        "same grid, ids and SM90a gate; the Q_PAGED_KV_NHD rows pin FMHA v2 "
        "(rejected); chunked_attention_size has no unified knob; non-paged layout "
        "rows skip (out of scope)",
    ),
    (
        "tests/attention/test_fmha_v2_prefill.py::test_trtllm_fmha_v2_chunked_prefill_chunked_attention",
        ["test_trtllm_fmha_v2_chunked_prefill_chunked_attention"],
        "unsupported-by-design",
        "same grid, ids and SM90a gate; pins FMHA v2 (rejected); "
        "chunked_attention_size has no unified knob",
    ),
]

# the legacy kernel, named as a backend: PagedAttention has no such backend
FMHA_V2 = "trtllm-fmha-v2"
PAGED_LAYOUTS = {"Q_PAGED_KV_NHD": "NHD", "Q_PAGED_KV_HND": "HND"}


def _legacy_case_skips(
    input_layout,
    head_dim,
    dtype,
    logits_soft_cap,
    save_softmax_stats,
    skip_softmax_threshold_scale_factor,
):
    """The skip block of the legacy ``run_trtllm_fmha_v2_prefill_case``."""
    device = torch.device("cuda")
    if not is_sm90a_supported(device) and not is_sm12x_supported(device):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) or SM12x GPUs.")
    is_sm12x = is_sm12x_supported(device)
    if dtype == torch.float8_e4m3fn and is_sm12x:
        pytest.skip("FP8 FMHAv2 not yet supported on SM12x")
    if input_layout == "SEPARATE_Q_K_V" and dtype == torch.float8_e4m3fn:
        pytest.skip("FP8 not supported for SEPARATE_Q_K_V layout")
    if input_layout == "SEPARATE_Q_K_V" and is_sm12x:
        pytest.skip(
            "SEPARATE_Q_K_V requires SM90 warp-specialization, not available on SM12x"
        )
    if head_dim > 256:
        if not is_sm12x:
            pytest.skip("head_dim > 256 FMHAv2 is only supported on SM12x")
        if dtype == torch.float8_e4m3fn:
            pytest.skip("head_dim > 256 FMHAv2 does not support fp8")
        if input_layout == "SEPARATE_Q_K_V":
            pytest.skip("head_dim > 256 FMHAv2 does not support SEPARATE_Q_K_V")
    if input_layout == "SEPARATE_Q_K_V" and logits_soft_cap > 0:
        pytest.skip("Logits soft capping not supported for SEPARATE_Q_K_V layout")
    if save_softmax_stats and input_layout != "CONTIGUOUS_Q_KV":
        pytest.skip(
            "For normal attention, Only CONTIGUOUS_Q_KV layout supports "
            "save_softmax_stats. For MLA only SEPARATE_Q_K_V layout supports "
            "save_softmax_stats."
        )
    if skip_softmax_threshold_scale_factor > 0 and not is_sm90a_supported(device):
        pytest.skip("Skip softmax attention is only supported on SM90+ (Hopper) GPUs.")


def _assert_fmha_v2_rejected(
    *,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_size,
    kv_layout,
    causal=True,
    window_left=-1,
    logits_soft_cap=None,
):
    """The legacy static configuration with FMHA v2 pinned: not a backend."""
    with pytest.raises(ValueError, match=f"unknown backend '{FMHA_V2}'"):
        resolve_paged_attention(
            device=torch.device("cuda:0"),
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            q_dtype=dtype,
            kv_dtype=dtype,
            page_size=page_size,
            kv_layout=kv_layout,
            causal=causal,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            backend=FMHA_V2,
        )


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    ("dtype", "o_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float16),
    ],
)
@pytest.mark.parametrize(
    ("input_layout", "page_size", "save_softmax_stats"),
    [
        ("PACKED_QKV", None, False),
        ("CONTIGUOUS_Q_KV", None, False),
        ("CONTIGUOUS_Q_KV", None, True),
        ("SEPARATE_Q_K_V", None, False),
        ("Q_PAGED_KV_NHD", 32, False),
        ("Q_PAGED_KV_NHD", 128, False),
        ("Q_PAGED_KV_HND", 32, False),
        ("Q_PAGED_KV_HND", 128, False),
    ],
)
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 512, "SLIDING_WINDOW"),
    ],
)
@pytest.mark.parametrize("pos_encoding_mode", [None])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_trtllm_fmha_v2_prefill(
    input_layout,
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    o_dtype,
    causal,
    mask_mode,
    window_left,
    logits_soft_cap,
    pos_encoding_mode,
    save_softmax_stats,
):
    _legacy_case_skips(
        input_layout, head_dim, dtype, logits_soft_cap, save_softmax_stats, 0.0
    )
    if input_layout not in PAGED_LAYOUTS:
        pytest.skip(f"{input_layout} is not a paged layout (out of scope)")

    _assert_fmha_v2_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        page_size=page_size,
        kv_layout=PAGED_LAYOUTS[input_layout],
        causal=causal,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
    )


@pytest.mark.parametrize("input_layout", ["Q_PAGED_KV_NHD", "Q_PAGED_KV_HND"])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_trtllm_fmha_v2_prefill_non_interleaved_kv(
    input_layout, page_size, dtype, num_kv_heads
):
    device = torch.device("cuda")
    if not is_sm90a_supported(device) and not is_sm12x_supported(device):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) or SM12x GPUs.")

    # legacy: B4, seq <= 1024, H8, D128, causal
    _assert_fmha_v2_rejected(
        num_qo_heads=8,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        dtype=dtype,
        page_size=page_size,
        kv_layout=PAGED_LAYOUTS[input_layout],
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [1024])
@pytest.mark.parametrize("num_qo_heads", [8])
@pytest.mark.parametrize("num_kv_heads", [2])
@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("input_layout", "page_size"),
    [
        ("CONTIGUOUS_Q_KV", None),
        ("Q_PAGED_KV_NHD", 32),
        ("Q_PAGED_KV_NHD", 128),
    ],
)
@pytest.mark.parametrize(
    ("causal", "window_left", "mask_mode"),
    [
        (True, -1, "CAUSAL"),
        (False, -1, "PADDING"),
        (True, 127, "SLIDING_WINDOW"),
        (True, 1024, "SLIDING_WINDOW"),
    ],
)
def test_trtllm_fmha_v2_prefill_sm120_large_head_dim(
    input_layout,
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    dtype,
    causal,
    window_left,
    mask_mode,
):
    if not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("This test targets SM12x (Blackwell) FMHAv2.")
    _legacy_case_skips(input_layout, head_dim, dtype, 0.0, False, 0.0)
    if input_layout not in PAGED_LAYOUTS:
        pytest.skip(f"{input_layout} is not a paged layout (out of scope)")

    _assert_fmha_v2_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        page_size=page_size,
        kv_layout=PAGED_LAYOUTS[input_layout],
        causal=causal,
        window_left=window_left,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [16384])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize(
    ("dtype", "o_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "input_layout", ["CONTIGUOUS_Q_KV", "Q_PAGED_KV_NHD", "Q_PAGED_KV_HND"]
)
@pytest.mark.parametrize(
    (
        "skip_softmax_threshold_scale_factor",
        "rtol",
        "atol",
    ),
    [
        (500.0, 2e-2, 1.2e-1),
        (10000.0, 2e-2, 2e-1),
    ],
)
def test_trtllm_fmha_v2_prefill_skip_softmax(
    input_layout,
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    o_dtype,
    skip_softmax_threshold_scale_factor,
    rtol,
    atol,
):
    _legacy_case_skips(
        input_layout, head_dim, dtype, 0.0, False, skip_softmax_threshold_scale_factor
    )
    if input_layout not in PAGED_LAYOUTS:
        pytest.skip(f"{input_layout} is not a paged layout (out of scope)")

    _assert_fmha_v2_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        page_size=32,
        kv_layout=PAGED_LAYOUTS[input_layout],
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_seq_len", [1024, 4096])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("input_layout", "page_size"),
    [
        ("CONTIGUOUS_Q_KV", None),
        ("SEPARATE_Q_K_V", None),
        ("Q_PAGED_KV_NHD", 32),
    ],
)
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_trtllm_fmha_v2_prefill_chunked_attention(
    batch_size,
    max_seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    input_layout,
    page_size,
    chunked_attention_size,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")
    if input_layout not in PAGED_LAYOUTS:
        pytest.skip(f"{input_layout} is not a paged layout (out of scope)")

    _assert_fmha_v2_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        page_size=page_size,
        kv_layout=PAGED_LAYOUTS[input_layout],
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("max_kv_len", [1024, 4096])
@pytest.mark.parametrize("max_new_tokens", [64, 256])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("chunked_attention_size", [64, 256])
def test_trtllm_fmha_v2_chunked_prefill_chunked_attention(
    batch_size,
    max_kv_len,
    max_new_tokens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    page_size,
    chunked_attention_size,
):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FMHA v2 requires SM90+ (Hopper) GPUs.")

    _assert_fmha_v2_rejected(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        page_size=page_size,
        kv_layout="NHD",
    )
