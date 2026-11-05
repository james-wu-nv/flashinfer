"""Legacy -> unified: tests/attention/test_modular_fmha_prefill_paged.py

The legacy file tests the paged mode of the modular cute-dsl prefill kernel
(``flashinfer.cute_dsl.attention``, also behind
``BatchPrefillWithPagedKVCacheWrapper(backend="cute-dsl")``) against the same
kernel's ragged mode, mostly bitwise.  No ``PagedAttention`` backend runs that
kernel (the unified backends are fa2, fa3, cudnn, trtllm-gen and cake), so
every case is an expected failure naming the missing backend, plus any other
missing API the case needs (fp8 q, a V cache of another dtype than K, the
sigmoid / ALiBi variants, non-causal + sliding window, the input NaN scan).
The legacy gates (CuTe DSL available, SM100a) come first, and the grids are
the legacy grids, so the node ids equal the legacy ids.
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_modular_fmha_prefill_paged.py"
LEGACY_MAP = [
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_bitwise_vs_ragged",
        ["test_paged_bitwise_vs_ragged"],
        "unsupported-by-design",
        "xfail: no PagedAttention backend runs the modular cute-dsl kernel (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_identity_table",
        ["test_paged_identity_table"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_d64_windowed_null_block",
        ["test_paged_d64_windowed_null_block"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_nan_pool_tail_clamp",
        ["test_paged_nan_pool_tail_clamp"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_windowed",
        ["test_paged_windowed"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit); the non-causal row also "
        "needs non-causal + sliding window",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_null_block_window_clamp",
        ["test_paged_null_block_window_clamp"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit); the non-causal row also "
        "needs non-causal + sliding window",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_hnd_bitwise_vs_nhd",
        ["test_paged_wrapper_hnd_bitwise_vs_nhd"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_lse",
        ["test_paged_wrapper_lse"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper",
        ["test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend and no fp8 q (API limits)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged",
        ["test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend, and one kv_dtype per plan (no fp8 V with "
        "a bf16 K) (API limits)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_variants_bitwise_vs_ragged",
        ["test_paged_wrapper_variants_bitwise_vs_ragged"],
        "unsupported-by-design",
        "xfail: no modular cute-dsl backend (API limit); sigmoid and ALiBi also have "
        "no PagedAttention spelling (sinks do)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_rejections",
        ["test_paged_wrapper_rejections"],
        "unsupported-by-design",
        "xfail: the cute-dsl wrapper's input validation; no modular cute-dsl backend "
        "(API limit)",
    ),
    (
        "tests/attention/test_modular_fmha_prefill_paged.py::test_paged_wrapper_validate_inputs_nan_scan",
        ["test_paged_wrapper_validate_inputs_nan_scan"],
        "unsupported-by-design",
        "xfail: the cute-dsl wrapper's FLASHINFER_VALIDATE_INPUTS NaN scan; no modular "
        "cute-dsl backend and no input NaN scan (API limits)",
    ),
]

NO_BACKEND_XFAIL = (
    "API limit: no PagedAttention backend runs the modular cute-dsl prefill kernel "
    "(flashinfer.cute_dsl.attention), the legacy subject (the unified backends are "
    "fa2, fa3, cudnn, trtllm-gen and cake)"
)
NONCAUSAL_WINDOW = "; non-causal + sliding window is declined as well (fa2: M20)"


def _skip_unless_sm100():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")


SHAPES = [
    ([512, 512], [512, 512]),  # uniform, page-aligned
    ([7, 300, 1291], [23, 300, 1547]),  # varlen, partial last pages, s_k > s_q
]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [8, 16, 64, 128])
@pytest.mark.parametrize("shapes", SHAPES)
def test_paged_bitwise_vs_ragged(causal, page_size, shapes):
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


def test_paged_identity_table():
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_d64_windowed_null_block(page_size):
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize(
    "causal,page_size",
    [(True, 8), (True, 16), (False, 64)],
)
def test_paged_nan_pool_tail_clamp(causal, page_size):
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([1024], [1024], False, 16, 255),
        ([1024], [1024], True, 8, 255),
        ([1291], [1547], True, 64, 511),
    ],
)
def test_paged_windowed(sq, sk, causal, page_size, window_left):
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL + ("" if causal else NONCAUSAL_WINDOW))


@pytest.mark.parametrize(
    "sq,sk,causal,page_size,window_left",
    [
        ([300], [1500], False, 16, 127),
        ([1291], [1547], True, 64, 511),
        ([7, 300, 1291], [23, 300, 1547], True, 16, 127),
        ([7, 300, 1291], [23, 300, 1547], True, 8, 127),
    ],
)
def test_paged_null_block_window_clamp(sq, sk, causal, page_size, window_left):
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL + ("" if causal else NONCAUSAL_WINDOW))


def test_paged_wrapper_windowed_bitwise_vs_ragged_wrapper():
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


def test_paged_wrapper_hnd_bitwise_vs_nhd():
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


def test_paged_wrapper_lse():
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_fp8_bitwise_vs_ragged_wrapper(page_size):
    _skip_unless_sm100()
    pytest.xfail(
        NO_BACKEND_XFAIL + "; fp8 q (with a bf16 output) is not declared either"
    )


@pytest.mark.parametrize("page_size", [8, 16])
def test_paged_wrapper_mixed_v_dtype_bitwise_vs_ragged(page_size):
    _skip_unless_sm100()
    pytest.xfail(
        NO_BACKEND_XFAIL + "; a plan has one kv_dtype (no fp8 V with a bf16 K) either"
    )


@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("variant_name", ["sigmoid", "alibi", "sink"])
def test_paged_wrapper_variants_bitwise_vs_ragged(variant_name, page_size):
    _skip_unless_sm100()
    reason = NO_BACKEND_XFAIL
    if variant_name != "sink":
        reason += f"; no {variant_name} attention variant in plan() either"
    pytest.xfail(reason)


def test_paged_wrapper_rejections():
    _skip_unless_sm100()
    pytest.xfail(NO_BACKEND_XFAIL)


def test_paged_wrapper_validate_inputs_nan_scan(monkeypatch):
    _skip_unless_sm100()
    pytest.xfail(
        NO_BACKEND_XFAIL + "; no FLASHINFER_VALIDATE_INPUTS input NaN scan either"
    )
