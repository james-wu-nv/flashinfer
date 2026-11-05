"""Legacy -> unified: tests/attention/test_sm120_fmha_paged.py

The legacy subject is ``sm120_fmha_fp8_paged_prefill``, the SM120 cute-dsl
FP8 paged FMHA (also behind the legacy wrapper's ``backend="cute-dsl-prims"``):
fp8 e4m3 q / K / V, an e4m3 P for P.V and an fp16 output, checked against an
fp32 reference at 0.2.  No ``PagedAttention`` backend runs that kernel, fp8 q
is not declared on any backend and the output is written in q's dtype, so
every case is an expected failure naming those API limits (the head_dim 32
rows also need (32, 32) declared).  The legacy gates come first (CuTe DSL
experimental available, compute capability exactly 12.0), so on B200 every
case skips like the legacy.  Same grids, so the node ids equal the legacy ids.
"""

import pytest
import torch

from flashinfer.cute_dsl.availability import is_cute_dsl_experimental_available
from flashinfer.utils import get_compute_capability

if not is_cute_dsl_experimental_available():
    pytest.skip("SM120 CuTe DSL dependencies are unavailable", allow_module_level=True)

_device = torch.device("cuda") if torch.cuda.is_available() else None

pytestmark = pytest.mark.skipif(
    _device is None or get_compute_capability(_device) != (12, 0),
    reason="SM120 FMHA requires SM120 GPU (compute capability 12.0)",
)

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_sm120_fmha_paged.py"
LEGACY_MAP = [
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_uniform_q",
        ["test_sm120_paged_uniform_q"],
        "unsupported-by-design",
        "xfail after the legacy SM120 gate: no backend for the SM120 cute-dsl FP8 "
        "paged FMHA, no fp8 q, no output dtype (API limits); D32 rows also need "
        "(32, 32)",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_variable_kv_lengths",
        ["test_sm120_paged_variable_kv_lengths"],
        "unsupported-by-design",
        "same xfail as the row above",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_varlen_q",
        ["test_sm120_paged_varlen_q"],
        "unsupported-by-design",
        "same xfail as the rows above",
    ),
    (
        "tests/attention/test_sm120_fmha_paged.py::test_sm120_paged_compile_cache",
        ["test_sm120_paged_compile_cache"],
        "unsupported-by-design",
        "xfail after the legacy SM120 gate: the compile cache of the SM120 cute-dsl "
        "kernel, which no backend runs (API limit)",
    ),
]

NO_BACKEND_XFAIL = (
    "API limit: no PagedAttention backend runs the SM120 cute-dsl FP8 paged FMHA "
    "(sm120_fmha_fp8_paged_prefill), the legacy subject; fp8 q is not declared on "
    "any backend and run() writes the output in q's dtype (the legacy writes fp16 "
    "from fp8 q)"
)


@pytest.mark.parametrize(
    "B,Sq,Skv,Hq,Hkv,D,page_size",
    [
        (1, 128, 128, 4, 4, 32, 64),  # head_dim=32
        (1, 128, 128, 8, 8, 128, 64),  # MHA, 2 pages/seq
        (2, 64, 128, 8, 2, 128, 64),  # GQA 4:1
        (1, 128, 128, 4, 4, 64, 64),  # head_dim=64
        (1, 129, 384, 2, 1, 256, 64),  # D=256 three-stage KV ring
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_uniform_q(B, Sq, Skv, Hq, Hkv, D, page_size, is_causal):
    reason = NO_BACKEND_XFAIL
    if D == 32:
        reason += "; head dims (32, 32) are not declared on any backend either"
    pytest.xfail(reason)


def test_sm120_paged_variable_kv_lengths():
    pytest.xfail(NO_BACKEND_XFAIL)


@pytest.mark.parametrize(
    "Hq,Hkv,D,page_size",
    [
        (8, 8, 128, 64),  # MHA
        (8, 2, 128, 64),  # GQA 4:1
    ],
)
@pytest.mark.parametrize("is_causal", [False, True])
def test_sm120_paged_varlen_q(Hq, Hkv, D, page_size, is_causal):
    pytest.xfail(NO_BACKEND_XFAIL)


def test_sm120_paged_compile_cache():
    pytest.xfail(NO_BACKEND_XFAIL)
