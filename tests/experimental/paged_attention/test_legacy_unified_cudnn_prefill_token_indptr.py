"""Legacy -> unified: tests/attention/test_cudnn_prefill_token_indptr.py

Every legacy function here calls the ragged cuDNN prefill (3-D K/V with a
token-unit ``kv_indptr``, no page table), which is outside the PagedAttention
scope, so this file holds no tests.
"""

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_cudnn_prefill_token_indptr.py"
LEGACY_MAP = [
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_token_indptr",
        [],
        "out-of-scope",
        "ragged cuDNN prefill (token-unit kv_indptr, no page table); not paged",
    ),
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_token_indptr_omit_actual_seq_lens",
        [],
        "out-of-scope",
        "ragged cuDNN prefill (token-unit kv_indptr, no page table); not paged",
    ),
    (
        "tests/attention/test_cudnn_prefill_token_indptr.py::test_cudnn_prefill_lse_is_base2",
        [],
        "out-of-scope",
        "ragged cuDNN prefill (token-unit kv_indptr, no page table); not paged",
    ),
]
