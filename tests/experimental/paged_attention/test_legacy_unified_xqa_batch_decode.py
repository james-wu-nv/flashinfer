"""Legacy -> unified: tests/attention/test_xqa_batch_decode.py

Every legacy function in this file tests the XQA decode kernel
(``flashinfer.decode.xqa_batch_decode_with_kv_cache``), including the
speculative-decode rows (``q_len_per_req > 1``), the ragged draft lengths, the
nvfp4 KV rows and the mask-mode check.  The prefill / decode wrappers it calls
are only references.  ``PagedAttention`` is the paged-prefill API and has no
decode route, so every function is out of scope and this file holds no tests.
"""

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_xqa_batch_decode.py"
LEGACY_MAP = [
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode",
        [],
        "out-of-scope",
        "XQA decode kernel (xqa_batch_decode_with_kv_cache, q_len_per_req decode "
        "entries); PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_spec_dec_sliding_window",
        [],
        "out-of-scope",
        "delegates to test_xqa_batch_decode (XQA decode kernel); PagedAttention has "
        "no decode route",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_ragged_q",
        [],
        "out-of-scope",
        "XQA decode kernel with ragged draft lengths; PagedAttention has no decode "
        "route",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_nvfp4_kv",
        [],
        "out-of-scope",
        "XQA decode kernel with nvfp4 KV; PagedAttention has no decode route",
    ),
    (
        "tests/attention/test_xqa_batch_decode.py::test_xqa_batch_decode_mask_mode_deterministic",
        [],
        "out-of-scope",
        "XQA decode kernel against an analytic expectation; PagedAttention has no "
        "decode route",
    ),
]
