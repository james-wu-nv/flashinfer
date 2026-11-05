"""Legacy -> unified: tests/attention/test_batch_decode_kernels.py

Every function of the legacy file tests a decode API or kernel
(``BatchDecodeWithPagedKVCacheWrapper``, its CUDA-graph and tensor-core
multi-token variants, ``single_decode_with_kv_cache``), so the whole file is
out of scope: ``PagedAttention`` is the paged-prefill API and has no decode
route.  The file keeps only the map the parity report reads; it has no tests.
"""

# Read by the legacy <-> unified parity report:
# (legacy nodeid, [unified test functions], status, note)
LEGACY_SOURCE = "tests/attention/test_batch_decode_kernels.py"
LEGACY_MAP = [
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_with_fast_plan",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_tuple_paged_kv_cache",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_cuda_graph_batch_decode_with_paged_kv_cache",
        [],
        "out-of-scope",
        "CUDAGraphBatchDecodeWithPagedKVCacheWrapper: decode-only API.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_nvfp4",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.  "
        "(NVFP4 KV is also undeclared on every unified backend.)",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_with_paged_kv_cache_nvfp4_large_head",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_batch_decode_rejects_unequal_kv_strides_nvfp4_contract",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_single_decode_torch_compile_cuda_graph",
        [],
        "out-of-scope",
        "single_decode_with_kv_cache under torch.compile, not paged prefill.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_cuda_graph_uniform_multi_token_decode_with_paged_kv_cache",
        [],
        "out-of-scope",
        "multi-token DECODE workload (tensor-core decode wrapper, "
        "plan(q_len_per_req), CUDA graph); its fa2 prefill wrapper is only the "
        "reference.",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_tensor_core_decode_rejects_mismatched_q_len",
        [],
        "out-of-scope",
        "decode wrapper plan(q_len_per_req) contract, no unified counterpart (query "
        "lengths come from qo_indptr).",
    ),
    (
        "tests/attention/test_batch_decode_kernels.py::test_paged_decode_extreme_negative_logits",
        [],
        "out-of-scope",
        "decode-only API (BatchDecodeWithPagedKVCacheWrapper), not paged prefill.",
    ),
]
