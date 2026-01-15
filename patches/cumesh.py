"""Patch cumesh for Windows MSVC compatibility.

Fix: Move #if CUDART_VERSION directives outside CUDA_CHECK macro calls.
MSVC doesn't handle preprocessor directives inside macro arguments correctly.
"""
from pathlib import Path

atlas_file = Path("src/atlas.cu")
content = atlas_file.read_text()

# The issue: #if directives inside CUDA_CHECK() macro don't work on MSVC
# Solution: Define a type alias before the macro call, reuse it in both calls

old_block = '''    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
#if CUDART_VERSION >= 12090
        ::cuda::std::plus(),
#else
        cub::Sum(),
#endif
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
#if CUDART_VERSION >= 12090
        ::cuda::std::plus(),
#else
        cub::Sum(),
#endif
        M
    ));'''

new_block = '''#if CUDART_VERSION >= 12090
    using ReduceOp = ::cuda::std::plus<>;
#else
    using ReduceOp = cub::Sum;
#endif
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        nullptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
        ReduceOp(),
        M
    ));
    mesh.cub_temp_storage.resize(temp_storage_bytes);
    CUDA_CHECK(cub::DeviceReduce::ReduceByKey(
        mesh.cub_temp_storage.ptr, temp_storage_bytes,
        reinterpret_cast<uint64_t*>(mesh.temp_storage.ptr),
        mesh.atlas_chart_adj.ptr,
        cu_sorted_lengths,
        mesh.atlas_chart_adj_length.ptr,
        cu_num_chart_adjs,
        ReduceOp(),
        M
    ));'''

if old_block in content:
    content = content.replace(old_block, new_block)
    atlas_file.write_text(content)
    print("Fixed MSVC preprocessor issue in atlas.cu")
else:
    print("WARNING: Could not find expected code block in atlas.cu - may already be patched or source changed")
