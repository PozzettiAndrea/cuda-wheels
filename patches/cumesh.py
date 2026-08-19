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

# --- CUDA 13.2 / torch 2.13 compatibility (found by the first cu132/2.13 run) ---

# (1) torch 2.13's headers use C++20 features (designated initializers,
# bit-field default member init). GCC tolerates them under c++17; MSVC and
# Windows nvcc hard-error. cumesh pins c++17 in four places in setup.py.
_setup = Path("setup.py")
_t = _setup.read_text()
_t2 = _t.replace("c++17", "c++20")
if _t2 != _t:
    _setup.write_text(_t2)
    print("cumesh patch: setup.py c++17 -> c++20")

# (2) CCCL 3.x (shipped with CUDA 13.2) removed DeviceScan::ExclusiveSum's
# 4-arg IN-PLACE overload (d_temp, bytes, d_data, num). The arguments then
# bind to the new env-based overload and produce garbage template errors
# (InputIteratorT=nullptr_t, NumItemsT=int*). The 5-arg form with
# d_in == d_out is still in-place and works on every CUDA line, so duplicate
# the data argument. Idempotent: 5-arg calls are left alone.
import re as _re

def _fix_inplace_exclusive_sum(text: str) -> tuple[str, int]:
    out, pos, fixed = [], 0, 0
    for m in _re.finditer(r"cub::DeviceScan::ExclusiveSum\(", text):
        start, depth, j = m.end(), 1, m.end()
        while depth and j < len(text):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
            j += 1
        argstr = text[start:j-1]
        args, d, last = [], 0, 0
        for k, ch in enumerate(argstr):
            if ch in "([":
                d += 1
            elif ch in ")]":
                d -= 1
            elif ch == "," and d == 0:
                args.append(argstr[last:k])
                last = k + 1
        args.append(argstr[last:])
        if len(args) == 4:
            data = args[2]
            new_argstr = ",".join([args[0], args[1], data, " " + data.strip(), args[3]])
            out.append(text[pos:start]); out.append(new_argstr); out.append(")")
            pos = j
            fixed += 1
    out.append(text[pos:])
    return "".join(out), fixed

_total = 0
for _f in ["src/clean_up.cu", "src/connectivity.cu", "src/remesh/svox2vert.cu",
           "src/simplify.cu", "src/atlas.cu"]:
    _path = Path(_f)
    if not _path.exists():
        continue
    _text = _path.read_text()
    _new, _n = _fix_inplace_exclusive_sum(_text)
    if _n:
        _path.write_text(_new)
        _total += _n
        print(f"cumesh patch: {_f}: {_n} in-place ExclusiveSum call(s) -> 5-arg form")
print(f"cumesh patch: {_total} CCCL-3.x call sites fixed")
