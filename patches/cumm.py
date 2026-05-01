"""Patch cumm to add bfloat16 GEMM kernel support.

cumm has all the low-level bf16 primitives (MMA instructions, tensor ops,
dtype definitions, numeric converters) but never instantiates bf16 GEMM
kernels. This patch adds:
  - bf16 Ampere TensorOp GEMM params (aligned, fast path)
  - bf16 Simt GEMM params (unaligned fallback for non-power-of-2 channels)

Also forces package name to 'cumm' (not 'cumm-cu{version}').
Also populates third_party/cccl/libcudacxx/include/ from $CUDA_HOME so
setup.py bundles libcudacxx headers into the wheel for NVRTC.

bf16 tensor core MMA instructions require sm_80+ (Ampere).
bf16 Simt uses CUDA cores with f32 accumulation (works on any arch).
"""
import os
import re
import shutil
from pathlib import Path

# ─── Populate third_party/cccl/libcudacxx/include from CUDA toolkit ───
# cumm's setup.py copies from third_party/cccl/libcudacxx/include/ into
# cumm/libcudacxx_include/ during build. But v0.8.2 has no cccl submodule,
# so we populate it from the CUDA toolkit headers.
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cuda_include = Path(cuda_home) / "include"
cccl_dest = Path("third_party") / "cccl" / "libcudacxx" / "include"

if not cccl_dest.exists() and cuda_include.exists():
    cccl_dest.mkdir(parents=True, exist_ok=True)
    for subdir in ["cuda", "nv"]:
        src = cuda_include / subdir
        if src.exists():
            shutil.copytree(str(src), str(cccl_dest / subdir), dirs_exist_ok=True)
    # Also copy ALL CUDA runtime headers so NVRTC works on Windows without
    # a system CUDA install. Cherry-picking individual headers is fragile —
    # just copy everything. Headers are platform-independent text files.
    for item in cuda_include.iterdir():
        if item.name in ["cuda", "nv"]:
            continue  # already copied above as CCCL/libcudacxx
        dest_item = cccl_dest / item.name
        if not dest_item.exists():
            if item.is_dir():
                shutil.copytree(str(item), str(dest_item), dirs_exist_ok=True)
            else:
                shutil.copy2(str(item), str(dest_item))
    print(f"Populated {cccl_dest} from {cuda_include} (incl. CUDA runtime headers)")
else:
    if cccl_dest.exists():
        print(f"third_party/cccl already populated, skipping")
    else:
        print(f"WARNING: Could not find CUDA headers at {cuda_include}")

# ─── 0. Force package name to 'cumm' (ignore CUMM_CUDA_VERSION) ───
setup_py = Path("setup.py")
if setup_py.exists():
    setup_content = setup_py.read_text()
    # Neutralize the CUDA version suffix logic:
    #   cuda_ver = os.getenv("CUMM_CUDA_VERSION", None)
    # Replace with always-None so RELEASE_NAME stays as 'cumm'
    setup_content = setup_content.replace(
        'os.getenv("CUMM_CUDA_VERSION"',
        'os.getenv("_DISABLED_CUMM_CUDA_VERSION"'
    )
    setup_py.write_text(setup_content)
    print("Patched setup.py: forced package name to 'cumm'")

main_py = Path("cumm/gemm/main.py")
content = main_py.read_text()

# ─── 1. Add SHUFFLE_AMPERE_PARAMS with bf16 GEMM kernels ───
# Insert after SHUFFLE_TURING_PARAMS definition.
# cumm's gen_shuffle_params signature (NOT gen_shuffle_params_v2):
#   gen_shuffle_params(ts, wts, dss: List[str], stage: int, algo, tensorop)

AMPERE_PARAMS_BLOCK = '''
# bf16 Ampere GEMM kernels (sm_80+)
# TensorOp((16, 8, 16)) is the Ampere-optimized shape for 16-bit types.
# Uses f32 accumulator for numerical stability (bf16 accumulation not supported by hardware).
SHUFFLE_AMPERE_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 64, 64),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 64),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
]

'''

# Find the end of SHUFFLE_TURING_PARAMS and insert AMPERE_PARAMS after it
# The list ends with a "]" line that is NOT inside a gen_shuffle_params call
# Look for the pattern: end of SHUFFLE_TURING_PARAMS list followed by a blank line
turing_match = re.search(
    r'(SHUFFLE_TURING_PARAMS\s*:.*?\n(?:.*\n)*?^]\s*$)',
    content,
    re.MULTILINE
)
if not turing_match:
    raise RuntimeError("Could not find SHUFFLE_TURING_PARAMS in cumm/gemm/main.py")

insert_pos = turing_match.end()
content = content[:insert_pos] + '\n' + AMPERE_PARAMS_BLOCK + content[insert_pos:]

# ─── 2. Populate the empty ampere_params list in GemmMainUnitTest.__init__ ───
# The non-debug branch has:
#     ampere_params = [
#
#     ]
# Replace with:
#     ampere_params = [
#         *SHUFFLE_AMPERE_PARAMS,
#     ]

content = re.sub(
    r'(ampere_params\s*=\s*\[)\s*\n\s*(\])',
    r'\1\n                    *SHUFFLE_AMPERE_PARAMS,\n                \2',
    content,
    count=1
)

# ─── 3. Add bf16 Simt fallback kernels to SHUFFLE_SIMT_PARAMS ───
# Simt kernels handle unaligned dimensions (e.g. 9-channel input).
# Without these, bf16 GEMM fails when LDA is not a multiple of 8.
# Mirrors the existing f16 Simt fallback entries in spconv/core.py.
# cumm's gen_shuffle_params: gen_shuffle_params(ts, wts, dss, stage, algo, tensorop)

BF16_SIMT_PARAMS = '''
    # bf16 Simt fallback kernels for misaligned dimensions
    *gen_shuffle_params(
        (128, 128, 8),
        (32, 64, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 64, 32),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (32, 32, 32),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 128, 16),
        (32, 64, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params(
        (64, 64, 8),
        (32, 32, 8), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Simt, None),
'''

# Find the closing ] of SHUFFLE_SIMT_PARAMS and insert bf16 entries before it
simt_match = re.search(
    r'(SHUFFLE_SIMT_PARAMS\s*:.*?\n(?:.*\n)*?)(^\]\s*$)',
    content,
    re.MULTILINE
)
if simt_match:
    insert_pos = simt_match.start(2)
    content = content[:insert_pos] + BF16_SIMT_PARAMS + content[insert_pos:]
    print("  - Added bf16 Simt fallback params to SHUFFLE_SIMT_PARAMS")
else:
    print("WARNING: Could not find SHUFFLE_SIMT_PARAMS closing bracket")

main_py.write_text(content)
print("Patched cumm/gemm/main.py with bf16 GEMM params")
print("  - Added SHUFFLE_AMPERE_PARAMS with 9 tile configs (bf16 TensorOp)")
print("  - Added bf16 Simt fallback params (5 tile configs, unaligned support)")
print("  - Populated ampere_params in GemmMainUnitTest.__init__")

# ─── 4. Fix zero_whole_storage_ pybind11 binding (missing default Context arg) ───
# The binding for zero_whole_storage_ doesn't declare a default Context arg,
# even though the C++ signature has `Context ctx = Context()`.
# This causes spconv's algo.py to fail when calling `tensor.zero_whole_storage_()`
# without arguments. Fix: add `py::arg("ctx") = tv::Context()` like clone_whole_storage has.
bind_py = Path("cumm/tensorview_bind.py")
bind_content = bind_py.read_text()
old_binding = '.def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_)'
new_binding = '.def("zero_whole_storage_", &tv::Tensor::zero_whole_storage_, py::arg("ctx") = tv::Context())'
if old_binding in bind_content:
    bind_content = bind_content.replace(old_binding, new_binding)
    bind_py.write_text(bind_content)
    print("Patched cumm/tensorview_bind.py: added default Context arg to zero_whole_storage_")
else:
    print("WARNING: Could not find zero_whole_storage_ binding to patch")

# ─── 5. Fix NVRTC include path resolution in constants.py ───
# cumm's constants.py picks the first existing `include/` directory from
# [site-packages/include, cumm/include]. Other packages (Eigen, embreex)
# create site-packages/include/, so cumm picks that instead of its own
# cumm/include/ which has tensorview headers. NVRTC then fails with
# "could not open source file tensorview/core/all.h".
# Fix: check for the actual header file, not just directory existence.
const_py = Path("cumm/constants.py")
const_content = const_py.read_text()
old_resolve = """TENSORVIEW_INCLUDE_PATH = _TENSORVIEW_INCLUDE_PATHS[0]
if not TENSORVIEW_INCLUDE_PATH.exists():
    for p in _TENSORVIEW_INCLUDE_PATHS[1:]:
        if p.exists():
            TENSORVIEW_INCLUDE_PATH = p

assert TENSORVIEW_INCLUDE_PATH.exists()"""
new_resolve = """_HEADER_SENTINEL = Path("tensorview") / "core" / "all.h"
TENSORVIEW_INCLUDE_PATH = None
for p in _TENSORVIEW_INCLUDE_PATHS:
    if (p / _HEADER_SENTINEL).exists():
        TENSORVIEW_INCLUDE_PATH = p
        break

assert TENSORVIEW_INCLUDE_PATH is not None and TENSORVIEW_INCLUDE_PATH.exists()"""
if old_resolve in const_content:
    const_content = const_content.replace(old_resolve, new_resolve)
    const_py.write_text(const_content)
    print("Patched cumm/constants.py: fixed NVRTC include path resolution")
else:
    print("WARNING: Could not find include path resolution block in constants.py")

# ─── 5b. Patch _locate_cudart_includes_for_nvrtc to fallback to bundled headers ───
# On Windows (no system CUDA, no triton), the function raises ValueError.
# Add a fallback that uses the bundled libcudacxx_include dir which now
# also contains CUDA runtime headers (cuda.h, cuda_fp16.h, etc.).
_common_py = Path("cumm/common.py")
_common_content = _common_py.read_text()
_old_raise = '''    raise ValueError("can't find cudart include for nvrtc, you must either install cuda to your system "
        "or use nvidia pip package (nvidia-cuda-runtime-cu12) (see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).")'''
_new_fallback = '''    # Fallback: use bundled headers in libcudacxx_include (contains CUDA runtime headers too)
    if TENSORVIEW_LIBCUDACXX_PATH.exists() and (Path(TENSORVIEW_LIBCUDACXX_PATH) / "cuda_fp16.h").exists():
        return [str(TENSORVIEW_LIBCUDACXX_PATH)]
    raise ValueError("can't find cudart include for nvrtc, you must either install cuda to your system "
        "or use nvidia pip package (nvidia-cuda-runtime-cu12) (see https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/).")'''
if _old_raise in _common_content:
    # Also need to import TENSORVIEW_LIBCUDACXX_PATH in the function scope
    _common_content = _common_content.replace(_old_raise, _new_fallback)
    # Ensure TENSORVIEW_LIBCUDACXX_PATH is imported at the top of the file
    if "from cumm.constants import" in _common_content and "TENSORVIEW_LIBCUDACXX_PATH" not in _common_content.split("def ")[0]:
        # It's already imported via constants — check if it's in the import line
        pass  # Already imported at module level in common.py
    _common_py.write_text(_common_content)
    print("Patched cumm/common.py: added libcudacxx_include fallback for NVRTC cudart headers")
else:
    print("WARNING: Could not find cudart ValueError raise in common.py")

# ─── 6. Fix NVRTC "qualified name is not allowed" in dtype headers ───
# Under __CUDACC_RTC__, CUDA_NAMESPACE_STD expands to cuda::std.
# `namespace cuda::std {` is C++17 nested namespace syntax which NVRTC
# rejects in its default C++14 mode. Fix: use C++14-compatible nesting.
_dtype_dir = Path("include") / "tensorview" / "gemm" / "dtypes"
_patched_ns = 0
for _hdr in ["half.h", "bfloat16.h", "tf32.h", "float8.h"]:
    _hdr_path = _dtype_dir / _hdr
    if not _hdr_path.exists():
        print(f"WARNING: {_hdr_path} not found, skipping namespace fix")
        continue
    _hdr_content = _hdr_path.read_text()
    if "namespace CUDA_NAMESPACE_STD {" not in _hdr_content:
        continue
    _hdr_content = _hdr_content.replace(
        "namespace CUDA_NAMESPACE_STD {",
        "namespace cuda { namespace std {"
    )
    _hdr_content = re.sub(
        r'\}\s*//\s*namespace std\b',
        '}} // namespace cuda::std',
        _hdr_content
    )
    _hdr_path.write_text(_hdr_content)
    _patched_ns += 1
print(f"Patched {_patched_ns} dtype headers: fixed NVRTC namespace syntax (C++17 -> C++14)")

# ─── 7. Fix NVRTC missing std::abs/min/max in nvrtc_std.h ───
# NVRTC's std namespace (via cuda::std) doesn't include abs, min, max.
# spconv/cumm codegen emits std::abs(), std::min(), std::max() in NVRTC
# kernels. Fix: add shims to nvrtc_std.h so they're available.
_nvrtc_std = Path("include") / "tensorview" / "core" / "nvrtc_std.h"
if _nvrtc_std.exists():
    _nvrtc_content = _nvrtc_std.read_text()
    _shims = '''
// Shims for std::abs, std::min, std::max — not provided by cuda::std/libcudacxx.
// These are normally in <cstdlib>/<algorithm> which NVRTC doesn't have.
#if defined(__CUDACC_RTC__)
namespace std {
    template<typename T>
    __device__ inline T abs(T x) { return x < T(0) ? -x : x; }
    template<typename T>
    __device__ inline const T& min(const T& a, const T& b) { return a < b ? a : b; }
    template<typename T>
    __device__ inline const T& max(const T& a, const T& b) { return a > b ? a : b; }
}
#endif
'''
    if "std::abs" not in _nvrtc_content:
        # Insert after the M_PI define block, before the #endif that closes #ifndef __APPLE__
        _marker = '#define M_PI 3.14159265358979323846\n#endif'
        if _marker in _nvrtc_content:
            _nvrtc_content = _nvrtc_content.replace(
                _marker,
                '#define M_PI 3.14159265358979323846\n#endif\n' + _shims
            )
            _nvrtc_std.write_text(_nvrtc_content)
            print("Patched nvrtc_std.h: added std::abs/min/max shims")
    else:
        print("nvrtc_std.h already has std::abs, skipping")

# ─── 6. Fix NVRTC "qualified name is not allowed" in dtype headers ───
# Under __CUDACC_RTC__, CUDA_NAMESPACE_STD expands to cuda::std.
# `namespace cuda::std {` is C++17 nested namespace syntax which NVRTC
# rejects in its default C++14 mode. Fix: use C++14-compatible
# `namespace cuda { namespace std {` and `}}` closing.
_dtype_dir = Path("include") / "tensorview" / "gemm" / "dtypes"
_patched_ns = 0
for _hdr in ["half.h", "bfloat16.h", "tf32.h", "float8.h"]:
    _hdr_path = _dtype_dir / _hdr
    if not _hdr_path.exists():
        print(f"WARNING: {_hdr_path} not found, skipping namespace fix")
        continue
    _hdr_content = _hdr_path.read_text()
    if "namespace CUDA_NAMESPACE_STD {" not in _hdr_content:
        continue
    # Replace opening: namespace CUDA_NAMESPACE_STD { -> namespace cuda { namespace std {
    _hdr_content = _hdr_content.replace(
        "namespace CUDA_NAMESPACE_STD {",
        "namespace cuda { namespace std {"
    )
    # Replace closing: } // namespace std -> }} // namespace cuda::std
    # (some files have extra spaces before //)
    _hdr_content = re.sub(
        r'\}\s*//\s*namespace std\b',
        '}} // namespace cuda::std',
        _hdr_content
    )
    _hdr_path.write_text(_hdr_content)
    _patched_ns += 1
print(f"Patched {_patched_ns} dtype headers: fixed NVRTC namespace syntax (C++17 -> C++14)")

# ─── 8. Fix cu++filt unavailable on Windows — inline Itanium demangler ───
# cumm calls cu++filt to demangle CUDA symbol names. On Windows, cu++filt
# doesn't exist and tv.cufilt() uses __cxa_demangle (GCC-only).
# NVRTC always uses Itanium ABI mangling on all platforms.
# Fix: inline a simple Itanium demangler for the common case (namespace::name).
_nvrtc_init = Path("cumm/nvrtc/__init__.py")
_nvrtc_init_content = _nvrtc_init.read_text()
_old_cufilt = '''        res = subprocess.check_output(["cu++filt",
                                       name]).decode("utf-8").strip()
        return res'''
_new_cufilt = '''        # Inline Itanium ABI demangler for simple names (namespace::var)
        # NVRTC uses Itanium mangling on all platforms including Windows.
        # Pattern: _ZN<len1><name1><len2><name2>...E
        if name.startswith("_ZN") and name.endswith("E"):
            parts = []
            i = 3  # skip _ZN
            s = name[:-1]  # strip trailing E
            try:
                while i < len(s):
                    n = 0
                    while i < len(s) and s[i].isdigit():
                        n = n * 10 + int(s[i])
                        i += 1
                    if n > 0 and i + n <= len(s):
                        parts.append(s[i:i+n])
                        i += n
                    else:
                        break
                if parts and i == len(s):
                    return "::".join(parts)
            except Exception:
                pass
        try:
            res = subprocess.check_output(["cu++filt",
                                           name]).decode("utf-8").strip()
            return res
        except (FileNotFoundError, subprocess.CalledProcessError):
            return name'''
if _old_cufilt in _nvrtc_init_content:
    _nvrtc_init_content = _nvrtc_init_content.replace(_old_cufilt, _new_cufilt)
    _nvrtc_init.write_text(_nvrtc_init_content)
    print("Patched cumm/nvrtc/__init__.py: inline Itanium demangler for Windows")
