"""Patch cumm to add bfloat16 GEMM kernel support.

cumm has all the low-level bf16 primitives (MMA instructions, tensor ops,
dtype definitions, numeric converters) but never instantiates bf16 GEMM
kernels. This patch adds:
  - bf16 Ampere TensorOp GEMM params (aligned, fast path)
  - bf16 Simt GEMM params (unaligned fallback for non-power-of-2 channels)

Also forces package name to 'cumm' (not 'cumm-cu{version}').
Also bundles libcudacxx (CCCL) headers into the wheel so NVRTC can find them
at runtime without needing a system CUDA installation.

bf16 tensor core MMA instructions require sm_80+ (Ampere).
bf16 Simt uses CUDA cores with f32 accumulation (works on any arch).
"""
import os
import re
import shutil
from pathlib import Path

# ─── Bundle libcudacxx headers into cumm/libcudacxx_include ───
# cumm's NVRTC looks for cumm/libcudacxx_include/ at runtime to resolve
# #include <cuda/std/limits> etc. Without this, users need a full CUDA
# toolkit on the system. We copy the headers from $CUDA_HOME/include.
cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
cuda_include = Path(cuda_home) / "include"
dest = Path("cumm") / "libcudacxx_include"

# The headers we need are under include/cuda/ and include/nv/
# (cuda/std/*, cuda/atomic, nv/target etc.)
copied = False
for subdir in ["cuda", "nv"]:
    src = cuda_include / subdir
    if src.exists():
        shutil.copytree(str(src), str(dest / subdir), dirs_exist_ok=True)
        copied = True

if copied:
    # Ensure the headers are included in the wheel via MANIFEST.in
    manifest = Path("MANIFEST.in")
    manifest_content = manifest.read_text() if manifest.exists() else ""
    manifest_line = "recursive-include cumm/libcudacxx_include *.h *.hpp"
    if manifest_line not in manifest_content:
        with manifest.open("a") as f:
            f.write(f"\n{manifest_line}\n")
    print(f"Bundled libcudacxx headers from {cuda_include} into cumm/libcudacxx_include/")
else:
    print(f"WARNING: Could not find CCCL headers at {cuda_include}/cuda/ - NVRTC may fail at runtime")

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
