"""Patch SageAttention for build compatibility.

1. Fix arch parser to handle space-separated TORCH_CUDA_ARCH_LIST.
2. Replace GCC-specific CXX_FLAGS with MSVC equivalents on Windows.
3. Skip _GLIBCXX_USE_CXX11_ABI on Windows.
4. Reduce nvcc --threads from 8 to 4 to avoid OOM on CI runners.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

# Fix TORCH_CUDA_ARCH_LIST parser: upstream only handles comma/semicolon
# separators but PyTorch convention uses spaces (e.g. "7.0 7.5 8.0 8.6 8.9 9.0").
# Without this, HAS_SM80/89/90 are never set and qattn CUDA kernels are skipped.
old_parser = '    for item in arch_list_env.replace(",", ";").split(";"):'
new_parser = '    for item in arch_list_env.replace(",", " ").replace(";", " ").split():'
if old_parser in content:
    content = content.replace(old_parser, new_parser)
    print("Patched arch parser to handle space-separated TORCH_CUDA_ARCH_LIST")
else:
    print("WARNING: Could not find arch parser line - source may have changed")

# Replace hardcoded GCC CXX_FLAGS with platform-aware version
old_flags = '    CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]'

new_flags = """    import platform
    if platform.system() == "Windows":
        CXX_FLAGS = ["/O2", "/Zi", "/openmp", "/std:c++17", "-DENABLE_BF16"]
    else:
        CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]"""

if old_flags in content:
    content = content.replace(old_flags, new_flags)
    print("Patched CXX_FLAGS for MSVC compatibility")
else:
    print("WARNING: Could not find CXX_FLAGS block - source may have changed")

# Skip _GLIBCXX_USE_CXX11_ABI on Windows (GCC/libstdc++ only)
old_abi = """    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]"""

new_abi = """    if platform.system() != "Windows":
        ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
        CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
        NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]"""

if old_abi in content:
    content = content.replace(old_abi, new_abi)
    print("Patched _GLIBCXX_USE_CXX11_ABI to skip on Windows")
else:
    print("WARNING: Could not find ABI block - source may have changed")

# Reduce nvcc --threads from 8 to 4 to avoid OOM on GitHub runners
content = content.replace('"--threads=8"', '"--threads=4"')
print("Patched nvcc --threads=8 -> --threads=4")

setup_file.write_text(content)
