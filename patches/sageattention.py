"""Patch SageAttention for Windows MSVC compatibility.

Fix: Replace GCC-specific CXX_FLAGS with MSVC equivalents on Windows.
SageAttention's setup.py hardcodes GCC flags (-g, -O3, -fopenmp, -lgomp, -std=c++17)
which MSVC silently ignores, resulting in unoptimized Windows builds.
Also skips the GCC-only _GLIBCXX_USE_CXX11_ABI flag on Windows.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

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

setup_file.write_text(content)
