"""Patch SageAttention3 (Blackwell) for cross-compilation on CI runners.

1. Replace torch.cuda.get_device_capability() with TORCH_CUDA_ARCH_LIST parsing.
2. Replace GCC-specific CXX_FLAGS with MSVC equivalents on Windows.
3. Guard _GLIBCXX_USE_CXX11_ABI on Windows.
"""
from pathlib import Path

setup_file = Path("sageattention3_blackwell/setup.py")
content = setup_file.read_text()

# 1. Replace GPU detection with TORCH_CUDA_ARCH_LIST parsing.
# The upstream code calls torch.cuda.get_device_capability() which fails
# on CI runners without a GPU. Instead, parse the arch list from the env var
# that the build-wheel action sets.
old_gpu_detect = """    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    elif (cc_major, cc_minor) == (12, 1):  # sm_121
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_121a,code=sm_121a")
    else:
        raise RuntimeError("Unsupported GPU")"""

new_gpu_detect = """    cc_flag = []
    # Parse TORCH_CUDA_ARCH_LIST for cross-compilation (no GPU needed)
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    arch_map = {
        "10.0": ("compute_100a", "sm_100a"),
        "12.0": ("compute_120a", "sm_120a"),
        "12.1": ("compute_121a", "sm_121a"),
    }
    for item in arch_list_env.replace(",", " ").replace(";", " ").split():
        item = item.strip()
        if item in arch_map:
            compute, sm = arch_map[item]
            cc_flag.extend(["-gencode", f"arch={compute},code={sm}"])
    if not cc_flag:
        raise RuntimeError(
            f"No supported Blackwell architectures found in TORCH_CUDA_ARCH_LIST={arch_list_env!r}. "
            "Expected one of: 10.0, 12.0, 12.1"
        )"""

if old_gpu_detect in content:
    content = content.replace(old_gpu_detect, new_gpu_detect)
    print("Patched GPU detection -> TORCH_CUDA_ARCH_LIST parsing")
else:
    # Try without the sm_121 block (v2.2.0 may not have it)
    old_gpu_detect_v2 = """    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    else:
        raise RuntimeError("Unsupported GPU")"""

    if old_gpu_detect_v2 in content:
        content = content.replace(old_gpu_detect_v2, new_gpu_detect)
        print("Patched GPU detection -> TORCH_CUDA_ARCH_LIST parsing (v2.2.0 variant)")
    else:
        print("WARNING: Could not find GPU detection block - source may have changed")

# 2. Platform-aware CXX flags for Windows MSVC support.
old_cxx = '''                "cxx": ["-O3", "-std=c++17"],'''
new_cxx = '''                "cxx": ["/O2", "/std:c++17"] if platform.system() == "Windows" else ["-O3", "-std=c++17"],'''

if old_cxx in content:
    content = content.replace(old_cxx, new_cxx)
    # Add platform import if not already present
    if "import platform" not in content:
        content = content.replace("import warnings", "import warnings\nimport platform")
    print("Patched CXX_FLAGS for MSVC compatibility")
else:
    print("WARNING: Could not find CXX_FLAGS block - source may have changed")

# 3. Guard FORCE_CXX11_ABI on non-Windows (GCC/libstdc++ only).
old_abi = """    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True"""
new_abi = """    if FORCE_CXX11_ABI and platform.system() != "Windows":
        torch._C._GLIBCXX_USE_CXX11_ABI = True"""

if old_abi in content:
    content = content.replace(old_abi, new_abi)
    print("Patched FORCE_CXX11_ABI to skip on Windows")
else:
    print("WARNING: Could not find FORCE_CXX11_ABI block - source may have changed")

setup_file.write_text(content)
