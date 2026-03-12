"""Patch DRTK for Windows MSVC compilation.

1. Remove /GR- flag — disables RTTI, but PyTorch headers require it
   (dynamic_cast / dynamic_pointer_cast → C2280 errors without RTTI).
2. Replace /MT with /MD — DRTK uses static CRT (/MT) but nvcc compiles
   .cu files with /MD (dynamic CRT), causing LNK2038 mismatch.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

# Remove /GR- from Windows compiler flags — PyTorch requires RTTI
if '"/GR-"' in content:
    content = content.replace('"/GR-", ', '')
    print("Patched setup.py: removed /GR- (RTTI required by PyTorch)")
elif "/GR-" in content:
    content = content.replace("/GR-", "")
    print("Patched setup.py: removed /GR- (RTTI required by PyTorch)")
else:
    print("WARNING: /GR- not found in setup.py — source may have changed")

# Replace /MT (static CRT) with /MD (dynamic CRT) — nvcc uses /MD,
# mixing /MT and /MD causes linker error LNK2038
if '"/MT"' in content:
    content = content.replace('"/MT"', '"/MD"')
    print("Patched setup.py: replaced /MT with /MD (CRT must match nvcc's /MD)")
elif "/MT" in content:
    content = content.replace("/MT", "/MD")
    print("Patched setup.py: replaced /MT with /MD (CRT must match nvcc's /MD)")
else:
    print("WARNING: /MT not found in setup.py — source may have changed")

setup_file.write_text(content)

# Fix CUDA 12.4 CUB + __half operator conflict in interpolate_kernel.cu
# PyTorch adds -D__CUDA_NO_HALF_OPERATORS__ etc. on the command line, but
# CUB's dispatch_histogram.cuh and agent_sub_warp_merge_sort.cuh need __half
# comparison operators.  #undef at the top of the file overrides the -D flags.
interp_cu = Path("src/interpolate/interpolate_kernel.cu")
if interp_cu.exists():
    cu_content = interp_cu.read_text()
    undef_block = (
        "// -- cuda-wheels patch: re-enable half operators for CUB compat --\n"
        "#undef __CUDA_NO_HALF_OPERATORS__\n"
        "#undef __CUDA_NO_HALF2_OPERATORS__\n"
        "#undef __CUDA_NO_HALF_CONVERSIONS__\n"
        "#undef __CUDA_NO_BFLOAT16_CONVERSIONS__\n"
        "// -- end patch --\n\n"
    )
    if "__CUDA_NO_HALF_OPERATORS__" not in cu_content:
        # undefs not already present
        interp_cu.write_text(undef_block + cu_content)
        print("Patched interpolate_kernel.cu: #undef half operator macros for CUB compat")
    else:
        print("interpolate_kernel.cu already has half operator handling")
else:
    print("WARNING: interpolate_kernel.cu not found — source structure may have changed")
