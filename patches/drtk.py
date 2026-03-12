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

# Re-enable half/bfloat16 operators in all .cu files.
# PyTorch adds -D__CUDA_NO_HALF_OPERATORS__ etc. on the command line, which
# breaks CUB headers (dispatch_histogram.cuh, agent_sub_warp_merge_sort.cuh)
# and disables native half-precision ops.  #undef at the top overrides the -D.
UNDEF_BLOCK = (
    "// -- cuda-wheels patch: re-enable half/bfloat16 operators --\n"
    "#undef __CUDA_NO_HALF_OPERATORS__\n"
    "#undef __CUDA_NO_HALF2_OPERATORS__\n"
    "#undef __CUDA_NO_HALF_CONVERSIONS__\n"
    "#undef __CUDA_NO_BFLOAT16_CONVERSIONS__\n"
    "// -- end patch --\n\n"
)
patched_cu = 0
for cu_file in Path("src").rglob("*.cu"):
    cu_content = cu_file.read_text()
    if "__CUDA_NO_HALF_OPERATORS__" not in cu_content:
        cu_file.write_text(UNDEF_BLOCK + cu_content)
        patched_cu += 1
print(f"Patched {patched_cu} .cu files: #undef half/bfloat16 operator macros")
