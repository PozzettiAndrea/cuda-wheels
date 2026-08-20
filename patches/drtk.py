"""Patch DRTK for Windows MSVC compilation.

1. Remove /GR- flag — disables RTTI, but PyTorch headers require it
   (dynamic_cast / dynamic_pointer_cast -> C2280 errors without RTTI).
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

# Downgrade CUDA -std=c++20 to c++17 — nvcc's C++20 parser misparses
# PyTorch template expressions like .to<List<Elem>>() in ivalue_inl.h
if '"-std=c++20"' in content:
    content = content.replace('"-std=c++20"', '"-std=c++17"')
    print("Patched setup.py: -std=c++20 -> -std=c++17 (nvcc C++20 breaks PyTorch headers)")
elif "-std=c++20" in content:
    content = content.replace("-std=c++20", "-std=c++17")
    print("Patched setup.py: -std=c++20 -> -std=c++17 (nvcc C++20 breaks PyTorch headers)")
else:
    print("INFO: -std=c++20 not found in setup.py (already c++17 or unset)")

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

# --- torch 2.13 compatibility (same class as the cumesh fix) ---
# torch 2.13 headers use C++20 features; MSVC and nvcc hard-error below it.
# Linux pins c++17; the win32 cxx block passes NO /std at all (cl defaults
# too old). nvcc already gets -std=c++20 upstream.
content = setup_file.read_text()
c2 = content.replace('"-std=c++17"', '"-std=c++20"')
if '"/std:c++20"' not in c2:
    c2 = c2.replace('"/EHsc"', '"/EHsc", "/std:c++20"', 1)
if c2 != content:
    setup_file.write_text(c2)
    print("drtk patch: C++20 std for linux cxx + win32 cxx")
