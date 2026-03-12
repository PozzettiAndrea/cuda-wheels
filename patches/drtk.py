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
