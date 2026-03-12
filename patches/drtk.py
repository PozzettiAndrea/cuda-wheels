"""Patch DRTK for Windows MSVC compilation.

Remove /GR- flag which disables RTTI — PyTorch headers require RTTI
(dynamic_cast / dynamic_pointer_cast) and fail with C2280 errors without it.
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

setup_file.write_text(content)
