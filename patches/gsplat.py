"""Patch gsplat for Windows MSVC compatibility.

Fix: Replace unconditional -O3 CXX flag with /O2 on Windows.
gsplat already handles OpenMP and warning flags per-platform, but
the base -O3 optimization flag is GCC-only.
"""
from pathlib import Path
import re

setup_file = Path("setup.py")
content = setup_file.read_text()

old = '    extra_compile_args = {"cxx": ["-O3"]}'
new = '    extra_compile_args = {"cxx": ["/O2"] if os.name == "nt" else ["-O3"]}'

if old in content:
    content = content.replace(old, new)
    setup_file.write_text(content)
    print("Patched gsplat CXX_FLAGS: -O3 -> /O2 on Windows")
else:
    print("WARNING: Could not find CXX_FLAGS block in setup.py - source may have changed")
