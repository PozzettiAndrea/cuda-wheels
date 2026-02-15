"""Patch pytorch3d for Windows MSVC compatibility.

Fix: Replace unconditional -std=c++17 CXX flag with /std:c++17 on Windows.
pytorch3d already gates NVCC's -std=c++17 behind os.name != "nt",
but the CXX flag is unconditional.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

old = '    extra_compile_args = {"cxx": ["-std=c++17"]}'
new = '    extra_compile_args = {"cxx": ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]}'

if old in content:
    content = content.replace(old, new)
    setup_file.write_text(content)
    print("Patched pytorch3d CXX_FLAGS: -std=c++17 -> /std:c++17 on Windows")
else:
    print("WARNING: Could not find CXX_FLAGS block in setup.py - source may have changed")
