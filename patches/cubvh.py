"""Patch cubvh for torch 2.13 (same class as the cumesh fix).

torch 2.13's headers use C++20 features (designated initializers, bit-field
default member init); MSVC and Windows nvcc hard-error under c++17. cubvh
routes every std flag through one `cpp_standard` knob, so bump it. GCC and
nvcc accept c++20 on every CUDA line in the grid, so this is unconditional.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()
new = content.replace("cpp_standard = 17", "cpp_standard = 20")
if new != content:
    setup_file.write_text(new)
    print("cubvh patch: cpp_standard 17 -> 20")
else:
    print("cubvh patch: cpp_standard already != 17 (no-op)")
