"""Patch flex_gemm_vb (visualbruno fork) for wheel building:
1. Rename package from flex_gemm to flex_gemm_vb to avoid conflicts
2. Update autotuner cache path and walk_package references

Note: visualbruno's fork already has triton-windows platform-specific deps
      and MSVC-compatible CXX_FLAGS in setup.py.
"""
from pathlib import Path

# --- 1. Rename package to flex_gemm_vb ---

# pyproject.toml
pyproject = Path("pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "flex_gemm"', 'name = "flex_gemm_vb"')
pyproject.write_text(content)
print("Renamed package to flex_gemm_vb in pyproject.toml")

# setup.py
setup_file = Path("setup.py")
content = setup_file.read_text()
content = content.replace('name="flex_gemm"', 'name="flex_gemm_vb"')
# Replace package list entries
content = content.replace('"flex_gemm"', '"flex_gemm_vb"')
content = content.replace('"flex_gemm.', '"flex_gemm_vb.')
# Replace source file paths (flex_gemm/kernels/cuda/...)
content = content.replace('"flex_gemm/', '"flex_gemm_vb/')
# Replace cache path
content = content.replace('~/.flex_gemm', '~/.flex_gemm_vb')
setup_file.write_text(content)
print("Renamed package to flex_gemm_vb in setup.py")

# Rename the actual package directory
src_dir = Path("flex_gemm")
dst_dir = Path("flex_gemm_vb")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed flex_gemm/ directory to flex_gemm_vb/")

# --- 2. Update autotuner walk_package references ---
autotuner = dst_dir / "utils" / "autotuner.py"
if autotuner.exists():
    content = autotuner.read_text()
    content = content.replace("walk_package('flex_gemm'", "walk_package('flex_gemm_vb'")
    autotuner.write_text(content)
    print("Updated walk_package references in autotuner.py")
