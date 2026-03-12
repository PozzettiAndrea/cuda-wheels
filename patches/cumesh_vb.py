"""Patch cumesh_vb (visualbruno fork) for wheel building:
1. Fetch missing Eigen submodule (cubvh committed directly, not as git submodule)
2. Rename package from cumesh to cumesh_vb to avoid conflicts
3. Fix GCC-only CXX_FLAGS for Windows MSVC builds
"""
import os
import subprocess
from pathlib import Path

# --- 0. Fetch Eigen (nested submodule not auto-fetched) ---
# visualbruno committed third_party/cubvh directly into the tree rather than
# as a git submodule. The cubvh directory has its own .gitmodules pointing to
# eigen, but since cubvh isn't a submodule, --recursive doesn't fetch it.
eigen_dir = Path("third_party/cubvh/third_party/eigen")
if not eigen_dir.exists() or not any(eigen_dir.iterdir()):
    eigen_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://gitlab.com/libeigen/eigen.git", str(eigen_dir)],
        check=True,
    )
    print(f"Cloned Eigen into {eigen_dir}")

# --- 1. Rename package to cumesh_vb ---

# pyproject.toml
pyproject = Path("pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "cumesh"', 'name = "cumesh_vb"')
pyproject.write_text(content)
print("Renamed package to cumesh_vb in pyproject.toml")

# setup.py
setup_file = Path("setup.py")
content = setup_file.read_text()
content = content.replace('name="cumesh"', 'name="cumesh_vb"')
content = content.replace("'cumesh'", "'cumesh_vb'")
content = content.replace('name="cumesh._C"', 'name="cumesh_vb._C"')
content = content.replace("name='cumesh._cubvh'", "name='cumesh_vb._cubvh'")
content = content.replace("name='cumesh._xatlas'", "name='cumesh_vb._xatlas'")
setup_file.write_text(content)
print("Renamed package to cumesh_vb in setup.py")

# Rename the actual package directory
src_dir = Path("cumesh")
dst_dir = Path("cumesh_vb")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed cumesh/ directory to cumesh_vb/")

# --- 2. Fix CXX/NVCC flags for Windows ---
# MSVC: -O3 -> /O2, -std=c++20 -> /std:c++17
# nvcc on Windows: c++20 triggers cub header bugs in CUDA 12.4, downgrade to c++17
content = setup_file.read_text()
content = content.replace(
    '"cxx": ["-O3", "-std=c++20"]',
    '"cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++20"]',
)
content = content.replace(
    '"nvcc": ["-O3","-std=c++20"]',
    '"nvcc": ["-O3", "-std=c++17", "--extended-lambda"] if os.name == "nt" else ["-O3", "-std=c++20"]',
)
setup_file.write_text(content)
print("Patched CXX/NVCC flags for Windows (c++17, /O2)")
