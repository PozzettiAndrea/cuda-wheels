"""Patch o_voxel for wheel building as o_voxel_vb_ap:
1. Rename package from o_voxel to o_voxel_vb_ap
2. Fix MSVC compatibility (size_t narrowing)
3. Fix GCC-only CXX_FLAGS for Windows MSVC builds

Note: postprocess.py and rasterize.py (nvdiffrast deps) are already
removed in the PozzettiAndrea/Trellis.2.drtk source.
"""
import re
import os
from pathlib import Path

# --- 1. Rename package to o_voxel_vb_ap ---

# pyproject.toml
pyproject = Path("o-voxel/pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "o_voxel"', 'name = "o_voxel_vb_ap"')
pyproject.write_text(content)
print("Renamed package to o_voxel_vb_ap in pyproject.toml")

# setup.py
setup_file = Path("o-voxel/setup.py")
content = setup_file.read_text()
content = content.replace('name="o_voxel"', 'name="o_voxel_vb_ap"')
content = content.replace("name=\"o_voxel._C\"", "name=\"o_voxel_vb_ap._C\"")
content = content.replace("'o_voxel'", "'o_voxel_vb_ap'")
content = content.replace("'o_voxel.convert'", "'o_voxel_vb_ap.convert'")
content = content.replace("'o_voxel.io'", "'o_voxel_vb_ap.io'")
setup_file.write_text(content)
print("Renamed package to o_voxel_vb_ap in setup.py")

# Rename the actual package directory
src_dir = Path("o-voxel/o_voxel")
dst_dir = Path("o-voxel/o_voxel_vb_ap")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed o_voxel/ directory to o_voxel_vb_ap/")

# Update internal imports
for py_file in dst_dir.rglob("*.py"):
    content = py_file.read_text()
    if "from o_voxel" in content or "import o_voxel" in content:
        content = content.replace("from o_voxel", "from o_voxel_vb_ap")
        content = content.replace("import o_voxel", "import o_voxel_vb_ap")
        py_file.write_text(content)
print("Updated internal imports to o_voxel_vb_ap")

# --- 2. Fix size_t narrowing for MSVC ---
for f in ["o-voxel/src/io/filter_neighbor.cpp", "o-voxel/src/io/filter_parent.cpp"]:
    fpath = Path(f)
    if fpath.exists():
        content = fpath.read_text()
        content = re.sub(r'torch::zeros\(\{(\w+),\s*(\w+)\}', r'torch::zeros({(int64_t)\1, (int64_t)\2}', content)
        fpath.write_text(content)
print("Fixed size_t narrowing in filter_*.cpp")

svo_file = Path("o-voxel/src/io/svo.cpp")
if svo_file.exists():
    content = svo_file.read_text()
    content = re.sub(r'\{(\w+)\.size\(\)\}', r'{(int64_t)\1.size()}', content)
    svo_file.write_text(content)
    print("Fixed size_t narrowing in svo.cpp")

# --- 3. Fix GCC-only CXX_FLAGS for MSVC ---
content = setup_file.read_text()
old_cxx = """            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }"""
new_cxx = """            extra_compile_args={
                "cxx": ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"] + cc_flag,
            }"""
if old_cxx in content:
    content = content.replace(old_cxx, new_cxx)
    setup_file.write_text(content)
    print("Patched setup.py CXX_FLAGS for MSVC compatibility")
else:
    print("WARNING: Could not find CXX_FLAGS block in setup.py - source may have changed")
