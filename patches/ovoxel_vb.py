"""Patch ovoxel_vb (visualbruno fork) for wheel building:
1. Rename package from o_voxel to o_voxel_vb to avoid conflicts
2. Add batched BVH queries to avoid GPU timeout (issue #19)
3. Fix MSVC compatibility (size_t narrowing)
4. Fix GCC-only CXX_FLAGS for Windows MSVC builds

Note: visualbruno's fork already fixes double literal suffixes and
      removes cumesh/flex_gemm git URL deps from pyproject.toml.
"""
import re
from pathlib import Path

# --- 1. Rename package to o_voxel_vb ---

# pyproject.toml
pyproject = Path("o-voxel/pyproject.toml")
content = pyproject.read_text()
content = content.replace('name = "o_voxel"', 'name = "o_voxel_vb"')
pyproject.write_text(content)
print("Renamed package to o_voxel_vb in pyproject.toml")

# setup.py
setup_file = Path("o-voxel/setup.py")
content = setup_file.read_text()
content = content.replace('name="o_voxel"', 'name="o_voxel_vb"')
content = content.replace("name=\"o_voxel._C\"", "name=\"o_voxel_vb._C\"")
content = content.replace("'o_voxel'", "'o_voxel_vb'")
content = content.replace("'o_voxel.convert'", "'o_voxel_vb.convert'")
content = content.replace("'o_voxel.io'", "'o_voxel_vb.io'")
setup_file.write_text(content)
print("Renamed package to o_voxel_vb in setup.py")

# Rename the actual package directory
src_dir = Path("o-voxel/o_voxel")
dst_dir = Path("o-voxel/o_voxel_vb")
if src_dir.exists() and not dst_dir.exists():
    src_dir.rename(dst_dir)
    print("Renamed o_voxel/ directory to o_voxel_vb/")

# Update internal imports
for py_file in dst_dir.rglob("*.py"):
    content = py_file.read_text()
    if "from .. import _C" in content or "from o_voxel" in content or "import o_voxel" in content:
        content = content.replace("from o_voxel", "from o_voxel_vb")
        content = content.replace("import o_voxel", "import o_voxel_vb")
        py_file.write_text(content)
print("Updated internal imports to o_voxel_vb")

# --- 2. Batched BVH queries ---
postprocess = dst_dir / "postprocess.py"
if postprocess.exists():
    content = postprocess.read_text()

    batched_func = '''
def _batched_unsigned_distance(bvh, positions, batch_size=500000, return_uvw=False):
    """Batch unsigned_distance queries to avoid GPU kernel timeout.
    See: https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2/issues/19
    """
    N = positions.shape[0]
    if N <= batch_size:
        return bvh.unsigned_distance(positions, return_uvw=return_uvw)
    import torch
    distances_list, face_id_list, uvw_list = [], [], []
    for i in range(0, N, batch_size):
        d, f, u = bvh.unsigned_distance(positions[i:min(i+batch_size, N)], return_uvw=return_uvw)
        distances_list.append(d)
        face_id_list.append(f)
        if return_uvw:
            uvw_list.append(u)
    return (
        torch.cat(distances_list),
        torch.cat(face_id_list),
        torch.cat(uvw_list) if return_uvw else None
    )

'''

    # Replace bare flex_gemm/cumesh imports with try/except fallbacks
    # so the wheel works whether the _vb or non-vb variants are installed
    content = content.replace(
        'from flex_gemm.ops.grid_sample import grid_sample_3d',
        'try:\n    from flex_gemm_vb.ops.grid_sample import grid_sample_3d\nexcept ImportError:\n    from flex_gemm.ops.grid_sample import grid_sample_3d',
    )
    content = content.replace(
        'import cumesh\n',
        'try:\n    import cumesh_vb as cumesh\nexcept ImportError:\n    import cumesh\n',
    )

    content = re.sub(r'(import cumesh\n)', r'\1' + batched_func, content)
    content = content.replace(
        '_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)',
        '_, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)'
    )
    postprocess.write_text(content)
    print("Patched postprocess.py for batched BVH queries and flexible imports")

# --- 3. Fix size_t narrowing for MSVC ---
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

# --- 4. Fix GCC-only CXX_FLAGS for MSVC ---
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
