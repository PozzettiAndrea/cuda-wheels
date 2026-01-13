"""Patch ovoxel for wheel building:
1. Remove git URL dependencies from pyproject.toml
2. Add batched BVH queries to avoid GPU timeout (issue #19)
"""
import re
from pathlib import Path

# Remove git URL deps from pyproject.toml
pyproject = Path("o-voxel/pyproject.toml")
content = pyproject.read_text()
content = re.sub(r'.*cumesh@.*\n', '', content)
content = re.sub(r'.*flex_gemm@.*\n', '', content)
pyproject.write_text(content)
print("Removed git URL dependencies from pyproject.toml")

# Patch postprocess.py for batched BVH queries
postprocess = Path("o-voxel/o_voxel/postprocess.py")
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

content = re.sub(r'(import cumesh\n)', r'\1' + batched_func, content)
content = content.replace(
    '_, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)',
    '_, face_id, uvw = _batched_unsigned_distance(bvh, valid_pos, return_uvw=True)'
)
postprocess.write_text(content)
print("Patched postprocess.py for batched BVH queries")
