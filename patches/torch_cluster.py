"""Patch pytorch_cluster to add BFloat16 support to all CUDA kernels.

The upstream CUDA kernels only dispatch for Float, Double, and Half.
BFloat16 is missing, causing NotImplementedError when models run in bf16.

radius_cuda.cu already has bf16 support — this patch adds it to the rest:
fps_cuda.cu, knn_cuda.cu, nearest_cuda.cu, grid_cuda.cu, graclus_cuda.cu.
"""
from pathlib import Path

cuda_dir = Path("csrc/cuda")
patched = 0

# Generic patcher: find AT_DISPATCH_*_AND(Half, and replace with AND2(Half, BFloat16,
# This handles all variations: FLOATING_TYPES_AND, ALL_TYPES_AND, any scalar_type expression
for cu_file in sorted(cuda_dir.glob("*.cu")):
    content = cu_file.read_text()
    fname = cu_file.name

    # Skip radius_cuda.cu — already has BFloat16
    if fname == "radius_cuda.cu":
        continue

    # Match the pattern: _AND(at::ScalarType::Half, -> _AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    old = "_AND(at::ScalarType::Half,"
    new = "_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,"
    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        cu_file.write_text(content)
        patched += count
        print(f"Patched {fname}: {count} dispatch site(s) -> +BFloat16")

print(f"\nDone. Patched {patched} total dispatch sites.")
print("radius_cuda.cu already has BFloat16 support - no changes needed.")
