"""Patch pytorch_sparse to add BFloat16 support to CUDA kernels.

spmm_cuda.cu only dispatches for Float, Double, and Half.
BFloat16 is missing, causing NotImplementedError when models run in bf16.
"""
from pathlib import Path

cuda_dir = Path("csrc/cuda")
patched = 0

for cu_file in sorted(cuda_dir.glob("*.cu")):
    content = cu_file.read_text()
    fname = cu_file.name

    old = "_AND(at::ScalarType::Half,"
    new = "_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,"
    count = content.count(old)
    if count > 0:
        content = content.replace(old, new)
        cu_file.write_text(content)
        patched += count
        print(f"Patched {fname}: {count} dispatch site(s) -> +BFloat16")

if patched == 0:
    print("No dispatch sites needed patching - BFloat16 may already be supported.")
else:
    print(f"\nDone. Patched {patched} total dispatch sites.")
