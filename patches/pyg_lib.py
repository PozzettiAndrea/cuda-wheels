"""Patch installed torch's CMake config to fix the nvToolsExt link target.

PyTorch 2.4/2.5/2.6 ship a Caffe2/public/cuda.cmake that hard-references
CUDA::nvToolsExt. CUDA Toolkit >=12.5 dropped the standalone nvToolsExt
library (only header-only NVTX3 remains), so FindCUDAToolkit no longer
creates that imported target and downstream find_package(Torch) blows up
during configure with:

    The link interface of target "torch::nvtoolsext" contains:
      CUDA::nvToolsExt
    but the target was not found.

PyTorch 2.7+ already replaced this with CUDA::nvtx3, so the patch is a
no-op on those versions.
"""
from pathlib import Path
import torch

cuda_cmake = Path(torch.__file__).parent / "share/cmake/Caffe2/public/cuda.cmake"
content = cuda_cmake.read_text()
n = content.count("CUDA::nvToolsExt")
if n:
    cuda_cmake.write_text(content.replace("CUDA::nvToolsExt", "CUDA::nvtx3"))
    print(f"Patched {cuda_cmake}: rewrote {n} CUDA::nvToolsExt -> CUDA::nvtx3")
else:
    print(f"No CUDA::nvToolsExt references in {cuda_cmake} (torch>=2.7?), skipping.")
