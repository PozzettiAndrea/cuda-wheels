"""Patch installed torch's CMake config to neutralize the nvToolsExt block.

PyTorch 2.4/2.5/2.6 ship a Caffe2/public/cuda.cmake that:
  1. Errors out (FATAL_ERROR) if CUDA::nvToolsExt isn't an existing target.
  2. Creates torch::nvtoolsext as an INTERFACE IMPORTED lib that links to
     CUDA::nvToolsExt.

CUDA Toolkit >=12.5 dropped the standalone nvToolsExt library, and our
setup-cuda action's cache key doesn't include extra-components, so even
adding `nvtx` to extra_cuda_components doesn't reliably make the target
appear when the CUDA cache is restored. Easiest fix: rewrite the block
to be a no-op IMPORTED interface library — `find_package(Torch)` then
succeeds and any downstream link to `torch::nvtoolsext` is empty.

PyTorch 2.7+ already replaced this with CUDA::nvtx3, so this patch is a
no-op there.
"""
from pathlib import Path
import torch

OLD = """if(NOT TARGET CUDA::nvToolsExt)
  message(FATAL_ERROR "Failed to find nvToolsExt")
endif()"""

NEW = """# patched: nvToolsExt presence check removed (CUDA>=12.5 dropped it)
if(FALSE)
endif()"""

OLD_LINK = """# nvToolsExt
add_library(torch::nvtoolsext INTERFACE IMPORTED)
set_property(
    TARGET torch::nvtoolsext PROPERTY INTERFACE_LINK_LIBRARIES
    CUDA::nvToolsExt)"""

NEW_LINK = """# nvToolsExt (patched: empty INTERFACE so downstream links are no-ops)
add_library(torch::nvtoolsext INTERFACE IMPORTED)"""

cuda_cmake = Path(torch.__file__).parent / "share/cmake/Caffe2/public/cuda.cmake"
text = cuda_cmake.read_text()
original = text

if OLD in text:
    text = text.replace(OLD, NEW, 1)
if OLD_LINK in text:
    text = text.replace(OLD_LINK, NEW_LINK, 1)

if text != original:
    cuda_cmake.write_text(text)
    print(f"Patched {cuda_cmake}: neutralized nvToolsExt FATAL_ERROR + link interface")
else:
    print(f"No nvToolsExt block found in {cuda_cmake} (torch>=2.7?), skipping.")
