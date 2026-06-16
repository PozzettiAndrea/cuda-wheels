# Package Build Notes

Per-package CUDA constraints, patch summaries, and build quirks.

## CUDA Version Support

All 26 packages support CUDA 12.4 through 13.0 (cu124, cu126, cu128, cu129, cu130).
No packages have an upper CUDA version limit that would block cu129.

### Packages with CUDA-version-specific behavior

| Package | Detail |
|---------|--------|
| flash_attn | setup.py explicitly handles CUDA 12.9+ (SM 101/Thor, family-specific 100f/120f gencode flags) |
| sageattention | Needs CUDA >= 12.0; SM 12.0 (Blackwell) needs >= 12.8. No upper bound. |
| sageattn3 | Blackwell-only (SM 10.0, 12.0). Needs CUDA >= 12.8. Patch replaces runtime GPU detection with TORCH_CUDA_ARCH_LIST parsing. |
| cumesh/cumesh_vb | Patch handles `CUDART_VERSION >= 12090` for CUB's DeviceReduce API change |

## Per-Package Details

### cc_torch
- **Source:** ronghanghu/cc_torch
- **Quirks:** None. Simple PyTorch extension wrapper.

### cubvh
- **Source:** ashawkey/cubvh
- **Quirks:** `max_jobs: 2`, recursive clone. No patches.

### cumesh
- **Source:** JeffreyXiang/CuMesh
- **Quirks:** Needs `curand_dev cufft_dev nvtx` CUDA components. `max_jobs: 2`. Patch fixes CUB API for CUDA 12.9+ on MSVC.

### cumesh_vb
- **Source:** visualbruno/CuMesh
- **Quirks:** Same CUDA components as cumesh. Patch fetches Eigen submodule, renames package to cumesh_vb, MSVC CXX/NVCC flag fixes. Forces C++17 for nvcc (CUB bugs in CUDA 12.4 with C++20).

### cumm
- **Source:** FindDefinition/cumm (v0.8.2)
- **Quirks:** Generates BF16 GEMM kernels at build time. Needs `pccm>=0.4.16 ccimport>=0.4.4`. Patch adds BF16 Ampere TensorOp + Simt fallback configs, fixes pybind11 `zero_whole_storage_` binding.

### detectron2
- **Source:** facebookresearch/detectron2 (v0.6)
- **Quirks:** No patches. Standard PyTorch extension build.

### dpvo_cuda (dpvo-cuda)
- **Source:** princeton-vl/DPVO
- **Quirks:** Downloads Eigen 3.4.0 headers. Patch fixes PyTorch 2.0+ API (`.type()` -> `.scalar_type()`), MSVC compound literals, DLL exports.

### faithc_aot
- **Source:** PozzettiAndrea/faithc-aot (tag `v1.5.0-aot`) — packaging fork bundling FaithContour (Luo-Yihao/FaithC, Apache-2.0) + Atom3d (Luo-Yihao/Atom3d, MIT) into one wheel.
- **Quirks:** No patch (AOT committed in the fork). Builds **4** CUDAExtensions: `faithcontour._C` + `atom3d.kernels.{cumtv,bvh,floodfill}_cuda`. Upstream Atom3d JIT-compiles at runtime; the fork converts all kernels to AOT and the loaders import the prebuilt `.so` first (JIT fallback only for source installs). No extra CUDA components (torch + standard CUDA runtime only). `torch_scatter` left external (faithcontour has a pure-torch fallback; install the prebuilt `torch_scatter` wheel alongside). **Currently linux-only + trimmed to cu128/torch2.8/py3.12-3.13** for first validation — see the `build_matrix` block in `packages/faithc_aot.yml` to expand to the full grid.

### flash_attn
- **Source:** Dao-AILab/flash-attention (v2.8.3)
- **Quirks:** SM >= 8.0 (Ampere+). `max_jobs: 1`. `free_disk_space: true`. Patch inits only csrc/cutlass submodule (skips composable_kernel — ROCm only, breaks Windows paths). Bridges TORCH_CUDA_ARCH_LIST -> FLASH_ATTN_CUDA_ARCHS.
- **Arch list:** cu124-cu126 use default `"8.0 9.0"`. cu128+ use `"8.0 9.0 10.0 12.0"` (Blackwell).

### flexgemm / flexgemm_ap / flexgemm_vb
- **Source:** JeffreyXiang/FlexGEMM (main), PozzettiAndrea/FlexGEMM-ap, visualbruno/FlexGEMM
- **Quirks:** Need `cufft_dev nvtx` CUDA components. Triton dependency (platform-specific: triton on Linux, triton-windows on Windows). flexgemm_vb renames to flex_gemm_vb.

### gsplat
- **Source:** nerfstudio-project/gsplat (v1.5.3)
- **Quirks:** Recursive clone. Patch: MSVC -O3 -> /O2.

### nvdiffrast
- **Source:** NVlabs/nvdiffrast (v0.4.0)
- **Quirks:** No patches.

### nvdiffrec_render
- **Source:** NVlabs/nvdiffrec
- **Quirks:** `build_subdir: nvdiffrec_render`. Patch restructures render/ as standalone package, creates setup.py, MSVC CXX flags.

### ovoxel / ovoxel_vb
- **Source:** microsoft/TRELLIS.2 / PozzettiAndrea/Trellis.2.sparseflex
- **Quirks:** `build_subdir: o-voxel`. Recursive clone. Patch removes git URL deps, adds batched BVH queries (batch_size=500000) to avoid kernel timeout. MSVC fixes for double literals and size_t narrowing.

### pyg_lib
- **Source:** pyg-team/pyg-lib (0.5.0)
- **Quirks:** Recursive clone. No patches.

### pytorch3d
- **Source:** facebookresearch/pytorch3d (v0.7.9)
- **Quirks:** Recursive clone. Patch: MSVC -std=c++17 -> /std:c++17.

### sageattention
- **Source:** thu-ml/SageAttention (v2.2.0)
- **Quirks:** `max_jobs: 1`. Patch fixes arch parser (space-separated TORCH_CUDA_ARCH_LIST), MSVC CXX flags, Windows ABI guard, nvcc --threads=8->4, sm_90-only gencode for _qattn_sm90.
- **Arch list:** cu124-cu126 use `"8.0 8.6 8.9 9.0"`. cu128+ use `"8.0 8.6 8.9 9.0 10.0 12.0"` (Blackwell).

### sageattn3
- **Source:** thu-ml/SageAttention (v2.2.0), `build_subdir: sageattention3_blackwell`
- **Quirks:** Blackwell-only (SM 10.0, 12.0, 12.1). `max_jobs: 1`. Complex MSVC patches: kernel_traits.h dependent-name workaround, kernel_ws.h parameter passing (pointer vs CUTE_GRID_CONSTANT), launch.h device-side parameter packing.
- **Arch list:** Global `"10.0 12.0"`. Only builds for cu128+.

### spconv
- **Source:** traveller59/spconv (v2.3.8)
- **Quirks:** Depends on cumm (`extra_deps: "pccm>=0.4.16 ccimport>=0.4.4 cumm"`). Patch adds BF16 sparse convolution (Simt fallback + Ampere TensorOp + ImplGemm params).

### torch_cluster
- **Source:** rusty1s/pytorch_cluster (1.6.3+)
- **Quirks:** Patch adds BFloat16 dispatch to all CUDA kernels (fps, knn, nearest, grid, graclus).

### torch_generic_nms
- **Source:** ronghanghu/torch_generic_nms
- **Quirks:** No patches.

### torch_scatter
- **Source:** rusty1s/pytorch_scatter (2.1.2+)
- **Quirks:** No patches.

### torch_sparse
- **Source:** rusty1s/pytorch_sparse (0.6.18+)
- **Quirks:** Recursive clone. Patch adds BFloat16 dispatch to spmm_cuda.cu.

### torch_spline_conv
- **Source:** rusty1s/pytorch_spline_conv
- **Quirks:** No patches.

## Upstream PyTorch Phantom Gaps

These CUDA/torch/python/platform combos are in our build matrix but PyTorch never published a wheel:

| Combo | Reason |
|-------|--------|
| cu124/torch2.5/cp313/windows | PyTorch only published cp313 Linux for torch 2.5+cu124 |
| cu129/torch2.10/*/windows | torch 2.10+cu129 is linux-only upstream |
