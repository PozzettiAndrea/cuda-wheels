# cuda-wheels

Pre-built CUDA Python wheels for ML/3D packages that are painful to compile from source.

**[Package Index](https://pozzettiandrea.github.io/cuda-wheels)** · **[Package Index v2](https://pozzettiandrea.github.io/cuda-wheels/v2/)** · **[Dashboard](https://pozzettiandrea.github.io/cuda-wheels/dashboard/)** · **[Install Helper](https://pozzettiandrea.github.io/cuda-wheels/dashboard/install.html)** · **[PyTorch CUDA Wheel Matrix (P.C.W.M)](https://pozzettiandrea.github.io/cuda-wheels/matrix/)** · **[PyTorch Arches Matrix (P.A.M)](https://pozzettiandrea.github.io/cuda-wheels/arches/)**

## Built Packages

| Package | Source |
|---------|--------|
| sageattention | [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) |
| gsplat | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) |
| nvdiffrast | [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast) |
| pytorch3d | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) |
| cumesh | [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) |
| cubvh | [ashawkey/cubvh](https://github.com/ashawkey/cubvh) |
| ovoxel | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| flexgemm | [JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM) |
| nvdiffrec_render | [NVlabs/nvdiffrec](https://github.com/NVlabs/nvdiffrec) |
| torch_generic_nms | [ronghanghu/torch_generic_nms](https://github.com/ronghanghu/torch_generic_nms) |
| cc_torch | [ronghanghu/cc_torch](https://github.com/ronghanghu/cc_torch) |
| spconv | [traveller59/spconv](https://github.com/traveller59/spconv) |
| flash_attn | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) |
| torch_cluster | [pyg-team/pytorch_cluster](https://github.com/pyg-team/pytorch_cluster) |
| torch_scatter | [pyg-team/pytorch_scatter](https://github.com/pyg-team/pytorch_scatter) |
| torch_sparse | [pyg-team/pytorch_sparse](https://github.com/pyg-team/pytorch_sparse) |
| natten | [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN) |

## Usage

Used by [comfy-env](https://github.com/PozzettiAndrea/comfy-env) to automatically resolve and install CUDA packages for ComfyUI custom nodes.

## Adding a package

See [packages/README.md](packages/README.md) for build config format.

## License

MIT
