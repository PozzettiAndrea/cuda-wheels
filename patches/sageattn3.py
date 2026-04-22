"""Patch SageAttention3 (Blackwell) for cross-compilation on CI runners.

1. Replace torch.cuda.get_device_capability() with TORCH_CUDA_ARCH_LIST parsing.
2. Replace GCC-specific CXX_FLAGS with proper MSVC equivalents on Windows.
3. Guard _GLIBCXX_USE_CXX11_ABI on Windows.
4. Patch kernel_traits.h for MSVC dependent-name workaround.
5. Patch kernel_ws.h for MSVC kernel parameter passing (pointer vs CUTE_GRID_CONSTANT).
6. Patch launch.h for MSVC device-side parameter packing.

Patches 4-6 based on mengqin/SageAttention commit 8bb81e4.
"""
from pathlib import Path

# ===========================================================================
# setup.py patches
# ===========================================================================

setup_file = Path("sageattention3_blackwell/setup.py")
content = setup_file.read_text()

# 1. Replace GPU detection with TORCH_CUDA_ARCH_LIST parsing.
# The upstream code calls torch.cuda.get_device_capability() which fails
# on CI runners without a GPU. Instead, parse the arch list from the env var
# that the build-wheel action sets.
old_gpu_detect = """    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    elif (cc_major, cc_minor) == (12, 1):  # sm_121
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_121a,code=sm_121a")
    else:
        raise RuntimeError("Unsupported GPU")"""

new_gpu_detect = """    cc_flag = []
    # Parse TORCH_CUDA_ARCH_LIST for cross-compilation (no GPU needed)
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    arch_map = {
        "10.0": ("compute_100a", "sm_100a"),
        "12.0": ("compute_120a", "sm_120a"),
        "12.1": ("compute_121a", "sm_121a"),
    }
    for item in arch_list_env.replace(",", " ").replace(";", " ").split():
        item = item.strip()
        if item in arch_map:
            compute, sm = arch_map[item]
            cc_flag.extend(["-gencode", f"arch={compute},code={sm}"])
    if not cc_flag:
        raise RuntimeError(
            f"No supported Blackwell architectures found in TORCH_CUDA_ARCH_LIST={arch_list_env!r}. "
            "Expected one of: 10.0, 12.0, 12.1"
        )"""

if old_gpu_detect in content:
    content = content.replace(old_gpu_detect, new_gpu_detect)
    print("Patched GPU detection -> TORCH_CUDA_ARCH_LIST parsing")
else:
    # Try without the sm_121 block (v2.2.0 may not have it)
    old_gpu_detect_v2 = """    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("12.8"):
        raise RuntimeError("Sage3 is only supported on CUDA 12.8 and above")
    cc_major, cc_minor = torch.cuda.get_device_capability()
    if (cc_major, cc_minor) == (10, 0):  # sm_100
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_100a,code=sm_100a")
    elif (cc_major, cc_minor) == (12, 0):  # sm_120
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_120a,code=sm_120a")
    else:
        raise RuntimeError("Unsupported GPU")"""

    if old_gpu_detect_v2 in content:
        content = content.replace(old_gpu_detect_v2, new_gpu_detect)
        print("Patched GPU detection -> TORCH_CUDA_ARCH_LIST parsing (v2.2.0 variant)")
    else:
        print("WARNING: Could not find GPU detection block - source may have changed")

# 2. Platform-aware CXX flags and nvcc -Xcompiler forwarding for Windows MSVC.
# The mengqin fork adds /Zc:__cplusplus, /bigobj, /permissive- and forwards
# CXX flags to nvcc via -Xcompiler= so the host compiler gets them during
# nvcc's cudafe pass.
old_cxx_block = """    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )"""

new_cxx_block = """    if os.name == "nt":
        nvcc_flags += ["-D_WIN32=1", "-DUSE_CUDA=1"]
        cxx_flags = ["/std:c++17", "/Zc:__cplusplus", "/bigobj", "/MD", "/permissive-"]
        nvcc_flags += [f"-Xcompiler={flag}" for flag in cxx_flags]
        cxx_flags += ["/O2"]
    else:
        cxx_flags = ["-O3", "-std=c++17"]

    ext_modules.append(
        CUDAExtension(
            name="fp4attn_cuda",
            sources=["sageattn3/blackwell/api.cu"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="fp4quant_cuda",
            sources=["sageattn3/quantization/fp4_quantization_4d.cu"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": append_nvcc_threads(
                    nvcc_flags + ["-DEXECMODE=0"] + cc_flag
                ),
            },
            include_dirs=include_dirs,
            # Without this we get and error about cuTensorMapEncodeTiled not defined
            libraries=["cuda"]
        )
    )"""

if old_cxx_block in content:
    content = content.replace(old_cxx_block, new_cxx_block)
    print("Patched CXX_FLAGS + nvcc -Xcompiler forwarding for MSVC")
else:
    print("WARNING: Could not find CXX_FLAGS/ext_modules block - source may have changed")

# 3. Guard FORCE_CXX11_ABI on non-Windows (GCC/libstdc++ only).
old_abi = """    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True"""
new_abi = """    if FORCE_CXX11_ABI and os.name != "nt":
        torch._C._GLIBCXX_USE_CXX11_ABI = True"""

if old_abi in content:
    content = content.replace(old_abi, new_abi)
    print("Patched FORCE_CXX11_ABI to skip on Windows")
else:
    print("WARNING: Could not find FORCE_CXX11_ABI block - source may have changed")

setup_file.write_text(content)

# ===========================================================================
# kernel_traits.h patch — MSVC dependent-name workaround
# ===========================================================================
# MSVC's cl.exe (via nvcc cudafe) can't resolve dependent names like
# `typename BlkScaledConfig::LayoutSF` in class templates. The fix inlines
# the type definitions under #if defined(_MSC_VER).

kt_file = Path("sageattention3_blackwell/sageattn3/blackwell/kernel_traits.h")
kt = kt_file.read_text()

old_blkscaled = """\
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));"""

new_blkscaled = """\
#if defined(_MSC_VER)
    using BlkScaledConfig = ::flash::BlockScaledConfig<SFVectorSize>;
    // Inline the definitions to avoid MSVC dependent-name quirks
    using SfAtom = Layout<
        Shape< Shape<_16, _4>, Shape<Int<SFVectorSize>, Int<4>>>,
        Stride<Stride<_16, _4>, Stride<_0, _1>>
    >;
    using LayoutSF = decltype(
      blocked_product(
        SfAtom{},
        make_layout(
          make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
          make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0))
        )
      )
    );
    using SmemLayoutAtomSFQ = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(::flash::BlockScaledConfig<SFVectorSize>::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
#else
    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
#endif"""

if old_blkscaled in kt:
    kt = kt.replace(old_blkscaled, new_blkscaled)
    print("Patched kernel_traits.h: MSVC BlkScaledConfig workaround")
else:
    print("WARNING: Could not find BlkScaledConfig block in kernel_traits.h - source may have changed")

kt_file.write_text(kt)

# ===========================================================================
# kernel_ws.h patch — MSVC kernel parameter passing
# ===========================================================================
# MSVC can't pass over-aligned structs via CUTE_GRID_CONSTANT by-value kernel
# params. The fix extracts the body into a __device__ impl function taking
# references, with MSVC using pointer params and non-MSVC using the original.

kw_file = Path("sageattention3_blackwell/sageattn3/blackwell/kernel_ws.h")
kw = kw_file.read_text()

old_kernel_sig = """\
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,
                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,
                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params
                    ) {"""

new_kernel_sig = """\
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__device__ inline void
compute_attn_ws_impl(Flash_fwd_params const &params,
                     typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const &mainloop_params,
                     typename CollectiveEpilogueFwd<Ktraits>::Params const &epilogue_params,
                     typename TileScheduler::Params const &scheduler_params) {"""

old_kernel_end = """\
}

} // namespace flash"""

new_kernel_end = """\
}

#if defined(_MSC_VER)
// MSVC requires special handling for kernel parameters to avoid alignment issues.

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
compute_attn_ws(Flash_fwd_params const *params,
                typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const *mainloop_params,
                typename CollectiveEpilogueFwd<Ktraits>::Params const *epilogue_params,
                typename TileScheduler::Params const *scheduler_params) {
    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(
        *params, *mainloop_params, *epilogue_params, *scheduler_params);
}

#else

template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)
    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,
                    CUTE_GRID_CONSTANT
                        typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const
                            mainloop_params,
                    CUTE_GRID_CONSTANT
                        typename CollectiveEpilogueFwd<Ktraits>::Params const
                            epilogue_params,
                    CUTE_GRID_CONSTANT
                        typename TileScheduler::Params const scheduler_params) {

    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(
        params, mainloop_params, epilogue_params, scheduler_params);
}

#endif  // _MSC_VER

} // namespace flash"""

if old_kernel_sig in kw:
    kw = kw.replace(old_kernel_sig, new_kernel_sig)
    print("Patched kernel_ws.h: extracted compute_attn_ws_impl")
else:
    print("WARNING: Could not find compute_attn_ws signature in kernel_ws.h - source may have changed")

if old_kernel_end in kw:
    kw = kw.replace(old_kernel_end, new_kernel_end)
    print("Patched kernel_ws.h: added MSVC/non-MSVC kernel wrappers")
else:
    print("WARNING: Could not find kernel end block in kernel_ws.h - source may have changed")

kw_file.write_text(kw)

# ===========================================================================
# launch.h patch — MSVC device-side parameter packing
# ===========================================================================
# MSVC can't pass over-aligned CUTLASS params directly to kernels (C2719).
# The fix packs params into a device-allocated struct and passes pointers.

lh_file = Path("sageattention3_blackwell/sageattn3/blackwell/launch.h")
lh = lh_file.read_text()

old_launch = (
    "    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};\n"
    "    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);\n"
    "    \n"
    "    C10_CUDA_KERNEL_LAUNCH_CHECK();"
)

new_launch = """\
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};

#if defined(_MSC_VER)
    // MSVC: Use parameter packing to pass parameters to avoid over-aligned parameters reporting a C2719 error.

    struct DeviceParamsPack {
        Flash_fwd_params params;
        typename CollectiveMainloop::Params mainloop;
        typename CollectiveEpilogue::Params epilogue;
        typename Scheduler::Params scheduler;
    };

    DeviceParamsPack h_pack{params, mainloop_params, epilogue_params, scheduler_params};

    DeviceParamsPack *d_pack = nullptr;
    C10_CUDA_CHECK(cudaMallocAsync(&d_pack, sizeof(DeviceParamsPack), stream));
    C10_CUDA_CHECK(cudaMemcpyAsync(
        d_pack, &h_pack, sizeof(DeviceParamsPack),
        cudaMemcpyHostToDevice, stream));

    char *base_h = reinterpret_cast<char*>(&h_pack);
    auto off_params   = reinterpret_cast<char*>(&h_pack.params)    - base_h;
    auto off_mainloop = reinterpret_cast<char*>(&h_pack.mainloop)  - base_h;
    auto off_epilogue = reinterpret_cast<char*>(&h_pack.epilogue)  - base_h;
    auto off_sched    = reinterpret_cast<char*>(&h_pack.scheduler) - base_h;

    char *base_d = reinterpret_cast<char*>(d_pack);

    auto d_params = reinterpret_cast<Flash_fwd_params*>(base_d + off_params);
    auto d_mainloop_params =
        reinterpret_cast<typename CollectiveMainloop::Params *>(base_d + off_mainloop);
    auto d_epilogue_params =
        reinterpret_cast<typename CollectiveEpilogue::Params *>(base_d + off_epilogue);
    auto d_scheduler_params =
        reinterpret_cast<typename Scheduler::Params *>(base_d + off_sched);

    cutlass::launch_kernel_on_cluster(
        launch_params, kernel,
        d_params, d_mainloop_params, d_epilogue_params, d_scheduler_params);

    C10_CUDA_CHECK(cudaFreeAsync(d_pack, stream));

#else
    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);
#endif

    C10_CUDA_KERNEL_LAUNCH_CHECK();"""

if old_launch in lh:
    lh = lh.replace(old_launch, new_launch)
    print("Patched launch.h: MSVC device-side parameter packing")
else:
    print("WARNING: Could not find launch_kernel_on_cluster block in launch.h - source may have changed")

lh_file.write_text(lh)
