import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


requirements = ["torch"]


def get_extensions():
    srcs = ["cc_torch/connected_components.cu"]
    extra_compile_args = {
        "cxx": [],
        "nvcc": [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }

    CC = os.environ.get("CC", None)
    if CC is not None:
        extra_compile_args["nvcc"].append("-ccbin={}".format(CC))

    ext_modules = [
        CUDAExtension(
            "cc_torch._C",
            srcs,
            include_dirs=[],
            define_macros=[],
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    # Meta Data
    name="cc_torch",
    version="0.2",
    description="Connected Components Labeling for PyTorch",
    # Package Info
    zip_safe=False,
    packages=find_packages(exclude=("tests",)),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
)
