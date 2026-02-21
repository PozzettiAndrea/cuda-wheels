import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_flags = ['/O2', '-DNVDR_TORCH'] if os.name == 'nt' else ['-O3', '-DNVDR_TORCH']

setup(
    name='nvdiffrec_render',
    version='0.0.1',
    description='Render utilities from NVlabs/nvdiffrec for differentiable rendering',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='nvdiffrec_render.renderutils._C',
            sources=[
                'nvdiffrec_render/renderutils/c_src/mesh.cu',
                'nvdiffrec_render/renderutils/c_src/loss.cu',
                'nvdiffrec_render/renderutils/c_src/bsdf.cu',
                'nvdiffrec_render/renderutils/c_src/normal.cu',
                'nvdiffrec_render/renderutils/c_src/cubemap.cu',
                'nvdiffrec_render/renderutils/c_src/common.cpp',
                'nvdiffrec_render/renderutils/c_src/torch_bindings.cpp',
            ],
            extra_compile_args={
                'cxx': cxx_flags,
                'nvcc': ['-O3', '-DNVDR_TORCH'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=['torch', 'numpy'],
)
