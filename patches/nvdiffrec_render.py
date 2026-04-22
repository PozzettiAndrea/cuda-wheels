"""Patch nvdiffrec to create nvdiffrec_render package:
1. Restructure render/ as standalone package
2. Create setup.py for CUDA extension (with MSVC-compatible flags)
3. Patch ops.py to use pre-built extension with JIT fallback
"""
import shutil
from pathlib import Path

# Create package structure
pkg_dir = Path("nvdiffrec_render")
pkg_dir.mkdir(exist_ok=True)
shutil.copytree("render", pkg_dir / "nvdiffrec_render")
print("Created package structure: nvdiffrec_render/nvdiffrec_render/")

# Create setup.py
setup_py = '''import os
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
'''
(pkg_dir / "setup.py").write_text(setup_py)
print("Created setup.py")

# Patch ops.py to try pre-built extension first
ops_py = pkg_dir / "nvdiffrec_render/renderutils/ops.py"
content = ops_py.read_text()

# Insert after "if _cached_plugin is not None: return _cached_plugin"
patch = '''
    # Try pre-built extension first (wheel installations)
    try:
        from . import _C
        _cached_plugin = _C
        return _cached_plugin
    except ImportError:
        pass  # Fall through to JIT compilation
'''

content = content.replace(
    'if _cached_plugin is not None:\n        return _cached_plugin',
    'if _cached_plugin is not None:\n        return _cached_plugin\n' + patch
)
ops_py.write_text(content)
print("Patched ops.py to use pre-built extension with JIT fallback")
