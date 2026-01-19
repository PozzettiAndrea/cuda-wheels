"""Patch script for dpvo-cuda - downloads Eigen and patches setup.py."""
import subprocess
import shutil
from pathlib import Path

# Download and extract Eigen (required for cuda_ba)
subprocess.run([
    "curl", "-sL",
    "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    "-o", "eigen.tar.gz"
], check=True)
subprocess.run(["tar", "-xzf", "eigen.tar.gz"], check=True)

# Move to thirdparty/eigen as DPVO expects
thirdparty = Path("thirdparty")
thirdparty.mkdir(exist_ok=True)
eigen_target = thirdparty / "eigen-3.4.0"
if eigen_target.exists():
    shutil.rmtree(eigen_target)
shutil.move("eigen-3.4.0", str(eigen_target))

print("Eigen headers installed successfully")

# Patch setup.py to only build cuda_corr and cuda_ba (skip lietorch_backends)
setup_py = Path("setup.py")
content = setup_py.read_text()

# Remove lietorch_backends extension from ext_modules list
new_content = content.replace(
    """CUDAExtension('lietorch_backends',
            sources=[
                'lietorch/src/lietorch.cpp',
                'lietorch/src/lietorch_gpu.cu',
                'lietorch/src/lietorch_cpu.cpp'],
            include_dirs=[
                ROOT / 'lietorch/include',
                ROOT / 'thirdparty/eigen-3.4.0'],
            extra_compile_args=extra_compile_args),""",
    ""
)

# Change package name and remove Python packages (we only want the extensions)
new_content = new_content.replace("name='dpvo'", "name='dpvo-cuda'")
new_content = new_content.replace("packages=find_packages()", "packages=[]")

setup_py.write_text(new_content)
print("setup.py patched: removed lietorch_backends, renamed to dpvo-cuda")
