"""Patch script for dpvo-cuda - downloads Eigen headers and fixes PyTorch API compatibility."""
import subprocess
import shutil
from pathlib import Path

# Download and extract Eigen (required for cuda_ba and lietorch_backends)
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

# Rename package to dpvo-cuda
setup_py = Path("setup.py")
content = setup_py.read_text()
content = content.replace("name='dpvo'", "name='dpvo-cuda'")
content = content.replace("packages=find_packages()", "packages=[]")
setup_py.write_text(content)

print("setup.py patched: renamed to dpvo-cuda")

# Fix PyTorch API compatibility: .type() -> .scalar_type()
# This is needed for PyTorch 2.0+ which deprecated tensor.type()
cuda_files = [
    Path("dpvo/altcorr/correlation_kernel.cu"),
    Path("dpvo/fastba/ba_cuda.cu"),
]

for cuda_file in cuda_files:
    if cuda_file.exists():
        content = cuda_file.read_text()
        # Replace .type() with .scalar_type() in AT_DISPATCH macros
        new_content = content.replace(".type()", ".scalar_type()")
        if new_content != content:
            cuda_file.write_text(new_content)
            print(f"Patched {cuda_file}: .type() -> .scalar_type()")

print("PyTorch API compatibility patches applied")
