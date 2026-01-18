"""Patch script for lietorch - downloads Eigen headers."""
import subprocess
import shutil
from pathlib import Path

# Download and extract Eigen
subprocess.run([
    "curl", "-sL",
    "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    "-o", "eigen.tar.gz"
], check=True)

subprocess.run(["tar", "-xzf", "eigen.tar.gz"], check=True)

# Rename to 'eigen' as expected by lietorch
eigen_dir = Path("eigen-3.4.0")
target = Path("eigen")
if target.exists():
    shutil.rmtree(target)
shutil.move(str(eigen_dir), str(target))

print("Eigen headers installed successfully")
