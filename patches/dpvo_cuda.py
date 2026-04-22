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
content = content.replace("name='dpvo'", "name='dpvo_cuda'")
# Keep packages=find_packages() so the dpvo/ package ships and the
# compiled ext_modules (cuda_corr, cuda_ba, lietorch_backends) get bundled
# into the wheel. Without this the wheel is essentially empty.
setup_py.write_text(content)

print("setup.py patched: renamed to dpvo-cuda")

# Fix PyTorch API compatibility: .type() -> .scalar_type()
# This is needed for PyTorch 2.0+ which deprecated tensor.type()
files_to_patch = [
    # DPVO altcorr files
    Path("dpvo/altcorr/correlation_kernel.cu"),
    # DPVO fastba files - all of them
    Path("dpvo/fastba/ba.cpp"),
    Path("dpvo/fastba/ba_cuda.cu"),
    Path("dpvo/fastba/block_e.cu"),
    # Lietorch files (also use deprecated .type() API)
    Path("dpvo/lietorch/src/lietorch_cpu.cpp"),
    Path("dpvo/lietorch/src/lietorch_gpu.cu"),
]

for src_file in files_to_patch:
    if src_file.exists():
        content = src_file.read_text()
        original = content

        # Replace .type() with .scalar_type() in AT_DISPATCH macros
        content = content.replace(".type()", ".scalar_type()")

        # Fix Windows linker error: mutable_data_ptr<T> template not exported
        # Use data_ptr<T> instead which works on both platforms
        content = content.replace("mutable_data_ptr<", "data_ptr<")

        # Fix Windows linker error: long type not exported from PyTorch DLL
        # packed_accessor32<long,...> uses mutable_data_ptr<long> internally
        # Replace long with int64_t which is properly exported
        content = content.replace("<long,", "<int64_t,")

        # Also fix .item<long>() which has same Windows export issue
        content = content.replace(".item<long>()", ".item<int64_t>()")

        if content != original:
            src_file.write_text(content)
            print(f"Patched {src_file}")

print("PyTorch API compatibility patches applied")

# Fix MSVC compound literal syntax error in ba_cuda.cu
# MSVC doesn't support C99 compound literals: (float[6]){...}
# Replace with individual element assignments
ba_cuda = Path("dpvo/fastba/ba_cuda.cu")
if ba_cuda.exists():
    content = ba_cuda.read_text()
    original = content

    # Replace compound literal assignments with individual element assignments
    # Line ~323: Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};
    content = content.replace(
        "Jj = (float[6]){fx*W*d, 0, fx*-X*W*d2, fx*-X*Y*d2, fx*(1+X*X*d2), fx*-Y*d};",
        "Jj[0]=fx*W*d; Jj[1]=0; Jj[2]=fx*-X*W*d2; Jj[3]=fx*-X*Y*d2; Jj[4]=fx*(1+X*X*d2); Jj[5]=fx*-Y*d;"
    )

    # Line ~331: Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};
    content = content.replace(
        "Jj = (float[6]){0, fy*W*d, fy*-Y*W*d2, fy*(-1-Y*Y*d2), fy*(X*Y*d2), fy*X*d};",
        "Jj[0]=0; Jj[1]=fy*W*d; Jj[2]=fy*-Y*W*d2; Jj[3]=fy*(-1-Y*Y*d2); Jj[4]=fy*(X*Y*d2); Jj[5]=fy*X*d;"
    )

    if content != original:
        ba_cuda.write_text(content)
        print("Patched ba_cuda.cu: fixed MSVC compound literal syntax")

print("All patches applied successfully")
