"""Patch flash-attn for CI builds.

1. Init only csrc/cutlass submodule (skip composable_kernel — ROCm only, breaks Windows due to long filenames).
2. Bridge TORCH_CUDA_ARCH_LIST → FLASH_ATTN_CUDA_ARCHS.
3. Fix MAX_JOBS auto-detect to account for arch count (upstream assumes 2 archs = 9GB/job,
   but cu128+ has 4 archs = ~18GB/job, causing OOM on 32GB machines).
"""
import subprocess
from pathlib import Path

# Init only the CUDA submodule (composable_kernel is ROCm-only and has
# filenames exceeding Windows' 260-char path limit)
subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)
print("Initialized csrc/cutlass submodule")

setup_file = Path("setup.py")
content = setup_file.read_text()

# Remove submodule init calls from setup.py (cutlass already done, composable_kernel not needed)
content = content.replace(
    'subprocess.run(["git", "submodule", "update", "--init", "csrc/composable_kernel"])',
    "# skipped composable_kernel submodule (ROCm only)",
)
content = content.replace(
    'subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"])',
    "# skipped cutlass submodule (already initialized)",
)
print("Patched out submodule init calls from setup.py")

# Replace cuda_archs() to also read TORCH_CUDA_ARCH_LIST
old_func = '''def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")'''

new_func = '''def cuda_archs() -> str:
    archs = os.getenv("FLASH_ATTN_CUDA_ARCHS")
    if archs:
        return archs.split(";")
    torch_archs = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    if torch_archs:
        # Convert "8.0 9.0 10.0 12.0" -> ["80", "90", "100", "120"]
        return [a.replace(".", "") for a in torch_archs.split()]
    return ["80", "90", "100", "120"]'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("Patched cuda_archs() to read TORCH_CUDA_ARCH_LIST")
else:
    print("WARNING: Could not find cuda_archs() function - source may have changed")

# Fix MAX_JOBS auto-detect: use arch count instead of hardcoded /9
# Upstream assumes 2 archs (9GB/job). With 4 archs (cu128+), it's ~18GB/job.
old_memory_estimate = "max_num_jobs_memory = int(free_memory_gb / 9)  # each JOB peak memory cost is ~8-9GB when threads = 4"
new_memory_estimate = (
    "num_archs = len(cuda_archs())\n"
    "            per_job_gb = 4.5 * num_archs  # ~4.5GB per cicc process x num arch targets\n"
    "            max_num_jobs_memory = int(free_memory_gb / per_job_gb)"
)

if old_memory_estimate in content:
    content = content.replace(old_memory_estimate, new_memory_estimate)
    print(f"Patched MAX_JOBS auto-detect to use arch count (cuda_archs())")
else:
    print("WARNING: Could not find MAX_JOBS memory estimate - source may have changed")

setup_file.write_text(content)
