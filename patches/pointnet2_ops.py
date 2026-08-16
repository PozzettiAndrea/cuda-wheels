"""Let the farm's TORCH_CUDA_ARCH_LIST through to pointnet2_ops.

Upstream setup.py hardcodes, at import time:

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"

Two problems. It clobbers the arch list the build action exports, so every
wheel would be compiled for a 2021 GPU set regardless of config. And sm_37
(Kepler K80) was removed in CUDA 12.0, so under cu128 nvcc fails outright with
"Unsupported gpu architecture 'compute_37'" before compiling a single file.

Deleting the line is the whole fix: setup.py then leaves TORCH_CUDA_ARCH_LIST
alone and torch's BuildExtension picks up ours.

Run with cwd = repo root (the build action cds into source/ first).
"""

import re
from pathlib import Path

setup_py = Path("pointnet2_ops_lib/setup.py")
text = setup_py.read_text(encoding="utf-8")

pattern = re.compile(
    r'^os\.environ\[\s*["\']TORCH_CUDA_ARCH_LIST["\']\s*\]\s*=.*\n',
    re.MULTILINE,
)
patched, n = pattern.subn("", text)

# Fail loudly rather than silently shipping a wheel built for the wrong archs.
if n != 1:
    raise SystemExit(
        f"pointnet2_ops patch: expected exactly 1 TORCH_CUDA_ARCH_LIST "
        f"assignment in {setup_py}, found {n}. Upstream changed -- re-check "
        f"before building."
    )

setup_py.write_text(patched, encoding="utf-8")
print(f"Removed hardcoded TORCH_CUDA_ARCH_LIST from {setup_py}")
