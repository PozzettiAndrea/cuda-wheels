"""Patch fused-ssim for Windows + torch 2.10+ compatibility.

Bug: PyTorch 2.10 introduced `bool small` as a parameter name in
c10/cuda/CUDACachingAllocator.h:212. Windows SDK rpcndr.h (transitively
pulled in via windows.h through torch's own deps) defines:

    #define small char   /* MIDL helper from the 90s */

so nvcc preprocesses `bool small` -> `bool char` and bombs with
"invalid combination of type specifiers" on the StreamSegmentSize ctor.

Upstream: https://github.com/pytorch/pytorch/issues/173112

Fix: prepend each fused-ssim C++/CUDA source file with a Windows-only
header sequence that forces rpcndr.h to define `small`, then #undefs
it BEFORE any torch header gets a chance to reach CUDACachingAllocator.h.

  #ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN     /* skip most of windows.h's noise */
  #include <windows.h>            /* pulls rpcndr.h -> #define small char */
  #undef small                    /* kill the macro before torch headers */
  #endif

`-Usmall` on the nvcc command line does NOT fix this — that only
clears the initial preprocessor state; rpcndr.h then re-defines the
macro mid-compilation. The fix has to land in the source file's own
include sequence, sandwiched between windows.h and the torch headers.

This patch is gated to Windows + torch >= 2.10. On any other combo the
bug isn't present and the script is a no-op (so the 33 already-passing
combos in run 26460851532 stay green).
"""
import os
import sys
from pathlib import Path

import torch


def _torch_minor_version() -> tuple[int, int]:
    # torch.__version__ looks like "2.10.0+cu128"; we want (2, 10).
    base = torch.__version__.split("+", 1)[0]
    parts = base.split(".")
    return int(parts[0]), int(parts[1])


is_windows = os.name == "nt"
torch_ver = _torch_minor_version()

if not (is_windows and torch_ver >= (2, 10)):
    print(
        f"fused_ssim patch: not applicable "
        f"(is_windows={is_windows}, torch={torch_ver[0]}.{torch_ver[1]}); "
        f"skipping"
    )
    sys.exit(0)

print(
    f"fused_ssim patch: applying Windows + torch {torch_ver[0]}.{torch_ver[1]} "
    f"workaround for pytorch/pytorch#173112"
)


# Idempotency marker — embedded in the prologue. Re-runs detect it and skip.
MARKER = "pytorch/pytorch#173112"

PROLOGUE = f"""// Workaround for {MARKER}: rpcndr.h on Windows defines `#define small char`
// which collides with PyTorch 2.10+'s use of `bool small` as a parameter
// name in c10/cuda/CUDACachingAllocator.h:212. Force-include windows.h
// (triggers the macro definition) then immediately #undef it, BEFORE any
// torch header is parsed.
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef small
#endif

"""


# Source files seen in the failing build log (cwd=source/). If upstream
# renames or adds files, the missing-file warning surfaces it without
# failing the build.
SOURCE_FILES = ["ext.cpp", "ssim.cu"]

patched = 0
skipped = 0
for fname in SOURCE_FILES:
    fpath = Path(fname)
    if not fpath.exists():
        print(f"  WARN: {fname} not found in source/ — skipping")
        continue
    content = fpath.read_text()
    if MARKER in content:
        print(f"  {fname}: already patched (marker present)")
        skipped += 1
        continue
    fpath.write_text(PROLOGUE + content)
    print(f"  {fname}: prepended {len(PROLOGUE)} chars of Windows fix")
    patched += 1

print(
    f"fused_ssim patch: patched={patched}, "
    f"already-patched={skipped}, total-files={len(SOURCE_FILES)}"
)
