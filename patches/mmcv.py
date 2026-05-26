"""Patch mmcv 1.7.2 to always build with CUDA ops + keep wheel name `mmcv`.

mmcv 1.7.2's setup.py has two conditional behaviors gated on the env
var MMCV_WITH_OPS:

  1. get_extensions() returns [] when MMCV_WITH_OPS != "1" -> pure-Python
     wheel with no CUDA compile.
  2. setup(name=...) flips between `mmcv` (lite) and `mmcv-full` (with ops)
     based on the same env var.

We want behavior #1 = "always build ops" and behavior #2 = "always keep
the name `mmcv`" (matching the PyPI 1.x convention where the published
`mmcv` package IS the full-with-ops version; `mmcv-lite` is the
explicitly-lite sibling).

cuda-wheels' build action doesn't support per-package env vars (only
patch_script + pre_build_script run in a separate shell), so the
cleanest fix is to set `os.environ['MMCV_WITH_OPS'] = '1'` at the top
of setup.py and replace the conditional name expression with a literal
'mmcv'.

Idempotent via the MARKER string.
"""
from pathlib import Path
import sys

setup_file = Path("setup.py")
if not setup_file.exists():
    print("mmcv patch: setup.py not found in cwd, skipping")
    sys.exit(0)

content = setup_file.read_text()

MARKER = "# CUDA-WHEELS PATCH: force MMCV_WITH_OPS=1, name=mmcv"
if MARKER in content:
    print("mmcv patch: already applied (marker present), skipping")
    sys.exit(0)

# Step 1: inject env var override at the top of setup.py, after the
# initial imports. Doing this at module-import time means the env var
# is set before any os.getenv('MMCV_WITH_OPS', ...) check fires.
PROLOGUE = f"""{MARKER}
import os
os.environ['MMCV_WITH_OPS'] = '1'

"""

# Find a clean anchor near the top of setup.py — after the shebang/
# encoding line, after the initial 'import os' (which IS in mmcv's
# setup.py). Just prepend; setup.py's own `import os` later is a no-op.
content = PROLOGUE + content

# Step 2: replace the conditional name expression. mmcv 1.7.2 has
# something like:
#   name='mmcv' if os.getenv('MMCV_WITH_OPS', '0') == '0' else 'mmcv-full',
#
# We force the literal 'mmcv' name regardless of env. Use a tolerant
# replacement that handles slight whitespace variations.
import re
pattern = re.compile(
    r"name\s*=\s*'mmcv'\s+if\s+os\.getenv\(\s*'MMCV_WITH_OPS'\s*,\s*'0'\s*\)\s*==\s*'0'\s+else\s+'mmcv-full'",
)
new_content, n_subs = pattern.subn("name='mmcv'", content)
if n_subs == 0:
    print(
        "mmcv patch: WARNING — could not find the conditional name= "
        "expression in setup.py. The wheel may be published as 'mmcv-full' "
        "instead of 'mmcv'. Check upstream for refactors."
    )
elif n_subs > 1:
    print(f"mmcv patch: WARNING — replaced {n_subs} matches (expected 1)")
content = new_content

setup_file.write_text(content)
print(
    "mmcv patch: applied MMCV_WITH_OPS=1 prologue + forced name='mmcv'; "
    f"name-conditional substitutions: {n_subs}"
)
