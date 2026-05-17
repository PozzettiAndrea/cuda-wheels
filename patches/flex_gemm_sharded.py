"""Patch FlexGEMM for the sharded test bed.

The canonical flex_gemm package uses setup.py `name="flex_gemm"`. When we
build this package as a sharded variant, we rename the produced wheel to
"flex_gemm_sharded" so it doesn't collide with the canonical flex_gemm
release on cuda-wheels. The two builds (monolithic vs sharded) can then
be compared side-by-side under different package names.

Also keeps the triton-windows compatibility fix from patches/flexgemm.py
so this variant builds on Windows too if/when we exercise that path.
"""
import re
from pathlib import Path

# --- Rename the package so the sharded wheel doesn't shadow flex_gemm ---
setup_py = Path("setup.py")
setup_text = setup_py.read_text()
old = 'name="flex_gemm"'
new = 'name="flex_gemm_sharded"'
if old in setup_text:
    setup_py.write_text(setup_text.replace(old, new, 1))
    print(f"Renamed setup.py: {old} -> {new}")
else:
    raise SystemExit(
        f"FATAL: anchor {old!r} not found in setup.py -- upstream may have "
        f"changed. Re-check the patch against the pinned source_tag."
    )

# --- triton-windows compatibility (mirrors patches/flexgemm.py) ---
pyproject = Path("pyproject.toml")
if pyproject.exists():
    content = pyproject.read_text()
    new_content = re.sub(
        r'"triton(>=[\d.]+)"',
        r'"triton\1; platform_system != \'Windows\'", "triton-windows\1; platform_system == \'Windows\'"',
        content,
    )
    if new_content != content:
        pyproject.write_text(new_content)
        print("Patched pyproject.toml for triton-windows compatibility")
