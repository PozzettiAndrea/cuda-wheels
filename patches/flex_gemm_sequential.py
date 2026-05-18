"""Patch FlexGEMM for the sequential-checkpoint test bed.

Identical structure to patches/flex_gemm_sharded.py: rewrite the package
name in both setup.py and pyproject.toml so the produced wheel is
`flex_gemm_sequential-*` rather than `flex_gemm-*`. Lets us release the
sequential-checkpoint POC build side-by-side with the canonical flex_gemm
release and diff cubins.

Also keeps the triton-windows fix from patches/flexgemm.py so the variant
builds on Windows too if/when we extend the timeout/trigger plumbing
there.
"""
import re
from pathlib import Path

# --- Rename the package so the sequential-checkpoint wheel doesn't shadow flex_gemm ---
# Modern setuptools (PEP 621) reads name from pyproject.toml's [project]
# section. setup.py `name=` is the fallback. Rewrite both so the produced
# wheel is named flex_gemm_sequential-* regardless of which one setuptools
# resolves first.
setup_py = Path("setup.py")
setup_text = setup_py.read_text()
old_setup = 'name="flex_gemm"'
new_setup = 'name="flex_gemm_sequential"'
if old_setup in setup_text:
    setup_py.write_text(setup_text.replace(old_setup, new_setup, 1))
    print(f"Renamed setup.py: {old_setup} -> {new_setup}")
else:
    print(f"NOTE: anchor {old_setup!r} not in setup.py -- continuing; pyproject.toml is authoritative under PEP 621")

pyproject = Path("pyproject.toml")
if pyproject.exists():
    py_text = pyproject.read_text()
    old_proj = 'name = "flex_gemm"'
    new_proj = 'name = "flex_gemm_sequential"'
    if old_proj in py_text:
        pyproject.write_text(py_text.replace(old_proj, new_proj, 1))
        print(f"Renamed pyproject.toml [project] name: flex_gemm -> flex_gemm_sequential")
    else:
        old_alt = 'name="flex_gemm"'
        new_alt = 'name="flex_gemm_sequential"'
        if old_alt in py_text:
            pyproject.write_text(py_text.replace(old_alt, new_alt, 1))
            print(f"Renamed pyproject.toml [project] name (no-space form): flex_gemm -> flex_gemm_sequential")
        else:
            raise SystemExit(
                "FATAL: could not find name field in pyproject.toml -- upstream may have changed"
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
