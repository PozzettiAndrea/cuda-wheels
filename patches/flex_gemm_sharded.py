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
# Modern setuptools (PEP 621) reads name from pyproject.toml's [project]
# section. setup.py `name=` is the fallback. Rewrite both so the produced
# wheel is named flex_gemm_sharded-* regardless of which one setuptools
# resolves first.
setup_py = Path("setup.py")
setup_text = setup_py.read_text()
old_setup = 'name="flex_gemm"'
new_setup = 'name="flex_gemm_sharded"'
if old_setup in setup_text:
    setup_py.write_text(setup_text.replace(old_setup, new_setup, 1))
    print(f"Renamed setup.py: {old_setup} -> {new_setup}")
else:
    print(f"NOTE: anchor {old_setup!r} not in setup.py -- continuing; pyproject.toml is authoritative under PEP 621")

pyproject = Path("pyproject.toml")
if pyproject.exists():
    py_text = pyproject.read_text()
    old_proj = 'name = "flex_gemm"'
    new_proj = 'name = "flex_gemm_sharded"'
    if old_proj in py_text:
        pyproject.write_text(py_text.replace(old_proj, new_proj, 1))
        print(f"Renamed pyproject.toml [project] name: flex_gemm -> flex_gemm_sharded")
    else:
        # Maybe formatted as name="flex_gemm" (no spaces)
        old_alt = 'name="flex_gemm"'
        new_alt = 'name="flex_gemm_sharded"'
        if old_alt in py_text:
            pyproject.write_text(py_text.replace(old_alt, new_alt, 1))
            print(f"Renamed pyproject.toml [project] name (no-space form): flex_gemm -> flex_gemm_sharded")
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
