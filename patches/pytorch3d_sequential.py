"""Patch pytorch3d for the sequential-checkpoint test bed.

Two changes:
1. Rename the package so the test wheel doesn't shadow the canonical
   pytorch3d release: setup.py `name="pytorch3d"` -> `"pytorch3d_sequential"`.
   (pytorch3d v0.7.9 doesn't have a pyproject.toml [project] table -- setup.py
   `name=` is authoritative -- but we defensively rewrite pyproject.toml too
   if a future version adds one.)
2. Apply the same Windows MSVC c++17 flag fix as patches/pytorch3d.py
   (kept here so the variant builds on Windows too if/when we extend the
   timeout plumbing there).
"""
from pathlib import Path

# --- (1) Rename the package ---
setup_file = Path("setup.py")
setup_text = setup_file.read_text()
renamed = False
for old, new in (
    ('name="pytorch3d"', 'name="pytorch3d_sequential"'),
    ("name='pytorch3d'", "name='pytorch3d_sequential'"),
    ('name = "pytorch3d"', 'name = "pytorch3d_sequential"'),
    ("name = 'pytorch3d'", "name = 'pytorch3d_sequential'"),
):
    if old in setup_text:
        setup_text = setup_text.replace(old, new, 1)
        setup_file.write_text(setup_text)
        print(f"Renamed setup.py: {old} -> {new}")
        renamed = True
        break
if not renamed:
    raise SystemExit(
        "FATAL: could not find name= field in pytorch3d setup.py -- upstream may have changed; inspect the file and update this patch"
    )

pyproject = Path("pyproject.toml")
if pyproject.exists():
    py_text = pyproject.read_text()
    for old, new in (
        ('name = "pytorch3d"', 'name = "pytorch3d_sequential"'),
        ('name="pytorch3d"', 'name="pytorch3d_sequential"'),
    ):
        if old in py_text:
            pyproject.write_text(py_text.replace(old, new, 1))
            print(f"Renamed pyproject.toml [project] name: pytorch3d -> pytorch3d_sequential")
            break

# --- (2) Windows MSVC c++17 flag fix (mirrors patches/pytorch3d.py) ---
content = setup_file.read_text()
old_cxx = '    extra_compile_args = {"cxx": ["-std=c++17"]}'
new_cxx = '    extra_compile_args = {"cxx": ["/std:c++17"] if os.name == "nt" else ["-std=c++17"]}'
if old_cxx in content:
    setup_file.write_text(content.replace(old_cxx, new_cxx))
    print("Patched pytorch3d CXX_FLAGS: -std=c++17 -> /std:c++17 on Windows")
else:
    print("NOTE: could not find CXX_FLAGS block in setup.py (Windows c++17 patch skipped); source may have changed")
