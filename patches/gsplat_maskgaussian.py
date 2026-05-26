"""Patch gsplat_maskgaussian: bring in the GLM headers it expects.

HY-World 2.0's vendored gsplat_maskgaussian fork strips out the GLM
third-party submodule that upstream nerfstudio/gsplat carries at
`gsplat/cuda/csrc/third_party/glm`. setup.py still references that
exact include path (line 93-94) and Common.h does `#include
<glm/gtc/type_ptr.hpp>` — so the build dies on `fatal error:
glm/gtc/type_ptr.hpp: No such file or directory`.

The cuda-wheels build runs with cwd at the package root
(`hyworld2/worldgen/third_party/gsplat_maskgaussian/`), so we drop GLM
in at `gsplat/cuda/csrc/third_party/glm`. Pinned to GLM tag 1.0.1
(May 2024) — API-stable, header-only, ~1MB.

Upstream nerfstudio/gsplat uses g-truc/glm as a submodule without a
fixed commit pin (resolves to whatever the submodule pointer was at
the gsplat commit being cloned). We could mirror that, but pinning
gives reproducibility.
"""
import shutil
import subprocess
from pathlib import Path

GLM_REPO = "https://github.com/g-truc/glm.git"
GLM_TAG = "1.0.1"
# Patch script runs from source/ (the repo clone root); the package itself
# lives under build_subdir. Paths must be prefixed accordingly.
BUILD_SUBDIR = Path("hyworld2/worldgen/third_party/gsplat_maskgaussian")
TARGET = BUILD_SUBDIR / "gsplat/cuda/csrc/third_party/glm"

## --- Rename the wheel distribution to `gsplat_maskgaussian` so it doesn't
## clobber the published vanilla `gsplat` v1.5.3 wheels (same version
## string, would land on the same `gsplat-latest` release slot if not
## renamed). The python import name stays `gsplat` because the source
## directory is gsplat/ — that's fine; only one variant should be
## installed per env at a time.
setup_py = BUILD_SUBDIR / "setup.py"
setup_text = setup_py.read_text()
old_name = 'name="gsplat"'
new_name = 'name="gsplat_maskgaussian"'
if old_name in setup_text:
    setup_py.write_text(setup_text.replace(old_name, new_name, 1))
    print(f"gsplat_maskgaussian: patched setup.py name -> {new_name}")
else:
    if new_name in setup_text:
        print(f"gsplat_maskgaussian: setup.py name already renamed")
    else:
        raise RuntimeError(
            f"setup.py at {setup_py} doesn't contain {old_name!r}; "
            f"upstream may have changed — review and re-pin source_tag."
        )

if (TARGET / "glm" / "glm.hpp").exists() or (TARGET / "include" / "glm" / "glm.hpp").exists():
    print(f"gsplat_maskgaussian: GLM already present at {TARGET}; skipping")
else:
    if TARGET.exists():
        # Partial / broken — clear it
        shutil.rmtree(TARGET)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"gsplat_maskgaussian: cloning {GLM_REPO}@{GLM_TAG} -> {TARGET}")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", GLM_TAG, GLM_REPO, str(TARGET)],
        check=True,
    )
    # Verify the include we need is reachable.
    expected = TARGET / "glm" / "gtc" / "type_ptr.hpp"
    if not expected.exists():
        # Some GLM tarballs nest under include/ instead of at root.
        alt = TARGET / "include" / "glm" / "gtc" / "type_ptr.hpp"
        if alt.exists():
            raise RuntimeError(
                f"GLM cloned but headers nested at {alt}, expected at {expected}. "
                f"Adjust include_dirs in setup.py or restructure the checkout."
            )
        raise RuntimeError(
            f"GLM cloned to {TARGET} but {expected} not found. Tag {GLM_TAG!r} "
            f"may have restructured the layout — try a different tag."
        )
    print(f"gsplat_maskgaussian: GLM ready at {TARGET}")
