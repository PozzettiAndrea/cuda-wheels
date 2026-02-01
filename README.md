# cuda-wheels

Pre-built CUDA Python wheels for popular ML/3D packages. Installable via pip with `--extra-index-url`.

## Installation

```bash
pip install <package> --extra-index-url https://pozzettiandrea.github.io/cuda-wheels
```

Example:
```bash
pip install sageattention gsplat nvdiffrast --extra-index-url https://pozzettiandrea.github.io/cuda-wheels
```

## Available Packages

| Package | Source | Description |
|---------|--------|-------------|
| sageattention | thu-ml/SageAttention | Efficient attention implementation |
| gsplat | nerfstudio-project/gsplat | Gaussian splatting |
| nvdiffrast | NVlabs/nvdiffrast | Differentiable rasterization |
| flexgemm | JeffreyXiang/FlexGEMM | Flexible GEMM operations |
| cumesh | JeffreyXiang/CuMesh | CUDA mesh processing |
| ovoxel | microsoft/TRELLIS.2 | Voxel operations |
| nvdiffrec_render | NVlabs/nvdiffrec | Render utilities from nvdiffrec |
| torch_generic_nms | ronghanghu/torch_generic_nms | Generic NMS for PyTorch |
| cc_torch | ronghanghu/cc_torch | Connected components for PyTorch |

## Supported Configurations

- **CUDA**: 12.4, 12.6, 12.8, 13.0
- **PyTorch**: 2.4.0 - 2.9.1
- **Python**: 3.10, 3.11, 3.12, 3.13
- **Platforms**: Linux (x86_64), Windows (amd64)

## Adding a New Package

1. Create a config file in `packages/<name>.yml`:

```yaml
name: my_package
source_repo: owner/repo           # GitHub repository
version: "1.0.0"                  # Package version
source_tag: "v1.0.0"              # Git tag (optional, empty = main branch)

# Build options
clone_recursive: true             # Clone with --recursive for submodules
free_disk_space: true             # Free disk space before build (large builds)
patch_script: patches/my_pkg.py   # Python script to patch source (optional)
build_subdir: subdir              # Build from subdirectory (optional)
extra_deps: "numpy scipy"         # Extra pip dependencies (optional)

# Build matrix
build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.5.1"
      python_versions: ["3.10", "3.11", "3.12"]
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
  platforms: ["linux", "windows"]
```

2. (Optional) Create a patch script in `patches/<name>.py` if source modifications are needed.

3. Add the package to the workflow dropdown in `.github/workflows/build.yml`.

4. Run:
```bash
# Test matrix generation
python scripts/generate_matrix.py --package my_package --output /tmp/test.json

# Trigger build
gh workflow run build.yml -f package=my_package
```

## Architecture Detection

CUDA architectures are auto-detected based on CUDA and PyTorch versions:

| PyTorch | CUDA 12.4 | CUDA 12.6+ | CUDA 12.8+ |
|---------|-----------|------------|------------|
| 2.4.x, 2.5.x | 7.0-9.0 | 7.0-9.0 | 7.0-9.0 |
| 2.6.x+ | 7.0-9.0 | 7.0-9.0, 10.0 | 7.0-9.0, 10.0, 12.0 |

- **sm_100** (10.0): B200, requires CUDA 12.6+ and PyTorch 2.6+
- **sm_120** (12.0): RTX 50xx, requires CUDA 12.8+ and PyTorch 2.6+

Override with per-combination `arch_list` or package-level `arch_list` in config.

## Patch Scripts

For packages requiring source modifications, create `patches/<name>.py`:

```python
"""Patch script for my_package."""
from pathlib import Path
import re

# Modify source files
setup_py = Path("setup.py")
content = setup_py.read_text()
content = content.replace("old", "new")
setup_py.write_text(content)

print("Applied patches")
```

The script runs in the source directory after clone, before build.

## Project Structure

```
cuda-wheels/
├── .github/
│   ├── actions/
│   │   ├── build-wheel/    # Composite action for wheel building
│   │   └── setup-cuda/     # Composite action for CUDA setup
│   └── workflows/
│       └── build.yml       # Main build workflow
├── packages/               # Package configs (*.yml)
├── patches/                # Patch scripts (*.py)
├── scripts/
│   ├── generate_matrix.py  # Matrix generation from configs
│   └── generate_index.py   # PEP 503 index generation
└── docs/                   # Generated GitHub Pages index
```

## License

MIT
