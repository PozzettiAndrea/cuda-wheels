# Package Configuration

This directory contains YAML configurations for building CUDA wheel packages.

## Quick Start

Create a new package config `packages/<name>.yml`:

```yaml
name: mypackage
source_repo: owner/repo
version: "1.0.0"
source_tag: v1.0.0

build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.4.0"
      python_versions: ["3.10", "3.11", "3.12"]
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
  platforms: ["linux", "windows"]
```

Then trigger a build:
```bash
gh workflow run build.yml -f package=mypackage
```

## Configuration Reference

### Required Fields

| Field | Description |
|-------|-------------|
| `name` | Package name (must match filename without `.yml`) |
| `source_repo` | GitHub repository (e.g., `owner/repo`) |
| `version` | Package version string |
| `build_matrix` | Build configuration (see below) |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `source_tag` | `""` | Git tag or branch to checkout |
| `free_disk_space` | `false` | Clean up disk before build (needed for large builds like PyTorch 2.8+) |
| `clone_recursive` | `false` | Clone with `--recursive` for submodules |
| `max_jobs` | `1` | Max parallel compilation jobs |
| `extra_deps` | `""` | Additional pip dependencies (space-separated) |
| `pre_build_script` | `""` | Shell script to run before building |

### Build Matrix

The `build_matrix` defines which combinations to build:

```yaml
build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.4.0"
      python_versions: ["3.10", "3.11", "3.12"]
      arch_list: "7.0 7.5 8.0 8.6 8.9 9.0"  # Optional override
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
  platforms: ["linux", "windows"]
```

Each combination generates `len(python_versions) × len(platforms)` jobs.

### CUDA Architecture List

The `arch_list` specifies which GPU architectures to compile for.

**Auto-detection (recommended)**: If not specified, `arch_list` is automatically computed based on CUDA and PyTorch versions:

| PyTorch | CUDA | Architectures |
|---------|------|---------------|
| 2.4.x | any | 7.0 7.5 8.0 8.6 8.9 9.0 |
| 2.8.x | 12.4 | 7.0 7.5 8.0 8.6 8.9 9.0 10.0 |
| 2.8.x | 12.8+ | 7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0 |

**Manual override**: Set `arch_list` at the combination level or package level if you need different architectures (e.g., for older Pascal support):

```yaml
arch_list: "6.1 7.0 7.5 8.0 8.6 8.9 9.0"  # Include Pascal (GTX 10xx)
```

### Architecture Reference

| Compute Capability | GPU Examples |
|--------------------|--------------|
| 6.1 | GTX 1060/1070/1080 (Pascal) |
| 7.0 | V100 (Volta) |
| 7.5 | RTX 2080, T4 (Turing) |
| 8.0 | A100 (Ampere datacenter) |
| 8.6 | RTX 3090 (Ampere consumer) |
| 8.9 | RTX 4090 (Ada Lovelace) |
| 9.0 | H100 (Hopper) |
| 10.0 | B200 (Blackwell datacenter) |
| 12.0 | RTX 5090 (Blackwell consumer) |

## Build Process

1. **Matrix Generation**: `scripts/generate_matrix.py` reads package configs and generates the build matrix
2. **Upstream Check**: Before building, checks if the wheel already exists in the upstream index
3. **CUDA Setup**: Installs CUDA toolkit (cached for faster subsequent builds)
4. **Build Environment**: Sets up Python, PyTorch, and build dependencies
5. **Build Wheel**: Runs `pip wheel . --no-build-isolation --no-deps`
6. **Rename**: Adds CUDA+PyTorch suffix to wheel filename (e.g., `+cu124torch24`)
7. **Release**: Uploads wheels to GitHub Releases
8. **Index**: Updates the PEP 503 package index on GitHub Pages

### Environment Variables

During build, these environment variables are set:
- `CUDA_HOME=/usr/local/cuda-<version>`
- `TORCH_CUDA_ARCH_LIST=<arch_list>`
- `MAX_JOBS=<max_jobs>`
- `LIBRARY_PATH=/usr/local/cuda-<version>/lib64/stubs` (for libcuda.so linking)

## Examples

### Simple Package

```yaml
name: mypackage
source_repo: owner/mypackage
version: "1.0.0"
source_tag: v1.0.0

build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.4.0"
      python_versions: ["3.10", "3.11", "3.12"]
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
  platforms: ["linux", "windows"]
```

### Package with Submodules

```yaml
name: gsplat
source_repo: nerfstudio-project/gsplat
version: "1.5.3"
source_tag: v1.5.3

clone_recursive: true
free_disk_space: true

build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.4.0"
      python_versions: ["3.10", "3.11", "3.12"]
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
  platforms: ["linux", "windows"]
```

### Package with Pre-build Script

```yaml
name: mypackage
source_repo: owner/mypackage
version: "1.0.0"

pre_build_script: |
  # Apply patches
  sed -i 's/old/new/' setup.py
  # Copy additional files
  cp -r extra/* src/

build_matrix:
  # ...
```

### Package with Custom Architectures

```yaml
name: nvdiffrast
source_repo: NVlabs/nvdiffrast
version: "0.4.0"

build_matrix:
  combinations:
    - cuda: "12.4"
      pytorch: "2.4.0"
      python_versions: ["3.10", "3.11", "3.12"]
      arch_list: "6.1 7.0 7.5 8.0 8.6 8.9 9.0"  # Include Pascal
    - cuda: "12.8"
      pytorch: "2.8.0"
      python_versions: ["3.10", "3.11", "3.12", "3.13"]
      arch_list: "6.1 7.0 7.5 8.0 8.6 8.9 9.0 10.0 12.0"
  platforms: ["linux", "windows"]
```
