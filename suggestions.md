# Architectural Review: cuda-wheels

## What's Good

- **Consolidated build system** - Single repo instead of N repos is the right call
- **Dynamic arch detection** - Smart to compute sm_XX based on CUDA/PyTorch versions
- **Per-package config** - Clean separation of concerns
- **Skip existing builds** - Checking releases before building saves CI time

## Critical Issues

### 1. ABI Landmines (Severity: High)

Wheel filenames encode CUDA + PyTorch but NOT:
- GCC/MSVC version (ubuntu-22.04's GCC can change between runs)
- glibc version
- libstdc++ ABI

A wheel built today may segfault tomorrow if GitHub updates the runner image. Options:
- Pin runner image hash
- Add compiler version to wheel name
- Use manylinux containers with fixed toolchains

### 2. Cache Key Doesn't Include Extra Components (Severity: High)

```yaml
key: cuda-${{ inputs.cuda-version }}-linux-v2
```

If you change `extra_cuda_components` for a package, you'll restore a stale cache missing those components. The cache key needs to hash the component list.

### 3. No Import Smoke Test (Severity: High)

We build wheels but never verify they actually work:
```python
# Missing: python -c "import cumesh; print(cumesh.__version__)"
```

Could ship broken wheels for weeks before anyone notices.

### 4. Hardcoded Version Logic Will Rot (Severity: Medium)

```python
if (cuda_major, cuda_minor) >= (12, 8):
    archs.append("10.0")
```

When CUDA 13.1 adds sm_130, code changes needed. Consider a data-driven approach:

```yaml
# cuda_arch_support.yml
sm_100:
  min_cuda: "12.8"
  min_pytorch: "2.6"
```

### 5. Network Installer Fragility (Severity: Medium)

Network installers hit NVIDIA CDN which can:
- Rate limit
- Have regional outages
- Change component names between minor versions

For CI, local installers are more reliable despite the size.

### 6. No Reproducibility (Severity: Medium)

- `pip install torch==$VERSION` can resolve different transitive deps over time
- No `requirements.txt` lock file
- `@v4` action tags can change behavior

### 7. Scaling Cliff (Severity: Medium)

```
9 packages × 6 CUDA versions × 5 PyTorch versions × 4 Python versions × 2 platforms
= 2,160 potential jobs
```

GitHub Actions has concurrency limits. Consider:
- Self-hosted runners
- Splitting into multiple workflows
- Build priority tiers (popular combos first)

### 8. Release Strategy (Severity: Low)

`{pkg}-latest` releases mean:
- No rollback capability
- No way to install "last known good"
- No audit trail of what changed

Consider semver releases or at least timestamped tags.

## Quick Wins

### 1. Add smoke test after wheel build

```yaml
- name: Smoke test
  run: |
    pip install dist/*.whl
    python -c "import ${{ matrix.package }}"
```

### 2. Fix cache key

```yaml
key: cuda-${{ inputs.cuda-version }}-${{ hashFiles(inputs.extra-components) }}-linux-v3
```

### 3. Pin action versions to SHA

```yaml
uses: actions/cache/restore@0c45773b623bea8c8e75f6c82b208c3cf94ea4f9  # v4.0.2
```

## Priority Order

1. Add smoke tests
2. Fix cache keys
3. Pin toolchain versions
4. Add rollback capability

## Overall Assessment

**Solid foundation, production-hardening needed.** The architecture is sound but one bad GitHub runner update away from shipping broken wheels. The lack of import testing is the biggest gap - could be distributing non-functional packages right now and not know it.
