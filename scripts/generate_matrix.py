#!/usr/bin/env python3
"""Generate build matrix from package YAML configs."""
import argparse
import json
import yaml
from pathlib import Path


def get_default_arch_list(cuda_version: str, pytorch_version: str) -> str:
    """
    Auto-compute the CUDA arch_list based on CUDA and PyTorch versions.

    Base architectures (always included):
    - 7.0, 7.5: Volta/Turing (V100, RTX 20xx)
    - 8.0, 8.6, 8.9: Ampere/Ada (A100, RTX 30xx, RTX 40xx)
    - 9.0: Hopper (H100)

    Blackwell architectures (conditionally added):
    - 10.0: B200 (requires PyTorch 2.8+ and CUDA 12.4+)
    - 12.0: RTX 50xx (requires PyTorch 2.8+ and CUDA 12.8+)
    """
    # Base architectures up to Hopper
    archs = ["7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]

    # Parse versions
    cuda_major, cuda_minor = map(int, cuda_version.split(".")[:2])
    pytorch_major, pytorch_minor = map(int, pytorch_version.split(".")[:2])

    # Blackwell support requires PyTorch 2.8+
    pytorch_supports_blackwell = (pytorch_major, pytorch_minor) >= (2, 8)

    if pytorch_supports_blackwell:
        # sm_100 (B200) - needs CUDA 12.4+
        if (cuda_major, cuda_minor) >= (12, 4):
            archs.append("10.0")

        # sm_120 (RTX 50xx) - needs CUDA 12.8+
        if (cuda_major, cuda_minor) >= (12, 8):
            archs.append("12.0")

    return " ".join(archs)


def generate_matrix(package_filter: str) -> list:
    """Generate build matrix from package configs."""
    packages_dir = Path(__file__).parent.parent / "packages"
    matrix = []

    for pkg_file in packages_dir.glob("*.yml"):
        pkg = yaml.safe_load(pkg_file.read_text())

        if package_filter != "all" and pkg["name"] != package_filter:
            continue

        build = pkg["build_matrix"]

        # Support both old format (cuda_versions × pytorch_versions) and new format (combinations)
        if "combinations" in build:
            # New format: combinations with optional per-combination python_versions and arch_list
            combos = []
            for c in build["combinations"]:
                python_vers = c.get("python_versions", build.get("python_versions", []))
                combo_arch_list = c.get("arch_list")  # Per-combination arch_list
                combos.append((c["cuda"], c["pytorch"], python_vers, combo_arch_list))
        else:
            # Old format: cartesian product
            python_vers = build["python_versions"]
            combos = [(cuda, pytorch, python_vers, None)
                      for cuda in build["cuda_versions"]
                      for pytorch in build["pytorch_versions"]]

        for cuda, pytorch, python_versions, combo_arch_list in combos:
            for python_ver in python_versions:
                for platform in build["platforms"]:
                    matrix.append({
                        "package": pkg["name"],
                        "version": pkg["version"],
                        "source_repo": pkg["source_repo"],
                        "source_tag": pkg.get("source_tag", ""),
                        "cuda": cuda,
                        "cuda_short": cuda.replace(".", ""),
                        "cuda_apt": cuda.replace(".", "-"),
                        "pytorch": pytorch,
                        "python": python_ver,
                        "python_short": python_ver.replace(".", ""),
                        "platform": platform,
                        "arch_list": combo_arch_list or pkg.get("arch_list") or get_default_arch_list(cuda, pytorch),
                        "extra_deps": pkg.get("extra_deps", ""),
                        "pre_build_script": pkg.get("pre_build_script", ""),
                        "free_disk_space": pkg.get("free_disk_space", False),
                        "max_jobs": pkg.get("max_jobs", 1),
                        "clone_recursive": pkg.get("clone_recursive", False),
                    })

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Generate build matrix from package configs")
    parser.add_argument("--package", default="all", help="Package to build (or 'all')")
    parser.add_argument("--output", default="matrix.json", help="Output file path")
    args = parser.parse_args()

    matrix = generate_matrix(args.package)

    # Split by platform
    linux_jobs = [j for j in matrix if j["platform"] == "linux"]
    windows_jobs = [j for j in matrix if j["platform"] == "windows"]

    output = {
        "linux": {"include": linux_jobs},
        "windows": {"include": windows_jobs},
    }

    with open(args.output, "w") as f:
        # No indent - GitHub Actions needs single-line JSON for GITHUB_OUTPUT
        json.dump(output, f, separators=(',', ':'))

    print(f"Generated {len(matrix)} build jobs ({len(linux_jobs)} Linux, {len(windows_jobs)} Windows)")

    # Also print to stdout for debugging
    for job in matrix:
        print(f"  - {job['package']} py{job['python']} cu{job['cuda_short']} {job['platform']}")


if __name__ == "__main__":
    main()
