#!/usr/bin/env python3
"""Generate build matrix from package YAML configs."""
import argparse
import json
import yaml
from pathlib import Path


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
                        "arch_list": combo_arch_list or pkg.get("arch_list", "7.5;8.0;8.6;8.9;9.0"),
                        "extra_deps": pkg.get("extra_deps", ""),
                        "pre_build_script": pkg.get("pre_build_script", ""),
                        "free_disk_space": pkg.get("free_disk_space", False),
                        "max_jobs": pkg.get("max_jobs", 1),
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
