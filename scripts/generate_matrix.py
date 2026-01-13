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
        for cuda in build["cuda_versions"]:
            for pytorch in build["pytorch_versions"]:
                for python_ver in build["python_versions"]:
                    for platform in build["platforms"]:
                        matrix.append({
                            "package": pkg["name"],
                            "version": pkg["version"],
                            "source_repo": pkg["source_repo"],
                            "source_tag": pkg.get("source_tag", ""),
                            "cuda": cuda,
                            "cuda_short": cuda.replace(".", ""),
                            "pytorch": pytorch,
                            "python": python_ver,
                            "platform": platform,
                            "arch_list": pkg.get("arch_list", "7.5;8.0;8.6;8.9;9.0"),
                            "extra_deps": pkg.get("extra_deps", ""),
                            "pre_build_script": pkg.get("pre_build_script", ""),
                        })

    return matrix


def main():
    parser = argparse.ArgumentParser(description="Generate build matrix from package configs")
    parser.add_argument("--package", default="all", help="Package to build (or 'all')")
    parser.add_argument("--output", default="matrix.json", help="Output file path")
    args = parser.parse_args()

    matrix = generate_matrix(args.package)

    output = {"include": matrix}

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(matrix)} build jobs")

    # Also print to stdout for debugging
    for job in matrix:
        print(f"  - {job['package']} py{job['python']} cu{job['cuda_short']} {job['platform']}")


if __name__ == "__main__":
    main()
