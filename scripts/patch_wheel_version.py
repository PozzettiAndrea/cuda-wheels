#!/usr/bin/env python3
"""
Patch wheel METADATA to include the full version from the wheel filename.

After building, wheels are renamed to include a local version identifier
(e.g., +cu130torch29), but the internal METADATA still has the base version.
This script fixes that mismatch so tools like uv/pip see consistent versions.

Usage:
    python patch_wheel_version.py <wheel_or_directory> [...]

Examples:
    python patch_wheel_version.py dist/
    python patch_wheel_version.py my_package-0.2+cu130torch29-cp312-cp312-linux_x86_64.whl
"""

import os
import re
import sys
import zipfile
import tempfile
from pathlib import Path


def extract_version_from_filename(filename: str) -> tuple[str, str]:
    """Extract package name and full version from wheel filename.

    Returns (package_name, version) e.g. ('sageattention', '0.2+cu130torch29')
    """
    # Wheel format: {name}-{version}-{python}-{abi}-{platform}.whl
    # Name contains only [A-Za-z0-9_], version starts with a digit
    m = re.match(r"^([A-Za-z0-9_]+)-([^-]+)-(cp|py)", filename)
    if not m:
        raise ValueError(f"Could not parse wheel filename: {filename}")
    return m.group(1), m.group(2)


def fix_wheel(wheel_path: Path) -> bool:
    """Fix METADATA version in a wheel file in-place. Returns True if modified."""
    filename = wheel_path.name
    pkg_name, version = extract_version_from_filename(filename)

    # Only need to fix if there's a local version identifier
    if "+" not in version:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract wheel
        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find dist-info
        dist_info_dirs = list(tmpdir.glob("*.dist-info"))
        if not dist_info_dirs:
            print(f"  WARNING: No .dist-info found in {filename}, skipping")
            return False
        dist_info = dist_info_dirs[0]

        # Read METADATA
        metadata_path = dist_info / "METADATA"
        if not metadata_path.exists():
            print(f"  WARNING: No METADATA found in {filename}, skipping")
            return False

        content = metadata_path.read_text()

        # Check current version - skip if already correct
        m = re.search(r"^Version: (.+)$", content, re.MULTILINE)
        if m and m.group(1) == version:
            print(f"  {filename}: already correct ({version})")
            return False

        current_version = m.group(1) if m else "unknown"
        print(f"  {filename}: {current_version} -> {version}")

        # Update Version in METADATA
        content = re.sub(
            r"^Version: .+$",
            f"Version: {version}",
            content,
            flags=re.MULTILINE,
        )
        metadata_path.write_text(content)

        # Rename dist-info directory to match new version
        old_name = dist_info.name
        new_name = f"{pkg_name}-{version}.dist-info"
        if old_name != new_name:
            new_dist_info = dist_info.parent / new_name
            dist_info.rename(new_dist_info)
            dist_info = new_dist_info

        # Update RECORD file references
        record_path = dist_info / "RECORD"
        if record_path.exists():
            record_content = record_path.read_text()
            record_content = record_content.replace(old_name, new_name)
            record_path.write_text(record_content)

        # Repack wheel (overwrite original)
        with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(tmpdir.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(tmpdir))

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_wheel_version.py <wheel_or_directory> [...]")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    fixed = 0

    for path in paths:
        if path.is_dir():
            wheels = sorted(path.glob("*.whl"))
        elif path.suffix == ".whl":
            wheels = [path]
        else:
            print(f"Skipping non-wheel: {path}")
            continue

        for whl in wheels:
            if fix_wheel(whl):
                fixed += 1

    print(f"Fixed {fixed} wheel(s)")


if __name__ == "__main__":
    main()
