#!/usr/bin/env python3
"""
Fix wheel metadata for CUDA packages.

The wheel filenames have local version identifiers (e.g., 0.4.0+cu128torch28)
but the internal METADATA says just the base version. This causes pip to reject them.

Usage:
    python fix_wheel_metadata.py <package_name>

Example:
    python fix_wheel_metadata.py flex_gemm
    python fix_wheel_metadata.py o_voxel
    python fix_wheel_metadata.py nvdiffrec_render
"""

import os
import re
import sys
import zipfile
import tempfile
from pathlib import Path
from urllib.parse import unquote
import subprocess

CUDA_WHEELS_INDEX = "https://pozzettiandrea.github.io/cuda-wheels"


def get_wheel_urls(package_name: str) -> list[str]:
    """Fetch wheel URLs from GitHub release."""
    # Use gh to get release assets
    result = subprocess.run(
        ["gh", "release", "view", f"{package_name}-latest", "--json", "assets", "-q", ".assets[].url"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error fetching release: {result.stderr}")
        return []

    urls = [u.strip() for u in result.stdout.strip().split('\n') if u.strip()]
    return sorted(urls)


def extract_version_from_filename(filename: str, package_name: str) -> str:
    """Extract full version (including local part) from wheel filename."""
    # flex_gemm-1.0.0+cu128torch28-cp312-cp312-linux_x86_64.whl
    pattern = rf'{package_name}-([^-]+)-'
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract version from {filename}")


def fix_wheel_metadata(wheel_path: Path, output_dir: Path, package_name: str) -> Path:
    """Fix METADATA in wheel and save to output directory."""
    filename = wheel_path.name
    version = extract_version_from_filename(filename, package_name)

    print(f"  Fixing {filename} -> Version: {version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract wheel
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Find and fix METADATA
        dist_info_dirs = list(tmpdir.glob("*.dist-info"))
        if not dist_info_dirs:
            raise ValueError(f"No .dist-info found in {filename}")
        dist_info = dist_info_dirs[0]
        metadata_path = dist_info / "METADATA"

        with open(metadata_path, 'r') as f:
            content = f.read()

        # Check current version
        current_version = re.search(r'^Version: (.+)$', content, re.MULTILINE)
        if current_version:
            current = current_version.group(1)
            if current == version:
                print(f"    Already correct: {version}")
                # Just copy the file
                output_path = output_dir / filename
                import shutil
                shutil.copy(wheel_path, output_path)
                return output_path

        # Replace Version line
        content = re.sub(
            r'^Version: .+$',
            f'Version: {version}',
            content,
            flags=re.MULTILINE
        )

        with open(metadata_path, 'w') as f:
            f.write(content)

        # Rename dist-info directory to match new version
        old_dist_info_name = dist_info.name
        new_dist_info_name = f"{package_name}-{version}.dist-info"
        if old_dist_info_name != new_dist_info_name:
            new_dist_info = dist_info.parent / new_dist_info_name
            dist_info.rename(new_dist_info)
            dist_info = new_dist_info

        # Update RECORD file
        record_path = dist_info / "RECORD"
        if record_path.exists():
            with open(record_path, 'r') as f:
                record_content = f.read()
            record_content = record_content.replace(old_dist_info_name, new_dist_info_name)
            with open(record_path, 'w') as f:
                f.write(record_content)

        # Repack wheel
        output_path = output_dir / filename
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(tmpdir)
                    zf.write(file, arcname)

        return output_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_wheel_metadata.py <package_name>")
        print("Example: python fix_wheel_metadata.py flex_gemm")
        sys.exit(1)

    package_name = sys.argv[1]

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "fixed_wheels" / package_name
    download_dir = script_dir.parent / "downloads" / package_name

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching wheel URLs for {package_name}...")
    urls = get_wheel_urls(package_name)

    if not urls:
        print(f"No wheels found for {package_name}")
        sys.exit(1)

    print(f"Found {len(urls)} wheels")
    print(f"Output directory: {output_dir}")

    for url in urls:
        filename = unquote(url.split('/')[-1])
        download_path = download_dir / filename

        # Download if not exists
        if not download_path.exists():
            print(f"Downloading {filename}...")
            subprocess.run(
                ["curl", "-L", "-o", str(download_path), url],
                check=True,
                capture_output=True
            )
        else:
            print(f"Already downloaded: {filename}")

        # Fix metadata
        fix_wheel_metadata(download_path, output_dir, package_name)

    print(f"\nDone! Fixed wheels are in: {output_dir}")


if __name__ == "__main__":
    main()
