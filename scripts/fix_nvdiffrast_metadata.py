#!/usr/bin/env python3
"""
Download nvdiffrast wheels and fix their metadata.

The wheel filenames have local version identifiers (e.g., 0.4.0+cu128torch28)
but the internal METADATA says just 0.4.0. This causes pip to reject them.

This script:
1. Downloads all nvdiffrast wheels
2. Fixes the METADATA Version field to match the filename
3. Saves fixed wheels to output directory
"""

import os
import re
import zipfile
import tempfile
import shutil
from pathlib import Path
from urllib.parse import unquote
import subprocess

WHEEL_URLS = [
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch24-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu124torch25-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp313-cp313-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch26-cp313-cp313-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp313-cp313-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu126torch28-cp313-cp313-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp313-cp313-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch28-cp313-cp313-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp313-cp313-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu128torch29-cp313-cp313-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp310-cp310-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp310-cp310-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp311-cp311-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp311-cp311-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp312-cp312-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp312-cp312-win_amd64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp313-cp313-linux_x86_64.whl",
    "https://github.com/PozzettiAndrea/cuda-wheels/releases/download/nvdiffrast-latest/nvdiffrast-0.4.0%2Bcu130torch29-cp313-cp313-win_amd64.whl",
]


def extract_version_from_filename(filename: str) -> str:
    """Extract full version (including local part) from wheel filename."""
    # nvdiffrast-0.4.0+cu128torch28-cp312-cp312-linux_x86_64.whl
    match = re.match(r'nvdiffrast-([^-]+)-', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract version from {filename}")


def fix_wheel_metadata(wheel_path: Path, output_dir: Path) -> Path:
    """Fix METADATA in wheel and save to output directory."""
    filename = wheel_path.name
    version = extract_version_from_filename(filename)

    print(f"  Fixing {filename} -> Version: {version}")

    # Create temp directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract wheel
        with zipfile.ZipFile(wheel_path, 'r') as zf:
            zf.extractall(tmpdir)

        # Find and fix METADATA
        dist_info = list(tmpdir.glob("*.dist-info"))[0]
        metadata_path = dist_info / "METADATA"

        with open(metadata_path, 'r') as f:
            content = f.read()

        # Replace Version line
        content = re.sub(
            r'^Version: .+$',
            f'Version: {version}',
            content,
            flags=re.MULTILINE
        )

        with open(metadata_path, 'w') as f:
            f.write(content)

        # Also need to rename dist-info directory to match new version
        old_dist_info_name = dist_info.name
        new_dist_info_name = f"nvdiffrast-{version}.dist-info"
        if old_dist_info_name != new_dist_info_name:
            new_dist_info = dist_info.parent / new_dist_info_name
            dist_info.rename(new_dist_info)
            dist_info = new_dist_info

        # Update RECORD file (list of files with hashes)
        record_path = dist_info / "RECORD"
        if record_path.exists():
            with open(record_path, 'r') as f:
                record_content = f.read()
            # Update dist-info name references in RECORD
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
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "fixed_wheels" / "nvdiffrast"
    download_dir = script_dir.parent / "downloads" / "nvdiffrast"

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(WHEEL_URLS)} wheels...")
    print(f"Output directory: {output_dir}")

    for url in WHEEL_URLS:
        # Decode URL-encoded filename
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
        fix_wheel_metadata(download_path, output_dir)

    print(f"\nDone! Fixed wheels are in: {output_dir}")
    print(f"Upload these to the GitHub release to replace the broken ones.")


if __name__ == "__main__":
    main()
