#!/usr/bin/env python3
"""Repack wheels to fix internal metadata: torch28 → torch2.8 (v1 → v2).

Downloads v2-named wheels from GitHub releases, fixes internal metadata
(METADATA Version, dist-info directory name, RECORD hashes) to match
the v2 filename, then re-uploads with --clobber.

The wheel filename is already v2 (torch2.8). The problem is the internal
metadata still says torch28 (v1). This script fixes that mismatch.
"""
import argparse
import hashlib
import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

# Matches v1 torch version in internal metadata/paths: torch28, torch210
_V1_INTERNAL_RE = re.compile(r'(torch)(\d)(\d+)')
# Matches v2 torch version in filename: torch2.8, torch2.10
_V2_FILENAME_RE = re.compile(r'(\+cu\d+torch)(\d)\.(\d+)(-cp)')


def v2_to_v1(v2_str):
    """Extract v1 version from v2 filename pattern: torch2.8 → torch28"""
    m = _V2_FILENAME_RE.search(v2_str)
    if not m:
        return None
    return f"torch{m.group(2)}{m.group(3)}"


def v2_torch_version(v2_str):
    """Extract torch version from v2 filename: torch2.8 → 2.8"""
    m = _V2_FILENAME_RE.search(v2_str)
    if not m:
        return None
    return f"{m.group(2)}.{m.group(3)}"


def record_hash(data: bytes) -> str:
    """Compute RECORD-style hash: sha256=<urlsafe-base64-no-padding>"""
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
    return f"sha256={b64}"


def repack_wheel(whl_path: Path, output_dir: Path, dry_run=False) -> Path | None:
    """Repack a single wheel, fixing internal v1 metadata to v2.

    Returns output path on success, None if no changes needed.
    """
    filename = whl_path.name

    # Extract v1 and v2 torch strings from the filename
    m = _V2_FILENAME_RE.search(filename)
    if not m:
        return None  # not a torch-versioned wheel

    v2_torch = f"torch{m.group(2)}.{m.group(3)}"
    v1_torch = f"torch{m.group(2)}{m.group(3)}"

    with zipfile.ZipFile(whl_path, 'r') as zin:
        # Find the dist-info directory
        dist_info_dirs = set()
        for name in zin.namelist():
            if '.dist-info/' in name:
                dist_info_dirs.add(name.split('/')[0])

        if not dist_info_dirs:
            print(f"  WARNING: No dist-info found in {filename}")
            return None

        old_dist_info = list(dist_info_dirs)[0]

        # Read METADATA to check if it needs fixing
        meta_path = f"{old_dist_info}/METADATA"
        try:
            meta_text = zin.read(meta_path).decode('utf-8')
        except KeyError:
            print(f"  WARNING: No METADATA in {filename}")
            return None

        needs_v1_fix = v1_torch in meta_text

        # Check if RECORD hashes are valid
        needs_record_fix = False
        record_path = f"{old_dist_info}/RECORD"
        if record_path in zin.namelist():
            import csv
            record_text = zin.read(record_path).decode('utf-8')
            reader = csv.reader(io.StringIO(record_text))
            for row in reader:
                if len(row) < 3 or not row[1]:
                    continue
                try:
                    data = zin.read(row[0])
                    if record_hash(data) != row[1]:
                        needs_record_fix = True
                        break
                except KeyError:
                    needs_record_fix = True
                    break

        if not needs_v1_fix and not needs_record_fix:
            return None  # nothing to fix

        # Fix METADATA: replace v1 torch naming with v2
        if needs_v1_fix:
            meta_text = meta_text.replace(v1_torch, v2_torch)

        # Parse version and name from (fixed) METADATA
        old_version = new_version = None
        pkg_meta_name = None
        for line in meta_text.splitlines():
            if line.startswith('Version:'):
                new_version = line.split(':', 1)[1].strip()
            if line.startswith('Name:'):
                pkg_meta_name = line.split(':', 1)[1].strip()

        new_dist_info = f"{pkg_meta_name}-{new_version}.dist-info"

        if dry_run:
            print(f"  Would repack: {old_dist_info} → {new_dist_info}")
            return whl_path

        # Build new zip in memory
        output_path = output_dir / filename
        buf = io.BytesIO()

        # Track modified files for RECORD
        modified_files = {}

        with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                new_name = item.filename.replace(old_dist_info, new_dist_info)

                if item.filename.endswith('/METADATA'):
                    # Fix Version line
                    text = data.decode('utf-8')
                    text = text.replace(v1_torch, v2_torch)
                    data = text.encode('utf-8')
                    modified_files[new_name] = data

                elif item.filename.endswith('/RECORD'):
                    # Will regenerate RECORD at the end
                    continue

                else:
                    modified_files[new_name] = data

                # Preserve the ZipInfo metadata (timestamps, etc)
                new_info = zipfile.ZipInfo(new_name)
                new_info.compress_type = item.compress_type
                new_info.date_time = item.date_time
                new_info.external_attr = item.external_attr
                zout.writestr(new_info, data)

            # Regenerate RECORD
            record_lines = []
            for fname, fdata in modified_files.items():
                h = record_hash(fdata)
                record_lines.append(f"{fname},{h},{len(fdata)}")

            record_name = f"{new_dist_info}/RECORD"
            record_lines.append(f"{record_name},,")
            record_data = '\n'.join(record_lines) + '\n'

            record_info = zipfile.ZipInfo(record_name)
            record_info.compress_type = zipfile.ZIP_DEFLATED
            # Use same timestamp as METADATA
            for item in zin.infolist():
                if item.filename.endswith('/METADATA'):
                    record_info.date_time = item.date_time
                    break
            zout.writestr(record_info, record_data)

        output_path.write_bytes(buf.getvalue())
        return output_path


def get_releases(repo):
    """Get all releases via gh CLI."""
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "100", "--json", "tagName"],
        capture_output=True, text=True, check=True
    )
    import json
    return [r["tagName"] for r in json.loads(result.stdout)]


def get_release_assets(repo, tag):
    """Get assets for a release."""
    import json
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo, "--json", "assets"],
        capture_output=True, text=True, check=True
    )
    return json.loads(result.stdout)["assets"]


def main():
    parser = argparse.ArgumentParser(description="Repack wheels: fix internal v1 metadata to v2")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--package", default=None, help="Only process this package release tag")
    parser.add_argument("--repo", default="PozzettiAndrea/cuda-wheels")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between uploads")
    parser.add_argument("--local", nargs="+", help="Repack local wheel files instead of downloading")
    args = parser.parse_args()

    if args.local:
        # Local mode: repack files in place
        output_dir = Path(".")
        for whl_file in args.local:
            path = Path(whl_file)
            if not path.exists():
                print(f"File not found: {path}")
                continue
            print(f"Repacking: {path.name}")
            result = repack_wheel(path, output_dir, dry_run=args.dry_run)
            if result and not args.dry_run:
                print(f"  OK: {result}")
            elif result is None:
                print(f"  No changes needed")
        return

    # Remote mode: download from GitHub releases, repack, re-upload
    releases = get_releases(args.repo)
    if args.package:
        releases = [r for r in releases if args.package in r]

    print(f"Processing {len(releases)} releases")

    for tag in sorted(releases):
        assets = get_release_assets(args.repo, tag)
        wheels = [a for a in assets if a["name"].endswith(".whl") and _V2_FILENAME_RE.search(a["name"])]

        if not wheels:
            continue

        # Check if any actually need repacking (v1 internal metadata)
        # Quick check: if filename has torch2.X, check if we need to fix
        print(f"\n{tag}: {len(wheels)} v2-named wheels")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            repacked_dir = tmppath / "repacked"
            repacked_dir.mkdir()

            to_upload = []

            for i, asset in enumerate(wheels, 1):
                name = asset["name"]
                print(f"  [{i}/{len(wheels)}] {name}")

                if args.dry_run:
                    # Just check if it needs repacking
                    repack_wheel(Path("/dev/null"), repacked_dir, dry_run=True)
                    continue

                # Download
                dl_path = tmppath / name
                subprocess.run(
                    ["gh", "release", "download", tag, "--repo", args.repo,
                     "-p", name, "-D", str(tmppath), "--clobber"],
                    capture_output=True, check=True
                )

                # Repack
                result = repack_wheel(dl_path, repacked_dir)
                if result:
                    to_upload.append(result)
                    print(f"    Repacked OK")
                else:
                    print(f"    No changes needed (already v2 internally)")

            if to_upload and not args.dry_run:
                print(f"\n  Uploading {len(to_upload)} repacked wheels to {tag}...")
                # Upload in batches to avoid arg length issues
                batch_size = 20
                for batch_start in range(0, len(to_upload), batch_size):
                    batch = to_upload[batch_start:batch_start + batch_size]
                    cmd = ["gh", "release", "upload", tag, "--repo", args.repo, "--clobber"]
                    cmd.extend(str(p) for p in batch)
                    subprocess.run(cmd, check=True)
                    if args.delay and batch_start + batch_size < len(to_upload):
                        time.sleep(args.delay)

                print(f"  Uploaded {len(to_upload)} wheels")


if __name__ == "__main__":
    main()
