#!/usr/bin/env python3
"""Rename existing wheel assets on GitHub releases from v1 naming (torch29) to v2 (torch2.9).

For each release, downloads wheels with old naming, renames them, uploads with new name,
and deletes the old asset. Skips wheels that already have v2 naming.

Usage:
    python scripts/rename_wheels_v2.py [--dry-run] [--package PACKAGE]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Match old-style torch version: torch followed by digits only (no dots)
OLD_TORCH_RE = re.compile(r'(\+cu\d+torch)(\d)(\d+)(-cp)')


def v2_name(filename: str) -> str | None:
    """Convert v1 wheel filename to v2. Returns None if already v2 or no match."""
    m = OLD_TORCH_RE.search(filename)
    if not m:
        return None
    # e.g. +cu128torch29-cp -> +cu128torch2.9-cp
    prefix, major, minor, suffix = m.group(1), m.group(2), m.group(3), m.group(4)
    replacement = f"{prefix}{major}.{minor}{suffix}"
    start, end = m.start(), m.end()
    return filename[:start] + replacement + filename[end:]


def get_releases(repo: str) -> list:
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "100",
         "--json", "tagName", "-q", ".[].tagName"],
        capture_output=True, text=True
    )
    return result.stdout.strip().split("\n")


def get_assets(repo: str, release: str) -> list:
    # First get the release ID
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/tags/{release}",
         "--jq", ".id"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    release_id = result.stdout.strip()
    # Then paginate through assets
    result = subprocess.run(
        ["gh", "api", "--paginate",
         f"repos/{repo}/releases/{release_id}/assets",
         "--jq", ".[] | {id, name}"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    assets = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            try:
                assets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return assets


def rename_asset(repo: str, release: str, asset: dict, new_name: str,
                 dry_run: bool, keep_old: bool = True) -> bool:
    """Download, re-upload with new name. Returns True on success."""
    old_name = asset["name"]
    if dry_run:
        print(f"  [DRY RUN] {old_name} -> {new_name}")
        return True

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download
        dl = subprocess.run(
            ["gh", "release", "download", release, "--repo", repo,
             "-p", old_name, "-D", tmpdir],
            capture_output=True, text=True
        )
        if dl.returncode != 0:
            print(f"  FAILED to download {old_name}: {dl.stderr}", file=sys.stderr)
            return False

        old_path = Path(tmpdir) / old_name
        new_path = Path(tmpdir) / new_name
        old_path.rename(new_path)

        # Upload new
        up = subprocess.run(
            ["gh", "release", "upload", release, "--repo", repo,
             str(new_path), "--clobber"],
            capture_output=True, text=True
        )
        if up.returncode != 0:
            print(f"  FAILED to upload {new_name}: {up.stderr}", file=sys.stderr)
            return False

        if not keep_old:
            subprocess.run(
                ["gh", "api", "--method", "DELETE",
                 f"repos/{repo}/releases/assets/{asset['id']}"],
                capture_output=True, text=True
            )

        print(f"  {old_name} -> {new_name}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Rename v1 wheels to v2 naming on GitHub releases")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be renamed without doing it")
    parser.add_argument("--package", default=None, help="Only process this package (e.g. flash_attn)")
    parser.add_argument("--repo", default="PozzettiAndrea/cuda-wheels")
    parser.add_argument("--delete-old", action="store_true", help="Delete old-named assets after uploading")
    parser.add_argument("--threads", type=int, default=2, help="Number of parallel threads")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between operations (rate limiting)")
    args = parser.parse_args()

    releases = get_releases(args.repo)
    total_renamed = 0
    total_failed = 0
    total_skipped = 0

    for release in releases:
        if args.package and not release.startswith(args.package):
            continue

        assets = get_assets(args.repo, release)
        wheels = [a for a in assets if a["name"].endswith(".whl")]

        to_rename = []
        for w in wheels:
            new = v2_name(w["name"])
            if new and new != w["name"]:
                to_rename.append((w, new))

        if not to_rename:
            print(f"{release}: {len(wheels)} wheels, all already v2")
            total_skipped += len(wheels)
            continue

        print(f"{release}: {len(to_rename)}/{len(wheels)} to rename")

        if args.dry_run:
            for i, (asset, new_name) in enumerate(to_rename, 1):
                print(f"  [{i}/{len(to_rename)}] ", end="")
                rename_asset(args.repo, release, asset, new_name, dry_run=True)
            total_renamed += len(to_rename)
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                futures = {
                    pool.submit(rename_asset, args.repo, release, asset, new_name,
                                dry_run=False, keep_old=not args.delete_old): asset["name"]
                    for asset, new_name in to_rename
                }
                for future in as_completed(futures):
                    done += 1
                    if future.result():
                        total_renamed += 1
                        print(f"  [{done}/{len(to_rename)}] OK")
                    else:
                        total_failed += 1
                        print(f"  [{done}/{len(to_rename)}] ERROR: {futures[future]}", file=sys.stderr)
                    time.sleep(args.delay)

    print(f"\nDone: {total_renamed} renamed, {total_failed} failed, {total_skipped} already v2")


if __name__ == "__main__":
    main()
