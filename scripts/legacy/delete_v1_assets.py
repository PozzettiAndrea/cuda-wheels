#!/usr/bin/env python3
"""Delete v1-named wheel assets from GitHub releases (torch29 style, no dot).

Only deletes a v1 asset if a matching v2 copy (torch2.9) exists on the same release.

Usage:
    python scripts/delete_v1_assets.py --package spconv --dry-run
    python scripts/delete_v1_assets.py --package spconv
"""
import argparse
import json
import re
import subprocess
import sys
import time

V1_RE = re.compile(r'(\+cu\d+torch)(\d)(\d+)(-cp)')


def v2_name(filename: str) -> str | None:
    m = V1_RE.search(filename)
    if not m:
        return None
    return filename[:m.start()] + f"{m.group(1)}{m.group(2)}.{m.group(3)}{m.group(4)}" + filename[m.end():]


def get_releases(repo: str) -> list:
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "100",
         "--json", "tagName", "-q", ".[].tagName"],
        capture_output=True, text=True
    )
    return result.stdout.strip().split("\n")


def get_assets(repo: str, release: str) -> list:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/releases/tags/{release}", "--jq", ".id"],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    release_id = result.stdout.strip()
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


def main():
    parser = argparse.ArgumentParser(description="Delete v1-named wheel assets from releases")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--package", default=None, help="Only process this package")
    parser.add_argument("--repo", default="PozzettiAndrea/cuda-wheels")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between deletes (rate limiting)")
    args = parser.parse_args()

    releases = get_releases(args.repo)
    total_deleted = 0
    total_skipped = 0
    total_missing_v2 = 0

    for release in releases:
        if args.package and not release.startswith(args.package):
            continue

        assets = get_assets(args.repo, release)
        asset_names = {a["name"] for a in assets}
        wheels = [a for a in assets if a["name"].endswith(".whl")]

        to_delete = []
        for w in wheels:
            v2 = v2_name(w["name"])
            if v2 is None:
                total_skipped += 1
                continue
            if v2 in asset_names:
                to_delete.append(w)
            else:
                total_missing_v2 += 1
                print(f"  WARNING: no v2 copy for {w['name']} (expected {v2})", file=sys.stderr)

        if not to_delete:
            print(f"{release}: nothing to delete ({total_skipped} already v2)")
            continue

        print(f"{release}: {len(to_delete)} v1 assets to delete")

        for i, asset in enumerate(to_delete, 1):
            if args.dry_run:
                print(f"  [{i}/{len(to_delete)}] [DRY RUN] DELETE {asset['name']}")
            else:
                result = subprocess.run(
                    ["gh", "api", "--method", "DELETE",
                     f"repos/{args.repo}/releases/assets/{asset['id']}"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"  [{i}/{len(to_delete)}] DELETED {asset['name']}")
                    total_deleted += 1
                else:
                    print(f"  [{i}/{len(to_delete)}] FAILED {asset['name']}: {result.stderr}", file=sys.stderr)
                time.sleep(args.delay)

        if args.dry_run:
            total_deleted += len(to_delete)

    print(f"\nDone: {total_deleted} deleted, {total_skipped} skipped (already v2), {total_missing_v2} missing v2 copy")


if __name__ == "__main__":
    main()
