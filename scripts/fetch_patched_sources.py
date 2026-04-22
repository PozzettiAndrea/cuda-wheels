#!/usr/bin/env python3
"""Fetch package sources, apply patches, and stage them for the patched-sources branch.

Usage:
    python scripts/fetch_patched_sources.py --package pytorch3d --output-dir staging
    python scripts/fetch_patched_sources.py --package all --output-dir staging
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def get_all_package_names(packages_dir: Path) -> list[str]:
    return sorted(
        p.stem for p in packages_dir.glob("*.yml")
        if p.stem != "README"
    )


def process_package(pkg_name: str, packages_dir: Path, output_dir: Path):
    config_path = packages_dir / f"{pkg_name}.yml"
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}, skipping")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    source_repo = config["source_repo"]
    source_tag = config.get("source_tag", "")
    patch_script = config.get("patch_script")

    print(f"\n{'=' * 60}")
    print(f"Package: {pkg_name}")
    print(f"Source:  github.com/{source_repo} @ {source_tag or 'HEAD'}")
    print(f"Patch:   {patch_script or 'none'}")
    print(f"{'=' * 60}")

    clone_dir = Path(f"_clone_{pkg_name}")
    pkg_output = output_dir / pkg_name

    # Clean up previous attempts
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    if pkg_output.exists():
        shutil.rmtree(pkg_output)

    # Shallow clone (no submodules — keeps branch small)
    url = f"https://github.com/{source_repo}.git"
    cmd = ["git", "clone", "--depth=1"]
    if source_tag:
        cmd.extend(["--branch", source_tag])
    cmd.extend([url, str(clone_dir)])

    print(f"Cloning {url}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Clone failed for {pkg_name}: {result.stderr}")
        return
    print("Clone OK")

    # Apply patch script if defined
    patched = False
    if patch_script:
        # patch_script is relative to repo root (e.g. "patches/pytorch3d.py")
        patch_path = Path.cwd() / patch_script
        if patch_path.exists():
            print(f"Applying patch: {patch_script}")
            result = subprocess.run(
                [sys.executable, str(patch_path)],
                cwd=clone_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"WARNING: Patch failed: {result.stderr}")
            else:
                patched = True
                if result.stdout.strip():
                    print(f"  {result.stdout.strip()}")
        else:
            print(f"WARNING: Patch script not found: {patch_path}")

    # Remove .git (not needed, saves space)
    shutil.rmtree(clone_dir / ".git", ignore_errors=True)

    # Write metadata file
    meta = {
        "package": pkg_name,
        "source_repo": f"https://github.com/{source_repo}",
        "source_tag": source_tag or "HEAD",
        "patch_script": patch_script,
        "patched": patched,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(clone_dir / ".cuda-wheels-meta.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    # Move to output
    shutil.move(str(clone_dir), str(pkg_output))
    print(f"Staged: {pkg_output}")


def main():
    parser = argparse.ArgumentParser(description="Fetch and patch package sources")
    parser.add_argument("--package", required=True, help='Package name or "all"')
    parser.add_argument("--output-dir", required=True, help="Directory to stage sources")
    args = parser.parse_args()

    packages_dir = Path("packages")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.package == "all":
        packages = get_all_package_names(packages_dir)
    else:
        packages = [args.package]

    for pkg_name in packages:
        process_package(pkg_name, packages_dir, output_dir)

    print(f"\nDone. {len(packages)} package(s) staged in: {output_dir}")


if __name__ == "__main__":
    main()
