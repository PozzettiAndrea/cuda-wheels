#!/usr/bin/env python3
"""Repack all wheels across all releases: parallel download, repack, verify, upload.

Processes one release at a time to limit disk usage, but parallelizes
all steps within each release using 16 workers.
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "PozzettiAndrea/cuda-wheels"
WORKERS = 16
SCRIPTS = Path(__file__).parent


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def get_all_releases():
    r = run(["gh", "release", "list", "--repo", REPO, "--limit", "100", "--json", "tagName"])
    return [x["tagName"] for x in json.loads(r.stdout)]


def get_v2_wheel_names(tag):
    """Get v2-named wheel filenames from a release."""
    r = run(["gh", "release", "view", tag, "--repo", REPO, "--json", "assets"])
    assets = json.loads(r.stdout)["assets"]
    import re
    v2_re = re.compile(r'torch\d+\.\d+')
    return [a["name"] for a in assets if a["name"].endswith(".whl") and v2_re.search(a["name"])]


def download_wheel(tag, name, dest_dir, retries=3):
    """Download one wheel with retries. Returns (name, success)."""
    for attempt in range(retries):
        r = run(["gh", "release", "download", tag, "--repo", REPO,
                 "-p", name, "-D", str(dest_dir), "--clobber"])
        if r.returncode == 0:
            return (name, True)
        time.sleep(1 * (attempt + 1))
    return (name, False)


def repack_wheel_file(whl_path, output_dir):
    """Repack one wheel. Returns (name, output_path_or_None, error_or_None)."""
    # Import here to avoid circular issues
    from repack_wheels_v2 import repack_wheel
    try:
        result = repack_wheel(whl_path, output_dir)
        return (whl_path.name, result, None)
    except Exception as e:
        return (whl_path.name, None, str(e))


def verify_wheel_file(whl_path, original_dir):
    """Verify one wheel. Returns (name, errors_list)."""
    from verify_wheels import verify_wheel
    original = original_dir / whl_path.name
    orig = original if original.exists() else None
    errors = verify_wheel(whl_path, orig)
    return (whl_path.name, errors)


def upload_wheels(tag, wheel_paths, batch_size=20):
    """Upload wheels in batches."""
    total = 0
    for i in range(0, len(wheel_paths), batch_size):
        batch = wheel_paths[i:i + batch_size]
        cmd = ["gh", "release", "upload", tag, "--repo", REPO, "--clobber"]
        cmd.extend(str(p) for p in batch)
        r = run(cmd)
        if r.returncode != 0:
            print(f"  UPLOAD ERROR: {r.stderr[:200]}")
            return total
        total += len(batch)
    return total


def process_release(tag, dry_run=False):
    """Process one release: download → repack → verify → upload. All parallel."""
    import shutil

    workdir = Path(f"/tmp/wheel-repack/{tag}")
    orig_dir = workdir / "original"
    repack_dir = workdir / "repacked"

    # Clean
    if workdir.exists():
        shutil.rmtree(workdir)
    orig_dir.mkdir(parents=True)
    repack_dir.mkdir(parents=True)

    # Get wheel list
    wheels = get_v2_wheel_names(tag)
    if not wheels:
        print(f"  No v2 wheels, skipping")
        return 0, 0

    dl_workers = min(WORKERS, 8)  # cap download threads to avoid API rate limits
    print(f"  {len(wheels)} wheels — downloading ({dl_workers} threads)...")

    # === PARALLEL DOWNLOAD ===
    failed_dl = []
    with ThreadPoolExecutor(max_workers=dl_workers) as pool:
        futs = {pool.submit(download_wheel, tag, w, orig_dir): w for w in wheels}
        for fut in as_completed(futs):
            name, ok = fut.result()
            if not ok:
                failed_dl.append(name)

    dl_count = len(list(orig_dir.glob("*.whl")))
    if failed_dl:
        print(f"  WARNING: {len(failed_dl)} download failures: {failed_dl[:3]}...")
    print(f"  Downloaded {dl_count}/{len(wheels)}")

    if dl_count == 0:
        print(f"  No wheels downloaded, skipping")
        shutil.rmtree(workdir, ignore_errors=True)
        return 0, 0

    # === PARALLEL REPACK ===
    print(f"  Repacking ({WORKERS} threads)...")
    whl_files = sorted(orig_dir.glob("*.whl"))
    repacked = []
    skipped = 0
    errors = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(repack_wheel_file, w, repack_dir): w for w in whl_files}
        for fut in as_completed(futs):
            name, result, err = fut.result()
            if err:
                errors.append((name, err))
            elif result:
                repacked.append(result)
            else:
                skipped += 1

    print(f"  Repacked: {len(repacked)} | Skipped (already OK): {skipped} | Errors: {len(errors)}")
    for name, err in errors:
        print(f"    ERROR: {name}: {err}")

    if not repacked:
        print(f"  Nothing to upload")
        shutil.rmtree(workdir, ignore_errors=True)
        return 0, 0

    # === PARALLEL VERIFY ===
    print(f"  Verifying ({WORKERS} threads)...")
    verify_failures = []
    repacked_paths = sorted(repack_dir.glob("*.whl"))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(verify_wheel_file, w, orig_dir): w for w in repacked_paths}
        for fut in as_completed(futs):
            name, errs = fut.result()
            if errs:
                verify_failures.append((name, errs))

    if verify_failures:
        print(f"  VERIFICATION FAILED for {len(verify_failures)} wheels:")
        for name, errs in verify_failures[:5]:
            print(f"    {name}: {errs[0]}")
        print(f"  NOT UPLOADING {tag}")
        shutil.rmtree(workdir)
        return 0, len(verify_failures)

    print(f"  All {len(repacked_paths)} verified OK")

    # === UPLOAD ===
    if dry_run:
        print(f"  DRY RUN — would upload {len(repacked_paths)} wheels")
        shutil.rmtree(workdir)
        return len(repacked_paths), 0

    print(f"  Uploading {len(repacked_paths)} wheels...")
    uploaded = upload_wheels(tag, repacked_paths)
    print(f"  Uploaded {uploaded}/{len(repacked_paths)}")

    # Clean up
    shutil.rmtree(workdir, ignore_errors=True)
    return uploaded, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--package", default=None, help="Only this release tag (e.g. nvdiffrast-latest)")
    parser.add_argument("--skip", nargs="*", default=[], help="Release tags to skip")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    global WORKERS
    WORKERS = args.workers

    # Add scripts dir to path so we can import repack/verify
    sys.path.insert(0, str(SCRIPTS))

    releases = get_all_releases()
    if args.package:
        releases = [r for r in releases if args.package in r]

    # Skip already-done releases
    releases = [r for r in releases if r not in args.skip]

    print(f"Processing {len(releases)} releases with {WORKERS} workers")
    if args.dry_run:
        print("DRY RUN MODE")

    total_uploaded = 0
    total_failed = 0

    for i, tag in enumerate(sorted(releases), 1):
        print(f"\n[{i}/{len(releases)}] {tag}")
        uploaded, failed = process_release(tag, dry_run=args.dry_run)
        total_uploaded += uploaded
        total_failed += failed

    print(f"\n{'='*60}")
    print(f"DONE: {total_uploaded} uploaded, {total_failed} failed")


if __name__ == "__main__":
    main()
