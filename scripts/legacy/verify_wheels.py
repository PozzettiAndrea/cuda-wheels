#!/usr/bin/env python3
"""Thoroughly verify repacked wheels.

Checks:
1. Filename-metadata version match (the original bug)
2. dist-info directory name matches version
3. RECORD hashes are correct for every file
4. WHEEL file is unchanged/valid
5. All original files are present (no data loss)
6. Binary .so/.pyd files are byte-identical to originals
7. Zip structure is valid
"""
import hashlib
import base64
import csv
import io
import re
import sys
import zipfile
from pathlib import Path

_V2_FILENAME_RE = re.compile(r'(\+cu\d+torch)(\d)\.(\d+)(-cp)')


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b'=').decode()
    return f"sha256={b64}"


def version_from_filename(filename: str) -> str | None:
    """Extract version from wheel filename: pkg-VERSION+local-cpXX-..."""
    parts = filename.split('-')
    if len(parts) >= 2:
        return parts[1]
    return None


def verify_wheel(whl_path: Path, original_path: Path = None) -> list[str]:
    """Verify a wheel file. Returns list of errors (empty = pass)."""
    errors = []
    filename = whl_path.name

    # 1. Basic zip validity
    if not zipfile.is_zipfile(whl_path):
        return [f"NOT A VALID ZIP FILE"]

    try:
        with zipfile.ZipFile(whl_path, 'r') as z:
            # Test zip integrity
            bad = z.testzip()
            if bad:
                errors.append(f"Corrupt zip entry: {bad}")
                return errors

            names = z.namelist()

            # 2. Find dist-info
            dist_info_dirs = set()
            for name in names:
                if '.dist-info/' in name:
                    dist_info_dirs.add(name.split('/')[0])

            if not dist_info_dirs:
                errors.append("No .dist-info directory found")
                return errors

            if len(dist_info_dirs) > 1:
                errors.append(f"Multiple dist-info dirs: {dist_info_dirs}")

            dist_info = list(dist_info_dirs)[0]

            # 3. Read METADATA
            meta_path = f"{dist_info}/METADATA"
            if meta_path not in names:
                errors.append(f"Missing METADATA at {meta_path}")
                return errors

            meta_text = z.read(meta_path).decode('utf-8')
            meta_version = None
            meta_name = None
            for line in meta_text.splitlines():
                if line.startswith('Version:'):
                    meta_version = line.split(':', 1)[1].strip()
                if line.startswith('Name:'):
                    meta_name = line.split(':', 1)[1].strip()

            if not meta_version:
                errors.append("No Version in METADATA")
            if not meta_name:
                errors.append("No Name in METADATA")

            # 4. Filename version matches METADATA version
            file_version = version_from_filename(filename)
            if file_version and meta_version and file_version != meta_version:
                errors.append(f"VERSION MISMATCH: filename={file_version} metadata={meta_version}")

            # 5. dist-info dir name matches {Name}-{Version}.dist-info
            if meta_name and meta_version:
                expected_dist_info = f"{meta_name}-{meta_version}.dist-info"
                if dist_info != expected_dist_info:
                    # This is a warning, not always fatal (some packages omit local version)
                    # But check it doesn't have v1 naming when it should be v2
                    if 'torch' in meta_version and 'torch' in dist_info:
                        errors.append(f"dist-info mismatch: got={dist_info} expected={expected_dist_info}")

            # 6. No v1 torch naming in METADATA version (the whole point)
            if meta_version and _V2_FILENAME_RE.search(filename):
                # Filename is v2, check metadata is also v2
                m = _V2_FILENAME_RE.search(filename)
                v2_torch = f"torch{m.group(2)}.{m.group(3)}"
                v1_torch = f"torch{m.group(2)}{m.group(3)}"
                if v1_torch in meta_version and v2_torch not in meta_version:
                    errors.append(f"METADATA still has v1 naming: {meta_version}")

            # 7. Verify RECORD hashes
            record_path = f"{dist_info}/RECORD"
            if record_path in names:
                record_text = z.read(record_path).decode('utf-8')
                reader = csv.reader(io.StringIO(record_text))
                for row in reader:
                    if len(row) < 3:
                        continue
                    rec_file, rec_hash, rec_size = row[0], row[1], row[2]
                    if not rec_hash:  # RECORD itself has no hash
                        continue
                    if rec_file not in names:
                        errors.append(f"RECORD references missing file: {rec_file}")
                        continue
                    data = z.read(rec_file)
                    actual_hash = record_hash(data)
                    if actual_hash != rec_hash:
                        errors.append(f"RECORD hash mismatch for {rec_file}: expected={rec_hash} got={actual_hash}")
                    if rec_size and int(rec_size) != len(data):
                        errors.append(f"RECORD size mismatch for {rec_file}: expected={rec_size} got={len(data)}")
            else:
                errors.append("Missing RECORD file")

            # 8. WHEEL file exists and is valid
            wheel_path = f"{dist_info}/WHEEL"
            if wheel_path not in names:
                errors.append("Missing WHEEL file")
            else:
                wheel_text = z.read(wheel_path).decode('utf-8')
                if 'Wheel-Version:' not in wheel_text:
                    errors.append("WHEEL file missing Wheel-Version")

            # 9. Compare with original if provided
            if original_path and original_path.exists():
                with zipfile.ZipFile(original_path, 'r') as zorig:
                    orig_names = set(zorig.namelist())
                    new_names = set(names)

                    # Check all non-dist-info files are identical
                    for name in orig_names:
                        if '.dist-info/' in name:
                            continue
                        if name not in new_names:
                            errors.append(f"MISSING from repack: {name}")
                        else:
                            orig_data = zorig.read(name)
                            new_data = z.read(name)
                            if orig_data != new_data:
                                errors.append(f"DATA CHANGED: {name} (orig={len(orig_data)} new={len(new_data)})")

    except Exception as e:
        errors.append(f"Exception: {e}")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_wheels.py <wheel_or_dir> [original_dir]")
        sys.exit(1)

    target = Path(sys.argv[1])
    original_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if target.is_dir():
        wheels = sorted(target.glob("*.whl"))
    else:
        wheels = [target]

    total = len(wheels)
    passed = 0
    failed = 0

    for whl in wheels:
        original = None
        if original_dir:
            original = original_dir / whl.name

        errors = verify_wheel(whl, original)
        if errors:
            failed += 1
            print(f"FAIL: {whl.name}")
            for e in errors:
                print(f"  - {e}")
        else:
            passed += 1
            print(f"  OK: {whl.name}")

    print(f"\n{'='*60}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
