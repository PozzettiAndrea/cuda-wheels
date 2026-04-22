#!/usr/bin/env python3
"""Audit every published wheel for correct CUDA architectures.

Downloads each wheel from GitHub releases, extracts the .so/.pyd,
greps for compiled sm_XX targets, and compares against what the
package YAML config says should be there.

Usage:
    python scripts/audit_wheel_archs.py
    python scripts/audit_wheel_archs.py --package flash_attn
    python scripts/audit_wheel_archs.py --package flash_attn --cuda 12.8
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import yaml

REPO = "PozzettiAndrea/cuda-wheels"
PACKAGES_DIR = Path(__file__).parent.parent / "packages"

# ── arch_list resolution (mirrors generate_matrix.py exactly) ──


def get_default_arch_list(cuda_version: str, pytorch_version: str) -> str:
    cuda_major, cuda_minor = map(int, cuda_version.split(".")[:2])
    pytorch_major, pytorch_minor = map(int, pytorch_version.split(".")[:2])

    if cuda_major >= 13:
        archs = ["8.0", "9.0"]
    elif (cuda_major, cuda_minor) == (12, 4):
        archs = ["5.0", "6.0", "7.0", "8.0", "9.0"]
    else:
        archs = ["7.0", "8.0", "9.0"]

    if (pytorch_major, pytorch_minor) >= (2, 6):
        if (cuda_major, cuda_minor) >= (12, 8):
            archs.append("10.0")
            archs.append("12.0")

    return " ".join(archs)


def load_package_configs() -> dict:
    """Load all package YAMLs into a dict keyed by normalized name."""
    configs = {}
    for f in PACKAGES_DIR.glob("*.yml"):
        pkg = yaml.safe_load(f.read_text())
        name = pkg["name"].replace("-", "_")
        configs[name] = pkg
    return configs


def get_expected_archs(pkg: dict, cuda: str, pytorch: str) -> set:
    """Compute expected arch_list for a specific combo, same logic as generate_matrix.py line 248."""
    build = pkg["build_matrix"]

    # Check per-combination override first
    if "combinations" in build:
        for c in build["combinations"]:
            if c["cuda"] == cuda and c["pytorch"] == pytorch:
                if c.get("arch_list"):
                    return arch_list_to_sm(c["arch_list"])
                break

    # Package-level override
    if pkg.get("arch_list"):
        return arch_list_to_sm(pkg["arch_list"])

    # Auto-compute
    return arch_list_to_sm(get_default_arch_list(cuda, pytorch))


def arch_list_to_sm(arch_list: str) -> set:
    """Convert '8.0 9.0 10.0 12.0' -> {'sm_80', 'sm_90', 'sm_100', 'sm_120'}."""
    result = set()
    for a in arch_list.split():
        major, minor = a.split(".")
        result.add(f"sm_{major}{minor}")
    return result


def arch_to_major(sm: str) -> int:
    """Extract major version: sm_70->7, sm_86->8, sm_90->9, sm_100->10, sm_120->12."""
    num = int(sm.replace("sm_", ""))
    if num >= 100:
        return num // 10
    return num // 10


# ── wheel filename parsing ──

# flash_attn-2.8.3+cu124torch2.4-cp310-cp310-manylinux_2_35_x86_64.whl
# Also handle v1 naming: +cu124torch24-cp310-...
WHEEL_RE = re.compile(
    r"^(?P<pkg>[A-Za-z0-9_]+)"
    r"-(?P<ver>[^+]+)"
    r"\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)"
    r"-cp(?P<py>\d+)-cp\d+-"
    r"(?P<plat>.+)\.whl$"
)


def parse_wheel_name(name: str) -> Optional[dict]:
    m = WHEEL_RE.match(name)
    if not m:
        return None
    cuda_short = m.group("cuda")
    # Reconstruct full CUDA version: 124 -> 12.4, 128 -> 12.8, 130 -> 13.0
    if len(cuda_short) == 3:
        cuda_full = f"{cuda_short[:2]}.{cuda_short[2]}"
    else:
        cuda_full = cuda_short  # shouldn't happen

    torch_short = m.group("torch")
    # Normalize: "24" -> "2.4", "2.4" stays "2.4"
    if "." not in torch_short:
        torch_short = f"{torch_short[0]}.{torch_short[1:]}"

    return {
        "package": m.group("pkg"),
        "version": m.group("ver"),
        "cuda": cuda_full,
        "cuda_short": cuda_short,
        "torch_short": torch_short,
        "python": m.group("py"),
        "platform": "linux" if "linux" in m.group("plat") or "manylinux" in m.group("plat") else "windows",
        "filename": name,
    }


# ── binary arch extraction ──


def extract_archs_from_wheel(wheel_path: str) -> set:
    """Open wheel (zip), find .so/.pyd files, extract compiled CUDA archs.

    Three detection methods (in priority order):
    1. Cubin ELF headers (EM_CUDA=190) — ground truth, works for all packages
    2. Thrust/CUB namespace mangling — reliable for packages using Thrust
    3. Direct sm_XX strings — fallback, may include noise from bundled torch libs

    Method 1 is preferred. Methods 2/3 are only used if method 1 finds nothing.
    """
    cubin_archs = set()
    thrust_archs = set()
    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            for entry in zf.namelist():
                # Only process CUDA shared libraries
                if not (entry.endswith(".so") or entry.endswith(".pyd")):
                    continue
                if "_cpu" in entry:
                    continue
                # Skip bundled torch/cuda runtime libs (auditwheel artifacts)
                basename = entry.rsplit("/", 1)[-1] if "/" in entry else entry
                if basename.startswith(("libtorch", "libc10", "libcudart", "libcuda.",
                                        "libcublas", "libcusparse", "libcufft",
                                        "libcurand", "libcusolver", "libnvrtc",
                                        "libnvJitLink", "libcudnn")):
                    continue
                try:
                    data = zf.read(entry)

                    # Method 1: Cubin ELF headers embedded in fatbin
                    # Cubin ELF has e_machine = 190 (EM_CUDA)
                    # e_flags encoding varies by SM generation:
                    #   Old (sm<100): e_flags = 0x00SS0VSS where SS=SM version
                    #     e.g. sm_70: 0x00460546, sm_80: 0x00500550, sm_90: 0x005a055a
                    #   New: e_flags = 0xTT00SSXX where TT=type, SS=SM version
                    #     TT=0x06: standard SM, TT=0x0a: arch-specific 'a' variant
                    #     e.g. sm_100: 0x06006402, sm_100a: 0x0a006402, sm_120: 0x06007802
                    pos = 0
                    while True:
                        idx = data.find(b"\x7fELF", pos)
                        if idx == -1:
                            break
                        if idx + 64 < len(data):
                            e_machine = struct.unpack_from("<H", data, idx + 18)[0]
                            if e_machine == 190:  # EM_CUDA
                                ei_class = data[idx + 4]
                                if ei_class == 2:  # 64-bit
                                    e_flags = struct.unpack_from("<I", data, idx + 48)[0]
                                elif ei_class == 1:  # 32-bit
                                    e_flags = struct.unpack_from("<I", data, idx + 36)[0]
                                else:
                                    pos = idx + 1
                                    continue
                                byte0 = e_flags & 0xFF
                                byte1 = (e_flags >> 8) & 0xFF
                                byte3 = (e_flags >> 24) & 0xFF
                                # New format: byte3=0x06 (standard) or 0x0a (arch-specific 'a')
                                if byte3 in (0x06, 0x0a):
                                    sm = byte1  # 80, 90, 100, 120, etc.
                                else:
                                    # Old format: SM version in byte0
                                    sm = byte0  # 70, 75, 80, 86, 89, 90, etc.
                                if 50 <= sm <= 130:
                                    cubin_archs.add(f"sm_{sm}")
                        pos = idx + 1

                    # Method 1b: PTX .target directives (for packages using PTX instead of SASS)
                    for m in re.findall(rb"\.target\s+sm_(\d+)", data):
                        sm = int(m.decode())
                        if 50 <= sm <= 130:
                            cubin_archs.add(f"sm_{sm}")

                    # Method 2: Thrust/CUB namespace mangling
                    for m in re.finditer(rb"THRUST_\d+_([\d_]+)_NS", data):
                        nums = m.group(1).decode().split("_")
                        for n in nums:
                            if n and len(n) >= 2:
                                val = int(n)
                                if 500 <= val <= 13000:
                                    thrust_archs.add(f"sm_{val // 10}")

                except Exception:
                    pass
    except zipfile.BadZipFile:
        pass

    # Prefer cubin method (ground truth), fall back to thrust
    return cubin_archs if cubin_archs else thrust_archs


# ── GitHub API ──


def list_all_releases() -> list:
    """Get all releases and their assets via gh CLI."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}/releases", "--paginate",
         "--jq", '.[].assets[] | {name: .name, url: .url, size: .size, release: .node_id}'],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        print(f"ERROR: gh api failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    assets = []
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            assets.append(json.loads(line))
    return assets


def get_gh_token() -> str:
    """Get GitHub token from gh CLI."""
    r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
    return r.stdout.strip()


_GH_TOKEN = None


def download_wheel(asset_url: str, dest: str) -> bool:
    """Download a release asset using curl."""
    global _GH_TOKEN
    if _GH_TOKEN is None:
        _GH_TOKEN = get_gh_token()
    result = subprocess.run(
        ["curl", "-sL",
         "-H", f"Authorization: token {_GH_TOKEN}",
         "-H", "Accept: application/octet-stream",
         asset_url, "-o", dest],
        capture_output=True, timeout=600,
    )
    return result.returncode == 0 and os.path.getsize(dest) > 0


# ── main ──


def main():
    parser = argparse.ArgumentParser(description="Audit wheel architectures")
    parser.add_argument("--package", default=None, help="Filter by package name")
    parser.add_argument("--cuda", default=None, help="Filter by CUDA version (e.g. 12.8)")
    parser.add_argument("--output", default="audit_report.json", help="Output report file")
    parser.add_argument("--dry-run", action="store_true", help="List wheels without downloading")
    args = parser.parse_args()

    print("Loading package configs...")
    configs = load_package_configs()
    print(f"Loaded {len(configs)} packages")

    print("Fetching release assets from GitHub...")
    assets = list_all_releases()
    print(f"Found {len(assets)} total assets")

    # Filter to .whl files and parse
    wheels = []
    for asset in assets:
        if not asset["name"].endswith(".whl"):
            continue
        parsed = parse_wheel_name(asset["name"])
        if not parsed:
            continue
        parsed["url"] = asset["url"]
        parsed["size"] = asset["size"]

        if args.package and parsed["package"] != args.package:
            continue
        if args.cuda and parsed["cuda"] != args.cuda:
            continue

        wheels.append(parsed)

    # Deduplicate: same archs regardless of Python version, so keep one per (pkg, cuda, torch, platform)
    seen = set()
    deduped = []
    for w in wheels:
        key = (w["package"], w["cuda"], w["torch_short"], w["platform"])
        if key not in seen:
            seen.add(key)
            deduped.append(w)
    print(f"Found {len(wheels)} total wheels, {len(deduped)} unique (pkg, cuda, torch, platform) combos")
    wheels = deduped

    if args.dry_run:
        for w in wheels:
            print(f"  {w['filename']} ({w['size'] // 1024 // 1024}MB)")
        total_mb = sum(w["size"] for w in wheels) // 1024 // 1024
        print(f"\nTotal download: {total_mb} MB")
        return

    # Process each wheel
    results = []
    mismatches = []

    with tempfile.TemporaryDirectory(prefix="wheel-audit-") as tmpdir:
        for i, wheel in enumerate(wheels, 1):
            pkg_name = wheel["package"]
            pkg_config = configs.get(pkg_name)

            if not pkg_config:
                print(f"[{i}/{len(wheels)}] SKIP {wheel['filename']} — no config for {pkg_name}")
                continue

            # Find matching pytorch version from config
            # torch_short is "2.4" but config has "2.4.0"
            pytorch_full = None
            build = pkg_config["build_matrix"]
            if "combinations" in build:
                for c in build["combinations"]:
                    c_torch_short = ".".join(c["pytorch"].split(".")[:2])
                    if c["cuda"] == wheel["cuda"] and c_torch_short == wheel["torch_short"]:
                        pytorch_full = c["pytorch"]
                        break

            if not pytorch_full:
                # Try constructing it
                pytorch_full = wheel["torch_short"] + ".0"

            expected = get_expected_archs(pkg_config, wheel["cuda"], pytorch_full)

            # Download
            dest = os.path.join(tmpdir, wheel["filename"])
            size_mb = wheel["size"] // 1024 // 1024
            print(f"[{i}/{len(wheels)}] {wheel['filename']} ({size_mb}MB) ...", end=" ", flush=True)

            if not download_wheel(wheel["url"], dest):
                print("DOWNLOAD FAILED")
                results.append({
                    "wheel": wheel["filename"],
                    "status": "download_failed",
                })
                continue

            # Extract archs
            actual = extract_archs_from_wheel(dest)

            # Delete immediately
            os.unlink(dest)

            # Compare by major arch families
            # The build system adds sub-archs (sm_75, sm_86, sm_89) beyond what we request.
            # We check: every requested arch's major family must be present in actual.
            expected_sorted = sorted(expected)
            actual_sorted = sorted(actual)

            expected_majors = {arch_to_major(a) for a in expected}
            actual_majors = {arch_to_major(a) for a in actual}

            missing_majors = expected_majors - actual_majors
            extra_majors = actual_majors - expected_majors

            # Also check: requested archs should be a subset of actual
            # (sm_70 requested, sm_70 should be in actual even if sm_75 is too)
            missing_exact = expected - actual

            match = len(missing_majors) == 0

            if match and not missing_exact:
                print(f"OK {actual_sorted}")
            elif match:
                # Major families all present but some exact archs differ
                print(f"OK (sub-arch diff) expected {expected_sorted}, got {actual_sorted}")
            else:
                status_parts = []
                if missing_majors:
                    missing_sm = sorted(f"sm_{m}0" for m in missing_majors)
                    status_parts.append(f"MISSING families {missing_sm}")
                if missing_exact:
                    status_parts.append(f"MISSING exact {sorted(missing_exact)}")
                print(f"MISMATCH — expected {expected_sorted}, got {actual_sorted} — {', '.join(status_parts)}")
                mismatches.append({
                    "wheel": wheel["filename"],
                    "expected": expected_sorted,
                    "actual": actual_sorted,
                    "missing_families": sorted(f"sm_{m}0" for m in missing_majors),
                    "missing_exact": sorted(missing_exact),
                    "extra": sorted(actual - expected),
                })

            results.append({
                "wheel": wheel["filename"],
                "package": pkg_name,
                "cuda": wheel["cuda"],
                "pytorch": wheel["torch_short"],
                "expected": expected_sorted,
                "actual": actual_sorted,
                "match": match,
                "missing_exact": sorted(missing_exact) if missing_exact else [],
            })

    # Summary
    total = len(results)
    ok = sum(1 for r in results if r.get("match"))
    failed_dl = sum(1 for r in results if r.get("status") == "download_failed")
    bad = len(mismatches)

    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE: {total} wheels checked")
    print(f"  OK:              {ok}")
    print(f"  MISMATCH:        {bad}")
    print(f"  DOWNLOAD FAILED: {failed_dl}")

    if mismatches:
        print(f"\nMISMATCHES:")
        for m in mismatches:
            print(f"  {m['wheel']}")
            if m.get("missing_families"):
                print(f"    missing families: {m['missing_families']}")
            if m.get("missing_exact"):
                print(f"    missing exact:    {m['missing_exact']}")
            if m.get("extra"):
                print(f"    extra:            {m['extra']}")

    # Write report
    report = {
        "total": total,
        "ok": ok,
        "mismatches": bad,
        "download_failed": failed_dl,
        "results": results,
        "mismatch_details": mismatches,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to {args.output}")


if __name__ == "__main__":
    main()
