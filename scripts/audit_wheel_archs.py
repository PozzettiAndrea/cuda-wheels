#!/usr/bin/env python3
"""Audit every published wheel for correct CUDA architectures.

Downloads each wheel from GitHub releases, extracts the .so/.pyd,
greps for compiled sm_XX targets, and compares against what the
package YAML config says should be there.

Usage:
    python scripts/audit_wheel_archs.py
    python scripts/audit_wheel_archs.py --package flash_attn
    python scripts/audit_wheel_archs.py --package flash_attn --cuda 12.8

KNOWN FALSE POSITIVE -- compressed fatbins. Packages whose setup.py passes
`-Xfatbin -compress-all` (pointnet2_ops does) store their cubins compressed,
and this scanner cannot see the sm_XX markers inside them. Such wheels are
reported as MISMATCH with `actual_sass: []`, usually showing only the PTX
entry for the highest arch.

Before believing a MISMATCH, check the wheel for real:

    cuobjdump --list-elf <extracted .so or .pyd> | grep -o 'sm_[0-9]*' | sort -u

and cross-check the build log for the gencode flags nvcc actually received.
pointnet2_ops was verified this way: reported as missing 6 of 7 archs, but
cuobjdump shows all 7 present.
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

sys.path.insert(0, str(Path(__file__).parent))
import generate_matrix as _GM  # noqa: E402  -- single source of truth for arch resolution

REPO = "PozzettiAndrea/cuda-wheels"
PACKAGES_DIR = Path(__file__).parent.parent / "packages"

# ── arch_list resolution (mirrors generate_matrix.py exactly) ──


# NOTE: there is deliberately no hand-written arch table here any more.
# There used to be one, with a docstring claiming it "mirrors generate_matrix.py
# exactly". It did not: for cu124 it returned 5.0 6.0 7.0 8.0 9.0 while the grid
# specifies 5.0;6.0;7.0;7.5;8.0;8.6;9.0 -- omitting Turing and Ampere consumer,
# the two most common cards in this audience. A verifier that computes its own
# idea of the truth verifies nothing. Expectations now come from the same
# resolver the builds use.


def load_package_configs() -> dict:
    """Load all package YAMLs into a dict keyed by normalized name."""
    configs = {}
    for f in PACKAGES_DIR.glob("*.yml"):
        # _defaults.yml holds shared build matrix, not a package.
        if f.name.startswith("_"):
            continue
        pkg = yaml.safe_load(f.read_text())
        name = pkg["name"].replace("-", "_")
        configs[name] = pkg
    return configs


def get_expected_archs(pkg: dict, cuda: str, pytorch: str) -> set:
    """Expected arch set for one combo, resolved exactly as the build resolves it.

    Delegates to generate_matrix.resolve_arch_list so the planner and the auditor
    cannot disagree -- the same reason gap_analysis.py imports PHANTOM_COMBOS
    rather than keeping its own copy. That resolver also honours
    `arch_list_by_cuda`, which six packages use and which the previous
    implementation ignored entirely.
    """
    build = pkg.get("build_matrix") or {}

    combo_arch_list = None
    for c in build.get("combinations") or []:
        if str(c.get("cuda")) == str(cuda) and str(c.get("pytorch")) == str(pytorch):
            combo_arch_list = c.get("arch_list")
            break

    # _defaults.yml rows carry cells only; the arch source is the policy
    # file, resolved exactly as the build resolves it (CW-ADR-0012).
    try:
        default_arch_list = _GM.policy_arch_list(str(cuda), str(pytorch))
    except KeyError:
        default_arch_list = None

    resolved = _GM.resolve_arch_list(
        pkg, str(cuda),
        combo_arch_list=combo_arch_list,
        pytorch_version=str(pytorch),
        default_arch_list=default_arch_list,
    )
    return arch_list_to_sm(resolved)


def arch_list_to_sm(arch_list: str) -> set:
    """Convert '8.0 9.0 10.0 12.0' -> {'sm_80', 'sm_90', 'sm_100', 'sm_120'}.

    Accepts both separators in use: space-separated (as in the older package
    YAMLs) and semicolon-separated TORCH_CUDA_ARCH_LIST form (as in
    _defaults.yml, e.g. '7.0;7.5;9.0+PTX'). A trailing '+PTX' marks an extra
    PTX blob for that same arch, not a different one, so it is stripped.
    """
    result = set()
    for a in arch_list.replace(";", " ").split():
        a = a.split("+")[0].strip()
        if not a:
            continue
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


def extract_archs_from_wheel(wheel_path: str):
    """Open wheel (zip), find .so/.pyd files, extract compiled CUDA archs.

    Returns a dict {"sass": set(sm_XX), "ptx": set(sm_XX)}:
    - SASS comes from cubin ELF headers (EM_CUDA=190) embedded in fatbin.
    - PTX comes from `.target sm_XX` directives in PTX sections of fatbin.

    A wheel may ship SASS-only, PTX-only, or both. The two sets are kept
    separate so the audit can show coverage breakdown per cell.
    """
    sass_archs = set()
    ptx_archs = set()
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
                                        "libnvJitLink", "libcudnn",
                                        "libcaffe2_nvrtc")):
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
                                    sass_archs.add(f"sm_{sm}")
                        pos = idx + 1

                    # Method 1b: PTX .target directives — these are PTX (text)
                    # sections of the fatbin, separate from SASS cubin ELFs.
                    for m in re.findall(rb"\.target\s+sm_(\d+)", data):
                        sm = int(m.decode())
                        if 50 <= sm <= 130:
                            ptx_archs.add(f"sm_{sm}")

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

    # If we found nothing via SASS or PTX, fall back to thrust mangling for SASS.
    if not sass_archs and not ptx_archs and thrust_archs:
        sass_archs = thrust_archs
    return {"sass": sass_archs, "ptx": ptx_archs}


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

    print(f"Found {len(wheels)} wheels to audit")

    if args.dry_run:
        for w in wheels:
            print(f"  {w['filename']} ({w['size'] // 1024 // 1024}MB)")
        total_mb = sum(w["size"] for w in wheels) // 1024 // 1024
        print(f"\nTotal download: {total_mb} MB")
        return

    # Process each wheel
    results = []
    mismatches = []

    unverified = []
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
            build = pkg_config.get("build_matrix") or {}
            combos = build.get("combinations") or _GM.DEFAULTS.get("combinations") or []
            if combos:
                for c in combos:
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

            # Extract archs (SASS + PTX separately)
            extracted = extract_archs_from_wheel(dest)
            actual_sass = extracted["sass"]
            actual_ptx = extracted["ptx"]
            # Combined set is used for backwards-compatible mismatch logic;
            # the audit cares whether ANY representation (SASS or PTX) covers
            # the expected arch family.
            actual = actual_sass | actual_ptx

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

            # A scan that found NOTHING has not detected a defect -- it has
            # failed to look. nvcc compresses device code by default from CUDA
            # 12.8 (LZ4), so the SASS markers this scanner greps for are inside
            # a compressed payload and simply invisible. Reporting MISMATCH there
            # is a false alarm, and a verifier that cries wolf on most of the
            # grid trains you to ignore it. Say UNVERIFIED and mean it.
            if not actual:
                print(f"UNVERIFIED — expected {expected_sorted}, scan found no SASS "
                      f"(device code is compressed by default on CUDA 12.8+); "
                      f"confirm with: cuobjdump --list-elf <extracted .so>")
                unverified.append({
                    "wheel": wheel["filename"],
                    "expected": expected_sorted,
                    "reason": "no SASS visible to the byte scanner",
                })
                continue

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
                "pytorch_full": pytorch_full,
                "python": "cp" + wheel["python"],
                "platform": wheel["platform"],
                "size": wheel["size"],
                "expected": expected_sorted,
                "actual": actual_sorted,
                "actual_sass": sorted(actual_sass),
                "actual_ptx": sorted(actual_ptx),
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
    print(f"  UNVERIFIED:      {len(unverified)}  (scan blind -- compressed fatbins)")
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
