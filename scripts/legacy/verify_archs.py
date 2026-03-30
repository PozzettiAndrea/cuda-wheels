#!/usr/bin/env python3
"""Verify CUDA architectures embedded in built wheels.

Downloads one wheel per (package, cuda, platform) group, extracts .so/.pyd
files, and checks which SM architectures are compiled in. Compares against
the expected arch_list from package YAMLs.

Usage:
    python scripts/verify_archs.py                    # all packages
    python scripts/verify_archs.py --package gsplat   # single package
    python scripts/verify_archs.py --json             # JSON output
    python scripts/verify_archs.py --threads 16       # more parallelism
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

# Add scripts/ to path so we can import generate_matrix
sys.path.insert(0, str(Path(__file__).parent))
from generate_matrix import get_default_arch_list


# ---------------------------------------------------------------------------
# CUDA arch compatibility: sm_80 covers sm_86/sm_87/sm_89 etc.
# Within a major arch family, lower minor is forward-compatible.
# ---------------------------------------------------------------------------

ARCH_FAMILIES = {
    "7.0": 7, "7.5": 7,
    "8.0": 8, "8.6": 8, "8.7": 8, "8.9": 8,
    "9.0": 9,
    "10.0": 10, "10.1": 10,
    "12.0": 12, "12.1": 12,
    "13.0": 13,
}


def archs_covered(found_archs: set[str]) -> set[str]:
    """Given a set of found SM architectures, return all archs they cover
    via forward compatibility within the same major family."""
    covered = set(found_archs)
    # For each found arch, it covers all higher minor versions in the same family
    for arch in found_archs:
        family = ARCH_FAMILIES.get(arch)
        if family is None:
            continue
        try:
            major, minor = map(int, arch.split("."))
        except ValueError:
            continue
        # Add all known archs in the same family with >= minor version
        for known_arch, known_family in ARCH_FAMILIES.items():
            if known_family == family:
                try:
                    km, kn = map(int, known_arch.split("."))
                except ValueError:
                    continue
                if kn >= minor:
                    covered.add(known_arch)
    return covered


# ---------------------------------------------------------------------------
# Wheel URL collection (no GH API - scrape gh-pages or local docs/)
# ---------------------------------------------------------------------------

def collect_wheel_urls_from_index(base_url: str, packages: list[str] | None = None) -> dict:
    """Scrape PEP 503 HTML index for wheel download URLs.
    Returns {package_name: [{filename, url, cuda, torch, python, platform}, ...]}
    """
    link_re = re.compile(r'href="([^"]+)"[^>]*>([^<]+\.whl)</a>', re.IGNORECASE)
    wheel_re = re.compile(
        r'^(?P<pkg>[^-]+)-(?P<ver>[^-]+)\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)-'
        r'cp(?P<py>\d+)-[^-]+-(?P<plat>linux_x86_64|manylinux[^.]+_x86_64|win_amd64)\.whl$'
    )

    result = {}

    # Get root index to find all packages
    try:
        req = urllib.request.Request(f"{base_url}/index.html",
                                     headers={"User-Agent": "verify-archs/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            root_html = resp.read().decode()
    except Exception as e:
        print(f"Failed to fetch index: {e}", file=sys.stderr)
        return {}

    pkg_re = re.compile(r'href="([^"]+)/"', re.IGNORECASE)
    available_pkgs = [m.group(1) for m in pkg_re.finditer(root_html)]

    for pkg_name in available_pkgs:
        if packages and pkg_name not in packages:
            continue

        try:
            req = urllib.request.Request(f"{base_url}/{pkg_name}/index.html",
                                         headers={"User-Agent": "verify-archs/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                pkg_html = resp.read().decode()
        except Exception:
            continue

        wheels = []
        for match in link_re.finditer(pkg_html):
            url, filename = match.group(1), match.group(2)
            m = wheel_re.match(filename)
            if not m:
                continue
            cuda_short = m.group("cuda")
            cuda_ver = f"{cuda_short[:-1]}.{cuda_short[-1]}" if len(cuda_short) <= 3 else cuda_short
            plat = "linux" if "linux" in m.group("plat") else "windows"
            wheels.append({
                "filename": filename,
                "url": url,
                "cuda": cuda_ver,
                "torch": m.group("torch"),
                "python": m.group("py"),
                "platform": plat,
            })

        if wheels:
            result[pkg_name.replace("-", "_")] = wheels

    return result


def collect_wheel_urls_from_releases(packages: list[str] | None = None) -> dict:
    """Collect wheel URLs from GitHub releases using gh CLI."""
    result = {}
    wheel_re = re.compile(
        r'^(?P<pkg>[^-]+)-(?P<ver>[^-]+)\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)-'
        r'cp(?P<py>\d+)-[^-]+-(?P<plat>linux_x86_64|manylinux[^.]+_x86_64|win_amd64)\.whl$'
    )

    repo = "PozzettiAndrea/cuda-wheels"
    try:
        out = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases", "--paginate", "-q",
             '.[].assets[] | select(.name | endswith(".whl")) | "\(.name)\t\(.browser_download_url)"'],
            capture_output=True, text=True, timeout=60
        )
        if out.returncode != 0:
            return {}
    except Exception:
        return {}

    for line in out.stdout.strip().split("\n"):
        if not line or "\t" not in line:
            continue
        filename, url = line.split("\t", 1)
        m = wheel_re.match(filename)
        if not m:
            continue
        pkg = m.group("pkg")
        if packages and pkg not in packages:
            continue
        cuda_short = m.group("cuda")
        cuda_ver = f"{cuda_short[:-1]}.{cuda_short[-1]}" if len(cuda_short) <= 3 else cuda_short
        plat = "linux" if "linux" in m.group("plat") else "windows"
        result.setdefault(pkg, []).append({
            "filename": filename,
            "url": url,
            "cuda": cuda_ver,
            "torch": m.group("torch"),
            "python": m.group("py"),
            "platform": plat,
        })

    return result


def sample_wheels(wheels: list[dict]) -> list[dict]:
    """Pick one wheel per (cuda, platform) group — different python/torch
    versions compile the same CUDA kernels."""
    seen = set()
    sampled = []
    for w in wheels:
        key = (w["cuda"], w["platform"])
        if key not in seen:
            seen.add(key)
            sampled.append(w)
    return sampled


# ---------------------------------------------------------------------------
# CUDA architecture detection from binaries
# ---------------------------------------------------------------------------

# NVIDIA fatbin magic: 0xBA5EBA00 (little-endian: 00 BA 5E BA)
FATBIN_MAGIC = b"\xba\x5e\xba\x00"
# Older fatbin magic
FATBIN_MAGIC_OLD = b"\xed\xfe\xba\xbe"


def detect_archs_cuobjdump(filepath: str) -> set[str] | None:
    """Try cuobjdump to list embedded architectures. Returns None if unavailable."""
    try:
        out = subprocess.run(
            ["cuobjdump", "--list-elf", filepath],
            capture_output=True, text=True, timeout=30
        )
        if out.returncode != 0:
            return None
        archs = set()
        for line in out.stdout.split("\n"):
            m = re.search(r"sm_(\d+)([a-z]?)", line)
            if m:
                sm = int(m.group(1))
                major, minor = sm // 10, sm % 10
                archs.add(f"{major}.{minor}")
        return archs if archs else None
    except FileNotFoundError:
        return None
    except Exception:
        return None


def detect_archs_binary(data: bytes) -> set[str]:
    """Parse CUDA fatbin sections from binary data to find embedded SM architectures.

    The fatbin format:
    - Starts with magic 0xBA5EBA00
    - Header: 4 bytes magic, 2 bytes version, 2 bytes header_size, 8 bytes fat_size
    - Followed by entries, each with:
      - 2 bytes kind (1=PTX, 2=ELF/cubin)
      - 2 bytes unknown
      - 4 bytes header_size
      - 8 bytes padded_payload_size
      - 4 bytes unknown
      - 4 bytes sm_version (compute capability * 10, e.g. 80 for sm_80)
      ... more fields ...
      - payload data
    """
    archs = set()

    # Find all fatbin sections
    pos = 0
    while True:
        idx = data.find(FATBIN_MAGIC, pos)
        if idx == -1:
            break

        try:
            archs.update(_parse_fatbin(data, idx))
        except Exception:
            pass
        pos = idx + 4

    # Also try string-based detection as fallback
    archs.update(_detect_archs_strings(data))

    return archs


def _parse_fatbin(data: bytes, offset: int) -> set[str]:
    """Parse a fatbin starting at offset, extracting SM versions."""
    archs = set()

    if len(data) < offset + 16:
        return archs

    # Fatbin header
    magic = struct.unpack_from("<I", data, offset)[0]
    if magic != 0x00BA5EBA:
        return archs

    version = struct.unpack_from("<H", data, offset + 4)[0]
    header_size = struct.unpack_from("<H", data, offset + 6)[0]
    fat_size = struct.unpack_from("<Q", data, offset + 8)[0]

    # Iterate entries within the fatbin
    entry_offset = offset + header_size
    end = offset + header_size + fat_size if fat_size > 0 else len(data)

    while entry_offset + 24 <= end and entry_offset + 24 <= len(data):
        # Entry header
        kind = struct.unpack_from("<H", data, entry_offset)[0]
        if kind == 0:
            break
        entry_header_size = struct.unpack_from("<I", data, entry_offset + 4)[0]
        padded_size = struct.unpack_from("<Q", data, entry_offset + 8)[0]

        if entry_header_size < 24:
            break

        # SM version is at offset 20 in the entry header
        if entry_offset + 20 + 4 <= len(data):
            sm_val = struct.unpack_from("<I", data, entry_offset + 20)[0]
            if 10 <= sm_val <= 200:  # reasonable range
                major, minor = sm_val // 10, sm_val % 10
                archs.add(f"{major}.{minor}")

        # Move to next entry
        next_offset = entry_offset + entry_header_size + padded_size
        if next_offset <= entry_offset:
            break
        entry_offset = next_offset

    return archs


def _detect_archs_strings(data: bytes) -> set[str]:
    """Fallback: search for sm_XX patterns in binary strings."""
    archs = set()
    for m in re.finditer(rb"sm_(\d{2,3})(?![0-9])", data):
        sm = int(m.group(1))
        if 50 <= sm <= 200:
            major, minor = sm // 10, sm % 10
            archs.add(f"{major}.{minor}")
    return archs


def detect_archs_in_wheel(whl_path: Path) -> set[str]:
    """Extract .so/.pyd files from wheel and detect CUDA architectures."""
    archs = set()
    try:
        with zipfile.ZipFile(whl_path) as zf:
            for name in zf.namelist():
                if not (name.endswith(".so") or name.endswith(".pyd")):
                    continue
                # Skip non-CUDA libraries (e.g. pure C extensions)
                data = zf.read(name)
                if len(data) < 100:
                    continue

                # Try cuobjdump first (more reliable)
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(name)[1], delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    result = detect_archs_cuobjdump(tmp_path)
                    if result:
                        archs.update(result)
                    else:
                        # Fall back to binary parsing
                        archs.update(detect_archs_binary(data))
                finally:
                    os.unlink(tmp_path)

    except Exception as e:
        print(f"  Error inspecting {whl_path.name}: {e}", file=sys.stderr)

    return archs


# ---------------------------------------------------------------------------
# Expected arch_list resolution
# ---------------------------------------------------------------------------

def load_package_configs(packages_dir: Path) -> dict:
    """Load all package YAML configs."""
    configs = {}
    for yml in sorted(packages_dir.glob("*.yml")):
        pkg = yaml.safe_load(yml.read_text())
        configs[pkg["name"]] = pkg
    return configs


def get_expected_archs(pkg_config: dict, cuda_ver: str, platform: str) -> set[str] | None:
    """Resolve the expected arch_list for a specific (cuda, platform) combo.
    Returns None if this combo isn't in the build matrix."""
    build = pkg_config.get("build_matrix", {})
    combos = build.get("combinations", [])

    for c in combos:
        if c["cuda"] != cuda_ver:
            continue
        if platform == "linux" and "linux" not in build.get("platforms", []):
            continue
        if platform == "windows" and "windows" not in build.get("platforms", []):
            continue

        # Per-combo arch_list > package-level > default
        arch_str = (c.get("arch_list")
                    or pkg_config.get("arch_list")
                    or get_default_arch_list(cuda_ver, c["pytorch"]))
        return set(arch_str.split())

    return None


# ---------------------------------------------------------------------------
# Download + verify
# ---------------------------------------------------------------------------

def download_wheel(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download a wheel file. Returns True on success."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "verify-archs/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED to download after {max_retries} attempts: {e}", file=sys.stderr)
                return False
    return False


def verify_one_wheel(wheel_info: dict, pkg_config: dict, tmpdir: Path) -> dict:
    """Download, inspect, and compare one wheel. Returns result dict."""
    filename = wheel_info["filename"]
    cuda = wheel_info["cuda"]
    platform = wheel_info["platform"]

    result = {
        "package": pkg_config["name"],
        "filename": filename,
        "cuda": cuda,
        "platform": platform,
        "status": "unknown",
    }

    # Get expected archs
    expected = get_expected_archs(pkg_config, cuda, platform)
    if expected is None:
        result["status"] = "skip"
        result["reason"] = "combo not in build matrix"
        return result
    result["expected"] = sorted(expected)

    # Download
    dest = tmpdir / filename
    if not download_wheel(wheel_info["url"], dest):
        result["status"] = "error"
        result["reason"] = "download failed"
        return result

    try:
        # Detect architectures
        found = detect_archs_in_wheel(dest)
        result["found"] = sorted(found)

        if not found:
            result["status"] = "unknown"
            result["reason"] = "no CUDA archs detected (pure Python or parsing failed)"
            return result

        # Check coverage: found archs + forward compatibility
        covered = archs_covered(found)
        missing = expected - covered
        extra = found - expected

        if missing:
            result["status"] = "MISSING"
            result["missing"] = sorted(missing)
        else:
            result["status"] = "PASS"

        if extra:
            result["extra"] = sorted(extra)

    finally:
        dest.unlink(missing_ok=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify CUDA architectures in built wheels")
    parser.add_argument("--package", "-p", help="Check only this package")
    parser.add_argument("--threads", "-t", type=int, default=8, help="Download threads (default: 8)")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    parser.add_argument("--index-url", default="https://pozzettiandrea.github.io/cuda-wheels/v2",
                        help="Base URL of PEP 503 wheel index")
    parser.add_argument("--use-gh-api", action="store_true",
                        help="Use GitHub API instead of gh-pages index")
    args = parser.parse_args()

    packages_dir = Path(__file__).parent.parent / "packages"
    pkg_configs = load_package_configs(packages_dir)
    filter_pkgs = [args.package] if args.package else None

    # Collect wheel URLs
    print("Collecting wheel URLs...")
    if args.use_gh_api:
        all_wheels = collect_wheel_urls_from_releases(filter_pkgs)
    else:
        all_wheels = collect_wheel_urls_from_index(args.index_url, filter_pkgs)

    if not all_wheels:
        print("No wheels found. Try --use-gh-api if gh-pages index is not available.")
        sys.exit(1)

    # Sample: one wheel per (package, cuda, platform)
    total_sampled = 0
    total_wheels = 0
    work_items = []
    for pkg_name, wheels in sorted(all_wheels.items()):
        total_wheels += len(wheels)
        sampled = sample_wheels(wheels)
        total_sampled += len(sampled)
        pkg_config = pkg_configs.get(pkg_name)
        if not pkg_config:
            print(f"  Warning: no YAML config for {pkg_name}, skipping")
            continue
        for w in sampled:
            work_items.append((w, pkg_config))

    print(f"  {total_wheels} total wheels, {total_sampled} sampled (1 per cuda/platform)")
    print(f"  {len(work_items)} to verify across {len(all_wheels)} packages")
    print()

    # Verify in parallel
    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="verify_archs_"))

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {}
        for w, cfg in work_items:
            f = pool.submit(verify_one_wheel, w, cfg, tmpdir)
            futures[f] = w

        for i, f in enumerate(as_completed(futures), 1):
            r = f.result()
            results.append(r)

            # Progress output
            status = r["status"]
            pkg = r["package"]
            cuda = r["cuda"]
            plat = r["platform"]

            if status == "PASS":
                found_str = " ".join(r.get("found", []))
                extra_str = f" (+{' '.join(r['extra'])})" if r.get("extra") else ""
                print(f"  [{i}/{len(work_items)}] PASS  {pkg:<20} cu{cuda:<5} {plat:<8} archs: {found_str}{extra_str}")
            elif status == "MISSING":
                missing_str = " ".join(r.get("missing", []))
                found_str = " ".join(r.get("found", []))
                print(f"  [{i}/{len(work_items)}] FAIL  {pkg:<20} cu{cuda:<5} {plat:<8} found: {found_str}  MISSING: {missing_str}")
            elif status == "unknown":
                reason = r.get("reason", "")
                print(f"  [{i}/{len(work_items)}] ????  {pkg:<20} cu{cuda:<5} {plat:<8} {reason}")
            elif status == "error":
                reason = r.get("reason", "")
                print(f"  [{i}/{len(work_items)}] ERR   {pkg:<20} cu{cuda:<5} {plat:<8} {reason}")
            elif status == "skip":
                pass

    # Cleanup
    try:
        tmpdir.rmdir()
    except OSError:
        pass

    # Summary
    print()
    print("=" * 70)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "MISSING")
    unknown = sum(1 for r in results if r["status"] == "unknown")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skip")
    print(f"PASS: {passed}  MISSING: {failed}  UNKNOWN: {unknown}  ERROR: {errors}  SKIP: {skipped}")

    if failed:
        print(f"\nFailed wheels:")
        for r in results:
            if r["status"] == "MISSING":
                print(f"  {r['package']} cu{r['cuda']} {r['platform']}: "
                      f"expected {' '.join(r['expected'])}, "
                      f"found {' '.join(r.get('found', []))}, "
                      f"missing {' '.join(r['missing'])}")

    # JSON output
    if args.json:
        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": {"pass": passed, "missing": failed, "unknown": unknown,
                         "error": errors, "skip": skipped},
            "results": results,
        }
        report_path = Path("arch_verification.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report: {report_path}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
