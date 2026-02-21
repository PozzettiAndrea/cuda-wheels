#!/usr/bin/env python3
"""Inspect ALL wheels (built releases + external HTML indices).

Downloads each wheel, extracts metadata, groups into clusters,
deletes after inspection. Disk usage stays under 1GB.

A "cluster" is a group of wheels that share the same internal structure:
same .so/.pyd files, same dependencies, same compile artifacts.
Wheels in a cluster only differ by Python/CUDA/Torch/platform tags.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Wheel filename parsing
# ---------------------------------------------------------------------------

def parse_wheel_filename(filename: str) -> dict:
    m = re.match(
        r"^(?P<pkg>[^-]+)-(?P<ver>[^-]+)-(?P<pytag>cp\d+)-[^-]+-(?P<plat>.+)\.whl$",
        filename,
    )
    if not m:
        return {}

    info = {"package": m.group("pkg"), "version": m.group("ver"),
            "python_tag": m.group("pytag"), "platform": m.group("plat")}

    ver = m.group("ver")
    cuda_m = re.search(r"cu(\d{2,3})", ver)
    torch_m = re.search(r"torch(\d{2,3})", ver) or re.search(r"pt(\d{2,3})", ver)
    if cuda_m:
        c = cuda_m.group(1)
        info["cuda"] = f"{c[:-1]}.{c[-1]}" if len(c) <= 3 else c
    if torch_m:
        t = torch_m.group(1)
        info["torch"] = f"{t[0]}.{t[1:]}" if len(t) <= 3 else t

    py = m.group("pytag").replace("cp", "")
    if len(py) >= 2:
        info["python"] = f"{py[0]}.{py[1:]}"

    plat = m.group("plat")
    info["os"] = "Linux" if "linux" in plat else "Windows" if "win" in plat else "macOS" if "macos" in plat or "darwin" in plat else plat

    return info


# ---------------------------------------------------------------------------
# Wheel content inspection
# ---------------------------------------------------------------------------

def inspect_wheel(whl_path: Path) -> dict:
    """Extract metadata from a wheel file without installing it."""
    info = {
        "size_mb": round(whl_path.stat().st_size / 1024 / 1024, 2),
        "shared_libs": [],
        "shared_libs_sizes": {},
        "tags": [],
        "requires_python": None,
        "requires_dist": [],
        "summary": None,
        "top_level_dirs": [],
        "file_count": 0,
    }

    try:
        with zipfile.ZipFile(whl_path) as zf:
            names = zf.namelist()
            info["file_count"] = len(names)

            # Shared libraries
            for n in names:
                if n.endswith((".so", ".pyd", ".dll")):
                    zi = zf.getinfo(n)
                    # Use just the basename for clustering
                    basename = n.split("/")[-1]
                    info["shared_libs"].append(basename)
                    info["shared_libs_sizes"][basename] = round(zi.file_size / 1024 / 1024, 2)

            # WHEEL metadata (tags)
            for n in names:
                if n.endswith("/WHEEL"):
                    content = zf.read(n).decode(errors="replace")
                    info["tags"] = re.findall(r"Tag: (.+)", content)
                    break

            # METADATA
            for n in names:
                if n.endswith("/METADATA"):
                    content = zf.read(n).decode(errors="replace")
                    m = re.search(r"^Requires-Python: (.+)$", content, re.MULTILINE)
                    if m:
                        info["requires_python"] = m.group(1).strip()
                    info["requires_dist"] = re.findall(r"^Requires-Dist: (.+)$", content, re.MULTILINE)
                    m = re.search(r"^Summary: (.+)$", content, re.MULTILINE)
                    if m:
                        info["summary"] = m.group(1).strip()
                    break

            # Top-level dirs
            info["top_level_dirs"] = sorted(set(n.split("/")[0] for n in names if "/" in n))

    except Exception as e:
        info["error"] = str(e)

    return info


def download_wheel(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download a wheel file. Returns True on success."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cuda-wheels-inspector/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    FAILED to download after {max_retries} attempts: {e}", file=sys.stderr)
                return False
    return False


# ---------------------------------------------------------------------------
# Collect wheel URLs from all sources
# ---------------------------------------------------------------------------

def collect_built_wheels() -> list:
    """Collect wheel URLs from GitHub releases on this repo."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "PozzettiAndrea/cuda-wheels")

    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read().decode())

    wheels = []
    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"].endswith(".whl"):
                wheels.append({
                    "filename": asset["name"],
                    "url": asset["browser_download_url"],
                    "source_type": "built",
                    "release": release["tag_name"],
                })
    return wheels


def collect_external_wheels() -> list:
    """Collect wheel URLs from external_wheels/ HTML index files."""
    external_dir = Path("external_wheels")
    if not external_dir.is_dir():
        return []

    link_pattern = re.compile(r'href="([^"]+)"[^>]*>([^<]+\.whl)</a>', re.IGNORECASE)
    wheels = []

    for pkg_dir in sorted(external_dir.iterdir()):
        if not pkg_dir.is_dir():
            continue
        index_file = pkg_dir / "index.html"
        if not index_file.exists():
            continue

        html = index_file.read_text()
        for match in link_pattern.finditer(html):
            url, display = match.group(1), match.group(2)
            wheels.append({
                "filename": display,
                "url": url,
                "source_type": "external",
                "release": pkg_dir.name,
            })

    return wheels


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_key(file_info: dict, wheel_info: dict) -> str:
    """Cluster key: package + version_base + sorted shared libs + sorted deps.

    Wheels with the same key have identical internal structure,
    differing only by CUDA/torch/python/platform.
    """
    pkg = file_info.get("package", "unknown")
    version_base = file_info.get("version", "?").split("+")[0]
    libs = tuple(sorted(wheel_info.get("shared_libs", [])))
    deps = tuple(sorted(wheel_info.get("requires_dist", [])))
    return f"{pkg}|{version_base}|{hashlib.md5(str(libs).encode()).hexdigest()[:8]}|{hashlib.md5(str(deps).encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Collecting wheel URLs...")
    built = collect_built_wheels()
    external = collect_external_wheels()
    all_wheels = built + external
    print(f"  Built: {len(built)} wheels")
    print(f"  External: {len(external)} wheels")
    print(f"  Total: {len(all_wheels)} wheels")

    # Group by package for progress reporting
    by_package = {}
    for w in all_wheels:
        fi = parse_wheel_filename(w["filename"])
        pkg = fi.get("package", w["release"])
        by_package.setdefault(pkg, []).append(w)

    print(f"\n{len(by_package)} packages to inspect")
    print("=" * 60)

    clusters = {}  # cluster_key -> cluster data
    all_results = []
    failed = []
    tmpdir = Path(tempfile.mkdtemp())

    for pkg_idx, (pkg_name, pkg_wheels) in enumerate(sorted(by_package.items()), 1):
        print(f"\n[{pkg_idx}/{len(by_package)}] {pkg_name} ({len(pkg_wheels)} wheels)")

        # For large packages, sample a few wheels per cluster to avoid downloading hundreds
        # of identical-structure wheels. Inspect first 3 per (version_base, os) group,
        # then just record filename metadata for the rest.
        sampled = {}
        for w in pkg_wheels:
            fi = parse_wheel_filename(w["filename"])
            ver_base = fi.get("version", "?").split("+")[0]
            plat = fi.get("os", "?")
            group = f"{ver_base}|{plat}"
            sampled.setdefault(group, []).append(w)

        to_download = []
        to_skip = []
        for group, group_wheels in sampled.items():
            to_download.extend(group_wheels[:3])
            to_skip.extend(group_wheels[3:])

        # Inspect sampled wheels (download + full inspection)
        cluster_for_group = {}  # group -> cluster_key
        for i, w in enumerate(to_download):
            filename = w["filename"]
            fi = parse_wheel_filename(filename)
            dest = tmpdir / filename
            print(f"  [{i+1}/{len(to_download)}] {filename[:80]}...", end=" ", flush=True)

            ok = download_wheel(w["url"], dest)
            if not ok:
                failed.append({"filename": filename, "url": w["url"], "error": "download_failed"})
                print("FAILED")
                continue

            wi = inspect_wheel(dest)
            dest.unlink(missing_ok=True)  # Free disk immediately

            ck = cluster_key(fi, wi)
            ver_base = fi.get("version", "?").split("+")[0]
            plat = fi.get("os", "?")
            group = f"{ver_base}|{plat}"
            cluster_for_group[group] = ck

            if ck not in clusters:
                clusters[ck] = {
                    "package": fi.get("package", pkg_name),
                    "version": ver_base,
                    "shared_libs": wi["shared_libs"],
                    "shared_libs_sizes": wi["shared_libs_sizes"],
                    "requires_python": wi["requires_python"],
                    "requires_dist": wi["requires_dist"],
                    "summary": wi["summary"],
                    "top_level_dirs": wi["top_level_dirs"],
                    "file_count": wi["file_count"],
                    "tags_sample": wi["tags"][:5],
                    "variants": [],
                }

            clusters[ck]["variants"].append({
                "filename": filename,
                "cuda": fi.get("cuda"),
                "torch": fi.get("torch"),
                "python": fi.get("python"),
                "os": fi.get("os"),
                "size_mb": wi["size_mb"],
                "source_type": w["source_type"],
            })

            all_results.append({"filename": filename, "cluster": ck, "inspected": True})
            print(f"{wi['size_mb']}MB, {len(wi['shared_libs'])} libs")

        # Record skipped wheels (just filename metadata, assigned to same cluster)
        for w in to_skip:
            filename = w["filename"]
            fi = parse_wheel_filename(filename)
            ver_base = fi.get("version", "?").split("+")[0]
            plat = fi.get("os", "?")
            group = f"{ver_base}|{plat}"
            ck = cluster_for_group.get(group)

            if ck and ck in clusters:
                clusters[ck]["variants"].append({
                    "filename": filename,
                    "cuda": fi.get("cuda"),
                    "torch": fi.get("torch"),
                    "python": fi.get("python"),
                    "os": fi.get("os"),
                    "size_mb": None,
                    "source_type": w["source_type"],
                })
            all_results.append({"filename": filename, "cluster": ck, "inspected": False})

    # Cleanup
    try:
        tmpdir.rmdir()
    except OSError:
        pass

    # ---------------------------------------------------------------------------
    # Write JSON report
    # ---------------------------------------------------------------------------
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_wheels": len(all_wheels),
        "inspected": sum(1 for r in all_results if r.get("inspected")),
        "failed": len(failed),
        "clusters": len(clusters),
        "packages": {},
    }

    for ck, cdata in sorted(clusters.items()):
        pkg = cdata["package"]
        report["packages"].setdefault(pkg, {"clusters": []})
        cuda_versions = sorted(set(v["cuda"] for v in cdata["variants"] if v.get("cuda")))
        torch_versions = sorted(set(v["torch"] for v in cdata["variants"] if v.get("torch")))
        python_versions = sorted(set(v["python"] for v in cdata["variants"] if v.get("python")))
        platforms = sorted(set(v["os"] for v in cdata["variants"] if v.get("os")))

        report["packages"][pkg]["clusters"].append({
            "version": cdata["version"],
            "cuda_versions": cuda_versions,
            "torch_versions": torch_versions,
            "python_versions": python_versions,
            "platforms": platforms,
            "wheel_count": len(cdata["variants"]),
            "shared_libs": cdata["shared_libs"],
            "shared_libs_sizes_mb": cdata["shared_libs_sizes"],
            "requires_python": cdata["requires_python"],
            "requires_dist": cdata["requires_dist"],
            "summary": cdata["summary"],
            "top_level_dirs": cdata["top_level_dirs"],
            "file_count": cdata["file_count"],
        })

    if failed:
        report["failed_wheels"] = failed

    Path("inspection_report.json").write_text(json.dumps(report, indent=2))

    # ---------------------------------------------------------------------------
    # Write Markdown report
    # ---------------------------------------------------------------------------
    md = []
    md.append("# Wheel Inspection Report\n")
    md.append(f"Generated: {report['generated_at']}\n")
    md.append(f"- **{report['total_wheels']}** total wheels")
    md.append(f"- **{report['inspected']}** fully inspected")
    md.append(f"- **{report['clusters']}** clusters")
    if failed:
        md.append(f"- **{len(failed)}** failed downloads")
    md.append("")

    for pkg_name in sorted(report["packages"].keys()):
        pkg_data = report["packages"][pkg_name]
        md.append(f"## {pkg_name}\n")

        for cluster in pkg_data["clusters"]:
            md.append(f"### v{cluster['version']} ({cluster['wheel_count']} wheels)\n")
            md.append(f"| Property | Value |")
            md.append(f"|----------|-------|")
            md.append(f"| CUDA | {', '.join(cluster['cuda_versions'])} |")
            md.append(f"| Torch | {', '.join(cluster['torch_versions'])} |")
            md.append(f"| Python | {', '.join(cluster['python_versions'])} |")
            md.append(f"| Platforms | {', '.join(cluster['platforms'])} |")
            md.append(f"| Requires-Python | {cluster['requires_python'] or 'unspecified'} |")
            md.append(f"| Files in wheel | {cluster['file_count']} |")
            md.append("")

            if cluster["shared_libs"]:
                md.append("**Shared libraries:**\n")
                for lib in cluster["shared_libs"]:
                    size = cluster["shared_libs_sizes_mb"].get(lib, "?")
                    md.append(f"- `{lib}` ({size} MB)")
                md.append("")

            if cluster["requires_dist"]:
                md.append("**Dependencies:**\n")
                for dep in cluster["requires_dist"]:
                    md.append(f"- `{dep}`")
                md.append("")

    Path("inspection_report.md").write_text("\n".join(md))

    # Print summary
    print(f"\n{'='*60}")
    print(f"DONE: {report['total_wheels']} wheels, {report['inspected']} inspected, {report['clusters']} clusters")
    if failed:
        print(f"FAILED: {len(failed)} downloads")
    print(f"Reports: inspection_report.json, inspection_report.md")


if __name__ == "__main__":
    main()
