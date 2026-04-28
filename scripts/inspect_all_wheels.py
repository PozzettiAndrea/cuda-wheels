#!/usr/bin/env python3
"""Inspect all wheels in the built GitHub releases.

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
    all_wheels = collect_built_wheels()
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

    # ---------------------------------------------------------------------------
    # Write HTML report (for gh-pages at /inspection/)
    # ---------------------------------------------------------------------------
    _generate_html_report(report, Path("inspection_page"))

    # Print summary
    print(f"\n{'='*60}")
    print(f"DONE: {report['total_wheels']} wheels, {report['inspected']} inspected, {report['clusters']} clusters")
    if failed:
        print(f"FAILED: {len(failed)} downloads")
    print(f"Reports: inspection_report.json, inspection_report.md, inspection_page/index.html")


def _generate_html_report(report: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    def badge(text, cls="badge"):
        return f'<span class="{cls}">{text}</span>'

    def badges(items, cls="badge"):
        return " ".join(badge(v, cls) for v in items)

    pkg_sections = []
    for pkg_name in sorted(report["packages"].keys()):
        pkg_data = report["packages"][pkg_name]
        cluster_cards = []

        for c in pkg_data["clusters"]:
            libs_html = ""
            if c["shared_libs"]:
                lib_rows = "".join(
                    f'<tr><td><code>{lib}</code></td><td>{c["shared_libs_sizes_mb"].get(lib, "?")} MB</td></tr>'
                    for lib in c["shared_libs"]
                )
                libs_html = f"""<details><summary>{len(c["shared_libs"])} shared libraries</summary>
<table class="inner"><tr><th>File</th><th>Size</th></tr>{lib_rows}</table></details>"""

            deps_html = ""
            if c["requires_dist"]:
                dep_list = "".join(f"<li><code>{d}</code></li>" for d in c["requires_dist"])
                deps_html = f'<details><summary>{len(c["requires_dist"])} dependencies</summary><ul>{dep_list}</ul></details>'

            cluster_cards.append(f"""<div class="cluster">
<div class="cluster-header">
  <span class="version">v{c["version"]}</span>
  <span class="wheel-count">{c["wheel_count"]} wheels</span>
  <span class="file-count">{c["file_count"]} files</span>
</div>
<div class="matrix">
  <div class="matrix-row"><span class="label">CUDA</span> {badges(c["cuda_versions"], "badge cuda")}</div>
  <div class="matrix-row"><span class="label">Torch</span> {badges(c["torch_versions"], "badge torch")}</div>
  <div class="matrix-row"><span class="label">Python</span> {badges(c["python_versions"], "badge py")}</div>
  <div class="matrix-row"><span class="label">Platform</span> {badges(c["platforms"], "badge plat")}</div>
  <div class="matrix-row"><span class="label">Requires-Python</span> <code>{c["requires_python"] or "unspecified"}</code></div>
</div>
{libs_html}
{deps_html}
</div>""")

        total_wheels = sum(c["wheel_count"] for c in pkg_data["clusters"])
        pkg_sections.append(f"""<div class="package" id="{pkg_name}">
<h2>{pkg_name} <span class="pkg-count">{total_wheels} wheels, {len(pkg_data["clusters"])} cluster{"s" if len(pkg_data["clusters"]) != 1 else ""}</span></h2>
{"".join(cluster_cards)}
</div>""")

    # Table of contents
    toc_items = []
    for pkg_name in sorted(report["packages"].keys()):
        pkg_data = report["packages"][pkg_name]
        total = sum(c["wheel_count"] for c in pkg_data["clusters"])
        toc_items.append(f'<a href="#{pkg_name}" class="toc-item">{pkg_name} <span class="toc-count">{total}</span></a>')

    failed_html = ""
    if report.get("failed_wheels"):
        failed_rows = "".join(
            f'<tr><td><code>{f["filename"]}</code></td><td>{f.get("error", "")}</td></tr>'
            for f in report["failed_wheels"]
        )
        failed_html = f"""<details class="failed"><summary>{len(report["failed_wheels"])} failed downloads</summary>
<table><tr><th>Filename</th><th>Error</th></tr>{failed_rows}</table></details>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cuda-wheels inspection</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 2rem; max-width: 1200px; margin: 0 auto; }}
  h1 {{ color: #f0f6fc; margin-bottom: 0.3rem; }}
  h2 {{ color: #f0f6fc; margin-bottom: 1rem; font-size: 1.3rem; border-bottom: 1px solid #21262d; padding-bottom: 0.5rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 1.5rem; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 0.8rem 1.2rem; }}
  .stat-value {{ font-size: 1.8rem; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ color: #8b949e; font-size: 0.8rem; }}
  .toc {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem;
          margin-bottom: 2rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .toc-item {{ color: #58a6ff; text-decoration: none; padding: 0.3rem 0.7rem; border-radius: 4px;
               background: #21262d; font-size: 0.85rem; }}
  .toc-item:hover {{ background: #30363d; }}
  .toc-count {{ color: #8b949e; font-size: 0.75rem; }}
  .package {{ margin-bottom: 2.5rem; }}
  .pkg-count {{ color: #8b949e; font-size: 0.85rem; font-weight: normal; }}
  .cluster {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
              padding: 1rem; margin-bottom: 0.75rem; }}
  .cluster-header {{ display: flex; gap: 1rem; align-items: center; margin-bottom: 0.75rem; }}
  .version {{ font-weight: bold; color: #f0f6fc; font-size: 1.1rem; }}
  .wheel-count, .file-count {{ color: #8b949e; font-size: 0.85rem; }}
  .matrix {{ display: grid; gap: 0.3rem; margin-bottom: 0.5rem; }}
  .matrix-row {{ display: flex; align-items: center; gap: 0.5rem; }}
  .label {{ color: #8b949e; font-size: 0.8rem; min-width: 100px; }}
  .badge {{ display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
            font-size: 0.75rem; font-weight: 500; margin: 1px; }}
  .badge.cuda {{ background: #23882533; color: #3fb950; }}
  .badge.torch {{ background: #f0883e33; color: #f0883e; }}
  .badge.py {{ background: #8957e533; color: #bc8cff; }}
  .badge.plat {{ background: #38849633; color: #58a6ff; }}
  details {{ margin-top: 0.5rem; }}
  summary {{ cursor: pointer; color: #58a6ff; font-size: 0.85rem; }}
  summary:hover {{ color: #79c0ff; }}
  table.inner {{ margin-top: 0.5rem; border-collapse: collapse; width: 100%; }}
  table.inner th, table.inner td {{ text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #21262d; font-size: 0.8rem; }}
  table.inner th {{ color: #8b949e; }}
  ul {{ margin: 0.5rem 0 0 1.5rem; font-size: 0.8rem; }}
  li {{ margin-bottom: 0.2rem; }}
  code {{ font-size: 0.8rem; }}
  .failed {{ margin-bottom: 2rem; }}
  .failed table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
  .failed th, .failed td {{ text-align: left; padding: 0.3rem 0.6rem; border-bottom: 1px solid #21262d; font-size: 0.8rem; }}
  footer {{ margin-top: 2rem; color: #484f58; font-size: 0.8rem; }}
  a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<h1>cuda-wheels inspection</h1>
<p class="subtitle">Generated {report["generated_at"]}</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{report["total_wheels"]}</div><div class="stat-label">total wheels</div></div>
  <div class="stat"><div class="stat-value">{report["inspected"]}</div><div class="stat-label">inspected</div></div>
  <div class="stat"><div class="stat-value">{report["clusters"]}</div><div class="stat-label">clusters</div></div>
  <div class="stat"><div class="stat-value">{len(report["packages"])}</div><div class="stat-label">packages</div></div>
</div>

<div class="toc">{"".join(toc_items)}</div>

{failed_html}

{"".join(pkg_sections)}

<footer>
  <p><a href="https://github.com/PozzettiAndrea/cuda-wheels">GitHub</a> · <a href="../">Package Index</a> · <a href="../dashboard/">Dashboard</a></p>
</footer>
</body>
</html>"""

    (out_dir / "index.html").write_text(html)
    print(f"HTML report: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
