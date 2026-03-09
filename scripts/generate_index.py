#!/usr/bin/env python3
"""Generate PEP 503 compliant package index from GitHub releases + external wheels."""
import os
import json
import re
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

# Matches v2 torch naming: +cu128torch2.9-cp (dot between major.minor)
_V2_TORCH_RE = re.compile(r'(\+cu\d+torch)(\d)\.(\d+)(-cp)')


def get_releases(repo: str, token: str = None) -> list:
    """Fetch all releases from a GitHub repository."""
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "PozzettiAndrea/cuda-wheels")

    print(f"Generating index for {repo}")

    # Fetch releases
    releases = get_releases(repo, token)

    # Collect all wheels from releases
    packages = {}
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue

            # Extract package name (first part before -)
            pkg_name = name.split("-")[0].lower().replace("_", "-")

            url = asset["browser_download_url"]

            # Generate v1 display name by stripping dot: torch2.9 → torch29
            v1_name = _V2_TORCH_RE.sub(
                lambda x: f"{x.group(1)}{x.group(2)}{x.group(3)}{x.group(4)}", name
            )

            packages.setdefault(pkg_name, []).append({
                "filename": name,      # v2 (actual asset name)
                "v1_filename": v1_name, # v1 (display name for root index)
                "url": url,
            })

    # Create docs directory
    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    # Copy external_wheels/ into docs/ (pre-built index.html files for external packages)
    external_dir = Path("external_wheels")
    external_packages = set()
    if external_dir.is_dir():
        for pkg_dir in sorted(external_dir.iterdir()):
            if pkg_dir.is_dir() and (pkg_dir / "index.html").exists():
                dest = docs / pkg_dir.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(pkg_dir, dest)
                external_packages.add(pkg_dir.name)
        print(f"Copied {len(external_packages)} external packages: {', '.join(sorted(external_packages))}")

    # Merge all package names for root index
    all_packages = sorted(set(packages.keys()) | external_packages)

    # Generate root index
    with open(docs / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head><title>CUDA Wheels Index</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>CUDA Wheels</h1>\n")
        for pkg in all_packages:
            f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
        f.write("</body>\n</html>\n")

    # Generate per-package index (only for built packages, externals already have index.html)
    # Root index: v1 display names (torch29), hrefs point to v2 assets (torch2.9)
    for pkg, wheels in packages.items():
        pkg_dir = docs / pkg
        pkg_dir.mkdir(exist_ok=True)

        with open(pkg_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>{pkg}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{pkg}</h1>\n")
            for wheel in sorted(wheels, key=lambda w: w["v1_filename"]):
                f.write(f'<a href="{wheel["url"]}">{wheel["v1_filename"]}</a><br>\n')
            f.write("</body>\n</html>\n")

    print(f"Generated index for {len(packages)} built packages:")
    for pkg, wheels in packages.items():
        print(f"  - {pkg}: {len(wheels)} wheels")
    if external_packages:
        print(f"External packages: {', '.join(sorted(external_packages))}")
    print(f"Total: {len(all_packages)} packages in index")

    # Generate v2 index (built packages only, all wheels are v2-named now)
    v2_packages = packages

    v2_docs = docs / "v2"
    v2_docs.mkdir(parents=True, exist_ok=True)
    with open(v2_docs / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head><title>CUDA Wheels v2</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>CUDA Wheels v2</h1>\n")
        for pkg in sorted(v2_packages.keys()):
            f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
        f.write("</body>\n</html>\n")

    for pkg, wheels in v2_packages.items():
        pkg_dir = v2_docs / pkg
        pkg_dir.mkdir(exist_ok=True)
        with open(pkg_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>{pkg}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{pkg}</h1>\n")
            for wheel in sorted(wheels, key=lambda w: w["filename"]):
                f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
            f.write("</body>\n</html>\n")

    print(f"Generated v2 index for {len(v2_packages)} packages")

    # Generate dashboard (separate from PEP 503 index)
    try:
        from generate_dashboard import generate_dashboard, parse_external_wheels, parse_wheel_filename, get_workflow_runs

        built_for_dashboard = {}
        release_urls = {}
        for release in releases:
            for asset in release.get("assets", []):
                name = asset["name"]
                if not name.endswith(".whl"):
                    continue
                info = parse_wheel_filename(name)
                if not info:
                    continue
                info["url"] = asset["browser_download_url"]
                info["source"] = "built"
                info["size"] = asset.get("size")
                info["display_name"] = name
                pkg_name = name.split("-")[0].lower().replace("_", "-")
                built_for_dashboard.setdefault(pkg_name, []).append(info)
                if pkg_name not in release_urls:
                    release_urls[pkg_name] = release.get("html_url")

        print("Fetching workflow runs...")
        workflow_runs = get_workflow_runs(repo, token)
        total_runs = sum(len(v) for v in workflow_runs.values())
        print(f"  {total_runs} runs across {len(workflow_runs)} packages")

        ext_for_dashboard = parse_external_wheels(external_dir)
        generate_dashboard(built_for_dashboard, ext_for_dashboard, docs / "dashboard",
                           release_urls=release_urls, workflow_runs=workflow_runs, repo=repo,
                           token=token)
    except Exception as e:
        print(f"Dashboard generation failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
