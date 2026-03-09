#!/usr/bin/env python3
"""Generate PEP 503 compliant package index from GitHub releases + external wheels."""
import os
import json
import re
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

# Matches v1 torch naming: +cu128torch29-cp (no dot between major/minor)
_V1_TORCH_RE = re.compile(r'(\+cu\d+torch)(\d)(\d+)(-cp)')


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

            # For v1-named wheels (torch29), rewrite URL to point to v2 copy (torch2.9)
            url = asset["browser_download_url"]
            m = _V1_TORCH_RE.search(name)
            if m:
                v2_name = _V1_TORCH_RE.sub(
                    lambda x: f"{x.group(1)}{x.group(2)}.{x.group(3)}{x.group(4)}", name
                )
                url = url.replace(quote(name, safe=""), quote(v2_name, safe=""))

            packages.setdefault(pkg_name, []).append({
                "filename": name,
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
    # Root index keeps v1 filenames but URLs point to v2-named assets
    for pkg, wheels in packages.items():
        # Only v1-named wheels for root index (URLs already rewritten to v2 above)
        v1_wheels = [w for w in wheels if _V1_TORCH_RE.search(w["filename"])]
        if not v1_wheels:
            v1_wheels = wheels  # no v1/v2 distinction for this package

        pkg_dir = docs / pkg
        pkg_dir.mkdir(exist_ok=True)

        with open(pkg_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>{pkg}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{pkg}</h1>\n")
            for wheel in sorted(v1_wheels, key=lambda w: w["filename"]):
                f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
            f.write("</body>\n</html>\n")

    print(f"Generated index for {len(packages)} built packages:")
    for pkg, wheels in packages.items():
        print(f"  - {pkg}: {len(wheels)} wheels")
    if external_packages:
        print(f"External packages: {', '.join(sorted(external_packages))}")
    print(f"Total: {len(all_packages)} packages in index")

    # Generate v2 index (built packages only, v2-named wheels only)
    v2_packages = {}
    for pkg, wheels in packages.items():
        v2_wheels = [w for w in wheels if not _V1_TORCH_RE.search(w["filename"])]
        if v2_wheels:
            v2_packages[pkg] = v2_wheels

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

    print(f"Generated v2 index for {len(v2_packages)} built packages")

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
