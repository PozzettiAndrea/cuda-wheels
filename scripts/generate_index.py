#!/usr/bin/env python3
"""Generate PEP 503 compliant package index from GitHub releases."""
import os
import json
import urllib.request
from pathlib import Path


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

    # Collect all wheels
    packages = {}
    for release in releases:
        for asset in release.get("assets", []):
            name = asset["name"]
            if not name.endswith(".whl"):
                continue

            # Extract package name (first part before -)
            # Handle names like "sageattention-2.2.0+cu128torch28-cp312-..."
            pkg_name = name.split("-")[0].lower().replace("_", "-")

            packages.setdefault(pkg_name, []).append({
                "filename": name,
                "url": asset["browser_download_url"],
            })

    # Create docs directory
    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    # Generate root index
    with open(docs / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head><title>CUDA Wheels Index</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>CUDA Wheels</h1>\n")
        for pkg in sorted(packages.keys()):
            f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
        f.write("</body>\n</html>\n")

    # Generate per-package index
    for pkg, wheels in packages.items():
        pkg_dir = docs / pkg
        pkg_dir.mkdir(exist_ok=True)

        with open(pkg_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>{pkg}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>{pkg}</h1>\n")
            for wheel in sorted(wheels, key=lambda w: w["filename"]):
                # PEP 503 requires href to be the download URL
                f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
            f.write("</body>\n</html>\n")

    print(f"Generated index for {len(packages)} packages:")
    for pkg, wheels in packages.items():
        print(f"  - {pkg}: {len(wheels)} wheels")


if __name__ == "__main__":
    main()
