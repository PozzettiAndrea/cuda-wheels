#!/usr/bin/env python3
"""Generate a dashboard page showing all available wheels and their metadata."""
import json
import os
import re
import urllib.request
from pathlib import Path


def get_releases(repo: str, token: str = None) -> list:
    url = f"https://api.github.com/repos/{repo}/releases"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


def parse_wheel_filename(filename: str) -> dict:
    """Extract metadata from wheel filename.

    Formats:
      pkg-1.0.0+cu128torch28-cp312-cp312-linux_x86_64.whl
      pkg-1.0.0+pt28cu128-cp312-cp312-linux_x86_64.whl
    """
    m = re.match(
        r"^(?P<pkg>[^-]+)-(?P<ver>[^-]+)-(?P<pytag>cp\d+)-[^-]+-(?P<plat>.+)\.whl$",
        filename,
    )
    if not m:
        return {}

    info = {
        "package": m.group("pkg"),
        "version": m.group("ver"),
        "python": m.group("pytag"),
        "platform": m.group("plat"),
    }

    ver = m.group("ver")
    # Extract CUDA and torch from local version
    cuda_m = re.search(r"cu(\d{2,3})", ver)
    torch_m = re.search(r"torch(\d{2,3})", ver) or re.search(r"pt(\d{2,3})", ver)

    if cuda_m:
        c = cuda_m.group(1)
        info["cuda"] = f"{c[:-1]}.{c[-1]}" if len(c) <= 3 else c
    if torch_m:
        t = torch_m.group(1)
        info["torch"] = f"{t[0]}.{t[1:]}" if len(t) <= 3 else t

    # Python version
    py = m.group("pytag")  # e.g. cp312
    digits = py.replace("cp", "")
    if len(digits) >= 2:
        info["python_version"] = f"{digits[0]}.{digits[1:]}"

    # Platform
    plat = m.group("plat")
    if "linux" in plat:
        info["os"] = "Linux"
    elif "win" in plat:
        info["os"] = "Windows"
    elif "macos" in plat or "darwin" in plat:
        info["os"] = "macOS"
    else:
        info["os"] = plat

    return info


def parse_external_wheels(external_dir: Path) -> dict:
    """Parse external_wheels/ HTML index files for wheel info."""
    packages = {}
    if not external_dir.is_dir():
        return packages

    link_pattern = re.compile(r'href="([^"]+)"[^>]*>([^<]+)</a>', re.IGNORECASE)

    for pkg_dir in sorted(external_dir.iterdir()):
        if not pkg_dir.is_dir():
            continue
        index_file = pkg_dir / "index.html"
        if not index_file.exists():
            continue

        html = index_file.read_text()
        wheels = []
        for match in link_pattern.finditer(html):
            url, display = match.group(1), match.group(2)
            if display.endswith(".whl"):
                info = parse_wheel_filename(display)
                if info:
                    info["url"] = url
                    info["source"] = "external"
                    wheels.append(info)
        if wheels:
            packages[pkg_dir.name] = wheels

    return packages


def generate_dashboard(built_packages: dict, external_packages: dict, output_dir: Path):
    """Generate dashboard HTML."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_packages = {}
    for name, wheels in built_packages.items():
        all_packages[name] = {"wheels": wheels, "source": "built"}
    for name, wheels in external_packages.items():
        all_packages[name] = {"wheels": wheels, "source": "external"}

    # Build summary data per package
    pkg_summaries = []
    for name in sorted(all_packages.keys()):
        data = all_packages[name]
        wheels = data["wheels"]
        cuda_versions = sorted(set(w.get("cuda", "?") for w in wheels))
        torch_versions = sorted(set(w.get("torch", "?") for w in wheels))
        python_versions = sorted(set(w.get("python_version", "?") for w in wheels))
        platforms = sorted(set(w.get("os", "?") for w in wheels))
        versions = sorted(set(w.get("version", "?").split("+")[0] for w in wheels))

        pkg_summaries.append({
            "name": name,
            "source": data["source"],
            "count": len(wheels),
            "versions": versions,
            "cuda": cuda_versions,
            "torch": torch_versions,
            "python": python_versions,
            "platforms": platforms,
            "wheels": wheels,
        })

    # Generate HTML
    html = _render_dashboard(pkg_summaries)
    (output_dir / "index.html").write_text(html)
    print(f"Dashboard: {len(pkg_summaries)} packages, {sum(p['count'] for p in pkg_summaries)} total wheels")


def _render_dashboard(pkg_summaries: list) -> str:
    total_wheels = sum(p["count"] for p in pkg_summaries)
    built = [p for p in pkg_summaries if p["source"] == "built"]
    external = [p for p in pkg_summaries if p["source"] == "external"]

    def badge(items, cls="badge"):
        return " ".join(f'<span class="{cls}">{v}</span>' for v in items if v != "?")

    def pkg_row(p):
        source_badge = '<span class="badge built">built</span>' if p["source"] == "built" else '<span class="badge external">external</span>'
        return f"""<tr>
  <td><strong>{p["name"]}</strong> {source_badge}</td>
  <td>{", ".join(p["versions"])}</td>
  <td>{badge(p["cuda"], "badge cuda")}</td>
  <td>{badge(p["torch"], "badge torch")}</td>
  <td>{badge(p["python"], "badge py")}</td>
  <td>{badge(p["platforms"], "badge plat")}</td>
  <td>{p["count"]}</td>
</tr>"""

    rows = "\n".join(pkg_row(p) for p in pkg_summaries)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cuda-wheels dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 2rem; }}
  h1 {{ color: #f0f6fc; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #8b949e; margin-bottom: 2rem; }}
  .stats {{ display: flex; gap: 2rem; margin-bottom: 2rem; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 1rem 1.5rem; }}
  .stat-value {{ font-size: 2rem; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ color: #8b949e; font-size: 0.85rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #161b22;
           border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
  th {{ background: #21262d; color: #f0f6fc; text-align: left; padding: 0.75rem 1rem;
       font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  td {{ padding: 0.6rem 1rem; border-top: 1px solid #21262d; font-size: 0.9rem; }}
  tr:hover td {{ background: #1c2128; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
            font-size: 0.75rem; font-weight: 500; margin: 1px; }}
  .badge.built {{ background: #1f6feb33; color: #58a6ff; }}
  .badge.external {{ background: #f0883e33; color: #f0883e; }}
  .badge.cuda {{ background: #23882533; color: #3fb950; }}
  .badge.torch {{ background: #f0883e33; color: #f0883e; }}
  .badge.py {{ background: #8957e533; color: #bc8cff; }}
  .badge.plat {{ background: #38849633; color: #58a6ff; }}
  footer {{ margin-top: 2rem; color: #484f58; font-size: 0.8rem; }}
  a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<h1>cuda-wheels</h1>
<p class="subtitle">Pre-built CUDA Python wheels for ML/3D packages</p>

<div class="stats">
  <div class="stat"><div class="stat-value">{len(pkg_summaries)}</div><div class="stat-label">packages</div></div>
  <div class="stat"><div class="stat-value">{total_wheels}</div><div class="stat-label">wheels</div></div>
  <div class="stat"><div class="stat-value">{len(built)}</div><div class="stat-label">built by CI</div></div>
  <div class="stat"><div class="stat-value">{len(external)}</div><div class="stat-label">external links</div></div>
</div>

<table>
<thead>
<tr><th>Package</th><th>Version</th><th>CUDA</th><th>Torch</th><th>Python</th><th>Platform</th><th>#</th></tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<footer>
  <p><a href="https://github.com/PozzettiAndrea/cuda-wheels">GitHub</a> · <a href="../">Package Index</a></p>
</footer>
</body>
</html>"""


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "PozzettiAndrea/cuda-wheels")

    print(f"Generating dashboard for {repo}")

    releases = get_releases(repo, token)

    # Collect built wheels from releases
    built_packages = {}
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
            pkg_name = name.split("-")[0].lower().replace("_", "-")
            built_packages.setdefault(pkg_name, []).append(info)

    # Collect external wheels
    external_packages = parse_external_wheels(Path("external_wheels"))

    # Generate dashboard at docs/dashboard/
    generate_dashboard(built_packages, external_packages, Path("docs") / "dashboard")


if __name__ == "__main__":
    main()
