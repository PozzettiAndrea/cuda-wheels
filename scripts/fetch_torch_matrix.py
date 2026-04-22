#!/usr/bin/env python3
"""Fetch the full CUDA/PyTorch/Python build matrix from PyTorch's wheel index.

Scrapes https://download.pytorch.org/whl/{cuda}/torch/ for each CUDA version
and extracts all available (cuda, torch, python, platform) combinations.

Outputs a JSON file and optionally an HTML page for GitHub Pages.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

CUDA_VERSIONS = ["cu124", "cu126", "cu128", "cu130"]

# Only include these python versions
MIN_PYTHON = (3, 10)

# Match the link text (not href which is URL-encoded)
# Format: >torch-2.4.0+cu124-cp310-cp310-linux_x86_64.whl</a>
WHEEL_RE = re.compile(
    r">torch-(?P<torch>[\d.]+)\+(?P<cuda>cu\d+)"
    r"-(?P<pytag>cp\d+t?)-cp\d+t?"
    r"-(?P<platform>(?:manylinux[^.]+_(?:x86_64|aarch64)|linux_(?:x86_64|aarch64)|win_amd64))\.whl<"
)


def fetch_torch_wheels(cuda: str) -> list[dict]:
    """Fetch wheel list from PyTorch index for a given CUDA version."""
    url = f"https://download.pytorch.org/whl/{cuda}/torch/"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cuda-wheels/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode()
    except Exception as e:
        print(f"WARNING: Failed to fetch {url}: {e}", file=sys.stderr)
        return []

    wheels = []
    for m in WHEEL_RE.finditer(html):
        pytag = m.group("pytag")
        # Skip free-threaded variants (cp313t, cp314t)
        if "t" in pytag[2:]:
            continue
        # Parse python version
        py_major = int(pytag[2])
        py_minor = int(pytag[3:])
        if (py_major, py_minor) < MIN_PYTHON:
            continue

        plat = m.group("platform")
        if "aarch64" in plat:
            platform = "linux_aarch64"
        elif "linux" in plat or "manylinux" in plat:
            platform = "linux_x86_64"
        else:
            platform = "windows"
        wheels.append({
            "cuda": cuda,
            "torch": m.group("torch"),
            "python": f"{py_major}.{py_minor}",
            "platform": platform,
        })

    return wheels


def build_matrix() -> dict:
    """Build the full matrix from all CUDA versions."""
    all_wheels = []
    for cuda in CUDA_VERSIONS:
        print(f"Fetching {cuda}...", file=sys.stderr)
        wheels = fetch_torch_wheels(cuda)
        print(f"  {len(wheels)} combos", file=sys.stderr)
        all_wheels.extend(wheels)

    # Deduplicate
    seen = set()
    unique = []
    for w in all_wheels:
        key = (w["cuda"], w["torch"], w["python"], w["platform"])
        if key not in seen:
            seen.add(key)
            unique.append(w)

    # Sort: cuda, torch version (numeric), python, platform
    def sort_key(w):
        cuda_num = int(w["cuda"][2:])
        torch_parts = tuple(int(x) for x in w["torch"].split("."))
        py_parts = tuple(int(x) for x in w["python"].split("."))
        return (cuda_num, torch_parts, py_parts, w["platform"])

    unique.sort(key=sort_key)

    # Build summary
    summary = {}
    for w in unique:
        cuda = w["cuda"]
        torch_v = w["torch"]
        key = f"{cuda}/torch-{torch_v}"
        if key not in summary:
            summary[key] = {"cuda": cuda, "torch": torch_v, "python": [], "platforms": set()}
        if w["python"] not in summary[key]["python"]:
            summary[key]["python"].append(w["python"])
        summary[key]["platforms"].add(w["platform"])

    # Convert sets to sorted lists
    for v in summary.values():
        v["platforms"] = sorted(v["platforms"])

    return {
        "combos": unique,
        "summary": list(summary.values()),
        "cuda_versions": CUDA_VERSIONS,
        "total": len(unique),
    }


def generate_html(matrix: dict, output_path: Path):
    """Generate an HTML page showing the full build matrix."""
    summary = matrix["summary"]

    rows = []
    for entry in summary:
        cuda = entry["cuda"]
        torch_v = entry["torch"]
        pythons = ", ".join(entry["python"])
        platforms = ", ".join(entry["platforms"])
        rows.append(f"          <tr><td>{cuda}</td><td>{torch_v}</td><td>{pythons}</td><td>{platforms}</td></tr>")

    rows_html = "\n".join(rows)
    total = matrix["total"]
    num_combos = len(summary)

    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>CUDA Wheels – Full Build Matrix</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 2rem; background: #0d1117; color: #e6edf3; }}
    h1 {{ color: #58a6ff; }}
    p {{ color: #8b949e; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #30363d; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #161b22; color: #58a6ff; position: sticky; top: 0; }}
    tr:hover {{ background: #161b22; }}
    .stats {{ margin: 1rem 0; padding: 1rem; background: #161b22; border-radius: 6px; border: 1px solid #30363d; }}
    a {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    nav {{ margin-bottom: 1.5rem; }}
  </style>
</head>
<body>
  <nav><a href="../">&larr; Package Index</a> · <a href="../dashboard/">Dashboard</a></nav>
  <h1>Full Build Matrix</h1>
  <div class="stats">
    <strong>{total}</strong> total wheel targets across <strong>{num_combos}</strong> CUDA/PyTorch combinations
    (CUDA versions: {', '.join(matrix['cuda_versions'])})
  </div>
  <p>Derived from <a href="https://download.pytorch.org/whl/">download.pytorch.org/whl</a>. Filtered to cp310+, no free-threaded, linux + windows only.</p>
  <table>
    <thead>
      <tr><th>CUDA</th><th>PyTorch</th><th>Python</th><th>Platforms</th></tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "index.html").write_text(html)
    print(f"Wrote {output_path / 'index.html'}", file=sys.stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fetch PyTorch CUDA build matrix")
    parser.add_argument("--output", default="matrix.json", help="Output JSON file")
    parser.add_argument("--html", default=None, help="Output HTML directory (e.g. docs/matrix)")
    args = parser.parse_args()

    matrix = build_matrix()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"Wrote {len(matrix['combos'])} combos to {args.output}", file=sys.stderr)

    if args.html:
        generate_html(matrix, Path(args.html))


if __name__ == "__main__":
    main()
