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

# The set of CUDA indexes is DISCOVERED, never hardcoded. A literal list is a
# prediction about what NVIDIA will ship, and it silently omits whatever it
# guessed wrong -- this list used to read ["cu124","cu126","cu128","cu130"] and
# was blind to both cu129 and cu132, so the published matrix page could not show
# a fifth of the grid the farm actually builds.
CUDA_ROOT = "https://download.pytorch.org/whl/"
MIN_CUDA = (12, 4)   # older indexes exist back to cu75; the farm does not build them


def discover_cuda_indexes() -> list[str]:
    """Every cuXXX index upstream publishes, floored at MIN_CUDA, newest last."""
    req = urllib.request.Request(CUDA_ROOT, headers={"User-Agent": "cuda-wheels/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode()
    found = set(re.findall(r'href="(?:\./)?(cu\d+)/?"', html))
    if not found:
        # A parse that yields nothing is a failure to look, not an empty upstream.
        raise RuntimeError(f"no cuXXX indexes found at {CUDA_ROOT}; refusing to "
                           "publish a matrix that would silently show nothing")

    def ver(key):
        n = key[2:]
        return (int(n[:-1]), int(n[-1])) if len(n) == 3 else (int(n[:-1]), int(n[-1]))

    keys = sorted((k for k in found if ver(k) >= MIN_CUDA), key=ver)
    return keys

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
    """Build the full matrix from every discovered CUDA index."""
    cuda_versions = discover_cuda_indexes()
    print(f"Discovered {len(cuda_versions)} CUDA indexes: {', '.join(cuda_versions)}",
          file=sys.stderr)
    all_wheels = []
    for cuda in cuda_versions:
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
        "cuda_versions": cuda_versions,
        "total": len(unique),
    }


def generate_html(matrix: dict, output_path: Path):
    """A reference for what PyTorch actually publishes.

    One table per CUDA index: rows are torch versions, columns are python
    versions, and each cell names the platforms upstream ships for that
    combination. Nothing about this farm -- this is the upstream fact.
    """
    combos = matrix["combos"]

    by_cuda = {}
    for w in combos:
        by_cuda.setdefault(w["cuda"], []).append(w)

    def cuda_key(k):
        n = k[2:]
        return (int(n[:-1]), int(n[-1]))

    def vt(t):
        return tuple(int(x) for x in t.split("."))

    LABEL = {"linux_x86_64": "L", "linux_aarch64": "A", "windows": "W"}
    ORDER = ["linux_x86_64", "linux_aarch64", "windows"]

    blocks = []
    for cuda in sorted(by_cuda, key=cuda_key, reverse=True):
        ws = by_cuda[cuda]
        pys = sorted({w["python"] for w in ws},
                     key=lambda p: tuple(int(x) for x in p.split(".")))
        torches = sorted({w["torch"] for w in ws}, key=vt, reverse=True)

        head = "".join(f"<th>py{p}</th>" for p in pys)
        rows = []
        for t in torches:
            have = {w["python"] for w in ws if w["torch"] == t}
            cells = "".join(
                '<td class="yes">&#10003;</td>' if p in have else '<td class="no">&mdash;</td>'
                for p in pys
            )
            plats = {w["platform"] for w in ws if w["torch"] == t}
            tags = " ".join(
                f'<span class="p {LABEL[x].lower()}">{LABEL[x]}</span>'
                for x in ORDER if x in plats
            )
            rows.append(f'<tr><td class="tv">{t}</td>{cells}<td class="plats">{tags}</td></tr>')

        n = len(ws)
        blocks.append(
            f'<section><h2>{cuda} <span class="count">{n} wheels</span></h2>'
            f'<table><thead><tr><th>PyTorch</th>{head}<th>Platforms</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>PyTorch CUDA Wheel Matrix</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #0d1117; color: #e6edf3; }}
  h1 {{ color: #58a6ff; margin-bottom: .25rem; }}
  h2 {{ color: #e6edf3; font-size: 1rem; margin: 1.75rem 0 .4rem; }}
  .count {{ color: #6e7681; font-size: .75rem; font-weight: 400; }}
  nav, .sub {{ color: #8b949e; font-size: .85rem; }}
  a {{ color: #58a6ff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; font-size: .8rem; }}
  th, td {{ border: 1px solid #30363d; padding: .3rem .55rem; text-align: center; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 500; }}
  td.tv {{ text-align: left; color: #c9d1d9; font-weight: 600; }}
  td.yes {{ color: #4ade80; }}
  td.no {{ color: #30363d; }}
  td.plats {{ letter-spacing: .05em; }}
  .p {{ display: inline-block; font-size: .62rem; font-weight: 700;
       padding: 0 .22rem; margin: 0 1px; border-radius: 2px; }}
  .p.l {{ color: #4ade80; }} .p.w {{ color: #818cf8; }} .p.a {{ color: #f97316; }}
  .stats {{ margin: 1rem 0; padding: .9rem 1.1rem; background: #161b22;
           border: 1px solid #30363d; border-radius: 6px; }}
  .key span {{ margin-right: 1.1rem; }}
</style></head><body>
<nav><a href="../">&larr; Package Index</a> &middot; <a href="../dashboard/">Dashboard</a></nav>
<h1>PyTorch CUDA Wheel Matrix</h1>
<p class="sub">Every CUDA/PyTorch/Python combination PyTorch publishes. CUDA
indexes are discovered from
<a href="https://download.pytorch.org/whl/">download.pytorch.org/whl</a>, so a
new one appears here as soon as upstream ships it. Filtered to cp310+, no
free-threaded builds.</p>
<div class="stats">
  <strong>{matrix['total']}</strong> wheel targets across
  <strong>{len(matrix['cuda_versions'])}</strong> CUDA indexes
  ({', '.join(matrix['cuda_versions'])})
  <div class="key" style="margin-top:.5rem">
    <span><span class="p l">L</span> linux x86_64</span>
    <span><span class="p a">A</span> linux aarch64</span>
    <span><span class="p w">W</span> windows</span>
  </div>
</div>
{''.join(blocks)}
</body></html>
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
