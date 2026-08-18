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
        # cuXYZ -> (XY, Z); cuXY -> (X, Y). Last digit is always the minor.
        n = key[2:]
        return (int(n[:-1]), int(n[-1]))

    keys = sorted((k for k in found if ver(k) >= MIN_CUDA), key=ver)
    return keys

# Only include these python versions
MIN_PYTHON = (3, 10)

# Match the link text (not href which is URL-encoded)
# Format: >torch-2.4.0+cu124-cp310-cp310-linux_x86_64.whl</a>
# The +cuXXX local version is OPTIONAL. Upstream is not consistent about it:
# cu126 tags its aarch64 wheels +cu126, but the cu124 index publishes
# torch-2.5.1-cp310-cp310-linux_aarch64.whl with no local version at all -- 18
# such wheels, absent from /whl/cpu/, HEAD 200 under /whl/cu124/. Requiring the
# tag dropped every one of them and made the page claim PyTorch never shipped
# CUDA 12.4 for ARM. Which index a wheel belongs to is known from the URL being
# scraped; the filename does not have to repeat it.
WHEEL_RE = re.compile(
    r">torch-(?P<torch>[\d.]+)(?P<local>\+cu\d+)?"
    r"-(?P<pytag>cp\d+)-cp\d+(?P<ft>t)?"
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
        # Same rule as discover_cuda_indexes: an empty result must never be the
        # answer to "the fetch failed". Swallowing this made one 403 delete a
        # whole section from the page, and six of them publish an empty matrix --
        # which update-index.yml then deploys with force_orphan, erasing the
        # good page with no history.
        raise RuntimeError(f"failed to fetch {url}: {type(e).__name__}: {e}") from e

    # An untagged wheel counts only if THIS index also ships that torch version
    # WITH a +cuXXX tag. Every CUDA index mirrors a pile of ancient wheels that
    # carry no local version and are not CUDA builds at all -- cu124 alone
    # mirrors torch 0.1.x macOS wheels and 1.11-2.0 manylinux2014_aarch64. The
    # genuine untagged wheels (cu124's linux_aarch64 builds of 2.4.x/2.5.x) are
    # exactly those whose version also appears tagged in the same index, because
    # they are that release's build for a platform upstream forgot to tag.
    tagged_versions = set(re.findall(r">torch-([\d.]+)\+cu\d+-", html))

    wheels = []
    for m in WHEEL_RE.finditer(html):
        torch_v = m.group("torch")
        if not m.group("local") and torch_v not in tagged_versions:
            continue
        pytag = m.group("pytag")
        # Free-threaded (cp3XXt) wheels are captured in group "ft" and NOT
        # filtered: they dedup away against their standard sibling, and upstream
        # ships no ft-only target. If that ever changes this must become a real
        # filter, because a check mark would then stand for a wheel no standard
        # interpreter can install. See CW-ADR-0010.
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
            "torch": torch_v,
            "python": f"{py_major}.{py_minor}",
            "platform": platform,
        })

    if not wheels:
        raise RuntimeError(
            f"{url} returned {len(html)} bytes but no wheels matched; upstream "
            "markup probably changed. Refusing to publish a partial matrix.")

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
    """A reference for what PyTorch publishes, per CUDA index."""
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

    # (14) Uniform geometry: every table gets the same python columns, so a
    # missing column can never be mistaken for "I didn't scroll far enough".
    all_pys = sorted({w["python"] for w in combos},
                     key=lambda p: tuple(int(x) for x in p.split(".")))

    blocks = []
    for cuda in sorted(by_cuda, key=cuda_key, reverse=True):
        ws = by_cuda[cuda]
        torches = sorted({w["torch"] for w in ws}, key=vt, reverse=True)

        head = "".join(f"<th>py{p}</th>" for p in all_pys)
        rows = []
        for t in torches:
            cells = []
            for p in all_pys:
                # (5) Platforms live IN the cell. A per-row union was wrong on
                # 5 of 40 rows -- cu124/2.5.x has no Windows for py3.13, and
                # 2.13.0 has no Windows for py3.15 on cu126/cu130/cu132.
                plats = {w["platform"] for w in ws
                         if w["torch"] == t and w["python"] == p}
                if not plats:
                    cells.append('<td class="no">&mdash;</td>')
                else:
                    tags = "".join(
                        f'<span class="p {LABEL[x].lower()}">{LABEL[x]}</span>'
                        for x in ORDER if x in plats)
                    cells.append(f'<td class="has">{tags}</td>')
            rows.append(f'<tr><td class="tv">{t}</td>{"".join(cells)}</tr>')

        n = len(ws)
        idx_url = f"https://download.pytorch.org/whl/{cuda}/torch/"
        blocks.append(
            f'<section id="{cuda}"><h2><a class="anchor" href="#{cuda}">{cuda}</a> '
            f'<span class="count">{n} combinations &middot; '
            f'<a href="{idx_url}">upstream index</a></span></h2>'
            f'<table><thead><tr><th>PyTorch</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyTorch CUDA Wheel Matrix</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #0d1117; color: #e6edf3; }}
  h1 {{ color: #58a6ff; margin-bottom: .25rem; }}
  h2 {{ color: #e6edf3; font-size: 1rem; margin: 1.75rem 0 .4rem; }}
  h2 .anchor {{ color: inherit; }}
  h2 .anchor:hover::after {{ content: " #"; color: #58a6ff; }}
  .count {{ color: #8b949e; font-size: .75rem; font-weight: 400; }}
  nav, .sub {{ color: #8b949e; font-size: .85rem; }}
  a {{ color: #58a6ff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; font-size: .8rem; }}
  th, td {{ border: 1px solid #30363d; padding: .3rem .55rem; text-align: center; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 500; }}
  td.tv {{ text-align: left; color: #c9d1d9; font-weight: 600; }}
  td.no {{ color: #8b949e; }}
  td.has {{ background: #10281a; }}
  .p {{ font-size: .68rem; font-weight: 700; margin: 0 1px; }}
  .p.l {{ color: #4ade80; }} .p.w {{ color: #818cf8; }} .p.a {{ color: #f97316; }}
  .stats {{ margin: 1rem 0; padding: .9rem 1.1rem; background: #161b22;
           border: 1px solid #30363d; border-radius: 6px; }}
  .key span {{ margin-right: 1.1rem; }}
  .scope {{ margin-top: .6rem; color: #8b949e; font-size: .78rem; }}
</style></head><body>
<nav><a href="../">&larr; Package Index</a> &middot; <a href="../dashboard/">Dashboard</a>
 &middot; <a href="https://pozzettiandrea.github.io/comfy-forge-docs/cuda-wheels/">Why this is hard</a></nav>
<h1>PyTorch CUDA Wheel Matrix</h1>
<p class="sub">Every CUDA wheel PyTorch publishes for CUDA {MIN_CUDA[0]}.{MIN_CUDA[1]} and
newer, Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+. CUDA indexes are discovered from
<a href="{CUDA_ROOT}">download.pytorch.org/whl</a>, so a new one appears here the next
time this page is generated (on push to <code>main</code>).</p>
<div class="stats">
  <strong>{matrix['total']}</strong> combinations across
  <strong>{len(matrix['cuda_versions'])}</strong> CUDA indexes
  <div class="key" style="margin-top:.5rem">
    <span><span class="p l">L</span> linux x86_64</span>
    <span><span class="p a">A</span> linux aarch64</span>
    <span><span class="p w">W</span> windows</span>
  </div>
  <div class="scope"><strong>Not shown:</strong>
    <a href="https://download.pytorch.org/whl/cu121/torch/">cu121</a> and older
    (this farm does not build them, though the
    <a href="../">package index</a> still serves some cu121 wheels),
    <a href="https://download.pytorch.org/whl/cpu/torch/">CPU</a>, ROCm and XPU
    indexes, <a href="https://download.pytorch.org/whl/nightly/">nightlies</a>,
    and Python below {MIN_PYTHON[0]}.{MIN_PYTHON[1]}.
    Free-threaded builds (<code>cp3XXt</code>) are a separate ABI and are shown
    under their standard Python version. This farm does not build against them
    &mdash; see
    <a href="https://pozzettiandrea.github.io/comfy-forge-docs/cuda-wheels/adr/0010-no-free-threaded-builds/">CW-ADR-0010</a>.
    Machine-readable: <a href="matrix.json">matrix.json</a>.
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
