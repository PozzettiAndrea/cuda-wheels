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


def load_declared() -> tuple[set, set, list]:
    """What packages/_defaults.yml declares: (cuda_key, torch, python, platform)."""
    import yaml
    d = yaml.safe_load((Path(__file__).parent.parent / "packages" / "_defaults.yml").read_text())
    # _defaults.yml says "linux"; the wheel index says "linux_x86_64". Normalise,
    # or every Linux row silently falls through to "not built by this farm".
    NAME = {"linux": "linux_x86_64", "windows": "windows"}
    plats = [NAME.get(x, x) for x in d.get("platforms", ["linux", "windows"])]
    declared = set()
    for c in d.get("combinations", []):
        cuda_key = "cu" + str(c["cuda"]).replace(".", "")
        for py in c.get("python_versions", []):
            for pl in plats:
                declared.add((cuda_key, str(c["pytorch"]), str(py), pl))
    return declared, set(plats), d.get("combinations", [])


def load_phantoms() -> set:
    """Cells upstream never published, so they can never be built."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from generate_matrix import PHANTOM_COMBOS
        NAME = {"linux": "linux_x86_64", "windows": "windows"}
        return {("cu" + c, ".".join(t.split(".")[:2]), f"3.{p[1:]}", NAME.get(pl, pl))
                for c, t, p, pl in PHANTOM_COMBOS}
    except Exception:
        return set()


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

    declared, farm_plats, _ = load_declared()
    phantoms = load_phantoms()

    n_built = n_todo = n_phantom = n_patch = 0
    declared_minors = {(c, ".".join(t.split(".")[:2])) for c, t, _, _ in declared}
    for w in unique:
        key = (w["cuda"], w["torch"], w["python"], w["platform"])
        minor = (w["cuda"], ".".join(w["torch"].split(".")[:2]), w["python"], w["platform"])
        if w["platform"] not in farm_plats:
            w["farm"] = "n/a"          # aarch64: upstream ships it, this farm does not build it
        elif key in declared:
            w["farm"] = "declared"; n_built += 1
        elif minor in phantoms:
            w["farm"] = "phantom"; n_phantom += 1
        elif (w["cuda"], ".".join(w["torch"].split(".")[:2])) in declared_minors:
            # The grid declares a different patch of this same (cuda, minor) --
            # it tracks .0 releases and skips .1/.2. Not a gap, a policy.
            w["farm"] = "patch"; n_patch += 1
        else:
            w["farm"] = "todo"; n_todo += 1

    for entry in summary.values():
        entry["todo"] = sorted({w["python"] for w in unique
                                if w["cuda"] == entry["cuda"] and w["torch"] == entry["torch"]
                                and w.get("farm") == "todo"})

    return {
        "combos": unique,
        "summary": list(summary.values()),
        "cuda_versions": cuda_versions,
        "total": len(unique),
        "coverage": {"declared": n_built, "todo": n_todo,
                     "phantom": n_phantom, "patch": n_patch},
    }


def generate_html(matrix: dict, output_path: Path):
    """One block per CUDA index: rows are torch versions, columns are pythons.

    A cell is a wheel upstream publishes. Its colour says whether this farm
    already declares it, still needs to build it, or can never build it because
    upstream never shipped the matching torch.
    """
    combos = matrix["combos"]
    cov = matrix.get("coverage", {})

    by_cuda = {}
    for w in combos:
        by_cuda.setdefault(w["cuda"], []).append(w)

    def cuda_key(k):
        n = k[2:]
        return (int(n[:-1]), int(n[-1]))

    def vt(t):
        return tuple(int(x) for x in t.split("."))

    blocks = []
    for cuda in sorted(by_cuda, key=cuda_key, reverse=True):
        ws = by_cuda[cuda]
        pys = sorted({w["python"] for w in ws}, key=lambda p: tuple(int(x) for x in p.split(".")))
        torches = sorted({w["torch"] for w in ws}, key=vt, reverse=True)

        head = "".join(f"<th>py{p}</th>" for p in pys)
        rows = []
        for t in torches:
            cells = []
            for p in pys:
                here = [w for w in ws if w["torch"] == t and w["python"] == p
                        and w["platform"] in ("linux_x86_64", "windows")]
                if not here:
                    cells.append('<td class="none">&mdash;</td>')
                    continue
                states = {w["farm"] for w in here}
                plats = "".join(sorted({w["platform"][0].upper() for w in here}))
                if states == {"patch"}:
                    cells.append(f'<td class="patch" title="patch release; the grid tracks .0">{plats}</td>')
                elif "todo" in states:
                    cells.append(f'<td class="todo" title="not in the grid yet">{plats}</td>')
                elif states == {"phantom"}:
                    cells.append(f'<td class="phantom" title="upstream never shipped this">{plats}</td>')
                else:
                    cells.append(f'<td class="ok" title="declared in _defaults.yml">{plats}</td>')
            rows.append(f"<tr><td class=\"tv\">{t}</td>{''.join(cells)}</tr>")

        todo_here = sum(1 for w in ws if w.get("farm") == "todo")  # patch/phantom excluded
        badge = f'<span class="badge todo-badge">{todo_here} to build</span>' if todo_here else \
                '<span class="badge ok-badge">complete</span>'
        blocks.append(
            f'<section><h2>{cuda} {badge}</h2>'
            f'<table><thead><tr><th>PyTorch</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></section>'
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CUDA Wheels - Full Build Matrix</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 2rem; background: #0d1117; color: #e6edf3; }}
  h1 {{ color: #58a6ff; margin-bottom: .25rem; }}
  h2 {{ color: #e6edf3; font-size: 1rem; margin: 1.75rem 0 .4rem; }}
  nav, .sub {{ color: #8b949e; font-size: .85rem; }}
  a {{ color: #58a6ff; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; font-size: .8rem; }}
  th, td {{ border: 1px solid #30363d; padding: .3rem .55rem; text-align: center; }}
  th {{ background: #161b22; color: #8b949e; font-weight: 500; }}
  td.tv {{ text-align: left; color: #c9d1d9; font-weight: 600; }}
  td.ok {{ background: #10281a; color: #4ade80; }}
  td.todo {{ background: #2a2410; color: #fbbf24; font-weight: 600; }}
  td.phantom {{ background: #1a1a1a; color: #6e7681; }}
  td.patch {{ background: #14202e; color: #7d95b8; }}
  td.none {{ color: #30363d; }}
  .stats {{ margin: 1rem 0; padding: .9rem 1.1rem; background: #161b22;
           border: 1px solid #30363d; border-radius: 6px; }}
  .badge {{ font-size: .7rem; padding: .1rem .45rem; border-radius: 3px;
           font-weight: 600; vertical-align: middle; margin-left: .5rem; }}
  .todo-badge {{ background: #2a2410; color: #fbbf24; }}
  .ok-badge {{ background: #10281a; color: #4ade80; }}
  .key span {{ margin-right: 1rem; }}
</style></head><body>
<nav><a href="../">&larr; Package Index</a> &middot; <a href="../dashboard/">Dashboard</a></nav>
<h1>Full Build Matrix</h1>
<p class="sub">Every wheel PyTorch publishes, against what this farm declares.
CUDA indexes are discovered from
<a href="https://download.pytorch.org/whl/">download.pytorch.org/whl</a>, not hardcoded.
Filtered to cp310+, no free-threaded builds. <strong>L</strong> = linux, <strong>W</strong> = windows.</p>
<div class="stats">
  <strong>{matrix['total']}</strong> upstream wheel targets across
  <strong>{len(matrix['cuda_versions'])}</strong> CUDA indexes
  ({', '.join(matrix['cuda_versions'])})<br>
  <span class="key" style="display:inline-block;margin-top:.5rem">
    <span style="color:#4ade80">&#9632; {cov.get('declared', 0)} declared</span>
    <span style="color:#fbbf24">&#9632; {cov.get('todo', 0)} still to build</span>
    <span style="color:#7d95b8">&#9632; {cov.get('patch', 0)} patch releases (grid tracks .0)</span>
    <span style="color:#6e7681">&#9632; {cov.get('phantom', 0)} phantom (upstream never shipped)</span>
  </span>
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
