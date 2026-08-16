#!/usr/bin/env python3
"""Analyze build gaps: compare package YAMLs against published release assets.

Reads each packages/*.yml, builds the expected wheel matrix (excluding patch
releases), fetches actual assets from GitHub, and prints a summary of what's
missing.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = "PozzettiAndrea/cuda-wheels"

# Torch minor versions to leave out of the expected grid, e.g.
#   python scripts/gap_analysis.py --exclude-torch 2.11
# Useful when a torch release is still rolling out across the farm and its
# absence would otherwise swamp the real gaps.
EXCLUDE_TORCH = set()
PACKAGES_DIR = Path(__file__).parent.parent / "packages"
PATCH_VERSIONS = {"2.4.1", "2.5.1", "2.7.1", "2.9.1"}

# Combos upstream PyTorch never shipped. Imported from generate_matrix so the
# planner and this report cannot disagree: they previously kept separate
# literals and drifted, this one missing the five cu129/torch2.11 Windows
# entries, which silently inflated "missing" by 5 per package.
sys.path.insert(0, str(Path(__file__).parent))
from generate_matrix import PHANTOM_COMBOS  # noqa: E402

# Parse wheel filename: extract cuda, torch, python, os
# e.g. torch_cluster-1.6.3+cu124torch2.4-cp310-cp310-manylinux...whl
_WHL_RE = re.compile(
    r'\+cu(\d+)torch(\d+\.\d+)-cp(\d+)-.*?(manylinux|linux|win)'
)


def parse_wheel(name):
    """Extract (cuda_short, torch_ver, python_ver, os) from wheel filename."""
    m = _WHL_RE.search(name)
    if not m:
        return None
    cuda = m.group(1)       # "124"
    torch_v = m.group(2)    # "2.4"
    py = m.group(3)         # "310"
    os_ = "linux" if "linux" in m.group(4) else "windows"
    return (cuda, torch_v, py, os_)


def cuda_short(cuda_str):
    """'12.4' -> '124', '13.0' -> '130'"""
    return cuda_str.replace(".", "")


def torch_short(torch_str):
    """'2.4.0' -> '2.4', '2.10.0' -> '2.10'"""
    parts = torch_str.split(".")
    return f"{parts[0]}.{parts[1]}"


def py_short(py_str):
    """'3.10' -> '310', '3.14' -> '314'"""
    return py_str.replace(".", "")


def _ver_tuple(v):
    """'2.10' -> (2, 10), so 2.10 sorts above 2.9 rather than below it."""
    return tuple(int(x) for x in v.split("."))


_DEFAULTS_CACHE = {}


def load_defaults():
    """Shared build matrix from packages/_defaults.yml."""
    if not _DEFAULTS_CACHE:
        with open(PACKAGES_DIR / "_defaults.yml") as f:
            _DEFAULTS_CACHE.update(yaml.safe_load(f) or {})
    return _DEFAULTS_CACHE


def load_expected(yml_path):
    """Load a package YAML and return set of expected (cuda, torch, py, os) tuples,
    excluding patch releases."""
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)

    name = cfg["name"]
    matrix = cfg.get("build_matrix", {})
    # _defaults.yml supplies combinations/platforms for any package that does
    # not define its own -- which is most of them. Without this fallback every
    # such package reported "expected 0" and looked complete no matter what
    # was actually published.
    defaults = load_defaults()
    combos = matrix.get("combinations") or defaults.get("combinations", [])
    platforms = matrix.get("platforms") or defaults.get("platforms", ["linux"])

    # Some packages only support torch >= N (natten).
    min_pt = cfg.get("min_pytorch")

    expected = set()
    for combo in combos:
        pt = combo["pytorch"]
        if pt in PATCH_VERSIONS:
            continue
        if min_pt and _ver_tuple(torch_short(pt)) < _ver_tuple(str(min_pt)):
            continue
        if EXCLUDE_TORCH and torch_short(pt) in EXCLUDE_TORCH:
            continue
        cu = cuda_short(combo["cuda"])
        tv = torch_short(pt)
        for py in combo["python_versions"]:
            for plat in platforms:
                expected.add((cu, tv, py_short(py), plat))

    # Remove combos that don't exist upstream
    expected -= PHANTOM_COMBOS

    # Derive release tag from package name (hyphens → underscores to match GH tags)
    tag_name = name.replace("-", "_")
    tag = f"{tag_name}-latest"
    return name, tag, expected


def get_actual_wheels(tag):
    """Fetch wheel filenames from a GitHub release."""
    r = subprocess.run(
        ["gh", "release", "view", tag, "--repo", REPO, "--json", "assets"],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return []
    assets = json.loads(r.stdout).get("assets", [])
    return [a["name"] for a in assets if a["name"].endswith(".whl")]


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    if "--exclude-torch" in sys.argv:
        EXCLUDE_TORCH.update(
            sys.argv[sys.argv.index("--exclude-torch") + 1].split(",")
        )

    # _defaults.yml holds the shared build matrix, not a package.
    ymls = [p for p in sorted(PACKAGES_DIR.glob("*.yml")) if not p.name.startswith("_")]
    if not ymls:
        print(f"No YAML files found in {PACKAGES_DIR}")
        return

    print(f"{'Package':<25} {'Expected':>8} {'Actual':>8} {'Missing':>8} {'%':>6}")
    print("-" * 60)

    total_expected = 0
    total_actual_matched = 0
    total_missing = 0
    incomplete = []

    for yml in ymls:
        name, tag, expected = load_expected(yml)
        wheels = get_actual_wheels(tag)

        # Parse actual wheels into combo tuples
        actual = set()
        for w in wheels:
            parsed = parse_wheel(w)
            if parsed:
                actual.add(parsed)

        missing = expected - actual
        pct = (len(expected) - len(missing)) / len(expected) * 100 if expected else 0

        print(f"{name:<25} {len(expected):>8} {len(actual):>8} {len(missing):>8} {pct:>5.0f}%")

        total_expected += len(expected)
        total_actual_matched += len(expected) - len(missing)
        total_missing += len(missing)

        if missing:
            incomplete.append((name, tag, expected, actual, missing))

    print("-" * 60)
    total_pct = total_actual_matched / total_expected * 100 if total_expected else 0
    print(f"{'TOTAL':<25} {total_expected:>8} {total_actual_matched:>8} {total_missing:>8} {total_pct:>5.0f}%")

    if verbose and incomplete:
        print("\n\n=== MISSING COMBOS ===\n")
        for name, tag, expected, actual, missing in incomplete:
            print(f"\n{name} ({len(missing)} missing):")
            # Group by cuda+torch
            by_ct = {}
            for cu, tv, py, os_ in sorted(missing):
                key = f"cu{cu}/torch{tv}"
                by_ct.setdefault(key, []).append(f"cp{py}-{os_}")
            for key in sorted(by_ct):
                combos_str = ", ".join(sorted(by_ct[key]))
                print(f"  {key}: {combos_str}")


if __name__ == "__main__":
    main()
