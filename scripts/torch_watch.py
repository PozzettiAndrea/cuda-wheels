#!/usr/bin/env python3
"""Watch upstream PyTorch for combinations the grid does not yet declare.

    python scripts/torch_watch.py                 # report what is new
    python scripts/torch_watch.py --json out.json # machine-readable, for CI
    python scripts/torch_watch.py --apply         # append new rows to _defaults.yml

WHAT COUNTS AS TRUTH
The wheel index at download.pytorch.org is the only source consulted, because
it is the only one that describes what people can actually download -- which is
what this farm has to pin against at build time and what a user's machine will
already have installed.

PyTorch also *declares* a build matrix in pytorch/test-infra
(generate_binary_build_matrix.py). It is deliberately NOT used here: measured
against the server it is wrong in all three directions. It omits cu124, cu128
and cu129, which carry hundreds of downloadable wheels; its CUDA_ARCHES_NO_WINDOWS
lists only 13.4 -- a version that is not published at all -- while missing the
real gap, cu129 having no Windows build for torch 2.12/2.13; and it excludes
Python 3.15 from the release channel although cp315 wheels are on the server.
Sourcing phantom cells from that declaration would have queued Windows jobs
that cannot succeed. Observation beats declaration.

DISTINGUISHING "NEVER BUILT" FROM "WE FAILED TO LOOK"
Absence only means something inside a result we know is good, so:

  * an index must return HTTP 200 AND parse to at least one wheel before any
    conclusion is drawn from it; anything else is a hard error, never a
    "removal" (cu134 currently answers 403 -- that is not evidence of anything)
  * a phantom cell is only recorded when the (cuda, torch) pair is present with
    other platforms and this one is missing -- absence *within* a populated
    result
  * rows are only ever added. Removing a row is a human's decision, because a
    wrong removal is silent and permanent.
"""

import argparse
import collections
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PACKAGES_DIR = Path(__file__).parent.parent / "packages"
DEFAULTS = PACKAGES_DIR / "_defaults.yml"
BASE_URL = "https://download.pytorch.org/whl/{key}/torch/"

# Probed every run so a new CUDA index cannot be missed by omission. Being on
# this list is not a claim that it exists -- a 403/404 simply means "not
# published", which is reported, not acted on.
CANDIDATE_CUDA = [
    "cu124", "cu126", "cu128", "cu129", "cu130", "cu132", "cu134", "cu136",
]

MIN_PYTHON = (3, 10)

# Final releases only. Anything with a suffix -- 2.13.0.dev20260101, 2.13.0rc1 --
# is excluded: the farm must not pin a moving target.
WHEEL_RE = re.compile(
    r">torch-(?P<torch>\d+\.\d+\.\d+)\+(?P<cuda>cu\d+)"
    r"-(?P<pytag>cp\d+t?)-cp\d+t?"
    r"-(?P<plat>[^.<]*(?:x86_64|aarch64|win_amd64))[^<]*\.whl<"
)


class FetchError(RuntimeError):
    """The index could not be read. Never interpreted as absence."""


def fetch_index(key: str) -> str | None:
    """Return index HTML, None if upstream says it does not exist."""
    try:
        req = urllib.request.Request(
            BASE_URL.format(key=key), headers={"User-Agent": "cuda-wheels-torch-watch/1.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status != 200:
                raise FetchError(f"{key}: HTTP {r.status}")
            return r.read().decode()
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None          # not published; a fact, but not a removal
        raise FetchError(f"{key}: HTTP {e.code}") from e
    except Exception as e:
        raise FetchError(f"{key}: {e}") from e


def parse_index(key: str, html: str) -> dict:
    """{(torch, python): {platforms}}. Raises if nothing parses."""
    out = collections.defaultdict(set)
    for m in WHEEL_RE.finditer(html):
        pytag = m.group("pytag")
        if "t" in pytag[2:]:                       # free-threaded build
            continue
        minor = int(pytag[3:])
        if (3, minor) < MIN_PYTHON:
            continue
        plat = m.group("plat")
        platform = ("windows" if "win" in plat
                    else "linux_aarch64" if "aarch64" in plat
                    else "linux")
        out[(m.group("torch"), f"3.{minor}")].add(platform)
    if not out:
        # 200 with nothing parseable means the page changed shape, not that
        # upstream deleted every wheel.
        raise FetchError(f"{key}: 200 but no wheels parsed -- index format may have changed")
    return dict(out)


def survey() -> tuple[dict, list[str], list[str]]:
    """(upstream, unpublished, errors). Errors never become conclusions."""
    upstream, unpublished, errors = {}, [], []
    for key in CANDIDATE_CUDA:
        try:
            html = fetch_index(key)
        except FetchError as e:
            errors.append(str(e))
            continue
        if html is None:
            unpublished.append(key)
            continue
        try:
            upstream[key] = parse_index(key, html)
        except FetchError as e:
            errors.append(str(e))
    return upstream, unpublished, errors


def load_declared() -> dict:
    """({cuda: {(minor, python)}}, {(cuda, minor): exact version}, cfg)."""
    cfg = yaml.safe_load(DEFAULTS.read_text())
    declared = collections.defaultdict(set)
    targets = {}
    for combo in cfg["combinations"]:
        key = "cu" + combo["cuda"].replace(".", "")
        exact = str(combo["pytorch"])
        minor = ".".join(exact.split(".")[:2])
        targets[(key, minor)] = exact
        for py in combo["python_versions"]:
            declared[key].add((minor, py))
    return declared, targets, cfg


def minor_of(v: str) -> str:
    return ".".join(v.split(".")[:2])


def vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def analyse(upstream: dict, declared: dict, targets: dict) -> dict:
    """New CUDA indexes, new (torch, python) cells, and phantom cells.

    Phantoms are judged against the EXACT patch release the grid targets, not
    against the minor as a whole. Patch releases disagree: cu129 torch 2.9.0
    has a Windows wheel and 2.9.1 does not, so folding them together would mark
    cu129/2.9/windows phantom and suppress builds that already exist.
    """
    new_cuda, new_cells, phantoms = [], [], []

    for key, cells in upstream.items():
        if key not in declared:
            versions = sorted({minor_of(t) for t, _ in cells}, key=vkey)
            new_cuda.append({"cuda": key, "torch_minors": versions})
            continue

        for (torch, py), platforms in cells.items():
            minor = minor_of(torch)
            if (minor, py) not in declared[key]:
                new_cells.append({
                    "cuda": key, "torch": torch, "torch_minor": minor,
                    "python": py, "platforms": sorted(platforms),
                })

        # One target patch per (cuda, minor): what the grid declares, or the
        # newest upstream patch for a minor we are about to adopt.
        minors = {minor_of(t) for t, _ in cells}
        for minor in minors:
            target = targets.get((key, minor)) or newest_patch(upstream, key, minor)
            for (torch, py), platforms in cells.items():
                if torch != target:
                    continue
                # Absence *within* a populated result -- never from a failed fetch.
                if "linux" in platforms and "windows" not in platforms:
                    phantoms.append({
                        "cuda": key, "torch": target, "torch_minor": minor,
                        "python": py, "missing": "windows",
                    })

    new_cells.sort(key=lambda c: (c["cuda"], vkey(c["torch"]), vkey(c["python"])))
    phantoms.sort(key=lambda p: (p["cuda"], vkey(p["torch"]), vkey(p["python"])))
    return {"new_cuda": new_cuda, "new_cells": new_cells, "phantoms": phantoms}


def newest_patch(upstream: dict, key: str, minor: str) -> str:
    return max((t for t, _ in upstream[key] if minor_of(t) == minor), key=vkey)


def apply_rows(result: dict, upstream: dict, cfg: dict) -> int:
    """Append new (cuda, torch minor) rows to _defaults.yml. Additive only."""
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_pytorch_arch_lists import fetch as fetch_archs

    wanted = collections.defaultdict(set)
    for c in result["new_cells"]:
        wanted[(c["cuda"], c["torch_minor"])].add(c["python"])

    added = 0
    for (key, minor), pys in sorted(wanted.items()):
        cuda_dotted = f"{key[2:-1]}.{key[-1]}"
        patch = newest_patch(upstream, key, minor)
        # Rows are cells only; arch lists live in packages/_arch_policy.yml
        # and are resolved at build time (CW-ADR-0012).
        cfg["combinations"].append({
            "cuda": cuda_dotted,
            "pytorch": f"{minor}.0" if patch.endswith(".0") else patch,
            "python_versions": sorted(pys, key=vkey),
        })
        print(f"  + cu{cuda_dotted} torch {patch} py{','.join(sorted(pys, key=vkey))}")
        added += 1

    if added:
        cfg["combinations"].sort(key=lambda c: (vkey(c["cuda"]), vkey(str(c["pytorch"]))))
        DEFAULTS.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        print(f"\nWrote {added} row(s) to {DEFAULTS}")
        print("REVIEW BEFORE COMMITTING: arch lists and phantom cells both need eyes.")
    return added


def report(result: dict, unpublished: list, errors: list) -> None:
    if errors:
        print("ERRORS -- treated as 'could not look', not as 'upstream removed it':")
        for e in errors:
            print(f"  ! {e}")
        print()

    if result["new_cuda"]:
        print("NEW CUDA INDEX (needs a human: build.yml choice list, the Windows")
        print("installer URL in setup-cuda, per-package arch_list_by_cuda review)")
        for c in result["new_cuda"]:
            print(f"  ** {c['cuda']}: torch {', '.join(c['torch_minors'])}")
        print()

    if result["new_cells"]:
        by = collections.defaultdict(list)
        for c in result["new_cells"]:
            by[(c["cuda"], c["torch"])].append(c["python"])
        print(f"NEW COMBINATIONS ({len(result['new_cells'])} cells)")
        for (key, torch), pys in sorted(by.items(), key=lambda kv: (kv[0][0], vkey(kv[0][1]))):
            print(f"  + {key} torch {torch}: py {', '.join(sorted(pys, key=vkey))}")
        print()

    if unpublished:
        print(f"probed, not published: {', '.join(unpublished)}\n")

    if not result["new_cuda"] and not result["new_cells"]:
        print("Grid is current with upstream.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH", help="write machine-readable result")
    ap.add_argument("--apply", action="store_true",
                    help="append new rows to _defaults.yml (review before committing)")
    args = ap.parse_args()

    upstream, unpublished, errors = survey()
    if not upstream:
        print("FATAL: no index could be read. Refusing to conclude anything.", file=sys.stderr)
        return 1

    declared, targets, cfg = load_declared()
    result = analyse(upstream, declared, targets)
    report(result, unpublished, errors)

    if args.apply:
        print("\nApplying:")
        apply_rows(result, upstream, cfg)

    if args.json:
        Path(args.json).write_text(json.dumps({
            **result,
            "unpublished": unpublished,
            "errors": errors,
            "has_news": bool(result["new_cuda"] or result["new_cells"]),
        }, indent=2))

    # 1 is reserved for failure, so "news" is signalled in the JSON, not the
    # exit code -- a watcher that cannot tell news from breakage is useless.
    return 1 if errors and not upstream else 0


if __name__ == "__main__":
    sys.exit(main())
