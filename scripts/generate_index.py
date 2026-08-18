#!/usr/bin/env python3
"""Generate PEP 503 compliant package index from GitHub releases."""
import os
import json
import re
import urllib.request
from pathlib import Path

import yaml

# Matches v2 torch naming: +cu128torch2.9-cp (dot between major.minor)
_V2_TORCH_RE = re.compile(r'(\+cu\d+torch)(\d)\.(\d+)(-cp)')

# Pulls the combo out of a wheel filename: +cu128torch2.8 -> ("cu128", "torch2.8")
_COMBO_RE = re.compile(r'\+(cu\d+)(torch[\d.]+)')


def _next_link(link_header):
    """Return the rel="next" URL from a GitHub Link header, or None."""
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.split(";")
        if len(section) >= 2 and 'rel="next"' in section[1].strip():
            return section[0].strip().strip("<>")
    return None


# Pulls the Python tag out of a wheel filename: -cp313-cp313t- -> "cp313"
_PYTAG_RE = re.compile(r"-(cp\d+)-cp\d+t?-")


def load_torch_free_packages(pkg_dir=Path("packages")) -> set:
    """Index-normalised names of packages declaring `links_torch: false`.

    These do not link libtorch, so one built wheel is valid for every torch in
    its CUDA line. See CW-ADR-0011.
    """
    names = set()
    if not pkg_dir.is_dir():
        return names
    for path in sorted(pkg_dir.glob("*.yml")):
        if path.stem.startswith("_"):
            continue
        try:
            cfg = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError as e:
            raise RuntimeError(f"{path}: unparseable package config: {e}") from None
        if cfg.get("links_torch") is False:
            name = (cfg.get("name") or path.stem).lower().replace("_", "-")
            names.add(name)
    return names


def load_grid(defaults_path=Path("packages/_defaults.yml")) -> dict:
    """{"cu128": {"torch2.9": {"cp310", ...}, ...}, ...} from the shared grid.

    The alias set is derived from the grid rather than hardcoded so that adding
    a CUDA or torch line to _defaults.yml automatically widens the aliases.
    """
    grid = {}
    if not defaults_path.is_file():
        return grid
    cfg = yaml.safe_load(defaults_path.read_text()) or {}
    for combo in cfg.get("combinations", []):
        cuda = "cu" + str(combo["cuda"]).replace(".", "")
        torch = "torch" + ".".join(str(combo["pytorch"]).split(".")[:2])
        pys = {"cp" + str(v).replace(".", "") for v in combo.get("python_versions", [])}
        grid.setdefault(cuda, {}).setdefault(torch, set()).update(pys)
    return grid


def expand_torch_free_aliases(packages: dict, torch_free: set, grid: dict) -> int:
    """List one torch-free asset under every torch in its CUDA line.

    The wheel is built once and uploaded once. comfy-env's index resolver
    filters on the anchor *text* and downloads from the *href*
    (`packages/cuda_wheels.py`), and the two are independent -- so emitting the
    same href under several display names makes a single asset resolvable for
    every torch, at the cost of nothing but anchor tags.

    Known limits, recorded here because they are invisible at the call site:

    - comfy-env's tier-2 fallback walks the Releases API and matches on the
      real asset name, which carries exactly one torch tag. An aliased wheel is
      therefore findable under every torch via the index but only under its
      built torch via the fallback -- a gap that opens only when GH Pages is
      unreachable.
    - pip takes the filename from the URL, not the anchor text, so a user
      resolving for torch 2.11 installs a distribution whose version reads
      `+cu128torch2.8`. `pip freeze` will disagree with the environment.

    Both are accepted for the transition. The durable fix is a torch-less local
    tag understood by both resolvers (CW-ADR-0011).
    """
    aliased = 0
    for pkg in sorted(torch_free & set(packages)):
        wheels = packages[pkg]
        seen = {w["filename"] for w in wheels}
        for wheel in list(wheels):
            m = _COMBO_RE.search(wheel["filename"])
            pm = _PYTAG_RE.search(wheel["filename"])
            if not m or not pm:
                continue
            cuda, built_torch, py_tag = m.group(1), m.group(2), pm.group(1)
            for torch, pys in sorted(grid.get(cuda, {}).items()):
                if torch == built_torch:
                    continue
                # Don't advertise a (torch, python) pairing upstream never
                # shipped -- nothing would ever ask for it, and it is noise.
                if py_tag not in pys:
                    continue
                alias = wheel["filename"].replace(
                    f"+{cuda}{built_torch}", f"+{cuda}{torch}", 1
                )
                if alias in seen:
                    continue
                seen.add(alias)
                wheels.append({
                    "filename": alias,
                    "v1_filename": _V2_TORCH_RE.sub(
                        lambda x: f"{x.group(1)}{x.group(2)}{x.group(3)}{x.group(4)}",
                        alias,
                    ),
                    "url": wheel["url"],          # same asset, no second upload
                    "alias_of": wheel["filename"],
                })
                aliased += 1
    return aliased


def get_releases(repo: str, token: str = None) -> list:
    """Fetch ALL releases from a GitHub repository.

    This endpoint is paginated -- 30 per page by default. A single unpaginated
    fetch silently truncates once the repo passes that many releases, and the
    resulting short index looks exactly like a healthy one: the packages that
    fall off simply stop existing as far as consumers are concerned. Ask for the
    maximum page size and follow the Link: rel="next" chain to the end.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    releases = []
    pages = 0
    while url:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            releases.extend(json.loads(response.read().decode()))
            url = _next_link(response.headers.get("Link"))
        pages += 1
        if pages > 50:  # 5000 releases; a runaway Link chain is a bug, not a repo
            raise RuntimeError("release pagination did not terminate")
    print(f"Fetched {len(releases)} releases across {pages} page(s)")
    return releases


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

    # One built wheel, many display names, for packages that never link
    # libtorch. Must run before the guard so aliases are counted in the index.
    torch_free = load_torch_free_packages()
    n_aliases = expand_torch_free_aliases(packages, torch_free, load_grid())
    if torch_free:
        print(f"torch-independent packages: {', '.join(sorted(torch_free))}")
        print(f"  emitted {n_aliases} alias listing(s) (0 extra wheels built or stored)")

    # Guard: never publish an index shorter than the last one. A truncated
    # fetch, an auth failure, or an API hiccup must fail the run rather than
    # quietly dropping packages -- losing them produces no error anywhere
    # downstream, and the per-package pages keep serving stale wheel lists.
    v2_dir = Path("docs/v2")
    previous = {d.name for d in v2_dir.iterdir() if d.is_dir()} if v2_dir.is_dir() else set()
    lost = previous - set(packages)
    if lost:
        print(f"ERROR: {len(lost)} package(s) in the previous index are absent now:")
        for name in sorted(lost):
            print(f"  - {name}")
        print("Refusing to publish a shorter index. If a package was removed")
        print("deliberately, delete its docs/v2/<name>/ directory in the same commit.")
        raise SystemExit(1)
    print(f"{len(packages)} packages, {sum(len(v) for v in packages.values())} wheels")

    # Create docs directory
    docs = Path("docs")
    docs.mkdir(exist_ok=True)

    all_packages = sorted(packages.keys())

    # Generate root index
    with open(docs / "index.html", "w") as f:
        f.write("<!DOCTYPE html>\n")
        f.write("<html>\n<head><title>CUDA Wheels Index</title></head>\n")
        f.write("<body>\n")
        f.write("<h1>CUDA Wheels</h1>\n")
        for pkg in all_packages:
            f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
        f.write("</body>\n</html>\n")

    # Generate per-package index.
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

    # Per-combo indexes: docs/<cuda>/<torch>/<pkg>/
    #
    # The flat index cannot be resolved. A wheel's CUDA and torch versions live
    # only in its local version tag, and pip matches neither -- so an unpinned
    # install against /v2/ picks the highest combo present, not the one the
    # machine can load. GPU architecture is not expressible at all. Putting the
    # combo in the URL is the only way to make selection unambiguous, and it is
    # what download.pytorch.org does (/whl/cu128/).
    #
    # Additive: /v2/ is untouched, so existing consumers are unaffected.
    combos = {}
    for pkg, wheels in packages.items():
        for wheel in wheels:
            m = _COMBO_RE.search(wheel["filename"])
            if not m:
                continue
            combos.setdefault((m.group(1), m.group(2)), {}).setdefault(pkg, []).append(wheel)

    for (cuda, torch), pkgs in sorted(combos.items()):
        combo_dir = docs / cuda / torch
        combo_dir.mkdir(parents=True, exist_ok=True)
        with open(combo_dir / "index.html", "w") as f:
            f.write("<!DOCTYPE html>\n")
            f.write(f"<html>\n<head><title>CUDA Wheels {cuda}/{torch}</title></head>\n")
            f.write("<body>\n")
            f.write(f"<h1>CUDA Wheels -- {cuda} / {torch}</h1>\n")
            for pkg in sorted(pkgs):
                f.write(f'<a href="{pkg}/">{pkg}</a><br>\n')
            f.write("</body>\n</html>\n")

        for pkg, wheels in pkgs.items():
            pkg_dir = combo_dir / pkg
            pkg_dir.mkdir(exist_ok=True)
            with open(pkg_dir / "index.html", "w") as f:
                f.write("<!DOCTYPE html>\n")
                f.write(f"<html>\n<head><title>{pkg} {cuda}/{torch}</title></head>\n")
                f.write("<body>\n")
                f.write(f"<h1>{pkg} -- {cuda} / {torch}</h1>\n")
                for wheel in sorted(wheels, key=lambda w: w["filename"]):
                    f.write(f'<a href="{wheel["url"]}">{wheel["filename"]}</a><br>\n')
                f.write("</body>\n</html>\n")

    print(f"Generated {len(combos)} per-combo indexes "
          f"({sum(len(p) for p in combos.values())} package entries)")

    # Generate dashboard (separate from PEP 503 index)
    try:
        from generate_dashboard import generate_dashboard, parse_wheel_filename, get_workflow_runs

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

        generate_dashboard(built_for_dashboard, docs / "dashboard",
                           release_urls=release_urls, workflow_runs=workflow_runs, repo=repo,
                           token=token)
    except Exception as e:
        print(f"Dashboard generation failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
