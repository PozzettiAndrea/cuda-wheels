#!/usr/bin/env python3
"""Check wheel status: counts, naming syntax, and version matching."""
import re
import subprocess
import urllib.request
import yaml
from pathlib import Path

# PEP 427 wheel filename pattern with our CUDA+PyTorch suffix
WHEEL_PATTERN = re.compile(
    r'^(?P<name>[a-zA-Z0-9_]+)-'
    r'(?P<version>[0-9][^-]*)'
    r'\+cu(?P<cuda>\d+)torch(?P<torch>[\d.]+)-'
    r'cp(?P<py>\d+)-cp\d+-'
    r'(?P<platform>linux_x86_64|win_amd64)\.whl$'
)


def get_release_wheels(package_name: str, repo: str = "PozzettiAndrea/cuda-wheels") -> list:
    """Get wheel filenames from a GitHub release."""
    result = subprocess.run(
        ["gh", "release", "view", f"{package_name}-latest", "--repo", repo,
         "--json", "assets", "-q", ".assets[].name"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []
    return [w for w in result.stdout.strip().split("\n") if w.endswith(".whl")]


def fetch_url(url: str) -> str | None:
    """Fetch content from URL, return None on failure."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.read().decode()
    except:
        return None


def get_source_version(source_repo: str, source_tag: str, build_subdir: str = "",
                       pkg_name: str = "", wheels: list = None) -> str | None:
    """
    Fetch version from source, trying multiple locations:
    1. pyproject.toml
    2. {pkg_name}/version.py
    3. {pkg_name}/__init__.py
    4. setup.py
    5. Infer from existing wheel filenames
    """
    ref = source_tag if source_tag else "main"
    base = f"https://raw.githubusercontent.com/{source_repo}/{ref}"
    subdir = f"{build_subdir}/" if build_subdir else ""

    # 1. Try pyproject.toml
    content = fetch_url(f"{base}/{subdir}pyproject.toml")
    if content:
        match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
        if match:
            return match.group(1)

    # 2. Try {pkg_name}/version.py (like gsplat)
    if pkg_name:
        content = fetch_url(f"{base}/{subdir}{pkg_name}/version.py")
        if content:
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)

    # 3. Try {pkg_name}/__init__.py
    if pkg_name:
        content = fetch_url(f"{base}/{subdir}{pkg_name}/__init__.py")
        if content:
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)

    # 4. Try setup.py
    content = fetch_url(f"{base}/{subdir}setup.py")
    if content:
        match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)

    # 5. Infer from existing wheel filenames
    if wheels:
        for w in wheels:
            m = WHEEL_PATTERN.match(w)
            if m:
                return m.group("version")

    return None


def load_defaults() -> dict:
    """The shared grid from packages/_defaults.yml, which most packages inherit."""
    path = Path(__file__).parent.parent / "packages" / "_defaults.yml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def get_expected_counts(pkg_config: dict, defaults: dict = None) -> tuple[int, int]:
    """Calculate expected Linux and Windows wheel counts from config.

    Resolution is per-key, matching generate_matrix.py: a package may override
    `combinations`, `platforms`, both, or neither. Most declare no build_matrix
    at all and inherit the whole grid, so indexing it directly is wrong.
    """
    defaults = defaults if defaults is not None else load_defaults()
    build = pkg_config.get("build_matrix") or {}
    platforms = build.get("platforms") or defaults.get("platforms", ["linux"])
    combinations = build.get("combinations") or defaults.get("combinations", [])
    linux_count = win_count = 0

    for combo in combinations:
        py_count = len(combo.get("python_versions", []))
        if "linux" in platforms:
            linux_count += py_count
        if "windows" in platforms:
            win_count += py_count

    return linux_count, win_count


def check_wheel_syntax(wheels: list) -> list:
    """Check wheel filenames match expected pattern."""
    errors = []
    for w in wheels:
        if not WHEEL_PATTERN.match(w):
            errors.append(f"Bad syntax: {w}")
    return errors


def check_wheel_versions(wheels: list, expected_version: str, pkg_name: str) -> list:
    """Check wheel versions match expected version."""
    errors = []
    if not expected_version:
        return []

    for w in wheels:
        match = WHEEL_PATTERN.match(w)
        if match:
            wheel_ver = match.group("version")
            if wheel_ver != expected_version:
                errors.append(f"Version mismatch: {w} has {wheel_ver}, expected {expected_version}")
    return errors


def main():
    packages_dir = Path(__file__).parent.parent / "packages"

    print("Wheel Status Report")
    print("=" * 80)

    total_expected = total_actual = 0
    all_errors = []

    # Summary table
    print(f"\n{'Package':<20} {'Expected':<10} {'Actual':<10} {'Missing':<10} {'Version':<12} {'Status'}")
    print("-" * 80)

    for yml in sorted(packages_dir.glob("*.yml")):
        # _defaults.yml is inherited config, not a package -- it has no `name`.
        # Same skip as generate_matrix.py and gap_analysis.py.
        if yml.name.startswith("_"):
            continue
        pkg = yaml.safe_load(yml.read_text())
        name = pkg["name"]
        source_repo = pkg["source_repo"]
        source_tag = pkg.get("source_tag", "")
        build_subdir = pkg.get("build_subdir", "")

        # Get counts
        linux_exp, win_exp = get_expected_counts(pkg)
        expected = linux_exp + win_exp
        wheels = get_release_wheels(name)
        actual = len(wheels)
        missing = expected - actual

        # Get source version (pass wheels for fallback)
        source_ver = get_source_version(source_repo, source_tag, build_subdir, name, wheels)
        ver_display = source_ver or "?"

        # Status indicator
        if missing == 0:
            status = "OK"
        else:
            status = f"MISSING"

        print(f"{name:<20} {expected:<10} {actual:<10} {missing:<10} {ver_display:<12} {status}")

        # Check syntax and versions
        syntax_errors = check_wheel_syntax(wheels)
        version_errors = check_wheel_versions(wheels, source_ver, name) if source_ver else []

        for err in syntax_errors + version_errors:
            all_errors.append(f"[{name}] {err}")

        total_expected += expected
        total_actual += actual

    print("-" * 80)
    total_missing = total_expected - total_actual
    print(f"{'TOTAL':<20} {total_expected:<10} {total_actual:<10} {total_missing:<10}")

    # Show errors
    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        print("-" * 80)
        for err in all_errors[:20]:
            print(f"  {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    else:
        print("\nAll wheel names and versions OK!")

    if total_missing > 0:
        print(f"\nRun: gh workflow run build.yml -f package=all")


if __name__ == "__main__":
    main()
