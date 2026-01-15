#!/usr/bin/env python3
"""Check wheel status: counts, naming syntax, and version matching."""
import re
import subprocess
import urllib.request
import yaml
from pathlib import Path

# PEP 427 wheel filename pattern
WHEEL_PATTERN = re.compile(
    r'^(?P<name>[a-zA-Z0-9_]+)-'
    r'(?P<version>[0-9][^-]*)'
    r'\+cu(?P<cuda>\d+)torch(?P<torch>\d+)-'
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


def get_source_version(source_repo: str, source_tag: str, build_subdir: str = "") -> str:
    """Fetch version from source pyproject.toml."""
    ref = source_tag if source_tag else "main"
    subdir = f"{build_subdir}/" if build_subdir else ""
    url = f"https://raw.githubusercontent.com/{source_repo}/{ref}/{subdir}pyproject.toml"

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            content = resp.read().decode()
            match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except:
        pass
    return None


def get_expected_counts(pkg_config: dict) -> tuple[int, int]:
    """Calculate expected Linux and Windows wheel counts from config."""
    build = pkg_config["build_matrix"]
    platforms = build["platforms"]
    linux_count = win_count = 0

    for combo in build.get("combinations", []):
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
        return [f"Could not fetch source version for {pkg_name}"]

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
    print(f"\n{'Package':<20} {'Expected':<10} {'Actual':<10} {'Missing':<10} {'Version'}")
    print("-" * 80)

    for yml in sorted(packages_dir.glob("*.yml")):
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

        # Get source version
        source_ver = get_source_version(source_repo, source_tag, build_subdir)
        ver_display = source_ver or "?"

        print(f"{name:<20} {expected:<10} {actual:<10} {missing:<10} {ver_display}")

        # Check syntax and versions
        syntax_errors = check_wheel_syntax(wheels)
        version_errors = check_wheel_versions(wheels, source_ver, name) if source_ver else []

        for err in syntax_errors + version_errors:
            all_errors.append(f"[{name}] {err}")

        total_expected += expected
        total_actual += actual

    print("-" * 80)
    print(f"{'TOTAL':<20} {total_expected:<10} {total_actual:<10} {total_expected - total_actual:<10}")

    # Show errors
    if all_errors:
        print(f"\nErrors ({len(all_errors)}):")
        print("-" * 80)
        for err in all_errors[:20]:  # Limit output
            print(f"  {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")
    else:
        print("\nAll wheel names and versions OK!")

    if total_actual < total_expected:
        print(f"\nRun: gh workflow run build.yml -f package=all")


if __name__ == "__main__":
    main()
