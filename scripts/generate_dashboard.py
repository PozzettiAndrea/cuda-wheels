#!/usr/bin/env python3
"""Generate a dashboard page showing all available wheels and their metadata."""
import json
import math
import os
import re
import shutil
import struct
import urllib.request
import yaml
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CACHE_FILE = SCRIPT_DIR / ".wheel_contents_cache.json"


# --- Wheel contents extraction via Range requests ---

def _load_contents_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_contents_cache(cache):
    CACHE_FILE.write_text(json.dumps(cache, separators=(",", ":")))


def _extract_contents_range(url, token=None):
    """Extract file listing and METADATA from a wheel using HTTP Range requests.

    Downloads only the zip central directory (~64KB) instead of the full file,
    plus a small Range request for the METADATA file.
    Returns (list of {path, size, dir}, metadata_str) or (None, None) on failure.
    """
    try:
        # Don't send auth tokens to download URLs (they redirect to CDNs that reject them)
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            final_url = resp.url  # after redirects

        if total_size == 0:
            return None, None

        # Download last 64KB (contains zip End of Central Directory + central directory)
        tail_size = min(65536, total_size)
        range_headers = {"Range": f"bytes={total_size - tail_size}-{total_size - 1}"}
        req = urllib.request.Request(final_url, headers=range_headers)
        with urllib.request.urlopen(req) as resp:
            tail = resp.read()

        # Find End of Central Directory record (signature 0x06054b50)
        eocd_sig = b"\x50\x4b\x05\x06"
        eocd_pos = tail.rfind(eocd_sig)
        if eocd_pos == -1:
            return None, None

        # Parse EOCD: skip sig(4), disk stuff(4), then num_entries(2), cd_size(4), cd_offset(4)
        if len(tail) - eocd_pos < 22:
            return None, None
        num_entries = struct.unpack_from("<H", tail, eocd_pos + 8)[0]
        cd_size = struct.unpack_from("<I", tail, eocd_pos + 12)[0]
        cd_offset = struct.unpack_from("<I", tail, eocd_pos + 16)[0]

        # Check if central directory is within our tail buffer
        cd_start_in_tail = cd_offset - (total_size - tail_size)
        if cd_start_in_tail >= 0:
            cd_data = tail[cd_start_in_tail:cd_start_in_tail + cd_size]
        else:
            # Need a separate Range request for the central directory
            range_headers = {"Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"}
            req = urllib.request.Request(final_url, headers=range_headers)
            with urllib.request.urlopen(req) as resp:
                cd_data = resp.read()

        # Parse central directory entries, also look for METADATA file
        files = []
        meta_entry = None  # (compression_method, compressed_size, local_header_offset)
        pos = 0
        cd_file_sig = b"\x50\x4b\x01\x02"
        while pos + 46 <= len(cd_data):
            if cd_data[pos:pos + 4] != cd_file_sig:
                break
            comp_method = struct.unpack_from("<H", cd_data, pos + 10)[0]
            comp_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
            uncomp_size = struct.unpack_from("<I", cd_data, pos + 24)[0]
            name_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
            extra_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
            comment_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
            local_offset = struct.unpack_from("<I", cd_data, pos + 42)[0]
            pos += 46
            if pos + name_len > len(cd_data):
                break
            name = cd_data[pos:pos + name_len].decode("utf-8", errors="replace")
            pos += name_len + extra_len + comment_len
            files.append({"path": name, "size": uncomp_size, "dir": name.endswith("/")})

            # Identify the METADATA file (inside .dist-info directory)
            if name.endswith(".dist-info/METADATA"):
                meta_entry = (comp_method, comp_size, uncomp_size, local_offset)

        # Extract METADATA file content via a small Range request
        metadata_str = None
        if meta_entry:
            comp_method, comp_size, uncomp_size, local_offset = meta_entry
            # Local file header: 30 fixed bytes + variable filename + extra field
            # Download enough to cover header + compressed data
            fetch_size = 30 + 256 + comp_size  # 256 bytes buffer for filename + extra
            range_start = local_offset
            range_end = min(local_offset + fetch_size - 1, total_size - 1)
            try:
                range_headers = {"Range": f"bytes={range_start}-{range_end}"}
                req = urllib.request.Request(final_url, headers=range_headers)
                with urllib.request.urlopen(req) as resp:
                    local_data = resp.read()

                # Parse local file header to find data start
                if len(local_data) >= 30 and local_data[:4] == b"\x50\x4b\x03\x04":
                    local_name_len = struct.unpack_from("<H", local_data, 26)[0]
                    local_extra_len = struct.unpack_from("<H", local_data, 28)[0]
                    data_start = 30 + local_name_len + local_extra_len
                    raw_data = local_data[data_start:data_start + comp_size]

                    if comp_method == 8:  # deflate
                        metadata_str = zlib.decompress(raw_data, -15).decode("utf-8", errors="replace")
                    elif comp_method == 0:  # stored
                        metadata_str = raw_data.decode("utf-8", errors="replace")
            except Exception as e:
                print(f"  Warning: failed to extract METADATA from {url}: {e}")

        return files, metadata_str
    except Exception as e:
        print(f"  Warning: failed to extract {url}: {e}")
        return None, None


def extract_all_contents(all_pkg_wheels, token=None):
    """Extract contents and metadata for all wheels using cache + parallel Range requests."""
    cache = _load_contents_cache()
    to_fetch = []

    for pkg, wheels in all_pkg_wheels.items():
        for i, w in enumerate(wheels):
            url = w.get("url", "")
            raw_size = w.get("raw_size")
            cache_key = url
            if cache_key and cache_key in cache and cache[cache_key].get("size") == raw_size and "metadata" in cache[cache_key]:
                w["contents"] = cache[cache_key]["files"]
                if cache[cache_key]["metadata"]:
                    w["metadata"] = cache[cache_key]["metadata"]
            elif url:
                to_fetch.append((pkg, i, url, raw_size))

    if to_fetch:
        print(f"  Extracting contents for {len(to_fetch)} wheels ({len(cache)} cached)...")
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_extract_contents_range, url, token): (pkg, i, url, raw_size)
                for pkg, i, url, raw_size in to_fetch
            }
            for future in as_completed(futures):
                pkg, i, url, raw_size = futures[future]
                files, metadata = future.result()
                if files is not None:
                    all_pkg_wheels[pkg][i]["contents"] = files
                    if metadata:
                        all_pkg_wheels[pkg][i]["metadata"] = metadata
                    cache[url] = {"size": raw_size, "files": files, "metadata": metadata}
                done += 1
                if done % 50 == 0:
                    print(f"  ... {done}/{len(to_fetch)}")

        print(f"  Done: extracted {done} wheels")
    else:
        print(f"  All {sum(len(v) for v in all_pkg_wheels.values())} wheels cached")

    _save_contents_cache(cache)


def _github_api(url: str, token: str = None) -> dict:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())


def get_releases(repo: str, token: str = None) -> list:
    return _github_api(f"https://api.github.com/repos/{repo}/releases", token)


def get_workflow_runs(repo: str, token: str = None) -> dict:
    """Fetch all workflow runs for build.yml and group by package name."""
    base = f"https://api.github.com/repos/{repo}/actions/workflows/build.yml/runs"
    per_page = 100

    data = _github_api(f"{base}?per_page=1", token)
    total = data.get("total_count", 0)
    pages = math.ceil(total / per_page)

    runs_by_pkg = {}
    title_re = re.compile(r"^Build (\S+) wheels")

    for page in range(1, pages + 1):
        data = _github_api(f"{base}?per_page={per_page}&page={page}", token)
        for run in data.get("workflow_runs", []):
            title = run.get("display_title", "")
            m = title_re.match(title)
            if not m:
                continue
            pkg = m.group(1).lower().replace("_", "-")
            runs_by_pkg.setdefault(pkg, []).append({
                "html_url": run["html_url"],
                "display_title": title,
                "conclusion": run.get("conclusion"),
                "created_at": run.get("created_at", ""),
            })

    return runs_by_pkg


def _format_duration(total_secs: int) -> str:
    if total_secs >= 3600:
        return f"{total_secs // 3600}h {(total_secs % 3600) // 60}m"
    elif total_secs >= 60:
        return f"{total_secs // 60}m {total_secs % 60}s"
    return f"{total_secs}s"


def _parse_job_duration(job: dict) -> str | None:
    started = job.get("started_at")
    completed = job.get("completed_at")
    if not started or not completed:
        return None
    s = datetime.fromisoformat(started.replace("Z", "+00:00"))
    e = datetime.fromisoformat(completed.replace("Z", "+00:00"))
    return _format_duration(int((e - s).total_seconds()))


def get_build_durations(repo: str, token: str = None) -> tuple[dict, dict]:
    """Fetch per-job build durations from ALL workflow runs.

    Fetches all runs for build.yml, then fetches jobs for each run.
    Handles two job name formats:
      New: "Linux sageattention py3.10 cu124 torch2.4.0"
      Old: "Windows torch_generic_nms py3.10 cu126"

    Returns (specific_durations, fallback_durations):
      specific: {pkg-cu{X}-torch{Y}-cp{Z}-{os} -> duration_str}
      fallback: {pkg-cu{X}-cp{Z}-{os} -> duration_str}  (for old format without torch)
    """
    # Two job name patterns
    new_re = re.compile(
        r"^(?P<os>\S+)\s+(?P<pkg>\S+)\s+py(?P<py>\d+\.\d+)\s+cu(?P<cu>\d+)\s+torch(?P<torch>\S+)$"
    )
    old_re = re.compile(
        r"^(?P<os>\S+)\s+(?P<pkg>\S+)\s+py(?P<py>\d+\.\d+)\s+cu(?P<cu>\d+)$"
    )

    specific = {}   # with torch version
    fallback = {}   # without torch version

    # Fetch ALL workflow runs (paginated)
    base = f"https://api.github.com/repos/{repo}/actions/workflows/build.yml/runs"
    data = _github_api(f"{base}?per_page=1", token)
    total = data.get("total_count", 0)
    per_page = 100
    pages = math.ceil(total / per_page)

    all_run_ids = []
    for page in range(1, pages + 1):
        data = _github_api(f"{base}?per_page={per_page}&page={page}", token)
        for run in data.get("workflow_runs", []):
            all_run_ids.append(run["id"])

    print(f"  Fetching jobs for {len(all_run_ids)} runs...")

    def _fetch_jobs(run_id):
        try:
            return _github_api(
                f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100",
                token,
            )
        except Exception:
            return {"jobs": []}

    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_jobs, rid): rid for rid in all_run_ids}
        for future in as_completed(futures):
            jobs_data = future.result()
            for job in jobs_data.get("jobs", []):
                if job.get("conclusion") != "success":
                    continue
                name = job.get("name", "")

                # Try new format first
                m = new_re.match(name)
                if m:
                    os_tag = "linux" if m.group("os").lower() == "linux" else "win"
                    pkg = m.group("pkg").lower().replace("_", "-")
                    py = m.group("py").replace(".", "")
                    cu = m.group("cu")
                    torch_raw = m.group("torch").replace(".", "")
                    # Normalize: "240" -> "24", "2100" -> "210"
                    if len(torch_raw) >= 3 and torch_raw.endswith("0"):
                        torch_raw = torch_raw.rstrip("0") or torch_raw[:-1]

                    key = f"{pkg}-cu{cu}-torch{torch_raw}-cp{py}-{os_tag}"
                    if key not in specific:
                        dur = _parse_job_duration(job)
                        if dur:
                            specific[key] = dur
                    # Also record as fallback
                    fb_key = f"{pkg}-cu{cu}-cp{py}-{os_tag}"
                    if fb_key not in fallback:
                        dur = _parse_job_duration(job)
                        if dur:
                            fallback[fb_key] = dur
                    continue

                # Try old format (no torch)
                m = old_re.match(name)
                if m:
                    os_tag = "linux" if m.group("os").lower() == "linux" else "win"
                    pkg = m.group("pkg").lower().replace("_", "-")
                    py = m.group("py").replace(".", "")
                    cu = m.group("cu")
                    fb_key = f"{pkg}-cu{cu}-cp{py}-{os_tag}"
                    if fb_key not in fallback:
                        dur = _parse_job_duration(job)
                        if dur:
                            fallback[fb_key] = dur

            done += 1
            if done % 25 == 0:
                print(f"  ... {done}/{len(all_run_ids)} runs")

    print(f"  Found {len(specific)} specific + {len(fallback)} fallback durations")
    return specific, fallback


def _build_duration_keys(wheel_name: str) -> tuple[str, str]:
    """Build config keys from a wheel filename. Returns (specific_key, fallback_key)."""
    m = re.match(
        r"^[^-]+-[^+]+\+cu(\d+)torch(\d+)-cp(\d+)-[^-]+-(.+)\.whl$",
        wheel_name,
    )
    if not m:
        return "", ""
    cu, torch_ver, py, plat = m.group(1), m.group(2), m.group(3), m.group(4)
    pkg = wheel_name.split("-")[0].lower().replace("_", "-")
    os_tag = "linux" if "linux" in plat or "manylinux" in plat else "win"
    specific = f"{pkg}-cu{cu}-torch{torch_ver}-cp{py}-{os_tag}"
    fallback = f"{pkg}-cu{cu}-cp{py}-{os_tag}"
    return specific, fallback


def parse_wheel_filename(filename: str) -> dict:
    """Extract metadata from wheel filename."""
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
    cuda_m = re.search(r"cu(\d{2,3})", ver)
    torch_m = re.search(r"torch(\d{2,3})", ver) or re.search(r"pt(\d{2,3})", ver)

    if cuda_m:
        c = cuda_m.group(1)
        info["cuda"] = f"{c[:-1]}.{c[-1]}" if len(c) <= 3 else c
    if torch_m:
        t = torch_m.group(1)
        info["torch"] = f"{t[0]}.{t[1:]}" if len(t) <= 3 else t

    py = m.group("pytag")
    digits = py.replace("cp", "")
    if len(digits) >= 2:
        info["python_version"] = f"{digits[0]}.{digits[1:]}"

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


# --- HTML rendering helpers ---

def _format_size(size_bytes):
    if size_bytes is None:
        return "-"
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _status_icon(conclusion):
    if conclusion == "success":
        return '<span style="color:#3fb950">&#x2713;</span>'
    elif conclusion == "failure":
        return '<span style="color:#f85149">&#x2717;</span>'
    elif conclusion is None:
        return '<span style="color:#d29922">&#x25cf;</span>'
    return '<span style="color:#8b949e">&#x25cb;</span>'


def _runs_cell(runs):
    if not runs:
        return "-"
    links = []
    for r in runs:
        icon = _status_icon(r["conclusion"])
        date = r["created_at"][:10] if r.get("created_at") else ""
        title = r.get("display_title", "")
        links.append(f'<a href="{r["html_url"]}" title="{title} ({date})">{icon} {date}</a>')
    return "<br>".join(links)


def _built_row(p):
    release_link = f'<a href="{p["release_url"]}">Release</a>' if p.get("release_url") else "-"
    runs_html = _runs_cell(p["runs"])
    return (
        f'<tr>'
        f'<td><strong>{p["name"]}</strong></td>'
        f'<td>{", ".join(p["versions"])}</td>'
        f'<td><a href="#pkg={p["name"]}" class="wheel-count">{p["count"]}</a></td>'
        f'<td class="runs-cell">{runs_html}</td>'
        f'<td>{release_link}</td>'
        f'</tr>'
    )


# --- Missing wheels computation ---

def _wheel_exists(wheel_names, cuda_short, torch_short, python_short, platform):
    """Check if a wheel matching this combo exists. Handles v1 (torch24) and v2 (torch2.4) naming."""
    torch_v1 = torch_short.replace(".", "")
    patterns = [
        f"+cu{cuda_short}torch{torch_short}-cp{python_short}-cp{python_short}-",
        f"+cu{cuda_short}torch{torch_v1}-cp{python_short}-cp{python_short}-",
    ]
    if platform == "linux":
        return any(p in w and ("manylinux" in w or "linux_x86_64" in w)
                   for p in patterns for w in wheel_names)
    else:
        return any(p in w and "win_amd64" in w
                   for p in patterns for w in wheel_names)


def compute_missing_wheels(built_packages, packages_dir):
    """Compare YAML-defined build matrix against actually built wheels."""
    result = {}
    for pkg_file in sorted(packages_dir.glob("*.yml")):
        pkg = yaml.safe_load(pkg_file.read_text())
        pkg_name = pkg["name"]

        # Collect actual wheel filenames - try both name forms
        lookup_names = [
            pkg_name.lower().replace("_", "-"),
            pkg_name.lower().replace("-", "_"),
            pkg_name.lower(),
        ]
        wheel_names = set()
        for ln in lookup_names:
            for w in built_packages.get(ln, []):
                wheel_names.add(w.get("display_name", ""))

        build = pkg["build_matrix"]
        platforms = build.get("platforms", ["linux", "windows"])

        expected = 0
        missing = []

        if "combinations" in build:
            combos = build["combinations"]
        else:
            combos = [{"cuda": c, "pytorch": p, "python_versions": build.get("python_versions", [])}
                      for c in build.get("cuda_versions", [])
                      for p in build.get("pytorch_versions", [])]

        for combo in combos:
            cuda = combo["cuda"]
            pytorch = combo["pytorch"]
            python_versions = combo.get("python_versions", build.get("python_versions", []))
            cuda_short = cuda.replace(".", "")
            torch_short = ".".join(pytorch.split(".")[:2])

            for py_ver in python_versions:
                py_short = py_ver.replace(".", "")
                for platform in platforms:
                    expected += 1
                    if not _wheel_exists(wheel_names, cuda_short, torch_short, py_short, platform):
                        missing.append({
                            "cuda": cuda,
                            "torch": pytorch,
                            "python": py_ver,
                            "platform": platform,
                        })

        result[pkg_name] = {
            "expected": expected,
            "built": expected - len(missing),
            "missing": missing,
        }

    return result


# --- Main generation ---

def generate_dashboard(built_packages: dict, output_dir: Path,
                       release_urls: dict = None, workflow_runs: dict = None,
                       repo: str = "PozzettiAndrea/cuda-wheels", token: str = None):
    """Generate dashboard HTML from template + static assets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    release_urls = release_urls or {}
    workflow_runs = workflow_runs or {}

    # Build summaries
    built_summaries = []
    all_pkg_wheels = {}
    for name in sorted(built_packages.keys()):
        wheels = built_packages[name]
        versions = sorted(set(w.get("version", "?").split("+")[0] for w in wheels))
        wheel_list = []
        for w in sorted(wheels, key=lambda x: x.get("display_name", x.get("version", ""))):
            wheel_list.append({
                "name": w.get("display_name", f"{w['package']}-{w['version']}-{w['python']}-{w['python']}-{w['platform']}.whl"),
                "url": w.get("url", ""),
                "size": _format_size(w.get("size")),
                "raw_size": w.get("size"),
            })
        all_pkg_wheels[name] = wheel_list
        built_summaries.append({
            "name": name,
            "count": len(wheels),
            "versions": versions,
            "release_url": release_urls.get(name),
            "runs": workflow_runs.get(name, []),
        })

    total_wheels = sum(p["count"] for p in built_summaries)

    # Compute missing wheels
    packages_dir = SCRIPT_DIR.parent / "packages"
    missing_data = compute_missing_wheels(built_packages, packages_dir)
    total_missing = sum(d["expected"] - d["built"] for d in missing_data.values())

    # Render rows
    built_rows = "\n".join(_built_row(p) for p in built_summaries)

    # Read template
    template = (SCRIPT_DIR / "dashboard_template.html").read_text()

    # Substitute placeholders
    html = template.replace("{{total_packages}}", str(len(built_summaries)))
    html = html.replace("{{total_wheels}}", str(total_wheels))
    html = html.replace("{{built_count}}", str(len(built_summaries)))
    html = html.replace("{{built_rows}}", built_rows)
    html = html.replace("{{repo}}", repo)
    html = html.replace("{{missing_count}}", str(total_missing))

    (output_dir / "index.html").write_text(html)

    # Extract wheel contents (Range requests + cache)
    print("Extracting wheel contents...")
    extract_all_contents(all_pkg_wheels, token=token)

    # Fetch build durations and merge into wheel data
    print("Fetching build durations...")
    specific_dur, fallback_dur = get_build_durations(repo, token)
    matched = 0
    for wheels in all_pkg_wheels.values():
        for w in wheels:
            sk, fk = _build_duration_keys(w.get("name", ""))
            dur = specific_dur.get(sk) or fallback_dur.get(fk)
            if dur:
                w["build_time"] = dur
                matched += 1
    print(f"  {matched} wheels matched with build durations")

    # Strip raw_size from output (only needed for cache matching)
    for wheels in all_pkg_wheels.values():
        for w in wheels:
            w.pop("raw_size", None)

    # Write package data as a separate JS file
    (output_dir / "packages.js").write_text(f"window.__WHEEL_DATA__ = {json.dumps(all_pkg_wheels)};\n")

    # Write missing wheels data
    (output_dir / "missing-data.js").write_text(f"window.__MISSING_DATA__ = {json.dumps(missing_data)};\n")

    # Write lightweight install data (just name + url, no contents/metadata)
    install_data = {}
    for pkg, wheels in all_pkg_wheels.items():
        install_data[pkg] = [{"n": w["name"], "u": w["url"]} for w in wheels if w.get("url")]
    (output_dir / "install-data.js").write_text(f"window.__INSTALL_DATA__ = {json.dumps(install_data, separators=(',', ':'))};\n")

    # Copy static assets
    static_dir = SCRIPT_DIR / "dashboard_static"
    for f in static_dir.iterdir():
        shutil.copy2(f, output_dir / f.name)

    print(f"Dashboard: {len(built_summaries) + len(ext_summaries)} packages, {total_wheels} total wheels")


def main():
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "PozzettiAndrea/cuda-wheels")

    print(f"Generating dashboard for {repo}")

    releases = get_releases(repo, token)

    # Collect built wheels from releases + release URLs + sizes
    built_packages = {}
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
            built_packages.setdefault(pkg_name, []).append(info)
            if pkg_name not in release_urls:
                release_urls[pkg_name] = release.get("html_url")

    # Collect workflow runs
    print("Fetching workflow runs...")
    workflow_runs = get_workflow_runs(repo, token)
    total_runs = sum(len(v) for v in workflow_runs.values())
    print(f"  {total_runs} runs across {len(workflow_runs)} packages")

    # Generate dashboard
    generate_dashboard(built_packages, Path("docs") / "dashboard",
                       release_urls=release_urls, workflow_runs=workflow_runs, repo=repo,
                       token=token)


if __name__ == "__main__":
    main()
