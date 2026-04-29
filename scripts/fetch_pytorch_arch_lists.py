"""Fetch authoritative TORCH_CUDA_ARCH_LIST from PyTorch's build_cuda.sh.

PyTorch's CI `pytorch/.ci/manywheel/build_cuda.sh` (at each release tag) is the
authoritative source for what goes into TORCH_CUDA_ARCH_LIST when a wheel is
built — i.e. exactly which SASS arches and which `+PTX` tail end up in
libtorch_cuda.so / torch_cuda.dll.

We pull that script for a given release tag, extract the relevant base
assignment + `case ${CUDA_VERSION}` block, evaluate it in bash with
$CUDA_VERSION set, and parse the resulting variable. This is more reliable
than scraping the binary fatbin (which since CUDA 12.8+ uses LZ4 compression
that hides most arches from naive byte scans).

Verified against actual wheel binaries on 2026-04-29:
- cu126/2.9.0 lib has only sm_50/60/70/75/80/86/90 strings; build_cuda.sh agrees.
- cu130/2.10.0 lib has compute_120 PTX (603 string hits — heavily compiled);
  build_cuda.sh declares 12.0+PTX. Match.

Schema:
    fetch("v2.7.0", "12.8") -> {"sass": ["sm_75",...,"sm_120"], "ptx": ["sm_120"]}
"""
from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

URL_TEMPLATES = [
    # Modern: pytorch/pytorch tag (v2.6.0 onward — manywheel scripts moved here)
    "https://raw.githubusercontent.com/pytorch/pytorch/{tag}/.ci/manywheel/build_cuda.sh",
    # Legacy: pytorch/builder release branch (v2.4 / v2.5 era)
    "https://raw.githubusercontent.com/pytorch/builder/release/{major_minor}/manywheel/build_cuda.sh",
]

CACHE_FILE = Path(__file__).resolve().parent / ".pytorch-build-cuda-cache.json"
# PyTorch tags don't change post-release, so a long TTL is safe.
CACHE_TTL_DAYS = 30

_script_cache: dict[str, tuple[str | None, str | None]] = {}


def _load_disk_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def _save_disk_cache(cache: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    os.replace(tmp, CACHE_FILE)


def _cache_fresh(entry: dict) -> bool:
    fetched_at = entry.get("fetched_at")
    if not fetched_at:
        return False
    try:
        fetched = datetime.datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    age = datetime.datetime.now(datetime.timezone.utc) - fetched
    return age.days < CACHE_TTL_DAYS


def fetch_script(torch_tag: str) -> tuple[str | None, str | None]:
    """Fetch build_cuda.sh for a torch tag. In-memory cached.

    Returns (script_text, source_url) or (None, None) on every-template-failed."""
    if torch_tag in _script_cache:
        return _script_cache[torch_tag]
    m = re.match(r"v(\d+\.\d+)", torch_tag)
    major_minor = m.group(1) if m else ""
    for tpl in URL_TEMPLATES:
        url = tpl.format(tag=torch_tag, major_minor=major_minor)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                _script_cache[torch_tag] = (r.read().decode(), url)
                return _script_cache[torch_tag]
        except Exception:
            continue
    _script_cache[torch_tag] = (None, None)
    return _script_cache[torch_tag]


def parse_arch_list_token(tok: str) -> dict:
    """'X.Y;X.Y;X.Y+PTX' -> {'sass':['sm_XY',...], 'ptx':['sm_XY',...]}.

    +PTX-suffixed entries contribute to BOTH the sass AND the ptx lists, since
    nvcc emits both `arch=compute_X,code=sm_X` and `arch=compute_X,code=compute_X`
    for those entries.
    """
    sass, ptx = [], []
    for token in re.split(r"[;\s]+", tok.strip()):
        if not token:
            continue
        has_ptx = token.endswith("+PTX")
        a = token[:-4] if has_ptx else token
        try:
            major, minor = a.split(".")
        except ValueError:
            continue
        sm = f"sm_{major}{minor}"
        sass.append(sm)
        if has_ptx:
            ptx.append(sm)
    return {"sass": sass, "ptx": ptx}


def fetch(torch_tag: str, cuda_version: str, arch: str = "x86_64",
          *, use_cache: bool = True, _disk_cache: dict | None = None) -> dict | None:
    """Returns {'sass': [...], 'ptx': [...]} or None on failure.

    Failure modes:
    - GitHub returns 404 for every URL template (tag/path doesn't exist).
    - The fetched script has no TORCH_CUDA_ARCH_LIST or `case CUDA_VERSION`.
    - bash exits non-zero (e.g. cuda_version isn't in the script's case branches —
      typically means the (torch, cuda) combo never existed).
    """
    cache_key = f"{torch_tag}/{cuda_version}/{arch}"
    if use_cache:
        if _disk_cache is None:
            _disk_cache = _load_disk_cache()
        cached = _disk_cache.get(cache_key)
        if cached and _cache_fresh(cached):
            return {"sass": cached["sass"], "ptx": cached["ptx"]}

    script, _ = fetch_script(torch_tag)
    if script is None:
        return None

    base = re.search(r'^TORCH_CUDA_ARCH_LIST=".*?"', script, re.MULTILINE)
    case = re.search(r"case\s+\$\{?CUDA_VERSION\}?\s+in.*?esac", script, re.DOTALL)
    helpers = re.findall(r"(?:^|\n)(filter_aarch64_archs\s*\(\)\s*\{.*?\n\})", script, re.DOTALL)
    aarch_filter = re.search(
        r'\[\[\s*"\$ARCH"\s*==\s*"aarch64"\s*\]\]\s*&&\s*TORCH_CUDA_ARCH_LIST=.*',
        script,
    )
    if not case and not base:
        return None

    parts = list(helpers)
    if base:
        parts.append(base.group(0))
    if case:
        parts.append(case.group(0))
    if aarch_filter:
        parts.append(aarch_filter.group(0))
    parts.append('echo "ARCHS=$TORCH_CUDA_ARCH_LIST"')
    snippet = "\n".join(parts)

    result = subprocess.run(
        ["bash", "-c", snippet],
        env={"PATH": os.environ["PATH"], "CUDA_VERSION": cuda_version, "ARCH": arch},
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return None
    line = next((l for l in result.stdout.splitlines() if l.startswith("ARCHS=")), None)
    if not line:
        return None
    raw = line[len("ARCHS="):].strip()
    if not raw:
        return None

    info = parse_arch_list_token(raw)
    if use_cache:
        if _disk_cache is None:
            _disk_cache = _load_disk_cache()
        _disk_cache[cache_key] = {
            **info,
            "raw": raw,
            "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _save_disk_cache(_disk_cache)
    return info


if __name__ == "__main__":
    # Smoke test: probe known-good combos
    import sys
    tests = [
        ("v2.4.0", "12.4"),
        ("v2.6.0", "12.6"),
        ("v2.7.0", "12.8"),
        ("v2.8.0", "12.9"),
        ("v2.10.0", "13.0"),
        ("v2.11.0", "13.0"),
    ]
    for tag, cuda in tests:
        info = fetch(tag, cuda, use_cache=False)
        if info:
            sass = " ".join(s.replace("sm_", "") for s in info["sass"])
            ptx = " ".join(s.replace("sm_", "") for s in info["ptx"]) or "—"
            print(f"  {tag:8s} cu{cuda}  sass=[{sass}]  ptx=[{ptx}]")
        else:
            print(f"  {tag:8s} cu{cuda}  FAILED", file=sys.stderr)
