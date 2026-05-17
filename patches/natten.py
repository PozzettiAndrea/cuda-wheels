"""Patch NATTEN for cuda-wheels build compatibility.

NATTEN's setup.py reads its own env vars (NATTEN_CUDA_ARCH, NATTEN_N_WORKERS)
and ignores PyTorch/cuda-wheels conventions (TORCH_CUDA_ARCH_LIST, MAX_JOBS).
This patch injects a small shim near the top of setup.py that translates
between the two so cuda-wheels' standard build env "just works":

  1. If NATTEN_CUDA_ARCH is unset, fall back to TORCH_CUDA_ARCH_LIST. Strip
     the '+PTX' suffix (NATTEN's parser doesn't accept it) and normalize
     space-separated to semicolon-separated (NATTEN's expected separator,
     per get_cuda_arch_list() at setup.py).
  2. If NATTEN_N_WORKERS is unset, fall back to MAX_JOBS. Without this,
     NATTEN defaults to cpu_count()//4 which is unrelated to the cuda-wheels
     max_jobs cap — and CUTLASS template instantiations need the cap.
"""
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

anchor = 'CUDA_ARCH = os.getenv("NATTEN_CUDA_ARCH", "")'
shim = '''# cuda-wheels shim: bridge TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH
# and MAX_JOBS -> NATTEN_N_WORKERS so the cuda-wheels build harness can
# drive NATTEN with its standard env vars. See patches/natten.py.
if not os.getenv("NATTEN_CUDA_ARCH"):
    _torch_arch = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    _parts = [p.replace("+PTX", "").strip() for p in _torch_arch.replace(";", " ").split()]
    _parts = [p for p in _parts if p]
    if _parts:
        os.environ["NATTEN_CUDA_ARCH"] = ";".join(_parts)
if not os.getenv("NATTEN_N_WORKERS"):
    _mj = os.getenv("MAX_JOBS", "")
    if _mj.isdigit() and int(_mj) > 0:
        os.environ["NATTEN_N_WORKERS"] = _mj
''' + anchor

if anchor in content:
    content = content.replace(anchor, shim, 1)
    print("Patched setup.py: TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH and MAX_JOBS -> NATTEN_N_WORKERS shim inserted")
else:
    raise SystemExit(
        "FATAL: anchor 'CUDA_ARCH = os.getenv(\"NATTEN_CUDA_ARCH\", \"\")' "
        "not found in setup.py -- upstream may have changed. Re-check the "
        "patch against the pinned source_tag."
    )

setup_file.write_text(content)
