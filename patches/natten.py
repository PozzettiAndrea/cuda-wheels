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

Also patches pyproject.toml: setuptools.packages.find.where = ["src/"] has
a trailing slash that newer setuptools' convert_path rejects on Windows
("path 'src/' cannot end with '/'"), failing metadata generation before
the build even starts. Linux's convert_path is permissive and ignores it.
"""
from pathlib import Path

# pyproject.toml: strip trailing slash from packages.find.where (Windows fix).
pyproject_file = Path("pyproject.toml")
pyproject_text = pyproject_file.read_text()
old_where = 'where = ["src/"]'
new_where = 'where = ["src"]'
if old_where in pyproject_text:
    pyproject_file.write_text(pyproject_text.replace(old_where, new_where, 1))
    print("Patched pyproject.toml: packages.find.where 'src/' -> 'src' (Windows fix)")
else:
    print("NOTE: pyproject.toml didn't contain 'where = [\"src/\"]' -- skipping (may already be fixed upstream)")

# csrc/CMakeLists.txt: strip GCC-only flags forwarded to host compiler that
# MSVC chokes on. `-Wconversion` triggers cl error D8021 ("invalid numeric
# argument '/Wconversion'") because MSVC parses '-W<digit>' as warning level.
# `-fno-strict-aliasing` is a GCC aliasing knob with no MSVC equivalent;
# MSVC errors out the same way. Neither is load-bearing for libnatten
# correctness — they're a non-critical warning + a GCC-specific optimizer
# safety hint. Strip both unconditionally; behavior on Linux is unchanged
# in any way that matters for the built kernels.
cmake_file = Path("csrc/CMakeLists.txt")
cmake_text = cmake_file.read_text()
patched_cmake = cmake_text
for line in (
    'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")',
    'set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")',
):
    if line in patched_cmake:
        patched_cmake = patched_cmake.replace(line + "\n", "", 1)
        print(f"Stripped from csrc/CMakeLists.txt: {line}")
    else:
        print(f"NOTE: csrc/CMakeLists.txt didn't contain {line!r} -- skipping (may already be removed upstream)")
if patched_cmake != cmake_text:
    cmake_file.write_text(patched_cmake)

# csrc/include/natten/helpers.h: CHECK_CONTIGUOUS uses the C++ alternative
# token `not` (`TORCH_CHECK(not x.is_sparse(), ...)`). GCC/Clang accept this
# without <ciso646>; MSVC errors with `identifier "not" is undefined` unless
# /permissive- or /Za is set. Replace with the standard `!` operator — this
# is a one-liner in the macro definition, but the macro expands inside other
# helpers.h checks (lines ~325, ~366 in the build error), so every consumer
# is fixed by the single substitution.
helpers_file = Path("csrc/include/natten/helpers.h")
helpers_text = helpers_file.read_text()
old_check = '(not x.is_sparse(),'
new_check = '(!x.is_sparse(),'
if old_check in helpers_text:
    helpers_file.write_text(helpers_text.replace(old_check, new_check))
    print(f"Patched csrc/include/natten/helpers.h: replaced {old_check!r} with {new_check!r} (MSVC fix)")
else:
    print("NOTE: csrc/include/natten/helpers.h didn't contain 'not x.is_sparse(),' -- skipping")

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
# Pin NATTEN_BUILD_DIR to a predictable in-source location so the cuda-wheels
# shard/link harness can find the .o files. Default is a temporary directory
# whose name changes per run, which doesn't survive the upload/restore
# handoff between shard compile jobs and the downstream link job.
if not os.getenv("NATTEN_BUILD_DIR"):
    os.environ["NATTEN_BUILD_DIR"] = os.path.abspath("build/natten_cmake")
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

# Shard filter for cmake-style build:
# Inject a post-autogen filter that deletes .cu files NOT in this shard's
# slice, gated on CUDA_WHEELS_SHARD_INDEX/COUNT env vars (set by the
# build-wheel action in compile-shard mode). The shard then compiles only
# its slice (1/N of the 144 autogen .cu files). When the env vars are
# unset (link-only mode, full-mode), this is a no-op and natten builds
# normally.
#
# Sorted round-robin partition: each .cu file at sorted-index i goes to
# shard (i % shard_count) + 1. Deterministic across shards (each shard sees
# the same sort order from autogen's deterministic output).
autogen_anchor = """            autogen_kernel_instantitations(
                this_dir=this_dir,
                autogen_dir=autogen_dir,
                scripts_dir=scripts_dir,
                policy=AUTOGEN_POLICY,
                cuda_arch_list=cuda_arch_list,
            )"""

autogen_filter = autogen_anchor + """

            # cuda-wheels shard filter: when CUDA_WHEELS_SHARD_COUNT > 0,
            # delete .cu files not in this shard's slice so cmake only
            # builds 1/N of the autogen output. See patches/natten.py.
            _cuw_shard_count = int(os.environ.get('CUDA_WHEELS_SHARD_COUNT', '0'))
            if _cuw_shard_count > 0:
                import glob
                _cuw_shard_index = int(os.environ.get('CUDA_WHEELS_SHARD_INDEX', '1'))
                _cuw_pattern = path.join(autogen_dir, 'src', 'cuda', '**', '*.cu')
                _cuw_all = sorted(glob.glob(_cuw_pattern, recursive=True))
                _cuw_kept = [f for i, f in enumerate(_cuw_all)
                             if i % _cuw_shard_count == _cuw_shard_index - 1]
                _cuw_to_delete = set(_cuw_all) - set(_cuw_kept)
                for _f in _cuw_to_delete:
                    os.remove(_f)
                print(f'[cuda-wheels natten shard {_cuw_shard_index}/{_cuw_shard_count}] '
                      f'kept {len(_cuw_kept)}/{len(_cuw_all)} autogen .cu files; '
                      f'deleted {len(_cuw_to_delete)}')
"""

if autogen_anchor in content:
    content = content.replace(autogen_anchor, autogen_filter, 1)
    print("Patched setup.py: autogen shard filter injected after autogen_kernel_instantitations() call")
else:
    raise SystemExit(
        "FATAL: anchor for autogen call not found in setup.py -- upstream may have changed. "
        "Re-check the patch against the pinned source_tag."
    )

setup_file.write_text(content)
