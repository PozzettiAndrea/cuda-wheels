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
  3. On Windows, strip 10.0 / 10.3 (Blackwell DC) from the arch list.
     NATTEN's setup.py enables -DNATTEN_WITH_BLACKWELL_FNA=1 when those
     archs are present, which compiles `sm100_fmha_bwd_kernel_tma_warpspecialized.hpp`.
     That kernel uses CUTLASS template idioms MSVC's strict mode rejects
     (C2061 syntax error: identifier 'PipelineState'). Stripping the archs
     skips the Blackwell-DC code path entirely. Windows users with RTX 5090
     (sm_120 consumer Blackwell) are still covered because sm_120 doesn't
     trigger NATTEN_WITH_BLACKWELL_FNA — it compiles via the regular
     CUTLASS-FNA path.

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

# Windows shard mode: MSVC linker fails the partial-link in each shard
# with "LNK1120: 2709 unresolved externals" because only ~1/16 of the .o
# files are present per shard. Linux's ld is permissive enough that the
# partial .so still gets emitted (we discard it anyway -- only the .o
# files matter from a shard). MSVC needs /FORCE:UNRESOLVED to tell it
# to produce the .pyd despite unresolved symbols.
#
# In the link job, CUDA_WHEELS_SHARD_COUNT is not exported (the
# build-wheel action only sets it in compile-shard mode), so this
# conditional is a no-op there -- the link runs strict-linked with all
# .o files present and no unresolved externals.
cmake_text = cmake_file.read_text()
shard_link_block = '''

# --- cuda-wheels Windows shard mode (injected) ---
# Tell MSVC's linker to ignore unresolved externals during a compile-shard
# build (env var CUDA_WHEELS_SHARD_COUNT set). The resulting .pyd is
# discarded; only the .o files are uploaded to the link job.
if(${NATTEN_IS_WINDOWS} AND DEFINED ENV{CUDA_WHEELS_SHARD_COUNT})
    target_link_options(natten PRIVATE "/FORCE:UNRESOLVED")
    message(STATUS "cuda-wheels Windows shard mode: /FORCE:UNRESOLVED enabled")
endif()
# --- end cuda-wheels Windows shard mode ---
'''
if 'cuda-wheels Windows shard mode' not in cmake_text:
    cmake_file.write_text(cmake_text + shard_link_block)
    print("Appended Windows-shard-mode /FORCE:UNRESOLVED block to csrc/CMakeLists.txt")
else:
    print("NOTE: /FORCE:UNRESOLVED block already present in csrc/CMakeLists.txt -- skipping")

# Restrict Blackwell autogen .cu files to sm_100/103 only. NATTEN gates these
# files with NATTEN_WITH_BLACKWELL_FNA (a single global flag), not per-arch
# #if __CUDA_ARCH__ guards, so when the flag is on cmake still compiles each
# Blackwell .cu against the target's full CUDA_ARCHITECTURES list (sm_80, 86,
# 89, 90, 100, 120, etc.). That's ~5x wasted nvcc template-instantiation work
# per Blackwell file -- nvcc parses + emits CUTLASS template stubs for archs
# that the source's #ifdefs make empty. Setting per-file CUDA_ARCHITECTURES
# overrides the target-global list for just those sources.
cmake_text = cmake_file.read_text()
blackwell_restrict_block = '''

# --- cuda-wheels blackwell arch restrict (injected) ---
# Compile Blackwell autogen .cu files only for sm_100/103. They're gated
# by NATTEN_WITH_BLACKWELL_FNA (global flag) rather than per-arch #ifdefs,
# so emitting code for sm_80/86/89/90/120 is pure wasted nvcc work.
if(${NATTEN_WITH_BLACKWELL_FNA})
    set_source_files_properties(
        ${AUTOGEN_BLACKWELL_FNA} ${AUTOGEN_BLACKWELL_FMHA}
        PROPERTIES CUDA_ARCHITECTURES "100;103"
    )
    message(STATUS "cuda-wheels: Blackwell sources restricted to sm_100/103")
endif()
# --- end cuda-wheels blackwell arch restrict ---
'''
if 'cuda-wheels blackwell arch restrict' not in cmake_text:
    cmake_file.write_text(cmake_text + blackwell_restrict_block)
    print("Appended Blackwell arch-restrict block to csrc/CMakeLists.txt")
else:
    print("NOTE: Blackwell arch-restrict block already present in csrc/CMakeLists.txt -- skipping")

# Restrict Hopper autogen .cu files to sm_90 only. Same mechanism as the
# Blackwell block above: NATTEN gates Hopper kernels with the global
# NATTEN_WITH_HOPPER_FNA flag, so nvcc compiles each Hopper .cu for every
# arch in the target's CUDA_ARCHITECTURES list even though only sm_90 emits
# useful code. Impact is wider than Blackwell because every NATTEN matrix
# cell includes sm_90: cu12.4/12.6 builds (no Blackwell at all) save 75%
# per Hopper file, cu12.8/12.9 save 83%, cu13.0 saves 86%.
# Runtime safety verified: can_run_cutlass_hopper_fna/fmha in
# src/natten/backends/configs/checks.py reject device_cc != 90, and
# csrc/src/hopper_fna_forward.cu re-asserts TORCH_CHECK(cc == 90, ...) at
# the host entry. Host symbols unreachable on non-Hopper GPUs.
cmake_text = cmake_file.read_text()
hopper_restrict_block = '''

# --- cuda-wheels hopper arch restrict (injected) ---
# Compile Hopper autogen .cu files only for sm_90. Same logic as the
# Blackwell-restrict block above.
if(${NATTEN_WITH_HOPPER_FNA})
    set_source_files_properties(
        ${AUTOGEN_HOPPER_FNA} ${AUTOGEN_HOPPER_FMHA}
        PROPERTIES CUDA_ARCHITECTURES "90"
    )
    message(STATUS "cuda-wheels: Hopper sources restricted to sm_90")
endif()
# --- end cuda-wheels hopper arch restrict ---
'''
if 'cuda-wheels hopper arch restrict' not in cmake_text:
    cmake_file.write_text(cmake_text + hopper_restrict_block)
    print("Appended Hopper arch-restrict block to csrc/CMakeLists.txt")
else:
    print("NOTE: Hopper arch-restrict block already present in csrc/CMakeLists.txt -- skipping")

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
# Windows: strip Blackwell DC archs (10.0, 10.3) from NATTEN_CUDA_ARCH to
# avoid enabling NATTEN_WITH_BLACKWELL_FNA and pulling in the
# sm100_fmha_bwd_kernel_tma_warpspecialized.hpp template that MSVC's
# strict mode rejects with C2061. RTX 5090 (sm_120) is unaffected and
# stays in the arch list (it doesn't trigger the Blackwell DC code path).
import platform as _cuw_platform
if _cuw_platform.system() == "Windows":
    _na = os.environ.get("NATTEN_CUDA_ARCH", "")
    _kept = [a for a in _na.split(";") if a.strip() and a.strip() not in ("10.0", "10.3", "100", "103")]
    os.environ["NATTEN_CUDA_ARCH"] = ";".join(_kept)
    print(f"[cuda-wheels] Windows: stripped Blackwell DC archs from NATTEN_CUDA_ARCH; result: {os.environ['NATTEN_CUDA_ARCH']!r}")
# Pin NATTEN_BUILD_DIR to a predictable in-source location so the cuda-wheels
# shard/link harness can find the .o files. Default is a temporary directory
# whose name changes per run, which doesn't survive the upload/restore
# handoff between shard compile jobs and the downstream link job.
# IMPORTANT: NATTEN's setup.py at line 67-68 falls back to the tempdir if
# the directory doesn't exist:
#   if not os.path.isdir(BUILD_DIR):
#       BUILD_DIR = tmp_dir.name
# so we must create the directory before the env var is read.
if not os.getenv("NATTEN_BUILD_DIR"):
    _cuw_natten_build_dir = os.path.abspath("build/natten_cmake")
    os.makedirs(_cuw_natten_build_dir, exist_ok=True)
    os.environ["NATTEN_BUILD_DIR"] = _cuw_natten_build_dir
    print(f"[cuda-wheels] NATTEN_BUILD_DIR set to {_cuw_natten_build_dir}")
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

# Sequential-checkpoint: skip cmake configure on resume. Proven on
# natten_sequential (run 26052223468) -- combined with CMAKE_SUPPRESS_REGENERATION
# below and the action.yml's sudo-touch of CUDA targets/ headers, ninja
# correctly resumes the build mid-chain. Without this patch, NATTEN's setup.py
# always runs cmake configure, which regenerates build.ninja with subtly
# different command strings and trips ninja's command_hash check.
old_cmake_pair = '''            # Config and build the extension
            subprocess.check_call(
                ["cmake", cmake_lists_dir] + cmake_args, cwd=build_dir
            )
            cmake_build_args = [
                "--build",
                build_dir,
                "-j",
                str(N_WORKERS),
            ]
            if VERBOSE:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])'''

new_cmake_pair = '''            # cuda-wheels sequential-checkpoint: skip the cmake configure call
            # on resume so the restored build.ninja is honored byte-identically
            # by ninja. A fresh configure re-emits build.ninja with subtly
            # different compile-command strings, forcing a full rebuild via
            # the command_hash check.
            _cuw_cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
            _cuw_build_ninja = os.path.join(build_dir, "build.ninja")
            if os.path.isfile(_cuw_cmake_cache) and os.path.isfile(_cuw_build_ninja):
                print(f"[cuda-wheels] reusing CMakeCache.txt + build.ninja in {build_dir}; skipping configure")
            else:
                subprocess.check_call(
                    ["cmake", cmake_lists_dir] + cmake_args, cwd=build_dir
                )
            cmake_build_args = [
                "--build",
                build_dir,
                "-j",
                str(N_WORKERS),
            ]
            if VERBOSE:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])'''

if old_cmake_pair in content:
    content = content.replace(old_cmake_pair, new_cmake_pair, 1)
    print("Patched setup.py: skip cmake configure on resume when CMakeCache.txt + build.ninja exist")
else:
    raise SystemExit(
        "FATAL: cmake configure block not found in NATTEN setup.py -- upstream may have changed."
    )

# Inject -DCMAKE_SUPPRESS_REGENERATION=TRUE into cmake_args. Without it,
# cmake's Ninja generator emits a RERUN_CMAKE edge that ninja itself fires
# on chain-link resume (visible as "[0/1] Re-running CMake..."), re-running
# cmake configure as a ninja edge -- defeating the setup.py-level skip above.
old_cmake_args = '''            cmake_args = [
                f"-DPYTHON_PATH={sys.executable}",'''

new_cmake_args = '''            cmake_args = [
                "-DCMAKE_SUPPRESS_REGENERATION=TRUE",
                f"-DPYTHON_PATH={sys.executable}",'''

if old_cmake_args in content:
    content = content.replace(old_cmake_args, new_cmake_args, 1)
    print("Patched setup.py: injected -DCMAKE_SUPPRESS_REGENERATION=TRUE into cmake_args")
else:
    raise SystemExit(
        "FATAL: cmake_args initializer not found in NATTEN setup.py -- upstream may have changed."
    )

setup_file.write_text(content)
