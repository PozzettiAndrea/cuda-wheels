"""Patch NATTEN for cuda-wheels sequential-checkpoint build.

Identical structure to patches/natten.py except:
  - Renames the package to natten_sequential so wheels don't collide with
    the canonical natten release while we validate the chain mechanism.
  - Drops the autogen shard filter (sequential-checkpoint mode doesn't
    partition .cu files across parallel jobs; it builds the full file set
    sequentially across timeout-bounded jobs).
  - Drops the Windows /FORCE:UNRESOLVED block (chain mode is Linux-only
    for this POC).

Everything else (the TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH shim, the
pyproject.toml fix, the csrc/CMakeLists.txt strip, the helpers.h MSVC
fix, the NATTEN_BUILD_DIR pin) is identical to patches/natten.py.
"""
from pathlib import Path

# pyproject.toml: strip trailing slash from packages.find.where (Windows fix),
# AND rewrite the package name to natten_sequential.
pyproject_file = Path("pyproject.toml")
pyproject_text = pyproject_file.read_text()
old_where = 'where = ["src/"]'
new_where = 'where = ["src"]'
if old_where in pyproject_text:
    pyproject_text = pyproject_text.replace(old_where, new_where, 1)
    print("Patched pyproject.toml: packages.find.where 'src/' -> 'src' (Windows fix)")
# Rename the package in pyproject.toml's [project] table if present.
for old, new in (
    ('name = "natten"', 'name = "natten_sequential"'),
    ('name="natten"', 'name="natten_sequential"'),
):
    if old in pyproject_text:
        pyproject_text = pyproject_text.replace(old, new, 1)
        print(f"Renamed pyproject.toml [project] name: {old} -> {new}")
        break
pyproject_file.write_text(pyproject_text)

# csrc/CMakeLists.txt: strip GCC-only flags forwarded to host compiler that
# MSVC chokes on. Harmless on Linux too.
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
if patched_cmake != cmake_text:
    cmake_file.write_text(patched_cmake)

# Restrict Blackwell autogen .cu files to sm_100/103 only. NATTEN gates these
# files with NATTEN_WITH_BLACKWELL_FNA (a single global flag), not per-arch
# #if __CUDA_ARCH__ guards, so when the flag is on cmake still compiles each
# Blackwell .cu against the target's full CUDA_ARCHITECTURES list. Mirrors
# the block in patches/natten.py.
cmake_text = cmake_file.read_text()
blackwell_restrict_block = '''

# --- cuda-wheels blackwell arch restrict (injected) ---
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

# Restrict Hopper autogen .cu files to sm_90 only. Mirrors patches/natten.py.
cmake_text = cmake_file.read_text()
hopper_restrict_block = '''

# --- cuda-wheels hopper arch restrict (injected) ---
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

# csrc/include/natten/helpers.h: MSVC alt-token fix (no-op on Linux but kept
# for consistency with patches/natten.py in case we extend chain to Windows).
helpers_file = Path("csrc/include/natten/helpers.h")
helpers_text = helpers_file.read_text()
old_check = '(not x.is_sparse(),'
new_check = '(!x.is_sparse(),'
if old_check in helpers_text:
    helpers_file.write_text(helpers_text.replace(old_check, new_check))
    print(f"Patched csrc/include/natten/helpers.h: {old_check!r} -> {new_check!r}")

setup_file = Path("setup.py")
content = setup_file.read_text()

# Rename in setup.py's name="..." field.
for old, new in (
    ('name="natten"', 'name="natten_sequential"'),
    ("name='natten'", "name='natten_sequential'"),
    ('name = "natten"', 'name = "natten_sequential"'),
    ("name = 'natten'", "name = 'natten_sequential'"),
):
    if old in content:
        content = content.replace(old, new, 1)
        print(f"Renamed setup.py: {old} -> {new}")
        break

anchor = 'CUDA_ARCH = os.getenv("NATTEN_CUDA_ARCH", "")'
shim = '''# cuda-wheels shim: bridge TORCH_CUDA_ARCH_LIST -> NATTEN_CUDA_ARCH
# and MAX_JOBS -> NATTEN_N_WORKERS so the cuda-wheels build harness can
# drive NATTEN with its standard env vars. See patches/natten_sequential.py.
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
# Windows: strip 10.0 / 10.3 (Blackwell DC) from the arch list. NATTEN's
# setup.py enables -DNATTEN_WITH_BLACKWELL_FNA=1 when those archs are
# present, which compiles sm100_fmha_bwd_kernel_tma_warpspecialized.hpp
# -- that kernel uses CUTLASS template idioms MSVC's strict mode rejects
# (C2061). RTX 5090 (sm_120 consumer Blackwell) doesn't trigger
# NATTEN_WITH_BLACKWELL_FNA and stays in the arch list.
import platform as _cuw_platform
if _cuw_platform.system() == "Windows":
    _na = os.environ.get("NATTEN_CUDA_ARCH", "")
    _kept = [a for a in _na.split(";") if a.strip() and a.strip() not in ("10.0", "10.3", "100", "103")]
    os.environ["NATTEN_CUDA_ARCH"] = ";".join(_kept)
    print(f"[cuda-wheels] Windows: stripped Blackwell DC archs from NATTEN_CUDA_ARCH; result: {os.environ['NATTEN_CUDA_ARCH']!r}")
# Pin NATTEN_BUILD_DIR to a predictable in-source location so the
# sequential-checkpoint chain can find the .o files across resume.
# NATTEN's setup.py at line 67-68 falls back to a tempdir if the
# directory doesn't exist, so we must create the directory before the
# env var is read.
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
        "not found in setup.py -- upstream may have changed."
    )

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
            # different compile-command strings (csrc/CMakeLists.txt has four
            # execute_process() discovery calls + file(GLOB) over autogen), and
            # ninja's command_hash check then forces a full rebuild regardless
            # of how well our mtime-preservation tricks worked.
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
        "FATAL: cmake configure block not found in NATTEN setup.py -- upstream may have changed. "
        "Re-check against the pinned source_tag."
    )

# Inject -DCMAKE_SUPPRESS_REGENERATION=TRUE into cmake_args. Without this,
# cmake's Ninja generator emits a RERUN_CMAKE edge that ninja itself fires
# at start-of-build (visible as "[0/1] Re-running CMake..." in the log) --
# this re-runs cmake configure AS A NINJA EDGE, regenerating build.ninja
# with different command_hashes. Even though our setup.py patch above
# skips the EXPLICIT cmake configure call on resume, the implicit one
# inside ninja still fires. CMAKE_SUPPRESS_REGENERATION=TRUE tells cmake
# to never emit that edge in the first place. Safe to set on link-0 too
# (build.ninja just won't auto-regenerate if CMakeLists.txt changes mid-run,
# which isn't a thing we do).
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
