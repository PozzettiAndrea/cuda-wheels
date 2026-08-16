"""Make Hunyuan3D-2.1's custom_rasterizer build on Windows, and rename it.

TWO JOBS.

1. Windows int64 portability. Upstream assumes LP64 (Linux), where `long` is
   64-bit. On MSVC `long` is 32-bit, which breaks the extension two ways:

     * `.data_ptr<long>()` on an int64 tensor -- torch instantiates data_ptr
       for int64_t, and on Windows `long` is a distinct 32-bit type, so there
       is no matching template and the build fails to link/compile.
     * `torch::zeros({ some_container.size(), 9 }, ...)` -- the braced list
       must convert to IntArrayRef (int64_t); size_t -> int64_t inside braces
       is a narrowing conversion, which MSVC rejects as an error.

   This is why no prebuilt Windows wheel of the 2.1 rasterizer exists.

   The fix is not invented here: Tencent already shipped it in the Hunyuan3D
   2.0 tree (hy3dgen/texgen/custom_rasterizer). Diffing 2.0 against 2.1 shows
   these int64_t/static_cast edits are the ONLY difference between the two
   kernel trees. So this patch ports 2.1 to 2.0's already-proven form, and the
   resulting wheel is valid for both families.

2. Rename + slim the distribution. `custom_rasterizer` is far too generic a
   name to occupy in an index, and both Hunyuan3D families claim it. The dist
   becomes `custom_rasterizer_hy3d2`. The pure-Python `custom_rasterizer`
   package is dropped from the wheel so the only thing installed is the
   `custom_rasterizer_kernel` extension -- consumers vendor their own Python
   wrapper anyway (ComfyUI-3D-Pack-enved does, in both HY families).

   The EXTENSION module keeps its name, `custom_rasterizer_kernel`. That is
   deliberate: it is what upstream's render.py does `import
   custom_rasterizer_kernel` on. Renaming it would force every consumer to be
   patched too.

Every substitution below asserts its expected hit count, so if upstream edits
these files the build fails here instead of silently producing a wheel that is
missing the fix.

Run with cwd = repo root (the build action cds into source/ first).
"""

from pathlib import Path

KERNEL = Path("hy3dpaint/custom_rasterizer/lib/custom_rasterizer_kernel")

# (path, [(old, new, expected_occurrences), ...])
EDITS = [
    (
        KERNEL / "grid_neighbor.cpp",
        [
            # Braced IntArrayRef init from size_t -> narrowing on MSVC.
            (
                "torch::zeros({seq2pos.size() / 3, 3}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3), "
                "static_cast<int64_t>(3)}, float_options)",
                2,
            ),
            (
                "torch::zeros({seq2pos.size() / 3}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2pos.size() / 3)}, float_options)",
                2,
            ),
            (
                "torch::zeros({seq2feat.size() / feat_channel, feat_channel}, float_options)",
                "torch::zeros({static_cast<int64_t>(seq2feat.size() / feat_channel), "
                "static_cast<int64_t>(feat_channel)}, float_options)",
                1,
            ),
            (
                "torch::zeros({grids[i].seq2grid.size(), 9}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2grid.size()), "
                "static_cast<int64_t>(9)}, int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].seq2evencorner.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2evencorner.size())}, "
                "int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].seq2oddcorner.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].seq2oddcorner.size())}, "
                "int64_options)",
                2,
            ),
            (
                "torch::zeros({grids[i].downsample_seq.size()}, int64_options)",
                "torch::zeros({static_cast<int64_t>(grids[i].downsample_seq.size())}, "
                "int64_options)",
                2,
            ),
            # 32-bit `long` pointers into 64-bit tensors.
            ("long* nptr", "int64_t* nptr", 2),
            ("long* dptr", "int64_t* dptr", 4),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 8),
        ],
    ),
    (
        KERNEL / "rasterizer.cpp",
        [
            ("(long)maxint", "(int64_t)maxint", 1),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 3),
        ],
    ),
    (
        KERNEL / "rasterizer_gpu.cu",
        [
            ("(long)maxint", "(int64_t)maxint", 1),
            ("data_ptr<long>()", "data_ptr<int64_t>()", 3),
        ],
    ),
    (
        Path("hy3dpaint/custom_rasterizer/setup.py"),
        [
            ('name="custom_rasterizer"', 'name="custom_rasterizer_hy3d2"', 1),
            # Ship only the CUDA extension, not the generic Python package.
            ("packages=find_packages()", "packages=[]", 1),
        ],
    ),
]

for path, edits in EDITS:
    if not path.exists():
        raise SystemExit(f"custom_rasterizer_hy3d2 patch: missing {path}")
    text = path.read_text(encoding="utf-8")
    for old, new, expected in edits:
        found = text.count(old)
        if found != expected:
            raise SystemExit(
                f"custom_rasterizer_hy3d2 patch: {path}: expected {expected} "
                f"occurrence(s) of {old!r}, found {found}. Upstream changed -- "
                f"re-diff against the Hunyuan3D 2.0 tree before building."
            )
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path} ({len(edits)} substitutions)")

# Nothing may still reach for a 32-bit long view of an int64 tensor.
for path, _ in EDITS[:3]:
    leftover = path.read_text(encoding="utf-8").count("data_ptr<long>")
    if leftover:
        raise SystemExit(f"{path}: {leftover} data_ptr<long> site(s) survived")

print("custom_rasterizer -> custom_rasterizer_hy3d2, int64_t port applied")
