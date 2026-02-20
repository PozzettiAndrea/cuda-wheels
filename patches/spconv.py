"""Patch spconv to add bfloat16 sparse convolution support.

Adds bf16 GEMM shuffle params (Ampere TensorOp + Simt fallback) and implicit
GEMM conv params to spconv/core.py. Requires a cumm build that includes bf16
GEMM kernels (see patches/cumm.py).

Also forces package name to 'spconv' (not 'spconv-cu{version}').

spconv uses gen_shuffle_params_v2 (imported as gen_shuffle_params) which has
an extra ds_for_sab parameter compared to cumm's gen_shuffle_params.
"""
import re
from pathlib import Path

# ─── 0. Force package name to 'spconv' (ignore CUMM_CUDA_VERSION) ───
setup_py = Path("setup.py")
if setup_py.exists():
    setup_content = setup_py.read_text()
    # Neutralize the CUDA version suffix logic and cumm dep renaming:
    #   cuda_ver = os.environ.get("CUMM_CUDA_VERSION", "")
    # Replace env var name so it's never found
    setup_content = setup_content.replace(
        '"CUMM_CUDA_VERSION"',
        '"_DISABLED_CUMM_CUDA_VERSION"'
    )
    setup_py.write_text(setup_content)
    print("Patched setup.py: forced package name to 'spconv'")

core_py = Path("spconv/core.py")
content = core_py.read_text()

# ─── 1a. Add bf16 Simt fallback entries to SHUFFLE_SIMT_PARAMS ───
# Simt kernels handle unaligned dimensions (e.g. 9-channel input where LDA=9).
# Without these, bf16 fails on the first conv layer since TensorOp requires
# LDA alignment to 8 elements. Mirrors existing f16 Simt fallback entries.

BF16_SIMT_FALLBACK = '''
    # bf16 Simt fallback kernels for misaligned dimensions
    *gen_shuffle_params((128, 128, 8), (32, 64, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((32, 64, 32), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((32, 32, 32), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((64, 128, 16), (32, 64, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
    *gen_shuffle_params((64, 64, 8), (32, 32, 8), ["bf16,bf16,bf16,f32,f32"],
                        "bf16,bf16,bf16,f32,f32", 2, kernel.GemmAlgo.Simt, None),
'''

# Insert before the closing ] of SHUFFLE_SIMT_PARAMS
simt_match = re.search(
    r'(SHUFFLE_SIMT_PARAMS\s*:.*?\n(?:.*\n)*?)(^\]\s*$)',
    content,
    re.MULTILINE
)
if simt_match:
    insert_pos = simt_match.start(2)
    content = content[:insert_pos] + BF16_SIMT_FALLBACK + content[insert_pos:]
    print("Added bf16 Simt fallback params to SHUFFLE_SIMT_PARAMS")
else:
    print("WARNING: Could not find SHUFFLE_SIMT_PARAMS closing bracket")

# ─── 1. Replace empty SHUFFLE_AMPERE_PARAMS with bf16 entries ───
# spconv's gen_shuffle_params is actually gen_shuffle_params_v2 with signature:
#   gen_shuffle_params_v2(ts, wts, dss: List[str], ds_for_sab, stage, algo, tensorop)

BF16_SHUFFLE_PARAMS = '''SHUFFLE_AMPERE_PARAMS: List[GemmAlgoParams] = [
    # bf16 with f32 accumulator - Ampere TensorOp (16, 8, 16)
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], "bf16,bf16,bf16,f32,f32", 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
]'''

# Replace the entire SHUFFLE_AMPERE_PARAMS block (including commented-out entries).
# Use MULTILINE so ^ matches line starts — match up to a ] on its own line
# (the closing bracket), not ] characters inside commented-out code.
content = re.sub(
    r'SHUFFLE_AMPERE_PARAMS\s*(?::\s*List\[GemmAlgoParams\]\s*)?=\s*\[.*?^\]',
    BF16_SHUFFLE_PARAMS,
    content,
    count=1,
    flags=re.DOTALL | re.MULTILINE
)

# ─── 2. Add bf16 entries to IMPLGEMM_AMPERE_PARAMS ───
# Add bf16 conv params alongside existing f16/tf32 entries.
# Uses gen_conv_params with ConvFwdAndBwdInput and ConvBwdWeight.
# Pattern mirrors existing f16 Ampere entries but with bf16 dtypes.

BF16_CONV_PARAMS = '''
    # bf16 FwdAndBwdInput - Ampere TensorOp (16, 8, 16)
    *gen_conv_params(ConvFwdAndBwdInput, (64, 64, 32), (32, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvFwdAndBwdInput, (64, 128, 32), (32, 64, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvFwdAndBwdInput, (128, 64, 32), (64, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    # bf16 BwdWeight - Ampere TensorOp (16, 8, 16)
    *gen_conv_params(ConvBwdWeight, (64, 64, 32), (32, 32, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4, 5], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
    *gen_conv_params(ConvBwdWeight, (64, 128, 32), (32, 64, 32),
                     NDIM_DONT_CARE,
                     ConvIterAlgo.Optimized,
                     [2, 3, 4, 5], ["bf16,bf16,bf16,f32,f32"],
                     NHWC, NHWC, NHWC,
                     GemmAlgo.Ampere,
                     TensorOp((16, 8, 16)),
                     mask_sparse=True,
                     increment_k_first=True,
                     access_per_vector=1),
'''

# Insert bf16 conv params at the end of IMPLGEMM_AMPERE_PARAMS, before the
# int8 inference section. Find the first s8 entry (int8) in the list and
# insert before it.
s8_match = re.search(
    r'(\n\s*# (?:always loaded|s8|int8).*?\n\s*\*gen_conv_params\(ConvFwdAndBwdInput.*?"s8)',
    content,
    re.DOTALL | re.IGNORECASE
)
if s8_match:
    insert_pos = s8_match.start()
    content = content[:insert_pos] + BF16_CONV_PARAMS + content[insert_pos:]
else:
    # Fallback: insert before the closing ] of IMPLGEMM_AMPERE_PARAMS
    # Find the end of the list by looking for the ] after the last gen_conv_params
    implgemm_match = re.search(
        r'(IMPLGEMM_AMPERE_PARAMS\s*(?::\s*List\[GemmAlgoParams\]\s*)?=\s*\[)',
        content
    )
    if implgemm_match:
        # Find the matching closing bracket
        start = implgemm_match.end()
        depth = 1
        pos = start
        while depth > 0 and pos < len(content):
            if content[pos] == '[':
                depth += 1
            elif content[pos] == ']':
                depth -= 1
            pos += 1
        # Insert before the closing ]
        content = content[:pos-1] + BF16_CONV_PARAMS + content[pos-1:]
        print("  - Inserted bf16 conv params at end of IMPLGEMM_AMPERE_PARAMS (fallback)")
    else:
        print("WARNING: Could not find IMPLGEMM_AMPERE_PARAMS to insert bf16 conv params")

core_py.write_text(content)
print("Patched spconv/core.py with bf16 support")
print("  - Added bf16 Simt fallback to SHUFFLE_SIMT_PARAMS (5 tile configs)")
print("  - Added bf16 SHUFFLE_AMPERE_PARAMS (7 tile configs)")
print("  - Added bf16 IMPLGEMM_AMPERE_PARAMS (3 FwdBwd + 2 BwdWeight)")

# ─── 3. Add bf16 to _TORCH_DTYPE_TO_TV in cppcore.py ───
cppcore_py = Path("spconv/pytorch/cppcore.py")
cppcore_content = cppcore_py.read_text()

cppcore_content = cppcore_content.replace(
    "torch.float16: tv.float16,",
    "torch.float16: tv.float16,\n    torch.bfloat16: tv.bfloat16,"
)

cppcore_py.write_text(cppcore_content)
print("Patched spconv/pytorch/cppcore.py with bf16 dtype mapping")
