"""Patch cumm to add bfloat16 GEMM kernel support.

cumm has all the low-level bf16 primitives (MMA instructions, tensor ops,
dtype definitions, numeric converters) but never instantiates bf16 GEMM
kernels. This patch adds bf16 Ampere GEMM params to cumm/gemm/main.py.

Also forces package name to 'cumm' (not 'cumm-cu{version}').

bf16 tensor core MMA instructions require sm_80+ (Ampere).
"""
import re
from pathlib import Path

# ─── 0. Force package name to 'cumm' (ignore CUMM_CUDA_VERSION) ───
setup_py = Path("setup.py")
if setup_py.exists():
    setup_content = setup_py.read_text()
    # Neutralize the CUDA version suffix logic:
    #   cuda_ver = os.getenv("CUMM_CUDA_VERSION", None)
    # Replace with always-None so RELEASE_NAME stays as 'cumm'
    setup_content = setup_content.replace(
        'os.getenv("CUMM_CUDA_VERSION"',
        'os.getenv("_DISABLED_CUMM_CUDA_VERSION"'
    )
    setup_py.write_text(setup_content)
    print("Patched setup.py: forced package name to 'cumm'")

main_py = Path("cumm/gemm/main.py")
content = main_py.read_text()

# ─── 1. Add SHUFFLE_AMPERE_PARAMS with bf16 GEMM kernels ───
# Insert after SHUFFLE_TURING_PARAMS definition.
# cumm's gen_shuffle_params signature (NOT gen_shuffle_params_v2):
#   gen_shuffle_params(ts, wts, dss: List[str], stage: int, algo, tensorop)

AMPERE_PARAMS_BLOCK = '''
# bf16 Ampere GEMM kernels (sm_80+)
# TensorOp((16, 8, 16)) is the Ampere-optimized shape for 16-bit types.
# Uses f32 accumulator for numerical stability (bf16 accumulation not supported by hardware).
SHUFFLE_AMPERE_PARAMS: List[GemmAlgoParams] = [
    *gen_shuffle_params(
        (64, 64, 32),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 128, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 64, 64),
        (32, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 64),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 256, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (256, 128, 32),
        (64, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (128, 64, 32),
        (64, 32, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
    *gen_shuffle_params(
        (64, 128, 32),
        (32, 64, 32), ["bf16,bf16,bf16,f32,f32"], 2,
        kernel.GemmAlgo.Ampere, TensorOp((16, 8, 16))),
]

'''

# Find the end of SHUFFLE_TURING_PARAMS and insert AMPERE_PARAMS after it
# The list ends with a "]" line that is NOT inside a gen_shuffle_params call
# Look for the pattern: end of SHUFFLE_TURING_PARAMS list followed by a blank line
turing_match = re.search(
    r'(SHUFFLE_TURING_PARAMS\s*:.*?\n(?:.*\n)*?^]\s*$)',
    content,
    re.MULTILINE
)
if not turing_match:
    raise RuntimeError("Could not find SHUFFLE_TURING_PARAMS in cumm/gemm/main.py")

insert_pos = turing_match.end()
content = content[:insert_pos] + '\n' + AMPERE_PARAMS_BLOCK + content[insert_pos:]

# ─── 2. Populate the empty ampere_params list in GemmMainUnitTest.__init__ ───
# The non-debug branch has:
#     ampere_params = [
#
#     ]
# Replace with:
#     ampere_params = [
#         *SHUFFLE_AMPERE_PARAMS,
#     ]

content = re.sub(
    r'(ampere_params\s*=\s*\[)\s*\n\s*(\])',
    r'\1\n                    *SHUFFLE_AMPERE_PARAMS,\n                \2',
    content,
    count=1
)

main_py.write_text(content)
print("Patched cumm/gemm/main.py with bf16 Ampere GEMM params")
print("  - Added SHUFFLE_AMPERE_PARAMS with 9 tile configs (bf16 in/out, f32 acc)")
print("  - Populated ampere_params in GemmMainUnitTest.__init__")
