"""Patch flash-attn setup.py to read TORCH_CUDA_ARCH_LIST for architecture selection.

flash-attn reads FLASH_ATTN_CUDA_ARCHS (semicolon-separated, no dots, e.g. "80;90")
but the build action sets TORCH_CUDA_ARCH_LIST (space-separated, with dots, e.g. "8.0 9.0").
This patch bridges the two formats.
"""
import re
from pathlib import Path

setup_file = Path("setup.py")
content = setup_file.read_text()

# Replace cuda_archs() to also read TORCH_CUDA_ARCH_LIST
old_func = '''def cuda_archs() -> str:
    return os.getenv("FLASH_ATTN_CUDA_ARCHS", "80;90;100;120").split(";")'''

new_func = '''def cuda_archs() -> str:
    archs = os.getenv("FLASH_ATTN_CUDA_ARCHS")
    if archs:
        return archs.split(";")
    torch_archs = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    if torch_archs:
        # Convert "8.0 9.0 10.0 12.0" -> ["80", "90", "100", "120"]
        return [a.replace(".", "") for a in torch_archs.split()]
    return ["80", "90", "100", "120"]'''

if old_func in content:
    content = content.replace(old_func, new_func)
    print("Patched cuda_archs() to read TORCH_CUDA_ARCH_LIST")
else:
    print("WARNING: Could not find cuda_archs() function - source may have changed")

setup_file.write_text(content)
