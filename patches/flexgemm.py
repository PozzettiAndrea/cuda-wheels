"""Patch FlexGEMM for Windows compatibility.

Fix: Make triton dependency platform-specific (triton on Linux, triton-windows on Windows).
"""
import re
from pathlib import Path

pyproject = Path("pyproject.toml")
content = pyproject.read_text()

# Replace triton dependency with platform-specific versions
# "triton>=X.Y.Z" -> "triton>=X.Y.Z; platform_system != 'Windows'", "triton-windows>=X.Y.Z; platform_system == 'Windows'"
content = re.sub(
    r'"triton(>=[\d.]+)"',
    r'"triton\1; platform_system != \'Windows\'", "triton-windows\1; platform_system == \'Windows\'"',
    content
)

pyproject.write_text(content)
print("Patched pyproject.toml for triton-windows compatibility")
