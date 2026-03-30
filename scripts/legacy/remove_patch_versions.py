#!/usr/bin/env python3
"""Remove patch release entries (2.4.1, 2.5.1, 2.7.1, 2.9.1) from all package YAMLs.

Removes the entire combination block (including comments on the same line)
for patch pytorch versions. Preserves all other formatting.
"""
import re
from pathlib import Path

PACKAGES_DIR = Path(__file__).parent.parent / "packages"
PATCH_VERSIONS = {"2.4.1", "2.5.1", "2.7.1", "2.9.1"}


def remove_patch_entries(text):
    """Remove combination blocks for patch pytorch versions."""
    lines = text.split("\n")
    result = []
    skip_block = False
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start of a combination entry: "    - cuda: ..."
        if re.match(r'\s+- cuda:', line):
            # Look ahead for the pytorch line
            block = [line]
            j = i + 1
            is_patch = False
            while j < len(lines) and not re.match(r'\s+- cuda:', lines[j]) and not lines[j].strip().startswith("# ──"):
                block.append(lines[j])
                if re.match(r'\s+pytorch:\s+"(\d+\.\d+\.\d+)"', lines[j]):
                    ver = re.match(r'\s+pytorch:\s+"(\d+\.\d+\.\d+)"', lines[j]).group(1)
                    if ver in PATCH_VERSIONS:
                        is_patch = True
                # Stop at next combination or section
                if j + 1 < len(lines) and (re.match(r'\s+- cuda:', lines[j + 1]) or re.match(r'\s+platforms:', lines[j + 1])):
                    break
                j += 1

            if is_patch:
                # Skip this entire block
                i = j + 1
                continue
            else:
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def main():
    for yml in sorted(PACKAGES_DIR.glob("*.yml")):
        text = yml.read_text()
        new_text = remove_patch_entries(text)
        if new_text != text:
            old_count = text.count("pytorch:")
            new_count = new_text.count("pytorch:")
            removed = old_count - new_count
            yml.write_text(new_text)
            print(f"{yml.name}: removed {removed} patch entries")
        else:
            print(f"{yml.name}: no changes")


if __name__ == "__main__":
    main()
