#!/usr/bin/env bash
set -euo pipefail

# Fix METADATA version in all existing release wheels.
# Downloads each release's wheels, patches metadata, re-uploads.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="/tmp/fix-wheel-metadata"
PATCH_SCRIPT="$SCRIPT_DIR/patch_wheel_version.py"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# Get all release tags
TAGS=$(gh release list --limit 100 --json tagName -q '.[].tagName')

total_fixed=0
total_skipped=0
total_failed=0

for tag in $TAGS; do
    echo "============================================"
    echo "Processing release: $tag"
    echo "============================================"

    dl_dir="$WORK_DIR/$tag/download"
    fixed_dir="$WORK_DIR/$tag/fixed"
    mkdir -p "$dl_dir" "$fixed_dir"

    # Download all wheels from this release
    echo "Downloading wheels..."
    gh release download "$tag" -p "*.whl" -D "$dl_dir" --skip-existing

    wheel_count=$(ls "$dl_dir"/*.whl 2>/dev/null | wc -l)
    echo "Downloaded $wheel_count wheels"

    if [ "$wheel_count" -eq 0 ]; then
        echo "No wheels found, skipping"
        continue
    fi

    # Copy to fixed dir and patch
    cp "$dl_dir"/*.whl "$fixed_dir/"
    echo "Fixing metadata..."
    python3 "$PATCH_SCRIPT" "$fixed_dir"

    # Find which wheels were actually modified (compare checksums)
    upload_wheels=()
    for whl in "$fixed_dir"/*.whl; do
        fname=$(basename "$whl")
        orig="$dl_dir/$fname"
        if ! cmp -s "$whl" "$orig"; then
            upload_wheels+=("$whl")
        else
            total_skipped=$((total_skipped + 1))
        fi
    done

    if [ ${#upload_wheels[@]} -eq 0 ]; then
        echo "No wheels needed fixing in $tag"
        continue
    fi

    echo "Uploading ${#upload_wheels[@]} fixed wheels..."
    gh release upload "$tag" "${upload_wheels[@]}" --clobber
    total_fixed=$((total_fixed + ${#upload_wheels[@]}))
    echo "Done with $tag: fixed ${#upload_wheels[@]} wheels"

    # Clean up to save disk space
    rm -rf "$WORK_DIR/$tag"
done

echo ""
echo "============================================"
echo "All done!"
echo "  Fixed:   $total_fixed"
echo "  Skipped: $total_skipped (already correct)"
echo "  Failed:  $total_failed"
echo "============================================"
