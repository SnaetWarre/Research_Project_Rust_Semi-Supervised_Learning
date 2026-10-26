#!/usr/bin/env bash
# Downloads the Kaggle "New Plant Diseases" dataset (87,867 images, 38 classes) into
# data/plantvillage/{train,valid}. Usage: scripts/download-dataset.sh [target-dir]
set -euo pipefail

target="${1:-data/plantvillage}"
url="https://www.kaggle.com/api/v1/datasets/download/vipoooool/new-plant-diseases-dataset"

if [[ -d "$target/train" && -d "$target/valid" ]]; then
    echo "$target already has train/ and valid/. Delete it to download again."
    exit 0
fi
command -v unzip >/dev/null || { echo "unzip is required" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "Downloading about 2.7 GB to $work"
curl -L --fail --progress-bar -o "$work/dataset.zip" "$url"
unzip -q "$work/dataset.zip" -d "$work/extracted"

# The archive nests train/ and valid/ a few folders deep
source_dir="$(dirname "$(find "$work/extracted" -type d -name train -print -quit)")"
[[ -d "$source_dir/valid" ]] || { echo "unexpected archive layout" >&2; exit 1; }

mkdir -p "$target"
mv "$source_dir/train" "$source_dir/valid" "$target/"
echo "Saved $(find "$target" -type f | wc -l) images to $target"
