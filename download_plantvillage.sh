#!/usr/bin/env bash
set -euo pipefail

# Plant Diseases dataset downloader.
# Using curl to download the balanced dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
# Based on original script, it was putting it in PROJECT_ROOT/data/plantvillage or SCRIPT_DIR/data/plantvillage.
# We'll use SCRIPT_DIR/data/plantvillage to keep it inside Source.
OUTPUT_DIR="$SCRIPT_DIR/data/plantvillage"
FORCE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [--output-dir PATH] [--force]

Downloads the PlantVillage balanced dataset via curl.

This dataset contains:
  - RGB images of healthy and diseased crop leaves
  - Balanced classes
  - Pre-split into train/ and valid/ folders

Options:
  --output-dir PATH   Target directory (default: data/plantvillage)
  --force             Re-download even if data exists
  -h, --help          Show this help

After download, run training with:
  cargo run --release -- train --data-dir data/plantvillage --epochs 50 --cuda

EOF
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if dataset already exists
if [[ -d "$OUTPUT_DIR/train" && -d "$OUTPUT_DIR/valid" && $FORCE -eq 0 ]]; then
    log_warn "Dataset already exists at $OUTPUT_DIR"
    echo "  Use --force to re-download"

    # Show quick stats
    TRAIN_COUNT=$(find "$OUTPUT_DIR/train" -type f -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    VALID_COUNT=$(find "$OUTPUT_DIR/valid" -type f -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l)
    TRAIN_CLASSES=$(find "$OUTPUT_DIR/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)

    echo ""
    log_info "Current dataset stats:"
    echo "  Train images: $TRAIN_COUNT"
    echo "  Valid images: $VALID_COUNT"
    echo "  Classes: $TRAIN_CLASSES"
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Downloading PlantVillage Dataset via curl..."
log_info "  Target: $OUTPUT_DIR"
echo ""

# Download dataset
cd "$OUTPUT_DIR"
ZIP_FILE="plantvillage-balanced.zip"

if [[ ! -f "$ZIP_FILE" && ! -d "plantvillage_split_balanced" && ! -d "train" ]]; then
    curl -L -o "$ZIP_FILE" "https://www.kaggle.com/api/v1/datasets/download/chandraguptsingh/plantvillage-balanced"
else
    log_info "Dataset already downloaded or partially extracted, skipping download."
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    log_error "'unzip' command not found. Please install it (e.g., pacman -S unzip)."
    exit 1
fi

if [[ -f "$ZIP_FILE" ]]; then
    log_info "Extracting dataset..."
    unzip -q -o "$ZIP_FILE" 2>/dev/null || true
    rm -f "$ZIP_FILE"
fi

# The dataset might extract with a nested structure, let's fix it by finding train/valid dirs
if [[ ! -d "train" || ! -d "valid" ]]; then
    log_info "Reorganizing dataset structure..."

    # Find directories named 'train' and 'valid' (or 'val')
    FOUND_TRAIN=$(find . -mindepth 2 -type d -name "train" | head -n 1)
    FOUND_VALID=$(find . -mindepth 2 -type d \( -name "valid" -o -name "val" \) | head -n 1)

    if [[ -n "$FOUND_TRAIN" ]]; then
        mkdir -p ./train
        mv "$FOUND_TRAIN"/* ./train/ 2>/dev/null || cp -r "$FOUND_TRAIN"/* ./train/
        rm -rf "$FOUND_TRAIN"
    fi
    if [[ -n "$FOUND_VALID" ]]; then
        mkdir -p ./valid
        mv "$FOUND_VALID"/* ./valid/ 2>/dev/null || cp -r "$FOUND_VALID"/* ./valid/
        rm -rf "$FOUND_VALID"
    fi

    # Clean up empty parent directories if needed
    rm -rf plantvillage_split_balanced 2>/dev/null || true
    find . -type d -empty -delete 2>/dev/null || true
fi

# Verify the structure
if [[ ! -d "$OUTPUT_DIR/train" || ! -d "$OUTPUT_DIR/valid" ]]; then
    log_error "Dataset extraction failed. Expected train/ and valid/ directories."
    log_info "Contents of $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"
    exit 1
fi

# Count images and classes
TRAIN_COUNT=$(find "$OUTPUT_DIR/train" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
VALID_COUNT=$(find "$OUTPUT_DIR/valid" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" \) 2>/dev/null | wc -l)
TRAIN_CLASSES=$(find "$OUTPUT_DIR/train" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
VALID_CLASSES=$(find "$OUTPUT_DIR/valid" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
TOTAL_COUNT=$((TRAIN_COUNT + VALID_COUNT))

echo ""
log_success "Dataset downloaded successfully!"
echo ""
echo "  Dataset Statistics:"
echo "  -------------------"
echo "  Total images:     $TOTAL_COUNT"
echo "  Training images:  $TRAIN_COUNT ($TRAIN_CLASSES classes)"
echo "  Validation images: $VALID_COUNT ($VALID_CLASSES classes)"
echo "  Location:         $OUTPUT_DIR"
echo ""
echo "  Dataset Structure:"
echo "  $OUTPUT_DIR/"
echo "  ├── train/           (~images, classes)"
echo "  │   ├── Apple___Apple_scab/"
echo "  │   ├── Apple___Black_rot/"
echo "  │   └── ..."
echo "  └── valid/           (~images, classes)"
echo "      ├── Apple___Apple_scab/"
echo "      └── ..."
echo ""
log_info "Next steps:"
echo "  1. Train the model:"
echo "     cargo run --release -- train --data-dir data/plantvillage --epochs 50 --cuda"
echo ""
echo "  2. Run SSL simulation after training:"
echo "     cargo run --release -- simulate --model output/models/plant_classifier_*.mpk --data-dir data/plantvillage --cuda"
