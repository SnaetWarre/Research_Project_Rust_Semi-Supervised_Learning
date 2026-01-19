#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# New Plant Diseases Dataset Downloader (Augmented & Balanced PlantVillage)
# ============================================================================
# Downloads the balanced, augmented PlantVillage dataset from Kaggle
# Dataset: vipoooool/new-plant-diseases-dataset
# 
# This dataset contains ~87K RGB images across 38 balanced classes.
# It comes pre-split into train/ and valid/ folders.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATASET="vipoooool/new-plant-diseases-dataset"
OUTPUT_DIR="$PROJECT_ROOT/data/plantvillage"
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

Downloads the New Plant Diseases Dataset (balanced, augmented PlantVillage) from Kaggle.

This dataset contains:
  - ~87,000 RGB images of healthy and diseased crop leaves
  - 38 balanced classes (same as original PlantVillage)
  - Pre-split into train/ (~70K) and valid/ (~17K) folders

Options:
  --output-dir PATH   Target directory (default: data/plantvillage)
  --force             Re-download even if data exists
  -h, --help          Show this help

Requirements:
  - kaggle CLI tool (pip install kaggle)
  - ~/.kaggle/kaggle.json with API credentials

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

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    log_error "kaggle CLI not found. Please install it:"
    echo "  pip install kaggle"
    echo ""
    echo "Then configure your API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save kaggle.json to ~/.kaggle/kaggle.json"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Check if kaggle credentials exist
if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    log_error "Kaggle credentials not found at ~/.kaggle/kaggle.json"
    echo ""
    echo "To set up Kaggle API credentials:"
    echo "  1. Go to https://www.kaggle.com/settings"
    echo "  2. Click 'Create New Token' under API section"
    echo "  3. Save the downloaded kaggle.json to ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

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

log_info "Downloading New Plant Diseases Dataset..."
log_info "  Source: kaggle.com/datasets/$DATASET"
log_info "  Target: $OUTPUT_DIR"
echo ""

# Download dataset
cd "$OUTPUT_DIR"
kaggle datasets download -d "$DATASET" --unzip

# The dataset extracts with a nested structure, let's fix it
# Expected: New Plant Diseases Dataset(Augmented)/train and .../valid
if [[ -d "New Plant Diseases Dataset(Augmented)" ]]; then
    log_info "Reorganizing dataset structure..."
    
    # Move train and valid folders up
    if [[ -d "New Plant Diseases Dataset(Augmented)/train" ]]; then
        mv "New Plant Diseases Dataset(Augmented)/train" ./train
    fi
    if [[ -d "New Plant Diseases Dataset(Augmented)/valid" ]]; then
        mv "New Plant Diseases Dataset(Augmented)/valid" ./valid
    fi
    
    # Remove the wrapper directory
    rm -rf "New Plant Diseases Dataset(Augmented)"
fi

# Also check for alternate naming
if [[ -d "new plant diseases dataset(augmented)" ]]; then
    log_info "Reorganizing dataset structure..."
    
    if [[ -d "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train" ]]; then
        mv "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/train" ./train
        mv "new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)/valid" ./valid
    elif [[ -d "new plant diseases dataset(augmented)/train" ]]; then
        mv "new plant diseases dataset(augmented)/train" ./train
        mv "new plant diseases dataset(augmented)/valid" ./valid
    fi
    
    rm -rf "new plant diseases dataset(augmented)"
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
echo "  ├── train/           (~70K images, 38 classes)"
echo "  │   ├── Apple___Apple_scab/"
echo "  │   ├── Apple___Black_rot/"
echo "  │   └── ... (38 balanced classes)"
echo "  └── valid/           (~17K images, 38 classes)"
echo "      ├── Apple___Apple_scab/"
echo "      └── ..."
echo ""
log_info "Next steps:"
echo "  1. Train the model:"
echo "     cargo run --release -- train --data-dir data/plantvillage --epochs 50 --cuda"
echo ""
echo "  2. Run SSL simulation after training:"
echo "     cargo run --release -- simulate --model output/models/plant_classifier_*.mpk --data-dir data/plantvillage --cuda"
