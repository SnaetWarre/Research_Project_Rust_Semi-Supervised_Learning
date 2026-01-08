#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# PlantVillage Dataset Downloader
# ============================================================================
# Single canonical way to download and organize the dataset from Kaggle
# Works for both plantvillage_ssl/ and incremental_learning/
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="abdallahalidev/plantvillage-dataset"
OUTPUT_DIR="$SCRIPT_DIR/plantvillage_ssl/data/plantvillage"
FORCE=0
SKIP_DOWNLOAD=0

usage() {
  cat <<EOF
Usage: $0 [--output-dir PATH] [--force] [--skip-download]

Downloads PlantVillage dataset from Kaggle and organizes it into:
  organized/      - Class-per-directory structure
  splits/         - Train/val/test split manifests

Options:
  --output-dir PATH   Target directory (default: plantvillage_ssl/data/plantvillage)
  --force             Re-download even if data exists
  --skip-download     Reuse existing raw data, only rebuild manifests
  -h, --help          Show this help

Requirements:
  - kaggle CLI tool (pip install kaggle)
  - ~/.kaggle/kaggle.json with API credentials

