#!/bin/bash
# ==============================================================================
# PlantVillage Semi-Supervised Learning - Benchmark Comparison Script
# ==============================================================================
#
# This script runs benchmarks for both the Rust/Burn and Python/PyTorch
# implementations, then generates a comparison report.
#
# Usage:
#   ./scripts/run_benchmarks.sh [--epochs N] [--batch-size N] [--skip-training]
#
# Author: Warre Snaet
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
EPOCHS=5
BATCH_SIZE=32
IMAGE_SIZE=128
ITERATIONS=100
DATA_DIR="data/plantvillage"
OUTPUT_DIR="output/benchmarks"
SKIP_TRAINING=false
SKIP_PYTORCH=false
SKIP_BURN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-pytorch)
            SKIP_PYTORCH=true
            shift
            ;;
        --skip-burn)
            SKIP_BURN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N         Number of training epochs (default: 5)"
            echo "  --batch-size N     Batch size (default: 32)"
            echo "  --image-size N     Image size (default: 128)"
            echo "  --iterations N     Benchmark iterations (default: 100)"
            echo "  --data-dir PATH    Path to PlantVillage dataset"
            echo "  --output-dir PATH  Output directory for results"
            echo "  --skip-training    Skip training, only run inference benchmarks"
            echo "  --skip-pytorch     Skip PyTorch benchmarks"
            echo "  --skip-burn        Skip Burn/Rust benchmarks"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Convert relative paths to absolute paths
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
fi

if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$PROJECT_ROOT/$DATA_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${CYAN}"
echo "=============================================================="
echo "  PlantVillage Benchmark Comparison"
echo "  Burn (Rust) vs PyTorch"
echo "=============================================================="
echo -e "${NC}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Epochs:       $EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Image size:   ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Iterations:   $ITERATIONS"
echo "  Data dir:     $DATA_DIR"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

# Check for required tools
echo -e "${CYAN}Checking prerequisites...${NC}"

if ! command -v cargo &> /dev/null && [ "$SKIP_BURN" = false ]; then
    echo -e "${RED}Error: cargo not found. Please install Rust.${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null && [ "$SKIP_PYTORCH" = false ]; then
    echo -e "${RED}Error: python3 not found. Please install Python 3.${NC}"
    exit 1
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA available:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    DEVICE="cuda"
else
    echo -e "${YELLOW}CUDA not available, using CPU${NC}"
    DEVICE="cpu"
fi

echo ""

# ==============================================================================
# Burn/Rust Benchmarks
# ==============================================================================
if [ "$SKIP_BURN" = false ]; then
    echo -e "${CYAN}=============================================================="
    echo "  BURN (RUST) BENCHMARKS"
    echo "==============================================================${NC}"

    cd "$PROJECT_ROOT/plantvillage_ssl"

    # Build release binary
    echo -e "${YELLOW}Building Rust project...${NC}"
    cargo build --release

    BURN_BINARY="./target/release/plantvillage_ssl"

    # Run inference benchmark
    echo ""
    echo -e "${GREEN}Running Burn inference benchmark...${NC}"
    $BURN_BINARY benchmark \
        --iterations "$ITERATIONS" \
        --warmup 10 \
        --batch-size "$BATCH_SIZE" \
        --image-size "$IMAGE_SIZE" \
        --output "$OUTPUT_DIR/burn_benchmark.json" \
        --verbose

    # Run training benchmark if not skipped and dataset exists
    if [ "$SKIP_TRAINING" = false ] && [ -d "$DATA_DIR" ]; then
        echo ""
        echo -e "${GREEN}Running Burn training benchmark...${NC}"

        START_TIME=$(date +%s)
        $BURN_BINARY train \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR/burn_models" \
            --quick
        END_TIME=$(date +%s)

        BURN_TRAIN_TIME=$((END_TIME - START_TIME))
        echo "Burn training time: ${BURN_TRAIN_TIME}s"
        echo "{\"training_time_s\": $BURN_TRAIN_TIME}" > "$OUTPUT_DIR/burn_training_time.json"
    fi

    cd "$PROJECT_ROOT"
fi

# ==============================================================================
# PyTorch Benchmarks
# ==============================================================================
if [ "$SKIP_PYTORCH" = false ]; then
    echo ""
    echo -e "${CYAN}=============================================================="
    echo "  PYTORCH BENCHMARKS"
    echo "==============================================================${NC}"

    cd "$PROJECT_ROOT/pytorch_reference"

    # Check Python dependencies
    echo -e "${YELLOW}Checking Python dependencies...${NC}"
    if ! python3 -c "import torch" 2>/dev/null; then
        echo -e "${YELLOW}Installing PyTorch dependencies...${NC}"
        pip install -r requirements.txt
    fi

    # Run PyTorch benchmark
    echo ""
    echo -e "${GREEN}Running PyTorch benchmarks...${NC}"

    if [ "$SKIP_TRAINING" = false ]; then
        MODE="both"
    else
        MODE="benchmark"
    fi

    python3 trainer.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/pytorch" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --image-size "$IMAGE_SIZE" \
        --mode "$MODE"

    cd "$PROJECT_ROOT"
fi

# ==============================================================================
# Generate Comparison Report
# ==============================================================================
echo ""
echo -e "${CYAN}=============================================================="
echo "  GENERATING COMPARISON REPORT"
echo "==============================================================${NC}"

cd "$PROJECT_ROOT/benchmarks"

# Run comparison script if both benchmarks were run
if [ "$SKIP_BURN" = false ] && [ "$SKIP_PYTORCH" = false ]; then
    python3 compare_frameworks.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --image-size "$IMAGE_SIZE" \
        --device "$DEVICE" \
        --skip-pytorch \
        --skip-burn
fi

cd "$PROJECT_ROOT"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo -e "${GREEN}=============================================================="
echo "  BENCHMARK COMPLETE"
echo "==============================================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
ls -la "$OUTPUT_DIR/" 2>/dev/null || echo "  (no files yet)"
echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Review benchmark results in $OUTPUT_DIR/"
echo "  2. Check comparison charts: ${OUTPUT_DIR}/*.png"
echo "  3. See JSON data: ${OUTPUT_DIR}/*.json"
echo ""
