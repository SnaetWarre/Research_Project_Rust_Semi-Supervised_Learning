#!/bin/bash
# ==============================================================================
# Full Research Pipeline Script
# ==============================================================================
#
# This script runs the complete research pipeline:
# 1. Downloads dataset (if needed)
# 2. Trains supervised model (Burn)
# 3. Trains supervised model (PyTorch)
# 4. Runs semi-supervised simulation
# 5. Benchmarks inference and training
# 6. Compares frameworks and generates reports
#
# Usage:
#   ./run_research_pipeline.sh [command] [options]
#
# Commands:
#   all           - Run entire pipeline
#   train         - Train both frameworks
#   ssl           - Run semi-supervised simulation
#   benchmark     - Run benchmarks
#   compare       - Compare and generate reports
#   clean         - Clean output directories
#   help          - Show help
#
# Examples:
#   ./run_research_pipeline.sh all                    # Run everything
#   ./run_research_pipeline.sh train --epochs 10      # Train with custom epochs
#   ./run_research_pipeline.sh benchmark --iter 200   # Benchmark with 200 iterations
#
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Default configuration
CONFIG_FILE="$PROJECT_ROOT/pipeline_config.yaml"
DATA_DIR="$PROJECT_ROOT/plantvillage_ssl/data/plantvillage/organized"
OUTPUT_DIR="output/research_pipeline"
EPOCHS=50
BATCH_SIZE=32
ITERATIONS=100
WARMUP=10
IMAGE_SIZE=224

# Flags
SKIP_DOWNLOAD=false
SKIP_TRAIN_BURN=false
SKIP_TRAIN_PYTORCH=false
SKIP_SSL=false
SKIP_BENCHMARK=false
SKIP_COMPARE=false
DRY_RUN=false
VERBOSE=false
DATASET_SCRIPT="$PROJECT_ROOT/plantvillage_ssl/scripts/download_dataset.sh"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    local title="$1"
    echo ""
    echo -e "${CYAN}========================================================================${NC}"
    echo -e "${CYAN}  $title${NC}"
    echo -e "${CYAN}========================================================================${NC}"
    echo ""
}

print_stage() {
    local stage="$1"
    echo -e "${BLUE}[STAGE]${NC} $stage"
}

print_success() {
    local msg="$1"
    echo -e "${GREEN}✓${NC} $msg"
}

print_error() {
    local msg="$1"
    echo -e "${RED}✗${NC} $msg"
}

print_info() {
    local msg="$1"
    echo -e "${YELLOW}ℹ${NC} $msg"
}

print_command() {
    local cmd="$1"
    if [ "$VERBOSE" = true ]; then
        echo -e "${MAGENTA}➜${NC} $cmd"
    fi
}

check_command() {
    local cmd="$1"
    if ! command -v "$cmd" &> /dev/null; then
        print_error "$cmd not found. Please install $cmd."
        return 1
    fi
    return 0
}

# ==============================================================================
# Stage 1: Dataset Download
# ==============================================================================

download_dataset() {
    print_stage "Dataset Download"

    if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
        print_success "Dataset already exists at $DATA_DIR"
        return 0
    fi

    print_info "PlantVillage dataset not found. Downloading..."

    local data_root
    data_root="$(cd "$(dirname "$DATA_DIR")" && pwd)"

    if [ -x "$DATASET_SCRIPT" ]; then
        print_command "bash $DATASET_SCRIPT --output-dir $data_root"
        if [ "$DRY_RUN" = false ]; then
            bash "$DATASET_SCRIPT" --output-dir "$data_root"
        fi
        print_success "Dataset downloaded and organized"
    else
        print_error "Download script not found at $DATASET_SCRIPT"
        return 1
    fi
}

# ==============================================================================
# Stage 2: Train Burn Model
# ==============================================================================

train_burn() {
    print_stage "Training Burn (Rust) Model"

    cd "$PROJECT_ROOT/plantvillage_ssl"

    # Build project
    print_info "Building Burn project..."
    print_command "cargo build --release --features cuda"
    if [ "$DRY_RUN" = false ]; then
        cargo build --release --features cuda
    fi

    BURN_BINARY="./target/release/plantvillage_ssl"

    # Run training
    print_info "Starting supervised training..."
    print_command "$BURN_BINARY train --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR/burn"

    if [ "$DRY_RUN" = false ]; then
        START_TIME=$(date +%s)

        $BURN_BINARY train \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR/burn"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        echo "{\"training_time_s\": $TRAIN_TIME}" > "$OUTPUT_DIR/burn/training_time.json"
        print_success "Burn training completed in ${TRAIN_TIME}s"
    fi

    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Stage 3: Train PyTorch Model
# ==============================================================================

train_pytorch() {
    print_stage "Training PyTorch Model"

    cd "$PROJECT_ROOT/pytorch_reference"

    # Check dependencies
    print_info "Checking Python dependencies..."
    if ! python3 -c "import torch" 2>/dev/null; then
        print_info "Installing PyTorch dependencies..."
        print_command "pip install -r requirements.txt"
        if [ "$DRY_RUN" = false ]; then
            pip install -r requirements.txt
        fi
    fi

    # Run training
    print_info "Starting PyTorch training..."
    print_command "python3 trainer.py --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR/pytorch"

    if [ "$DRY_RUN" = false ]; then
        START_TIME=$(date +%s)

        python3 trainer.py \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR/pytorch"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        echo "{\"training_time_s\": $TRAIN_TIME}" > "$OUTPUT_DIR/pytorch/training_time.json"
        print_success "PyTorch training completed in ${TRAIN_TIME}s"
    fi

    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Stage 4: Semi-Supervised Simulation
# ==============================================================================

run_ssl_simulation() {
    print_stage "Semi-Supervised Simulation"

    cd "$PROJECT_ROOT/plantvillage_ssl"

    BURN_BINARY="./target/release/plantvillage_ssl"

    if [ ! -f "$BURN_BINARY" ]; then
        print_error "Burn binary not found. Run train_burn first."
        return 1
    fi

    print_info "Running semi-supervised simulation with pseudo-labeling..."
    print_command "$BURN_BINARY simulate --data-dir $DATA_DIR --batch-size 500 --days 10 --confidence-threshold 0.90 --output-dir $OUTPUT_DIR/ssl"

    if [ "$DRY_RUN" = false ]; then
        $BURN_BINARY simulate \
            --data-dir "$DATA_DIR" \
            --batch-size 500 \
            --days 10 \
            --confidence-threshold 0.90 \
            --output-dir "$OUTPUT_DIR/ssl"

        print_success "Semi-supervised simulation completed"
    fi

    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Stage 5: Benchmarking
# ==============================================================================

benchmark_inference() {
    print_stage "Inference Benchmarking"

    # Burn benchmark
    print_info "Benchmarking Burn (Rust) inference..."
    cd "$PROJECT_ROOT/plantvillage_ssl"

    BURN_BINARY="./target/release/plantvillage_ssl"
    if [ -f "$BURN_BINARY" ]; then
        print_command "$BURN_BINARY benchmark --iterations $ITERATIONS --warmup $WARMUP --batch-size $BATCH_SIZE --output $OUTPUT_DIR/benchmark/burn.json"

        if [ "$DRY_RUN" = false ]; then
            $BURN_BINARY benchmark \
                --iterations "$ITERATIONS" \
                --warmup "$WARMUP" \
                --batch-size "$BATCH_SIZE" \
                --output "$OUTPUT_DIR/benchmark/burn.json" \
                --verbose
        fi
        print_success "Burn benchmark completed"
    else
        print_error "Burn binary not found"
    fi

    # PyTorch benchmark
    print_info "Benchmarking PyTorch inference..."
    cd "$PROJECT_ROOT/pytorch_reference"

    print_command "python3 trainer.py --mode benchmark --data-dir $DATA_DIR --iterations $ITERATIONS --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR/benchmark"

    if [ "$DRY_RUN" = false ]; then
        python3 trainer.py \
            --mode benchmark \
            --data-dir "$DATA_DIR" \
            --iterations "$ITERATIONS" \
            --batch-size "$BATCH_SIZE" \
            --output-dir "$OUTPUT_DIR/benchmark"
    fi
    print_success "PyTorch benchmark completed"

    cd "$PROJECT_ROOT"
}

benchmark_training() {
    print_stage "Training Time Benchmarking"

    print_info "Training time already measured in Stage 2 and 3"
    print_info "Results saved to:"
    print_info "  - $OUTPUT_DIR/burn/training_time.json"
    print_info "  - $OUTPUT_DIR/pytorch/training_time.json"
}

# ==============================================================================
# Stage 6: Comparison and Reporting
# ==============================================================================

compare_frameworks() {
    print_stage "Framework Comparison and Reporting"

    cd "$PROJECT_ROOT/benchmarks"

    # Determine device
    if command -v nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi

    print_info "Running comparison script..."
    print_command "python3 compare_frameworks.py --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --device $DEVICE --skip-burn --skip-pytorch --no-charts"

    if [ "$DRY_RUN" = false ]; then
        python3 compare_frameworks.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --skip-burn \
            --skip-pytorch

        print_success "Comparison report generated"
    fi

    cd "$PROJECT_ROOT"
}

generate_summary_report() {
    print_stage "Generating Summary Report"

    REPORT_FILE="$OUTPUT_DIR/research_summary.txt"

    {
        echo "======================================="
        echo "  RESEARCH PIPELINE SUMMARY REPORT"
        echo "======================================="
        echo ""
        echo "Date: $(date)"
        echo "Configuration:"
        echo "  - Epochs: $EPOCHS"
        echo "  - Batch Size: $BATCH_SIZE"
        echo "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
        echo "  - Benchmark Iterations: $ITERATIONS"
        echo ""
        echo "Dataset: $DATA_DIR"
        echo "Output Directory: $OUTPUT_DIR"
        echo ""
        echo "======================================="
        echo "  OUTPUT FILES"
        echo "======================================="
        echo ""

        if [ -d "$OUTPUT_DIR/burn" ]; then
            echo "Burn (Rust) Outputs:"
            ls -lh "$OUTPUT_DIR/burn" | grep -v "^total" | awk '{print "  -", $9, "("$5")"}'
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/pytorch" ]; then
            echo "PyTorch Outputs:"
            ls -lh "$OUTPUT_DIR/pytorch" | grep -v "^total" | awk '{print "  -", $9, "("$5")"}'
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/benchmark" ]; then
            echo "Benchmark Results:"
            ls -lh "$OUTPUT_DIR/benchmark" | grep -v "^total" | awk '{print "  -", $9, "("$5")"}'
            echo ""
        fi

        echo "======================================="
        echo "  COMPARISON CHARTS"
        echo "======================================="
        echo ""
        find "$OUTPUT_DIR" -name "*.png" -o -name "*.jpg" | while read chart; do
            echo "  - $chart"
        done
        echo ""

        echo "======================================="
        echo "  DATA FILES (JSON)"
        echo "======================================="
        echo ""
        find "$OUTPUT_DIR" -name "*.json" | while read json; do
            echo "  - $json"
        done
        echo ""

    } > "$REPORT_FILE"

    print_success "Summary report saved to: $REPORT_FILE"
    echo ""
    cat "$REPORT_FILE"
}

# ==============================================================================
# Clean Function
# ==============================================================================

clean() {
    print_stage "Cleaning Output Directories"

    print_info "Removing output directories..."
    if [ -d "$OUTPUT_DIR" ]; then
        print_command "rm -rf $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
        print_success "Removed $OUTPUT_DIR"
    fi

    if [ -d "output/models" ]; then
        print_command "rm -rf output/models"
        rm -rf output/models
        print_success "Removed output/models"
    fi

    if [ -d "output/benchmarks" ]; then
        print_command "rm -rf output/benchmarks"
        rm -rf output/benchmarks
        print_success "Removed output/benchmarks"
    fi

    # Clean Rust build artifacts
    cd "$PROJECT_ROOT/plantvillage_ssl"
    print_info "Cleaning Rust build artifacts..."
    print_command "cargo clean"
    if [ "$DRY_RUN" = false ]; then
        cargo clean
    fi
    print_success "Cleaned Rust build artifacts"
    cd "$PROJECT_ROOT"

    print_success "Clean complete!"
}

# ==============================================================================
# Show Help
# ==============================================================================

show_help() {
    cat << EOF
Research Pipeline Script - Full Research Automation

Usage: $0 [command] [options]

Commands:
  all           Run entire research pipeline (default)
  train         Train both Burn and PyTorch models
  ssl           Run semi-supervised simulation
  benchmark     Run inference and training benchmarks
  compare       Compare frameworks and generate reports
  clean         Clean all output directories and build artifacts
  help          Show this help message

Options:
  --epochs N                Number of training epochs (default: 50)
  --batch-size N            Batch size (default: 32)
  --iterations N            Benchmark iterations (default: 100)
  --image-size N            Image size (default: 224)
  --data-dir PATH           Dataset directory (default: data/plantvillage)
  --output-dir PATH         Output directory (default: output/research_pipeline)
  --skip-download           Skip dataset download
  --skip-train-burn         Skip Burn training
  --skip-train-pytorch      Skip PyTorch training
  --skip-ssl                Skip semi-supervised simulation
  --skip-benchmark          Skip benchmarking
  --skip-compare            Skip comparison
  --dry-run                 Show commands without executing
  --verbose                 Show detailed command output
  -h, --help                Show this help message

Examples:
  # Run full pipeline
  $0 all

  # Run full pipeline with custom epochs
  $0 all --epochs 20

  # Train only
  $0 train --epochs 10

  # Run benchmarks only
  $0 benchmark --iterations 200

  # Clean everything
  $0 clean

Pipeline Stages (when running 'all'):
  1. Download Dataset      - Download PlantVillage dataset if not present
  2. Train Burn           - Train Rust/Burn model with supervised learning
  3. Train PyTorch        - Train PyTorch model with supervised learning
  4. SSL Simulation       - Run semi-supervised pseudo-labeling simulation
  5. Benchmarking          - Run inference and training benchmarks
  6. Compare              - Generate comparison charts and reports
  7. Summary              - Generate final summary report

EOF
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

COMMAND="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        all|train|ssl|benchmark|compare|clean|help)
            COMMAND="$1"
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --image-size)
            IMAGE_SIZE="$2"
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
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-train-burn)
            SKIP_TRAIN_BURN=true
            shift
            ;;
        --skip-train-pytorch)
            SKIP_TRAIN_PYTORCH=true
            shift
            ;;
        --skip-ssl)
            SKIP_SSL=true
            shift
            ;;
        --skip-benchmark)
            SKIP_BENCHMARK=true
            shift
            ;;
        --skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use '$0 help' for usage information."
            exit 1
            ;;
    esac
done

# ==============================================================================
# Main Pipeline
# ==============================================================================

main() {
    print_header "PlantVillage Semi-Supervised Learning Research Pipeline"

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Show configuration
    print_info "Pipeline Configuration:"
    echo "  Epochs:       $EPOCHS"
    echo "  Batch size:   $BATCH_SIZE"
    echo "  Image size:   ${IMAGE_SIZE}x${IMAGE_SIZE}"
    echo "  Iterations:   $ITERATIONS"
    echo "  Data dir:     $DATA_DIR"
    echo "  Output dir:   $OUTPUT_DIR"
    echo "  Dry run:      $DRY_RUN"
    echo ""

    # Check prerequisites
    print_info "Checking prerequisites..."
    check_command cargo || exit 1
    check_command python3 || exit 1

    if command -v nvidia-smi &> /dev/null; then
        print_success "CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        print_info "CUDA not available, using CPU"
    fi

    echo ""

    # Execute based on command
    case $COMMAND in
        all)
            print_info "Running full pipeline..."

            [ "$SKIP_DOWNLOAD" = false ] && download_dataset

            [ "$SKIP_TRAIN_BURN" = false ] && train_burn
            [ "$SKIP_TRAIN_PYTORCH" = false ] && train_pytorch

            [ "$SKIP_SSL" = false ] && run_ssl_simulation

            [ "$SKIP_BENCHMARK" = false ] && benchmark_inference
            [ "$SKIP_BENCHMARK" = false ] && benchmark_training

            [ "$SKIP_COMPARE" = false ] && compare_frameworks
            generate_summary_report

            print_success "Full pipeline completed!"
            echo ""
            print_info "Results saved to: $OUTPUT_DIR"
            ;;

        train)
            print_info "Running training stage..."

            [ "$SKIP_DOWNLOAD" = false ] && download_dataset
            [ "$SKIP_TRAIN_BURN" = false ] && train_burn
            [ "$SKIP_TRAIN_PYTORCH" = false ] && train_pytorch

            print_success "Training completed!"
            ;;

        ssl)
            print_info "Running SSL simulation..."

            [ "$SKIP_SSL" = false ] && run_ssl_simulation

            print_success "SSL simulation completed!"
            ;;

        benchmark)
            print_info "Running benchmarks..."

            [ "$SKIP_BENCHMARK" = false ] && benchmark_inference
            [ "$SKIP_BENCHMARK" = false ] && benchmark_training

            [ "$SKIP_COMPARE" = false ] && compare_frameworks

            print_success "Benchmarking completed!"
            ;;

        compare)
            print_info "Running comparison..."

            [ "$SKIP_COMPARE" = false ] && compare_frameworks
            generate_summary_report

            print_success "Comparison completed!"
            ;;

        clean)
            clean
            ;;

        help)
            show_help
            ;;
    esac

    echo ""
    print_success "Done! 🎉"
}

main
