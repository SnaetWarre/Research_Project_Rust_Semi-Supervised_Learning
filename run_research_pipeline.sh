#!/bin/bash
# ==============================================================================
# Full Research Pipeline Script
# ==============================================================================
#
# This script runs the complete research pipeline, reading configuration from
# pipeline_config.yaml (single source of truth).
#
# Pipeline stages:
# 1. Downloads dataset (if needed)
# 2. Trains supervised model (Burn/Rust)
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
#   config        - Show current configuration
#   help          - Show help
#
# Configuration is read from pipeline_config.yaml. CLI args override YAML values.
#
# ==============================================================================

set -e

# ==============================================================================
# Initialize Conda (handles lazy-loaded conda setups)
# ==============================================================================
init_conda() {
    # If conda is already available, nothing to do
    if command -v conda &> /dev/null; then
        return 0
    fi
    
    # Try to find and source conda initialization
    local conda_paths=(
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "/opt/anaconda3/etc/profile.d/conda.sh"
        "/opt/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )
    
    for conda_sh in "${conda_paths[@]}"; do
        if [ -f "$conda_sh" ]; then
            source "$conda_sh"
            return 0
        fi
    done
    
    # Could not find conda
    return 1
}

# Initialize conda early (before we need it)
init_conda 2>/dev/null || true

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
CONFIG_FILE="$PROJECT_ROOT/pipeline_config.yaml"

# ==============================================================================
# YAML Parsing Functions
# ==============================================================================

# Check if yq is available, try different variants
YQ_CMD=""
detect_yq() {
    if command -v yq &> /dev/null; then
        # Check if it's the Go version (mikefarah) or Python version
        if yq --version 2>&1 | grep -q "mikefarah"; then
            YQ_CMD="yq_go"
        else
            YQ_CMD="yq_python"
        fi
    elif command -v python3 &> /dev/null && python3 -c "import yaml" 2>/dev/null; then
        YQ_CMD="python"
    else
        YQ_CMD="grep"
        echo -e "${YELLOW}Warning: yq not found. Using basic grep parsing (limited).${NC}"
        echo -e "${YELLOW}Install yq for full YAML support: sudo pacman -S yq${NC}"
    fi
}

# Read a value from YAML config
# Usage: read_config ".path.to.key" "default_value"
read_config() {
    local key="$1"
    local default="$2"
    local value=""

    if [ ! -f "$CONFIG_FILE" ]; then
        echo "$default"
        return
    fi

    case "$YQ_CMD" in
        yq_go)
            value=$(yq eval "$key // \"\"" "$CONFIG_FILE" 2>/dev/null)
            ;;
        yq_python)
            value=$(yq -r "$key // \"\"" "$CONFIG_FILE" 2>/dev/null)
            ;;
        python)
            value=$(python3 -c "
import yaml
import sys
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
keys = '$key'.strip('.').split('.')
val = config
try:
    for k in keys:
        val = val[k]
    print(val if val is not None else '')
except (KeyError, TypeError):
    print('')
" 2>/dev/null)
            ;;
        grep)
            # Basic grep fallback - only works for simple keys
            local simple_key=$(echo "$key" | sed 's/.*\.//')
            value=$(grep -E "^\s*${simple_key}:" "$CONFIG_FILE" 2>/dev/null | head -1 | sed 's/.*:\s*//' | sed 's/#.*//' | xargs)
            ;;
    esac

    # Return default if value is empty or null
    if [ -z "$value" ] || [ "$value" = "null" ]; then
        echo "$default"
    else
        echo "$value"
    fi
}

# Read boolean from YAML (returns "true" or "false")
read_config_bool() {
    local key="$1"
    local default="$2"
    local value=$(read_config "$key" "$default")
    
    # Normalize to lowercase
    value=$(echo "$value" | tr '[:upper:]' '[:lower:]')
    
    if [ "$value" = "true" ] || [ "$value" = "yes" ] || [ "$value" = "1" ]; then
        echo "true"
    else
        echo "false"
    fi
}

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
# Load Configuration from YAML
# ==============================================================================

load_config() {
    detect_yq
    
    # Paths
    DATA_DIR="$PROJECT_ROOT/$(read_config '.paths.data_dir' 'plantvillage_ssl/data/plantvillage')"
    OUTPUT_DIR="$(read_config '.paths.output_dir' 'output/research_pipeline')"
    
    # Dataset settings
    NUM_CLASSES=$(read_config '.dataset.num_classes' '38')
    IMAGE_SIZE=$(read_config '.dataset.image_size' '224')
    LABELED_RATIO=$(read_config '.dataset.labeled_ratio' '0.20')
    VALIDATION_SPLIT=$(read_config '.dataset.validation_split' '0.10')
    TEST_SPLIT=$(read_config '.dataset.test_split' '0.10')
    
    # Training settings
    EPOCHS=$(read_config '.training.epochs' '50')
    BATCH_SIZE=$(read_config '.training.batch_size' '32')
    LEARNING_RATE=$(read_config '.training.learning_rate' '0.001')
    WEIGHT_DECAY=$(read_config '.training.weight_decay' '0.0001')
    DROPOUT_RATE=$(read_config '.training.dropout_rate' '0.3')
    
    # SSL settings
    SSL_ENABLED=$(read_config_bool '.ssl.enabled' 'true')
    SSL_CONFIDENCE=$(read_config '.ssl.confidence_threshold' '0.90')
    SSL_USE_ALL=$(read_config_bool '.ssl.use_all_unlabeled' 'true')
    SSL_RETRAIN_THRESHOLD=$(read_config '.ssl.retrain_threshold' '500')
    SSL_RETRAIN_EPOCHS=$(read_config '.ssl.retrain_epochs' '20')
    
    # Benchmarking settings
    BENCHMARK_ENABLED=$(read_config_bool '.benchmarking.enabled' 'true')
    ITERATIONS=$(read_config '.benchmarking.iterations' '100')
    WARMUP=$(read_config '.benchmarking.warmup' '10')
    COMPARE_FRAMEWORKS=$(read_config_bool '.benchmarking.compare_frameworks' 'true')
    
    # Pipeline stages (from YAML)
    STAGE_DOWNLOAD=$(read_config_bool '.stages.download_dataset' 'true')
    STAGE_TRAIN_BURN=$(read_config_bool '.stages.train_burn' 'true')
    STAGE_TRAIN_PYTORCH=$(read_config_bool '.stages.train_pytorch' 'true')
    STAGE_SSL=$(read_config_bool '.stages.ssl_simulation' 'true')
    STAGE_BENCHMARK=$(read_config_bool '.stages.benchmark' 'true')
    STAGE_COMPARE=$(read_config_bool '.stages.compare' 'true')
    
    # Python environment
    CONDA_ENV=$(read_config '.python.conda_env' '')
    
    # Script paths
    DATASET_SCRIPT="$PROJECT_ROOT/download_plantvillage.sh"
}

# ==============================================================================
# Python Environment Functions
# ==============================================================================

# Get Python executable (from conda env if specified, otherwise system python3)
get_python() {
    if [ -n "$CONDA_ENV" ] && command -v conda &> /dev/null; then
        # Use conda run with --live-stream to avoid output buffering
        echo "conda run --live-stream -n $CONDA_ENV python"
    else
        echo "python3"
    fi
}

# Get pip executable (from conda env if specified, otherwise system pip)
get_pip() {
    if [ -n "$CONDA_ENV" ] && command -v conda &> /dev/null; then
        echo "conda run --live-stream -n $CONDA_ENV pip"
    else
        echo "pip"
    fi
}

# Check if conda environment exists
check_conda_env() {
    if [ -n "$CONDA_ENV" ]; then
        if ! conda env list | grep -q "^${CONDA_ENV}\s"; then
            print_error "Conda environment '$CONDA_ENV' not found!"
            print_info "Available environments:"
            conda env list | grep -v "^#" | awk '{print "  - " $1}'
            return 1
        fi
        print_success "Using conda environment: $CONDA_ENV"
    fi
    return 0
}

# ==============================================================================
# Show Configuration
# ==============================================================================

show_config() {
    print_header "Current Pipeline Configuration"
    
    echo -e "${CYAN}Configuration file:${NC} $CONFIG_FILE"
    echo ""
    
    echo -e "${YELLOW}Paths:${NC}"
    echo "  Data directory:    $DATA_DIR"
    echo "  Output directory:  $OUTPUT_DIR"
    echo ""
    
    echo -e "${YELLOW}Dataset:${NC}"
    echo "  Classes:           $NUM_CLASSES"
    echo "  Image size:        ${IMAGE_SIZE}x${IMAGE_SIZE}"
    echo "  Labeled ratio:     $LABELED_RATIO ($(echo "$LABELED_RATIO * 100" | bc)%)"
    echo "  Validation split:  $VALIDATION_SPLIT ($(echo "$VALIDATION_SPLIT * 100" | bc)%)"
    echo "  Test split:        $TEST_SPLIT ($(echo "$TEST_SPLIT * 100" | bc)%)"
    echo ""
    
    echo -e "${YELLOW}Training:${NC}"
    echo "  Epochs:            $EPOCHS"
    echo "  Batch size:        $BATCH_SIZE"
    echo "  Learning rate:     $LEARNING_RATE"
    echo "  Weight decay:      $WEIGHT_DECAY"
    echo "  Dropout rate:      $DROPOUT_RATE"
    echo ""
    
    echo -e "${YELLOW}Semi-Supervised Learning:${NC}"
    echo "  Enabled:           $SSL_ENABLED"
    echo "  Confidence:        $SSL_CONFIDENCE"
    echo "  Use all unlabeled: $SSL_USE_ALL"
    echo "  Retrain threshold: $SSL_RETRAIN_THRESHOLD"
    echo "  Retrain epochs:    $SSL_RETRAIN_EPOCHS"
    echo ""
    
    echo -e "${YELLOW}Benchmarking:${NC}"
    echo "  Enabled:           $BENCHMARK_ENABLED"
    echo "  Iterations:        $ITERATIONS"
    echo "  Warmup:            $WARMUP"
    echo "  Compare frameworks: $COMPARE_FRAMEWORKS"
    echo ""
    
    echo -e "${YELLOW}Python Environment:${NC}"
    if [ -n "$CONDA_ENV" ]; then
        echo "  Conda env:         $CONDA_ENV"
        echo "  Python:            $(get_python)"
    else
        echo "  Using:             system python3"
    fi
    echo ""
    
    echo -e "${YELLOW}Pipeline Stages:${NC}"
    echo "  Download dataset:  $STAGE_DOWNLOAD"
    echo "  Train Burn:        $STAGE_TRAIN_BURN"
    echo "  Train PyTorch:     $STAGE_TRAIN_PYTORCH"
    echo "  SSL Simulation:    $STAGE_SSL"
    echo "  Benchmark:         $STAGE_BENCHMARK"
    echo "  Compare:           $STAGE_COMPARE"
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
        print_success "Dataset downloaded successfully"
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

    # Run training with YAML config values
    print_info "Starting supervised training..."
    print_info "  Epochs: $EPOCHS, Batch size: $BATCH_SIZE, Labeled ratio: $LABELED_RATIO"
    print_command "$BURN_BINARY train --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --labeled-ratio $LABELED_RATIO --learning-rate $LEARNING_RATE --output-dir $OUTPUT_DIR/burn"

    if [ "$DRY_RUN" = false ]; then
        START_TIME=$(date +%s)

        $BURN_BINARY train \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --labeled-ratio "$LABELED_RATIO" \
            --learning-rate "$LEARNING_RATE" \
            --output-dir "$OUTPUT_DIR/burn"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Post-processing: Find latest model and copy to best_model.mpk
        # We are in plantvillage_ssl directory here, so path is relative
        # OUTPUT_DIR in config is "output/research_pipeline", so "$OUTPUT_DIR/burn" is correct relative path
        LATEST_MODEL=$(ls -t "$OUTPUT_DIR/burn"/plant_classifier_*.mpk 2>/dev/null | head -n 1)
        
        if [ -n "$LATEST_MODEL" ]; then
            cp "$LATEST_MODEL" "best_model.mpk"
            print_success "Latest model copied to plantvillage_ssl/best_model.mpk"
            
            # Also keep a reference in output dir
            cp "$LATEST_MODEL" "$OUTPUT_DIR/burn/model_final.mpk"
        else
            print_error "No model file found in $OUTPUT_DIR/burn"
        fi

        mkdir -p "$OUTPUT_DIR/burn"
        echo "{\"training_time_s\": $TRAIN_TIME, \"epochs\": $EPOCHS, \"batch_size\": $BATCH_SIZE, \"labeled_ratio\": $LABELED_RATIO}" > "$OUTPUT_DIR/burn/training_time.json"
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

    PYTHON_CMD=$(get_python)
    PIP_CMD=$(get_pip)

    # Check dependencies
    print_info "Checking Python dependencies..."
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        print_info "Installing PyTorch dependencies..."
        print_command "$PIP_CMD install -r requirements.txt"
        if [ "$DRY_RUN" = false ]; then
            $PIP_CMD install -r requirements.txt
        fi
    else
        print_success "PyTorch already available in environment"
    fi

    # Run training with YAML config values
    print_info "Starting PyTorch training..."
    print_info "  Epochs: $EPOCHS, Batch size: $BATCH_SIZE, Labeled ratio: $LABELED_RATIO"
    print_info "  Python: $PYTHON_CMD"
    print_command "$PYTHON_CMD trainer.py --data-dir $DATA_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --labeled-ratio $LABELED_RATIO --lr $LEARNING_RATE --image-size $IMAGE_SIZE --output-dir $OUTPUT_DIR/pytorch"

    if [ "$DRY_RUN" = false ]; then
        START_TIME=$(date +%s)

        $PYTHON_CMD trainer.py \
            --data-dir "$DATA_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --labeled-ratio "$LABELED_RATIO" \
            --lr "$LEARNING_RATE" \
            --image-size "$IMAGE_SIZE" \
            --output-dir "$OUTPUT_DIR/pytorch"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        mkdir -p "$OUTPUT_DIR/pytorch"
        echo "{\"training_time_s\": $TRAIN_TIME, \"epochs\": $EPOCHS, \"batch_size\": $BATCH_SIZE, \"labeled_ratio\": $LABELED_RATIO}" > "$OUTPUT_DIR/pytorch/training_time.json"
        print_success "PyTorch training completed in ${TRAIN_TIME}s"
    fi

    cd "$PROJECT_ROOT"
}

# ==============================================================================
# Stage 4: Semi-Supervised Simulation
# ==============================================================================

run_ssl_simulation() {
    print_stage "Semi-Supervised Simulation"

    if [ "$SSL_ENABLED" != "true" ]; then
        print_info "SSL simulation disabled in config"
        return 0
    fi

    cd "$PROJECT_ROOT/plantvillage_ssl"

    BURN_BINARY="./target/release/plantvillage_ssl"

    if [ ! -f "$BURN_BINARY" ]; then
        print_error "Burn binary not found. Run train_burn first."
        return 1
    fi

    # Find the latest trained model (most recent timestamp)
    # We look for files matching "plant_classifier_*.mpk" or "model_final.mpk" in the burn output directory
    MODEL_PATH=$(ls -t "$OUTPUT_DIR/burn"/plant_classifier_*.mpk 2>/dev/null | head -n 1)
    
    if [ -z "$MODEL_PATH" ]; then
        # Fallback to model_final.mpk if timestamped one not found
        MODEL_PATH=$(find "$OUTPUT_DIR/burn" -name "model_final.mpk" -type f 2>/dev/null | head -1)
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        # Fallback to just any mpk file
        MODEL_PATH=$(find "$OUTPUT_DIR/burn" -name "*.mpk" -type f 2>/dev/null | head -1)
    fi

    if [ -z "$MODEL_PATH" ]; then
        print_error "No model found in $OUTPUT_DIR/burn. Run train_burn first."
        return 1
    fi
    
    print_info "Using model: $MODEL_PATH"

    # Calculate images per day to use ALL unlabeled data
    # With use_all_unlabeled=true, we want to process all stream pool images
    # Default: simulate 30 days, images_per_day calculated to use all data
    DAYS=30
    if [ "$SSL_USE_ALL" = "true" ]; then
        # Use larger batch to process more images
        IMAGES_PER_DAY=1000
        print_info "Using ALL unlabeled images (50% of dataset) over $DAYS simulated days"
    else
        IMAGES_PER_DAY=500
    fi

    print_info "Running semi-supervised simulation with pseudo-labeling..."
    print_info "  Confidence threshold: $SSL_CONFIDENCE"
    print_info "  Retrain threshold: $SSL_RETRAIN_THRESHOLD pseudo-labels"
    print_info "  Retrain epochs: $SSL_RETRAIN_EPOCHS"
    print_command "$BURN_BINARY simulate --data-dir $DATA_DIR --model $MODEL_PATH --days $DAYS --images-per-day $IMAGES_PER_DAY --confidence-threshold $SSL_CONFIDENCE --retrain-threshold $SSL_RETRAIN_THRESHOLD --output-dir $OUTPUT_DIR/ssl"

    if [ "$DRY_RUN" = false ]; then
        $BURN_BINARY simulate \
            --data-dir "$DATA_DIR" \
            --model "$MODEL_PATH" \
            --days "$DAYS" \
            --images-per-day "$IMAGES_PER_DAY" \
            --confidence-threshold "$SSL_CONFIDENCE" \
            --retrain-threshold "$SSL_RETRAIN_THRESHOLD" \
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

    if [ "$BENCHMARK_ENABLED" != "true" ]; then
        print_info "Benchmarking disabled in config"
        return 0
    fi

    mkdir -p "$OUTPUT_DIR/benchmark"

    # Burn benchmark
    print_info "Benchmarking Burn (Rust) inference..."
    cd "$PROJECT_ROOT/plantvillage_ssl"

    BURN_BINARY="./target/release/plantvillage_ssl"
    if [ -f "$BURN_BINARY" ]; then
        print_command "$BURN_BINARY benchmark --iterations $ITERATIONS --warmup $WARMUP --batch-size $BATCH_SIZE --image-size $IMAGE_SIZE --output $OUTPUT_DIR/benchmark/burn.json"

        if [ "$DRY_RUN" = false ]; then
            $BURN_BINARY benchmark \
                --iterations "$ITERATIONS" \
                --warmup "$WARMUP" \
                --batch-size "$BATCH_SIZE" \
                --image-size "$IMAGE_SIZE" \
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

    PYTHON_CMD=$(get_python)
    print_command "$PYTHON_CMD trainer.py --mode benchmark --data-dir $DATA_DIR --iterations $ITERATIONS --batch-size $BATCH_SIZE --image-size $IMAGE_SIZE --output-dir $OUTPUT_DIR/benchmark"

    if [ "$DRY_RUN" = false ]; then
        $PYTHON_CMD trainer.py \
            --mode benchmark \
            --data-dir "$DATA_DIR" \
            --iterations "$ITERATIONS" \
            --batch-size "$BATCH_SIZE" \
            --image-size "$IMAGE_SIZE" \
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

    if [ "$COMPARE_FRAMEWORKS" != "true" ]; then
        print_info "Framework comparison disabled in config"
        return 0
    fi

    cd "$PROJECT_ROOT/benchmarks"

    # Determine device
    if command -v nvidia-smi &> /dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi

    PYTHON_CMD=$(get_python)
    print_info "Running comparison script..."
    print_command "$PYTHON_CMD compare_frameworks.py --data-dir $DATA_DIR --output-dir $OUTPUT_DIR --epochs $EPOCHS --batch-size $BATCH_SIZE --image-size $IMAGE_SIZE --device $DEVICE --skip-burn --skip-pytorch"

    if [ "$DRY_RUN" = false ]; then
        $PYTHON_CMD compare_frameworks.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --image-size "$IMAGE_SIZE" \
            --device "$DEVICE" \
            --skip-burn \
            --skip-pytorch

        print_success "Comparison report generated"
    fi

    cd "$PROJECT_ROOT"
}

generate_summary_report() {
    print_stage "Generating Summary Report"

    mkdir -p "$OUTPUT_DIR"
    REPORT_FILE="$OUTPUT_DIR/research_summary.txt"

    {
        echo "======================================="
        echo "  RESEARCH PIPELINE SUMMARY REPORT"
        echo "======================================="
        echo ""
        echo "Date: $(date)"
        echo "Config: $CONFIG_FILE"
        echo ""
        echo "Configuration (from YAML):"
        echo "  - Epochs: $EPOCHS"
        echo "  - Batch Size: $BATCH_SIZE"
        echo "  - Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
        echo "  - Labeled Ratio: $LABELED_RATIO ($(echo "$LABELED_RATIO * 100" | bc)%)"
        echo "  - SSL Confidence Threshold: $SSL_CONFIDENCE"
        echo "  - Benchmark Iterations: $ITERATIONS"
        echo ""
        echo "Dataset: $DATA_DIR"
        echo "Output Directory: $OUTPUT_DIR"
        echo ""
        echo "======================================="
        echo "  DATA SPLIT"
        echo "======================================="
        echo "  - Labeled Training: $(echo "$LABELED_RATIO * 100" | bc)%"
        echo "  - Validation: $(echo "$VALIDATION_SPLIT * 100" | bc)%"
        echo "  - Test: $(echo "$TEST_SPLIT * 100" | bc)%"
        echo "  - SSL Stream Pool: $(echo "(1 - $LABELED_RATIO - $VALIDATION_SPLIT - $TEST_SPLIT) * 100" | bc)%"
        echo ""
        echo "======================================="
        echo "  OUTPUT FILES"
        echo "======================================="
        echo ""

        if [ -d "$OUTPUT_DIR/burn" ]; then
            echo "Burn (Rust) Outputs:"
            ls -lh "$OUTPUT_DIR/burn" 2>/dev/null | grep -v "^total" | awk '{print "  -", $9, "("$5")"}' || echo "  (empty)"
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/pytorch" ]; then
            echo "PyTorch Outputs:"
            ls -lh "$OUTPUT_DIR/pytorch" 2>/dev/null | grep -v "^total" | awk '{print "  -", $9, "("$5")"}' || echo "  (empty)"
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/ssl" ]; then
            echo "SSL Simulation Outputs:"
            ls -lh "$OUTPUT_DIR/ssl" 2>/dev/null | grep -v "^total" | awk '{print "  -", $9, "("$5")"}' || echo "  (empty)"
            echo ""
        fi

        if [ -d "$OUTPUT_DIR/benchmark" ]; then
            echo "Benchmark Results:"
            ls -lh "$OUTPUT_DIR/benchmark" 2>/dev/null | grep -v "^total" | awk '{print "  -", $9, "("$5")"}' || echo "  (empty)"
            echo ""
        fi

        echo "======================================="
        echo "  COMPARISON CHARTS"
        echo "======================================="
        echo ""
        find "$OUTPUT_DIR" -name "*.png" -o -name "*.jpg" 2>/dev/null | while read chart; do
            echo "  - $chart"
        done
        echo ""

        echo "======================================="
        echo "  DATA FILES (JSON)"
        echo "======================================="
        echo ""
        find "$OUTPUT_DIR" -name "*.json" 2>/dev/null | while read json; do
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

Configuration is read from: pipeline_config.yaml
CLI arguments override YAML values.

Usage: $0 [command] [options]

Commands:
  all           Run entire research pipeline (default)
  train         Train both Burn and PyTorch models
  ssl           Run semi-supervised simulation
  benchmark     Run inference and training benchmarks
  compare       Compare frameworks and generate reports
  config        Show current configuration from YAML
  clean         Clean all output directories and build artifacts
  help          Show this help message

Options (override YAML config):
  --epochs N                Number of training epochs
  --batch-size N            Batch size
  --iterations N            Benchmark iterations
  --warmup N                Warmup iterations
  --image-size N            Image size
  --labeled-ratio N         Labeled data ratio (0.0-1.0)
  --data-dir PATH           Dataset directory
  --output-dir PATH         Output directory
  --confidence N            SSL confidence threshold
  --dry-run                 Show commands without executing
  --verbose                 Show detailed command output
  -h, --help                Show this help message

Examples:
  # Run full pipeline (uses YAML config)
  $0 all

  # Show current configuration
  $0 config

  # Override epochs from command line
  $0 all --epochs 20

  # Train only
  $0 train

  # Run benchmarks only
  $0 benchmark

  # Clean everything
  $0 clean

Pipeline Stages (controlled by YAML stages section):
  1. Download Dataset      - Download PlantVillage dataset if not present
  2. Train Burn            - Train Rust/Burn model with supervised learning
  3. Train PyTorch         - Train PyTorch model with supervised learning
  4. SSL Simulation        - Run semi-supervised pseudo-labeling simulation
  5. Benchmarking          - Run inference and training benchmarks
  6. Compare               - Generate comparison charts and reports
  7. Summary               - Generate final summary report

EOF
}

# ==============================================================================
# Parse Arguments
# ==============================================================================

COMMAND="all"
DRY_RUN=false
VERBOSE=false

# First load config from YAML
load_config

# Then parse CLI args (these override YAML values)
while [[ $# -gt 0 ]]; do
    case $1 in
        all|train|ssl|benchmark|compare|clean|config|help)
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
        --labeled-ratio)
            LABELED_RATIO="$2"
            shift 2
            ;;
        --confidence)
            SSL_CONFIDENCE="$2"
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
    print_info "Pipeline Configuration (from $CONFIG_FILE):"
    echo "  Epochs:         $EPOCHS"
    echo "  Batch size:     $BATCH_SIZE"
    echo "  Image size:     ${IMAGE_SIZE}x${IMAGE_SIZE}"
    echo "  Labeled ratio:  $LABELED_RATIO"
    echo "  Iterations:     $ITERATIONS"
    echo "  Data dir:       $DATA_DIR"
    echo "  Output dir:     $OUTPUT_DIR"
    echo "  Dry run:        $DRY_RUN"
    echo ""

    # Check prerequisites
    print_info "Checking prerequisites..."
    check_command cargo || exit 1
    
    # Check Python environment
    if [ -n "$CONDA_ENV" ]; then
        check_conda_env || exit 1
        PYTHON_CMD=$(get_python)
        if ! $PYTHON_CMD --version &>/dev/null; then
            print_error "Cannot execute Python from conda environment '$CONDA_ENV'"
            exit 1
        fi
        print_success "Python: $($PYTHON_CMD --version 2>&1)"
    else
        check_command python3 || exit 1
        print_success "Python: $(python3 --version 2>&1)"
    fi

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

            [ "$STAGE_DOWNLOAD" = "true" ] && download_dataset
            [ "$STAGE_TRAIN_BURN" = "true" ] && train_burn
            [ "$STAGE_TRAIN_PYTORCH" = "true" ] && train_pytorch
            [ "$STAGE_SSL" = "true" ] && run_ssl_simulation
            [ "$STAGE_BENCHMARK" = "true" ] && benchmark_inference
            [ "$STAGE_BENCHMARK" = "true" ] && benchmark_training
            [ "$STAGE_COMPARE" = "true" ] && compare_frameworks
            generate_summary_report

            print_success "Full pipeline completed!"
            echo ""
            print_info "Results saved to: $OUTPUT_DIR"
            ;;

        train)
            print_info "Running training stage..."

            [ "$STAGE_DOWNLOAD" = "true" ] && download_dataset
            [ "$STAGE_TRAIN_BURN" = "true" ] && train_burn
            [ "$STAGE_TRAIN_PYTORCH" = "true" ] && train_pytorch

            print_success "Training completed!"
            ;;

        ssl)
            print_info "Running SSL simulation..."

            run_ssl_simulation

            print_success "SSL simulation completed!"
            ;;

        benchmark)
            print_info "Running benchmarks..."

            benchmark_inference
            benchmark_training
            [ "$STAGE_COMPARE" = "true" ] && compare_frameworks

            print_success "Benchmarking completed!"
            ;;

        compare)
            print_info "Running comparison..."

            compare_frameworks
            generate_summary_report

            print_success "Comparison completed!"
            ;;

        config)
            show_config
            ;;

        clean)
            clean
            ;;

        help)
            show_help
            ;;
    esac

    echo ""
    print_success "Done!"
}

main
