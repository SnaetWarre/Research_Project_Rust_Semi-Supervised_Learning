#!/bin/bash
# ============================================================================
# PlantVillage SSL - Jetson Orin Nano Setup Script
# ============================================================================
#
# This script sets up the NVIDIA Jetson Orin Nano for running the PlantVillage
# semi-supervised learning application.
#
# Requirements:
#   - NVIDIA Jetson Orin Nano (8GB recommended)
#   - JetPack 5.x or later
#   - Internet connection for downloading dependencies
#
# Usage:
#   chmod +x setup_jetson.sh
#   ./setup_jetson.sh
#
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUST_VERSION="1.75.0"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================================================================
# Helper Functions
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_jetson() {
    log_info "Checking if running on Jetson..."

    if [ -f /etc/nv_tegra_release ]; then
        log_success "Jetson platform detected"
        cat /etc/nv_tegra_release
        return 0
    else
        log_warning "Not running on Jetson - some features may not work"
        return 1
    fi
}

check_cuda() {
    log_info "Checking CUDA installation..."

    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
        log_success "CUDA $CUDA_VERSION found"
        return 0
    else
        log_warning "CUDA not found in PATH"

        # Check common Jetson CUDA locations
        if [ -d "/usr/local/cuda" ]; then
            log_info "Found CUDA at /usr/local/cuda"
            export PATH="/usr/local/cuda/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
        fi
        return 1
    fi
}

install_system_dependencies() {
    log_info "Installing system dependencies..."

    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        libssl-dev \
        libclang-dev \
        libjpeg-dev \
        libpng-dev \
        curl \
        git

    log_success "System dependencies installed"
}

install_rust() {
    log_info "Checking Rust installation..."

    if command -v rustc &> /dev/null; then
        CURRENT_VERSION=$(rustc --version | awk '{print $2}')
        log_success "Rust $CURRENT_VERSION already installed"

        # Update if needed
        log_info "Updating Rust toolchain..."
        rustup update stable
    else
        log_info "Installing Rust $RUST_VERSION..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        log_success "Rust installed"
    fi

    # Add required targets
    log_info "Adding Rust targets..."
    rustup target add aarch64-unknown-linux-gnu || true
}

setup_cuda_for_rust() {
    log_info "Setting up CUDA environment for Rust..."

    # Create/update .bashrc with CUDA paths
    CUDA_PATHS='
# CUDA Configuration for PlantVillage SSL
export CUDA_HOME="/usr/local/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
'

    if ! grep -q "CUDA Configuration for PlantVillage" "$HOME/.bashrc"; then
        echo "$CUDA_PATHS" >> "$HOME/.bashrc"
        log_success "Added CUDA paths to .bashrc"
    else
        log_info "CUDA paths already in .bashrc"
    fi

    # Source for current session
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
}

build_project() {
    log_info "Building PlantVillage SSL project..."

    cd "$PROJECT_DIR"

    # Build with Jetson optimizations
    log_info "Building in release mode with Jetson optimizations..."

    # Set environment for CUDA
    export LIBTORCH_USE_PYTORCH=1

    # Build with CUDA feature
    cargo build --release --features cuda 2>&1 | tee build.log

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "Build completed successfully"

        # Show binary size
        BINARY="target/release/plantvillage_ssl"
        if [ -f "$BINARY" ]; then
            SIZE=$(du -h "$BINARY" | cut -f1)
            log_info "Binary size: $SIZE"
        fi
    else
        log_error "Build failed - check build.log for details"
        return 1
    fi
}

create_output_directories() {
    log_info "Creating output directories..."

    mkdir -p "$PROJECT_DIR/data/plantvillage"
    mkdir -p "$PROJECT_DIR/output/models"
    mkdir -p "$PROJECT_DIR/output/logs"
    mkdir -p "$PROJECT_DIR/output/benchmarks"
    mkdir -p "$PROJECT_DIR/output/simulation"

    log_success "Output directories created"
}

run_benchmark() {
    log_info "Running initial benchmark..."

    cd "$PROJECT_DIR"

    # Check if binary exists
    if [ ! -f "target/release/plantvillage_ssl" ]; then
        log_warning "Binary not found, skipping benchmark"
        return
    fi

    # Run benchmark (placeholder - needs trained model)
    log_info "Benchmark requires a trained model - skipping for now"
    log_info "Run 'plantvillage_ssl benchmark' after training a model"
}

check_gpu_memory() {
    log_info "Checking GPU memory..."

    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    elif [ -f /sys/kernel/debug/nvmap/iovmm/clients ]; then
        # Jetson-specific memory check
        log_info "Tegra memory status:"
        cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
    fi
}

setup_systemd_service() {
    log_info "Setting up systemd service (optional)..."

    SERVICE_FILE="/etc/systemd/system/plantvillage-ssl.service"

    read -p "Do you want to create a systemd service? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=PlantVillage Semi-Supervised Learning Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/target/release/plantvillage_ssl simulate --data-dir data/plantvillage --model output/models/best_model.bin
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

        sudo systemctl daemon-reload
        log_success "Systemd service created"
        log_info "Enable with: sudo systemctl enable plantvillage-ssl"
        log_info "Start with: sudo systemctl start plantvillage-ssl"
    fi
}

print_summary() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}PlantVillage SSL Setup Complete!${NC}"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Download the PlantVillage dataset:"
    echo "   - Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    echo "   - Extract to: $PROJECT_DIR/data/plantvillage/"
    echo ""
    echo "2. Train the model:"
    echo "   cd $PROJECT_DIR"
    echo "   ./target/release/plantvillage_ssl train --data-dir data/plantvillage --epochs 50"
    echo ""
    echo "3. Run inference:"
    echo "   ./target/release/plantvillage_ssl infer --input <image.jpg> --model output/models/best.bin"
    echo ""
    echo "4. Run benchmarks:"
    echo "   ./target/release/plantvillage_ssl benchmark --model output/models/best.bin"
    echo ""
    echo "============================================================"
    check_gpu_memory
    echo "============================================================"
}

# ============================================================================
# Main Script
# ============================================================================

main() {
    echo ""
    echo "============================================================"
    echo "  PlantVillage SSL - Jetson Orin Nano Setup"
    echo "============================================================"
    echo ""

    # Check platform
    check_jetson || true

    # Check CUDA
    check_cuda || true

    # Install dependencies
    install_system_dependencies

    # Install Rust
    install_rust

    # Setup CUDA environment
    setup_cuda_for_rust

    # Create directories
    create_output_directories

    # Build project
    build_project

    # Run initial benchmark
    run_benchmark || true

    # Optional systemd setup
    # setup_systemd_service

    # Print summary
    print_summary
}

# Run main function
main "$@"
