#!/bin/bash
# Setup script for PyTorch on NVIDIA Jetson (JetPack 6.x / R36)
# This enables the burn-tch backend to use CUDA via LibTorch
#
# Run this script on the Jetson device:
#   bash setup-jetson-pytorch.sh

set -e

echo "=== Setting up PyTorch for Jetson (JetPack 6.x) ==="

# Check if we're on a Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: /etc/nv_tegra_release not found. This might not be a Jetson device."
fi

# Get JetPack version info
echo "Tegra release info:"
cat /etc/nv_tegra_release 2>/dev/null || echo "Could not read tegra release"

# Install pip if not available
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip3..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y libopenblas-dev python3-dev

# Install cusparselt (required for PyTorch 24.06+)
echo "Installing cusparselt..."
if [ ! -f /usr/local/lib/libcusparseLt.so ]; then
    wget -q https://raw.githubusercontent.com/pytorch/pytorch/5c6af2b583709f6176898c017424dc9981023c28/.ci/docker/common/install_cusparselt.sh -O /tmp/install_cusparselt.sh
    export CUDA_VERSION=12.6
    sudo bash /tmp/install_cusparselt.sh || echo "cusparselt installation may have failed, continuing..."
    rm -f /tmp/install_cusparselt.sh
else
    echo "cusparselt already installed"
fi

# Determine which PyTorch wheel to install based on JetPack version
# R36.4.x is JetPack 6.1+ (use 24.09 wheel for Python 3.10)
# See: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/pytorch-jetson-rel.html

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# For JetPack 6.1 (R36.4.x), use the 24.09 wheel
# PyTorch 2.5.0a0 with CUDA 12.6 support
TORCH_WHEEL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+b465a5843b.nv24.09-cp310-cp310-linux_aarch64.whl"

echo "Installing PyTorch from NVIDIA wheel..."
echo "Wheel: $TORCH_WHEEL"

# Upgrade pip first
python3 -m pip install --upgrade pip

# Install numpy (required by PyTorch)
python3 -m pip install 'numpy<2'

# Install PyTorch
python3 -m pip install --no-cache "$TORCH_WHEEL"

# Verify installation
echo ""
echo "=== Verifying PyTorch installation ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
    
    # Quick GPU test
    x = torch.randn(100, 100, device='cuda')
    y = torch.randn(100, 100, device='cuda')
    z = torch.matmul(x, y)
    print(f'GPU compute test: PASSED (result shape: {z.shape})')
else:
    print('WARNING: CUDA not available!')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To build plantvillage_ssl with LibTorch backend:"
echo "  export LIBTORCH_USE_PYTORCH=1"
echo "  cargo build --profile release-jetson --features jetson --no-default-features"
echo ""
echo "You may want to add this to your ~/.bashrc:"
echo "  export LIBTORCH_USE_PYTORCH=1"
