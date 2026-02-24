# PlantVillage SSL Installation Guide

This guide provides step-by-step instructions for setting up the PlantVillage Semi-Supervised Learning project on both desktop systems (for training) and embedded CUDA-capable devices (for edge deployment).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Desktop Installation (Linux)](#desktop-installation-linux)
- [Desktop Installation (Windows)](#desktop-installation-windows)
 - [Embedded Device Installation](#embedded-device-installation)
- [Dataset Setup](#dataset-setup)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Desktop (for training):**
- CPU: 4+ cores recommended
- RAM: 16GB minimum, 32GB recommended
- GPU: NVIDIA GPU with 6GB+ VRAM (CUDA 12.x compatible)
- Storage: 10GB free space (5GB for dataset, 5GB for models/outputs)

- **Edge Device (for inference):**
- CUDA-capable embedded device (8GB)
- microSD card 64GB+ (or NVMe SSD recommended)
- USB camera (optional, for live inference)

### Software Requirements

- Rust 1.70 or later
- CUDA Toolkit 12.x (for GPU acceleration)
- Python 3.10+ (for dataset download and PyTorch reference)
- Git

---

## Desktop Installation (Linux)

### 1. Install System Dependencies

```bash
# Ubuntu/Debian
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
```

### 2. Install CUDA Toolkit

```bash
# Download CUDA 12.x from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Add CUDA to your PATH (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

## Embedded Device Installation

### 1. Install vendor SDK (if applicable)

For many embedded platforms the vendor provides an SDK that bundles CUDA/TensorRT components. Follow your device vendor's guide to install the appropriate SDK.

### 2. Run Setup Script

The project includes an automated setup script for the target platform. Run it on the device when applicable:

```bash
# Clone the repository
git clone <repository-url>
cd Source/plantvillage_ssl

# Make script executable and run
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 3. Manual Installation (if script fails)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libssl-dev libclang-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Set up CUDA paths
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Build (enable CUDA support)
cd Source/plantvillage_ssl
cargo build --release --features cuda
```

## Desktop Installation (Windows)

### 1. Install Visual Studio Build Tools

Download and install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Select "C++ build tools" workload.

### 2. Install CUDA Toolkit

Download and install [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-downloads) for Windows.

### 3. Install Rust

Download and run the [Rust installer](https://www.rust-lang.org/tools/install) for Windows.

### 4. Clone and Build

```powershell
# Clone the repository
git clone <repository-url>
cd Source\plantvillage_ssl

# Build with CUDA support
cargo build --release --features cuda
```

---

## Embedded Device Installation

### 1. Install vendor SDK (if applicable)

Ensure your target embedded device has the appropriate vendor SDK installed (for example, JetPack on NVIDIA platforms). Follow the vendor's installation guide for flashing and SDK setup.

### 2. Run Setup Script

The project includes an automated setup script for the target platform. Run it on the device when applicable:

```bash
# Clone the repository
git clone <repository-url>
cd Source/plantvillage_ssl

# Make script executable and run
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

The script will:
- Install system dependencies
- Install Rust
- Configure CUDA paths
- Build the project with Jetson optimizations
- Create output directories

### 3. Manual Installation (if script fails)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libssl-dev libclang-dev

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Set up CUDA paths
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Build
cd Source/plantvillage_ssl
cargo build --release --features cuda,jetson
```

---

## Dataset Setup

### Option 1: Download from Kaggle (Recommended)

```bash
# Install Kaggle CLI
pip install kaggle

# Set up Kaggle API credentials
# 1. Go to https://www.kaggle.com/settings/account
# 2. Click "Create New Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
cd Source
./download_plantvillage.sh
```

### Option 2: Manual Download

1. Visit [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
2. Download the dataset (click "Download" button)
3. Extract to `Source/plantvillage_ssl/data/plantvillage/`
4. Ensure `train/` and `valid/` directories are present under `data/plantvillage/`

### Expected Directory Structure

After setup, your data directory should look like:

```
data/plantvillage/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ... (38 class directories)
└── valid/
    ├── Apple___Apple_scab/
    └── ... (38 class directories)
```

---

## Verifying Installation

### 1. Check the Build

```bash
# Run the CLI with help
./target/release/plantvillage_ssl --help

# Expected output:
# PlantVillage Semi-Supervised Plant Disease Classification
# ...
```

### 2. Verify Dataset

```bash
# Check dataset statistics
./target/release/plantvillage_ssl stats --data-dir data/plantvillage
```

### 3. Test CUDA (GPU only)

```bash
# Check CUDA availability
nvidia-smi

# The application will automatically detect CUDA when running with --cuda flag
./target/release/plantvillage_ssl train --cuda --help
```

### 4. Run Unit Tests

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA not found

```
Error: CUDA not found
```

**Solution:**
```bash
# Ensure CUDA paths are set
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Rebuild
cargo clean
cargo build --release --features cuda
```

#### 2. Out of Memory (GPU)

```
Error: CUDA out of memory
```

**Solution:** Reduce batch size:
```bash
./target/release/plantvillage_ssl train --batch-size 8
```

#### 3. Linker errors on build

```
Error: linking with `cc` failed
```

**Solution:**
```bash
# Install missing dependencies
sudo apt-get install build-essential cmake libclang-dev
```

#### 4. Dataset not found

```
Error: Dataset directory does not exist
```

**Solution:** Ensure the dataset path is correct:
```bash
# Check the path
ls data/plantvillage/train/

# If empty, re-download the dataset
cd ..
./download_plantvillage.sh
```

#### 5. Permission denied on Jetson

```
Error: Permission denied
```

**Solution:**
```bash
# Add user to dialout group (for USB devices)
sudo usermod -a -G dialout $USER

# Reboot
sudo reboot
```

### Getting Help

If you encounter issues not covered here:

1. Check the [README.md](../README.md) for additional information
2. Review the error message carefully - it often contains hints
3. Search existing issues in the repository
4. Create a new issue with:
   - Your operating system and version
   - Rust version (`rustc --version`)
   - CUDA version (`nvcc --version`)
   - Full error message
   - Steps to reproduce

---

## Next Steps

After successful installation:

1. **Train a model:**
   ```bash
   ./target/release/plantvillage_ssl train --data-dir data/plantvillage --epochs 50
   ```

2. **Run inference:**
   ```bash
   ./target/release/plantvillage_ssl infer --input <image.jpg> --model output/models/best_model.mpk
   ```

3. **Benchmark performance:**
   ```bash
   ./target/release/plantvillage_ssl benchmark --model output/models/best_model.mpk
   ```

4. **Read the [User Guide](user_guide.md)** for detailed usage instructions.