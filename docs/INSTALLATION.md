# PlantVillage SSL - Installation Manual (Cold Start)
**Target Device:** NVIDIA Jetson Orin Nano / Desktop Linux
**Version:** 1.0

This guide explains how to deploy the application from scratch (Cold Start).

## 1. Prerequisites

Before starting, ensure your hardware is ready:
*   **Device:** NVIDIA Jetson Orin Nano (8GB) or Linux PC with NVIDIA GPU.
*   **OS:** Ubuntu 20.04 or 22.04 (JetPack 5.x/6.x for Jetson).
*   **Internet:** Required for initial download of crates and dataset.

## 2. Install System Dependencies

Open a terminal and run the following commands to install build tools and libraries:

```bash
# Update package list
sudo apt update

# Install build tools and SSL libraries
sudo apt install -y build-essential cmake libssl-dev pkg-config libclang-dev curl git
```

## 3. Install Rust (The Compiler)

We use Rust for the core machine learning engine.

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# configure current shell
source "$HOME/.cargo/env"

# Verify installation
rustc --version
# Expected output: rustc 1.7x.x ...
```

## 4. Install Tauri Dependencies (For GUI)

The GUI requires specific libraries for the window manager.

```bash
sudo apt install -y libwebkit2gtk-4.0-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
```

And install the **Bun** runtime (faster than Node.js) for the frontend:

```bash
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

## 5. Clone & Build the Project

```bash
# 1. Clone Repository
git clone https://github.com/YourUsername/plantvillage_ssl.git
cd plantvillage_ssl

# 2. Build the Rust Backend (Release mode for speed)
# Note: This takes 5-10 minutes on the first run.
cd plantvillage_ssl
cargo build --release --features cuda

# 3. Setup the GUI
cd gui
bun install
```

## 6. Download Dataset

The application needs the PlantVillage dataset to function.

```bash
# From the plantvillage_ssl folder:
./scripts/download_dataset.sh
```
*Note: This script requires `kaggle` CLI installed or manual download. See `README.md` for manual instructions.*

## 7. Running the Application

Everything is now installed. To start the app:

```bash
# From plantvillage_ssl/gui folder:
bun run tauri:dev
```

The application window should appear on your screen.

---

## Troubleshooting

*   **"Error: CUDA not found"**: Ensure `nvcc` is in your PATH.
    *   `export PATH=/usr/local/cuda/bin:$PATH`
    *   `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH`
*   **"Linker error"**: You are missing a system library. Run step 2 again.
