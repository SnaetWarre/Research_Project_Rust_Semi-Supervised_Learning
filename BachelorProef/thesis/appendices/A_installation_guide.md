# Appendix A: Installation and User Guide

## A.1 Prerequisites

| Requirement | Version | Purpose |
|:---|:---|:---|
| Rust toolchain | 1.78+ | Compiler and package manager |
| NVIDIA CUDA Toolkit | 12.x | GPU-accelerated training and inference |
| Git | 2.x | Source code retrieval |
| Bun | 1.x | Frontend dependency management (GUI only) |

For CPU-only builds, the CUDA Toolkit is not required.

## A.2 Installation

### A.2.1 Clone the Repository

```bash
git clone https://github.com/[TODO: repository-url]
cd Fast_Research_Project
```

### A.2.2 Download the Dataset

The project uses the New Plant Diseases Dataset from Kaggle:

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d plantvillage_ssl/data/plantvillage/

# Option 2: Manual download
# Download from https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
# Extract to plantvillage_ssl/data/plantvillage/
```

Expected directory structure after extraction:

```
plantvillage_ssl/data/plantvillage/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   └── ... (38 class folders)
└── valid/
    ├── Apple___Apple_scab/
    └── ... (38 class folders)
```

### A.2.3 Build the Project

**With GPU (CUDA):**
```bash
cd plantvillage_ssl
cargo build --release
```

**CPU only:**
```bash
cd plantvillage_ssl
cargo build --release --features cpu
```

## A.3 Training the Model

### A.3.1 Step 1: Initial CNN Training (Supervised)

```bash
cd plantvillage_ssl
cargo run --release --bin plantvillage_ssl -- train \
    --epochs 30 \
    --cuda \
    --labeled-ratio 0.2
```

| Parameter | Description |
|:---|:---|
| `--epochs 30` | Number of training epochs |
| `--cuda` | Use GPU acceleration |
| `--labeled-ratio 0.2` | Use 20% of data for supervised training |

The trained model is saved to `output/models/plant_classifier_TIMESTAMP`.

### A.3.2 Step 2: SSL Simulation (Pseudo-Labeling)

```bash
cargo run --release --bin plantvillage_ssl -- simulate \
    --model "output/models/plant_classifier_TIMESTAMP" \
    --data-dir "data/plantvillage" \
    --cuda \
    --days 0 \
    --images-per-day 100 \
    --labeled-ratio 0.2 \
    --retrain-threshold 200 \
    --confidence-threshold 0.9
```

| Parameter | Description |
|:---|:---|
| `--model` | Path to the trained model from Step 1 |
| `--days 0` | Process all available SSL stream data |
| `--images-per-day 100` | Batch size per streaming "day" |
| `--retrain-threshold 200` | Retrain after accumulating 200 pseudo-labels |
| `--confidence-threshold 0.9` | Only accept predictions with >90% confidence |

### A.3.3 Step 3: Copy Best Model

```bash
cp plantvillage_ssl/output/simulation/plant_classifier_ssl_TIMESTAMP.mpk \
   plantvillage_ssl/best_model.mpk
```

## A.4 Running Experiments

```bash
# Label efficiency experiment
cargo run --release --bin plantvillage_ssl -- experiment label-efficiency

# Class scaling experiment
cargo run --release --bin plantvillage_ssl -- experiment class-scaling

# New class position experiment
cargo run --release --bin plantvillage_ssl -- experiment new-class-position
```

Results are written to `output/experiments/<experiment-name>/results.json` and `conclusions.txt`. SVG plots are generated automatically.

## A.5 Running the GUI Application

### A.5.1 Development Mode

```bash
cd plantvillage_ssl/gui
bun install
bun run tauri:dev
```

### A.5.2 Production Build

```bash
cd plantvillage_ssl/gui
bun run tauri:build
```

The compiled application is in `gui/src-tauri/target/release/`.

### A.5.3 iOS Deployment (Tauri)

```bash
cargo tauri ios build
# Deploy via Xcode or TestFlight
```

## A.6 CLI Reference

```bash
# View all available commands
cargo run --release --bin plantvillage_ssl -- --help

# View training options
cargo run --release --bin plantvillage_ssl -- train --help

# View simulation options
cargo run --release --bin plantvillage_ssl -- simulate --help

# View experiment options
cargo run --release --bin plantvillage_ssl -- experiment --help
```

## A.7 Troubleshooting

| Issue | Solution |
|:---|:---|
| CUDA not found | Ensure CUDA Toolkit is installed and `nvcc` is on PATH |
| Out of GPU memory | Reduce batch size or use `--features cpu` |
| Dataset not found | Verify the dataset is extracted to `data/plantvillage/` with `train/` and `valid/` subdirectories |
| Slow compilation | Use `cargo check` for development; reserve `--release` for benchmarks |
| Model file not found | Check the exact timestamp in the model filename |
