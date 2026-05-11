# PlantVillage SSL

Semi-supervised learning implementation for plant disease classification using Rust + Burn framework.

## Quick Start

```bash
# 1. Download dataset (from repo root)
cd .. && ./download_plantvillage.sh && cd plantvillage_ssl

# 2. Build
cargo build --release

# 3. Train (20% labeled)
./target/release/plantvillage_ssl train \
    --data-dir data/plantvillage \
    --labeled-ratio 0.2 \
    --epochs 30 --cuda

# Inference
./target/release/plantvillage_ssl infer \
    --model-path output/models/best_model.mpk \
    --image-path /path/to/leaf.jpg

# Benchmark (embedded device)
./target/release/plantvillage_ssl benchmark \
    --model-path output/models/best_model.mpk
```

## Dataset Splitting

When the Kaggle dataset contains `train/` and `valid/`, the loader merges both
folders first. The project then creates its own deterministic SSL split:
10% test, 10% validation, 20% labeled training, and 60% unlabeled stream by
default. The Kaggle `valid/` folder is therefore treated as additional source
data, not as the validation set used in reported metrics.

## Structure

```
src/
├── model/      # CNN architecture (32→64→128→256 filters)
├── training/   # SSL pseudo-labeling
├── inference/  # Inference pipeline
├── dataset/    # Data loading
└── utils/      # Helpers
```

## Deployment

Works on CUDA-capable embedded devices without code changes (CUDA backend).
See `docs/` for detailed installation and user guide.
