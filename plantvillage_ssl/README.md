# PlantVillage SSL

Semi-supervised learning implementation for plant disease classification using Rust + Burn framework.

## Quick Start

```bash
# 1. Download dataset (from repo root)
cd .. && ./download_plantvillage.sh && cd plantvillage_ssl

# 2. Build
cargo build --release

# 3. Train with SSL (30% labeled)
./target/release/plantvillage_ssl ssl-train \
    --data-dir data/plantvillage/organized \
    --labeled-ratio 0.3 \
    --epochs 30 --cuda

# Inference
./target/release/plantvillage_ssl infer \
    --model-path output/models/best_model.mpk \
    --image-path /path/to/leaf.jpg

# Benchmark (embedded device)
./target/release/plantvillage_ssl benchmark \
    --model-path output/models/best_model.mpk
```

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
