# Incremental Learning

Adding new plant disease classes to an existing model without forgetting old ones.

## Methods Implemented

- **Fine-tuning** - Baseline (will forget old classes)
- **LwF** - Learning without Forgetting (knowledge distillation)
- **EWC** - Elastic Weight Consolidation (protects important weights)
- **Rehearsal** - Replay old samples when learning new

## Quick Start

```bash
# 1. Download dataset (from repo root)
cd .. && ./download_plantvillage.sh && cd incremental_learning

# 2. Build
cargo build --release

# 3. Example: 5→6 class experiment with LwF
./target/release/plant-incremental experiment \
    --method lwf \
    --base-classes 5 \
    --new-classes 1 \
    --data-dir ../plantvillage_ssl/data/plantvillage/organized

# Example: 30→31 class experiment
./target/release/plant-incremental experiment \
    --method ewc \
    --base-classes 30 \
    --new-classes 1 \
    --data-dir ../plantvillage_ssl/data/plantvillage/organized
```

## Research Questions

1. Is 5→6 easier than 30→31? (expected: yes, more capacity left)
2. How many images needed per new class? (10? 50? 100?)
3. Which method works best on Jetson?

## Structure

```
crates/
├── plant-core/        # Shared utilities
├── plant-dataset/     # Data loading
├── plant-incremental/ # LwF, EWC, rehearsal, finetuning
└── plant-cli/         # CLI runner
```
