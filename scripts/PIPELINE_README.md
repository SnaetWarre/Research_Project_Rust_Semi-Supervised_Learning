# Research Pipeline

This directory contains the complete research automation pipeline for the PlantVillage semi-supervised learning project.

## Quick Start

**Edit the config once, run everything:**

```bash
# 1. Edit configuration (optional - defaults are sensible)
vim pipeline_config.yaml

# 2. Run the entire pipeline
./run_research_pipeline.sh all
```

That's it! The script reads all settings from `pipeline_config.yaml`.

## Configuration (pipeline_config.yaml)

The YAML file is the **single source of truth** for all pipeline settings:

```yaml
# Key sections:

paths:
  data_dir: "plantvillage_ssl/data/plantvillage/organized"
  output_dir: "output/research_pipeline"

dataset:
  labeled_ratio: 0.30      # 30% for supervised training
  validation_split: 0.10   # 10% for validation
  test_split: 0.10         # 10% for final testing
  # Remaining 50% → SSL stream pool (uses ALL unlabeled images)

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

ssl:
  enabled: true
  confidence_threshold: 0.90
  use_all_unlabeled: true  # Process ALL remaining images

stages:
  download_dataset: true
  train_burn: true
  train_pytorch: true
  ssl_simulation: true
  benchmark: true
  compare: true
```

### Data Split Strategy

| Split | Percentage | Purpose |
|-------|------------|---------|
| Labeled Training | 30% | Initial supervised learning (Burn + PyTorch) |
| Validation | 10% | Monitor training progress |
| Test | 10% | Final evaluation only |
| **SSL Stream Pool** | **50%** | All remaining images for semi-supervised learning |

With ~61,486 PlantVillage images:
- ~18,446 labeled training images
- ~30,742 images for SSL (maximized!)

## Pipeline Commands

| Command | Description |
|---------|-------------|
| `./run_research_pipeline.sh all` | Run entire pipeline |
| `./run_research_pipeline.sh train` | Train Burn and PyTorch models |
| `./run_research_pipeline.sh ssl` | Run SSL simulation only |
| `./run_research_pipeline.sh benchmark` | Run benchmarks |
| `./run_research_pipeline.sh compare` | Generate comparison reports |
| `./run_research_pipeline.sh config` | Show current configuration |
| `./run_research_pipeline.sh clean` | Clean all outputs |
| `./run_research_pipeline.sh help` | Show help |

### Override Config from CLI

CLI arguments override YAML values:

```bash
# Override epochs
./run_research_pipeline.sh all --epochs 20

# Override multiple settings
./run_research_pipeline.sh train --epochs 10 --batch-size 64

# Dry run (show commands without executing)
./run_research_pipeline.sh all --dry-run --verbose
```

## Pipeline Stages

When running `all`, stages execute in order:

```
1. Download Dataset  → Download PlantVillage if not present
         ↓
2. Train Burn       → Train Rust/Burn model (30% labeled data)
         ↓
3. Train PyTorch    → Train PyTorch model (30% labeled data)
         ↓
4. SSL Simulation   → Pseudo-labeling on ALL unlabeled data (50%)
         ↓
5. Benchmark        → Inference benchmarks for both frameworks
         ↓
6. Compare          → Generate comparison charts and reports
         ↓
7. Summary          → Final summary report
```

Toggle stages on/off in `pipeline_config.yaml`:

```yaml
stages:
  download_dataset: true
  train_burn: true
  train_pytorch: false   # Skip PyTorch training
  ssl_simulation: true
  benchmark: true
  compare: true
```

## Output Structure

```
output/research_pipeline/
├── burn/                       # Burn (Rust) outputs
│   ├── model_*.mpk             # Trained model
│   └── training_time.json      # Training metrics
├── pytorch/                    # PyTorch outputs
│   ├── *.pth                   # Trained model
│   └── training_time.json      # Training metrics
├── ssl/                        # SSL simulation results
│   ├── pseudo_labels.json      # Pseudo-label history
│   └── accuracy_progress.json  # Accuracy over time
├── benchmark/                  # Benchmark results
│   ├── burn.json               # Burn latency/throughput
│   └── pytorch.json            # PyTorch latency/throughput
├── *.png                       # Comparison charts
└── research_summary.txt        # Final summary report
```

## Prerequisites

- **Rust** 1.70+ with cargo
- **Python** 3.10+ with pip
- **yq** (YAML parser) - optional but recommended
  ```bash
  # Arch Linux
  sudo pacman -S yq
  
  # Or via pip
  pip install yq
  ```
- **NVIDIA GPU** with CUDA 12.x (optional, for GPU acceleration)
- ~5GB disk space for dataset

## Troubleshooting

### Check Configuration

```bash
# View current config values
./run_research_pipeline.sh config
```

### Dry Run

```bash
# See what would be executed without running
./run_research_pipeline.sh all --dry-run --verbose
```

### Start Fresh

```bash
# Clean everything and start over
./run_research_pipeline.sh clean
./run_research_pipeline.sh all
```

### YAML Parser Not Found

If `yq` is not installed, the script falls back to basic grep parsing (limited). Install `yq` for full YAML support:

```bash
sudo pacman -S yq
```
