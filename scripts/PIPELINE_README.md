# Research Pipeline

This directory contains the complete research automation pipeline for the PlantVillage semi-supervised learning project.

## Quick Start

Run the entire research pipeline with a single command:

```bash
./run_research_pipeline.sh all
```

This will:
1. Download the PlantVillage dataset (if needed)
2. Train the Burn/Rust model
3. Train the PyTorch model
4. Run semi-supervised simulation
5. Benchmark both frameworks
6. Generate comparison reports and charts

## Pipeline Script

The `run_research_pipeline.sh` script is your main entry point.

### Commands

| Command | Description |
|---------|-------------|
| `all` | Run entire pipeline (default) |
| `train` | Train both Burn and PyTorch models |
| `ssl` | Run semi-supervised simulation |
| `benchmark` | Run benchmarks and comparisons |
| `compare` | Compare frameworks and generate reports |
| `clean` | Clean all outputs and build artifacts |
| `help` | Show help message |

### Examples

```bash
# Run full pipeline with 20 epochs
./run_research_pipeline.sh all --epochs 20

# Train only with batch size 64
./run_research_pipeline.sh train --batch-size 64 --epochs 10

# Run benchmarks with 500 iterations
./run_research_pipeline.sh benchmark --iterations 500

# Clean everything
./run_research_pipeline.sh clean
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs N` | Number of training epochs | 50 |
| `--batch-size N` | Batch size | 32 |
| `--iterations N` | Benchmark iterations | 100 |
| `--image-size N` | Image size | 224 |
| `--data-dir PATH` | Dataset directory | data/plantvillage |
| `--output-dir PATH` | Output directory | output/research_pipeline |
| `--skip-*` | Skip specific stages | false |
| `--dry-run` | Show commands without executing | false |
| `--verbose` | Show detailed output | false |

## Configuration

Edit `pipeline_config.yaml` to change default settings:

```yaml
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

semi_supervised:
  enabled: true
  confidence_threshold: 0.90

benchmarking:
  iterations: 100
  compare_with_pytorch: true
```

## Pipeline Stages

When running `all`, the following stages are executed in order:

1. **Download Dataset** - Downloads PlantVillage dataset if not present
2. **Train Burn** - Trains Rust/Burn model with supervised learning
3. **Train PyTorch** - Trains PyTorch model with supervised learning
4. **SSL Simulation** - Runs semi-supervised pseudo-labeling simulation
5. **Benchmarking** - Runs inference and training benchmarks
6. **Compare** - Generates comparison charts and reports
7. **Summary** - Generates final summary report

## Output Structure

After running the pipeline, you'll find:

```
output/research_pipeline/
├── burn/                    # Burn model outputs
│   ├── plant_classifier_*.mpk  # Trained model (timestamped)
│   ├── training_time.json   # Training metrics
│   └── metrics.json        # Performance metrics
├── pytorch/                # PyTorch model outputs
│   ├── model.pth           # Trained model
│   ├── training_time.json   # Training metrics
│   └── metrics.json        # Performance metrics
├── ssl/                    # Semi-supervised simulation results
│   ├── pseudo_labels.json   # Pseudo-label history
│   └── accuracy_progress.json
├── benchmark/              # Benchmark results
│   ├── burn_benchmark.json  # Burn metrics
│   ├── pytorch_benchmark.json
│   └── benchmark_comparison.json
├── *.png                   # Comparison charts
└── research_summary.txt    # Final summary report
```

## Individual Scripts

### run_benchmarks.sh

Legacy benchmarking script. Use `run_research_pipeline.sh benchmark` instead.

### run_research_pipeline.sh

Main pipeline script that orchestrates everything.

### download_dataset.py

Located in `plantvillage_ssl/scripts/`. Downloads the PlantVillage dataset.

## Prerequisites

- Rust 1.70+ with cargo
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (optional, for GPU acceleration)
- ~5GB disk space for dataset

## Examples

### Quick Test Run

```bash
# Run a quick test with 5 epochs
./run_research_pipeline.sh all --epochs 5 --batch-size 16
```

### Production Training

```bash
# Full training with 50 epochs
./run_research_pipeline.sh all --epochs 50 --batch-size 32 --verbose
```

### Benchmarking Only

```bash
# Run extensive benchmarks
./run_research_pipeline.sh benchmark --iterations 1000 --warmup 50
```

### Troubleshooting

```bash
# Run with verbose output to see all commands
./run_research_pipeline.sh all --verbose

# Dry run to see what will be executed
./run_research_pipeline.sh all --dry-run

# Clean everything and start fresh
./run_research_pipeline.sh clean
```

## Integration with Existing Workflows

The pipeline scripts are designed to work with your existing code:

- Uses `plantvillage_ssl/` CLI commands for Burn
- Uses `pytorch_reference/trainer.py` for PyTorch
- Uses `benchmarks/compare_frameworks.py` for comparisons
- Outputs are organized in `output/` directory

## Next Steps

1. Run `./run_research_pipeline.sh help` to see all options
2. Try a test run: `./run_research_pipeline.sh all --epochs 2`
3. Check outputs in `output/research_pipeline/`
4. Review charts and summary report
5. Adjust `pipeline_config.yaml` as needed
