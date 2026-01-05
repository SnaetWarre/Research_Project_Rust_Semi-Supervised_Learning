# Semi-Supervised Plant Disease Classification with Burn

A semi-supervised neural network for plant disease classification implemented in Rust using the [Burn](https://burn.dev/) framework, designed for deployment on NVIDIA Jetson Orin Nano edge devices.

## Research Project

**Goal:** Implement and compare semi-supervised learning approaches for plant disease classification on edge devices.

**Research Question:** How can a semi-supervised neural network be efficiently implemented in Rust for automatic labeling of partially labeled datasets on an edge device?

## Dataset

- **PlantVillage Dataset**: 61,486 images of plant leaves
- **39 Classes**: Various plant diseases across 15 crop types
- **Resolution**: 256x256 RGB images
- **Semi-Supervised Setup**: 20-30% labeled data, 70-80% unlabeled for pseudo-labeling

## Features

- **GPU Acceleration**: CUDA backend for training and inference
- **Semi-Supervised Learning**: Pseudo-labeling with confidence thresholds
- **Stream Simulation**: Simulates real-time camera input from edge devices
- **Edge Deployment**: Optimized for NVIDIA Jetson Orin Nano
- **Benchmarking**: Full comparison with PyTorch reference implementation

## Project Structure

```
plantvillage_ssl/          # Main Rust/Burn implementation
├── src/
│   ├── dataset/           # Data loading and augmentation
│   ├── model/             # CNN architecture
│   ├── training/          # Training loop & pseudo-labeling
│   ├── inference/         # Inference & benchmarking
│   └── utils/             # Logging & metrics
├── scripts/               # Setup & dataset download scripts
└── output/                # Model weights and results

pytorch_reference/         # PyTorch comparison implementation
benchmarks/                 # Performance comparison results
docs/                      # Documentation (installation, user guide)
archive/                   # Old CIFAR-10 implementation (archived)
```

## Requirements

### Desktop (Training)
- Rust 1.70+
- NVIDIA GPU with CUDA 12.x drivers
- Python 3.10+ (for PyTorch reference)

### Jetson Orin Nano (Inference)
- NVIDIA Jetson Orin Nano 8GB
- JetPack SDK
- CUDA 12.x + cuDNN 9.x

## Installation

### Desktop Setup

```bash
# Clone repository
git clone <repository-url>
cd Source

# Install Rust dependencies
cd plantvillage_ssl
cargo build --release --features cuda

# Download PlantVillage dataset
python scripts/download_dataset.py
```

### Jetson Setup

```bash
# Run setup script on Jetson device
./scripts/setup_jetson.sh
```

See [docs/installation.md](docs/installation.md) for detailed instructions.

## 🚀 Quick Start - Run Full Research Pipeline

The easiest way to run the entire research project is with the pipeline script:

```bash
# Run everything with one command
./run_research_pipeline.sh all

# Run with custom configuration
./run_research_pipeline.sh all --epochs 20 --batch-size 32
```

This will:
1. Download PlantVillage dataset
2. Train Burn (Rust) model
3. Train PyTorch model
4. Run semi-supervised simulation
5. Benchmark both frameworks
6. Generate comparison reports

See [scripts/PIPELINE_README.md](scripts/PIPELINE_README.md) for complete pipeline documentation.

## Usage

### Quick Start (Recommended)

```bash
# Run full research pipeline
./run_research_pipeline.sh all

# Or run specific stages
./run_research_pipeline.sh train --epochs 50    # Train only
./run_research_pipeline.sh benchmark            # Benchmark only
./run_research_pipeline.sh ssl                 # SSL simulation only
```

### Individual Commands (Desktop)

#### Training

```bash
# Train with semi-supervised learning
cd plantvillage_ssl
cargo run --release -- train \
    --data-dir data/plantvillage \
    --epochs 50 \
    --batch-size 32 \
    --labeled-ratio 0.3 \
    --pseudo-threshold 0.9
```

#### Inference

```bash
# Run single image prediction
cargo run --release -- predict \
    --model output/models/plant_classifier_YYYYMMDD_HHMMSS.mpk \
    --image path/to/image.jpg

# Benchmark inference performance
cargo run --release -- benchmark \
    --iterations 100
```

#### Semi-Supervised Simulation

```bash
# Simulate camera stream with pseudo-labeling
cargo run --release -- simulate \
    --data-dir data/plantvillage \
    --batch-size 500 \
    --days 10
```

See [docs/user_guide.md](docs/user_guide.md) for complete usage instructions.

## Performance Targets

- **Primary Metric**: Inference latency < 200ms per image on Jetson Orin Nano
- **Secondary Metric**: Accuracy ≥ 85% on test set
- **Semi-Supervised Impact**: ≥ 5% accuracy improvement over labeled-only baseline

## Architecture

**CNN Model (ResNet-18 lite)**
- Input: 224x224x3
- 4 Convolutional blocks with BatchNorm and ReLU
- AdaptiveAvgPool + Dropout(0.3)
- Output: 39 classes (Softmax)

## Research Deliverables

- [ ] Working Rust/Burn codebase
- [ ] Trained model weights on PlantVillage
- [ ] Benchmark comparison (Burn vs PyTorch)
- [ ] Jetson Orin Nano deployment
- [ ] Demo-ready application with real-time predictions
- [ ] Complete documentation

## Documentation

- [Installation Guide](docs/installation.md) - Setup for desktop and Jetson
- [User Guide](docs/user_guide.md) - Usage and troubleshooting
- [Research Contract](ResearchProject_Contractplan_Warre_Snaet%20(1).md) - Full research plan
- [Implementation Plan](STAPPENPLAN_UPDATE_TO_PLANTVILLAGE.md) - Technical details

## Comparison: Burn vs PyTorch

### Using Pipeline Script (Recommended)

```bash
# Run full benchmark comparison
./run_research_pipeline.sh benchmark --iterations 100

# Or use the full pipeline which includes comparison
./run_research_pipeline.sh all
```

### Manual Comparison

```bash
# Train PyTorch baseline
cd pytorch_reference
python trainer.py --data-dir ../plantvillage_ssl/data/plantvillage --epochs 50

# Run comparison benchmarks
cd ../benchmarks
python compare_frameworks.py
```

## License

Research Project - Howest University

## Acknowledgments

- PlantVillage Dataset: [PSU PlantVillage](https://plantvillage.psu.edu/)
- Burn Framework: [Tracel AI](https://github.com/tracel-ai/burn)
- Research Supervisor: Gilles Depypere
