# GPU Neural Network for CIFAR-10 Semi-Supervised Learning

A GPU-only neural network library built from scratch using CUDA for semi-supervised learning on CIFAR-10.

## Features

- **Full GPU acceleration**: All operations run on GPU with minimal CPU transfers
- **Custom CUDA kernels**: Element-wise operations, activations, and matrix operations
- **Semi-supervised learning**: Pseudo-labeling support for learning from unlabeled data
- **No external ML frameworks**: Built entirely from scratch using CUDA and cuBLAS

## Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- Rust toolchain

## Building

```bash
# Build with CUDA support
cargo build --release --features cuda

# Build training binary
cargo build --release --features cuda --bin gpu_train

# Build labeling binary
cargo build --release --features cuda --bin gpu_label
```

## Usage

### Training

Train a model with semi-supervised learning:

```bash
cargo run --release --features cuda --bin gpu_train -- \
    --data-root data/cifar10/cifar-10-batches-bin \
    --labeled-fraction 0.1 \
    --batch 256 \
    --epochs 100 \
    --lr 0.01 \
    --momentum 0.9 \
    --dropout 0.3 \
    --pseudo-threshold 0.95 \
    --warmup-epochs 5
```

**Arguments:**
- `--data-root`: Path to CIFAR-10 binary directory
- `--labeled-fraction`: Fraction of data to use as labeled (default: 0.1)
- `--batch`: Batch size (default: 256)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.01)
- `--momentum`: Momentum for SGD (default: 0.9)
- `--dropout`: Dropout rate (default: 0.3)
- `--pseudo-threshold`: Confidence threshold for pseudo-labeling (default: 0.95)
- `--warmup-epochs`: Epochs of supervised training before pseudo-labeling (default: 5)
- `--ema`: Use exponential moving average (0 or 1, default: 0)

**Outputs:**
- `artifacts/model.json`: Model architecture metadata
- `artifacts/weights.bin`: Model weights (binary)
- `artifacts/metrics.json`: Training metrics
- `artifacts/pseudo_labeling_history.json`: Pseudo-labeling history

### Labeling

Label images using a trained model:

```bash
cargo run --release --features cuda --bin gpu_label -- \
    --model artifacts \
    --input data/cifar10/cifar-10-batches-bin \
    --output labels.csv \
    --confidence-out confidences.csv
```

**Arguments:**
- `--model`: Path to model directory (contains model.json and weights.bin)
- `--input`: Path to CIFAR-10 binary directory or image directory
- `--output`: Output CSV file path (default: labels.csv)
- `--confidence-out`: Optional path to save confidence scores

**Output CSV format:**
```csv
filename,predicted_class,class_name,confidence
image_00000,0,airplane,0.923456
image_00001,1,automobile,0.876543
...
```

## Architecture

The network architecture:
- Input: 3072 (32×32×3 RGB)
- 1024 → 512 → 256 → 128 → 10
- Each dense layer followed by BatchNorm and Dropout
- ReLU activations, Linear output

## Library Structure

- `neural_net/`: Core GPU neural network library
  - `gpu_tensor.rs`: GPU tensor operations and CUDA kernels
  - `gpu_layer.rs`: GPU layers (Dense, BatchNorm, Dropout, Network, Optimizer, Loss)
  - `cifar10.rs`: CIFAR-10 dataset loading
  - `pseudo_label.rs`: Semi-supervised learning utilities
  - `model.rs`: Model save/load functionality

- `cifar10_semi_supervised/`: Training and labeling binaries
  - `bin/gpu_train.rs`: Training binary with CLI
  - `bin/gpu_label.rs`: Labeling binary

## Technical Details

- All operations run on GPU using CUDA kernels
- cuBLAS for optimized matrix multiplication
- Pseudo-labeling with confidence thresholding
- Learning rate scheduling (step decay every 30 epochs)
- Model serialization: JSON metadata + binary weights

## Performance

- Batch size: 256
- Training time: ~18-20s per epoch on RTX 3060
- Full GPU utilization with minimal CPU transfers

