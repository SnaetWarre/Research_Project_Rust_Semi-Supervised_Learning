# PlantVillage Semi-Supervised Learning - User Guide

This guide covers the complete workflow for using the PlantVillage SSL system, from dataset preparation to deployment on CUDA-capable embedded devices.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Semi-Supervised Learning](#semi-supervised-learning)
6. [Benchmarking](#benchmarking)
7. [Edge Deployment](#edge-deployment)
8. [Troubleshooting](#troubleshooting)

---

## Overview

PlantVillage SSL is a semi-supervised learning system for plant disease classification. It uses the Burn framework in Rust for efficient training and inference, with a focus on edge deployment.

### Key Features

- **39-class plant disease classification** using the PlantVillage dataset
- **Semi-supervised learning** with pseudo-labeling for efficient use of unlabeled data
-- **Edge deployment** optimized for CUDA-capable embedded devices
- **Stream simulation** to mimic real-world agricultural camera setups

### System Requirements

**Desktop (Training):**
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU with CUDA 11.x or 12.x
- 16GB+ RAM
- 10GB+ disk space

**Edge Device (Inference):**
- CUDA-capable embedded device (8GB recommended)
- JetPack 5.x or later

---

## Dataset Preparation

### 1. Download the PlantVillage Dataset

The PlantVillage dataset contains ~87,000 images of plant leaves across 38 classes.

**Option A: Kaggle Download (Recommended)**

```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (download from Kaggle settings)
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
cd ..
./download_plantvillage.sh
```

**Option B: Manual Download**

1. Visit: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Download and extract the dataset
3. Ensure the extracted directory contains `train/` and `valid/`

### 2. Dataset Structure

After downloading, your dataset should look like:

```
data/plantvillage/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── ... (38 class directories)
└── valid/
    ├── Apple___Apple_scab/
    └── ... (38 class directories)
```

### 3. Verify Dataset

```bash
ls data/plantvillage/train
```

---

## Training

### Basic Training

Train a model with default settings:

```bash
./target/release/plantvillage_ssl train \
    --data-dir data/plantvillage \
    --epochs 50 \
    --batch-size 32 \
    --output-dir output/models
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 32 | Batch size for training |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--labeled-ratio` | 0.2 | Fraction of data to use as labeled |
| `--confidence-threshold` | 0.9 | Threshold for pseudo-labeling |
| `--cuda` | false | Enable CUDA GPU acceleration |

### Training with Semi-Supervised Learning

Enable pseudo-labeling for semi-supervised learning:

```bash
./target/release/plantvillage_ssl train \
    --data-dir data/plantvillage \
    --epochs 50 \
    --labeled-ratio 0.2 \
    --confidence-threshold 0.9 \
    --cuda
```

This simulates a realistic scenario where only 20% of data is labeled.

### Monitoring Training

Training logs are saved to `output/logs/`. Monitor progress with:

- Training loss per epoch
- Validation accuracy
- Pseudo-label quality metrics
- Learning rate schedule

---

## Inference

### Single Image Prediction

```bash
./target/release/plantvillage_ssl infer \
    --input path/to/leaf_image.jpg \
    --model output/models/best_model.mpk
```

### Batch Prediction

```bash
./target/release/plantvillage_ssl infer \
    --input path/to/image_directory/ \
    --model output/models/best_model.mpk
```

### Output Format

Predictions include:
- Predicted class name (e.g., "Tomato___Late_blight")
- Confidence score (0.0 - 1.0)
- Top-5 predictions with probabilities
- Inference time in milliseconds

---

## Semi-Supervised Learning

### Stream Simulation

Simulate real-world camera data collection:

```bash
./target/release/plantvillage_ssl simulate \
    --data-dir data/plantvillage \
    --model output/models/initial_model.mpk \
    --days 30 \
    --images-per-day 50 \
    --confidence-threshold 0.9 \
    --retrain-threshold 200 \
    --output-dir output/simulation
```

### Simulation Parameters

| Parameter | Description |
|-----------|-------------|
| `--days` | Number of simulated days |
| `--images-per-day` | Images "captured" per day |
| `--confidence-threshold` | Threshold for pseudo-label acceptance |
| `--retrain-threshold` | Retrain after N pseudo-labels accumulated |

### Understanding the Split Strategy

The dataset is split into pools to simulate real-world conditions:

1. **Test Set (10%)** - Never seen during training, for final evaluation
2. **Validation Set (10%)** - For hyperparameter tuning
3. **Labeled Pool (20%)** - Initial labeled training data
4. **Stream Pool (60%)** - Simulates incoming camera images

---

## Benchmarking

### Run Latency Benchmark

```bash
./target/release/plantvillage_ssl benchmark \
    --model output/models/best_model.mpk \
    --test-dir data/plantvillage/valid/Tomato___healthy \
    --iterations 100 \
    --cuda
```

### Benchmark Metrics

- **Mean latency** - Average inference time
- **P95/P99 latency** - Tail latency percentiles
- **Throughput** - Images per second
- **Memory usage** - GPU/CPU memory consumption

### Target Performance

| Metric | Target | Acceptable |
|--------|--------|------------|
| Mean latency | <200ms | <500ms |
| P95 latency | <300ms | <600ms |
| Throughput | >5 img/s | >2 img/s |

---

## Edge Deployment

### 1. Platform Setup

Run the provided setup script on your target embedded device (if applicable):

```bash
cd plantvillage_ssl
chmod +x scripts/setup_jetson.sh
./scripts/setup_jetson.sh
```

### 2. Transfer Model

Copy your trained model to the device (adjust user and host as needed):

```bash
scp output/models/best_model.mpk user@<device-ip>:/home/user/plantvillage/models/
```

### 3. Run Inference on Device

```bash
./target/release/plantvillage_ssl infer \
    --input test_image.jpg \
    --model models/best_model.mpk \
    --cuda
```

### 4. Monitor Performance

Check device utilization using vendor tools (e.g., `tegrastats` on some NVIDIA devices) or standard NVIDIA/Linux tools such as `nvidia-smi`, `top`, or `htop`.

---

## Troubleshooting

### CUDA Not Found

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Out of Memory

- Reduce batch size: `--batch-size 8`
- Use smaller input images: Configure model for 224x224 instead of 256x256
- Enable gradient checkpointing (if available)

### Slow Training

- Ensure CUDA is being used: `--cuda`
- Check GPU utilization with `nvidia-smi`
- Increase batch size if memory allows

### Poor Accuracy

- Increase training epochs
- Adjust learning rate
- Use data augmentation
- Lower pseudo-label confidence threshold (with caution)

### Model Loading Errors

- Ensure model file exists and is not corrupted
- Check that model was saved with compatible Burn version
- Verify feature flags match (e.g., `--features cuda`)

---

## Configuration Reference

### Model Configuration

Edit `config/model_config.json`:

```json
{
    "num_classes": 39,
    "input_size": 256,
    "dropout_rate": 0.5,
    "conv_filters": [32, 64, 128, 256],
    "fc_units": [512, 256]
}
```

### Training Configuration

Edit `config/training_config.json`:

```json
{
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "early_stopping_patience": 10
}
```

### Semi-Supervised Configuration

Edit `config/ssl_config.json`:

```json
{
    "confidence_threshold": 0.9,
    "max_pseudo_per_class": 500,
    "retrain_threshold": 200,
    "epochs_per_retrain": 10
}
```

---

## Getting Help

- **Issues**: Open an issue on the project repository
- **Documentation**: Check the API documentation with `cargo doc --open`
- **Examples**: See the `examples/` directory for usage examples

---

*Last updated: 2024*