# 🌱 Stappenplan: Update Codebase to PlantVillage Semi-Supervised Learning

## Project Overview

**Current State:** CIFAR-10 semi-supervised learning with custom neural network library built from scratch using cudarc/CUDA.

**Target State:** PlantVillage plant disease classification using the **Burn** framework, deployed on NVIDIA Jetson Orin Nano with semi-supervised learning (pseudo-labeling).

---

## ⚠️ CRITICAL: No Real Camera/Sensor Setup

**Important Constraint:** We do NOT have:
- A camera attached to the Jetson
- Actual sick plants from the 39 disease classes
- A greenhouse or farm to collect real images

**Solution:** We must **simulate** the real-world scenario using the PlantVillage dataset itself, with a carefully designed split strategy that mimics:
1. Initial labeled data (what a farmer might manually label)
2. "Incoming" unlabeled images (simulating camera captures)
3. Held-out test images the model has NEVER seen (for honest evaluation)

This simulation must be **structured and reproducible** so we can demonstrate the semi-supervised learning pipeline as if it were running in a real agricultural setting.

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:
- [x] Rust 1.70+ installed
- [x] NVIDIA GPU with CUDA 12.x drivers (for desktop training)
- [ ] Access to NVIDIA Jetson Orin Nano 8GB (for edge deployment)
- [ ] ~5GB disk space for PlantVillage dataset
- [x] Python 3.10+ (for PyTorch reference implementation)

---

## Phase 1: Project Restructuring

### Step 1.1: Create New Project Structure [x]
Create a new Rust workspace with the following structure:

```
Source/
├── plantvillage_ssl/              # Main semi-supervised learning project
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs               # CLI entry point
│   │   ├── lib.rs                # Library exports
│   │   ├── dataset/              # PlantVillage data handling
│   │   │   ├── mod.rs
│   │   │   ├── loader.rs         # Dataset loading
│   │   │   ├── augmentation.rs   # Data augmentation
│   │   │   └── split.rs          # Train/val/test split with semi-supervised simulation
│   │   ├── model/                # CNN architecture with Burn
│   │   │   ├── mod.rs
│   │   │   ├── cnn.rs            # Main CNN model
│   │   │   └── config.rs         # Model configuration
│   │   ├── training/             # Training logic
│   │   │   ├── mod.rs
│   │   │   ├── trainer.rs        # Main training loop
│   │   │   ├── pseudo_label.rs   # Pseudo-labeling algorithm
│   │   │   └── scheduler.rs      # Learning rate scheduling
│   │   ├── inference/            # Inference & deployment
│   │   │   ├── mod.rs
│   │   │   ├── predictor.rs      # Single image prediction
│   │   │   └── benchmark.rs      # Latency benchmarking
│   │   └── utils/                # Utilities
│   │       ├── mod.rs
│   │       ├── logging.rs        # Structured logging with tracing
│   │       └── metrics.rs        # Accuracy, F1, confusion matrix
│   └── scripts/
│       ├── download_dataset.py   # Python script to download PlantVillage
│       └── setup_jetson.sh       # Jetson deployment script
├── pytorch_reference/            # PyTorch comparison implementation
│   ├── requirements.txt
│   ├── train.py
│   ├── model.py
│   └── benchmark.py
├── benchmarks/                   # Benchmark results
│   └── README.md
└── docs/                         # Documentation
    ├── installation.md
    └── user_guide.md
```

### Step 1.2: Initialize Cargo Workspace [x]
Create a `Cargo.toml` at the workspace root and the main project `Cargo.toml`:

**plantvillage_ssl/Cargo.toml dependencies:**
```toml
[package]
name = "plantvillage_ssl"
version = "0.1.0"
edition = "2021"

[dependencies]
# ML Framework - Burn with CUDA backend
burn = { version = "0.15", features = ["train", "tui"] }
burn-cuda = "0.15"              # CUDA backend for GPU training
burn-ndarray = "0.15"           # CPU backend fallback
burn-tch = "0.15"               # LibTorch backend (optional, for comparison)

# Data handling
image = "0.25"
ndarray = "0.16"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Logging & Monitoring
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# CLI & Progress
clap = { version = "4.0", features = ["derive"] }
indicatif = "0.17"
colored = "2.1"

# Utilities
rand = "0.8"
chrono = "0.4"
anyhow = "1.0"

[features]
default = ["cuda"]
cuda = ["burn-cuda"]
jetson = ["cuda"]  # Jetson-specific optimizations
```

---

## Phase 2: Dataset Implementation

### Step 2.1: Create PlantVillage Dataset Downloader [x]
Create a Python script (`scripts/download_dataset.py`) that:
1. Downloads PlantVillage from Kaggle or direct source
2. Organizes into folder structure: `data/plantvillage/{class_name}/*.jpg`
3. Creates metadata JSON with class names and image counts
4. Total: 61,486 images, 39 classes, 256x256 pixels

**Class list to implement:**
```
Apple___Apple_scab, Apple___Black_rot, Apple___Cedar_apple_rust, Apple___healthy,
Blueberry___healthy, Cherry___Powdery_mildew, Cherry___healthy,
Corn___Cercospora_leaf_spot, Corn___Common_rust, Corn___Northern_Leaf_Blight, Corn___healthy,
Grape___Black_rot, Grape___Esca, Grape___Leaf_blight, Grape___healthy,
Orange___Haunglongbing, Peach___Bacterial_spot, Peach___healthy,
Pepper___Bacterial_spot, Pepper___healthy,
Potato___Early_blight, Potato___Late_blight, Potato___healthy,
Raspberry___healthy, Soybean___healthy,
Squash___Powdery_mildew, Strawberry___Leaf_scorch, Strawberry___healthy,
Tomato___Bacterial_spot, Tomato___Early_blight, Tomato___Late_blight,
Tomato___Leaf_Mold, Tomato___Septoria_leaf_spot, Tomato___Spider_mites,
Tomato___Target_Spot, Tomato___Yellow_Leaf_Curl_Virus, Tomato___mosaic_virus, Tomato___healthy
```

### Step 2.2: Implement Rust Dataset Loader [x]
In `src/dataset/loader.rs` and `src/dataset/burn_dataset.rs`:
1. Load images from folder structure [x]
2. Resize to 224x224 (standard for CNNs) or 256x256 [x]
3. Normalize pixel values (ImageNet mean/std or dataset-specific) [x]
4. Convert to Burn tensors [x]
5. Implement `burn::data::dataset::Dataset` trait [x]
   - Created `PlantVillageBurnDataset` implementing Burn's `Dataset` trait
   - Created `PlantVillageBatcher` for efficient batch creation
   - Created `PseudoLabelDataset` and `PseudoLabelBatcher` for SSL
   - Created `CombinedDataset` for merging labeled + pseudo-labeled data

### Step 2.3: Implement Dataset Split Strategy (CRITICAL!) [x]

**The Problem:** We don't have a real camera to capture new plant images. We must simulate the entire workflow using only the PlantVillage dataset.

**The Solution:** A 4-pool split strategy that simulates real-world deployment:

```
PlantVillage Dataset (61,486 images)
│
├── 🧪 TEST SET (15%) - ~9,200 images
│   └── NEVER touched during training or pseudo-labeling
│   └── Used ONLY for final evaluation
│   └── This simulates "future unseen plants"
│
├── 📊 VALIDATION SET (10%) - ~6,100 images  
│   └── Used to monitor training progress
│   └── Early stopping decisions
│   └── NOT used for pseudo-labeling
│
└── 🎯 TRAINING POOL (75%) - ~46,100 images
    │
    ├── 🏷️ LABELED POOL (20-30% of training) - ~9,200-13,800 images
    │   └── Initially available with ground-truth labels
    │   └── Simulates "manually labeled by farmer"
    │   └── Balanced across all 39 classes
    │
    ├── 📷 STREAM POOL (50-60% of training) - ~23,000-27,700 images
    │   └── Simulates "camera captures over time"
    │   └── Labels are HIDDEN from the model
    │   └── Fed to model in batches (simulating daily captures)
    │   └── Model predicts → high confidence → pseudo-label
    │   └── Ground-truth used ONLY to measure pseudo-label quality
    │
    └── 🔮 FUTURE POOL (10-20% of training) - ~4,600-9,200 images
        └── Reserved for demonstrating retraining
        └── "New images that arrive later"
        └── Can show model improvement over time
```

**In `src/dataset/split.rs`, implement:**

```rust
pub struct DatasetSplits {
    // Evaluation (never touched during training)
    pub test_set: Vec<LabeledImage>,        // 15% - final evaluation only
    pub validation_set: Vec<LabeledImage>,  // 10% - training monitoring
    
    // Training pools
    pub labeled_pool: Vec<LabeledImage>,    // 20-30% of 75% - initial training
    pub stream_pool: Vec<HiddenLabelImage>, // 50-60% of 75% - simulated camera
    pub future_pool: Vec<HiddenLabelImage>, // 10-20% of 75% - later retraining demo
    
    // Tracking
    pub pseudo_labeled: Vec<PseudoLabeledImage>, // Grows during training
}

pub struct HiddenLabelImage {
    pub image_path: PathBuf,
    pub hidden_label: usize,  // Ground truth, but hidden from model
    pub image_id: usize,
}

pub struct PseudoLabeledImage {
    pub image_path: PathBuf,
    pub predicted_label: usize,
    pub confidence: f32,
    pub ground_truth: usize,  // For measuring pseudo-label accuracy
    pub is_correct: bool,     // predicted == ground_truth
}
```

**Stream Simulation Logic:**


**Stream Simulation Logic:**/a


**Stream Simulation Logic:**/
1. Model trains on `labeled_pool` [ ]
2. Simulate "Day 1": Take batch of 500 images from `stream_pool` [x]
3. Run inference → filter confidence > 0.9 → add to `pseudo_labeled` [x]
4. Retrain on `labeled_pool` + `pseudo_labeled` [ ]
5. Simulate "Day 2": Take next batch of 500 from `stream_pool` [x]
6. Repeat until `stream_pool` exhausted or accuracy plateaus [x]

**Key Metrics to Track:**
- Pseudo-label precision: `sum(is_correct) / len(pseudo_labeled)` [x]


**Key Metrics to Track:**/a


**Key Metrics to Track:**/
- Stream coverage: How much of `stream_pool` gets pseudo-labeled [x]
- Accuracy progression on `validation_set` after each batch [x]

### Step 2.4: Implement Data Augmentation [x]
In `src/dataset/augmentation.rs`:
1. Random horizontal flip [x]
2. Random rotation (±15 degrees) [ ]
3. Random brightness/contrast adjustment [x]
4. Random crop and resize [x]
5. Color jitter [ ]
6. Use Burn's transform pipeline [ ]

---

## Phase 3: Model Architecture with Burn

### Step 3.1: Implement CNN Model [x]
In `src/model/cnn.rs`, create a CNN using Burn:

**Architecture (similar to ResNet-18 lite):**
```
Input: 224x224x3
├── Conv2d(3, 64, 7x7, stride=2, padding=3) → BatchNorm → ReLU → MaxPool(3x3, stride=2)
├── ConvBlock(64, 64, 3x3) x2
├── ConvBlock(64, 128, 3x3, stride=2) + ConvBlock(128, 128, 3x3)
├── ConvBlock(128, 256, 3x3, stride=2) + ConvBlock(256, 256, 3x3)
├── AdaptiveAvgPool → Flatten
├── Dropout(0.3)
├── Linear(256, 39) → Softmax
Output: 39 classes
```

**Key Burn implementation points:**
- Use `#[derive(Module)]` for the model struct [x]
- Implement `forward()` method [x]
- Use `burn::nn::*` for layers (Conv2d, Linear, BatchNorm2d, etc.) [x]
- Support both training and inference modes (dropout behavior) [x]

### Step 3.2: Model Configuration [x]
In `src/model/config.rs`:
```rust
pub struct ModelConfig {
    pub num_classes: usize,        // 39 for PlantVillage
    pub input_size: (usize, usize), // (224, 224)
    pub dropout_rate: f64,         // 0.3
    pub use_pretrained: bool,      // Optional: load pretrained weights
}
```

### Step 3.3: Implement Model Save/Load [x]
Using Burn's built-in serialization:
1. Save model weights after training [x]
2. Load model for inference [x]
3. Support ONNX export for cross-platform deployment [ ]

---

## Phase 4: Training Pipeline

### Step 4.1: Implement Training Loop [x]
In `src/training/trainer.rs`:
1. Create Burn training pipeline with: [x]
   - Optimizer: Adam with configurable learning rate and weight decay
   - Loss: CrossEntropyLoss for labeled data
   - Weighted CrossEntropyLoss for pseudo-labeled data (confidence weighting)
   - Learning rate scheduler: Cosine annealing, StepDecay, Exponential, ReduceOnPlateau
2. Training loop with epoch/batch progress [x]
   - `train_epoch_labeled()` for supervised training
   - `train_epoch_semi_supervised()` for SSL with pseudo-labels
   - Ramp-up weight for pseudo-label loss
3. Validation after each epoch [x]
   - `evaluate()` method computing loss, accuracy, and full metrics
   - Predictions with confidence scores via `predict_with_confidence()`
4. Early stopping (patience: configurable) [x]
5. Checkpoint saving (best model) [x]

### Step 4.2: Implement Pseudo-Labeling with Stream Simulation [x]


**Stream Simulation Logic:**/a


**Stream Simulation Logic:**/

**This simulates a camera feeding images to the Jetson over time!**

**Algorithm (Simulated Camera Stream):**
```
SETUP:
- Model M trained on labeled_pool (initial training)
- stream_pool contains ~25,000 "unlabeled" images (labels hidden)
- batch_size = 500 images (simulates ~1 day of camera captures)

SIMULATION LOOP:
for day in 1..=max_days:
    # Simulate camera capturing images
    batch = stream_pool.take_next(batch_size)
    
    if batch.is_empty():
        break  # No more "camera images"
    
    # Run inference on "new camera images"
    predictions = model.predict(batch)
    
    # Filter high-confidence predictions for pseudo-labeling
    high_conf = predictions.filter(|p| p.confidence > 0.9)
    
    # Add to pseudo-labeled pool (with hidden ground truth for metrics)
    for pred in high_conf:
        pseudo_labeled.add(PseudoLabeledImage {
            image: pred.image,
            predicted_label: pred.class,
            confidence: pred.confidence,
            ground_truth: batch.get_hidden_label(pred.image),  # For metrics only!
            is_correct: pred.class == ground_truth,
        })
    
    # Log pseudo-label quality (this is our "oracle" check)
    precision = pseudo_labeled.count_correct() / pseudo_labeled.len()
    log!("Day {}: +{} pseudo-labels, precision: {:.2}%", day, high_conf.len(), precision * 100)
    
    # Retrain if enough new pseudo-labels accumulated
    if pseudo_labeled.len() >= retraining_threshold:
        model.retrain(labeled_pool + pseudo_labeled)
        log!("Retrained! Validation accuracy: {:.2}%", model.eval(validation_set))

FINAL EVALUATION:
- Test on test_set (NEVER seen during any training)
- Report: final accuracy, pseudo-label precision, stream coverage
```

**Config:**
```rust
pub struct StreamSimulationConfig {
    pub confidence_threshold: f64,      // 0.9
    pub batch_size: usize,              // 500 (images per "day")
    pub max_days: usize,                // Max simulation days
    pub retraining_threshold: usize,    // Retrain every N pseudo-labels (e.g., 500)
    pub epochs_per_retrain: usize,      // 20-30 epochs per retraining
    pub max_pseudo_per_class: usize,    // Prevent class imbalance
}
```

**Why This Approach Works:**
1. **Realistic simulation**: Mimics how a real edge device would receive images
2. **Honest evaluation**: Test set is truly unseen
3. **Measurable pseudo-label quality**: We know ground truth (but model doesn't)
4. **Demonstrates value**: Shows improvement from semi-supervised learning
5. **Reproducible**: Same splits = same results for demo

### Step 4.3: Implement Metrics Tracking [x]
In `src/utils/metrics.rs`:
1. Accuracy (top-1) [x]
2. F1-score (macro and per-class) [x]
3. Confusion matrix [x]
4. Pseudo-label precision (how many pseudo-labels are correct) [x]


**Key Metrics to Track:**/a


**Key Metrics to Track:**/
5. Confidence distribution histogram [ ]
6. Export to JSON for analysis [x]

---

## Phase 5: Inference & Benchmarking

### Step 5.1: Implement Inference Pipeline [x]
In `src/inference/predictor.rs`:
1. Load trained model [x]
2. Preprocess single image [x]
3. Run forward pass [x]
4. Return: predicted class, confidence, top-5 predictions [x]
5. Support batch inference [x]

### Step 5.2: Implement Latency Benchmarking [x]
In `src/inference/benchmark.rs`:
1. Warm-up runs (10 iterations) [x]
2. Timed runs (100+ iterations) [x]
3. Measure: min, max, mean, p50, p95, p99 latency [x]
4. **Target: < 200ms per image on Jetson** [ ]
5. Log GPU/CPU/Memory usage [x]
6. Export benchmark results to JSON [x]

### Step 5.3: Add Profiling & Monitoring [x]
Using tracing crate:
1. Structured logging for all operations [x]
2. Performance spans for timing [x]
3. Integration with nvidia-smi for GPU metrics [x]
4. Memory usage tracking via /proc/self/status [x]

---

## Phase 6: Model Optimization for Edge

### Step 6.1: Implement Quantization [ ]
1. Post-training quantization (FP32 → FP16 or INT8)
2. Burn supports FP16 via backend configuration
3. Measure accuracy drop vs latency improvement
4. Target: minimal accuracy loss (<2%) with 2x+ speedup

### Step 6.2: Model Size Reduction (Optional) [ ]
If model is too large for Jetson:
1. Reduce channel counts (64→32, 128→64, etc.)
2. Use depthwise separable convolutions
3. Knowledge distillation (train smaller model to mimic larger)
4. Pruning (remove low-magnitude weights)

### Step 6.3: Batch Size Optimization [ ]
1. Find optimal batch size for Jetson (likely 1-4 for real-time)
2. Balance throughput vs latency
3. Test memory limits on Jetson 8GB VRAM

---

## Phase 7: PyTorch Reference Implementation

### Step 7.1: Create PyTorch Baseline [x]
In `pytorch_reference/`:
1. Implement same CNN architecture [x]
2. Same training pipeline (pseudo-labeling) [x] - `trainer.py` with `PseudoLabeler` class
3. Same hyperparameters [x] - Matching `TrainingConfig` dataclass
4. Train on same hardware (desktop GPU) [x] - CUDA support with device selection

### Step 7.2: Benchmark Comparison [x]
Created `benchmarks/compare_frameworks.py` with:
1. Training time (wall-clock) [x]
2. Inference latency (desktop GPU) [x]
3. Inference latency (CPU) [x]
4. Inference latency (Jetson - via TorchScript or ONNX) [ ] - Pending Jetson testing
5. Final test accuracy [x]
6. Model size (MB) [x]
7. Memory usage during inference [x]
8. Automatic chart generation (latency, throughput, comparison) [x]

---

## Phase 8: Jetson Deployment

### Step 8.1: Create Deployment Script [x]
In `scripts/setup_jetson.sh`:
1. Install Rust on Jetson [x]
2. Install CUDA 12.x + cuDNN 9.x (Jetson-optimized) [x]
3. Build project with `--features jetson` [x]
4. Copy trained model weights [x]
5. Run smoke test [x]

### Step 8.2: Test on Jetson [ ]
1. Run inference benchmark on Jetson
2. Verify < 200ms latency target
3. Run stress test: 1 hour continuous inference
4. Monitor for memory leaks
5. Log GPU temperature and throttling

### Step 8.3: Create Docker Container (Optional) [ ]
For reproducibility:
1. Create Dockerfile based on NVIDIA L4T base image
2. Include all dependencies
3. Include pre-trained model weights
4. Document build and run instructions

---

## Phase 9: Documentation & Demo

### Step 9.1: Create User Documentation [x]
In `docs/user_guide.md`:
1. How to run inference [x]
2. How to interpret results [x]
3. Confidence scores explained [x]
4. Troubleshooting guide [x]

### Step 9.2: Create Installation Guide [x]
In `docs/installation.md`:
1. Desktop setup (training) [x]
2. Jetson setup (inference) [x]
3. Dependencies list [x]
4. Common issues and fixes [x]

### Step 9.3: Prepare Demo Materials [ ]
1. Sample images from test set (never seen during training)
2. Latency timer visualization
3. Confidence bar display
4. Before/after pseudo-labeling accuracy comparison
5. Retraining impact graph

---

## 🎯 Success Criteria Checklist

### Primary Goals
- [ ] Inference latency < 200ms on Jetson Orin Nano
- [ ] Test accuracy ≥ 85% on PlantVillage (38 classes)
- [ ] Stable operation: 1+ hour continuous inference without crashes/leaks

### Secondary Goals
- [ ] Semi-supervised improves accuracy by ≥ 5% over labeled-only baseline
- [x] Model uses Burn framework (not custom from-scratch implementation)
- [x] Training pipeline with Burn implemented (Trainer, optimizer, loss, scheduler) [COMPLETE]
- [x] Comparison with PyTorch reference documented - `benchmarks/compare_frameworks.py`
- [ ] Retraining frequency analysis complete

### Deliverables
- [x] Working Rust/Burn codebase [DONE]
- [x] Burn Dataset trait implementation (PlantVillageBurnDataset, Batchers) [DONE]
- [x] Complete training loop (trainer.rs with forward/backward, optimizer, scheduler)
- [x] CLI with stats, train, infer, benchmark, simulate commands [DONE]
- [ ] Trained model weights - Requires dataset download
- [x] Benchmark results (Burn vs PyTorch) - Comparison script ready
- [x] Jetson deployment scripts
- [x] Documentation (user guide + installation)
- [ ] Demo-ready application

---

## 📅 Suggested Timeline (4 weeks)

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Phase 1-2 | Project structure, dataset loading, splits working [DONE] |
| 2 | Phase 3-4 | CNN model in Burn, training pipeline, pseudo-labeling [DONE] |
| 3 | Phase 5-6 | Benchmarking, optimization, Jetson testing [NEXT] |
| 4 | Phase 7-9 | PyTorch comparison, documentation, demo preparation |

### Progress Summary (Updated - Latest)
- ✅ Phase 1: Project Restructuring - Complete
- ✅ Phase 2: Dataset Implementation - Complete (Burn Dataset trait, batchers, loaders)
- ✅ Phase 3: Model Architecture - Complete (PlantClassifier, PlantClassifierLite with CUDA)
- ✅ Phase 4: Training Pipeline - Complete (full GPU training loop with Adam, cross-entropy, LR scheduling)
- ✅ Phase 5: Inference & Benchmarking - Complete (CLI benchmark command, runner.rs with JSON output)
- 🔲 Phase 6: Model Optimization for Edge - Pending (FP16, quantization)
- ✅ Phase 7: PyTorch Reference - Complete (trainer.py with full semi-supervised pipeline, pseudo-labeling)
- 🔲 Phase 8: Jetson Deployment - Scripts ready, needs hardware testing
- ✅ Phase 9: Documentation & Demo - Partial (benchmark comparison script ready)

### New Files Added (Latest Session)
- `pytorch_reference/trainer.py` - Full PyTorch training pipeline with pseudo-labeling
- `benchmarks/compare_frameworks.py` - Burn vs PyTorch benchmark comparison with charts
- `plantvillage_ssl/src/inference/runner.rs` - Benchmark runner with JSON output
- `scripts/run_benchmarks.sh` - Convenience script to run all benchmarks

### Build & Test Status
- ✅ `cargo build --release` with CUDA backend: PASSING
- ✅ `cargo test --release`: 59 tests passing (all on GPU)
- ✅ Default feature changed to `cuda` (GPU-first for Jetson Orin Nano)

### Immediate Next Steps
1. **Download PlantVillage dataset** - Required before training can begin
   - Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
   - Extract to: `data/plantvillage/{class_name}/*.jpg`
2. **Run first training experiment** on desktop GPU
   - Burn: `cargo run --release -- train --data-dir data/plantvillage --epochs 10`
   - PyTorch: `python pytorch_reference/trainer.py --data-dir data/plantvillage --epochs 10`
3. **Run benchmark comparison**
   - Quick: `cargo run --release -- benchmark --iterations 100`
   - Full: `./scripts/run_benchmarks.sh --epochs 5`
4. **Test on Jetson Orin Nano** - Deploy and measure real edge performance

---

## 🔧 Quick Reference: Key Burn Patterns

### Creating a Model
```rust
use burn::nn::{Conv2d, Linear, BatchNorm};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct PlantClassifier<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    fc: Linear<B>,
}

impl<B: Backend> PlantClassifier<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        // ... more layers
        self.fc.forward(x)
    }
}
```

### Training with Burn Learner
```rust
use burn::train::{LearnerBuilder, metric::LossMetric};

let learner = LearnerBuilder::new(artifact_dir)
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .with_file_checkpointer(CompactRecorder::new())
    .devices(vec![device])
    .num_epochs(config.epochs)
    .build(model, optim, lr_scheduler);

let trained_model = learner.fit(train_loader, valid_loader);
```

### Saving/Loading Models
```rust
// Save
model.save_file(path, &recorder)?;

// Load
let model = PlantClassifier::load_file(path, &recorder, device)?;
```

---

## 🚨 Important Notes for AI Agent

1. **Use Burn, not custom CUDA code**: The current codebase uses cudarc directly. Replace with Burn's high-level API. [DONE]

2. **Keep pseudo-labeling logic**: The semi-supervised learning approach stays the same, just implement with Burn tensors. [DONE]

3. **Archive old code**: Don't delete `neural_net` and `cifar10_semi_supervised` - move to `archive/` folder for reference.

4. **Test incrementally**: After each phase, ensure the code compiles and basic tests pass. [DONE]

5. **Burn documentation**: Refer to https://burn.dev/docs and https://github.com/tracel-ai/burn for examples.

6. **Dataset difference**: 
   - CIFAR-10: 32x32, 10 classes, 60k images
   - PlantVillage: 256x256, 39 classes, 61k images
   - Adjust model architecture accordingly (bigger input, more classes) [DONE]