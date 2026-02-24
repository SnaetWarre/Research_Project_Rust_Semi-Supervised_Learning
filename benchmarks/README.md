# 📊 Benchmarks

This directory contains benchmark results comparing different configurations and implementations of the PlantVillage semi-supervised learning system.

## Directory Structure

```
benchmarks/
├── README.md                    # This file
├── results/                     # Benchmark result files
│   ├── desktop_gpu/            # Desktop GPU (NVIDIA RTX/GTX) results
│   ├── embedded_device/       # Embedded edge device results
│   └── cpu_only/               # CPU-only baseline results
├── comparisons/                 # Cross-implementation comparisons
│   ├── burn_vs_pytorch.csv     # Burn vs PyTorch comparison
│   └── backend_comparison.csv   # Burn backend comparison (CUDA/NdArray/Tch)
└── scripts/                     # Helper scripts for running benchmarks
```

## Key Metrics

### Primary Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **Inference Latency** | <200ms (ideal), <500ms (max) | Time per single image inference |
| **Throughput** | >5 img/s | Images processed per second |
| **Accuracy** | >85% | Classification accuracy on test set |
| **F1-Score** | >0.80 | Macro-averaged F1 score |

### Resource Metrics

| Metric | Edge Device Limit | Description |
|--------|--------------|-------------|
| **GPU Memory** | <4GB | Peak GPU memory usage |
| **CPU Usage** | <80% | Average CPU utilization |
| **Power Consumption** | <15W | Average power draw |

## Running Benchmarks

### Quick Benchmark

```bash
# Run a quick benchmark (20 iterations)
./target/release/plantvillage_ssl benchmark \
    --model output/models/best_model.mpk \
    --test-dir data/plantvillage/valid \
    --iterations 20
```

### Full Benchmark

```bash
# Run comprehensive benchmark (500 iterations)
./target/release/plantvillage_ssl benchmark \
    --model output/models/best_model.mpk \
    --test-dir data/plantvillage/valid \
    --iterations 500 \
    --cuda
```

### Stress Test

```bash
# Run 1-hour stress test
./target/release/plantvillage_ssl benchmark \
    --model output/models/best_model.mpk \
    --test-dir data/plantvillage/valid \
    --duration 3600 \
    --monitor-memory
```

## Expected Results

### Desktop GPU (NVIDIA RTX 3080)

| Model Size | Batch Size | Latency (ms) | Throughput (img/s) | Memory (MB) |
|------------|------------|--------------|--------------------| ------------|
| Full | 1 | ~15 | ~65 | ~1200 |
| Full | 32 | ~45 | ~710 | ~3500 |
| Lite | 1 | ~8 | ~125 | ~400 |
| Lite | 32 | ~25 | ~1280 | ~1100 |

### Embedded Device (8GB)

| Model Size | Batch Size | Latency (ms) | Throughput (img/s) | Memory (MB) |
|------------|------------|--------------|--------------------| ------------|
| Full | 1 | ~120 | ~8 | ~800 |
| Full | 8 | ~350 | ~23 | ~2000 |
| Lite | 1 | ~60 | ~17 | ~300 |
| Lite | 8 | ~180 | ~44 | ~700 |

### CPU-Only (Intel i7)

| Model Size | Batch Size | Latency (ms) | Throughput (img/s) |
|------------|------------|--------------|-------------------|
| Full | 1 | ~250 | ~4 |
| Lite | 1 | ~80 | ~12 |

## Burn vs PyTorch Comparison

Preliminary comparison results (to be updated after implementation):

| Metric | Burn (Rust) | PyTorch (Python) | Notes |
|--------|-------------|------------------|-------|
| Inference Latency | TBD | TBD | Single image |
| Training Time | TBD | TBD | Per epoch |
| Model Size | TBD | TBD | Binary size |
| Memory Usage | TBD | TBD | Peak during inference |
| Accuracy | TBD | TBD | Test set |

## Semi-Supervised Learning Results

### Pseudo-Labeling Effectiveness

| Labeled Ratio | Accuracy (Supervised Only) | Accuracy (Semi-Supervised) | Improvement |
|---------------|----------------------------|----------------------------|-------------|
| 10% | TBD | TBD | TBD |
| 20% | TBD | TBD | TBD |
| 30% | TBD | TBD | TBD |
| 50% | TBD | TBD | TBD |

### Pseudo-Label Quality

| Confidence Threshold | Acceptance Rate | Pseudo-Label Accuracy |
|---------------------|-----------------|----------------------|
| 0.80 | TBD | TBD |
| 0.85 | TBD | TBD |
| 0.90 | TBD | TBD |
| 0.95 | TBD | TBD |

## Result File Format

Benchmark results are stored in JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "device_info": {
    "name": "Embedded Device",
    "device_type": "CUDA",
    "cuda_version": "12.2"
  },
  "config": {
    "iterations": 100,
    "batch_size": 1,
    "warmup_iterations": 10
  },
  "latency": {
    "mean_ms": 125.3,
    "std_ms": 8.2,
    "min_ms": 112.1,
    "max_ms": 156.8,
    "p50_ms": 124.1,
    "p95_ms": 139.4,
    "p99_ms": 152.7
  },
  "throughput": 7.98,
  "memory": {
    "gpu_used_mb": 812.0,
    "gpu_total_mb": 7680.0,
    "cpu_rss_mb": 245.0
  }
}
```

## Notes

- All benchmarks should be run after a system warmup period
- GPU benchmarks require CUDA-enabled builds (`--features cuda`)
For accurate embedded device measurements, follow your device vendor's guidance to enable a high-performance power profile (if available). Memory reporting on some embedded platforms uses unified memory where GPU and CPU share the same pool.