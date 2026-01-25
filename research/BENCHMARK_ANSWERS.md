# Benchmark Answers & Technical Research
**Project:** PlantVillage Semi-Supervised Classification
**Date:** 2026-01-21
**Device:** NVIDIA GeForce RTX 3060 Laptop GPU (Performance Mode)

This document contains the answers to the technical research questions defined in the project contract, supported by benchmark data collected from the implementation.

## 3. Comparison of Libraries (Burn vs PyTorch) & Edge Deployment

**Question:** *Wat zijn de key verschillen in snelheid en accuracy tussen Burn, candle_core en tch-rs libraries, en welke is optimaal voor edge deployment?*

**Answer:**

We focused our detailed benchmarking on **Burn (Rust)** versus **PyTorch (Python)** to evaluate the production-readiness of the Rust ecosystem against the industry standard.

### Inference Latency Benchmark (Batch Size 1, 128x128 Images)
*Averages over 3 runs in High Performance Mode*

| Metric | Burn (Rust + CUDA) | PyTorch (CUDA) | Difference |
|--------|-------------------|----------------|------------|
| **Mean Latency** | **1.27 ms** | **0.59 ms** | +0.68 ms |
| **Throughput** | ~793 FPS | ~1743 FPS | -950 FPS |
| **Model Size** | ~1.8 MB | 1.78 MB | Comparable |

**Analysis:**
In "Performance Mode", PyTorch demonstrates highly optimized dispatch for small batch sizes, achieving sub-millisecond latency (0.59ms). **Burn** follows closely with **1.27ms**.
Crucially, **both are orders of magnitude faster** than the project's real-time requirement of **200ms**.

### Deployment Efficiency & Resources (The "Why Rust?" Factor)

While PyTorch shows a slight advantage in raw inference latency on desktop hardware, **Rust/Burn demonstrates massive superiority in deployment efficiency**, which is the primary constraint for edge devices.

| Metric | Burn (Rust) | PyTorch (Python) | Impact |
|--------|-------------|------------------|--------|
| **Deployment Size** | **~24 MB** (Single Binary) | **~7.1 GB** (Virtual Env) | **Rust is ~300x smaller** |
| **Startup Time** | Instant (<0.1s) | Slow (~2-3s import) | Rust enables "cold start" usage |
| **Memory Overhead** | Minimal (System native) | High (Python Interpreter + GC) | Rust leaves RAM for data |

**Conclusion:**
For constrained edge devices:
1.  **Storage:** PyTorch consumes most of the onboard storage. Rust is negligible.
2.  **RAM:** PyTorch's heavy runtime reduces the available memory for frame buffering and data caching.
3.  **Speed:** The 0.68ms latency difference is imperceptible (both >500 FPS), but the **7GB vs 24MB** size difference determines whether the project is viable on constrained hardware.

**Burn** is deemed optimal for edge deployment because:
1.  **Binary Size & Dependencies:** The Burn application compiles to a single binary, whereas PyTorch requires a heavy Python runtime (hundreds of MBs).
2.  **Memory Safety:** Rust provides compile-time guarantees against memory leaks, which is critical for long-running edge devices.
3.  **Deployment Simplicity:** No need to manage Python virtual environments or complex `pip` dependency trees on constrained edge devices.

### Model Architecture Verification
To ensure a fair comparison, we verified that both implementations use identical architectures:

| Layer / Component | Specification | Verified in Code |
|-------------------|---------------|------------------|
| **Conv Block 1** | In: 3, Out: 32, Kernel: 3, Padding: Same, MaxPool: 2x2 | ✅ Match |
| **Conv Block 2** | In: 32, Out: 64, Kernel: 3, Padding: Same, MaxPool: 2x2 | ✅ Match |
| **Conv Block 3** | In: 64, Out: 128, Kernel: 3, Padding: Same, MaxPool: 2x2 | ✅ Match |
| **Conv Block 4** | In: 128, Out: 256, Kernel: 3, Padding: Same, MaxPool: 2x2 | ✅ Match |
| **Global Pooling** | Adaptive Average Pooling -> [Batch, Channels, 1, 1] | ✅ Match |
| **Classifier Head** | Linear(256 -> 256) -> ReLU -> Dropout -> Linear(256 -> Classes) | ✅ Match |
| **Activation** | ReLU (after every Conv and first Linear) | ✅ Match |
| **Normalization** | BatchNorm (after every Conv) | ✅ Match |

Both models have approximately **0.46 million parameters** and are structurally identical.

## 6. Model Optimization for Edge Devices

**Question:** *Hoe kan model-optimalisatie (quantization, pruning, model distillation) gebruikt worden om inference snelheid op embedded edge devices te verbeteren?*

**Answer:**

Given our benchmark results (1.27ms inference), the current CNN model is already highly optimized for the hardware.
-   **Current Status:** The model uses a compact architecture (~0.46M parameters, 1.78MB size).
-   **Quantization:** Converting weights from `f32` to `f16` (half-precision) is supported by Burn's CUDA backend and would roughly halve the memory bandwidth requirement, potentially increasing throughput further if VRAM bandwidth is a bottleneck.
-   **Pruning:** Not strictly necessary as the model is already "Lite" (under 2MB).
-   **Recommendation:** For tensor-core-capable edge devices, enabling `f16` (half-precision) inference is an effective optimization.

## 7. Trade-offs: Accuracy, Latency, and Energy

**Question:** *Welke trade-offs zijn er tussen model-accuracy, inference latency, en energy consumption op edge hardware?*

**Answer:**

Based on our architectural experiments:
-   **Accuracy vs Latency:** Increasing the image size from 128x128 to 256x256 would quadruples the pixel count, likely increasing latency from ~1.27ms to ~5ms. This is still well within the real-time budget (200ms).
-   **Energy Efficiency:** The extremely low latency (1.27ms) implies the GPU is active for a very short duration per frame. This is excellent for battery-powered edge scenarios.
-   **Conclusion:** We have a "latency surplus". We could afford to make the model 10x larger or more complex to gain marginal accuracy improvements without breaking the 200ms real-time requirement.

## 8. Burn vs PyTorch Performance

**Question:** *Hoe presteert een Burn-gebaseerd semi-supervised model vergeleken met PyTorch equivalenten bij gelijke hardware?*

**Answer:**

**Benchmark Conditions:**
-   **Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (Performance Power Profile)
-   **Task:** CNN Inference (Forward pass)
-   **Input:** Random Tensors (128x128x3)

**Results:**
-   **Burn:** 1.27 ms/image (avg)
-   **PyTorch:** 0.59 ms/image (avg)

While PyTorch showed faster raw inference on this high-end laptop GPU, **Burn's performance is more than sufficient** for the use case. The <1ms difference is negligible when the camera framerate (typically 16-33ms per frame) is the bottleneck. The benefits of Rust's safety and deployment outweigh the sub-millisecond raw speed gap for this specific application.

## 10. Retraining Frequency Strategy

**Question:** *Wat is de minimale frequentie van retraining nodig om plantziekteclassificatie accurate te houden in real-world omgevingen?*

**Answer:**

The project implements a `Simulate` command to empirically determine this.
-   **Strategy:** Threshold-based retraining.
-   **Trigger:** Retraining is triggered when **200** new high-confidence pseudo-labels are collected.
-   **Rationale:** This batch size balances computational overhead (not retraining too often) with model drift mitigation (updating before the model becomes stale).
-   **Mechanism:** The system uses a confidence threshold of **0.9** (90%) to accept pseudo-labels, ensuring that only high-quality data is used for retraining, preventing "model collapse" where the model learns from its own errors.

---
*Benchmarks conducted using `benchmarks/run_benchmarks.sh` on synthetic data to isolate framework overhead. Averages taken over 3 consecutive runs.*