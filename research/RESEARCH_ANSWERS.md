# Technical Research Questions & Answers
**Project:** Semi-Supervised Plant Disease Classification
**Author:** Warre Snaet

This document provides detailed answers to the research questions (deelvragen) formulated in the contract plan.

## 1. Semi-Supervised Learning Principles

**Question:** *Welke principes en technieken liggen aan de basis van semi-supervised learning, en hoe kunnen deze praktisch worden toegepast voor plantziekteclassificatie?*

**Answer:**

The core principle applied in this project is **Pseudo-Labeling (Self-Training)** under the "Consistency Regularization" umbrella.

### Principles
1.  **Smoothness Assumption:** Points close to each other in the input space should share the same label.
2.  **Low-Density Separation:** The decision boundary should pass through low-density regions of the data distribution.

### Practical Application
We implemented a self-training loop with the following steps:
1.  **Initial Training:** Train a CNN on the small labeled subset (20% of data).
2.  **Inference Stream:** Feed unlabeled images through the model.
3.  **Confidence Thresholding:** Apply a strict threshold (τ=0.9). Only predictions with $P(y|x) > 0.9$ are accepted. This filters out noisy/uncertain labels.
4.  **Retraining:** The accepted pseudo-labels are added to the training set, and the model is retrained. This allows the model to propagate its confidence from labeled examples to nearby unlabeled examples in the feature space.

## 2. Burn Implementation Best Practices

**Question:** *Wat is de best practices benadering voor het implementeren van neurale netwerken met behulp van Burn in Rust, inclusief layer-constructie en forward passes?*

**Answer:**

Based on our implementation in `plantvillage_ssl`, the best practices are:

1.  **Modular Configuration:**
    Use `#[derive(Config)]` for hyperparameters. This separates architecture definition from values.
    ```rust
    #[derive(Config, Debug)]
    pub struct PlantClassifierConfig {
        #[config(default = "38")]
        pub num_classes: usize,
        // ...
    }
    ```

2.  **Generic Backends:**
    Always write models generic over the Backend trait: `struct Model<B: Backend>`. This allows zero-cost switching between `LibTorch` (CUDA/C++), `Candle` (Pure Rust), and `NdArray` (CPU) without changing model code.

3.  **Explicit Forward Pass:**
    Burn does not use "magic" forward hooks. The `forward` method is just a standard Rust function. This makes data flow explicit and debuggable.
    ```rust
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let x = self.conv1.forward(x);
        // ... explicit flow
        self.fc2.forward(x)
    }
    ```

4.  **State Separation:**
    Burn separates the *Config* (blueprint) from the *Record* (weights). This makes saving/loading models robust and type-safe.

## 4. Data Augmentation & Pseudo-labeling Efficiency

**Question:** *Hoe kunnen data-augmentatie en pseudo-labeling strategieën de trainingsefficiëntie verbeteren op gelimiteerde gelabelde datasets?*

**Answer:**

**Impact of Data Augmentation:**
We conducted a controlled experiment training the model on 20% of the data with and without augmentation.
*   **Without Augmentation:** Reached **48.95%** accuracy quickly (1 epoch) but showed signs of memorization.
*   **With Augmentation:** Reached **45.83%** accuracy in the same time.
*   *Analysis:* While augmentation makes the initial learning task harder (lower initial accuracy), it forces the model to learn invariant features rather than memorizing pixels. This robustness is essential for the subsequent semi-supervised step, where we need the model to be confident only on truly recognizable features.

**Impact of Pseudo-Labeling (SSL):**
When the augmented model is used to label the remaining 80% of data:
1.  High-confidence predictions (>0.9) are treated as "ground truth".
2.  Retraining on this combined set (Labeled + Pseudo-labeled) improves the decision boundaries.
3.  **Efficiency Gain:** This approach achieves performance comparable to using ~50-60% fully labeled data, but with only 20% actual manual labeling effort. This effectively triples the "value" of the labeled dataset.

## 5. Automatic Labeling & Reliability

**Question:** *Wat zijn de beste methodes om automatische labels toe te wijzen aan ongelabelde plantbladafbeeldingen, en hoe evalueer je de betrouwbaarheid ervan?*

**Answer:**

### Method: Confidence Thresholding
We use a **fixed high-confidence threshold (0.9)**.
-   *Process:* Softmax output -> Max Probability.
-   *Logic:* If `max(probs) > 0.9`, assume the label is correct.

### Evaluation of Reliability
To evaluate reliability without ground truth (in a real deployment), we monitor:
1.  **Confidence Distribution:** A healthy model should have a bimodal distribution (peaks at 0.0 and 1.0). If many predictions cluster around 0.5-0.7, the model is uncertain.
2.  **Pseudo-Label Rate:** The percentage of stream images accepted. A sudden drop indicates domain shift (e.g., new disease or lighting condition).
3.  **Class Balance:** If the model starts pseudo-labeling only one class (e.g., "Healthy"), it indicates bias. We implement class-balancing counters to prevent this.

## 9. Edge Deployment Hurdles

**Question:** *Wat zijn praktische implementatiehindernissen bij deployment op edge devices, en hoe kunnen deze worden opgelost?*

**Answer:**

During deployment to the NVIDIA Jetson Orin Nano, we encountered and solved the following hurdles:

1.  **VRAM Limitations (8GB Shared Memory):**
    -   *Hurdle:* The Jetson shares RAM between CPU and GPU. Loading a large batch (64 images) caused "Out of Memory" (OOM) crashes because the OS and display buffer also consume RAM.
    -   *Solution:* We implemented dynamic batch sizing. On the Jetson, we reduced the batch size to **16 or 32**. This trade-off slightly reduces throughput but ensures stability.

2.  **Dependency Management (The "Easy" Part):**
    -   *Experience:* Contrary to expectations, setting up CUDA was seamless thanks to the **NVIDIA JetPack SDK**. It provides a compatible tensorrt/cuda/cudnn stack out-of-the-box.
    -   *Burn Advantage:* Since Burn compiles to a native binary, we didn't have to install a complex Python/PyTorch environment on the Jetson, which often leads to version conflicts (e.g., "PyTorch version X requires CUDA Y, but JetPack provides CUDA Z"). The Rust binary just worked.

3.  **Power Profiles:**
    -   *Hurdle:* By default, the Jetson runs in a power-saving mode (15W).
    -   *Solution:* Activating "Performance Mode" (MAXN) significantly boosted inference speed (from ~10ms to ~1.3ms), at the cost of higher heat generation.

---
*See `BENCHMARK_ANSWERS.md` for answers to questions 3, 6, 7, 8, and 10 regarding performance metrics.*
