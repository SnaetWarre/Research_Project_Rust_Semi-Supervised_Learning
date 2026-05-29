# How Can a Semi-Supervised Neural Network Be Efficiently Implemented in Rust for the Automatic Labeling of Partially Labeled Datasets on an Edge Device?

**Bachelor Thesis: Howest MCT 2025-2026**

---

| | |
|---|---|
| **Student** | Warre Snaet |
| **Student Email** | warresnaet@student.howest.be |
| **Program** | Multimedia & Creative Technologies (MCT) |
| **Institution** | Howest University of Applied Sciences |
| **Internal Promoter** | Gilles Depypere |
| **External Promoter** | Sandro Queirós |
| **Academic Year** | 2025-2026 |

---

# Abstract

Deploying machine learning models on edge devices without cloud connectivity remains difficult because the common Python/PyTorch stack brings large runtime dependencies, while expert-labeled datasets are expensive to create. This thesis investigates whether a semi-supervised neural network can be implemented efficiently in Rust with the Burn framework, so that partially labeled image datasets can be labeled locally on an edge device.

Plant disease classification on the PlantVillage dataset is used as the benchmark. The system combines a lightweight convolutional neural network with a pseudo-labeling pipeline written end to end in Rust. Starting from 20% labeled data, the model assigns pseudo-labels to unlabeled images above a 90% confidence threshold and retrains on the enlarged dataset.

The experiments evaluate label efficiency, catastrophic forgetting when new classes are added, and the effect of adding a new class to a small versus large taxonomy. The saved SSL checkpoint reaches 94.90% top-1 accuracy on the held-out test split. The trained weights are about 916 KB in Burn's native format and about 1.8 MB as ONNX, while the compiled release binary is roughly 26 MB. On an RTX 3060 laptop, inference reaches 0.42 ms per image, and an iPhone 12 through Tauri reaches about 80 ms per image.

The results show that Rust and Burn are a realistic route for offline, portable edge AI, with deployment size and inference speed as the main advantages over Python-based alternatives.

**Keywords:** semi-supervised learning, pseudo-labeling, edge AI, Rust, Burn framework, plant disease detection, incremental learning, Tauri, offline inference

---
