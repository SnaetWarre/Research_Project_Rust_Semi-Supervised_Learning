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

Deploying machine learning models on edge devices without cloud connectivity is still a real engineering challenge. The standard Python stack needs several gigabytes of runtime dependencies, which makes it impractical for offline applications on phones or field laptops. Getting enough labeled training data for supervised learning is also expensive, especially in specialised domains where expert annotation is needed.

This thesis investigates how a semi-supervised neural network can be efficiently implemented in Rust, using the Burn framework, so that an edge device can automatically label partially labeled datasets without a network connection. Plant disease classification on the PlantVillage dataset serves as the concrete benchmark. The system combines a custom lightweight convolutional neural network with a pseudo-labeling pipeline, written end to end in Rust. Starting from only 20% labeled data, the model assigns pseudo-labels to unlabeled images when the prediction is above a 90% confidence threshold, and then retrains on the enlarged dataset.

Three controlled experiments examine the parts that matter most for practical deployment: (1) the minimum number of labeled samples per class needed for acceptable accuracy, (2) the effect of model scale on catastrophic forgetting when new classes are added, and (3) how the position of a new class within an existing taxonomy changes the difficulty of learning it.

The trained model weights are about 916 KB in Burn's native format and about 1.8 MB when exported to ONNX. The compiled release binary is roughly 26 MB, which includes the runtime and application code. A comparable Python/PyTorch deployment needs a multi-gigabyte environment on the target device. On an NVIDIA RTX 3060 GPU, the saved SSL checkpoint reaches 94.90% top-1 accuracy on the held-out test split and an inference latency of 0.42 ms, or roughly 2,406 FPS. Desktop and CPU-only deployment were tested without a network dependency. An iPhone 12 through Tauri was tested at approximately 80 ms per inference. The experiments show that 100 labeled images per class are enough to exceed 80% accuracy, that catastrophic forgetting grows six-fold when the base model is scaled from 5 to 30 classes, and that adding a class to a large taxonomy needs much more labeled data than adding it to a small one.

The findings show that Rust and the Burn framework can be a realistic route for deploying semi-supervised machine learning on edge devices. Compared with Python-based alternatives, the main advantages are deployment size, inference speed and cross-platform portability.

**Keywords:** semi-supervised learning, pseudo-labeling, edge AI, Rust, Burn framework, plant disease detection, incremental learning, Tauri, offline inference

---
