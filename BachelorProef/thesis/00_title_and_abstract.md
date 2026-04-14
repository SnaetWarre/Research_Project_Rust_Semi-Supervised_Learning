# How Can a Semi-Supervised Neural Network Be Efficiently Implemented in Rust for the Automatic Labeling of Partially Labeled Datasets on an Edge Device?

**Bachelor Thesis: Howest MCT 2025–2026**

---

| | |
|---|---|
| **Student** | Warre Snaet |
| **Student Email** | warresnaet@student.howest.be |
| **Program** | Multimedia & Creative Technologies (MCT) |
| **Institution** | Howest University of Applied Sciences |
| **Internal Promoter** | [TODO: Name of internal promoter] |
| **External Promoter** | [TODO: Name of external promoter / 2nd reader] |
| **Academic Year** | 2025–2026 |

---

## Abstract

Plant diseases account for an estimated 20–40% of global crop losses annually. Early detection is critical, yet existing solutions depend on cloud-based inference or costly laboratory analysis, neither of which is viable for farmers operating in areas with limited or no internet connectivity. This thesis investigates how a semi-supervised neural network can be efficiently implemented in Rust to automatically label partially labeled plant disease datasets on an edge device, fully offline.

The research combines a custom lightweight convolutional neural network (CNN) with a pseudo-labeling pipeline built entirely in Rust using the Burn machine learning framework. Starting from only 20% labeled data, the system iteratively assigns pseudo-labels to unlabeled images that exceed a 90% confidence threshold, then retrains on the expanded dataset. Three controlled experiments evaluate critical aspects of real-world deployment: (1) the minimum number of labeled samples per class required for acceptable accuracy, (2) the effect of model scale on catastrophic forgetting when adding new disease classes, and (3) how the position of a new class within an existing taxonomy affects learning difficulty.

The resulting system compiles to a 24 MB binary (300 times smaller than an equivalent PyTorch deployment) and achieves 0.39 ms inference latency (2,579 FPS) on an NVIDIA RTX 3060 GPU. Cross-platform deployment is demonstrated on desktop, iPhone 12 via Tauri (80 ms inference), and CPU-only environments, all without any network dependency. Experimental results show that 100 labeled images per class suffice for over 80% accuracy, that catastrophic forgetting increases six-fold when scaling from 5 to 30 base classes, and that adding a class to a larger taxonomy requires substantially more labeled samples than adding it to a smaller one.

The findings confirm that Rust and the Burn framework provide a production-viable path for deploying semi-supervised machine learning on edge devices, with significant advantages in deployment size, inference speed, and cross-platform portability compared to Python-based alternatives.

**Keywords:** semi-supervised learning, pseudo-labeling, edge AI, Rust, Burn framework, plant disease detection, incremental learning, Tauri, offline inference

---

## Table of Contents

1. [Introduction](01_introduction.md)
2. [Research: Literature Study](02_research.md)
3. [Research Results](03_results.md)
4. [Reflection](04_reflection.md)
5. [Advice](05_advice.md)
6. [Conclusion](06_conclusion.md)
7. [References](07_references.md)
8. Appendices
   - [A: Installation & User Guide](appendices/A_installation_guide.md)
   - [B: Interview Question Template](appendices/B_interview_template.md)
   - [C: Guest Session Report: NVISO AI Threats](appendices/C_guest_session_nviso.md)
   - [D: Guest Session Report: Session 2](appendices/D_guest_session_2.md)
