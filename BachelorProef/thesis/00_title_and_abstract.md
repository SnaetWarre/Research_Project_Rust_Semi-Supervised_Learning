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

Plant diseases are responsible for an estimated 20 to 40% of global crop losses every year. Early detection matters, yet the tools that are currently available rely either on cloud-based inference or on expensive laboratory analysis. Neither option is realistic for farmers who work in areas with limited or no internet connectivity. This thesis investigates how a semi-supervised neural network can be implemented efficiently in Rust so that partially labeled plant disease datasets can be labeled automatically on an edge device, fully offline.

The research combines a custom lightweight convolutional neural network (CNN) with a pseudo-labeling pipeline written end to end in Rust using the Burn machine learning framework. Starting from only 20% of the data being labeled, the system iteratively assigns pseudo-labels to unlabeled images whose predictions exceed a 90% confidence threshold, and then retrains on the expanded dataset. Three controlled experiments look at the aspects that matter most for real-world deployment: (1) the minimum number of labeled samples per class needed for acceptable accuracy, (2) the effect of model scale on catastrophic forgetting when new disease classes are added, and (3) how the position of a new class within an existing taxonomy changes the difficulty of learning it.

The resulting system compiles into a single binary of roughly 26 MB that already contains every runtime dependency, whereas a comparable Python/PyTorch deployment requires a multi-gigabyte environment on the target device. On an NVIDIA RTX 3060 GPU the model reaches an inference latency of 0.39 ms, which corresponds to 2,579 FPS. Cross-platform deployment has been demonstrated on desktop, on an iPhone 12 through Tauri (80 ms per inference) and in CPU-only environments, all without any network dependency. The experiments show that 100 labeled images per class are enough to exceed 80% accuracy, that catastrophic forgetting grows six-fold when the base is scaled from 5 to 30 classes, and that adding a class to a large taxonomy requires considerably more labeled samples than adding it to a small one.

The findings confirm that Rust and the Burn framework offer a production-viable route for deploying semi-supervised machine learning on edge devices, with meaningful advantages over Python-based alternatives in terms of deployment size, inference speed and cross-platform portability.

**Keywords:** semi-supervised learning, pseudo-labeling, edge AI, Rust, Burn framework, plant disease detection, incremental learning, Tauri, offline inference

---

## Table of Contents

**Front matter**
- [Foreword](00a_foreword.md)
- [List of Figures and Tables](00b_list_of_figures.md)
- [List of Abbreviations](00c_abbreviations.md)
- [Glossary](00d_glossary.md)

**Chapters**
1. [Introduction](01_introduction.md)
2. [Research: Literature Study](02_research.md)
3. [Research Results](03_results.md)
4. [Reflection](04_reflection.md)
5. [Advice](05_advice.md)
6. [Conclusion](06_conclusion.md)
7. [References](07_references.md)

**Appendices**
- [A: Installation & User Guide](appendices/A_installation_guide.md)
- [B: Interview Question Template](appendices/B_interview_template.md)
- [C: Guest Session Report: NVISO AI Threats](appendices/C_guest_session_nviso.md)
- [D: Guest Session Report: Session 2](appendices/D_guest_session_2.md)
