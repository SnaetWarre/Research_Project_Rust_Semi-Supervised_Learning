# 1. Introduction

## 1.1 Context and Motivation

Deploying machine learning models on edge devices introduces requirements that differ fundamentally from cloud-based deployments. The standard machine learning stack, consisting of Python and PyTorch, is optimised for research flexibility and data-centre GPU throughput. Its deployment characteristics, however, present significant challenges for edge scenarios. A typical PyTorch deployment requires the Python interpreter, the PyTorch library and a set of supporting packages, which together occupy several gigabytes on the target device. While this footprint is acceptable on cloud infrastructure, it is problematic for edge environments where storage is limited and network connectivity may be unreliable.

Rust offers an alternative deployment model. It compiles to a single binary without requiring an interpreter or garbage collector. Its type system enforces memory safety at compile time, which is valuable for machine learning processes that must run reliably over extended periods. Furthermore, Cargo, the Rust build system, supports cross-compilation to ARM, iOS, Android and WebAssembly targets. In this project, these characteristics enable a deployment footprint reduction from a multi-gigabyte Python environment to a 26 MB compiled binary that can be distributed over a USB drive or a short data transfer.

The more demanding aspect is model training. Rust's machine learning ecosystem is comparatively young, and several existing frameworks focus primarily on inference rather than full training loops. Semi-supervised learning (SSL) combines a small labeled dataset with a large pool of unlabeled data. Implementing SSL requires pseudo-labeling, confidence filtering and iterative retraining cycles, which places greater demands on the training application programming interface (API) than simple inference does. The central research question addressed in this thesis is whether Rust's machine learning ecosystem, specifically the Burn framework, can support this workflow while preserving its deployment advantages.

Plant disease detection is employed as the evaluation benchmark. The PlantVillage dataset, comprising 38 classes and approximately 87,000 images, provides a realistic and widely used test case. Plant disease detection is a particularly suitable application for edge deployment: end users frequently operate in areas with limited connectivity, and existing diagnostic tools either depend on cloud access or require expensive laboratory analysis. By constructing an SSL pipeline for this use case, the project evaluates whether Rust can support a complete machine learning workflow—from training through deployment—in settings where offline operation is essential.

![Deployment footprint comparison: Python/PyTorch stack versus Rust single binary](figures/deployment_comparison.svg)
*Figure 1.1: Conceptual comparison of the runtime footprint of a Python/PyTorch deployment versus a compiled Rust binary. The sizes are based on the measurements described in Chapter 3.*

## 1.2 The Labeling Problem

Image classification models require labeled training data, and expert annotation is expensive. In the agricultural domain, labeling by plant pathologists demands specialised expertise and is both time-consuming and costly [1]. For a dataset of 87,000 images across 38 classes, the total annotation cost can reach thousands of euros, which is prohibitive for many research groups and smaller projects.

Semi-supervised learning offers a mechanism to reduce these costs. By training an initial model on a small labeled subset and then using that model to generate pseudo-labels for the remaining unlabeled data, SSL can approach the accuracy of fully supervised training at a substantially reduced annotation budget. The critical challenge is ensuring that pseudo-labels are sufficiently accurate to improve the model rather than degrade its performance. This requirement makes the choice of confidence threshold and retraining strategy central to the pipeline design.

## 1.3 Research Question

The central research question of this thesis is:

> **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

This question is decomposed into the following sub-questions:

1. Which principles underpin semi-supervised learning, and how can pseudo-labeling be applied to image classification?
2. What is the best-practice approach for implementing neural networks with the Burn framework in Rust?
3. How can data augmentation and pseudo-labeling strategies improve training efficiency on limited labeled datasets?
4. What trade-offs exist between model accuracy, inference latency and deployment size on edge hardware?
5. How does a Burn-based semi-supervised model compare to a PyTorch equivalent on identical hardware?
6. How much labeled data is needed for acceptable accuracy, and what happens when new classes are added incrementally?
7. Which practical obstacles stand in the way of deploying Rust-based machine learning on edge devices?

## 1.4 Scope and Approach

This research focuses on implementing a complete SSL pipeline in Rust with the Burn framework, validated on the PlantVillage dataset with 38 classes and approximately 87,000 images. The model is a custom lightweight convolutional neural network (CNN) designed for edge deployment. It is neither a pretrained model nor a Vision Transformer. The full pipeline, from training to deployment, compiles into a single binary that runs fully offline.

The experimental work is organised around three axes:

1. **Label efficiency**: determining the minimum number of labeled samples per class required for acceptable classification accuracy.
2. **Class scaling**: measuring how catastrophic forgetting varies when new classes are added to models of different sizes (a 5-class base versus a 30-class base).
3. **New class position**: evaluating whether a new class is more difficult to learn as the 6th class in a small taxonomy than as the 31st class in a large one.

Deployment is validated across four hardware configurations: a laptop with an NVIDIA RTX 3060 GPU, an iPhone 12 through Tauri, a Jetson Orin Nano and a CPU-only environment.

## 1.5 Thesis Structure

This thesis is organised as follows:

- **Chapter 2: Research** presents the literature study. It covers semi-supervised learning techniques, the Rust ML ecosystem, incremental learning theory, edge AI deployment strategies and the PlantVillage dataset.
- **Chapter 3: Research Results** describes the technical implementation. It covers the system architecture, the SSL training pipeline, the three controlled experiments and their quantitative results, the cross-platform benchmarks and the Tauri-based GUI application.
- **Chapter 4: Reflection** offers a critical evaluation of the results. It includes external expert feedback, a self-reflection on strengths and weaknesses, and an analysis of the broader implications, including implementation barriers, privacy considerations and possible directions for future research.
- **Chapter 5: Advice** provides practical recommendations for anyone tackling a similar research question, grounded in the experimental findings and the literature review.
- **Chapter 6: Conclusion** answers the research question directly by bringing together the key findings from the preceding chapters.
