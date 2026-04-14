# 1. Introduction

## 1.1 Context and Motivation

Plant diseases are responsible for an estimated 20–40% of global agricultural production losses each year [15]. For smallholder farmers, who produce roughly one-third of the world's food, a single undetected outbreak can mean the difference between a profitable harvest and financial ruin. Early and accurate identification of plant diseases is therefore not merely a technological convenience but an economic necessity.

Current diagnostic methods fall into two broad categories. The first is laboratory-based analysis, where leaf samples are physically sent to an expert for examination. This approach is accurate but slow, with turnaround times of several days, and it is economically prohibitive at scale. The second category consists of cloud-based machine learning systems, where images of diseased plants are uploaded to a remote server for classification. While faster than laboratory methods, these systems introduce a hard dependency on internet connectivity, a resource that is unavailable or unreliable in many of the rural regions where agriculture is most vulnerable.

This creates a clear gap: farmers need fast, accurate, and accessible disease detection that works without internet connectivity, on the devices they already own. The solution must be lightweight enough to run on a smartphone or a low-power laptop, and it must not require a data connection after initial installation.

## 1.2 The Labeling Problem

Building a machine learning model for plant disease classification requires large quantities of labeled training data. Expert annotation of agricultural imagery costs approximately €2 per image [16]. For a dataset of 50,000 images covering 38 disease classes, this translates to a labeling budget of €100,000, a cost that is prohibitive for most research projects and impractical for field deployment in developing regions.

Semi-supervised learning (SSL) offers a way to reduce this dependency on labeled data. By training an initial model on a small labeled subset and then using that model to generate pseudo-labels for the remaining unlabeled data, SSL can achieve accuracy levels comparable to fully supervised training at a fraction of the annotation cost. The key challenge lies in ensuring that the pseudo-labels are sufficiently accurate to improve rather than degrade the model during retraining.

## 1.3 Research Question

The central research question of this thesis is:

> **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

This question is decomposed into the following sub-questions:

1. Which principles and techniques underpin semi-supervised learning, and how can they be practically applied to plant disease classification?
2. What is the best-practice approach for implementing neural networks with the Burn framework in Rust, including layer construction and forward passes?
3. What are the key differences in speed and accuracy between Burn, Candle, and tch-rs, and which is optimal for edge deployment?
4. How can data augmentation and pseudo-labeling strategies improve training efficiency on limited labeled datasets?
5. What are the best methods for automatically assigning labels to unlabeled plant leaf images, and how can the reliability of those labels be evaluated?
6. How can model optimization techniques such as quantization or pruning improve inference speed on embedded edge devices?
7. What trade-offs exist between model accuracy, inference latency, and energy consumption on edge hardware?
8. How does a Burn-based semi-supervised model compare to PyTorch equivalents on identical hardware?
9. What are the practical implementation obstacles for deployment on edge devices, and how can they be resolved?
10. What is the minimum retraining frequency needed to keep plant disease classification accurate in real-world environments?

## 1.4 Scope and Approach

This research focuses on the PlantVillage dataset (38 disease classes, approximately 87,000 images) and uses a custom lightweight convolutional neural network (CNN) rather than pretrained models or Vision Transformers. The entire pipeline, from training to deployment, is implemented in Rust using the Burn framework. The system is designed to work fully offline, with no network calls during inference.

The experimental work is organized around three axes:

1. **Label efficiency**: determining the minimum number of labeled samples per class needed for acceptable classification accuracy.
2. **Class scaling**: measuring how catastrophic forgetting changes when adding new classes to models of different sizes (5-class vs. 30-class base).
3. **New class position**: evaluating whether a new class is harder to learn as the 6th class in a small taxonomy versus the 31st class in a large one.

Deployment is validated across four hardware configurations: a laptop with an NVIDIA RTX 3060 GPU, an iPhone 12 via Tauri, a Jetson Orin Nano, and a CPU-only environment.

## 1.5 Thesis Structure

This thesis is organized as follows:

- **Chapter 2: Research** presents the literature study covering semi-supervised learning techniques, the Rust ML ecosystem, incremental learning theory, edge AI deployment strategies, and the PlantVillage dataset.
- **Chapter 3: Research Results** describes the technical implementation: the system architecture, the SSL training pipeline, the three controlled experiments with their quantitative results, cross-platform benchmarks, and the Tauri-based GUI application.
- **Chapter 4: Reflection** provides a critical evaluation of the results through interviews with external experts and an analysis of broader implications including implementation barriers, business value, societal impact, and future research directions.
- **Chapter 5: Advice** offers a practical step-by-step guide for someone facing the same research question, grounded in both the experimental findings and the feedback from external reflection.
- **Chapter 6: Conclusion** directly answers the research question by synthesizing the key findings from the preceding chapters.
