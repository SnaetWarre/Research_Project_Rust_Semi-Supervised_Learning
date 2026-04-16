# 1. Introduction

## 1.1 Context and Motivation

Plant diseases are responsible for an estimated 20 to 40% of global agricultural production losses each year [15]. For smallholder farmers, who together produce roughly one third of the world's food, a single undetected outbreak can be the difference between a profitable harvest and financial ruin. Early and accurate identification of plant diseases is therefore not merely a technological convenience but an economic necessity.

Current diagnostic methods fall into two broad categories. The first one is laboratory analysis, where leaf samples are physically sent to an expert for examination. This approach is accurate but slow, with turnaround times measured in days, and it does not scale economically. The second category covers cloud-based machine learning systems, which upload images of diseased plants to a remote server for classification. These are faster than laboratory analysis, but they introduce a hard dependency on internet connectivity, which is precisely what is missing in many of the rural regions where agriculture is at its most vulnerable.

This creates a clear gap. Farmers need fast, accurate and accessible disease detection that works without an internet connection, on the devices they already own. The solution has to be light enough to run on a smartphone or a low-power laptop, and it must not require a data connection beyond the initial installation.

## 1.2 The Labeling Problem

Any machine learning model for plant disease classification needs large amounts of labeled training data. Expert annotation of agricultural imagery costs roughly €2 per image [16]. For a dataset of 50,000 images covering 38 disease classes, that amounts to a labeling budget of €100,000, a figure that is prohibitive for most research projects and entirely impractical for field deployment in developing regions.

Semi-supervised learning (SSL) offers a way out of this dependency. By training an initial model on a small labeled subset and then using that model to generate pseudo-labels for the remaining unlabeled data, SSL can reach accuracy levels that are comparable to fully supervised training at a fraction of the annotation cost. The main challenge is making sure that the pseudo-labels are accurate enough to improve the model rather than degrade it during retraining.

## 1.3 Research Question

The central research question of this thesis is:

> **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

This question is broken down into the following sub-questions:

1. Which principles and techniques underpin semi-supervised learning, and how can they be applied in practice to plant disease classification?
2. What is the best-practice approach for implementing neural networks with the Burn framework in Rust, including layer construction and forward passes?
3. What are the key differences in speed and accuracy between Burn, Candle and tch-rs, and which of them is the most suitable for edge deployment?
4. How can data augmentation and pseudo-labeling strategies improve training efficiency on limited labeled datasets?
5. What are the best methods for automatically assigning labels to unlabeled plant leaf images, and how can the reliability of those labels be evaluated?
6. How can model optimisation techniques such as quantisation or pruning improve inference speed on embedded edge devices?
7. What trade-offs exist between model accuracy, inference latency and energy consumption on edge hardware?
8. How does a Burn-based semi-supervised model compare to a PyTorch equivalent on identical hardware?
9. Which practical implementation obstacles stand in the way of deployment on edge devices, and how can they be resolved?
10. What is the minimum retraining frequency needed to keep plant disease classification accurate in real-world environments?

## 1.4 Scope and Approach

The research focuses on the PlantVillage dataset (38 disease classes and roughly 87,000 images) and uses a custom lightweight convolutional neural network (CNN) rather than pretrained models or Vision Transformers. The full pipeline, from training to deployment, is implemented in Rust using the Burn framework. The system is designed to run fully offline, with no network calls during inference.

The experimental work is organised around three axes:

1. **Label efficiency**: determining the minimum number of labeled samples per class that is needed for acceptable classification accuracy.
2. **Class scaling**: measuring how catastrophic forgetting changes when new classes are added to models of different sizes (a 5-class base versus a 30-class base).
3. **New class position**: evaluating whether a new class is harder to learn as the 6th class in a small taxonomy than as the 31st class in a large one.

Deployment is validated across four hardware configurations: a laptop with an NVIDIA RTX 3060 GPU, an iPhone 12 through Tauri, a Jetson Orin Nano and a CPU-only environment.

## 1.5 Thesis Structure

This thesis is organised as follows:

- **Chapter 2: Research** presents the literature study. It covers semi-supervised learning techniques, the Rust ML ecosystem, incremental learning theory, edge AI deployment strategies and the PlantVillage dataset.
- **Chapter 3: Research Results** describes the technical implementation. It covers the system architecture, the SSL training pipeline, the three controlled experiments and their quantitative results, the cross-platform benchmarks and the Tauri-based GUI application.
- **Chapter 4: Reflection** offers a critical evaluation of the results through interviews with external experts, together with an analysis of the broader implications, including implementation barriers, business value, societal impact and possible directions for future research.
- **Chapter 5: Advice** gives a practical, step-by-step guide for anyone tackling the same research question, grounded in both the experimental findings and the feedback from external reflection.
- **Chapter 6: Conclusion** answers the research question directly by bringing together the key findings from the preceding chapters.
