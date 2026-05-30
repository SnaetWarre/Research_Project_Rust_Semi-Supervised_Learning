# Thesis Humanization Report

This document shows the original sentences/paragraphs and their humanized versions.
You can use this to copy-paste changes into your Google Doc.

## Foreword (`00a_foreword.md`)

**Original:**

Plant disease detection turned out to be a good fit for this kind of question. The PlantVillage dataset is publicly available and well documented, the economic stakes behind early disease detection are real and measurable, and the distance between existing solutions (cloud-dependent or lab-based) and what farmers in rural regions have access to is concrete. Together, those three factors made it a natural topic for research on offline, edge-deployable machine learning.
 
Choosing Rust for the whole implementation was deliberate, but also a bit experimental. I had been exploring the language on my own for approximately a year before starting this project, and I wanted to see whether the ML ecosystem around it had matured enough to be a real alternative to Python for this kind of work. The answer, as this thesis documents in detail, is that it mostly has, although there are still some specific limitations that should be understood before making that choice.
 
 I would like to thank Gilles Depypere for the consistent feedback throughout the research and for the practical guidance on scoping the work. I also want to thank Sandro Queirós for reading the drafts critically and for pushing back on certain conclusions in a way that made them stronger. I also want to thank Helena Torres and Pedro Morais from 2AI-IPCA for reviewing the technical approach and sharing their expertise on image classification, deployment and data augmentation.

**Humanized:**

Plant disease detection turned out to be a good fit for this kind of question. The PlantVillage dataset is publicly available and well documented, the economic stakes behind early disease detection are real and measurable, and the distance between existing solutions (cloud-dependent or lab-based) and what farmers in rural regions have access to is concrete. Together, those three factors made it a natural topic for research on offline, edge-deployable machine learning.
 
Choosing Rust for the whole implementation was deliberate, but also a bit experimental. I had been exploring the language on my own for about a year before starting this project, and I wanted to see whether the ML ecosystem around it had matured enough to be a real alternative to Python for this kind of work. This thesis documents the answer in detail: it mostly has, though a few specific limitations are worth understanding before you commit to that choice.
 
 I would like to thank Gilles Depypere for the consistent feedback throughout the research and for the practical guidance on scoping the work. I also want to thank Sandro Queirós for reading the drafts critically and for pushing back on certain conclusions in a way that made them stronger. I also want to thank Helena Torres and Pedro Morais from 2AI-IPCA for reviewing the technical approach and sharing their expertise on image classification, deployment and data augmentation.

---

## Title and Abstract (`00_title_and_abstract.md`)

**Original:**

# Abstract
 
Deploying machine learning models on edge devices without cloud connectivity remains difficult because the common Python/PyTorch stack brings large runtime dependencies, while expert-labeled datasets are expensive to create. This thesis investigates whether a semi-supervised neural network can be implemented efficiently in Rust with the Burn framework, so that partially labeled image datasets can be labeled locally on an edge device.
 
Plant disease classification on the PlantVillage dataset is used as the benchmark. The system combines a lightweight convolutional neural network with a pseudo-labeling pipeline written end to end in Rust. Starting from 20% labeled data, the model assigns pseudo-labels to unlabeled images above a 90% confidence threshold and retrains on the enlarged dataset.
 
 The experiments evaluate label efficiency, catastrophic forgetting when new classes are added, and the effect of adding a new class to a small versus large taxonomy. The saved SSL checkpoint reaches 94.90% top-1 accuracy on the held-out test split. The trained weights are approximately 916 KB in Burn's native format and approximately 1.8 MB as ONNX, while the compiled release binary is approximately 26 MB. On an RTX 3060 laptop, inference reaches 0.42 ms per image, and an iPhone 12 through Tauri reaches approximately 80 ms per image.

**Humanized:**

# Abstract
 
Deploying machine learning models on edge devices without cloud connectivity remains difficult because the common Python/PyTorch stack brings large runtime dependencies, while expert-labeled datasets are expensive to create. This thesis investigates whether a semi-supervised neural network can be implemented efficiently in Rust with the Burn framework, enabling local labeling of partially labeled image datasets on an edge device.
 
The benchmark is plant disease classification on the PlantVillage dataset. The system combines a lightweight convolutional neural network with a pseudo-labeling pipeline written end to end in Rust. Starting from 20% labeled data, the model assigns pseudo-labels to unlabeled images above a 90% confidence threshold and retrains on the enlarged dataset.
 
 The experiments evaluate label efficiency, catastrophic forgetting when new classes are added, and the effect of adding a new class to a small versus large taxonomy. The saved SSL checkpoint reaches 94.90% top-1 accuracy on the held-out test split. The trained weights are approximately 916 KB in Burn's native format and approximately 1.8 MB as ONNX, while the compiled release binary is approximately 26 MB. On an RTX 3060 laptop, inference reaches 0.42 ms per image, and an iPhone 12 through Tauri reaches approximately 80 ms per image.

---

## Introduction (`01_introduction.md`)

**Original:**

Deploying machine learning models on edge devices introduces requirements that differ fundamentally from cloud-based deployments. The standard machine learning stack, consisting of Python and PyTorch, is optimised for research flexibility and data-centre GPU throughput. Its deployment characteristics, however, present significant challenges for edge scenarios. A typical PyTorch deployment requires the Python interpreter, the PyTorch library and a set of supporting packages, which together occupy several gigabytes on the target device. While this footprint is acceptable on cloud infrastructure, it is problematic for edge environments where storage is limited and network connectivity may be unreliable.
 
Rust offers an alternative deployment model. It compiles to a single binary without requiring an interpreter or garbage collector. Its type system enforces memory safety at compile time, which is valuable for machine learning processes that must run reliably over extended periods. Furthermore, Cargo, the Rust build system, supports cross-compilation to ARM, iOS, Android and WebAssembly targets. In this project, these characteristics enable a deployment footprint reduction from a multi-gigabyte Python environment to a 26 MB compiled binary that can be distributed over a USB drive or a short data transfer.
 
The more demanding aspect is model training. Rust's machine learning ecosystem is comparatively young, and several existing frameworks focus primarily on inference rather than full training loops. Semi-supervised learning (SSL) combines a small labeled dataset with a large pool of unlabeled data. Implementing SSL requires pseudo-labeling, confidence filtering and iterative retraining cycles, which places greater demands on the training application programming interface (API) than simple inference does. The central research question addressed in this thesis is whether Rust's machine learning ecosystem, specifically the Burn framework, can support this workflow while preserving its deployment advantages.
 
Plant disease detection is employed as the evaluation benchmark. The PlantVillage dataset, comprising 38 classes and approximately 87,000 images, provides a realistic and widely used test case. Plant disease detection is a particularly suitable application for edge deployment: end users frequently operate in areas with limited connectivity, and existing diagnostic tools either depend on cloud access or require expensive laboratory analysis. By constructing an SSL pipeline for this use case, the project evaluates whether Rust can support a complete machine learning workflow—from training through deployment—in settings where offline operation is essential.
 
 ![Deployment footprint comparison: Python/PyTorch stack versus Rust single binary](figures/deployment_comparison.svg)

**Humanized:**

Deploying machine learning models on edge devices introduces requirements that differ fundamentally from cloud-based deployments. The standard machine learning stack, consisting of Python and PyTorch, is optimised for research flexibility and data-centre GPU throughput. Its deployment characteristics, however, present significant challenges for edge scenarios. A typical PyTorch deployment requires the Python interpreter, the PyTorch library and a set of supporting packages, which together occupy several gigabytes on the target device. While this footprint is acceptable on cloud infrastructure, it is problematic for edge environments where storage is limited and network connectivity may be unreliable.
 
Rust offers an alternative deployment model. It compiles to a single binary without requiring an interpreter or garbage collector. Its type system enforces memory safety at compile time, which is valuable for machine learning processes that must run reliably over extended periods. Cargo, the Rust build system, supports cross-compilation to ARM, iOS, Android and WebAssembly targets. In this project, these characteristics reduce the deployment footprint from a multi-gigabyte Python environment to a 26 MB compiled binary that can be distributed over a USB drive or a short data transfer.
 
The more demanding aspect is model training. Rust's machine learning ecosystem is comparatively young, and several existing frameworks focus primarily on inference rather than full training loops. Semi-supervised learning (SSL) combines a small labeled dataset with a large pool of unlabeled data. Implementing SSL requires pseudo-labeling, confidence filtering and iterative retraining cycles, which places greater demands on the training API than simple inference does. The central research question of this thesis is whether Rust's machine learning ecosystem, specifically the Burn framework, can support this workflow while preserving its deployment advantages.
 
Plant disease detection is the evaluation benchmark. The PlantVillage dataset, comprising 38 classes and approximately 87,000 images, provides a realistic and widely used test case. Plant disease detection is a particularly suitable application for edge deployment: end users frequently operate in areas with limited connectivity, and existing diagnostic tools either depend on cloud access or require expensive laboratory analysis. By constructing an SSL pipeline for this use case, the project evaluates whether Rust can support a complete machine learning workflow from training through deployment in settings where offline operation is essential.
 
 ![Deployment footprint comparison: Python/PyTorch stack versus Rust single binary](figures/deployment_comparison.svg)

---

## Research (`02_research.md`)

**Original:**

# 2. Research: Literature Study
 
Deep learning-based plant disease classification typically requires large amounts of labeled training data. In practice, however, annotating agricultural images by plant pathologists is labor-intensive and expensive, which limits the scalability of fully supervised approaches [1]. This problem is compounded when deploying on edge devices, where compute, memory and storage impose additional constraints. This chapter situates the research within the existing literature and discusses the theoretical foundations needed to address these challenges: semi-supervised learning as a response to the shortage of labeled data, the Rust machine learning ecosystem as an alternative to the classical Python stack, incremental learning in light of catastrophic forgetting, and the specific boundary conditions of edge AI deployments. The topics discussed fall largely outside the standard MCT curriculum and are therefore explained in depth.
 
 ## 2.1 Semi-Supervised Learning

**Humanized:**

# 2. Research: Literature Study
 
Deep learning-based plant disease classification typically requires large amounts of labeled training data. In practice, however, annotating agricultural images by plant pathologists is labor-intensive and expensive, which limits the scalability of fully supervised approaches [1]. This problem is compounded when deploying on edge devices, where compute, memory and storage impose additional constraints. This chapter situates the research in the existing literature and discusses the theoretical foundations needed to address these challenges: semi-supervised learning as a response to the shortage of labeled data, the Rust machine learning ecosystem as an alternative to the classical Python stack, incremental learning in light of catastrophic forgetting, and the specific boundary conditions of edge AI deployments. The topics discussed fall largely outside the standard MCT curriculum and are therefore explained in depth.
 
 ## 2.1 Semi-Supervised Learning

---

**Original:**

### 2.1.1 Fundamentals
 
The availability of labeled data constitutes a structural bottleneck in training deep learning models for specialised domains such as plant disease recognition. Semi-supervised learning (SSL) offers a solution to this problem by combining a small set of labeled examples with a larger pool of unlabeled data. Under certain assumptions, a model trained in this manner can achieve comparable performance to one trained on a fully labeled dataset [1]. The core idea is that the structure of the unlabeled data contains information about the optimal position of the decision boundary. This assumption is referred to in the literature as the **cluster assumption**: data points that lie close together in feature space are likely to share the same label [1].
 
 ### 2.1.2 Pseudo-Labeling

**Humanized:**

### 2.1.1 Fundamentals
 
Labeled data is a structural bottleneck in training deep learning models for specialised domains such as plant disease recognition. Semi-supervised learning (SSL) offers a solution to this problem by combining a small set of labeled examples with a larger pool of unlabeled data. Under certain assumptions, a model trained in this manner can achieve comparable performance to one trained on a fully labeled dataset [1]. The core idea is that the structure of the unlabeled data contains information about the optimal position of the decision boundary. This assumption is called the **cluster assumption** in the literature: data points that lie close together in feature space are likely to share the same label [1].
 
 ### 2.1.2 Pseudo-Labeling

---

**Original:**

5. Repeat until convergence.
 
The critical design choice here is the **confidence threshold**. If the threshold is too low, noisy pseudo-labels enter the training set and can make the model worse. This is usually called confirmation bias. If the threshold is too high, too many samples are rejected and the unlabeled data is not used enough. This is the **quantity to quality trade-off** described by Chen et al. in SoftMatch [2], where an adaptive weighting scheme is proposed to balance both concerns.
 
 ### 2.1.3 Related Work in Plant Disease Classification
 
Several recent studies have applied SSL to plant disease detection with promising results:
 
 **Ambiguity-Aware Semi-Supervised Learning (AaSSL).** Pham et al. (2025) address the problem of ambiguous samples near the decision boundary. Their method explicitly filters these samples out rather than accepting them as pseudo-labels. With only 5% of the data labeled, accuracy improved from 90.74% to 94.09% [3]. This project therefore adopts confidence-threshold filtering.
 
**Mean Teacher and Consistency Regularization.** Ilsever and Baz (2024) apply a student-teacher architecture to the PlantVillage dataset. The teacher's weights are an exponential moving average (EMA) of the student's weights, and a consistency loss is applied under different augmentations. That approach reached 88.50% accuracy with 5% labeled data [4]. It is effective, but Mean Teacher requires two models in memory at the same time. This constitutes a significant constraint on edge devices with limited VRAM.
 
 **Semi-supervised jute leaf disease classification.** Jannat (2025) shows that a lightweight CNN combined with SSL on 10% labeled and 90% unlabeled data can reach 97.89% accuracy, specifically with mobile and edge deployment in mind [1]. These results demonstrate that simple architectures paired with effective SSL can outperform more complex models in constrained environments.

**Humanized:**

5. Repeat until convergence.
 
The critical design choice here is the **confidence threshold**. If the threshold is too low, noisy pseudo-labels enter the training set and can make the model worse. This is called confirmation bias. If the threshold is too high, too many samples are rejected and the unlabeled data is not used enough. This is the **quantity to quality trade-off** described by Chen et al. in SoftMatch [2], who propose an adaptive weighting scheme to balance both concerns.
 
 ### 2.1.3 Related Work in Plant Disease Classification
 
Several recent studies have applied SSL to plant disease detection:
 
 **Ambiguity-Aware Semi-Supervised Learning (AaSSL).** Pham et al. (2025) address the problem of ambiguous samples near the decision boundary. Their method explicitly filters these samples out rather than accepting them as pseudo-labels. With only 5% of the data labeled, accuracy improved from 90.74% to 94.09% [3]. This project therefore adopts confidence-threshold filtering.
 
**Mean Teacher and Consistency Regularization.** Ilsever and Baz (2024) apply a student-teacher architecture to the PlantVillage dataset. The teacher's weights are an exponential moving average (EMA) of the student's weights, and a consistency loss is applied under different augmentations. That approach reached 88.50% accuracy with 5% labeled data [4]. It is effective, but Mean Teacher requires two models in memory at the same time. This is a significant constraint on edge devices with limited VRAM.
 
 **Semi-supervised jute leaf disease classification.** Jannat (2025) shows that a lightweight CNN combined with SSL on 10% labeled and 90% unlabeled data can reach 97.89% accuracy, specifically with mobile and edge deployment in mind [1]. These results demonstrate that simple architectures paired with effective SSL can outperform more complex models in constrained environments.

---

**Original:**

- **Memory safety.** Python's garbage collector and the C++ backend (LibTorch) can cause unpredictable memory behaviour, which is problematic for long-running processes on an edge device.
 
Rust addresses these constraints in a different way. It compiles to a single binary with no interpreter. Its ownership model gives memory safety at compile time without a garbage collector. Its build system, Cargo, also supports cross-compilation to ARM, WASM, iOS and Android targets [8].
 
 ### 2.2.2 Framework Comparison

**Humanized:**

- **Memory safety.** Python's garbage collector and the C++ backend (LibTorch) can cause unpredictable memory behaviour, which is problematic for long-running processes on an edge device.
 
Rust addresses these constraints differently. It compiles to a single binary with no interpreter. Its ownership model gives memory safety at compile time without a garbage collector. Its build system, Cargo, supports cross-compilation to ARM, WASM, iOS and Android targets [8].
 
 ### 2.2.2 Framework Comparison

---

**Original:**

**tch-rs** provides direct Rust bindings to LibTorch, which is PyTorch's C++ backend. This provides full PyTorch compatibility, but it also reintroduces the dependency on a large C++ shared library (around 1.5 GB), which undoes the deployment size advantage of Rust.
 
Burn was therefore selected because it combines backend-agnostic deployment, a training API that is suitable for custom SSL loops, and the option to produce a self-contained binary for edge devices [9][10][11].
 
 ![Conceptual comparison of deployment models for Burn, Candle and tch-rs](figures/framework_deployment.svg)

**Humanized:**

**tch-rs** provides direct Rust bindings to LibTorch, which is PyTorch's C++ backend. This provides full PyTorch compatibility, but it also reintroduces the dependency on a large C++ shared library (around 1.5 GB), which undoes the deployment size advantage of Rust.
 
Burn was therefore selected because it combines backend-agnostic deployment, a training API suitable for custom SSL loops, and the option to produce a self-contained binary for edge devices [9][10][11].
 
 ![Conceptual comparison of deployment models for Burn, Candle and tch-rs](figures/framework_deployment.svg)

---

**Original:**

Several deployment paths exist for ML models on edge devices:
 
- **ONNX (Open Neural Network Exchange):** a vendor-neutral model format supported by ONNX Runtime, with backends for CPU, GPU, CoreML (iOS), NNAPI (Android) and WebAssembly. It is widely used, but it always requires a conversion step from the training framework.
- **TensorFlow Lite (TFLite):** Google's edge inference runtime, optimised for mobile devices. It requires models to be converted from TensorFlow format and has limited support for custom operations.
 - **WebAssembly (WASM):** allows ML models to run inside web browsers at near-native performance. It makes Progressive Web Apps (PWAs) possible and those can work fully offline after the first load.
 - **Native compilation (Rust/C++):** compiling the model and the inference runtime into a single binary removes all dependency management entirely. This is the approach used in this project via Burn.

**Humanized:**

Several deployment paths exist for ML models on edge devices:
 
- **ONNX (Open Neural Network Exchange):** a vendor-neutral model format supported by ONNX Runtime, with backends for CPU, GPU, CoreML (iOS), NNAPI (Android) and WebAssembly. It is widely used, but it always requires conversion from the training framework.
- **TensorFlow Lite (TFLite):** Google's edge inference runtime, optimised for mobile devices. It requires conversion from TensorFlow format and has limited support for custom operations.
 - **WebAssembly (WASM):** allows ML models to run inside web browsers at near-native performance. It makes Progressive Web Apps (PWAs) possible and those can work fully offline after the first load.
 - **Native compilation (Rust/C++):** compiling the model and the inference runtime into a single binary removes all dependency management entirely. This is the approach used in this project via Burn.

---

**Original:**

### 2.4.3 Tauri for Cross-Platform Deployment
 
Tauri [15] is a framework for building desktop and mobile applications with a Rust backend and a web-based frontend. Unlike Electron, which bundles a full Chromium browser, Tauri uses the operating system's native webview. Because of that, the resulting applications are much smaller.
 
 For this project, Tauri makes it possible to produce, from a single codebase:

**Humanized:**

### 2.4.3 Tauri for Cross-Platform Deployment
 
Tauri [15] is a framework for building desktop and mobile applications with a Rust backend and a web-based frontend. Unlike Electron, which bundles a full Chromium browser, Tauri uses the operating system's native webview. The resulting applications are much smaller.
 
 For this project, Tauri makes it possible to produce, from a single codebase:

---

**Original:**

### 2.4.4 MicroFlow and Rust-Based Inference Engines
 
Zhang et al. (2024) present MicroFlow, an efficient Rust-based inference engine that is designed specifically for TinyML deployments [16]. MicroFlow shows that Rust's zero-cost abstractions and its lack of a garbage collector make it realistic for inference on microcontrollers with as little as 256 KB of RAM. This project targets more capable devices, such as smartphones and laptops, but MicroFlow supports the broader idea that Rust is a viable language for production ML inference at the edge.
 
 ## 2.5 The PlantVillage Dataset

**Humanized:**

### 2.4.4 MicroFlow and Rust-Based Inference Engines
 
Zhang et al. (2024) present MicroFlow, an efficient Rust-based inference engine designed specifically for TinyML deployments [16]. MicroFlow shows that Rust's zero-cost abstractions and its lack of a garbage collector make it realistic for inference on microcontrollers with as little as 256 KB of RAM. This project targets more capable devices, such as smartphones and laptops, but MicroFlow supports the broader idea that Rust is a viable language for production ML inference at the edge.
 
 ## 2.5 The PlantVillage Dataset

---

**Original:**

The dataset includes both healthy and diseased classes across 38 categories, so the model can learn to distinguish between different disease states and healthy tissue. Note that not every crop has both a healthy and a diseased class. Classes follow the naming convention `Crop___Condition` (for example `Apple___Apple_scab`, `Tomato___healthy`).
 
For this project, the existing train and valid split is merged and then re-split according to the four-pool strategy described in Chapter 3 (20% labeled, 60% stream, 10% validation, 10% test). This makes sure that the SSL pipeline has access to a large pool of unlabeled data while also keeping a held-out test set that is never seen during training.
 
 Because the dataset is pre-balanced, with approximately 2,000 images per class, the experimental setup is simpler. Class imbalance should not distort the results of the label efficiency and class scaling experiments.

**Humanized:**

The dataset includes both healthy and diseased classes across 38 categories, so the model can learn to distinguish between different disease states and healthy tissue. Note that not every crop has both a healthy and a diseased class. Classes follow the naming convention `Crop___Condition` (for example `Apple___Apple_scab`, `Tomato___healthy`).
 
For this project, the existing train and valid split is merged and then re-split according to the four-pool strategy described in Chapter 3 (20% labeled, 60% stream, 10% validation, 10% test). This gives the SSL pipeline access to a large pool of unlabeled data while keeping a held-out test set that is never seen during training.
 
 Because the dataset is pre-balanced, with approximately 2,000 images per class, the experimental setup is simpler. Class imbalance should not distort the results of the label efficiency and class scaling experiments.

---

## Results (`03_results.md`)

**Original:**

# 3. Research Results
 
This chapter describes the system that was built to answer the research question. It covers the architecture, the semi-supervised learning pipeline, three controlled experiments that are relevant to deployment, cross-platform benchmarks and the graphical user interface.
 
 ## 3.1 System Architecture

**Humanized:**

# 3. Research Results
 
This chapter describes the system that was built to answer the research question. It covers the architecture, the semi-supervised learning pipeline, three controlled experiments relevant to deployment, cross-platform benchmarks and the graphical user interface.
 
 ## 3.1 System Architecture

---

**Original:**

- **`incremental_learning`**: a dedicated workspace for the incremental learning experiments, built on Burn 0.21. It is split into library crates (`plant-core`, `plant-dataset`, `plant-training`, `plant-incremental`) and CLI tools (`train`, `evaluate`, `experiment-runner`).
 
The split into two workspaces was a deliberate choice. The incremental learning crate was developed earlier in the project and keeps the class-incremental experiments separate from the main SSL pipeline. Both workspaces share the same CNN architecture and dataset handling logic, so the experimental results remain comparable.
 
 ### 3.1.2 CNN Architecture

**Humanized:**

- **`incremental_learning`**: a dedicated workspace for the incremental learning experiments, built on Burn 0.21. It is split into library crates (`plant-core`, `plant-dataset`, `plant-training`, `plant-incremental`) and CLI tools (`train`, `evaluate`, `experiment-runner`).
 
The two workspaces are separate by design. The incremental learning crate was developed earlier in the project and keeps the class-incremental experiments separate from the main SSL pipeline. Both workspaces share the same CNN architecture and dataset handling logic, so the experimental results remain comparable.
 
 ### 3.1.2 CNN Architecture

---

**Original:**

```
 
As a result, the same model code compiles for CUDA (for GPU-accelerated training) and ndarray (for CPU-only environments). Burn also supports a wgpu backend for cross-platform GPU inference, but that backend was not tested in this project.
 
 ### 3.1.3 Model Size

**Humanized:**

```
 
The same model code compiles for CUDA (GPU-accelerated training) and ndarray (CPU-only environments). Burn also supports a wgpu backend for cross-platform GPU inference, but that backend was not tested in this project.
 
 ### 3.1.3 Model Size

---

**Original:**

The trained model weights take up approximately 916 KB in Burn's native CompactRecorder format. The compiled Rust release binary, which includes the model, the inference runtime and the application code, is approximately 26 MB. The PyTorch checkpoint for the same CNN architecture is similar in size, at only a few MB. **The model weights are similar in both of these stacks.**
 
The important difference is **what has to be present on the end user's device to run inference.** With Rust, the release binary is the only artefact that has to be there. It is a single 26 MB file that contains the compiled runtime and the dependencies. With Python, running the same model requires the Python interpreter, the PyTorch library, with or without CUDA support, and several extra packages. A CUDA-enabled PyTorch wheel alone is typically in the low gigabytes once unpacked [6][7], and a practical environment with TorchVision, NumPy, Pillow and similar packages grows further from there.

To put that in perspective, the Rust `target/` build directory, which is comparable to `node_modules` or a Python virtual environment, is itself around 2.1 GB. This is similar to a PyTorch virtual environment. **Both stacks require gigabytes of tooling during development.** The difference is that Rust's compilation step reduces all of that to a single portable binary, while a Python deployment has to carry its interpreter and library tree to the target device.

For edge deployment, this means that the Rust binary can be distributed over Bluetooth, on a USB stick or through a brief mobile data connection. A Python-based deployment either requires a multi-gigabyte environment to be pre-installed on every device or forces the team to ship a container or bundle that includes the interpreter and wheels.
 
 ## 3.2 Semi-Supervised Learning Pipeline

**Humanized:**

The trained model weights take up approximately 916 KB in Burn's native CompactRecorder format. The compiled Rust release binary, which includes the model, the inference runtime and the application code, is approximately 26 MB. The PyTorch checkpoint for the same CNN architecture is similar in size, at only a few MB. **The model weights are similar in both of these stacks.**
 
The important difference is **what must be present on the end user's device to run inference.** With Rust, the release binary is the only artefact required. It is a single 26 MB file containing the compiled runtime and all dependencies. With Python, running the same model requires the Python interpreter, the PyTorch library (with or without CUDA support) and several extra packages. A CUDA-enabled PyTorch wheel alone is typically in the low gigabytes once unpacked [6][7], and a practical environment with TorchVision, NumPy, Pillow and similar packages grows further from there.

To put that in perspective, the Rust `target/` build directory, which is comparable to `node_modules` or a Python virtual environment, is itself around 2.1 GB. This is similar to a PyTorch virtual environment. **Both stacks require gigabytes of tooling during development.** Rust's compilation step reduces all of that to a single portable binary, whereas a Python deployment must carry its interpreter and library tree to the target device.

For edge deployment, the Rust binary can be distributed over Bluetooth, on a USB stick or through a brief mobile data connection. A Python-based deployment either requires a multi-gigabyte environment on every device or forces the team to ship a container or bundle that includes the interpreter and wheels.
 
 ## 3.2 Semi-Supervised Learning Pipeline

---

**Original:**

| Test | 10% | Final evaluation (never seen during training) |
 
The labeled ratio is intentionally kept low at 20%. This simulates a realistic situation where only a limited amount of expert-annotated data is available.
 
 ### 3.2.2 Training Pipeline
 
**Step 1: Initial supervised training.** The CNN is trained on the 20% labeled pool for 30 epochs using cross-entropy loss, the Adam optimizer and standard data augmentations (horizontal and vertical flip, rotation, brightness, contrast, saturation, blur and noise). This produces a baseline model with approximately 70 to 75% validation accuracy.

**Step 2: Pseudo-labeling simulation.** The trained model is then used to classify images from the 60% unlabeled stream pool. Images are processed in batches of 100, which are referred to as "images per day" in the streaming simulation. For every image, the model produces a softmax probability distribution over all 38 classes. If the maximum predicted probability is above the **confidence threshold of 0.9**, the image is accepted as a pseudo-labeled sample with the predicted class as its label. Images that fall below this threshold are discarded.

**Step 3: Retraining.** Once 200 pseudo-labeled samples have accumulated (the retrain threshold), the model is retrained on the union of the original labeled data and the accepted pseudo-labels. This cycle repeats until all stream data has been processed or validation accuracy plateaus.
 
 The pipeline is exposed as a CLI command:

**Humanized:**

| Test | 10% | Final evaluation (never seen during training) |
 
The labeled ratio is intentionally low at 20%. This simulates a realistic situation where only a limited amount of expert-annotated data is available.
 
 ### 3.2.2 Training Pipeline
 
**Step 1: Initial supervised training.** The CNN is trained on the 20% labeled pool for 30 epochs using cross-entropy loss, the Adam optimizer and standard data augmentations (horizontal and vertical flip, rotation, brightness, contrast, saturation, blur and noise). This produces a baseline model with approximately 70-75% validation accuracy.

**Step 2: Pseudo-labeling simulation.** The trained model is then used to classify images from the 60% unlabeled stream pool. Images are processed in batches of 100, referred to as "images per day" in the streaming simulation. For every image, the model produces a softmax probability distribution over all 38 classes. If the maximum predicted probability exceeds the **confidence threshold of 0.9**, the image is accepted as a pseudo-labeled sample using the predicted class. Images that fall below this threshold are discarded.

**Step 3: Retraining.** Once 200 pseudo-labeled samples have accumulated (the retrain threshold), the model is retrained on the original labeled data plus the accepted pseudo-labels. This cycle repeats until all stream data has been processed or validation accuracy plateaus.
 
 The pipeline is exposed as a CLI command:

---

**Original:**

### 3.2.3 SSL Results
 
The saved checkpoints were re-evaluated with Burn 0.21.0 on the held-out test split using the CUDA backend. The evaluation used the same split configuration as the training pipeline: 20% labeled data, 60% stream data, 10% validation data and 10% test data. The test split contains 8,786 images and was not used during training or pseudo-label selection.
 
 **Table 3.1:** Held-out test evaluation of saved checkpoints

**Humanized:**

### 3.2.3 SSL Results
 
Saved checkpoints were re-evaluated with Burn 0.21.0 on the held-out test split using the CUDA backend. The evaluation used the same split configuration as the training pipeline: 20% labeled data, 60% stream data, 10% validation data and 10% test data. The test split contains 8,786 images and was not used during training or pseudo-label selection.
 
 **Table 3.1:** Held-out test evaluation of saved checkpoints

---

**Original:**

| SSL checkpoint | 8,786 | 94.90% | 94.74% |
 
The saved SSL checkpoint improves held-out test accuracy by 8.84 percentage points and macro F1 by 8.66 percentage points compared with the saved supervised baseline. This supports the central SSL claim more strongly than the earlier validation-only wording, because the comparison is now made on the held-out test split. These values are still single-checkpoint results rather than averages over multiple random seeds.
 
 ## 3.3 Incremental Learning Experiments
 
Three controlled experiments were carried out to evaluate parts of the system that are relevant to real-world deployment: how much labeled data is needed, what happens when new classes have to be added to an existing model, and whether the difficulty of adding a class depends on the size of the existing taxonomy.
 
 ### 3.3.1 Experiment 1: Label Efficiency Curve

**Humanized:**

| SSL checkpoint | 8,786 | 94.90% | 94.74% |
 
The saved SSL checkpoint improves held-out test accuracy by 8.84 percentage points and macro F1 by 8.66 percentage points compared with the saved supervised baseline. This supports the central SSL claim more strongly than earlier validation-only comparisons because the comparison is now on the held-out test split. These values are still single-checkpoint results rather than averages over multiple random seeds.
 
 ## 3.3 Incremental Learning Experiments
 
Three controlled experiments were carried out to evaluate parts of the system relevant to real-world deployment: how much labeled data is needed, what happens when new classes are added to an existing model, and whether adding a class depends on the size of the existing taxonomy.
 
 ### 3.3.1 Experiment 1: Label Efficiency Curve

---

**Original:**

**Research question:** How many labeled images per class are needed for acceptable classification accuracy?
 
The model was trained from scratch at seven different labeled data quantities, ranging from 5 up to 500 images per class. All other variables (architecture, augmentation, training schedule) were kept constant.
 
 **Table 3.2:** Label efficiency results

**Humanized:**

**Research question:** How many labeled images per class are needed for acceptable classification accuracy?
 
The model was trained from scratch at seven labeled data quantities, ranging from 5 to 500 images per class. All other variables (architecture, augmentation, training schedule) remained constant.
 
 **Table 3.2:** Label efficiency results

---

**Original:**

**Key findings:**
 
1. With only 5 labeled images per class, the model reaches 34.21% accuracy. This exceeds random chance for 38 classes (2.63%), but it is still too low for practical use.
 2. The sharpest improvement happens between 25 and 100 images per class, where accuracy jumps from 57.89% to 85.53%.
3. Beyond 100 images per class, returns diminish quickly: going from 100 to 200 yields only a 3.22 percentage point gain.
4. **Practical recommendation:** a minimum of 100 labeled images per class is needed for production-viable accuracy, meaning above 80%. SSL methods are useful for bridging the gap whenever fewer labels are available.
 
 ### 3.3.2 Experiment 2: Class Scaling Effect

**Humanized:**

**Key findings:**
 
1. With only 5 labeled images per class, the model reaches 34.21% accuracy. This exceeds random chance for 38 classes (2.63%), but remains too low for practical use.
 2. The sharpest improvement happens between 25 and 100 images per class, where accuracy jumps from 57.89% to 85.53%.
3. Beyond 100 images per class, returns diminish quickly: the jump from 100 to 200 yields only a 3.22 percentage point gain.
4. **Practical recommendation:** at least 100 labeled images per class are needed for production-viable accuracy, meaning above 80%. SSL methods are useful for bridging the gap whenever fewer labels are available.
 
 ### 3.3.2 Experiment 2: Class Scaling Effect

---

**Original:**

**Research question:** Is adding a new class to a small model (5 classes) harder or easier than adding one to a large model (30 classes)? Does the model become more biased towards existing classes as the base grows?
 
Two scenarios were compared. In Scenario A, a model was trained on 5 base classes and then a 6th class was added through incremental learning. In Scenario B, a model was trained on 30 base classes and then a 31st class was added. Both scenarios used the same incremental learning procedure and the same number of labeled samples for the new class.
 
 **Table 3.3:** Class scaling results

**Humanized:**

**Research question:** Is adding a new class to a small model (5 classes) harder or easier than adding one to a large model (30 classes)? Does the model become more biased towards existing classes as the base grows?
 
Two scenarios were compared. In Scenario A, a model was trained on 5 base classes, then a 6th class was added through incremental learning. In Scenario B, a model was trained on 30 base classes, then a 31st class was added. Both scenarios used the same incremental learning procedure and the same number of labeled samples for the new class.
 
 **Table 3.3:** Class scaling results

---

**Original:**

1. The large-base model (30 classes) shows **6× more forgetting** than the small-base model (1.26 percentage points versus 0.21 percentage points). The model is measurably more biased towards existing classes when the base is larger.
2. New class accuracy drops by 3.02 percentage points in the large-base scenario (96.98% versus 100.00%), which confirms that class competition increases as the number of existing classes grows.
 3. Training time scales approximately linearly with the number of classes (5.3× longer for 6× more base classes).
4. **Practical recommendation:** for production systems with many existing classes, use incremental learning methods such as Learning without Forgetting (LwF), Elastic Weight Consolidation (EWC) or rehearsal-based approaches to keep catastrophic forgetting under control. Accuracy on existing classes should be checked after every model update.
 
 ### 3.3.3 Experiment 3: New Class Position Effect

**Humanized:**

1. The large-base model (30 classes) shows **6× more forgetting** than the small-base model (1.26 percentage points versus 0.21 percentage points). The model is measurably more biased towards existing classes when the base is larger.
2. New class accuracy drops by 3.02 percentage points in the large-base scenario (96.98% versus 100.00%), confirming that class competition increases as the number of existing classes grows.
 3. Training time scales approximately linearly with the number of classes (5.3× longer for 6× more base classes).
4. **Practical recommendation:** for production systems with many existing classes, use incremental learning methods such as Learning without Forgetting (LwF), Elastic Weight Consolidation (EWC) or rehearsal-based approaches to control catastrophic forgetting. Accuracy on existing classes should be checked after every model update.
 
 ### 3.3.3 Experiment 3: New Class Position Effect

---

**Original:**

2. The 6th class passes 70% accuracy with only 50 samples. The 31st class does not reach 70% accuracy at any of the tested sample counts (up to 100).
 3. Negative forgetting values in the small-base scenario (for example -2.84% at 50 samples) show that the model occasionally improves on existing classes during incremental training, probably because the additional data acts as implicit regularisation.
4. **Practical recommendation:** when the deployment scenario assumes that new classes will be added over time, start with a broad base model. Adding classes to a large taxonomy requires much more labeled data than adding them to a small one. SSL pseudo-labeling can help bridge that gap by generating extra training samples for the new class.
 
 ## 3.4 Deployment and Benchmarks

**Humanized:**

2. The 6th class passes 70% accuracy with only 50 samples. The 31st class does not reach 70% accuracy at any of the tested sample counts (up to 100).
 3. Negative forgetting values in the small-base scenario (for example -2.84% at 50 samples) show that the model occasionally improves on existing classes during incremental training, probably because the additional data acts as implicit regularisation.
4. **Practical recommendation:** when new classes will be added over time, start with a broad base model. Adding classes to a large taxonomy requires much more labeled data than adding them to a small one. SSL pseudo-labeling can help bridge that gap by generating extra training samples for the new class.
 
 ## 3.4 Deployment and Benchmarks

---

**Original:**

**Mobile performance.** The iPhone 12, running the model through Tauri's Rust backend, reached approximately 80 ms per inference (around 12 FPS) in local testing. This falls within the usability threshold for a camera-based application where a farmer points a phone at a leaf and waits for a classification, though this measurement was taken on a single device and may vary across iOS versions and hardware revisions.
 
**The Jetson result.** The Jetson Orin Nano, which is a dedicated edge AI device costing €350, performed worse than the iPhone 12 in this test, with 120 ms compared to 80 ms. This result shaped the project's deployment strategy. Consumer devices that many users already own can outperform dedicated low-end edge hardware for this specific model. Consequently, the project shifted to a BYOD (Bring Your Own Device) model, which removes extra hardware cost.
 
 **Deployment size advantage.** The compiled binary of approximately 26 MB can be distributed over Bluetooth, a USB drive or a short mobile data connection. A Python/PyTorch deployment requires a multi-gigabyte environment on the target device, which is not practical over those same channels and makes offline-first deployment harder.

**Humanized:**

**Mobile performance.** The iPhone 12, running the model through Tauri's Rust backend, reached approximately 80 ms per inference (around 12 FPS) in local testing. This falls within the usability threshold for a camera-based application where a farmer points a phone at a leaf and waits for a classification, though this measurement was taken on a single device and may vary across iOS versions and hardware revisions.
 
**The Jetson result.** The Jetson Orin Nano, which is a dedicated edge AI device costing €350, performed worse than the iPhone 12 in this test, with 120 ms compared to 80 ms. This result shaped the deployment strategy. Consumer devices that many users already own can outperform dedicated low-end edge hardware for this specific model. Consequently, the project shifted to a BYOD (Bring Your Own Device) model, removing extra hardware cost.
 
 **Deployment size advantage.** The compiled binary of approximately 26 MB can be distributed over Bluetooth, a USB drive or a short mobile data connection. A Python/PyTorch deployment requires a multi-gigabyte environment on the target device, which is not practical over those same channels and makes offline-first deployment harder.

---

**Original:**

Three deployment targets were implemented:
 
1. **Desktop GUI:** a native application with a Svelte 5 and TailwindCSS frontend and a Tauri backend running the Rust Burn model. The GUI offers real-time classification, confidence visualisation and model diagnostics.

2. **Browser (PWA):** an export pipeline converts the Burn model weights to ONNX format (approximately 1.8 MB). The ONNX model can be loaded into an ONNX Runtime Web deployment via a Progressive Web App. The PWA can cache the model through a Service Worker, which would make offline operation possible after the first load. This path was prepared but not fully end-to-end tested on all target browsers.

3. **iPhone 12 (Tauri Mobile):** the same Tauri application, compiled for iOS. The Rust inference backend runs natively on the A14 chip, and the web-based UI takes care of the camera interface. Deployment goes through Xcode or TestFlight.
 
 ## 3.5 Tauri GUI Application
 
The desktop and mobile application was built using Tauri 2.0 with a Svelte 5 frontend. The architecture follows a clear separation of concerns:
 
 - **Frontend (Svelte 5 + TailwindCSS):** handles the user interface, camera access, image upload and result visualisation. It uses the Svelte 5 runes syntax (`$props()`, `$state()`) for reactive state management.

**Humanized:**

Three deployment targets were implemented:
 
1. **Desktop GUI:** a native application with a Svelte 5 frontend, TailwindCSS styling and a Tauri backend running the Rust Burn model. The GUI offers real-time classification, confidence visualisation and model diagnostics.

2. **Browser (PWA):** an export pipeline converts the Burn model weights to ONNX format (approximately 1.8 MB). The ONNX model can be loaded into an ONNX Runtime Web deployment via a Progressive Web App. The PWA can cache the model through a Service Worker, making offline operation possible after the first load. This path was prepared but not fully end-to-end tested on all target browsers.

3. **iPhone 12 (Tauri Mobile):** the same Tauri application, compiled for iOS. The Rust inference backend runs natively on the A14 chip, and the web-based UI handles the camera interface. Deployment goes through Xcode or TestFlight.
 
 ## 3.5 Tauri GUI Application
 
The desktop and mobile application was built with Tauri 2.0 and a Svelte 5 frontend. The architecture follows a clear separation of concerns:
 
 - **Frontend (Svelte 5 + TailwindCSS):** handles the user interface, camera access, image upload and result visualisation. It uses the Svelte 5 runes syntax (`$props()`, `$state()`) for reactive state management.

---

**Original:**

## 3.6 Challenges Encountered
 
Several technical problems arose during development that are worth writing down for reproducibility.
 
 ### 3.6.1 Burn API Boundaries
 
Development started in the `incremental_learning` workspace, while the main SSL pipeline lives in `plantvillage_ssl`. The two workspaces depend on different Burn APIs, especially around the `Module` trait, the optimizer API and the tensor serialisation format. Instead of forcing the incremental learning experiments into the SSL workspace, the two workspaces were kept separate. This kept the experimental results from the incremental learning workspace reproducible, but it also meant maintaining two parallel codebases with the same model architecture.

Model weights cannot be transferred directly between these two workspaces. To share trained models across them, a JSON-based weight export and import mechanism was added. It introduces an extra conversion step, but it preserves weight compatibility across the project.
 
 ### 3.6.2 CUDA Memory Management
 
During the pseudo-labeling simulation, the training loop creates and destroys thousands of tensors per epoch. Burn's CUDA backend allocates GPU memory through a caching allocator, but under sustained load, fragmentation can cause out-of-memory errors even when the total allocated memory is still below the device limit. The fix was to insert explicit synchronisation points at the end of each retraining cycle, so the allocator could compact its memory pools. On the 6 GB laptop RTX 3060 used for development, this reduced peak memory usage from approximately 5.8 GB to 4.2 GB.
 
 ### 3.6.3 Cross-Platform Image Preprocessing
 
The Tauri mobile deployment brought up preprocessing inconsistencies. Desktop image loading through the `image` crate returns images in RGB format, while the iOS camera API returns images in BGRA format. The initial deployment to the iPhone 12 produced incorrect classifications until the colour channel order was corrected in the preprocessing pipeline. This kind of bug is silent: the model still produces a valid probability distribution, but the classifications are systematically wrong because the input channels no longer match what the model was trained on.
 
 ### 3.6.4 Compilation Times
 
Full release builds of the `plantvillage_ssl` workspace take approximately 5 to 7 minutes on the development machine (AMD Ryzen 7, 32 GB RAM, NVMe SSD). This is a known characteristic of Rust's monomorphisation and optimisation passes, especially for generic code that is instantiated across multiple backends. During development, `cargo check` (type-checking without code generation) was used for fast iteration, and `--release` builds were reserved for benchmarking and deployment.
 
 ## 3.7 Limitations of the Experimental Results
 
The results in this chapter should be read with the following limitations in mind. All experiments were carried out on the PlantVillage dataset, which consists of relatively uniform lab-like images. Real-world field images differ in lighting, background, camera quality and disease progression, so accuracy on field data is likely to be lower. The held-out SSL comparison is based on two saved checkpoints with one fixed split and one random seed, not on averages over multiple seeds or datasets. Pseudo-label precision was not independently recomputed in the final Burn 0.21 evaluation run. The mobile inference measurement was taken on one iPhone 12 under controlled conditions. Finally, the wgpu and WASM backends were prepared theoretically but not validated with end-to-end experiments in this project.

**Humanized:**

## 3.6 Challenges Encountered
 
Several technical problems arose during development that are worth recording for reproducibility.
 
 ### 3.6.1 Burn API Boundaries
 
Development started in the `incremental_learning` workspace, whereas the main SSL pipeline lives in `plantvillage_ssl`. The two workspaces depend on different Burn APIs, especially around the `Module` trait, the optimizer API and the tensor serialisation format. Instead of forcing the incremental learning experiments into the SSL workspace, they were kept separate. This kept the incremental learning results reproducible, but also meant maintaining two parallel codebases with the same model architecture.

Model weights cannot be transferred directly between the two workspaces. To share trained models across them, a JSON-based weight export and import mechanism was added. This introduces an extra conversion step, but preserves weight compatibility across the project.
 
 ### 3.6.2 CUDA Memory Management
 
During the pseudo-labeling simulation, the training loop creates and destroys thousands of tensors per epoch. Burn's CUDA backend allocates GPU memory through a caching allocator, but under sustained load fragmentation can cause out-of-memory errors even when total allocated memory is still below the device limit. The fix was to insert explicit synchronisation points at the end of each retraining cycle so the allocator could compact its memory pools. On the 6 GB laptop RTX 3060 used for development, this reduced peak memory usage from approximately 5.8 GB to 4.2 GB.
 
 ### 3.6.3 Cross-Platform Image Preprocessing
 
The Tauri mobile deployment revealed preprocessing inconsistencies. Desktop image loading through the `image` crate returns images in RGB format, while the iOS camera API returns images in BGRA format. The initial deployment to the iPhone 12 produced incorrect classifications until the colour channel order was corrected in the preprocessing pipeline. This kind of bug is silent: the model still produces a valid probability distribution, but the classifications are systematically wrong because the input channels no longer match what the model was trained on.
 
 ### 3.6.4 Compilation Times
 
Full release builds of the `plantvillage_ssl` workspace take approximately 5-7 minutes on the development machine (AMD Ryzen 7, 32 GB RAM, NVMe SSD). This is a known characteristic of Rust's monomorphisation and optimisation passes, especially for generic code instantiated across multiple backends. During development, `cargo check` (type-checking without code generation) was used for fast iteration. `--release` builds were reserved for benchmarking and deployment.
 
 ## 3.7 Limitations of the Experimental Results
 
The results in this chapter should be read with the following limitations in mind. All experiments were carried out on the PlantVillage dataset, which consists of relatively uniform lab-like images. Real-world field images differ in lighting, background, camera quality and disease progression, so accuracy on field data is likely lower. The held-out SSL comparison is based on two saved checkpoints with one fixed split and one random seed, not on averages over multiple seeds or datasets. Pseudo-label precision was not independently recomputed in the final Burn 0.21 evaluation run. The mobile inference measurement was taken on one iPhone 12 under controlled conditions. Finally, the wgpu and WASM backends were prepared theoretically but not validated with end-to-end experiments in this project.

---

## Reflection (`04_reflection.md`)

**Original:**

# 4. Reflection
 
This chapter presents a critical evaluation of the research results. It is organised into three parts: an assessment of the external feedback that was collected, a reflection on the project's strengths and limitations, and a discussion of the broader implications for deployment.
 
 ## 4.1 External Feedback

**Humanized:**

# 4. Reflection
 
This chapter presents a critical evaluation of the research results. It is organised into three parts: an assessment of the external feedback collected, a reflection on the project's strengths and limitations, and a discussion of the broader implications for deployment.
 
 ## 4.1 External Feedback

---

**Original:**

Their feedback is referenced throughout this chapter. Pedro Morais's full answers are provided in Appendix B, and Helena Torres's in Appendix C.
 
The most valuable aspect of this feedback was that it did not reject the project's direction, but it did sharpen the identification of weak points. Both experts recognised the value of a simple pseudo-labeling setup for edge deployment, particularly when the goal is a small and portable system rather than a large research stack. At the same time, both shifted the focus from whether the pipeline executes correctly to whether the generated labels are sufficiently reliable. This distinction is significant: a semi-supervised system can appear successful by accepting large numbers of pseudo-labels, yet remain unsafe if it accepts incorrect samples with high confidence.
 
 Helena Torres raised an important consideration regarding dataset size. In this thesis, PlantVillage is treated as a dataset where labels are expensive relative to unlabeled samples. While this holds for the agricultural annotation setting, Torres noted that 17,000 labeled images would already constitute a large dataset in many medical imaging contexts. In such domains, the primary concern is often not merely the number of labels, but also their quality, consistency and inter-observer variability. This nuance affects how the results should be interpreted. The project demonstrates that Rust and Burn can support an SSL workflow, but it does not establish that the same thresholds and data splits would transfer directly to domains where labels are noisier or where experts disagree.
 
The feedback also clarified the engineering trade-offs involved. Pedro suggested EfficientNet, while Torres mentioned MobileNetV3 and EfficientNet-Lite as natural mobile baselines. A custom CNN was chosen for this project because it offered greater experimental control, was sufficiently small for Burn and fitted within the available video memory. Although this was a defensible thesis decision, it also means that the model comparison remains incomplete. A more comprehensive evaluation would compare the custom CNN against at least one established mobile architecture under identical Rust or ONNX deployment constraints.
 
 ## 4.2 Self-Reflection

**Humanized:**

Their feedback is referenced throughout this chapter. Pedro Morais's full answers are provided in Appendix B, and Helena Torres's in Appendix C.
 
The most valuable aspect of this feedback was that it did not reject the project's direction, but sharpened the identification of weak points. Both experts recognised the value of a simple pseudo-labeling setup for edge deployment, particularly when the goal is a small and portable system rather than a large research stack. At the same time, both shifted the focus from whether the pipeline executes correctly to whether the generated labels are sufficiently reliable. This distinction is significant: a semi-supervised system can appear successful by accepting large numbers of pseudo-labels, yet remain unsafe if it accepts incorrect samples with high confidence.
 
 Helena Torres raised an important consideration regarding dataset size. In this thesis, PlantVillage is treated as a dataset where labels are expensive relative to unlabeled samples. While this holds for the agricultural annotation setting, Torres noted that 17,000 labeled images would already constitute a large dataset in many medical imaging contexts. In such domains, the primary concern is often not merely the number of labels, but also their quality, consistency and inter-observer variability. This nuance affects how the results should be interpreted. The project demonstrates that Rust and Burn can support an SSL workflow, but it does not establish that the same thresholds and data splits would transfer directly to domains where labels are noisier or where experts disagree.
 
The feedback also clarified the engineering trade-offs involved. Pedro suggested EfficientNet, while Torres mentioned MobileNetV3 and EfficientNet-Lite as natural mobile baselines. A custom CNN was chosen for this project because it offered greater experimental control, was sufficiently small for Burn and fitted within the available video memory. Although this was a defensible thesis decision, the model comparison remains incomplete. A more comprehensive evaluation would compare the custom CNN against at least one established mobile architecture under identical Rust or ONNX deployment constraints.
 
 ## 4.2 Self-Reflection

---

**Original:**

### 4.2.1 Strengths
 
**Deployment size and portability.** The compiled binary of approximately 26 MB represents a meaningful improvement over a Python/PyTorch deployment. Both stacks require gigabytes of tooling during development: Rust's `target/` directory is approximately 2.1 GB, which is comparable to a PyTorch virtual environment. The decisive difference is that Rust's compilation step reduces the entire environment to a single portable binary. A Python deployment, by contrast, must carry its interpreter and library tree to the target device. This distinction determines which distribution channels are viable for edge deployment. A file that can be transferred over Bluetooth or a USB drive differs fundamentally from a system that first requires a multi-gigabyte environment.
 
 **The BYOD pivot.** The benchmark results (Table 3.7) provided a clear, data-driven basis for moving away from dedicated edge hardware. The Jetson Orin Nano, priced at €350, proved slower in this test than a consumer smartphone that many users already own. This pivot removed the single largest capital cost associated with deployment.

**Humanized:**

### 4.2.1 Strengths
 
**Deployment size and portability.** The compiled binary of approximately 26 MB represents a meaningful improvement over a Python/PyTorch deployment. Both stacks require gigabytes of tooling during development: Rust's `target/` directory is approximately 2.1 GB, comparable to a PyTorch virtual environment. The decisive difference is that Rust's compilation step reduces the entire environment to a single portable binary. A Python deployment, by contrast, must carry its interpreter and library tree to the target device. This distinction determines which distribution channels are viable for edge deployment. A file that can be transferred over Bluetooth or a USB drive differs fundamentally from a system that first requires a multi-gigabyte environment.
 
 **The BYOD pivot.** The benchmark results (Table 3.7) provided a clear, data-driven basis for moving away from dedicated edge hardware. The Jetson Orin Nano, priced at €350, proved slower in this test than a consumer smartphone that many users already own. This pivot removed the single largest capital cost associated with deployment.

---

**Original:**

**External expert feedback.** Pedro Morais reviewed the approach and confirmed that the 90% confidence threshold is a sensible starting point for controlled imaging conditions. He recommended retaining a fixed threshold for this application, which the current pipeline does. Torres agreed that pseudo-labeling is appropriate when deployment simplicity is prioritised, but cautioned that the initial model should be examined carefully at that 90% threshold. She specifically highlighted the need to check for systematic errors, class-specific bias and confusion between visually similar classes. Rather than relying on a single global threshold, she suggested investigating per-class thresholds or uncertainty estimation.
 
Both experts also commented on the model architecture and preprocessing. Pedro suggested evaluating EfficientNet, and Torres recommended comparing the custom CNN with MobileNetV3 or EfficientNet-Lite. This confirms that the custom architecture is a valid choice for an edge-focused prototype, but it is probably not the strongest mobile baseline available. Regarding augmentation, Pedro cautioned that contrast and brightness manipulations can reduce stability. Torres offered a broader warning: intensity and spatial augmentations are useful when labeled data is scarce, but deformable transformations should be used with care in medical imaging because they can distort clinically meaningful structures. For this project, controlled augmentation ablations are therefore needed, rather than assuming that every augmentation automatically improves robustness.
 
**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is constrained by the accuracy of the model trained on the 20% labeled subset. If the initial model systematically misclassifies certain classes, those errors propagate through the pseudo-labeling cycle. Techniques such as co-training, in which two models examine different views of the data, could mitigate this risk, but they were not implemented because of video memory constraints on edge devices.
 
 This is also where the fixed confidence threshold becomes a limitation. The 90% threshold is straightforward to explain and implement, which is valuable for an edge prototype. However, confidence is not equivalent to correctness. A model can be overconfident on visually similar plant diseases, on images with unusual lighting, or on classes that were under-represented during training. The thesis reports global accuracy, but a more robust SSL system would also examine per-class precision, recall, F1-score and confusion matrices before trusting accepted pseudo-labels. This would reveal whether the model performs strongly only on common or visually distinct classes, or whether it behaves consistently across the full taxonomy.

**Humanized:**

**External expert feedback.** Pedro Morais reviewed the approach and confirmed that the 90% confidence threshold is a sensible starting point for controlled imaging conditions. He recommended retaining a fixed threshold for this application, which the current pipeline does. Torres agreed that pseudo-labeling is appropriate when deployment simplicity is prioritised, but cautioned that the initial model should be examined carefully at that 90% threshold. She specifically highlighted the need to check for systematic errors, class-specific bias and confusion between visually similar classes. Rather than relying on a single global threshold, she suggested investigating per-class thresholds or uncertainty estimation.
 
Both experts also commented on the model architecture and preprocessing. Pedro suggested evaluating EfficientNet, and Torres recommended comparing the custom CNN with MobileNetV3 or EfficientNet-Lite. This confirms that the custom architecture is a valid choice for an edge-focused prototype, but is probably not the strongest mobile baseline available. Regarding augmentation, Pedro cautioned that contrast and brightness manipulations can reduce stability. Torres offered a broader warning: intensity and spatial augmentations are useful when labeled data is scarce, but deformable transformations should be used with care in medical imaging because they can distort clinically meaningful structures. For this project, controlled augmentation ablations are needed, rather than assuming that every augmentation automatically improves robustness.
 
**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is constrained by the accuracy of the model trained on the 20% labeled subset. If the initial model systematically misclassifies certain classes, those errors propagate through the pseudo-labeling cycle. Techniques such as co-training, in which two models examine different views of the data, could mitigate this risk, but were not implemented because of video memory constraints on edge devices.
 
 This is also where the fixed confidence threshold becomes a limitation. The 90% threshold is straightforward to explain and implement, which is valuable for an edge prototype. However, confidence is not equivalent to correctness. A model can be overconfident on visually similar plant diseases, on images with unusual lighting, or on classes that were under-represented during training. The thesis reports global accuracy, but a more robust SSL system would also examine per-class precision, recall, F1-score and confusion matrices before trusting accepted pseudo-labels. This would reveal whether the model performs strongly only on common or visually distinct classes, or whether it behaves consistently across the full taxonomy.

---

**Original:**

Another missing element is calibration analysis. If a prediction score of 90% does not correspond to approximately 90% correctness, the threshold has a different meaning than assumed. Calibration plots or expected calibration error could help determine whether a single global threshold is appropriate. If certain classes are consistently overconfident, per-class thresholds would be more defensible. This was not implemented, but Torres's feedback makes it clear that calibration analysis would be one of the first improvements before applying this approach in a higher-risk setting.
 
**No field validation.** All experiments were conducted on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important respects: varying lighting, background vegetation, leaf angle, camera quality and the presence of multiple diseases on the same leaf. The model's performance on field-captured images is unknown and is likely lower than the figures reported here.
 
 The absence of field validation also affects the preprocessing pipeline. The iOS channel-ordering bug demonstrated that even a small difference between BGRA and RGB can silently break predictions while the application still appears to function. For future deployments, preprocessing should be validated with fixed reference images on every target platform. The same image should produce identical tensor values and predictions on desktop, mobile and any exported ONNX path. Although this appears to be a minor engineering detail, it is exactly the kind of detail that determines whether a portable machine learning system is reliable.

**Humanized:**

Another missing element is calibration analysis. If a prediction score of 90% does not correspond to approximately 90% correctness, the threshold has a different meaning than assumed. Calibration plots or expected calibration error could help determine whether a single global threshold is appropriate. If certain classes are consistently overconfident, per-class thresholds would be more defensible. This was not implemented, but Torres's feedback makes it clear that calibration analysis would be one of the first improvements before applying this approach in a higher-risk setting.
 
**No field validation.** All experiments were conducted on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important respects: varying lighting, background vegetation, leaf angle, camera quality and the presence of multiple diseases on the same leaf. The model's performance on field-captured images is unknown and likely lower than the figures reported here.
 
 The absence of field validation also affects the preprocessing pipeline. The iOS channel-ordering bug demonstrated that even a small difference between BGRA and RGB can silently break predictions while the application still appears to function. For future deployments, preprocessing should be validated with fixed reference images on every target platform. The same image should produce identical tensor values and predictions on desktop, mobile and any exported ONNX path. Although this appears to be a minor engineering detail, it is exactly the kind of detail that determines whether a portable machine learning system is reliable.

---

**Original:**

Building a working prototype and deploying it to end users are distinct challenges. Several barriers emerged during development that stand between a functional system and a practical tool.
 
The most significant barrier is trust. If the model predicts "bacterial spot" with high confidence and a farmer treats accordingly, but the actual condition is different, the tool has caused harm rather than benefit. This is why the graphical user interface displays a confidence bar rather than a single prediction. Even so, non-technical users may not know how to interpret a 72% confidence score. The interface must communicate uncertainty honestly without becoming confusing, which constitutes a user experience problem as much as a machine learning problem.
 
Device diversity is another concern. The BYOD strategy requires the system to function across whatever phone or laptop a user happens to own. The cross-platform benchmarks (Table 3.7) demonstrate that the system can operate across hardware configurations, but the preprocessing bug on iOS (Section 3.6.3), where BGRA versus RGB channel ordering silently produced incorrect classifications, is representative of the kind of problem that only appears on real devices. Additional bugs of this nature are likely to emerge on untested hardware.
 
 Finally, there is the update problem. The initial installation is small enough (26 MB) to distribute offline, but distributing model improvements or new classes presents challenges. Agricultural extension workers or local community centres could serve as distribution points, but this requires coordination beyond software engineering.

**Humanized:**

Building a working prototype and deploying it to end users are distinct challenges. Several barriers emerged during development that stand between a functional system and a practical tool.
 
The most significant barrier is trust. If the model predicts "bacterial spot" with high confidence and a farmer treats accordingly, but the actual condition is different, the tool has caused harm rather than benefit. This is why the graphical user interface displays a confidence bar rather than a single prediction. Even so, non-technical users may not know how to interpret a 72% confidence score. The interface must communicate uncertainty honestly without becoming confusing. This is a user experience problem as much as a machine learning problem.
 
Device diversity is another concern. The BYOD strategy requires the system to function across whatever phone or laptop a user happens to own. The cross-platform benchmarks (Table 3.7) show that the system can operate across hardware configurations, but the preprocessing bug on iOS (Section 3.6.3), where BGRA versus RGB channel ordering silently produced incorrect classifications, is representative of problems that only appear on real devices. Additional bugs of this nature are likely to emerge on untested hardware.
 
 Finally, there is the update problem. The initial installation is small enough (26 MB) to distribute offline, but distributing model improvements or new classes presents challenges. Agricultural extension workers or local community centres could serve as distribution points, but this requires coordination beyond software engineering.

---

**Original:**

The measurements from this project support a straightforward economic argument.
 
First, the SSL pipeline reduces the labeling requirement from 100% of the dataset to approximately 20%, which translates directly into annotation budget savings. For an 87,000-image dataset, this represents the difference between a full expert-annotation budget and a substantially smaller one, which matters for research groups and agricultural extensions with limited funding.
 
Second, the BYOD pivot eliminates hardware costs entirely. The Jetson Orin Nano benchmark (Table 3.7) demonstrated that a €350 dedicated device was slower than a consumer phone, so there is no economic justification for purchasing one.
 
 Third, once installed, the marginal cost of each classification is zero: no cloud API calls, no per-prediction charges and no bandwidth costs.

**Humanized:**

The measurements from this project support a straightforward economic argument.
 
First, the SSL pipeline reduces the labeling requirement from 100% of the dataset to approximately 20%, which translates directly into annotation budget savings. For an 87,000-image dataset, this is the difference between a full expert-annotation budget and a substantially smaller one, which matters for research groups and agricultural extensions with limited funding.
 
Second, the BYOD pivot eliminates hardware costs entirely. The Jetson Orin Nano benchmark (Table 3.7) showed that a €350 dedicated device was slower than a consumer phone, so there is no economic justification for purchasing one.
 
 Third, once installed, the marginal cost of each classification is zero: no cloud API calls, no per-prediction charges and no bandwidth costs.

---

**Original:**

The offline-first architecture offers a practical advantage beyond connectivity. Because the model runs entirely on the user's device, no image data is transmitted to an external server. This removes server-side data-processing risks and reduces the compliance surface for privacy regulations such as the General Data Protection Regulation (GDPR), because the developer does not need to store or process user images on a backend. This does not eliminate all legal considerations: if the device is managed by an organisation, or if inference logs are collected, GDPR may still apply. Device-level security measures such as local encryption and access control therefore remain important.
 
Whether this kind of tool would reduce crop losses cannot be answered from a computer science thesis alone. What can be stated is that the technical barriers—deployment size, connectivity requirements and inference speed—are no longer the principal bottleneck. The remaining barriers concern trust, usability and distribution. Addressing those requires field trials and user research rather than additional engineering alone.
 
 ### 4.3.4 Future Research Directions

**Humanized:**

The offline-first architecture offers a practical advantage beyond connectivity. Because the model runs entirely on the user's device, no image data is transmitted to an external server. This removes server-side data-processing risks and reduces the compliance surface for privacy regulations such as the General Data Protection Regulation (GDPR), because the developer does not need to store or process user images on a backend. This does not eliminate all legal considerations: if the device is managed by an organisation, or if inference logs are collected, GDPR may still apply. Device-level security measures such as local encryption and access control therefore remain important.
 
Whether this kind of tool would reduce crop losses cannot be answered from a computer science thesis alone. What can be stated is that the technical barriers, deployment size, connectivity requirements and inference speed, are no longer the principal bottleneck. The remaining barriers concern trust, usability and distribution. Addressing those requires field trials and user research rather than additional engineering alone.
 
 ### 4.3.4 Future Research Directions

---

**Original:**

Several directions for future work merit attention:
 
1. **Field validation:** deploying the system with actual users and measuring classification accuracy on real-world images, under varying lighting, backgrounds and camera quality. The PlantVillage results are promising, but controlled benchmarks and real-world performance are not equivalent.
 2. **Active learning:** instead of discarding every low-confidence sample, the system could flag uncertain predictions and request human input. This would transform the SSL loop into a targeted labeling tool.
 3. **Federated learning:** multiple deployed devices could share model updates without sharing raw images, which would allow the model to improve over time while keeping data local.

**Humanized:**

Several directions for future work merit attention:
 
1. **Field validation:** deploying the system with actual users and measuring classification accuracy on real-world images under varying lighting, backgrounds and camera quality. The PlantVillage results are promising, but controlled benchmarks and real-world performance are not equivalent.
 2. **Active learning:** instead of discarding every low-confidence sample, the system could flag uncertain predictions and request human input. This would transform the SSL loop into a targeted labeling tool.
 3. **Federated learning:** multiple deployed devices could share model updates without sharing raw images, which would allow the model to improve over time while keeping data local.

---

## Recommendations (`05_advice.md`)

**Original:**

# 5. Recommendations
 
This chapter presents practical recommendations derived from the experimental results in Chapter 3 and the reflections in Chapter 4. These recommendations are intended for researchers and practitioners who intend to implement semi-supervised neural networks in Rust for edge deployment. The guidance is grounded in empirical findings and is structured by topic rather than as a chronological procedure.
 
 ## 5.1 Framework Selection

**Humanized:**

# 5. Recommendations
 
This chapter presents practical recommendations derived from the experimental results in Chapter 3 and the reflections in Chapter 4. These recommendations are intended for researchers and practitioners who intend to implement semi-supervised neural networks in Rust for edge deployment. The guidance is grounded in empirical findings and structured by topic rather than as a chronological procedure.
 
 ## 5.1 Framework Selection

---

**Original:**

For use cases involving SSL or iterative training loops, Burn is the recommended choice. Its `Module` derive macro and backend generics allow the same model code to compile for CUDA training and mobile inference without modification.
 
A practical consideration is Rust's compilation time. A full release build of the `plantvillage_ssl` workspace requires five minutes or more. It is advisable to use `cargo check` during development and reserve `--release` builds for testing and deployment. Enabling the `sccache` compiler cache can reduce rebuild times.
 
 ## 5.2 Minimum Labeled Data Requirements

**Humanized:**

For use cases involving SSL or iterative training loops, Burn is the recommended choice. Its `Module` derive macro and backend generics allow the same model code to compile for CUDA training and mobile inference without modification.
 
A practical consideration is Rust's compilation time. A full release build of the `plantvillage_ssl` workspace requires five minutes or more. Use `cargo check` during development and reserve `--release` builds for testing and deployment. Enabling the `sccache` compiler cache can reduce rebuild times.
 
 ## 5.2 Minimum Labeled Data Requirements

---

**Original:**

- **200 or more images per class:** diminishing returns. Effort spent on labeling beyond 200 images per class is better invested in improving the pseudo-labeling pipeline or collecting field data.
 
It is recommended to collect at least 100 labeled images per class before initiating SSL. If this is not feasible, the limited labeling budget should be allocated to class pairs that are most easily confused, such as diseases with similar visual symptoms. This strategy gives the initial model the most informative decision boundary in the regions where it matters most.
 
 ## 5.3 Pseudo-Labeling Pipeline Design
 
Based on the experimental results and the literature review, the parameter settings in Table 5.2 are recommended as initial values.
 
 **Table 5.2:** Recommended pseudo-labeling parameters

**Humanized:**

- **200 or more images per class:** diminishing returns. Effort spent on labeling beyond 200 images per class is better invested in improving the pseudo-labeling pipeline or collecting field data.
 
Collect at least 100 labeled images per class before initiating SSL. If this is not feasible, allocate the limited labeling budget to class pairs that are most easily confused, such as diseases with similar visual symptoms. This strategy gives the initial model the most informative decision boundary in the regions where it matters most.
 
 ## 5.3 Pseudo-Labeling Pipeline Design
 
Based on the experimental results and literature review, the parameter settings in Table 5.2 are recommended as initial values.
 
 **Table 5.2:** Recommended pseudo-labeling parameters

---

**Original:**

5. Evaluate the final model on the held-out test set, which must not be used during training or pseudo-label selection.
 
A critical methodological constraint is that the test set must not be used for any decision during training, including pseudo-label threshold tuning. Using the test set for any optimisation during training is one of the most common causes of optimistic accuracy estimates in SSL research.
 
 The global 0.9 threshold should not be treated as permanently fixed. It is a sensible starting point, but per-class acceptance rates, confidence histograms and validation accuracy should be logged after every retraining cycle. If one class receives far more pseudo-labels than the others, or if visually similar classes are repeatedly confused, switching to per-class thresholds or adding uncertainty estimation should be considered before continuing the SSL cycle.

**Humanized:**

5. Evaluate the final model on the held-out test set, which must not be used during training or pseudo-label selection.
 
A critical methodological constraint: the test set must not be used for any decision during training, including pseudo-label threshold tuning. Using the test set for any optimisation during training is one of the most common causes of optimistic accuracy estimates in SSL research.
 
 The global 0.9 threshold should not be treated as permanently fixed. It is a sensible starting point, but per-class acceptance rates, confidence histograms and validation accuracy should be logged after every retraining cycle. If one class receives far more pseudo-labels than the others, or if visually similar classes are repeatedly confused, switching to per-class thresholds or adding uncertainty estimation should be considered before continuing the SSL cycle.

---

**Original:**

## 5.4 Incremental Class Addition
 
In deployment scenarios where new classes are added over time—which is expected in most real-world agricultural applications—the experimental results from Chapter 3 yield the following guidelines:
 
 1. **Start with a broad base model.** The class scaling experiment (Table 3.3) shows that adding a class to a larger base causes more forgetting. A larger base, however, means that the model covers more diseases from the outset, which reduces the frequency of subsequent updates.

**Humanized:**

## 5.4 Incremental Class Addition
 
In deployment scenarios where new classes are added over time, which is expected in most real-world agricultural applications, the experimental results from Chapter 3 yield the following guidelines:
 
 1. **Start with a broad base model.** The class scaling experiment (Table 3.3) shows that adding a class to a larger base causes more forgetting. A larger base, however, means that the model covers more diseases from the outset, which reduces the frequency of subsequent updates.

---

**Original:**

- The deployment size of approximately 26 MB is small enough to install over Bluetooth or a brief mobile connection.
 
The exception to this recommendation is headless deployments, such as camera traps or automated greenhouse systems. In those cases, a Raspberry Pi 4 or 5 with the CPU backend is usually more suitable than a GPU-based edge device.
 
 ## 5.6 Early Device Testing

**Humanized:**

- The deployment size of approximately 26 MB is small enough to install over Bluetooth or a brief mobile connection.
 
The exception is headless deployments, such as camera traps or automated greenhouse systems. In those cases, a Raspberry Pi 4 or 5 with the CPU backend is usually more suitable than a GPU-based edge device.
 
 ## 5.6 Early Device Testing

---

**Original:**

- **Latency differences:** the CPU backend can be orders of magnitude slower than the GPU backend. The wgpu backend may behave differently on mobile GPUs than on desktop GPUs.
 - **Memory pressure:** mobile operating systems aggressively kill background applications that use excessive memory. A model that runs correctly in isolation may still fail when the device is also running a camera preview.
- **Image preprocessing mismatches:** camera APIs return images in various formats (NV21, BGRA, JPEG). Ensuring that the preprocessing pipeline handles all of these correctly is non-trivial.
 - **Permissions and sandboxing:** iOS and Android restrict file system access, camera access and background processing. These restrictions affect how the model is loaded and where inference results can be stored.
 
It is recommended to establish a minimal deployment on the target device within the first two weeks of development. A minimal Tauri application that loads the model and runs inference on a single image is sufficient to validate the deployment pipeline and surface integration issues while they are still inexpensive to fix.
 
For image preprocessing specifically, a small set of fixed reference images should be retained, and the model's numerical outputs should be compared across desktop, mobile and web runtimes. This catches errors in channel ordering, resizing and normalisation before they become silent deployment bugs.
 
 ## 5.7 Common Pitfalls and Mitigations

**Humanized:**

- **Latency differences:** the CPU backend can be orders of magnitude slower than the GPU backend. The wgpu backend may behave differently on mobile GPUs than on desktop GPUs.
 - **Memory pressure:** mobile operating systems aggressively kill background applications that use excessive memory. A model that runs correctly in isolation may still fail when the device is also running a camera preview.
- **Image preprocessing mismatches:** camera APIs return images in various formats (NV21, BGRA, JPEG). Ensuring that the preprocessing pipeline handles all of them correctly is non-trivial.
 - **Permissions and sandboxing:** iOS and Android restrict file system access, camera access and background processing. These restrictions affect how the model is loaded and where inference results can be stored.
 
Establish a minimal deployment on the target device within the first two weeks of development. A minimal Tauri application that loads the model and runs inference on a single image is sufficient to validate the deployment pipeline and surface integration issues while they are still inexpensive to fix.
 
For image preprocessing specifically, retain a small set of fixed reference images and compare the model's numerical outputs across desktop, mobile and web runtimes. This catches errors in channel ordering, resizing and normalisation before they become silent deployment bugs.
 
 ## 5.7 Common Pitfalls and Mitigations

---

## Conclusion (`06_conclusion.md`)

**Original:**

This thesis set out to answer the question: **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**
 
The short answer is that it can be done by combining a lightweight custom CNN built with the Burn framework, an iterative pseudo-labeling pipeline with confidence-based filtering, and a cross-platform deployment strategy built around Tauri. The longer answer involves the specific trade-offs, numbers and lessons learned that resulted from this project.
 
 ## 6.1 What Was Built
 
The implementation is efficient in the ways that are relevant to edge deployment. The trained model weights are approximately 916 KB in Burn's native format. The compiled binary is approximately 26 MB, and that is the only file that needs to be present on the target device. A comparable Python/PyTorch deployment requires several gigabytes of interpreter, libraries and supporting packages, even though the model weights themselves are approximately the same size in both stacks. The Burn binary starts in under 100 ms, compared with approximately 3 seconds for a PyTorch application. On a laptop GPU, an NVIDIA RTX 3060, the saved SSL checkpoint reaches an inference latency of 0.42 ms, which corresponds to approximately 2,406 FPS. On an iPhone 12 through Tauri, it reached approximately 80 ms per inference in local testing, which is fast enough for a camera-based classification app.
 
 The automatic labeling pipeline was implemented as an offline pseudo-labeling workflow. Starting from only 20% labeled data, it generates labels for the remaining unlabeled pool. On the held-out test split of 8,786 images, the saved supervised baseline reached 86.06% top-1 accuracy and 86.08% macro F1. The saved SSL checkpoint reached 94.90% top-1 accuracy and 94.74% macro F1. This represents an improvement of 8.84 percentage points in accuracy and 8.66 percentage points in macro F1 without extra human annotation, although these figures come from one saved split and should not be treated as guaranteed performance across datasets. The system also runs fully offline, with no cloud API calls, no data leaving the device and no network connection required after installation.

**Humanized:**

This thesis set out to answer the question: **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**
 
The short answer is that it can be done by combining a lightweight custom CNN built with the Burn framework, an iterative pseudo-labeling pipeline with confidence-based filtering, and a cross-platform deployment strategy built around Tauri. The longer answer involves the specific trade-offs, numbers and lessons learned from this project.
 
 ## 6.1 What Was Built
 
The implementation is efficient in ways relevant to edge deployment. The trained model weights are approximately 916 KB in Burn's native format. The compiled binary is approximately 26 MB, and is the only file that must be present on the target device. A comparable Python/PyTorch deployment requires several gigabytes of interpreter, libraries and supporting packages, even though the model weights themselves are approximately the same size in both stacks. The Burn binary starts in under 100 ms, compared with approximately 3 seconds for a PyTorch application. On a laptop GPU, an NVIDIA RTX 3060, the saved SSL checkpoint reaches an inference latency of 0.42 ms, corresponding to approximately 2,406 FPS. On an iPhone 12 through Tauri, it reached approximately 80 ms per inference in local testing, which is fast enough for a camera-based classification app.
 
 The automatic labeling pipeline was implemented as an offline pseudo-labeling workflow. Starting from only 20% labeled data, it generates labels for the remaining unlabeled pool. On the held-out test split of 8,786 images, the saved supervised baseline reached 86.06% top-1 accuracy and 86.08% macro F1. The saved SSL checkpoint reached 94.90% top-1 accuracy and 94.74% macro F1. This represents an improvement of 8.84 percentage points in accuracy and 8.66 percentage points in macro F1 without extra human annotation, although these figures come from one saved split and should not be treated as guaranteed performance across datasets. The system also runs fully offline, with no cloud API calls, no data leaving the device and no network connection required after installation.

---

**Original:**

## 6.2 What the Experiments Show
 
The three controlled experiments give the quantitative answers that are needed for deployment decisions. The label efficiency curve shows that 100 labeled images per class is the minimum for useful accuracy, exceeding 80%, with diminishing returns beyond 200. The class scaling experiment shows that adding a new class to a 30-class model causes six times more catastrophic forgetting than adding one to a 5-class model, 1.26 percentage points compared with 0.21 percentage points. This is a clear signal that incremental learning methods such as rehearsal or EWC become important once the model reaches production scale. The new class position experiment shows the extra effort that is needed. At 50 labeled samples, learning a new class as the 6th class in a small taxonomy reaches 84% accuracy, while learning it as the 31st class in a large one only reaches 26%. These numbers are useful planning inputs for anyone extending a deployed model.
 
 ## 6.3 Hardware and Deployment Strategy
 
One unexpected finding was the hardware comparison. The Jetson Orin Nano, a dedicated edge AI device of around €350, reached 120 ms inference. The iPhone 12 reached 80 ms, without extra hardware cost. This result pushed the project away from dedicated edge hardware and towards a Bring Your Own Device strategy, because it removes the largest capital cost for deployment.
 
 Three deployment paths were prepared: a desktop Tauri application, an iOS build through Tauri Mobile, and a browser-based Progressive Web App using ONNX Runtime Web. The desktop and iOS paths were tested end to end. The PWA path was prepared but not fully validated across all browsers.

**Humanized:**

## 6.2 What the Experiments Show
 
The three controlled experiments give the quantitative answers needed for deployment decisions. The label efficiency curve shows that 100 labeled images per class is the minimum for useful accuracy, exceeding 80%, with diminishing returns beyond 200. The class scaling experiment shows that adding a new class to a 30-class model causes six times more catastrophic forgetting than adding one to a 5-class model: 1.26 percentage points compared with 0.21 percentage points. This is a clear signal that incremental learning methods such as rehearsal or EWC become important once the model reaches production scale. The new class position experiment shows the extra effort needed. At 50 labeled samples, learning a new class as the 6th class in a small taxonomy reaches 84% accuracy, while learning it as the 31st class in a large one only reaches 26%. These numbers are useful planning inputs for anyone extending a deployed model.
 
 ## 6.3 Hardware and Deployment Strategy
 
One unexpected finding was the hardware comparison. The Jetson Orin Nano, a dedicated edge AI device at around €350, reached 120 ms inference. The iPhone 12 reached 80 ms without extra hardware cost. This result pushed the project away from dedicated edge hardware and towards a Bring Your Own Device strategy, removing the largest capital cost for deployment.
 
 Three deployment paths were prepared: a desktop Tauri application, an iOS build through Tauri Mobile, and a browser-based Progressive Web App using ONNX Runtime Web. The desktop and iOS paths were tested end to end. The PWA path was prepared but not fully validated across all browsers.

---

**Original:**

## 6.4 Limitations and Honest Assessment
 
The reflection in Chapter 4 also points out the main limitations. The system has not yet been validated on real agricultural images from the field. Pseudo-label quality is still limited by the quality of the first model. The Burn ecosystem is less mature than PyTorch. The external expert feedback supports the main direction of the project, but it also clarifies the next steps: compare the custom CNN with established mobile architectures, analyse pseudo-label quality per class, test multiple input resolutions and validate the model under different acquisition conditions.
 
The most important next step is field validation, meaning tests on real images, captured by real users, under uncontrolled conditions. The PlantVillage results are encouraging, but controlled benchmarks and real-world performance are not the same thing. Without that step, the thesis answers the technical research question without proving that the solution works in practice for farmers.
 
 ## 6.5 Answer to the Research Question

**Humanized:**

## 6.4 Limitations and Honest Assessment
 
The reflection in Chapter 4 also points out the main limitations. The system has not yet been validated on real agricultural images from the field. Pseudo-label quality is still limited by the quality of the first model. The Burn ecosystem is less mature than PyTorch. The external expert feedback supports the main direction of the project, but also clarifies the next steps: compare the custom CNN with established mobile architectures, analyse pseudo-label quality per class, test multiple input resolutions and validate the model under different acquisition conditions.
 
The most important next step is field validation: tests on real images, captured by real users, under uncontrolled conditions. The PlantVillage results are encouraging, but controlled benchmarks and real-world performance are not the same thing. Without that step, the thesis answers the technical research question without proving that the solution works in practice for farmers.
 
 ## 6.5 Answer to the Research Question

---
