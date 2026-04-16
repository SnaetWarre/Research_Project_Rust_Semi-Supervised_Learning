# Glossary

**Backend (Burn)**
In the context of the Burn ML framework, a backend is the underlying computation engine that executes tensor operations. Burn supports multiple backends including CUDA (for NVIDIA GPU acceleration), ndarray (for CPU-only environments), and wgpu (for cross-platform GPU access including WebGPU and WASM). The same model code compiles against any backend without modification.

**Burn**
An open-source machine learning framework written in Rust, developed by Tracel AI. Burn provides a PyTorch-like API with a type-safe, backend-agnostic design. Models are defined using Rust's trait system and compiled to target-specific computation backends at build time.

**Catastrophic forgetting**
The tendency of a neural network to abruptly lose previously learned information when it is fine-tuned on new data. Fine-tuning updates the weights responsible for recognizing new classes, which partially overwrites the weights encoding knowledge of existing classes. Also referred to as catastrophic interference.

**Confidence threshold**
A minimum predicted probability value that a model's output must exceed before that prediction is accepted as a pseudo-label. In this project, a confidence threshold of 0.9 means only predictions where the model assigns at least 90% probability to a single class are used as training targets during the SSL pipeline.

**Edge AI**
The deployment of artificial intelligence models directly on end-user devices or local hardware, without sending data to a remote server for processing. Edge AI enables offline inference, reduces latency, eliminates cloud costs, and avoids transmitting potentially sensitive data over a network.

**Elastic Weight Consolidation (EWC)**
A regularization-based approach to incremental learning that selectively slows down the updating of weights that were important for previously learned tasks. Importance is estimated using the Fisher information matrix. EWC allows a model to learn new tasks while retaining performance on old ones.

**Incremental learning**
The process of updating a trained neural network to recognize new categories, without retraining from scratch on the full combined dataset. Incremental learning is relevant in scenarios where new classes emerge over time and full retraining is computationally prohibitive or where historical data is no longer available.

**Learning without Forgetting (LwF)**
An incremental learning technique that uses knowledge distillation to preserve a model's output on old tasks while training on new data. During incremental training, the outputs of the previous model version serve as soft targets for the existing classes, acting as a regularizer against forgetting.

**PlantVillage dataset**
A publicly available dataset of 87,000 plant leaf images, labeled across 38 disease and healthy categories covering 14 crop species. The dataset is pre-balanced and split into a training set and a validation set. It is widely used as a benchmark for plant disease classification research.

**Pseudo-labeling**
A semi-supervised learning technique in which a trained model is used to generate predicted labels for unlabeled data. Predictions that exceed a confidence threshold are treated as ground truth labels and added to the training set for subsequent retraining cycles. The quality of pseudo-labels is bounded by the accuracy of the model that generates them.

**Rehearsal**
An incremental learning technique that maintains a small memory buffer containing a selection of examples from previously learned classes. When training on new classes, examples from the buffer are included in each training batch, preventing the model from forgetting earlier classes by continuing to train on representative samples of them.

**Semi-supervised learning (SSL)**
A machine learning paradigm that trains a model using a small labeled dataset together with a much larger pool of unlabeled data. SSL reduces the dependency on expensive expert annotation while achieving accuracy levels closer to fully supervised training. Pseudo-labeling is one of the simplest and most effective SSL techniques.

**Tauri**
An open-source framework for building cross-platform desktop and mobile applications. Tauri uses a Rust core for application logic and system access, and a web-based frontend (HTML, CSS, JavaScript or a framework such as Svelte) for the user interface. Tauri applications compile to native binaries without bundling a full browser engine, resulting in significantly smaller deployment sizes compared to Electron.

**Tensor**
The fundamental data structure in machine learning frameworks, representing an n-dimensional array of numerical values. Neural network inputs, outputs, weights, and intermediate activations are all represented as tensors.

**WebAssembly (WASM)**
A binary instruction format for a stack-based virtual machine designed to enable near-native performance execution of code in web browsers and other environments. WASM allows Rust code to be compiled and run in a browser context, which is relevant for deploying the inference pipeline as a Progressive Web App.
