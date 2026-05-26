# Glossary

**Backend (Burn)**
In Burn, a backend is the engine that actually runs tensor operations. Burn supports several: CUDA for NVIDIA GPUs, ndarray for CPU-only work, and wgpu for cross-platform GPU access including WebGPU and WASM. The same model code compiles for any of these without modification.

**Burn**
An open-source machine learning framework written in Rust and developed by Tracel AI. Burn offers a PyTorch-like API with a type-safe, backend-agnostic design. Models are defined through Rust's trait system and compiled against a target-specific backend at build time.

**Catastrophic forgetting**
The tendency of a neural network to suddenly forget what it already knew when it is fine-tuned on new data. Updating weights for the new classes partially overwrites the weights that encoded knowledge of the old ones. This is also called catastrophic interference.

**Confidence threshold**
The minimum predicted probability a model output has to reach before it is accepted as a pseudo-label. In this project, a threshold of 0.9 means only predictions where the model assigns at least 90% probability to a single class are used as training targets in the SSL pipeline.

**Edge AI**
Running AI models directly on end-user devices or local hardware, without sending data to a remote server. Edge AI makes offline inference possible, reduces latency, removes recurring cloud costs and avoids transmitting sensitive data over a network.

**Elastic Weight Consolidation (EWC)**
A regularization approach to incremental learning that selectively slows down updates to weights that were important for earlier tasks. Importance is estimated using the Fisher information matrix. EWC lets a model learn new tasks without losing performance on old ones.

**Incremental learning**
Updating a trained neural network so it can recognise new categories without retraining from scratch on the full combined dataset. This matters when new classes emerge over time and a full retrain is too expensive or impossible because the original data is no longer available.

**Learning without Forgetting (LwF)**
An incremental learning technique that uses knowledge distillation to preserve a model's output on old tasks while it trains on new data. The outputs of the previous model version act as soft targets for existing classes, which serves as a regulariser against forgetting.

**PlantVillage dataset**
A publicly available dataset of roughly 87,000 plant leaf images, labeled across 38 categories that include both diseases and healthy variants, covering 14 crop species. The dataset is pre-balanced and comes split into a training set and a validation set. It is widely used as a benchmark for plant disease classification research.

**Pseudo-labeling**
A semi-supervised learning technique where a trained model generates predicted labels for unlabeled data. Predictions that exceed a confidence threshold are treated as ground-truth labels and added to the training set for the next retraining cycle. The quality of the resulting pseudo-labels is bounded by the accuracy of the model that generates them.

**Rehearsal**
An incremental learning technique that keeps a small memory buffer with selected examples from previously learned classes. When the model trains on new classes, buffer examples are included in every batch, which prevents forgetting because the model keeps seeing representative samples of earlier classes.

**Semi-supervised learning (SSL)**
A machine learning paradigm that trains a model on a small labeled dataset together with a much larger pool of unlabeled data. SSL reduces the need for expensive expert annotation while still achieving accuracy close to fully supervised training. Pseudo-labeling is one of the simplest and most effective SSL techniques.

**Tauri**
An open-source framework for building cross-platform desktop and mobile applications. Tauri uses a Rust core for application logic and system access, and a web-based frontend for the user interface. Tauri apps compile into native binaries without bundling a full browser engine, which makes them much smaller than Electron equivalents.

**Tensor**
The fundamental data structure in machine learning frameworks: an n-dimensional array of numerical values. Neural network inputs, outputs, weights and intermediate activations are all represented as tensors.

**WebAssembly (WASM)**
A binary instruction format for a stack-based virtual machine that runs at near-native performance inside web browsers. WASM lets Rust code execute in a browser context, which enables the inference pipeline to be shipped as a Progressive Web App.
