# Glossary

**Backend (Burn)**
In the context of the Burn ML framework, a backend is the underlying computation engine that actually executes tensor operations. Burn supports several backends, including CUDA (for NVIDIA GPU acceleration), ndarray (for CPU-only environments) and wgpu (for cross-platform GPU access, including WebGPU and WASM). The same model code compiles against any of these backends without modification.

**Burn**
An open-source machine learning framework written in Rust and developed by Tracel AI. Burn offers a PyTorch-like API together with a type-safe, backend-agnostic design. Models are defined through Rust's trait system and compiled against a target-specific computation backend at build time.

**Catastrophic forgetting**
The tendency of a neural network to lose previously learned information suddenly when it is fine-tuned on new data. Fine-tuning updates the weights that are responsible for recognising the new classes, and in doing so it partially overwrites the weights that encoded knowledge of the existing ones. The phenomenon is sometimes referred to as catastrophic interference.

**Confidence threshold**
The minimum predicted probability that a model's output has to exceed before that prediction is accepted as a pseudo-label. In this project, a confidence threshold of 0.9 means that only predictions where the model assigns at least 90% probability to a single class are used as training targets during the SSL pipeline.

**Edge AI**
The deployment of artificial intelligence models directly on end-user devices or on local hardware, without sending data to a remote server for processing. Edge AI makes offline inference possible, reduces latency, removes recurring cloud costs and avoids transmitting sensitive data over a network.

**Elastic Weight Consolidation (EWC)**
A regularization-based approach to incremental learning that selectively slows down the updating of weights that were important for previously learned tasks. Importance is estimated using the Fisher information matrix. EWC allows a model to learn new tasks while still retaining its performance on the older ones.

**Incremental learning**
The process of updating a trained neural network so that it can recognise new categories, without retraining from scratch on the full combined dataset. This matters in any scenario where new classes emerge over time and a full retrain is either too expensive to run or impossible because the original training data is no longer available.

**Learning without Forgetting (LwF)**
An incremental learning technique that relies on knowledge distillation to preserve a model's output on old tasks while it is being trained on new data. During incremental training, the outputs of the previous version of the model act as soft targets for the existing classes and therefore serve as a regulariser against forgetting.

**PlantVillage dataset**
A publicly available dataset of 87,000 plant leaf images, labeled across 38 disease and healthy categories that cover 14 crop species. The dataset is pre-balanced and comes split into a training set and a validation set. It is widely used as a benchmark for plant disease classification research.

**Pseudo-labeling**
A semi-supervised learning technique in which a trained model is used to generate predicted labels for unlabeled data. Predictions that exceed a confidence threshold are then treated as ground-truth labels and added to the training set for the next retraining cycle. The quality of the resulting pseudo-labels is bounded by the accuracy of the model that generates them.

**Rehearsal**
An incremental learning technique that keeps a small memory buffer containing a selection of examples from previously learned classes. When the model is trained on new classes, examples from the buffer are included in every training batch, which prevents the model from forgetting earlier classes because it keeps seeing representative samples of them.

**Semi-supervised learning (SSL)**
A machine learning paradigm that trains a model using a small labeled dataset together with a much larger pool of unlabeled data. SSL reduces the dependency on expensive expert annotation while still achieving accuracy levels that are close to fully supervised training. Pseudo-labeling is one of the simplest and most effective SSL techniques.

**Tauri**
An open-source framework for building cross-platform desktop and mobile applications. Tauri uses a Rust core for application logic and system access, and a web-based frontend (HTML, CSS, JavaScript, or a framework such as Svelte) for the user interface. Tauri applications compile into native binaries without bundling a full browser engine, which results in much smaller deployments than Electron.

**Tensor**
The fundamental data structure in machine learning frameworks: an n-dimensional array of numerical values. Neural network inputs, outputs, weights and intermediate activations are all represented as tensors.

**WebAssembly (WASM)**
A binary instruction format for a stack-based virtual machine designed so that code can run at near-native performance inside web browsers and similar environments. WASM makes it possible for Rust code to be compiled and executed in a browser context, which matters here because it enables the inference pipeline to be shipped as a Progressive Web App.
