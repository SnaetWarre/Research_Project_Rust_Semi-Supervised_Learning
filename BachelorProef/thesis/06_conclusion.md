# 6. Conclusion

This thesis set out to answer the question: **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

The short answer is that it can be done by combining a lightweight custom CNN built with the Burn framework, an iterative pseudo-labeling pipeline with confidence-based filtering, and a cross-platform deployment strategy built around Tauri. The longer answer involves the specific trade-offs, numbers and lessons learned from this project.

## 6.1 What Was Built

The implementation is efficient in ways relevant to edge deployment. The trained model weights are approximately 916 KB in Burn's native format. The compiled binary is approximately 26 MB, and is the only file that must be present on the target device. A comparable Python/PyTorch deployment requires several gigabytes of interpreter, libraries and supporting packages, even though the model weights themselves are approximately the same size in both stacks. The Burn binary starts in under 100 ms, compared with approximately 3 seconds for a PyTorch application. On a laptop GPU, an NVIDIA RTX 3060, the saved SSL checkpoint reaches an inference latency of 0.42 ms, corresponding to approximately 2,406 FPS. On an iPhone 12 through Tauri, it reached approximately 80 ms per inference in local testing, which is fast enough for a camera-based classification app.

The automatic labeling pipeline was implemented as an offline pseudo-labeling workflow. Starting from only 20% labeled data, it generates labels for the remaining unlabeled pool. On the held-out test split of 8,786 images, the saved supervised baseline reached 86.06% top-1 accuracy and 86.08% macro F1. The saved SSL checkpoint reached 94.90% top-1 accuracy and 94.74% macro F1. This represents an improvement of 8.84 percentage points in accuracy and 8.66 percentage points in macro F1 without extra human annotation, although these figures come from one saved split and should not be treated as guaranteed performance across datasets. The system also runs fully offline, with no cloud API calls, no data leaving the device and no network connection required after installation.

## 6.2 What the Experiments Show

The three controlled experiments give the quantitative answers needed for deployment decisions. The label efficiency curve shows that 100 labeled images per class is the minimum for useful accuracy, exceeding 80%, with diminishing returns beyond 200. The class scaling experiment shows that adding a new class to a 30-class model causes six times more catastrophic forgetting than adding one to a 5-class model: 1.26 percentage points compared with 0.21 percentage points. This is a clear signal that incremental learning methods such as rehearsal or EWC become important once the model reaches production scale. The new class position experiment shows the extra effort needed. At 50 labeled samples, learning a new class as the 6th class in a small taxonomy reaches 84% accuracy, while learning it as the 31st class in a large one only reaches 26%. These numbers are useful planning inputs for anyone extending a deployed model.

## 6.3 Hardware and Deployment Strategy

One unexpected finding was the hardware comparison. The Jetson Orin Nano, a dedicated edge AI device at around €350, reached 120 ms inference. The iPhone 12 reached 80 ms without extra hardware cost. This result pushed the project away from dedicated edge hardware and towards a Bring Your Own Device strategy, removing the largest capital cost for deployment.

Three deployment paths were prepared: a desktop Tauri application, an iOS build through Tauri Mobile, and a browser-based Progressive Web App using ONNX Runtime Web. The desktop and iOS paths were tested end to end. The PWA path was prepared but not fully validated across all browsers.

## 6.4 Limitations and Honest Assessment

The reflection in Chapter 4 also points out the main limitations. The system has not yet been validated on real agricultural images from the field. Pseudo-label quality is still limited by the quality of the first model. The Burn ecosystem is less mature than PyTorch. The external expert feedback supports the main direction of the project, but also clarifies the next steps: compare the custom CNN with established mobile architectures, analyse pseudo-label quality per class, test multiple input resolutions and validate the model under different acquisition conditions.

The most important next step is field validation: tests on real images, captured by real users, under uncontrolled conditions. The PlantVillage results are encouraging, but controlled benchmarks and real-world performance are not the same thing. Without that step, the thesis answers the technical research question without proving that the solution works in practice for farmers.

## 6.5 Answer to the Research Question

The central claim of this thesis is that Rust and the Burn framework can provide a viable path for deploying semi-supervised machine learning on edge devices. The experimental evidence supports that claim for the tested conditions. The system is smaller, faster and more portable than Python-based alternatives for the same model, and the SSL pipeline uses unlabeled data to reduce the need for expensive expert annotation. For plant disease detection in offline settings, this approach is technically feasible and deployable, as long as the remaining gaps in field validation, user testing and cross-dataset evaluation are filled.
