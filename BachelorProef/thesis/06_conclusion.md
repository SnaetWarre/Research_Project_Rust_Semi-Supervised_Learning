# 6. Conclusion

This thesis set out to answer the question: **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

The research demonstrates that this is not only feasible but that Rust, specifically the Burn framework, offers tangible advantages over the established Python/PyTorch stack for edge deployment scenarios.

## 6.1 Answering the Research Question

A semi-supervised neural network can be efficiently implemented in Rust for edge deployment through the combination of three components: a lightweight custom CNN built with the Burn framework, an iterative pseudo-labeling pipeline with confidence-based filtering, and a cross-platform deployment strategy using Tauri.

**The implementation is efficient** along multiple axes. The compiled binary is ~26 MB and is the only artefact needed on the target device; a Python/PyTorch deployment requires a multi-gigabyte environment (interpreter, wheels, supporting packages) even though the model weights themselves are comparable in size between both stacks. Inference latency is 0.39 ms on a desktop GPU (2,579 FPS) and 80 ms on an iPhone 12 via Tauri, both well below the real-time threshold. The model weights occupy 5.7 MB, making distribution feasible even over low-bandwidth channels. Cold start is under 100 ms, compared to approximately 3 seconds for a PyTorch application.

**The automatic labeling is effective.** Starting from only 20% labeled data, the pseudo-labeling pipeline generates labels for the remaining unlabeled pool with a precision exceeding 95% (at a 0.9 confidence threshold). This improves validation accuracy from approximately 70–75% (supervised baseline) to approximately 78–85%, a meaningful gain achieved without any additional human annotation.

**The system runs on edge devices** without internet connectivity. All inference happens locally: no cloud API calls, no data leaves the device, and no network connection is required after installation. This was validated on four hardware configurations, with the iPhone 12 (via Tauri) and the CPU-only backend confirming viability on consumer hardware that farmers already own.

## 6.2 Key Findings

The three controlled experiments provide quantitative answers to deployment-critical questions:

1. **Label efficiency.** A minimum of 100 labeled images per class is needed for production-viable accuracy (>80%). Below 25 images per class, the model is effectively unusable (34–37% accuracy on 38 classes). These numbers establish a concrete labeling budget for new deployments.

2. **Class scaling.** Adding a new class to a 30-class model causes 6× more catastrophic forgetting than adding one to a 5-class model (1.26% vs. 0.21% forgetting). The model becomes measurably more biased toward existing classes as the base grows. This finding has direct implications for production systems that must evolve over time: incremental learning methods (rehearsal, EWC, LwF) are not optional luxuries but practical necessities.

3. **New class position.** Learning a new class as the 31st in a large taxonomy is substantially harder than learning it as the 6th in a small one. At 50 labeled samples, the 6th class reaches 84.27% accuracy while the 31st class reaches only 25.62%. This quantifies the additional labeling effort required when extending mature production models.

## 6.3 The BYOD Pivot

An unexpected but significant finding was the performance comparison between dedicated edge hardware and consumer devices. The Jetson Orin Nano (€350) achieved 120 ms inference, which is slower than the iPhone 12 (80 ms, €0 additional cost). This data-driven pivot from dedicated hardware to a Bring Your Own Device strategy eliminates the largest capital expenditure barrier to deployment. Combined with the ~26 MB deployment size, this makes the system distributable to any farmer with a smartphone.

## 6.4 Reflection and External Perspectives

The external reflections (Chapter 4) will provide additional validation of these findings against industry experience. The self-reflection identifies key limitations: the absence of field validation on real-world agricultural images, the bounded effectiveness of pseudo-labeling when the initial model has systematic errors, and the relative immaturity of the Burn ecosystem compared to PyTorch.

The broader impact analysis highlights the potential of this approach for food security in regions with limited internet infrastructure. The offline-first architecture not only addresses the connectivity constraint but also provides a privacy advantage, as no agricultural data is transmitted to external servers, eliminating data exfiltration risks.

## 6.5 Recommendations for Future Work

Three directions for future research are most promising:

1. **Field validation** with real-world agricultural images captured under diverse conditions (lighting, angle, camera quality, multiple diseases per leaf). This is the critical missing step between the controlled experiments in this thesis and actual deployment.

2. **Active learning integration**, where the system identifies uncertain predictions and requests targeted human annotation rather than discarding all low-confidence samples. This could further reduce the labeled data requirement.

3. **Federated learning across deployed devices**, allowing multiple installations to improve the shared model without transmitting raw image data, preserving privacy while enabling continuous improvement.

## 6.6 Final Assessment

The central claim of this thesis is that Rust and the Burn framework provide a production-viable path for deploying semi-supervised machine learning on edge devices. The experimental evidence supports this claim: the system is smaller, faster, and more portable than Python-based alternatives, while the SSL pipeline effectively leverages unlabeled data to compensate for the high cost of expert annotation. For the specific use case of plant disease detection in offline agricultural settings, this approach represents a practical and deployable solution.
