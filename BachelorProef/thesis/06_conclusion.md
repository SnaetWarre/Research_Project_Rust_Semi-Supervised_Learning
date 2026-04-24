# 6. Conclusion

This thesis set out to answer the question: **How can a semi-supervised neural network be efficiently implemented in Rust for the automatic labeling of partially labeled datasets on an edge device?**

The research shows that this is not only feasible, but that Rust, and specifically the Burn framework, offers tangible advantages over the established Python/PyTorch stack for edge deployment scenarios.

## 6.1 Answering the Research Question

A semi-supervised neural network can be implemented efficiently in Rust for edge deployment through the combination of three components: a lightweight custom CNN built with the Burn framework, an iterative pseudo-labeling pipeline with confidence-based filtering, and a cross-platform deployment strategy built around Tauri.

**The implementation is efficient** along several axes. The compiled binary is roughly 26 MB and is the only artefact needed on the target device, whereas a Python/PyTorch deployment requires a multi-gigabyte environment (interpreter, wheels, supporting packages), even though the model weights themselves are comparable in size between both stacks. Inference latency is 0.39 ms on a desktop GPU (2,579 FPS) and 80 ms on an iPhone 12 through Tauri, both well below the real-time threshold. The model weights take up 5.7 MB, which makes distribution feasible even over low-bandwidth channels. A cold start takes less than 100 ms, compared to roughly 3 seconds for a PyTorch application.

**The automatic labeling is effective.** Starting from only 20% labeled data, the pseudo-labeling pipeline generates labels for the remaining unlabeled pool with a precision above 95% (at a 0.9 confidence threshold). Validation accuracy improves from roughly 70 to 75% (the supervised baseline) to roughly 78 to 85%, which is a meaningful gain achieved without any additional human annotation.

**The system runs on edge devices** without any internet connection. All inference happens locally: no cloud API calls, no data leaves the device and no network connection is required after installation. This was validated on four hardware configurations, with the iPhone 12 (via Tauri) and the CPU-only backend confirming that the approach is viable on the kind of consumer hardware that farmers already own.

## 6.2 Key Findings

The three controlled experiments provide quantitative answers to the questions that matter most for deployment:

1. **Label efficiency.** A minimum of 100 labeled images per class is needed to reach production-viable accuracy (above 80%). Below 25 images per class, the model is effectively unusable (34 to 37% accuracy on 38 classes). These numbers give a concrete labeling budget for any new deployment.

2. **Class scaling.** Adding a new class to a 30-class model causes 6× more catastrophic forgetting than adding one to a 5-class model (1.26% versus 0.21% forgetting). The model becomes measurably more biased towards existing classes as the base grows. That finding has direct consequences for production systems that have to evolve over time: incremental learning methods (rehearsal, EWC, LwF) are not optional luxuries but practical necessities.

3. **New class position.** Learning a new class as the 31st in a large taxonomy is substantially harder than learning it as the 6th in a small one. At 50 labeled samples, the 6th class reaches 84.27% accuracy while the 31st class only reaches 25.62%. This quantifies the additional labeling effort required when mature production models are extended.

## 6.3 The BYOD Pivot

One of the more unexpected, and significant, findings was the performance comparison between dedicated edge hardware and consumer devices. The Jetson Orin Nano (€350) managed 120 ms inference, which is slower than the iPhone 12 (80 ms, €0 additional cost). That data-driven pivot from dedicated hardware to a Bring Your Own Device strategy removes the largest capital expenditure barrier to deployment. Combined with the deployment size of roughly 26 MB, this makes the system distributable to any farmer who owns a smartphone.

## 6.4 Reflection and External Perspectives

The external reflections in Chapter 4 will provide additional validation of these findings against industry experience. The self-reflection identifies the main limitations: the absence of field validation on real-world agricultural images, the bounded effectiveness of pseudo-labeling when the initial model has systematic errors, and the relative immaturity of the Burn ecosystem compared to PyTorch.

The broader impact analysis highlights the potential of this approach for food security in regions with limited internet infrastructure. The offline-first architecture not only addresses the connectivity constraint, but also offers a privacy advantage: because no agricultural data is transmitted to an external server, the risk of data exfiltration is eliminated.

## 6.5 Recommendations for Future Work

Three directions for future research stand out as the most promising:

1. **Field validation** using real-world agricultural images captured under diverse conditions (lighting, angle, camera quality, multiple diseases per leaf). This is the critical missing step between the controlled experiments in this thesis and actual deployment.

2. **Active learning integration**, where the system identifies uncertain predictions and requests targeted human annotation rather than discarding every low-confidence sample. This could reduce the labeled data requirement even further.

3. **Federated learning across deployed devices**, which would allow multiple installations to improve the shared model without transmitting raw image data. This would preserve privacy while still enabling continuous improvement.

## 6.6 Final Assessment

The central claim of this thesis is that Rust and the Burn framework provide a production-viable path for deploying semi-supervised machine learning on edge devices. The experimental evidence supports that claim: the system is smaller, faster and more portable than Python-based alternatives, and the SSL pipeline makes effective use of unlabeled data to compensate for the high cost of expert annotation. For the specific use case of plant disease detection in offline agricultural settings, this approach represents a practical and deployable solution.
