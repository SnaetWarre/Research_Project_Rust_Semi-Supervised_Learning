# 4. Reflection

This chapter presents a critical evaluation of the research results. It is organised into three parts: an assessment of the external feedback that was collected, a reflection on the project's strengths and limitations, and a discussion of the broader implications for deployment.

## 4.1 External Feedback

Two researchers from the 2AI-IPCA research lab, Helena Torres and Pedro Morais, both working in image processing and deep learning, were consulted. Detailed interview questions were sent by email. Pedro Morais answered five of the twenty questions, and Helena Torres answered all twenty.

Their feedback is referenced throughout this chapter. Pedro Morais's full answers are provided in Appendix B, and Helena Torres's in Appendix C.

The most valuable aspect of this feedback was that it did not reject the project's direction, but it did sharpen the identification of weak points. Both experts recognised the value of a simple pseudo-labeling setup for edge deployment, particularly when the goal is a small and portable system rather than a large research stack. At the same time, both shifted the focus from whether the pipeline executes correctly to whether the generated labels are sufficiently reliable. This distinction is significant: a semi-supervised system can appear successful by accepting large numbers of pseudo-labels, yet remain unsafe if it accepts incorrect samples with high confidence.

Helena Torres raised an important consideration regarding dataset size. In this thesis, PlantVillage is treated as a dataset where labels are expensive relative to unlabeled samples. While this holds for the agricultural annotation setting, Torres noted that 17,000 labeled images would already constitute a large dataset in many medical imaging contexts. In such domains, the primary concern is often not merely the number of labels, but also their quality, consistency and inter-observer variability. This nuance affects how the results should be interpreted. The project demonstrates that Rust and Burn can support an SSL workflow, but it does not establish that the same thresholds and data splits would transfer directly to domains where labels are noisier or where experts disagree.

The feedback also clarified the engineering trade-offs involved. Pedro suggested EfficientNet, while Torres mentioned MobileNetV3 and EfficientNet-Lite as natural mobile baselines. A custom CNN was chosen for this project because it offered greater experimental control, was sufficiently small for Burn and fitted within the available video memory. Although this was a defensible thesis decision, it also means that the model comparison remains incomplete. A more comprehensive evaluation would compare the custom CNN against at least one established mobile architecture under identical Rust or ONNX deployment constraints.

## 4.2 Self-Reflection

### 4.2.1 Strengths

**Deployment size and portability.** The compiled binary of approximately 26 MB represents a meaningful improvement over a Python/PyTorch deployment. Both stacks require gigabytes of tooling during development: Rust's `target/` directory is approximately 2.1 GB, which is comparable to a PyTorch virtual environment. The decisive difference is that Rust's compilation step reduces the entire environment to a single portable binary. A Python deployment, by contrast, must carry its interpreter and library tree to the target device. This distinction determines which distribution channels are viable for edge deployment. A file that can be transferred over Bluetooth or a USB drive differs fundamentally from a system that first requires a multi-gigabyte environment.

**The BYOD pivot.** The benchmark results (Table 3.7) provided a clear, data-driven basis for moving away from dedicated edge hardware. The Jetson Orin Nano, priced at €350, proved slower in this test than a consumer smartphone that many users already own. This pivot removed the single largest capital cost associated with deployment.

**Experimental rigour.** The three controlled experiments provide quantitative answers to questions that are often discussed only qualitatively in the literature: how much labeled data is required, how forgetting scales with taxonomy size, and how the position of a new class affects the difficulty of learning it.

### 4.2.2 Weaknesses and Limitations

**External expert feedback.** Pedro Morais reviewed the approach and confirmed that the 90% confidence threshold is a sensible starting point for controlled imaging conditions. He recommended retaining a fixed threshold for this application, which the current pipeline does. Torres agreed that pseudo-labeling is appropriate when deployment simplicity is prioritised, but cautioned that the initial model should be examined carefully at that 90% threshold. She specifically highlighted the need to check for systematic errors, class-specific bias and confusion between visually similar classes. Rather than relying on a single global threshold, she suggested investigating per-class thresholds or uncertainty estimation.

Both experts also commented on the model architecture and preprocessing. Pedro suggested evaluating EfficientNet, and Torres recommended comparing the custom CNN with MobileNetV3 or EfficientNet-Lite. This confirms that the custom architecture is a valid choice for an edge-focused prototype, but it is probably not the strongest mobile baseline available. Regarding augmentation, Pedro cautioned that contrast and brightness manipulations can reduce stability. Torres offered a broader warning: intensity and spatial augmentations are useful when labeled data is scarce, but deformable transformations should be used with care in medical imaging because they can distort clinically meaningful structures. For this project, controlled augmentation ablations are therefore needed, rather than assuming that every augmentation automatically improves robustness.

**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is constrained by the accuracy of the model trained on the 20% labeled subset. If the initial model systematically misclassifies certain classes, those errors propagate through the pseudo-labeling cycle. Techniques such as co-training, in which two models examine different views of the data, could mitigate this risk, but they were not implemented because of video memory constraints on edge devices.

This is also where the fixed confidence threshold becomes a limitation. The 90% threshold is straightforward to explain and implement, which is valuable for an edge prototype. However, confidence is not equivalent to correctness. A model can be overconfident on visually similar plant diseases, on images with unusual lighting, or on classes that were under-represented during training. The thesis reports global accuracy, but a more robust SSL system would also examine per-class precision, recall, F1-score and confusion matrices before trusting accepted pseudo-labels. This would reveal whether the model performs strongly only on common or visually distinct classes, or whether it behaves consistently across the full taxonomy.

Another missing element is calibration analysis. If a prediction score of 90% does not correspond to approximately 90% correctness, the threshold has a different meaning than assumed. Calibration plots or expected calibration error could help determine whether a single global threshold is appropriate. If certain classes are consistently overconfident, per-class thresholds would be more defensible. This was not implemented, but Torres's feedback makes it clear that calibration analysis would be one of the first improvements before applying this approach in a higher-risk setting.

**No field validation.** All experiments were conducted on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important respects: varying lighting, background vegetation, leaf angle, camera quality and the presence of multiple diseases on the same leaf. The model's performance on field-captured images is unknown and is likely lower than the figures reported here.

The absence of field validation also affects the preprocessing pipeline. The iOS channel-ordering bug demonstrated that even a small difference between BGRA and RGB can silently break predictions while the application still appears to function. For future deployments, preprocessing should be validated with fixed reference images on every target platform. The same image should produce identical tensor values and predictions on desktop, mobile and any exported ONNX path. Although this appears to be a minor engineering detail, it is exactly the kind of detail that determines whether a portable machine learning system is reliable.

Input resolution is another unresolved design choice. The chosen image size was a practical compromise between accuracy, memory usage and inference speed. Torres suggested evaluating multiple input resolutions, which would make the trade-off more explicit. Higher resolution could preserve small disease symptoms, while lower resolution could improve speed and memory usage on phones. Without that comparison, the chosen resolution is defensible but not fully justified.

**Burn ecosystem maturity.** Burn proved capable for this project, but the framework remains under active development. Its documentation is less extensive than PyTorch's, and some features (such as mixed-precision training and distributed training) are not yet available. Teams considering Burn for production should factor in the cost of working with a less mature ecosystem.

The interviews also placed this choice in a broader software context. Rust and Burn are attractive for deployment because they produce compact binaries and avoid a large Python runtime on the target device. For research collaboration, however, PyTorch remains more accessible because a larger research community is familiar with it, more examples exist, and more pretrained models are available. A pragmatic production workflow could therefore be hybrid: train and compare models in PyTorch, export the selected model to ONNX, and run the deployed inference path in Rust. This thesis focused on a full Rust implementation to test feasibility, but this does not imply that every future project should avoid Python entirely.

**Single dataset evaluation.** The experiments were conducted exclusively on PlantVillage. Generalisation to other agricultural datasets (different crops, different disease profiles, different imaging conditions) has not been validated.

This limitation matters because PlantVillage is relatively clean. The leaves are centered, the backgrounds are controlled and the class definitions are stable. Real deployments would require tests across different cameras, farms, lighting conditions and disease stages. A model that performs well on PlantVillage might still fail on early symptoms, damaged leaves, mixed infections or images where the leaf occupies only part of the frame. Segmentation-assisted classification could help by forcing the model to focus on the relevant leaf or lesion area before classification.

## 4.3 Broader Implications

### 4.3.1 Practical Barriers to Deployment

Building a working prototype and deploying it to end users are distinct challenges. Several barriers emerged during development that stand between a functional system and a practical tool.

The most significant barrier is trust. If the model predicts "bacterial spot" with high confidence and a farmer treats accordingly, but the actual condition is different, the tool has caused harm rather than benefit. This is why the graphical user interface displays a confidence bar rather than a single prediction. Even so, non-technical users may not know how to interpret a 72% confidence score. The interface must communicate uncertainty honestly without becoming confusing, which constitutes a user experience problem as much as a machine learning problem.

Device diversity is another concern. The BYOD strategy requires the system to function across whatever phone or laptop a user happens to own. The cross-platform benchmarks (Table 3.7) demonstrate that the system can operate across hardware configurations, but the preprocessing bug on iOS (Section 3.6.3), where BGRA versus RGB channel ordering silently produced incorrect classifications, is representative of the kind of problem that only appears on real devices. Additional bugs of this nature are likely to emerge on untested hardware.

Finally, there is the update problem. The initial installation is small enough (26 MB) to distribute offline, but distributing model improvements or new classes presents challenges. Agricultural extension workers or local community centres could serve as distribution points, but this requires coordination beyond software engineering.

### 4.3.2 Economic Case

The measurements from this project support a straightforward economic argument.

First, the SSL pipeline reduces the labeling requirement from 100% of the dataset to approximately 20%, which translates directly into annotation budget savings. For an 87,000-image dataset, this represents the difference between a full expert-annotation budget and a substantially smaller one, which matters for research groups and agricultural extensions with limited funding.

Second, the BYOD pivot eliminates hardware costs entirely. The Jetson Orin Nano benchmark (Table 3.7) demonstrated that a €350 dedicated device was slower than a consumer phone, so there is no economic justification for purchasing one.

Third, once installed, the marginal cost of each classification is zero: no cloud API calls, no per-prediction charges and no bandwidth costs.

These savings are meaningful only if the model is sufficiently accurate to be useful. The held-out test evaluation suggests that it is for the PlantVillage benchmark (94.90% top-1 accuracy with the saved SSL checkpoint), but this has only been validated on PlantVillage imagery, not on field photographs.

### 4.3.3 Privacy and Local Operation

The offline-first architecture offers a practical advantage beyond connectivity. Because the model runs entirely on the user's device, no image data is transmitted to an external server. This removes server-side data-processing risks and reduces the compliance surface for privacy regulations such as the General Data Protection Regulation (GDPR), because the developer does not need to store or process user images on a backend. This does not eliminate all legal considerations: if the device is managed by an organisation, or if inference logs are collected, GDPR may still apply. Device-level security measures such as local encryption and access control therefore remain important.

Whether this kind of tool would reduce crop losses cannot be answered from a computer science thesis alone. What can be stated is that the technical barriers—deployment size, connectivity requirements and inference speed—are no longer the principal bottleneck. The remaining barriers concern trust, usability and distribution. Addressing those requires field trials and user research rather than additional engineering alone.

### 4.3.4 Future Research Directions

Several directions for future work merit attention:

1. **Field validation:** deploying the system with actual users and measuring classification accuracy on real-world images, under varying lighting, backgrounds and camera quality. The PlantVillage results are promising, but controlled benchmarks and real-world performance are not equivalent.
2. **Active learning:** instead of discarding every low-confidence sample, the system could flag uncertain predictions and request human input. This would transform the SSL loop into a targeted labeling tool.
3. **Federated learning:** multiple deployed devices could share model updates without sharing raw images, which would allow the model to improve over time while keeping data local.
4. **Multi-disease detection:** extending the model to handle images in which several diseases appear on the same leaf simultaneously.
5. **Segmentation-assisted classification:** using leaf or symptom segmentation before classification to improve explainability and reduce the effect of irrelevant backgrounds.
6. **Burn ecosystem contributions:** contributing missing features (mixed-precision training, model quantisation) back to the open-source framework.
