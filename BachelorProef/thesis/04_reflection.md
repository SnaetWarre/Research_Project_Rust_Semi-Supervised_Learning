# 4. Reflection

This chapter gives a critical evaluation of the research results. It is divided into three parts: an honest assessment of the external feedback that was sought, a self-reflection on what went well and what did not, and a discussion of the broader implications for deployment.

## 4.1 External Feedback

I contacted two researchers from the 2AI-IPCA research lab, Helena Torres and Pedro Morais, who work in image processing and deep learning. I sent them the detailed interview questions by email. Pedro Morais answered five of the twenty questions, and Helena Torres answered all twenty.

Their feedback is used throughout this chapter. Pedro Morais's full answers are in Appendix B, and Helena Torres's are in Appendix C.

The most useful part of this feedback was that it did not reject the direction of the project, but it did make the weak points sharper. Both experts saw value in a simple pseudo-labeling setup for edge deployment, especially when the goal is a small and portable system instead of a large research stack. At the same time, they both shifted the focus from "does the pipeline run?" to "how do we know the labels are reliable enough?" That distinction is important. A semi-supervised system can look successful if it accepts many pseudo-labels, but it can still be unsafe if it accepts the wrong samples with high confidence.

Helena Torres also made an important point about dataset size. In this thesis, PlantVillage is treated as a dataset where labels are expensive compared with unlabeled samples. That is true in the agricultural annotation setting, but Helena noted that 17,000 labeled images would already be a large dataset in many medical imaging tasks. In those domains, the main issue is often not only the number of labels, but their quality, consistency and inter-observer variability. This changes how the results should be interpreted. The project shows that Rust and Burn can support an SSL workflow, but it does not prove that the same thresholds and data splits would transfer directly to domains where labels are noisier or where experts disagree.

The feedback also made the engineering trade-off clearer. Pedro suggested EfficientNet, while Helena mentioned MobileNetV3 and EfficientNet-Lite as natural mobile baselines. I chose a custom CNN because it was easier to control, small enough for Burn, and suitable for the available VRAM. That was a reasonable thesis decision, but it also means that the model comparison is incomplete. A stronger final evaluation would compare the custom CNN against at least one established mobile architecture under the same Rust or ONNX deployment constraints.

## 4.2 Self-Reflection

### 4.2.1 What Went Well

**Deployment size and portability.** The binary of roughly 26 MB is a real improvement compared with a Python/PyTorch deployment. Both stacks need gigabytes of tooling during development. Rust's `target/` directory is around 2.1 GB, which is comparable to a PyTorch virtual environment. The difference is that Rust's compilation reduces everything to a single portable binary. A Python deployment still has to bring its interpreter and library tree to the target device. That changes which distribution channels are realistic for edge deployment. A file that fits on a Bluetooth transfer or a USB stick is very different from a system that first needs a multi-gigabyte environment.

**The BYOD pivot.** The benchmark results (Table 3.7) provided a clear, data-driven reason to move away from dedicated edge hardware. The Jetson Orin Nano, at €350, turned out to be slower than a phone that many people already own. That pivot removed the single largest cost barrier to deployment.

**Experimental rigour.** The three controlled experiments give quantitative answers to questions that are often only discussed qualitatively in the literature: how much labeled data is actually enough, how forgetting scales with model size, and how the position of a new class affects the difficulty of learning it.

### 4.2.2 Weaknesses and Limitations

**External expert feedback.** Pedro Morais from 2AI-IPCA reviewed the approach and confirmed that the 90% confidence threshold is a sensible starting point for controlled imaging conditions. He recommended keeping it fixed for this application, which is what the current pipeline does. Helena Torres agreed that pseudo-labeling makes sense when you want to keep deployment simple, but she warned that the initial model should be checked carefully at that 90% threshold. She specifically mentioned looking for systematic errors, class-specific bias and confusion between visually similar classes. Instead of relying on a single global threshold, she suggested trying per-class thresholds or uncertainty estimation.

Both experts also commented on the model and the preprocessing. Pedro suggested evaluating EfficientNet, and Helena recommended comparing the custom CNN with MobileNetV3 or EfficientNet-Lite. This confirms that the custom architecture is a valid choice for an edge-focused prototype, but it is probably not the strongest mobile baseline available. On augmentation, Pedro cautioned that contrast and brightness manipulations can hurt stability. Helena gave a broader warning: intensity and spatial augmentations are useful when labeled data is scarce, but deformable transformations should be used with care in medical imaging because they can distort clinically meaningful structures. For this project, that means controlled augmentation ablations are needed, rather than assuming that every augmentation automatically improves robustness.

**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is limited by the accuracy of the model that was trained on the 20% labeled subset. If the initial model systematically misclasses certain classes, those errors can continue through the pseudo-labeling cycle. Techniques such as co-training, where two models look at different views of the data, could reduce this risk, but they were not implemented because of the VRAM constraints on edge devices.

This is also where the fixed confidence threshold becomes a limitation. The 90% threshold is easy to explain and easy to implement, which is valuable for an edge prototype. However, confidence is not the same as correctness. A model can be overconfident on visually similar plant diseases, on images with unusual lighting, or on classes that were under-represented during training. The thesis reports global accuracy, but a safer SSL system would also inspect per-class precision, recall, F1-score and confusion matrices before trusting accepted pseudo-labels. That would show whether the model is only strong on common or visually easy classes, or whether it behaves consistently across the full taxonomy.

Another missing step is calibration analysis. If a prediction score of 90% does not correspond to roughly 90% correctness, the threshold has a different meaning than expected. Calibration plots or expected calibration error could help determine whether one global threshold is acceptable. If some classes are consistently overconfident, per-class thresholds would be more defensible. This was not implemented, but Helena's feedback makes it clear that it would be one of the first improvements before using this approach in a higher-risk setting.

**No field validation.** All experiments were carried out on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important ways: varying lighting, background vegetation, leaf angle, camera quality and the presence of several diseases on the same leaf. The model's performance on field-captured images is unknown and is very likely lower than the numbers reported here.

The lack of field validation also affects the preprocessing pipeline. The iOS channel-ordering bug showed that even a small difference between BGRA and RGB can silently break predictions while the application still appears to work. For future deployments, preprocessing should be validated with fixed reference images on every target platform. The same image should produce the same tensor values and the same prediction on desktop, mobile and any exported ONNX path. This sounds like a small engineering detail, but it is exactly the kind of detail that determines whether a portable ML system is actually reliable.

Input resolution is another unresolved part of the design. The chosen image size was a practical compromise between accuracy, memory usage and inference speed. Helena suggested testing multiple input resolutions, which would make the trade-off more explicit. Higher resolution could preserve small disease symptoms, while lower resolution could improve speed and memory usage on phones. Without that comparison, the chosen resolution is defensible, but not fully justified.

**Burn ecosystem maturity.** Burn proved capable for this project, but the framework is still under active development. Its documentation is sparse compared to PyTorch, and some features (for example mixed-precision training and distributed training) are not yet available. Teams considering Burn for production should factor in the cost of working with a less mature ecosystem.

The interviews also put this choice into a broader software context. Rust and Burn are attractive for deployment because they produce compact binaries and avoid a large Python runtime on the target device. For research collaboration, however, PyTorch is still easier because more researchers know it, more examples exist, and more pretrained models are available. A pragmatic production workflow could therefore be hybrid: train and compare models in PyTorch, export the selected model to ONNX, and then run the deployed inference path in Rust. This thesis focused on a full Rust implementation to test feasibility, but that does not mean every future project should avoid Python completely.

**Single dataset evaluation.** The experiments were carried out exclusively on PlantVillage. Generalisation to other agricultural datasets (different crops, different disease profiles, different imaging conditions) has not been validated.

This limitation matters because PlantVillage is relatively clean. The leaves are centered, the backgrounds are controlled and the class definitions are stable. Real deployments would need tests across different cameras, farms, lighting conditions and disease stages. A model that works on PlantVillage might still fail on early symptoms, damaged leaves, mixed infections or images where the leaf is only part of the frame. Segmentation-assisted classification could help by forcing the model to focus on the relevant leaf or lesion area before classification.

## 4.3 Broader Implications

### 4.3.1 Practical Barriers to Deployment

Building the system is one thing. Getting it into the hands of someone who would actually use it is another. During development, a few things became clear about what stands between a working prototype and a useful tool:

The biggest issue is trust. If the model confidently says "bacterial spot" and the farmer treats for that, but it turns out to be something else, the tool has done more harm than good. That is why the GUI shows a confidence bar instead of only a single answer. Even then, a non-technical user might not know how to interpret a 72% confidence score. The interface has to be honest about uncertainty without becoming confusing, and that is a UX problem as much as an ML problem.

Device diversity is another concern. The BYOD strategy means the system has to work on whatever phone or laptop someone happens to own. The cross-platform benchmarks (Table 3.7) show that it can work across hardware, but the preprocessing bug on iOS (Section 3.6.3), where BGRA versus RGB channel ordering silently produced wrong classifications, is exactly the kind of problem that only appears on real devices. There will probably be more bugs like that on devices I have not tested.

Finally, there is the update problem. The initial installation is small enough (26 MB) to distribute offline, but what happens when the model improves or a new class is added? Agricultural extension workers or local community centres could serve as distribution points, but that requires coordination that goes beyond software engineering.

### 4.3.2 Economic Case

The numbers from this project make a straightforward economic argument:

- The SSL pipeline reduces the labeling requirement from 100% of the dataset down to roughly 20%, which translates directly into annotation budget savings. For an 87,000-image dataset, that is the difference between a full expert-annotation budget and a much smaller one, which matters for research groups and agricultural extensions with limited funding.
- The BYOD pivot eliminates hardware costs entirely. The Jetson Orin Nano benchmark (Table 3.7) showed that a €350 dedicated device was actually slower than a phone, so there is no reason to buy one.
- Once installed, the marginal cost of each classification is zero: no cloud API calls, no per-prediction charges, no bandwidth costs.

These savings are real, but they only matter if the model is accurate enough to be useful. The held-out test evaluation suggests it is for the PlantVillage benchmark (94.90% top-1 accuracy with the saved SSL checkpoint), but that has only been validated on PlantVillage imagery, not on field photos.

### 4.3.3 Privacy and Local Operation

The offline-first architecture has a practical advantage that goes beyond connectivity. Because the model runs entirely on the user's device, no image data is sent to an external server. That removes server-side data-processing risks and reduces the compliance surface for privacy regulations such as GDPR, because the developer does not need to store or process user images on a backend. It does not remove all legal considerations. If the device itself is managed by an organisation, or if inference logs are collected, GDPR may still apply. Device-level security measures such as local encryption and access control remain important.

Whether this kind of tool would actually reduce crop losses is a question I cannot answer from a computer science thesis. What I can say is that the technical barriers, such as deployment size, connectivity requirements and inference speed, are no longer the main bottleneck. The remaining barriers are about trust, usability and distribution. Those need field trials and user research, not only more engineering.

### 4.3.4 Future Research Directions

A few directions for future work stand out:

1. **Field validation:** deploying the system with actual users and measuring classification accuracy on real-world images, with different lighting, backgrounds and camera quality. The PlantVillage results are promising, but controlled benchmarks and real-world performance are different things.
2. **Active learning:** instead of discarding every low-confidence sample, the system could flag uncertain predictions and ask for human input. That would turn the SSL loop into a targeted labeling tool.
3. **Federated learning:** multiple deployed devices could share model updates without sharing raw images, which would allow the model to improve over time while keeping data local.
4. **Multi-disease detection:** extending the model to handle images where several diseases appear on the same leaf simultaneously.
5. **Segmentation-assisted classification:** using leaf or symptom segmentation before classification to improve explainability and reduce the effect of irrelevant backgrounds.
6. **Burn ecosystem contributions:** contributing missing features (mixed-precision training, model quantisation) back to the open-source framework.
