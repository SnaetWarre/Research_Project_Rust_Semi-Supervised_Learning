# 4. Reflection

This chapter gives a critical evaluation of the research results. It is divided into three parts: an honest assessment of the external feedback that was sought, a self-reflection on what went well and what did not, and a discussion of the broader implications for deployment.

## 4.1 External Feedback

I contacted two researchers from the 2AI-IPCA research lab, Helena Torres and Pedro Morais, who work in image processing and deep learning. I sent them the detailed interview questions by email. Pedro Morais answered five of the twenty questions.

His feedback is integrated into the reflection below. The full responses are included in Appendix B. Helena Torres's responses will be included in Appendix C when available.

## 4.2 Self-Reflection

### 4.2.1 What Went Well

**Deployment size and portability.** The binary of roughly 26 MB is a real improvement compared with a Python/PyTorch deployment. Both stacks need gigabytes of tooling during development. Rust's `target/` directory is around 2.1 GB, which is comparable to a PyTorch virtual environment. The difference is that Rust's compilation reduces everything to a single portable binary. A Python deployment still has to bring its interpreter and library tree to the target device. That changes which distribution channels are realistic for edge deployment. A file that fits on a Bluetooth transfer or a USB stick is very different from a system that first needs a multi-gigabyte environment.

**The BYOD pivot.** The benchmark results (Table 3.6) provided a clear, data-driven reason to move away from dedicated edge hardware. The Jetson Orin Nano, at €350, turned out to be slower than a phone that many people already own. That pivot removed the single largest cost barrier to deployment.

**Experimental rigour.** The three controlled experiments give quantitative answers to questions that are often only discussed qualitatively in the literature: how much labeled data is actually enough, how forgetting scales with model size, and how the position of a new class affects the difficulty of learning it.

### 4.2.2 Weaknesses and Limitations

**External expert feedback.** Pedro Morais from 2AI-IPCA reviewed the approach and confirmed that the 90% confidence threshold is a sensible starting point for controlled imaging conditions. He recommended sticking with a fixed threshold for this application, which is the strategy used in the current pipeline. He also suggested evaluating EfficientNet rather than the custom CNN, which aligns with the broader literature on mobile architectures and is a valid direction for future work. On augmentation, he cautioned that contrast and brightness manipulations can damage stability, a point that was not fully quantified in this project and that would benefit from a dedicated ablation study.

**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is limited by the accuracy of the model that was trained on the 20% labeled subset. If the initial model systematically misclasses certain classes, those errors can continue through the pseudo-labeling cycle. Techniques such as co-training, where two models look at different views of the data, could reduce this risk, but they were not implemented because of the VRAM constraints on edge devices.

**No field validation.** All experiments were carried out on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important ways: varying lighting, background vegetation, leaf angle, camera quality and the presence of several diseases on the same leaf. The model's performance on field-captured images is therefore unknown and is very likely lower than the numbers reported here.

**Burn ecosystem maturity.** Burn proved capable for this project, but the framework is still under active development. Its documentation is sparse compared to PyTorch, and some features (for example mixed-precision training and distributed training) are not yet available. Teams considering Burn for production should factor in the cost of working with a less mature ecosystem.

**Single dataset evaluation.** The experiments were carried out exclusively on PlantVillage. Generalisation to other agricultural datasets (different crops, different disease profiles, different imaging conditions) has not been validated.

## 4.3 Broader Implications

### 4.3.1 Practical Barriers to Deployment

Building the system is one thing. Getting it into the hands of someone who would actually use it is another. During development, a few things became clear about what stands between a working prototype and a useful tool:

The biggest issue is trust. If the model confidently says "bacterial spot" and the farmer treats for that, but it turns out to be something else, the tool has done more harm than good. That is why the GUI shows a confidence bar instead of only a single answer. Even then, a non-technical user might not know how to interpret a 72% confidence score. The interface has to be honest about uncertainty without becoming confusing, and that is a UX problem as much as an ML problem.

Device diversity is another concern. The BYOD strategy means the system has to work on whatever phone or laptop someone happens to own. The cross-platform benchmarks (Table 3.6) show that it can work across hardware, but the preprocessing bug on iOS (Section 3.6.3), where BGRA versus RGB channel ordering silently produced wrong classifications, is exactly the kind of problem that only appears on real devices. There will probably be more bugs like that on devices I have not tested.

Finally, there is the update problem. The initial installation is small enough (26 MB) to distribute offline, but what happens when the model improves or a new class is added? Agricultural extension workers or local community centres could serve as distribution points, but that requires coordination that goes beyond software engineering.

### 4.3.2 Economic Case

The numbers from this project make a straightforward economic argument:

- The SSL pipeline reduces the labeling requirement from 100% of the dataset down to roughly 20%, which translates directly into annotation budget savings. For an 87,000-image dataset, that is the difference between a full expert-annotation budget and a much smaller one, which matters for research groups and agricultural extensions with limited funding.
- The BYOD pivot eliminates hardware costs entirely. The Jetson Orin Nano benchmark (Table 3.6) showed that a €350 dedicated device was actually slower than a phone, so there is no reason to buy one.
- Once installed, the marginal cost of each classification is zero: no cloud API calls, no per-prediction charges, no bandwidth costs.

These savings are real, but they only matter if the model is accurate enough to be useful. The experimental results suggest it is (85%+ with SSL), but that has only been validated on PlantVillage imagery, not on field photos.

### 4.3.3 Privacy and Local Operation

The offline-first architecture has a practical advantage that goes beyond connectivity. Because the model runs entirely on the user's device, no image data is sent to an external server. That removes server-side data-processing risks and reduces the compliance surface for privacy regulations such as GDPR, because the developer does not need to store or process user images on a backend. It does not remove all legal considerations. If the device itself is managed by an organisation, or if inference logs are collected, GDPR may still apply. Device-level security measures such as local encryption and access control remain important.

Whether this kind of tool would actually reduce crop losses is a question I cannot answer from a computer science thesis. What I can say is that the technical barriers, such as deployment size, connectivity requirements and inference speed, are no longer the main bottleneck. The remaining barriers are about trust, usability and distribution. Those need field trials and user research, not only more engineering.

### 4.3.4 Future Research Directions

A few directions for future work stand out:

1. **Field validation:** deploying the system with actual users and measuring classification accuracy on real-world images, with different lighting, backgrounds and camera quality. The PlantVillage results are promising, but controlled benchmarks and real-world performance are different things.
2. **Active learning:** instead of discarding every low-confidence sample, the system could flag uncertain predictions and ask for human input. That would turn the SSL loop into a targeted labeling tool.
3. **Federated learning:** multiple deployed devices could share model updates without sharing raw images, which would allow the model to improve over time while keeping data local.
4. **Multi-disease detection:** extending the model to handle images where several diseases appear on the same leaf simultaneously.
5. **Burn ecosystem contributions:** contributing missing features (mixed-precision training, model quantisation) back to the open-source framework.
