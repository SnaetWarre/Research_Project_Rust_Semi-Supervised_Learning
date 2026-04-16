# 4. Reflection

This chapter offers a critical evaluation of the research results through two lenses: feedback from external experts with relevant industry or academic experience, and a broader analysis of the implications of this work. The thesis guidelines require meaningful input from at least two external persons; their complete interviews are included in the appendices.

## 4.1 External Reflection

### 4.1.1 External Expert 1

> **[TODO: Insert name and company/organization]**
> *[TODO: Insert role/title and brief description of expertise]*

**Context of the conversation:** [TODO: Describe how you contacted this person and the setting of the interview (in person, video call, etc.)]

**Key discussion points:**

**On practical deployment in agricultural settings:**

[TODO: What did this expert say about whether farmers would actually use an offline plant disease detection app? What barriers do they see (digital literacy, trust in AI, device availability)?]

**On the value of SSL for reducing labeling costs:**

[TODO: How does this expert evaluate the pseudo-labeling approach from a practical standpoint? Is the €2/image annotation cost realistic in their experience? How do they typically handle labeled data acquisition in their work?]

**On the Rust/Burn technology choice:**

[TODO: What is their reaction to using Rust instead of Python for ML? Do they see advantages or risks from a maintenance and team perspective?]

**On limitations and potential improvements:**

[TODO: What weaknesses or blind spots did this expert identify? What would they recommend doing differently?]

### 4.1.2 External Expert 2

> **[TODO: Insert name and company/organization]**
> *[TODO: Insert role/title and brief description of expertise]*

**Context of the conversation:** [TODO: Describe how you contacted this person and the setting of the interview.]

**Key discussion points:**

**On the Burn framework and Rust ML maturity:**

[TODO: How does this expert evaluate the Burn framework's maturity for production use? What are the ecosystem gaps compared to PyTorch/TensorFlow?]

**On incremental learning in production:**

[TODO: What is their experience with adding new classes to deployed models? Do they use EWC, LwF or rehearsal methods in practice? How does the forgetting behavior observed in the experiments compare to what they see in the field?]

**On SSL limitations and pseudo-label quality:**

[TODO: How do they assess the risk of confirmation bias in pseudo-labeling? What techniques do they recommend for monitoring pseudo-label quality over time?]

**On deployment strategy and cross-platform targeting:**

[TODO: What is their opinion on the BYOD pivot away from dedicated edge hardware? What challenges do they foresee with targeting multiple platforms from a single Rust codebase?]

### 4.1.3 Synthesis of External Feedback

[TODO: After completing both interviews, write a synthesis that identifies common themes, contrasting viewpoints and the most actionable feedback. This synthesis should connect the external perspectives to the decisions made during the project.]

## 4.2 Self-Reflection on Results

### 4.2.1 Strengths

**Deployment size and portability.** The binary of roughly 26 MB is a genuine step change compared to a Python/PyTorch deployment. Both stacks need gigabytes of tooling during development (Rust's `target/` directory is around 2.1 GB, which is comparable to a PyTorch virtual environment), but Rust's compilation distils everything down into a single portable binary. A Python deployment, by contrast, still has to carry its interpreter and library tree along to the target device. That fundamentally changes which distribution channels are viable for edge deployment: a file that fits on a Bluetooth transfer or a USB stick is qualitatively different from one that requires a multi-gigabyte environment to be installed first.

**The BYOD pivot.** The benchmark results (Table 3.6) provided a clear, data-driven reason to walk away from dedicated edge hardware. The Jetson Orin Nano, at €350, turned out to be slower than a phone that farmers already own. That pivot removed the single largest cost barrier to deployment.

**Experimental rigour.** The three controlled experiments give quantitative answers to questions that are often only addressed qualitatively in the literature: how much labeled data is actually enough (100 per class), how forgetting scales with model size (6×), and how the position of a new class affects the difficulty of learning it (substantially).

### 4.2.2 Weaknesses and Limitations

**Pseudo-label quality is bounded by the initial model.** The effectiveness of the SSL pipeline is fundamentally limited by the accuracy of the model that was trained on the 20% labeled subset. If the initial model systematically misclassifies certain disease classes, those errors propagate through the pseudo-labeling cycle. Techniques such as co-training, where two models look at different views of the data, could mitigate this, but they were not implemented because of the VRAM constraints on edge devices.

**No field validation.** All experiments were carried out on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ in important ways: varying lighting, background vegetation, leaf angle, camera quality and the presence of several diseases on the same leaf. The model's performance on field-captured images is therefore unknown and is very likely lower than the numbers reported here.

**Burn ecosystem maturity.** Burn proved capable for this project, but the framework is still under active development. Its documentation is sparse compared to PyTorch, and some features (for example mixed-precision training and distributed training) are not yet available. Teams considering Burn for production should factor in the cost of working with a less mature ecosystem.

**Single dataset evaluation.** The experiments were carried out exclusively on PlantVillage. Generalisation to other agricultural datasets (different crops, different disease profiles, different imaging conditions) has not been validated.

## 4.3 Broader Impact Analysis

### 4.3.1 Implementation Barriers

Deploying an offline AI system in the field comes with challenges that go beyond the purely technical:

- **Digital literacy:** farmers in regions where offline operation is most needed may have limited experience with smartphone applications. The UI has to be simple, with clear visual feedback and minimal text.
- **Device diversity:** the BYOD model means that the system has to work on a wide range of Android and iOS devices with varying camera quality and processing power.
- **Trust in AI:** a farmer who receives an incorrect diagnosis may lose trust in the system altogether. False-positive rates have to be communicated transparently, and the system should report a confidence level rather than presenting a single definitive answer.
- **Update distribution:** the initial installation is small enough for offline distribution, but model updates (new classes, improved weights) still need a distribution mechanism, possibly through agricultural extension workers or community access points.

### 4.3.2 Business Value

The economic argument for this approach is relatively simple:

- **Annotation cost reduction:** SSL brings the labeled data requirement down by roughly 60 to 80%, which translates into tens of thousands of euros saved in annotation costs for every new deployment.
- **Hardware cost elimination:** the pivot from dedicated edge hardware (€350 per unit) to BYOD removes the capital expenditure entirely.
- **No recurring cloud costs:** zero inference API calls means zero per-prediction charges. The marginal cost of each classification after deployment is effectively zero.
- **Faster time-to-diagnosis:** sub-second inference replaces a multi-day laboratory turnaround, which allows farmers to act before diseases have time to spread.

### 4.3.3 Societal Impact

Plant disease detection has direct implications for food security. The FAO estimates that plant pests and diseases cause up to 40% of crop losses globally, with an annual economic impact of more than $220 billion [15]. Smallholder farmers in Sub-Saharan Africa and South Asia are disproportionately affected.

An offline, phone-based disease detection tool removes two critical access barriers: internet connectivity and specialised equipment. If the accuracy demonstrated in this research translates to field conditions, a tool of this kind could enable earlier intervention, reduce crop losses and improve food security for some of the world's most vulnerable farming communities.

The offline-first architecture also has a privacy advantage, as pointed out in the NVISO guest session on AI threats (Appendix C). Because the model runs entirely on the device, no agricultural data is transmitted to an external server. That eliminates the risk of data exfiltration and removes the need for data processing agreements, which is a practical benefit for deployment in regions with varying data protection regulations.

### 4.3.4 Future Research Directions

Several avenues for future work stand out:

1. **Field validation study:** deploying the system with actual farmers and measuring classification accuracy on real-world images captured under diverse conditions.
2. **Active learning integration:** rather than using a fixed confidence threshold, the system could identify uncertain samples and request targeted human annotation, which would turn the SSL loop into a human-in-the-loop labeling pipeline.
3. **Federated learning:** multiple deployed devices could periodically aggregate model updates without sharing raw data, which would allow the model to improve continuously while preserving privacy.
4. **Multi-disease detection:** extending the model so that it can handle images that contain several simultaneous diseases on a single leaf.
5. **Burn ecosystem contributions:** contributing missing features (mixed-precision training, model quantisation) back to the open-source Burn framework.
