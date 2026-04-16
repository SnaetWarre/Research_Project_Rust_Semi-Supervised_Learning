# 4. Reflection

This chapter provides a critical evaluation of the research results through two lenses: feedback from external experts with relevant industry or academic experience, and a broader analysis of the implications of this work. The thesis guidelines require input from a minimum of two external persons with meaningful feedback; their complete interviews are included in the appendices.

## 4.1 External Reflection

### 4.1.1 External Expert 1

> **[TODO: Insert name and company/organization]**
> *[TODO: Insert role/title and brief description of expertise]*

**Context of the conversation:** [TODO: Describe how you contacted this person and the setting of the interview (in person, video call, etc.)]

**Key discussion points:**

**On the practical deployment in agricultural settings:**

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

[TODO: What is their experience with adding new classes to deployed models? Do they use EWC, LwF, or rehearsal methods in practice? How does the forgetting behavior observed in the experiments compare to what they see in the field?]

**On SSL limitations and pseudo-label quality:**

[TODO: How do they assess the risk of confirmation bias in pseudo-labeling? What techniques do they recommend for monitoring pseudo-label quality over time?]

**On deployment strategy and cross-platform targeting:**

[TODO: What is their opinion on the BYOD pivot away from dedicated edge hardware? What challenges do they foresee with targeting multiple platforms from a single Rust codebase?]

### 4.1.3 Synthesis of External Feedback

[TODO: After completing both interviews, write a synthesis that identifies common themes, contrasting viewpoints, and the most actionable feedback. This synthesis should connect the external perspectives to the decisions made during the project.]

## 4.2 Self-Reflection on Results

### 4.2.1 Strengths

**Deployment size and portability.** The ~26 MB binary represents a genuine step change compared to a Python/PyTorch deployment. Both stacks require gigabytes of tooling during development (Rust's `target/` directory is ~2.1 GB, comparable to a PyTorch virtual environment), but Rust's compilation distills everything into a single portable binary. A Python deployment must carry its interpreter and library tree to the target device. This fundamentally changes which distribution channels are viable for edge deployment: a file that fits on a Bluetooth transfer or a USB stick is qualitatively different from one that requires installing a multi-gigabyte environment first.

**The BYOD pivot.** The benchmark results (Table 3.6) provided a clear, data-driven reason to abandon dedicated edge hardware. The Jetson Orin Nano, at €350, was slower than a phone that farmers already own. This pivot eliminated the single largest cost barrier to deployment.

**Experimental rigor.** The three controlled experiments provide quantitative answers to questions that are often addressed only qualitatively in the literature: how much labeled data is enough (100/class), how forgetting scales with model size (6×), and how class position affects learning difficulty (substantially).

### 4.2.2 Weaknesses and Limitations

**Pseudo-label quality is bounded by the initial model.** The SSL pipeline's effectiveness is fundamentally limited by the accuracy of the model trained on the 20% labeled subset. If the initial model systematically misclassifies certain disease classes, those errors will propagate through the pseudo-labeling cycle. Techniques such as co-training (using two models with different views of the data) could mitigate this, but were not implemented due to VRAM constraints on edge devices.

**No field validation.** All experiments were conducted on the PlantVillage dataset under controlled conditions. Real-world agricultural images differ significantly: varying lighting, background vegetation, leaf angle, camera quality, and the presence of multiple diseases on the same leaf. The model's performance on field-captured images is unknown and is likely lower than the numbers reported here.

**Burn ecosystem maturity.** While Burn proved capable for this project, the framework is still under active development. Documentation is sparse compared to PyTorch, and some features (e.g., mixed-precision training, distributed training) are not yet available. Teams considering Burn for production should factor in the cost of working with a less mature ecosystem.

**Single dataset evaluation.** The experiments were conducted exclusively on PlantVillage. Generalization to other agricultural datasets (e.g., different crops, different disease profiles, different imaging conditions) has not been validated.

## 4.3 Broader Impact Analysis

### 4.3.1 Implementation Barriers

Deploying an offline AI system in the field involves challenges beyond the technical:

- **Digital literacy:** farmers in regions where offline operation is most needed may have limited experience with smartphone applications. The UI must be simple, with clear visual feedback and minimal text.
- **Device diversity:** the "BYOD" model means the system must work on a wide range of Android and iOS devices with varying camera quality and processing power.
- **Trust in AI:** a farmer who receives an incorrect diagnosis may lose trust in the system entirely. False positive rates must be communicated transparently, and the system should indicate confidence levels rather than presenting a single definitive answer.
- **Update distribution:** while the initial installation is small enough for offline distribution, model updates (new classes, improved weights) still require a distribution mechanism, potentially through agricultural extension workers or community access points.

### 4.3.2 Business Value

The economic argument for this approach is straightforward:

- **Annotation cost reduction:** SSL reduces the labeled data requirement by approximately 60–80%, translating to tens of thousands of euros saved in annotation costs for new deployments.
- **Hardware cost elimination:** the pivot from dedicated edge hardware (€350/unit) to BYOD eliminates capital expenditure entirely.
- **No recurring cloud costs:** zero inference API calls means zero per-prediction charges. The marginal cost of each classification after deployment is effectively zero.
- **Faster time-to-diagnosis:** sub-second inference replaces a multi-day laboratory turnaround, enabling farmers to act before diseases spread.

### 4.3.3 Societal Impact

Plant disease detection has direct implications for food security. The FAO estimates that plant pests and diseases cause up to 40% of crop losses globally, with the economic impact exceeding $220 billion annually [15]. Smallholder farmers in Sub-Saharan Africa and South Asia are disproportionately affected.

An offline, phone-based disease detection tool removes two critical access barriers: internet connectivity and specialized equipment. If the accuracy demonstrated in this research translates to field conditions, such a tool could enable earlier intervention, reduce crop losses, and improve food security for some of the world's most vulnerable farming communities.

The offline-first architecture also has a privacy advantage, as noted in the NVISO guest session on AI threats (Appendix C): because the model runs entirely on-device, no agricultural data is transmitted to external servers. This eliminates risks of data exfiltration and removes the need for data processing agreements, a practical benefit for deployment in regions with varying data protection regulations.

### 4.3.4 Future Research Directions

Several avenues for future work emerge from this research:

1. **Field validation study:** deploying the system with actual farmers and measuring classification accuracy on real-world images captured under diverse conditions.
2. **Active learning integration:** instead of a fixed confidence threshold, the system could identify uncertain samples and request targeted human annotation, creating a human-in-the-loop labeling pipeline.
3. **Federated learning:** multiple deployed devices could periodically aggregate model updates without sharing raw data, enabling continuous improvement while preserving privacy.
4. **Multi-disease detection:** extending the model to handle images containing multiple simultaneous diseases on a single leaf.
5. **Burn ecosystem contributions:** contributing missing features (mixed-precision training, model quantization) back to the open-source Burn framework.
