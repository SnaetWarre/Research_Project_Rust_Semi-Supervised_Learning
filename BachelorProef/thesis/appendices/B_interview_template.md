# Appendix B: Interview Question Template

These questions are meant for the external reflection interviews (Chapter 4). They should be adapted to the expert's background and area of expertise.

---

## General Information

- **Date of interview:** ___
- **Name of interviewee:** ___
- **Organization / company:** ___
- **Role / title:** ___
- **Relevant expertise:** ___
- **Format:** (in person / video call / email exchange)

---

## Section 1: Context and Problem Relevance

1. In your experience, how significant is the problem of plant disease detection in agricultural practice? Is early detection something farmers actively seek solutions for?

2. How well do current solutions (laboratory analysis, cloud-based apps) actually perform in practice? What are the main pain points you tend to see?

3. What is your opinion on the requirement for offline, on-device operation? Is internet connectivity a genuine barrier in the agricultural contexts you work with?

## Section 2: Technical Approach

4. We use a semi-supervised learning approach (pseudo-labeling with a 90% confidence threshold) to bring the labeled data requirement down from 100% to 20%. How do you evaluate this approach? Do you see risks of error propagation?

5. The entire system is implemented in Rust using the Burn ML framework and compiles into a binary of roughly 26 MB. What is your reaction to this technology choice compared to the standard Python/PyTorch stack?

6. Our benchmarks show 0.39 ms inference on a desktop GPU and 80 ms on an iPhone 12. Are those latency numbers sufficient for the use cases you have in mind?

7. We chose to deploy on consumer devices (phones, laptops) rather than on dedicated edge hardware (Jetson), based on cost-performance benchmarks. Would you agree with that decision?

## Section 3: Incremental Learning and Scalability

8. Our experiments show that adding a new class to a 30-class model causes 6× more forgetting than adding one to a 5-class model. In your experience, how often do production models need to accommodate new classes? Is this a relevant concern for you?

9. Which methods do you use (or recommend) for mitigating catastrophic forgetting when deployed models are extended?

10. Do you see a practical path from 38 classes (PlantVillage) to coverage of the diseases that are relevant to your region or crop portfolio?

## Section 4: Deployment and Real-World Viability

11. What do you consider the biggest barriers to deploying an AI-based diagnostic tool to farmers in practice?

12. How would you distribute and update an offline application for end users who may have limited digital literacy?

13. What level of accuracy would you consider acceptable for a field-deployed plant disease detection tool? Is 85% sufficient, or does the threshold need to be higher?

## Section 5: Broader Impact

14. What economic value do you see in reducing labeling costs and in eliminating cloud inference charges?

15. Do you see societal or environmental benefits in making plant disease detection more accessible to smallholder farmers?

16. What future developments or research directions would you recommend for this type of project?

---

## Closing

17. Is there anything else you would like to add that we have not covered?

18. May I reference you by name and organization in the thesis?

---

*Note: The complete transcripts or written responses from each interview are included alongside this template in the final thesis submission.*

\newpage

---

# Tailored Questions: Helena R. Torres (2AI-IPCA)

**Interviewee:** Helena R. Torres  
**Organization:** 2AI-IPCA (Auxiliar Researcher)  
**Expertise:** Image processing, computer vision, deep learning, biomedical applications  
**Format:** Email exchange (or video call if preferred)

---

## Section 1: Semi-Supervised Learning & Pseudo-Labeling

1. In your experience with image classification tasks, how effective is pseudo-labeling as a semi-supervised strategy compared to other approaches (e.g., consistency regularization, MixMatch, FixMatch)? My system uses a confidence threshold of 90% to filter pseudo-labels and achieved ~82% accuracy using only 20% labeled data. Do you see risks of confirmation bias in this approach?

2. What techniques would you recommend for monitoring pseudo-label quality over time? In my SSL pipeline, pseudo-labels are accumulated and used for retraining after a threshold is reached. How would you detect and mitigate error propagation in such a loop?

3. My initial CNN is trained on only 20% of the dataset (~100 labeled images per class). From your experience with medical/biomedical image datasets, is this a realistic scenario? How does this compare to labeled data availability in your work (e.g., kidney segmentation, 3D head assessment)?

## Section 2: CNN Architecture & Image Processing

4. The model uses a relatively simple CNN architecture (3 conv blocks + 2 FC layers, ~850K parameters) designed for edge deployment. From your experience with image processing and deep learning, do you see this as sufficient for a 38-class classification task? Would you recommend a different architecture (e.g., MobileNet, EfficientNet-Lite) for this use case?

5. Data augmentation is a critical part of the training pipeline (random crops, flips, color jitter). In your work on medical image processing, which augmentation strategies have you found most effective for improving model robustness when dealing with limited labeled data?

6. The model processes 128×128 pixel input images. In your experience, how much does input resolution affect classification accuracy for fine-grained visual tasks? Would you recommend a different resolution for distinguishing between visually similar plant diseases?

## Section 3: Rust & Burn vs. Python/PyTorch

7. The entire system (training + inference + GUI) is implemented in Rust using the Burn ML framework, rather than the standard Python/PyTorch stack. The compiled binary is ~26 MB. As someone who works primarily with Python-based ML, what is your reaction to this technology choice? Do you see advantages or risks from a research reproducibility and team collaboration perspective?

8. One motivation for Rust was cross-platform deployment from a single codebase (desktop, mobile, embedded). In your experience with deploying deep learning models in practical/clinical settings, is portability an important factor? How do you typically handle deployment of your models?

## Section 4: Incremental Learning & Model Evolution

9. My experiments show that adding a new class to a 30-class model causes approximately 6× more catastrophic forgetting than adding one to a 5-class model. In your experience, how relevant is this problem in practice? Have you encountered situations where deployed models needed to learn new categories without forgetting existing ones?

10. Which methods for mitigating catastrophic forgetting (EWC, LwF, rehearsal buffers) have you seen work best in practice? Are there approaches from the medical imaging domain that you think could transfer well to agricultural image classification?

## Section 5: Edge Deployment & Real-World Viability

11. Our benchmarks show inference times of 0.39 ms on a desktop GPU and ~80 ms on an iPhone 12. We chose consumer devices (BYOD) over dedicated edge hardware (Jetson Orin Nano at €350) based on cost-performance analysis. What is your opinion on this deployment strategy?

12. In your work on real-time medical image processing (e.g., kidney segmentation in 3D US images), what latency thresholds do you consider acceptable for interactive applications? How does this compare to the requirements for field-based plant disease detection?

13. What do you consider the biggest technical barriers to deploying a deep learning model on consumer mobile devices? (e.g., model quantization, memory constraints, hardware fragmentation)

## Section 6: Broader Impact & Future Directions

14. From your perspective as a researcher in computer vision and deep learning, what future research directions would you recommend for this type of project? For example: active learning, federated learning, multi-task learning, attention mechanisms?

15. Do you see potential for the techniques used in this project (SSL + edge deployment + Rust) to transfer to other domains, such as medical imaging or industrial quality inspection?

16. Is there anything else you would like to add, or any aspect of the approach that you think deserves more attention?

---

## Closing

17. May I reference you by name and organization in my thesis?

---
