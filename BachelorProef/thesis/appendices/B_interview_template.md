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

*Note: No written interview responses were received in time for this thesis submission. This template is included as documentation of the questions that were sent.*

\newpage

---

# Tailored Questions: Image Analysis Experts

**Interviewees:** Miss Torres and Sir Morais  
**Organization:** 2AI-IPCA  
**Expertise:** Image processing, computer vision, deep learning, biomedical applications  
**Format:** Email exchange (or video call if preferred)

---

## Section 1: Semi-Supervised Learning & Pseudo-Labeling

1. In your experience with image classification tasks, how effective is pseudo-labeling as a semi-supervised strategy compared to other approaches, such as consistency regularization, MixMatch or FixMatch? My system uses a 90% confidence threshold to accept pseudo-labels, with 20% labeled data, 60% unlabeled stream data, 10% validation data and 10% test data. Do you see a risk of confirmation bias or error amplification in this setup?

2. The SSL pipeline accumulates accepted pseudo-labels and retrains the model once 200 pseudo-labeled images are available. What techniques would you recommend for monitoring pseudo-label quality over time? Would you rely on a fixed confidence threshold, or would you prefer calibration methods, per-class thresholds, uncertainty estimation or another strategy?

3. The initial CNN is trained on roughly 17,400 labeled images from a total dataset of about 87,000 images, across 38 classes. From your experience with medical or biomedical image datasets, is this a realistic labeled-data scenario? How does it compare with the amount and quality of annotation you typically see in your own work?

## Section 2: CNN Architecture & Image Processing

4. The model uses a lightweight custom CNN with four convolutional blocks, batch normalization, ReLU, max pooling, adaptive average pooling, dropout and two linear layers. It was chosen for edge deployment rather than maximum accuracy. From your experience, is this a defensible architecture for a 38-class image classification task, or would you recommend a mobile architecture such as MobileNet or EfficientNet-Lite?

5. The training pipeline uses augmentations such as horizontal and vertical flips, rotation, brightness, contrast, saturation, blur and noise. In your image-analysis work, which augmentation strategies have you found most useful when labeled data is limited? Are there augmentations you would avoid because they can damage clinically or visually meaningful image features?

6. The model mainly processes 128x128 RGB input images to keep memory use and inference time low. For fine-grained visual classification, how would you evaluate whether this resolution is sufficient? Would you test higher resolutions, crops around symptomatic regions, segmentation before classification or another preprocessing approach?

## Section 3: Evaluation, Failure Analysis & Preprocessing

7. Beyond global accuracy, which evaluation outputs would you expect for this type of image classification system? For example, would you consider per-class precision and recall, F1-score, confusion matrices, confidence histograms, calibration curves or visual failure-case analysis essential?

8. Some plant disease classes are visually similar. When a model confuses such classes, what would you inspect first to understand the cause? Would you focus on model capacity, input resolution, augmentation choices, dataset quality, class imbalance, label noise or something else?

9. During deployment, preprocessing consistency became important because image channel ordering and resizing can differ between platforms. In your experience, what validation steps are needed to make sure an image model behaves consistently across different devices or runtimes?

## Section 4: Rust & Burn vs. Python/PyTorch

10. The whole system, including training, inference and the GUI backend, is implemented in Rust with the Burn ML framework instead of the standard Python/PyTorch stack. The trained model is small and the compiled deployment binary is roughly 26 MB. What is your reaction to this technology choice from a research reproducibility, maintainability and collaboration perspective?

11. One motivation for Rust was cross-platform deployment from a single codebase, targeting desktop, mobile and edge-like devices. In your practical or clinical deployment experience, how important is portability? How do you usually move models from research code to usable applications?

## Section 5: Incremental Learning & Model Evolution

12. My experiments show that adding a new class to a 30-class model causes approximately 6x more catastrophic forgetting than adding one to a 5-class model. In your experience, how relevant is this problem in practice? Have you encountered situations where deployed models needed to learn new categories without forgetting existing ones?

13. Which methods for mitigating catastrophic forgetting, such as EWC, Learning without Forgetting or rehearsal buffers, have you seen work best in practice? Are there approaches from the medical imaging domain that could transfer well to agricultural image classification?

## Section 6: Edge Deployment & Real-World Viability

14. The benchmarks show inference times of about 0.39 ms on a desktop GPU and about 80 ms on an iPhone 12. The project therefore focuses on consumer devices instead of dedicated edge hardware. From a technical perspective, is this a reasonable deployment strategy for interactive image classification?

15. In your work with real-time or interactive image processing, what latency thresholds do you consider acceptable? How would those expectations compare with a field-based plant disease detection app where the user takes or selects one image at a time?

16. What do you consider the biggest technical barriers to deploying a deep learning model on consumer mobile devices? For example, model quantization, memory constraints, hardware fragmentation, preprocessing differences or update distribution.

## Section 7: Broader Impact & Future Directions

17. From your perspective as researchers in computer vision and deep learning, what future research directions would you recommend for this type of project? For example, active learning, federated learning, multi-task learning, attention mechanisms, better calibration or segmentation-assisted classification.

18. Do you see potential for the techniques used in this project, namely semi-supervised learning, edge deployment and a Rust-based implementation, to transfer to other domains such as medical imaging or industrial quality inspection?

19. Is there anything else you would like to add, or any aspect of the approach that you think deserves more attention?

---

## Closing

20. May I reference you by name and organization in my thesis?

---
