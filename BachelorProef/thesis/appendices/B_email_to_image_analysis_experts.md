# Email Draft: Interview Questions for Image Analysis Experts

Subject: Short expert feedback request for my bachelor's thesis

Dear Miss Torres and Sir Morais,

I hope you are both doing well.

I am currently finishing my bachelor's thesis at Howest, where I built a semi-supervised image classification pipeline for plant disease detection. The use case is agricultural, but the part where your feedback would be most valuable is the technical image-analysis side: pseudo-labeling, CNN design, preprocessing, evaluation, deployment and model reliability.

The system is implemented in Rust using the Burn machine learning framework. It trains an initial CNN on 20% labeled data, then uses pseudo-labeling on the remaining unlabeled stream data with a 90% confidence threshold. The model is a lightweight CNN with four convolutional blocks and 128x128 RGB input images, designed for edge deployment on consumer devices. The goal is to evaluate whether this kind of semi-supervised model can be implemented efficiently and reliably outside the standard Python/PyTorch stack.

Would you be willing to answer the questions below by email? Short answers are completely fine. I mainly want to validate whether the technical choices are reasonable from the perspective of experienced researchers in image processing.

If you agree, may I also reference your name, organization and role in the reflection chapter of my thesis?

Thank you very much for your time.

Kind regards,

Warre Snaet

## Questions

### Semi-Supervised Learning and Pseudo-Labeling

1. In your experience with image classification tasks, how effective is pseudo-labeling as a semi-supervised strategy compared to other approaches, such as consistency regularization, MixMatch or FixMatch? My system uses a 90% confidence threshold to accept pseudo-labels, with 20% labeled data, 60% unlabeled stream data, 10% validation data and 10% test data. Do you see a risk of confirmation bias or error amplification in this setup?

2. The SSL pipeline accumulates accepted pseudo-labels and retrains the model once 200 pseudo-labeled images are available. What techniques would you recommend for monitoring pseudo-label quality over time? Would you rely on a fixed confidence threshold, or would you prefer calibration methods, per-class thresholds, uncertainty estimation or another strategy?

3. The initial CNN is trained on roughly 17,400 labeled images from a total dataset of about 87,000 images, across 38 classes. From your experience with medical or biomedical image datasets, is this a realistic labeled-data scenario? How does it compare with the amount and quality of annotation you typically see in your own work?

### CNN Architecture and Image Processing

4. The model uses a lightweight custom CNN with four convolutional blocks, batch normalization, ReLU, max pooling, adaptive average pooling, dropout and two linear layers. It was chosen for edge deployment rather than maximum accuracy. From your experience, is this a defensible architecture for a 38-class image classification task, or would you recommend a mobile architecture such as MobileNet or EfficientNet-Lite?

5. The training pipeline uses augmentations such as random crops, flips and color jitter. In your image-analysis work, which augmentation strategies have you found most useful when labeled data is limited? Are there augmentations you would avoid because they can damage clinically or visually meaningful image features?

6. The model mainly processes 128x128 RGB input images to keep memory use and inference time low. For fine-grained visual classification, how would you evaluate whether this resolution is sufficient? Would you test higher resolutions, crops around symptomatic regions, segmentation before classification or another preprocessing approach?

### Evaluation, Failure Analysis and Preprocessing

7. Beyond global accuracy, which evaluation outputs would you expect for this type of image classification system? For example, would you consider per-class precision and recall, F1-score, confusion matrices, confidence histograms, calibration curves or visual failure-case analysis essential?

8. Some plant disease classes are visually similar. When a model confuses such classes, what would you inspect first to understand the cause? Would you focus on model capacity, input resolution, augmentation choices, dataset quality, class imbalance, label noise or something else?

9. During deployment, preprocessing consistency became important because image channel ordering and resizing can differ between platforms. In your experience, what validation steps are needed to make sure an image model behaves consistently across different devices or runtimes?

### Rust and Burn Compared with Python and PyTorch

10. The whole system, including training, inference and the GUI backend, is implemented in Rust with the Burn ML framework instead of the standard Python/PyTorch stack. The trained model is small and the compiled deployment binary is roughly 26 MB. What is your reaction to this technology choice from a research reproducibility, maintainability and collaboration perspective?

11. One motivation for Rust was cross-platform deployment from a single codebase, targeting desktop, mobile and edge-like devices. In your practical or clinical deployment experience, how important is portability? How do you usually move models from research code to usable applications?

### Incremental Learning and Model Evolution

12. My experiments show that adding a new class to a 30-class model causes approximately 6x more catastrophic forgetting than adding one to a 5-class model. In your experience, how relevant is this problem in practice? Have you encountered situations where deployed models needed to learn new categories without forgetting existing ones?

13. Which methods for mitigating catastrophic forgetting, such as EWC, Learning without Forgetting or rehearsal buffers, have you seen work best in practice? Are there approaches from the medical imaging domain that could transfer well to agricultural image classification?

### Edge Deployment and Real-World Viability

14. The benchmarks show inference times of about 0.39 ms on a desktop GPU and about 80 ms on an iPhone 12. The project therefore focuses on consumer devices instead of dedicated edge hardware. From a technical perspective, is this a reasonable deployment strategy for interactive image classification?

15. In your work with real-time or interactive image processing, what latency thresholds do you consider acceptable? How would those expectations compare with a field-based plant disease detection app where the user takes or selects one image at a time?

16. What do you consider the biggest technical barriers to deploying a deep learning model on consumer mobile devices? For example, model quantization, memory constraints, hardware fragmentation, preprocessing differences or update distribution.

### Broader Impact and Future Directions

17. From your perspective as researchers in computer vision and deep learning, what future research directions would you recommend for this type of project? For example, active learning, federated learning, multi-task learning, attention mechanisms, better calibration or segmentation-assisted classification.

18. Do you see potential for the techniques used in this project, namely semi-supervised learning, edge deployment and a Rust-based implementation, to transfer to other domains such as medical imaging or industrial quality inspection?

19. Is there anything else you would like to add, or any aspect of the approach that you think deserves more attention?

20. May I reference you by name and organization in my thesis?



