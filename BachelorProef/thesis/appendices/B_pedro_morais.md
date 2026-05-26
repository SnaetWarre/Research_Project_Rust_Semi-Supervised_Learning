# Appendix B: Expert Feedback - Pedro Morais

**Name:** Pedro Morais  
**Organisation:** 2AI – Applied Artificial Intelligence Laboratory (IPCA)  
**Role:** Researcher in image processing and deep learning  
**Email:** pmorais@ipca.pt  

Pedro Morais answered the following five questions from a broader questionnaire of twenty questions sent by email.

## Semi-Supervised Learning and Pseudo-Labeling

**Question 1:** In your experience with image classification tasks, how effective is pseudo-labeling as a semi-supervised strategy compared to other approaches, such as consistency regularization, MixMatch or FixMatch? My system uses a 90% confidence threshold to accept pseudo-labels, with 20% labeled data, 60% unlabeled stream data, 10% validation data and 10% test data. Do you see a risk of confirmation bias or error amplification in this setup?

> That depends on your image quality. Based on the description, and assuming controlled lighting conditions, I believe that's a good approach.

**Question 2:** The SSL pipeline accumulates accepted pseudo-labels and retrains the model once 200 pseudo-labeled images are available. What techniques would you recommend for monitoring pseudo-label quality over time? Would you rely on a fixed confidence threshold, or would you prefer calibration methods, per-class thresholds, uncertainty estimation or another strategy?

> Based on your application, I will begin with a fixed confidence threshold.

**Question 3:** The initial CNN is trained on roughly 17,400 labeled images from a total dataset of about 87,000 images, across 38 classes. From your experience with medical or biomedical image datasets, is this a realistic labeled-data scenario? How does it compare with the amount and quality of annotation you typically see in your own work?

> Medical case volumes are frequently low.

## CNN Architecture and Image Processing

**Question 4:** The model uses a lightweight custom CNN with four convolutional blocks, batch normalization, ReLU, max pooling, adaptive average pooling, dropout and two linear layers. It was chosen for edge deployment rather than maximum accuracy. From your experience, is this a defensible architecture for a 38-class image classification task, or would you recommend a mobile architecture such as MobileNet or EfficientNet-Lite?

> Given EfficientNet's strong performance in our team's previous work, I will evaluate its efficacy for this application.

**Question 5:** The training pipeline uses augmentations such as random crops, flips and color jitter. In your image-analysis work, which augmentation strategies have you found most useful when labeled data is limited? Are there augmentations you would avoid because they can damage clinically or visually meaningful image features?

> The majority of these experiments utilized a strictly supervised learning approach, where data augmentation consistently enhanced model performance. However, caution must be exercised regarding contrast and brightness manipulations, as they can negatively impact stability.
