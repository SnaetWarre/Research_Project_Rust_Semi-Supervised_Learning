# Literature Review: Incremental Learning for Plant Disease Detection

**Author**: [Your Name]  
**Date**: January 2024  
**Status**: In Progress

## Table of Contents

1. [Introduction](#introduction)
2. [Incremental Learning Background](#incremental-learning-background)
3. [Catastrophic Forgetting](#catastrophic-forgetting)
4. [Incremental Learning Methods](#incremental-learning-methods)
5. [Plant Disease Detection](#plant-disease-detection)
6. [Edge Computing and Mobile Deployment](#edge-computing-and-mobile-deployment)
7. [Relevant Datasets](#relevant-datasets)
8. [Research Gap](#research-gap)
9. [References](#references)

---

## Introduction

### Research Context

This literature review explores the intersection of three key research areas:
- **Incremental/Continual Learning**: Learning new tasks without forgetting old ones
- **Plant Disease Detection**: Computer vision for agricultural applications
- **Edge Deployment**: Running ML models on resource-constrained devices

### Research Questions

1. What are the most effective incremental learning strategies for image classification?
2. How severe is catastrophic forgetting when adding new plant disease classes?
3. Can incremental learning models maintain accuracy on edge devices?
4. What is the trade-off between model accuracy and inference speed on mobile hardware?

---

## Incremental Learning Background

### Definition

**Incremental Learning** (also called Continual Learning or Lifelong Learning) refers to the ability of a machine learning model to:
- Learn new tasks or classes sequentially
- Retain knowledge from previously learned tasks
- Adapt without requiring access to all previous training data

### Key Challenges

1. **Catastrophic Forgetting**: Dramatic performance drop on old tasks when learning new ones
2. **Memory Constraints**: Limited storage for previous data
3. **Computational Efficiency**: Need for fast adaptation
4. **Stability-Plasticity Dilemma**: Balancing retention vs. adaptation

### Evaluation Metrics

- **Average Accuracy**: Overall performance across all tasks
- **Forgetting Measure**: Performance drop on old tasks
- **Transfer Learning**: Positive/negative transfer between tasks
- **Memory Efficiency**: Storage requirements for the approach

---

## Catastrophic Forgetting

### Definition and Causes

**Catastrophic Forgetting** occurs when a neural network trained on new data completely overwrites previously learned representations.

**Primary Causes**:
- Weight updates that interfere with previously learned features
- Lack of replay/rehearsal of old data
- Overwriting of shared representations

### Measuring Forgetting

Common metrics:
```
Forgetting = Accuracy_old_after_new - Accuracy_old_before_new
Forgetting % = (Forgetting / Accuracy_old_before_new) × 100
```

### Literature Findings

| Study | Dataset | Task Type | Forgetting Rate |
|-------|---------|-----------|-----------------|
| TBD | TBD | TBD | TBD |

---

## Incremental Learning Methods

### 1. Fine-Tuning (Naive Baseline)

**Description**: Simply continue training on new data

**Pros**:
- Simple to implement
- Fast adaptation
- No additional memory

**Cons**:
- Severe catastrophic forgetting
- Poor retention of old knowledge

**Key Papers**:
- [ ] Add relevant citations

---

### 2. Learning without Forgetting (LwF)

**Citation**: Li, Z., & Hoiem, D. (2016). Learning without Forgetting. ECCV.

**Key Idea**: Use knowledge distillation to preserve old task performance
- Keep old model's predictions as "soft targets"
- New model must match both old predictions and new labels

**Algorithm**:
```
Loss = CrossEntropy(new_data, new_labels) + 
       λ × KL_Divergence(old_model_output, new_model_output)
```

**Hyperparameters**:
- Temperature (T): Controls softness of probability distribution (typical: 2-4)
- Distillation weight (λ): Balance between new and old knowledge (typical: 0.5-1.0)

**Pros**:
- No need to store old data
- Moderate forgetting reduction
- Computationally efficient

**Cons**:
- Requires old model for inference during training
- May not fully prevent forgetting
- Sensitive to hyperparameters

**Implementation Notes**:
- Use soft labels (probabilities) from old model
- Apply temperature scaling to both models
- Balance weights carefully based on task similarity

---

### 3. Elastic Weight Consolidation (EWC)

**Citation**: Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS.

**Key Idea**: Protect important weights from large changes
- Estimate importance of each weight using Fisher Information
- Add penalty term for changing important weights

**Algorithm**:
```
Loss = Task_Loss + (λ/2) × Σ F_i × (θ_i - θ*_i)²
```
Where:
- F_i: Fisher information for parameter i
- θ*_i: Optimal parameters from old task
- λ: Importance weight

**Fisher Information Estimation**:
```python
# Pseudo-code
for batch in old_data:
    loss = model(batch)
    gradients = compute_gradients(loss)
    fisher += gradients²  # Element-wise square
fisher /= num_batches
```

**Pros**:
- Principled approach from neuroscience
- No need for old data during training
- Can be computed efficiently

**Cons**:
- Requires Fisher information matrix computation
- Memory overhead for storing Fisher information
- May be too restrictive for highly different new tasks

**Hyperparameters**:
- Importance weight (λ): 100-10000 (dataset dependent)
- Number of samples for Fisher estimation: 200-1000

---

### 4. Rehearsal / Memory Replay

**Key Idea**: Store exemplars from old classes and replay during new training

**Variants**:
- **Random Sampling**: Randomly select samples to store
- **Herding**: Select samples closest to class mean
- **Core-set Selection**: Optimize for coverage of feature space

**Algorithm**:
```
1. Select k exemplars per old class
2. Store in memory buffer M
3. When learning new class:
   - Mix new data with exemplars from M
   - Train on combined dataset
```

**Pros**:
- Simple and effective
- Strong empirical performance
- Flexible (can combine with other methods)

**Cons**:
- Requires storing old data (privacy/storage concerns)
- Number of exemplars per class decreases as classes increase
- May not be representative of full distribution

**Memory Budget Strategies**:
- Fixed budget per class: k samples/class
- Fixed total budget: B samples total
- Growing budget: Allocate more to difficult classes

---

### 5. iCaRL (Incremental Classifier and Representation Learning)

**Citation**: Rebuffi, S. A., et al. (2017). iCaRL: Incremental Classifier and Representation Learning. CVPR.

**Key Idea**: Combine representation learning, classification, and rehearsal
- Use herding for exemplar selection
- Use distillation loss
- Nearest-mean-of-exemplars classifier

**Features**:
- Sophisticated exemplar selection
- Combines LwF with rehearsal
- Class-incremental learning focus

---

### Comparison Table

| Method | Memory Overhead | Computation | Forgetting | Old Data Required |
|--------|----------------|-------------|------------|-------------------|
| Fine-tuning | None | Low | High | No |
| LwF | Old Model | Medium | Medium | No |
| EWC | Fisher Matrix | Medium | Medium-Low | For Fisher calc |
| Rehearsal | Exemplars | Medium | Low | Yes (exemplars) |
| iCaRL | Exemplars + Old Model | High | Low | Yes (exemplars) |

---

## Plant Disease Detection

### Current State of the Art

**Traditional Approaches**:
- Hand-crafted features (SIFT, HOG, color histograms)
- Classical ML (SVM, Random Forests)
- Limited accuracy and generalization

**Deep Learning Approaches**:
- CNNs (VGG, ResNet, Inception)
- EfficientNet, MobileNet for mobile deployment
- Typical accuracy: 90-99% on standard datasets

### Key Papers

**Foundational Work**:
- [ ] Mohanty, S. P., et al. (2016). "Using Deep Learning for Image-Based Plant Disease Detection"
- [ ] Add more citations

**Recent Advances**:
- [ ] Add recent papers (2020-2024)

### Common Architectures

1. **EfficientNet-B0 to B7**
   - Accuracy: 95-98% on PlantVillage
   - Parameters: 5M (B0) to 66M (B7)
   - Best for: Balance of accuracy and efficiency

2. **ResNet-18/34/50**
   - Accuracy: 94-97%
   - Parameters: 11M (18) to 25M (50)
   - Best for: Strong baseline

3. **MobileNetV2/V3**
   - Accuracy: 92-95%
   - Parameters: 3-5M
   - Best for: Mobile deployment

### Challenges in Agriculture

1. **Data Collection**:
   - Limited labeled data for rare diseases
   - Imbalanced class distributions
   - Variable image quality (lighting, angle, occlusion)

2. **Real-world Deployment**:
   - Need for offline inference
   - Limited computational resources
   - Battery constraints

3. **Incremental Scenarios**:
   - New diseases emerge over time
   - Geographic variations in disease types
   - Seasonal disease patterns

---

## Edge Computing and Mobile Deployment

### Why Edge Deployment?

**Benefits**:
- Offline functionality (no internet required)
- Lower latency (instant predictions)
- Privacy preservation (data stays on device)
- Reduced server costs

**Challenges**:
- Limited compute (CPU, memory)
- Battery constraints
- Storage limitations
- Model size restrictions

### Mobile Optimization Techniques

1. **Quantization**
   - Convert FP32 → INT8
   - 4x size reduction, 2-4x speedup
   - Minimal accuracy loss (< 1%)

2. **Pruning**
   - Remove unimportant weights/neurons
   - 30-50% size reduction possible
   - May require fine-tuning

3. **Knowledge Distillation**
   - Train small model to mimic large model
   - Flexible architecture choice
   - Can maintain 95%+ of accuracy

4. **Architecture Design**
   - MobileNet: Depthwise separable convolutions
   - EfficientNet: Compound scaling
   - SqueezeNet: Fire modules

### Target Hardware

**Android Devices**:
- CPU: ARM Cortex-A76 or similar
- RAM: 4-8GB
- Storage: 64-128GB
- Inference: 50-200ms acceptable

**iOS Devices**:
- CPU: A13 Bionic or later
- RAM: 4GB+
- Neural Engine: 5+ TOPS
- Inference: 20-100ms acceptable

### Performance Benchmarks

| Model | Size (MB) | Inference Time (ms) | Accuracy (%) | Platform |
|-------|-----------|---------------------|--------------|----------|
| TBD | TBD | TBD | TBD | TBD |

---

## Relevant Datasets

### PlantVillage Dataset

**Overview**:
- **Size**: 54,000+ images
- **Classes**: 38 plant disease classes + healthy
- **Plants**: 14 crop species
- **Source**: Public domain, crowd-sourced
- **URL**: https://github.com/spMohanty/PlantVillage-Dataset

**Characteristics**:
- Controlled backgrounds (lab conditions)
- High image quality
- Balanced classes
- Limited real-world diversity

**Limitations**:
- Not representative of field conditions
- Limited background variation
- May overestimate real-world performance

### Other Relevant Datasets

1. **PlantDoc**
   - Real-world images
   - 2,598 images, 13 classes
   - More challenging than PlantVillage

2. **Crop Disease Dataset (Kaggle)**
   - Various crop diseases
   - Community contributed

3. **Custom Field Datasets**
   - Consider collecting own data for validation

---

## Research Gap

### Identified Gaps

1. **Incremental Learning for Agriculture**
   - Limited research combining IL and plant detection
   - No comprehensive comparison of IL methods in this domain
   - Lack of edge deployment considerations

2. **Practical Constraints**
   - Most research assumes unlimited compute
   - Few studies address mobile deployment
   - Limited work on realistic data scenarios

3. **Evaluation Metrics**
   - Need for domain-specific metrics
   - Consider farmer/agronomist usability
   - Real-world performance vs. benchmark performance

### Our Contribution

This research aims to:
1. Compare major IL methods (LwF, EWC, Rehearsal) on plant disease detection
2. Evaluate performance under realistic constraints (few examples, mobile hardware)
3. Provide reproducible baseline results for future research
4. Deploy working prototype on Android devices

---

## References

### Incremental Learning

1. Li, Z., & Hoiem, D. (2016). Learning without Forgetting. *ECCV*.

2. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

3. Rebuffi, S. A., et al. (2017). iCaRL: Incremental Classifier and Representation Learning. *CVPR*.

4. Lopez-Paz, D., & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. *NIPS*.

5. Chaudhry, A., et al. (2019). On Tiny Episodic Memories in Continual Learning. *arXiv*.

### Plant Disease Detection

6. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.

7. Too, E. C., Yujian, L., Njuki, S., & Yingchun, L. (2019). A comparative study of fine-tuning deep learning models for plant disease identification. *Computers and Electronics in Agriculture*, 161, 272-279.

### Edge Computing

8. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv*.

9. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.

### Continual Learning Surveys

10. Parisi, G. I., et al. (2019). Continual lifelong learning with neural networks: A review. *Neural Networks*, 113, 54-71.

11. De Lange, M., et al. (2021). A continual learning survey: Defying forgetting in classification tasks. *IEEE TPAMI*.

---

## Notes and Observations

### Key Insights

- [ ] Add insights as you read papers
- [ ] Note contradictions or debates in literature
- [ ] Identify promising directions

### Questions to Explore

- [ ] How does class similarity affect forgetting?
- [ ] What's the minimum number of exemplars needed?
- [ ] Can we predict which methods work best for which scenarios?

### Experimental Design Considerations

Based on literature review:
- Test with varying numbers of training samples (10, 50, 100, 500)
- Use 5 base classes + incrementally add 1-5 new classes
- Measure both accuracy and inference time
- Compare against naive fine-tuning baseline

---

**Status**: 
- [ ] Initial literature search complete
- [ ] Key papers identified and read
- [ ] Methodology comparison table complete
- [ ] Research gap clearly defined
- [ ] Experimental design informed by literature

**Next Steps**:
1. Read identified papers in depth
2. Update this document with findings
3. Refine research questions based on gaps
4. Design experiments based on best practices