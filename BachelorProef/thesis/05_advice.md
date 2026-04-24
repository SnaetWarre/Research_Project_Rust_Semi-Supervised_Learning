# 5. Advice

This chapter is written for someone who is facing the same research question: how to implement a semi-supervised neural network in Rust for automatic labeling on an edge device. The advice is grounded in the experimental results from Chapter 3 and shaped by the reflections in Chapter 4. It is structured as a practical, step-by-step guide.

## 5.1 Choose Your ML Framework Early

The choice of ML framework shapes everything else in the development stack, and changing frameworks halfway through a project is extremely costly. Based on this research, the decision matrix below applies.

| If you need... | Choose... | Reason |
|:---|:---|:---|
| Custom training loops (SSL, pseudo-labeling) | **Burn** | Extensive training API, backend-agnostic |
| LLM / transformer inference only | Candle | Optimised for inference, lightweight |
| Full PyTorch compatibility | tch-rs | Direct LibTorch bindings |
| Minimal deployment size | **Burn** or Candle | Static binary, no runtime dependencies |
| Cross-platform (iOS, Android, WASM, desktop) | **Burn** | wgpu backend covers all targets |

If the use case involves any form of SSL or an iterative training loop, Burn is the recommended choice. Its `Module` derive macro and backend generics mean that the same model code compiles for CUDA training and mobile inference without any modification.

**A pitfall to avoid:** do not underestimate Rust's compile times. A full release build of the `plantvillage_ssl` workspace takes 5 minutes or more. Use `cargo check` during development and keep `--release` builds for testing and deployment. The `sccache` compiler cache is also worth enabling in order to bring rebuild times down.

## 5.2 Start with Enough Labeled Data

The label efficiency experiment (Table 3.1) provides a clear empirical baseline:

- **Below 25 images per class:** accuracy is unacceptably low (under 60%). Do not try to train a model with this little data; the pseudo-labeling cycle will only propagate errors.
- **50 to 100 images per class:** the minimum viable range. At 100 images per class, accuracy reaches 85.53%, which is sufficient for the initial model in an SSL pipeline.
- **200 or more images per class:** diminishing returns. Any effort spent on labeling past 200 images per class is better invested in improving the pseudo-labeling pipeline.

**Practical recommendation:** collect at least 100 labeled images per class before starting SSL. If that is not feasible, spend the limited labeling budget on the class pairs that are most easily confused (for example diseases with similar visual symptoms). That gives the initial model the best possible decision boundary where it matters most.

## 5.3 Design the Pseudo-Labeling Pipeline

Based on the experimental results and the literature review, the following parameter settings are recommended as starting points:

| Parameter | Recommended value | Rationale |
|:---|:---|:---|
| Confidence threshold | 0.9 | Balances precision (>95%) against coverage |
| Retrain threshold | 150–200 samples | Batching reduces training overhead |
| Labeled data weight | 1.0 | Real labels are ground truth |
| Pseudo-label weight | 0.5–0.8 | Lower weight acknowledges the uncertainty |
| Labeled ratio | 0.2 (20%) | Leaves 60% for the SSL stream, 10% validation, 10% test |

**Step-by-step approach:**

1. Train the initial model on the labeled subset (20%) for 30 epochs. Validate on the held-out validation set.
2. Run inference on the unlabeled stream. Accept predictions that sit above the confidence threshold.
3. When the retrain threshold is reached, retrain the model on the combined dataset.
4. After each retraining cycle, evaluate on the validation set. If accuracy does not improve for two consecutive cycles, stop.
5. Evaluate the final model on the held-out test set, which was never used during training or pseudo-label selection.

**A pitfall to avoid:** do not use the test set for any decision during training, including pseudo-label threshold tuning. That is one of the most common sources of over-optimistic reported accuracy in SSL research.

## 5.4 Plan for Incremental Class Addition

If the deployment scenario involves adding new disease classes over time, which is very likely in any real-world agricultural application, the experimental results from Chapter 3 provide a few important guidelines:

1. **Start with a comprehensive base model.** The class scaling experiment (Table 3.2) shows that adding a class to a larger base causes 6× more forgetting. Starting with a larger base, however, means that the model already covers more diseases from the outset, which reduces how often updates are needed.

2. **Collect enough labeled data for new classes.** The new class position experiment (Table 3.3) shows that adding a 31st class to a 30-class model requires substantially more labeled samples than adding a 6th class to a 5-class model. At 50 labeled samples, the 6th class reaches 84% accuracy while the 31st class only reaches 26%.

3. **Use rehearsal methods when adding classes to large models.** Plain fine-tuning causes measurable forgetting on large models. Keeping a small buffer of examples per existing class and including them in the fine-tuning batches (experience replay) is the most practical mitigation.

4. **Monitor existing class accuracy after every update.** The experiments quantify forgetting rates, but these will vary by dataset and model. Automated testing against a held-out set for each existing class should be part of the update pipeline.

## 5.5 Target BYOD Over Dedicated Edge Hardware

The benchmark results (Table 3.6) lead to a strong, and perhaps counterintuitive, recommendation: **do not invest in dedicated edge hardware** (Jetson Nano, Coral and similar) for plant disease detection at inference time.

The reasoning is:

- An iPhone 12 (80 ms inference) outperforms a Jetson Orin Nano (120 ms), at zero additional hardware cost.
- Consumer devices have better displays, cameras and connectivity for distributing updates.
- The Tauri framework lets a single Rust codebase target iOS, Android and desktop.
- The deployment size of roughly 26 MB is small enough to install over Bluetooth, NFC or a brief mobile connection.

**Exception:** if the deployment requires running the model on a headless device (for example a camera trap or an automated greenhouse system), dedicated hardware can be justified. In that case, a Raspberry Pi 4 or 5 with the CPU backend is usually a better fit than a GPU-based edge device.

## 5.6 Test on Target Devices Early

One of the most valuable lessons from this project is to deploy to the target device **early in the development cycle**, not as a final step.

Testing on the actual device early reveals:

- **Unexpected latency:** the CPU backend can be 600× slower than the GPU backend. The wgpu backend may behave differently on mobile GPUs than it does on desktop GPUs.
- **Memory pressure:** mobile operating systems aggressively kill background apps that use too much memory. A model that runs fine in isolation can still fail when the phone is also running a camera preview.
- **Image preprocessing mismatches:** camera APIs return images in a variety of formats (NV21, BGRA, JPEG). Making sure the preprocessing pipeline handles all of these correctly is not a trivial task.
- **Permissions and sandboxing:** iOS and Android restrict file system access, camera access and background processing. Those restrictions affect how the model is loaded and where inference results can be stored.

**Recommendation:** by week 2 of development, have a minimal Tauri app that loads the model and runs inference on a single image on the target device. That establishes the deployment pipeline early and surfaces integration issues while they are still cheap to fix.

## 5.7 Practical Pitfalls and How to Avoid Them

| Pitfall | Symptom | Solution |
|:---|:---|:---|
| Burn compile times | 5 to 10 minute release builds | Use `cargo check`, `sccache`, incremental compilation |
| WASM binary size | > 50 MB WASM file | Enable `wasm-opt`, strip debug symbols, use `lto = true` |
| iOS sideloading | Cannot install without App Store | Use TestFlight for beta distribution, or Xcode direct install for development |
| Pseudo-label drift | Accuracy degrades over retraining cycles | Cap the ratio of pseudo-labels to real labels at 3:1; raise the confidence threshold if precision drops below 90% |
| GPU memory on mobile | Model fails to load | Switch to the ndarray (CPU) backend on devices with less than 4 GB of RAM; drop batch size to 1 |
| Model format compatibility | Weights trained on Burn 0.20 don't load on 0.14 | Keep version-locked workspaces; use weight export/import via JSON for cross-version compatibility |

## 5.8 Summary: The Recommended Workflow

For someone starting this research from scratch, the recommended workflow is:

```
Week 1:  Set up Burn + Tauri project scaffold
         Deploy a dummy model to the target device
         Verify the deployment pipeline end-to-end

Week 2:  Collect/prepare labeled dataset (≥100 images/class)
         Train initial CNN on labeled subset
         Measure baseline accuracy and inference latency

Week 3:  Implement pseudo-labeling pipeline
         Run SSL with confidence threshold 0.9
         Evaluate SSL improvement over baseline

Week 4:  Run incremental learning experiments
         Measure forgetting at different base sizes
         Determine minimum labeled data for new classes

Week 5:  Optimise deployment (binary size, startup time)
         Cross-platform testing (desktop, iOS, CPU-only)
         Benchmark all targets

Week 6:  Stress testing, edge-case handling
         Documentation and user guide
         Final evaluation on held-out test set
```

This timeline assumes familiarity with Rust and the availability of a labeled dataset. Collecting and annotating the dataset may require additional time, depending on the domain.
