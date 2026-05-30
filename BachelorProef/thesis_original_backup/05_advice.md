# 5. Recommendations

This chapter presents practical recommendations derived from the experimental results in Chapter 3 and the reflections in Chapter 4. These recommendations are intended for researchers and practitioners who intend to implement semi-supervised neural networks in Rust for edge deployment. The guidance is grounded in empirical findings and is structured by topic rather than as a chronological procedure.

## 5.1 Framework Selection

The choice of machine learning framework constrains the remainder of the development stack, and changing frameworks mid-project carries substantial cost. Based on the findings of this research, the decision criteria in Table 5.1 apply.

**Table 5.1:** Framework selection criteria

| Requirement | Recommended framework | Rationale |
|:---|:---|:---|
| Custom training loops (SSL, pseudo-labeling) | **Burn** | Extensive training API, backend-agnostic |
| LLM / transformer inference only | Candle | Optimised for inference, lightweight |
| Full PyTorch compatibility | tch-rs | Direct LibTorch bindings |
| Minimal deployment size | **Burn** or Candle | Static binary, no runtime dependencies |
| Cross-platform (iOS, Android, WASM, desktop) | **Burn** | wgpu backend covers all targets in principle |

For use cases involving SSL or iterative training loops, Burn is the recommended choice. Its `Module` derive macro and backend generics allow the same model code to compile for CUDA training and mobile inference without modification.

A practical consideration is Rust's compilation time. A full release build of the `plantvillage_ssl` workspace requires five minutes or more. It is advisable to use `cargo check` during development and reserve `--release` builds for testing and deployment. Enabling the `sccache` compiler cache can reduce rebuild times.

## 5.2 Minimum Labeled Data Requirements

The label efficiency experiment (Table 3.2) provides the following empirical baseline:

- **Below 25 images per class:** accuracy remains below 60%. Training with so little data is risky because the pseudo-labeling cycle will propagate initial errors.
- **50 to 100 images per class:** the minimum viable range. At 100 images per class, accuracy reaches 85.53%, which is sufficient for the initial model in an SSL pipeline.
- **200 or more images per class:** diminishing returns. Effort spent on labeling beyond 200 images per class is better invested in improving the pseudo-labeling pipeline or collecting field data.

It is recommended to collect at least 100 labeled images per class before initiating SSL. If this is not feasible, the limited labeling budget should be allocated to class pairs that are most easily confused, such as diseases with similar visual symptoms. This strategy gives the initial model the most informative decision boundary in the regions where it matters most.

## 5.3 Pseudo-Labeling Pipeline Design

Based on the experimental results and the literature review, the parameter settings in Table 5.2 are recommended as initial values.

**Table 5.2:** Recommended pseudo-labeling parameters

| Parameter | Recommended value | Rationale |
|:---|:---|:---|
| Confidence threshold | 0.9 | Balances precision against coverage |
| Retrain threshold | 150–200 samples | Batching reduces training overhead |
| Labeled data weight | 1.0 | Real labels are ground truth |
| Pseudo-label weight | 0.5–0.8 | Lower weight acknowledges uncertainty |
| Labeled ratio | 0.2 (20%) | Leaves 60% for the SSL stream, 10% validation, 10% test |

The recommended pipeline proceeds as follows:

1. Train the initial model on the labeled subset (20%) for 30 epochs. Validate on the held-out validation set.
2. Run inference on the unlabeled stream. Accept predictions that exceed the confidence threshold.
3. When the retrain threshold is reached, retrain the model on the combined labeled and pseudo-labeled dataset.
4. After each retraining cycle, evaluate on the validation set. If accuracy does not improve for two consecutive cycles, terminate the pipeline.
5. Evaluate the final model on the held-out test set, which must not be used during training or pseudo-label selection.

A critical methodological constraint is that the test set must not be used for any decision during training, including pseudo-label threshold tuning. Using the test set for any optimisation during training is one of the most common causes of optimistic accuracy estimates in SSL research.

The global 0.9 threshold should not be treated as permanently fixed. It is a sensible starting point, but per-class acceptance rates, confidence histograms and validation accuracy should be logged after every retraining cycle. If one class receives far more pseudo-labels than the others, or if visually similar classes are repeatedly confused, switching to per-class thresholds or adding uncertainty estimation should be considered before continuing the SSL cycle.

## 5.4 Incremental Class Addition

In deployment scenarios where new classes are added over time—which is expected in most real-world agricultural applications—the experimental results from Chapter 3 yield the following guidelines:

1. **Start with a broad base model.** The class scaling experiment (Table 3.3) shows that adding a class to a larger base causes more forgetting. A larger base, however, means that the model covers more diseases from the outset, which reduces the frequency of subsequent updates.

2. **Collect sufficient labeled data for new classes.** The new class position experiment (Table 3.4) demonstrates that adding a 31st class to a 30-class model requires substantially more labeled samples than adding a 6th class to a 5-class model. At 50 labeled samples, the 6th class reaches 84% accuracy, whereas the 31st class reaches only 26%.

3. **Use rehearsal methods when adding classes to large models.** Plain fine-tuning causes measurable forgetting on large models. Keeping a small buffer of examples per existing class and including them in fine-tuning batches (experience replay) is the most practical mitigation.

4. **Monitor existing class accuracy after every update.** The experiments quantify forgetting rates, but these will vary by dataset and model. Automated testing against a held-out set for each existing class should be part of the update pipeline.

## 5.5 Hardware Targeting

The benchmark results (Table 3.7) lead to a recommendation that may appear counterintuitive: for interactive plant disease detection on consumer devices, dedicated edge hardware is usually not the best investment.

The supporting evidence is as follows:

- An iPhone 12 (80 ms inference) outperformed a Jetson Orin Nano (120 ms) in this test, at zero additional hardware cost.
- Consumer devices offer superior displays, cameras and connectivity for distributing updates.
- The Tauri framework allows a single Rust codebase to target iOS, Android and desktop.
- The deployment size of approximately 26 MB is small enough to install over Bluetooth or a brief mobile connection.

The exception to this recommendation is headless deployments, such as camera traps or automated greenhouse systems. In those cases, a Raspberry Pi 4 or 5 with the CPU backend is usually more suitable than a GPU-based edge device.

## 5.6 Early Device Testing

One of the most important lessons from this project is that deployment to the target device should occur early in the development cycle, not only at the end.

Early device testing reveals the following issues:

- **Latency differences:** the CPU backend can be orders of magnitude slower than the GPU backend. The wgpu backend may behave differently on mobile GPUs than on desktop GPUs.
- **Memory pressure:** mobile operating systems aggressively kill background applications that use excessive memory. A model that runs correctly in isolation may still fail when the device is also running a camera preview.
- **Image preprocessing mismatches:** camera APIs return images in various formats (NV21, BGRA, JPEG). Ensuring that the preprocessing pipeline handles all of these correctly is non-trivial.
- **Permissions and sandboxing:** iOS and Android restrict file system access, camera access and background processing. These restrictions affect how the model is loaded and where inference results can be stored.

It is recommended to establish a minimal deployment on the target device within the first two weeks of development. A minimal Tauri application that loads the model and runs inference on a single image is sufficient to validate the deployment pipeline and surface integration issues while they are still inexpensive to fix.

For image preprocessing specifically, a small set of fixed reference images should be retained, and the model's numerical outputs should be compared across desktop, mobile and web runtimes. This catches errors in channel ordering, resizing and normalisation before they become silent deployment bugs.

## 5.7 Common Pitfalls and Mitigations

**Table 5.3:** Common pitfalls and their mitigations

| Pitfall | Symptom | Mitigation |
|:---|:---|:---|
| Burn compile times | 5 to 10 minute release builds | Use `cargo check`, `sccache`, incremental compilation |
| WASM binary size | Greater than 50 MB WASM file | Enable `wasm-opt`, strip debug symbols, use `lto = true` |
| iOS sideloading | Cannot install without App Store | Use TestFlight for beta distribution, or Xcode direct install for development |
| Pseudo-label drift | Accuracy degrades over retraining cycles | Cap the ratio of pseudo-labels to real labels at 3:1; raise the confidence threshold if precision drops below 90% |
| GPU memory on mobile | Model fails to load | Switch to the ndarray (CPU) backend on devices with less than 4 GB of RAM, and set batch size to 1 |
| Model format incompatibility | Weights from the main SSL workspace do not load in the incremental learning workspace | Keep workspaces version-locked; use weight export/import via JSON for cross-workspace compatibility |

## 5.8 Suggested Development Timeline

For a project starting from scratch, the following timeline is suggested. It assumes familiarity with Rust and the availability of a labeled dataset.

**Phase 1 (Week 1):** Scaffold the Burn and Tauri project, deploy a minimal model to the target device, and verify the deployment pipeline end-to-end.

**Phase 2 (Week 2):** Collect or prepare the labeled dataset (at least 100 images per class), train the initial CNN on the labeled subset, and measure baseline accuracy and inference latency.

**Phase 3 (Week 3):** Implement the pseudo-labeling pipeline, run SSL with a confidence threshold of 0.9, and evaluate the SSL improvement over the baseline.

**Phase 4 (Week 4):** Conduct incremental learning experiments, measure forgetting at different base sizes, and determine the minimum labeled data requirements for new classes.

**Phase 5 (Week 5):** Optimise deployment (binary size, startup time), perform cross-platform testing (desktop, iOS, CPU-only), and benchmark all targets.

**Phase 6 (Week 6):** Conduct stress testing and edge-case handling, write documentation and a user guide, and perform the final evaluation on the held-out test set.

Collecting and annotating the dataset may require additional time, depending on the domain.
