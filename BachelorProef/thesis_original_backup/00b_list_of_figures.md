# List of Figures

| Figure | Caption | Page |
|:---|:---|:---:|
| Figure 1.1 | Deployment footprint comparison: Python/PyTorch stack versus Rust single binary. | Ch. 1 |
| Figure 2.1 | Flowchart of the implemented pseudo-labeling pipeline, showing the confidence-based filtering loop and the retrain-trigger condition. | Ch. 2 |
| Figure 2.2 | Conceptual comparison of deployment models for Burn, Candle and tch-rs. | Ch. 2 |
| Figure 2.3 | Conceptual overview of catastrophic forgetting mitigation strategies. | Ch. 2 |
| Figure 3.1 | Accuracy as a function of labeled images per class. The steepest gain sits between 25 and 100 images per class. | Ch. 3 |
| Figure 3.2 | Bar chart comparison of classification accuracy at each labeling level, from 5 to 500 images per class. | Ch. 3 |
| Figure 3.3 | Visual comparison of accuracy metrics between the small-base (5 classes) and large-base (30 classes) incremental learning scenarios. | Ch. 3 |
| Figure 3.4 | New class accuracy as a function of labeled samples, plotted for both the 6th class and 31st class scenarios. | Ch. 3 |
| Figure 3.5 | Detailed metric comparison at 50 labeled samples for the new class, contrasting the small-base and large-base conditions. | Ch. 3 |
| Figure 3.6 | Catastrophic forgetting as a function of labeled samples for the new class, across both base sizes. | Ch. 3 |
| Figure 3.7 | Screenshot of the Tauri desktop application dashboard, showing dataset loading, model status and class distribution diagnostics. | Ch. 3 |

---

# List of Tables

| Table | Caption | Page |
|:---|:---|:---:|
| Table 2.1 | Comparison of Rust ML frameworks (Burn, Candle, tch-rs). | Ch. 2 |
| Table 2.2 | Dataset characteristics for the New Plant Diseases Dataset. | Ch. 2 |
| Table 3.1 | Held-out test evaluation of the saved supervised baseline and SSL checkpoints. | Ch. 3 |
| Table 3.2 | Label efficiency results: classification accuracy and training time at seven labeled-data quantities. | Ch. 3 |
| Table 3.3 | Class scaling results: accuracy, forgetting and training time when one class is added to a 5-class and a 30-class base model. | Ch. 3 |
| Table 3.4 | New class accuracy by label count and base size, at five labeling levels from 5 to 100 samples. | Ch. 3 |
| Table 3.5 | Catastrophic forgetting by label count and base size, at five labeling levels from 5 to 100 samples. | Ch. 3 |
| Table 3.6 | Burn CUDA backend inference benchmark for the saved baseline and SSL checkpoints. | Ch. 3 |
| Table 3.7 | Hardware benchmark comparison across four deployment targets for the SSL checkpoint model. | Ch. 3 |
