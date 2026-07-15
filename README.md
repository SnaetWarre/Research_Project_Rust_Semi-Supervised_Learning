# Semi-Supervised Plant Disease Detection in Rust

Bachelor research on data-efficient plant-disease classification for offline edge devices. The project implements pseudo-labeling and incremental-learning experiments in Rust with Burn, compares the approach with a PyTorch reference, and packages inference for desktop, mobile, and browser-oriented deployment paths.

## Results

| Saved checkpoint | Held-out images | Top-1 accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Supervised baseline | 8,786 | 86.06% | 86.08% |
| Semi-supervised learning | 8,786 | **94.90%** | **94.74%** |

The SSL checkpoint improved accuracy by 8.84 percentage points and macro F1 by 8.66 points while starting with 20% labeled data. Both checkpoints were evaluated on the same held-out split, which was excluded from training and pseudo-label selection.

These are results from one saved split and one random seed, not averages across repeated runs. The experiments use PlantVillage's relatively uniform, lab-like images; performance on field images with different lighting, backgrounds, cameras, and disease stages is expected to be lower.

## Edge deployment

- Burn model weights: approximately **916 KB**
- ONNX export: approximately **1.8 MB**
- Compiled Rust release application: approximately **26 MB**
- RTX 3060 laptop inference: **0.42 ms per image**
- iPhone 12 through the Tauri Rust backend: approximately **80 ms per image**

The deployment result is the main systems advantage: inference can run locally without a Python interpreter, cloud API, or network connection.

## Research scope

The repository studies three connected questions:

1. How effectively can pseudo-labeling use an unlabeled image stream when only 20% of the data starts labeled?
2. How does adding one class differ for a small taxonomy (5 to 6 classes) versus a larger taxonomy (30 to 31 classes)?
3. How much labeled data is required for a useful new class while limiting catastrophic forgetting?

## Repository structure

```text
plantvillage_ssl/       # Burn training, evaluation, inference, GUI, and export
incremental_learning/   # 5→6 and 30→31 class experiments
pytorch_reference/      # reference implementation for comparison
benchmarks/             # framework and deployment benchmarks
BachelorProef/thesis/   # research method, final results, limitations, and advice
research/               # literature study and research notes
```

## Reproduce the main pipeline

Install a recent stable Rust toolchain. CPU execution works everywhere; CUDA requires a supported NVIDIA environment.

```bash
git clone https://github.com/SnaetWarre/Research_Project_Rust_Semi-Supervised_Learning.git
cd Research_Project_Rust_Semi-Supervised_Learning

./download_plantvillage.sh

cd plantvillage_ssl
cargo run --release --features cuda -- train \
  --data-dir data/plantvillage \
  --labeled-ratio 0.2 \
  --epochs 30 \
  --cuda
```

For CPU validation:

```bash
cargo check --manifest-path plantvillage_ssl/Cargo.toml --no-default-features --features cpu
```

Incremental-learning experiments are documented under [incremental_learning](incremental_learning).

## Evidence and documentation

- [Final results and limitations](BachelorProef/thesis/03_results.md)
- [Research design](BachelorProef/thesis/02_research.md)
- [Conclusion](BachelorProef/thesis/06_conclusion.md)
- [Installation and user documentation](plantvillage_ssl/docs/)
- [Technical portfolio article](https://snaetwarre.github.io/My-Portofolio/blog/blog.html)

## Production considerations

A production agricultural system would need field-data validation, repeated evaluation across seeds and locations, monitoring for drift and low-confidence classes, an update and rollback mechanism for offline devices, and feedback from domain experts. The current repository demonstrates the research and deployment approach; it does not claim clinical or agronomic certification.
