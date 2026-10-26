# Semi-supervised plant disease detection in Rust

Bachelor research on classifying plant diseases from leaf photos when most photos have no label.
A small CNN is trained on the 5% of images that keep their labels, then improves itself on the
other images with pseudo-labels: its own confident predictions, used as training labels. Everything
runs in Rust with [Burn](https://burn.dev) 0.22, and one model at a time fits in 6 GB of VRAM.

## Results

RESULTS_TABLE

## How the experiment works

**Dataset.** The Kaggle [New Plant Diseases](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
dataset: 87,867 images in 38 classes. Kaggle built it by rotating and flipping PlantVillage photos
and saving every copy, so one leaf can appear up to 15 times. `plantssl split` groups the copies of
each photo and keeps every group inside one pool. Without that, rotated copies of test leaves end
up in the training data.

**Pools.** The split is computed once and saved to `runs/split.json`. Every later command reads
that file.

| Pool | Share | Used for |
| --- | ---: | --- |
| Labeled | 5% | supervised training |
| Stream | 75% | unlabeled images for pseudo-labeling |
| Validation | 10% | picking the best epoch and round |
| Test | 10% | the final numbers, used once |

**Pseudo-labeling.** The stream arrives in 10 rounds, like new field photos would. Each round the
model labels every stream image seen so far. A label it got wrong earlier can change or drop out.
Predictions at 95% confidence or more are kept, up to each class's expected share of the images,
so classes the model finds easy cannot flood the training set. The model then fine-tunes for
3 epochs on the labeled pool plus the kept pseudo-labels. The round with the best validation
accuracy is saved. `--fresh-student` trains a new model from scratch each round instead
(teacher-student self-training, as in Noisy Student).

Why 5%: with 20% labels and a leak-free split, the supervised model already reaches about 98%
and pseudo-labels have nothing left to fix (see the 20% row in the results). The SSL papers
the thesis builds on test at 5% labels too.

**Model.** A stride-2 stem convolution, three stages of two 3x3 convolutions with batch norm and
max pooling, global average pooling and one linear layer. 1.2M parameters, 128x128 input, a
4.7 MB weights file.

## Run it

You need Rust (the pinned toolchain installs itself through `rust-toolchain.toml`). Training on
the GPU needs an NVIDIA card with CUDA drivers.

```bash
scripts/download-dataset.sh                        # about 2.7 GB into data/plantvillage
scripts/reproduce.sh                               # everything below, plus the upper bound
```

Or step by step:

```bash
cargo run --release -p plantssl -- split           # runs/split.json
cargo run --release -p plantssl -- train           # runs/supervised
cargo run --release -p plantssl -- ssl             # runs/ssl
cargo run --release -p plantssl -- eval --model runs/ssl
cargo run --release -p plantssl -- predict --model runs/ssl leaf.jpg
```

Each command has `--help`. Without an NVIDIA GPU, add `--device cpu`, or build with
`--no-default-features`. The split stores absolute dataset paths, so run `split` again after
moving the dataset.

## Desktop app

`app/` is a Tauri app with three screens:

- **Classify** loads a model folder and classifies a photo on the CPU, like an edge device would.
- **Live SSL** starts from a trained model and runs the pseudo-labeling rounds on the GPU. Validation
  accuracy and pseudo-label precision update after every round, and the run can be stopped.
- **Runs** shows the training curves, SSL rounds and test scores of every folder in `runs/`.

```bash
cd app
bun install
bun run tauri dev
```

## Repository

```text
plantssl/       library and CLI: data, model, training, SSL, evaluation
app/            Tauri 2 + Svelte 5 desktop app
scripts/        dataset download and the full experiment
BachelorProef/  thesis and school deliverables
research/       literature notes
```

The thesis in `BachelorProef/` describes the earlier version of this code, which is at commit
[`e69074c7`](https://github.com/SnaetWarre/Research_Project_Rust_Semi-Supervised_Learning/tree/e69074c7).
Its numbers came from a split that leaked augmented copies of test leaves into training, so they
are higher than the numbers above.

## Limits

These results come from one split and one seed. PlantVillage photos are lab shots of single leaves
on plain backgrounds, so accuracy on field photos with other lighting, backgrounds and cameras
will be lower.
