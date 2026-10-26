# Semi-supervised plant disease detection in Rust

Bachelor research on classifying plant diseases from leaf photos when most photos have no label.
A small CNN is trained on the 5% of images that keep their labels, then improves itself on the
other images with pseudo-labels: its own confident predictions, used as training labels. Everything
runs in Rust with [Burn](https://burn.dev) 0.22, and one model at a time fits in 6 GB of VRAM.

## Results

All rows are scored on the same test pool: 8,832 images whose leaves never appear in training or
in model selection. Validation picked every setting and checkpoint; the test pool was scored once.

| Labels | Model | Test accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| 5% (4,380 images) | Supervised only | 92.33% | 92.33% |
| 5% | + pseudo-labels, fine-tuning, 10 rounds | 93.08% | 93.23% |
| 5% | + pseudo-labels, new student, 1 round | 95.20% | 95.20% |
| 5% | + pseudo-labels, new student, 3 rounds (default) | **95.41%** | **95.41%** |
| 80% (stream labels revealed) | Upper bound | 99.47% | 99.46% |
| 20% (17,568 images) | Supervised only | 97.85% | 97.83% |
| 20% | + pseudo-labels, fine-tuning, 10 rounds | 97.85% | 97.83% |

With 5% of the labels, pseudo-labels lift test accuracy by 3.1 points and close about 43% of the
gap to a model that sees every label. Validation accuracy climbs every round: 93.00% for the
supervised start, then 95.15%, 95.75% and 96.11%. 99.6% of the 32,944 pseudo-labels in the last
round were correct. With 20% labels the supervised model is already at 97.85%, and SSL kept its starting
model: there is too little left to fix.

The thesis reports 86.06% and 94.90% for the older code. Those came from a split that leaked
rotated copies of test leaves into training, so they cannot be compared with the table above.

**On the edge.** The model is 1.2M parameters and a 4.7 MB weights file. Classifying one photo
takes 14.7 ms on a laptop CPU (AMD Ryzen 7 6800H) with Burn's pure-Rust CPU backend, no GPU and no
Python. Training runs at about 360 images per second on an RTX 3060 laptop GPU with 6 GB.

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

**Pseudo-labeling.** The stream arrives in rounds, like new field photos would. Each round:

1. The current model (the teacher) labels every stream image seen so far, so a label it got
   wrong earlier can change or drop out.
2. For every disease, the most confident half of the images predicted as that disease is kept
   (class-balanced self-training, Zou et al. 2018). One global confidence threshold does not
   work here: at 0.95 it keeps 94% of the stream images of the easiest class and 18% of the
   hardest, and training on that skewed set kept validation accuracy below the supervised
   start for 9 of 10 rounds.
3. A new model (the student) trains from scratch on the labeled pool plus the kept
   pseudo-labels, with the same 30-epoch schedule as `train`. It becomes the next round's
   teacher (teacher-student self-training, as in Noisy Student). The teacher is done once its
   predictions are in, so only one model trains at a time and every run here fit in 6 GB of VRAM.

The default is 3 rounds. The round with the best validation accuracy is saved. `--mode fine-tune`
keeps training the current model for 3 epochs per round instead. Each round is about ten times
faster, but fine-tuning gained 0.3 points on validation against 3.1 for students.

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
incremental_learning/  separate project: adding new disease classes (EWC, LwF, rehearsal)
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
