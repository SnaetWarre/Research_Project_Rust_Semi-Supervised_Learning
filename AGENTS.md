# AGENTS.md

Semi-supervised plant disease classification in Rust with Burn 0.22. The research question is how
much pseudo-labeling recovers when only 20% of the images keep their labels, on hardware with
6 GB of VRAM.

## Layout

```
plantssl/           Library and `plantssl` CLI
  src/data.rs       Dataset scan and the leaf-grouped split
  src/images.rs     Decoding, augmentation, prefetching batch loader
  src/model.rs      PlantCnn and its save/load (model.bpk + model.json)
  src/train.rs      Supervised loop (AdamW, warmup + cosine, best epoch on validation)
  src/ssl.rs        Self-training rounds with confidence-thresholded pseudo-labels
  src/eval.rs       Batch prediction, accuracy, macro F1, confusion matrix
  src/predict.rs    Single-image classification, used by the app
app/                Tauri 2 desktop app (Svelte 5, Tailwind 4, Vite)
scripts/            Dataset download and the full experiment
BachelorProef/      Thesis and school deliverables (describe the older code at commit e69074c7)
research/           Literature notes
```

## Commands

```bash
scripts/download-dataset.sh               # data/plantvillage/{train,valid}
cargo run --release -p plantssl -- split              # runs/split.json
cargo run --release -p plantssl -- train              # runs/supervised
cargo run --release -p plantssl -- ssl                # runs/ssl
cargo run --release -p plantssl -- eval --model runs/ssl
scripts/reproduce.sh                      # all of the above plus the upper bound

cargo test                                # unit tests (CPU)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all

cd app && bun install && bun run tauri dev   # desktop app
cd app && bun run check                      # Svelte type check
```

CUDA is the default feature. Pass `--device cpu` to run on the CPU, or build with
`--no-default-features` on machines without CUDA. CI does the latter.

## Rules that protect the results

1. Every command reads the same `runs/split.json`. Never recompute a split inside `train`, `ssl`
   or `eval`.
2. The split groups augmented copies of one photo (`_90deg`, `_flipLR`, `_new30degFlipLR` and so
   on) by `data::source_id`. Keep that, or test accuracy measures memorized leaves.
3. In `ssl.rs`, stream labels may only be read to report pseudo-label precision. Training uses
   the predicted label.
4. Model selection uses the validation pool. The test pool is for `eval` only.
5. Do not use label smoothing: it caps softmax confidence and breaks the 0.95 threshold.

## Code style

- Rust 2024 edition, toolchain pinned in `rust-toolchain.toml`.
- `anyhow::Result` throughout; add context to file errors with `.with_context`.
- Every module starts with a `//!` comment saying what it is for.
- Tests sit in a `#[cfg(test)] mod tests` at the end of each file and run on `Device::flex()`.
- Burn 0.22 has no backend type parameter: models take `&Device`, `device.autodiff()` enables
  gradients, `model.valid()` gives an inference copy.
- Keep Burn's default features. Without them CUDA loses kernel fusion and autotuning and
  training is about 100 times slower.
- Svelte 5 runes (`$state`, `$derived`, `$props`), Tailwind classes inline, theme colours in
  `app/src/app.css`.
