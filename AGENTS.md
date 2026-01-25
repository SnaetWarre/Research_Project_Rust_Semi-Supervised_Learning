# AGENTS.md - Plant Disease Detection Research Project

This is a Rust-based research project for semi-supervised plant disease classification
with incremental learning, targeting edge devices.

## Project Structure

```
Source/
├── plantvillage_ssl/          # Main SSL library and CLI (Burn 0.20)
│   ├── src/                   # Core Rust library
│   └── gui/                   # Tauri desktop app (Svelte 5 + TailwindCSS)
├── incremental_learning/      # Incremental learning workspace (Burn 0.14)
│   ├── crates/                # Library crates (plant-core, plant-dataset, etc.)
│   └── tools/                 # CLI tools (train, evaluate, experiment-runner)
└── pytorch_reference/         # Python/PyTorch baseline for comparison
```

## Build Commands

### Rust (plantvillage_ssl)
```bash
cd plantvillage_ssl
cargo build --release              # Build with CUDA (default)
cargo build --release --features cpu   # Build CPU-only
cargo test                         # Run all tests
cargo test test_name               # Run single test
cargo test -- --nocapture          # Run tests with output
cargo clippy                       # Lint
cargo fmt                          # Format
```

### Rust (incremental_learning workspace)
```bash
cd incremental_learning
cargo build --release              # Build all workspace members
cargo test -p plant-core           # Test single crate
cargo test -p plant-training -- test_trainer_update_epoch  # Single test
cargo run -p train -- --help       # Run train tool
cargo run -p evaluate -- --help    # Run evaluate tool
cargo run -p experiment-runner -- --help  # Run experiment runner
```

### Tauri GUI (plantvillage_ssl/gui)
```bash
cd plantvillage_ssl/gui
bun install                        # Install dependencies
bun run dev                        # Development server
bun run tauri:dev                  # Run Tauri app in dev mode
bun run tauri:build                # Build production app
bun run check                      # Svelte type checking
```

## Code Style Guidelines

### Rust

**Imports**: Group and order imports as follows:
1. Standard library (`std::`)
2. External crates (alphabetical)
3. Internal crates/modules (`crate::`, `super::`)
4. Re-exports at module level

```rust
use std::path::PathBuf;

use burn::{config::Config, module::Module};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::utils::error::{PlantVillageError, Result};
```

**Module Documentation**: Every module file starts with `//!` doc comments:
```rust
//! Training infrastructure for plant disease classification models.
//!
//! This module provides:
//! - Training loop with epoch management
//! - Loss computation and backpropagation
```

**Naming Conventions**:
- Types/Structs: `PascalCase` (e.g., `PlantClassifier`, `TrainingConfig`)
- Functions/Methods: `snake_case` (e.g., `forward_softmax`, `load_state`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `NUM_CLASSES`, `IMAGE_SIZE`)
- Generic type parameters: Single uppercase letter with trait bound (e.g., `B: Backend`)

**Error Handling**: Use `thiserror` for custom error types:
```rust
#[derive(Error, Debug)]
pub enum PlantVillageError {
    #[error("Dataset error: {0}")]
    Dataset(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, PlantVillageError>;
```

**Structs with Derive Macros**: Order derives consistently:
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState { ... }

#[derive(Config, Debug)]  // Burn configs
pub struct PlantClassifierConfig { ... }

#[derive(Module, Debug)]  // Burn modules
pub struct PlantClassifier<B: Backend> { ... }
```

**Default Implementations**: Always implement `Default` for config structs:
```rust
impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            num_epochs: 100,
            batch_size: 32,
        }
    }
}
```

**Tests**: Place tests in a `#[cfg(test)]` module at the end of each file:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_name() {
        // Test implementation
    }
}
```

### Svelte 5 (GUI)

**Component Structure**: Use Svelte 5 runes syntax:
```svelte
<script lang="ts">
  interface Props {
    title?: string;
    class?: string;
    children?: import('svelte').Snippet;
  }

  let { title, class: className = '', children }: Props = $props();
</script>

<div class="bg-background-light rounded-xl p-6 {className}">
  {#if title}
    <h3 class="text-lg font-semibold text-white mb-4">{title}</h3>
  {/if}
  {@render children?.()}
</div>
```

**Styling**: Use TailwindCSS classes inline. Custom colors defined in tailwind.config.js.

**Tauri Commands**: Backend commands in `src-tauri/src/commands/`. Use `#[tauri::command]` macro.

## Key Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| burn | 0.14/0.20 | ML framework (different versions in workspaces) |
| burn-cuda | matching | CUDA backend for GPU |
| burn-ndarray | matching | CPU backend fallback |
| image | 0.25 | Image loading/processing |
| clap | 4.x | CLI argument parsing |
| serde | 1.0 | Serialization |
| thiserror | 1.0 | Error types |
| tracing | 0.1 | Logging |
| anyhow | 1.0 | Error handling in binaries |

## Common Patterns

**Burn Backend Generics**: Models are generic over backend:
```rust
pub struct PlantClassifier<B: Backend> {
    conv1: Conv2d<B>,
    // ...
}

impl<B: Backend> PlantClassifier<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> { ... }
}
```

**CLI with Clap**: Use derive macros for CLI parsing:
```rust
#[derive(Parser, Debug)]
#[command(name = "plantvillage_ssl")]
struct Cli {
    #[arg(short, long, default_value = "false")]
    verbose: bool,
    
    #[command(subcommand)]
    command: Commands,
}
```

**Logging**: Use `tracing` macros (`info!`, `warn!`, `debug!`):
```rust
use tracing::{info, warn};

info!("Training epoch {}: loss={:.4}", epoch, loss);
warn!("No improvement. Patience: {}/{}", counter, max_patience);
```

## Testing

Run tests with output visible:
```bash
cargo test -- --nocapture
```

Run a specific test:
```bash
cargo test test_plant_classifier_output_shape -- --nocapture
```

Test a specific crate in workspace:
```bash
cargo test -p plant-incremental
```

## Notes for AI Agents

1. **Two Burn versions**: `plantvillage_ssl` uses Burn 0.20-pre, `incremental_learning` uses 0.14
2. **CUDA default**: This project targets GPU. Use `--features cpu` for CPU-only builds
3. **Image size**: Default is 128x128 or 256x256 depending on context
4. **38 classes**: PlantVillage dataset has 38 disease classes
5. **Workspace structure**: `incremental_learning/` is a Cargo workspace with multiple crates
6. **GUI uses Svelte 5**: Modern runes syntax (`$props()`, `$state()`, etc.)
7. **New Plant Diseases Dataset**: Uses pre-balanced dataset from Kaggle (~87K images)

## Dataset

The project uses the **New Plant Diseases Dataset** from Kaggle (`vipoooool/new-plant-diseases-dataset`):
- ~87,000 images (pre-balanced and augmented)
- Same 38 classes as original PlantVillage
- Pre-split into `train/` (~70K) and `valid/` (~17K) folders
- No balancing needed - dataset is already balanced (~2K images per class)

### Download Dataset
```bash
cd plantvillage_ssl
./scripts/download_dataset.sh
```

Or manually via Kaggle CLI:
```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/plantvillage/
```

### Expected Structure
```
data/plantvillage/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ... (38 classes)
└── valid/
    ├── Apple___Apple_scab/
    └── ... (38 classes)
```

The loader automatically merges train/ and valid/ folders for SSL training.

## SSL Training Pipeline - IMPORTANT!

When the user asks to train the model for SSL (semi-supervised learning), use this workflow:

### Step 1: Initial CNN Training (20% labeled data)
```bash
cd plantvillage_ssl
cargo run --release --bin plantvillage_ssl -- train \
    --epochs 30 \
    --cuda \
    --labeled-ratio 0.2
```

**Key parameters:**
- `--labeled-ratio 0.2` = 20% for CNN training, 60% reserved for SSL stream, 10% validation, 10% test
- `--epochs 30` = Sufficient for initial model (can increase if needed)
- `--cuda` = Use GPU acceleration

### Step 2: SSL Simulation (Pseudo-labeling with all unlabeled data)
```bash
cd plantvillage_ssl
cargo run --release --bin plantvillage_ssl -- simulate \
    --model "output/models/plant_classifier_TIMESTAMP" \
    --data-dir "data/plantvillage" \
    --cuda \
    --days 0 \
    --images-per-day 100 \
    --labeled-ratio 0.2 \
    --retrain-threshold 200 \
    --confidence-threshold 0.9
```

**Key parameters:**
- `--days 0` = Unlimited - process ALL available SSL stream data
- `--images-per-day 100` = Batch size per "day" for streaming simulation
- `--labeled-ratio 0.2` = MUST match training! Ensures correct data split
- `--retrain-threshold 200` = Retrain after accumulating 200 pseudo-labels
- `--confidence-threshold 0.9` = Only accept predictions with >90% confidence

### Step 3: Copy Best Model for Demo
```bash
cp plantvillage_ssl/output/simulation/plant_classifier_ssl_TIMESTAMP.mpk plantvillage_ssl/best_model.mpk
```

### Data Split Strategy

| Pool | Fraction | Purpose |
|------|----------|---------|
| Test | 10% | Final evaluation (never seen during training) |
| Validation | 10% | Hyperparameter tuning, early stopping |
| Labeled (CNN) | 20% | Initial supervised training |
| Stream (SSL) | 60% | Unlabeled data for pseudo-labeling |

### Expected Results
- Initial CNN training: ~70-75% validation accuracy (with only 20% labeled data)
- After SSL pipeline: ~78-85%+ validation accuracy
- Pseudo-label precision: >95%

### Quick Reference Commands
```bash
# Full SSL workflow (train + simulate)
cd plantvillage_ssl
cargo run --release --bin plantvillage_ssl -- train --epochs 30 --cuda --labeled-ratio 0.2
cargo run --release --bin plantvillage_ssl -- simulate \
    --model "output/models/plant_classifier_LATEST" \
    --data-dir "data/plantvillage" \
    --cuda --days 0 --labeled-ratio 0.2

# Check available commands
cargo run --release --bin plantvillage_ssl -- --help
cargo run --release --bin plantvillage_ssl -- train --help
cargo run --release --bin plantvillage_ssl -- simulate --help
```
