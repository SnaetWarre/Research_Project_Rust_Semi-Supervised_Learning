//! Supervised Training Implementation
//!
//! This implements a working training loop using Burn 0.15's API directly
//! with a simple, custom training loop rather than the high-level LearnerBuilder.
//!
//! ## Features
//!
//! - On-the-fly data augmentation for better generalization
//! - Early stopping at target validation accuracy (leaves room for SSL improvement)

use chrono::Local;
use std::path::PathBuf;

use anyhow::Result;
use burn::{
    data::dataloader::batcher::Batcher,
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, ElementConversion},
};
use colored::Colorize;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::dataset::burn_dataset::{AugmentingBatcher, RawPlantVillageDataset};
use crate::dataset::split::{DatasetSplits, SplitConfig};
use crate::model::cnn::PlantClassifierConfig;
use crate::{PlantClassifier, PlantVillageBatcher, PlantVillageBurnDataset, PlantVillageDataset};

/// Configuration for early stopping based on validation accuracy
#[derive(Clone, Debug)]
pub struct EarlyStoppingConfig {
    /// Target validation accuracy to stop at (e.g., 0.88 for 88%)
    pub target_accuracy: f64,
    /// Number of epochs to maintain target accuracy before stopping
    pub patience: usize,
    /// Whether early stopping is enabled
    pub enabled: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            target_accuracy: 0.88, // 88% - leaves room for SSL to improve
            patience: 3,           // Stop after 3 epochs at target
            enabled: true,
        }
    }
}

impl EarlyStoppingConfig {
    /// Disable early stopping
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Create with custom target accuracy
    pub fn with_target(target_accuracy: f64, patience: usize) -> Self {
        Self {
            target_accuracy,
            patience,
            enabled: true,
        }
    }
}

/// Run training with the given configuration
///
/// # Type Parameters
/// * `B` - The autodiff backend to use (e.g., `Autodiff<NdArray>` or `Autodiff<Cuda>`)
///
/// # Arguments
/// * `data_dir` - Path to the PlantVillage dataset directory
/// * `epochs` - Number of training epochs
/// * `batch_size` - Batch size for training
/// * `learning_rate` - Initial learning rate
/// * `labeled_ratio` - Ratio of labeled data to use (0.0-1.0)
/// * `_confidence_threshold` - Threshold for pseudo-labeling (not used in supervised training)
/// * `output_dir` - Directory to save model checkpoints
/// * `seed` - Random seed for reproducibility
/// * `max_samples` - Maximum samples to use (for quick testing)
/// * `use_augmentation` - Enable data augmentation during training
/// * `early_stopping` - Configuration for early stopping at target accuracy
pub fn run_training<B>(
    data_dir: &str,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    labeled_ratio: f64,
    _confidence_threshold: f64,
    output_dir: &str,
    seed: u64,
    max_samples: Option<usize>,
    use_augmentation: bool,
    early_stopping: Option<EarlyStoppingConfig>,
) -> Result<()>
where
    B: AutodiffBackend,
{
    println!("{}", "Initializing Training...".green().bold());

    let device = B::Device::default();
    println!("  Device: {:?}", device);

    std::fs::create_dir_all(output_dir)?;

    // Load the dataset
    println!("{}", "Loading Dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    stats.print();

    if stats.total_samples == 0 {
        println!("{} No images found in dataset directory!", "Error:".red());
        println!();
        println!("Please download the PlantVillage dataset first:");
        println!("  1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset");
        println!("  2. Download and extract to: {}", data_dir);
        println!(
            "  3. Ensure directory structure: {}/{{class_name}}/*.jpg",
            data_dir
        );
        return Ok(());
    }

    // Use stratified splitting for proper train/val distribution
    println!("{}", "Creating Stratified Data Splits...".cyan());

    // Prepare all images with class names for stratified splitting
    let all_images: Vec<(PathBuf, usize, String)> = dataset
        .samples
        .iter()
        .map(|s| (s.path.clone(), s.label, s.class_name.clone()))
        .collect();

    // Limit samples if max_samples is specified
    let images_to_use: Vec<(PathBuf, usize, String)> = match max_samples {
        Some(max) => {
            // Shuffle first, then take max samples to maintain class diversity
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut shuffled = all_images.clone();
            shuffled.shuffle(&mut rng);
            shuffled.into_iter().take(max).collect()
        }
        None => all_images,
    };

    // Create stratified splits using the proper splitting logic
    let split_config = SplitConfig {
        test_fraction: 0.0, // We'll use validation as our held-out set for now
        validation_fraction: 0.10,
        labeled_fraction: labeled_ratio,
        stream_fraction: 1.0 - labeled_ratio, // Remaining for stream pool
        seed,
        stratified: true,
    };

    let splits = DatasetSplits::from_images(images_to_use, split_config)
        .map_err(|e| anyhow::anyhow!("Failed to create splits: {:?}", e))?;

    let split_stats = splits.stats();
    println!("  📊 Stratified split created:");
    println!("    Classes represented: {}", split_stats.num_classes);
    println!(
        "    Labeled pool: {} samples",
        split_stats.labeled_pool_size
    );
    println!(
        "    Validation set: {} samples",
        split_stats.validation_size
    );
    println!(
        "    Stream pool: {} samples (for SSL)",
        split_stats.stream_pool_size
    );

    // Convert splits to samples format
    let train_samples: Vec<(PathBuf, usize)> = splits
        .labeled_pool
        .iter()
        .map(|img| (img.image_path.clone(), img.label))
        .collect();

    let val_samples: Vec<(PathBuf, usize)> = splits
        .validation_set
        .iter()
        .map(|img| (img.image_path.clone(), img.label))
        .collect();

    if train_samples.len() < batch_size {
        println!(
            "{} Not enough labeled data ({}) for batch size {}",
            "Error:".red(),
            train_samples.len(),
            batch_size
        );
        return Ok(());
    }

    println!();
    println!("{}", "Dataset Splits:".cyan().bold());
    println!("  🏷️  Training samples:   {}", train_samples.len());
    println!("  ✅ Validation samples: {}", val_samples.len());

    // Create datasets and batcher
    // Use 128x128 images for better GPU memory usage on 6GB cards
    let image_size = 128;

    // Determine augmentation settings
    let early_stop_config = early_stopping.unwrap_or_else(EarlyStoppingConfig::default);

    println!();
    if use_augmentation {
        println!(
            "{}",
            "Pre-loading Training Data (with augmentation)..."
                .cyan()
                .bold()
        );
    } else {
        println!("{}", "Pre-loading Training Data...".cyan().bold());
    }

    // For augmentation, we need raw images; otherwise use preprocessed cache
    let train_dataset_raw = if use_augmentation {
        Some(
            RawPlantVillageDataset::new_cached(train_samples.clone())
                .expect("Failed to load raw training dataset"),
        )
    } else {
        None
    };

    let train_dataset = if !use_augmentation {
        Some(
            PlantVillageBurnDataset::new_cached(train_samples.clone(), image_size)
                .expect("Failed to load training dataset"),
        )
    } else {
        None
    };

    println!("{}", "Pre-loading Validation Data...".cyan().bold());
    let val_dataset = PlantVillageBurnDataset::new_cached(val_samples.clone(), image_size)
        .expect("Failed to load validation dataset");

    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);
    let aug_batcher = if use_augmentation {
        Some(AugmentingBatcher::<B>::new(
            device.clone(),
            image_size,
            seed,
        ))
    } else {
        None
    };

    // Create model
    println!();
    println!("{}", "Creating Model...".cyan());
    let model_config = PlantClassifierConfig {
        num_classes: 38, // PlantVillage has 38 classes
        input_size: image_size,
        dropout_rate: 0.3, // Moderate dropout
        in_channels: 3,
        base_filters: 32, // Proper sized model for good accuracy
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);

    // Create optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    // Print training config
    let total_samples = train_samples.len() + val_samples.len();
    println!();
    println!("{}", "Training Configuration:".cyan().bold());
    println!("  📊 Total samples:     {}", total_samples);
    println!("  🏷️  Training samples:  {}", train_samples.len());
    println!("  ✅ Validation samples: {}", val_samples.len());
    println!("  🔄 Epochs:            {}", epochs);
    println!("  📦 Batch size:        {}", batch_size);
    println!("  📈 Learning rate:     {}", learning_rate);
    println!(
        "  🎨 Augmentation:      {}",
        if use_augmentation {
            "enabled"
        } else {
            "disabled"
        }
    );
    if early_stop_config.enabled {
        println!(
            "  🛑 Early stopping:    at {:.0}% val acc ({} epochs)",
            early_stop_config.target_accuracy * 100.0,
            early_stop_config.patience
        );
    } else {
        println!("  🛑 Early stopping:    disabled");
    }
    println!("  🧠 Device:            {:?}", device);
    println!();

    println!("{}", "Starting Training...".green().bold());
    println!();

    let mut best_val_acc = 0.0f64;
    let mut epochs_at_target = 0usize; // Counter for early stopping

    // Create RNG for epoch shuffling
    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    // Get training dataset length for shuffling
    let train_len = if use_augmentation {
        use burn::data::dataset::Dataset;
        train_dataset_raw.as_ref().unwrap().len()
    } else {
        use burn::data::dataset::Dataset;
        train_dataset.as_ref().unwrap().len()
    };

    // Training loop
    for epoch in 0..epochs {
        println!(
            "{}",
            format!("Epoch {}/{}", epoch + 1, epochs).yellow().bold()
        );

        // Training phase
        let mut epoch_loss = 0.0f64;
        let mut correct = 0usize;
        let mut total_samples_count = 0usize;

        // Create shuffled indices for this epoch
        let mut shuffled_indices: Vec<usize> = (0..train_len).collect();
        shuffled_indices.shuffle(&mut epoch_rng);
        let num_batches = (shuffled_indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            // Create batch on-demand (lazy batching to avoid GPU OOM)
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(shuffled_indices.len());
            let batch_indices = &shuffled_indices[start..end];

            // Create batch based on whether augmentation is enabled
            let batch = if use_augmentation {
                use burn::data::dataset::Dataset;
                let items: Vec<_> = batch_indices
                    .iter()
                    .filter_map(|&i| train_dataset_raw.as_ref().unwrap().get(i))
                    .collect();

                if items.is_empty() {
                    continue;
                }

                aug_batcher.as_ref().unwrap().batch(items, &device)
            } else {
                use burn::data::dataset::Dataset;
                let items: Vec<_> = batch_indices
                    .iter()
                    .filter_map(|&i| train_dataset.as_ref().unwrap().get(i))
                    .collect();

                if items.is_empty() {
                    continue;
                }

                batcher.batch(items, &device)
            };

            // Forward pass
            let output = model.forward(batch.images.clone());

            // Compute cross-entropy loss
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output.clone(), batch.targets.clone());

            let loss_value: f64 = loss.clone().into_scalar().elem();

            // CRITICAL: Detect NaN/Inf immediately to avoid wasting training time
            if loss_value.is_nan() {
                anyhow::bail!(
                    "Loss became NaN at epoch {} batch {}. Training aborted. \
                    This usually indicates:\n  \
                    - Learning rate too high\n  \
                    - Gradient explosion\n  \
                    - Corrupted input data\n  \
                    Try reducing learning rate or adding gradient clipping.",
                    epoch + 1,
                    batch_idx + 1
                );
            }
            if loss_value.is_infinite() {
                anyhow::bail!(
                    "Loss became infinite at epoch {} batch {}. Training aborted. \
                    This usually indicates:\n  \
                    - Learning rate too high\n  \
                    - Numerical instability in loss calculation\n  \
                    Try reducing learning rate.",
                    epoch + 1,
                    batch_idx + 1
                );
            }

            epoch_loss += loss_value;

            // Calculate batch accuracy
            let predictions = output.argmax(1).squeeze::<1>();
            let batch_correct: i64 = predictions
                .equal(batch.targets.clone())
                .int()
                .sum()
                .into_scalar()
                .elem();
            correct += batch_correct as usize;
            total_samples_count += batch.targets.dims()[0];

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Update model
            model = optimizer.step(learning_rate, model, grads);

            // Progress logging
            if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
                let running_acc = 100.0 * correct as f64 / total_samples_count as f64;
                println!(
                    "  Batch {:>4}/{}: loss = {:.4}, acc = {:.2}%",
                    batch_idx + 1,
                    num_batches,
                    loss_value,
                    running_acc
                );
            }

            // Drop batch explicitly to free GPU memory before next iteration
            drop(batch);
        }

        let avg_loss = epoch_loss / num_batches.max(1) as f64;
        let train_acc = 100.0 * correct as f64 / total_samples_count.max(1) as f64;

        // Validation phase
        let val_acc = evaluate(&model, &val_dataset, &batcher, batch_size, image_size);

        // Track best model
        let is_best = val_acc > best_val_acc;
        if is_best {
            best_val_acc = val_acc;
        }

        // Check for early stopping at target accuracy
        let val_acc_ratio = val_acc / 100.0;
        let mut early_stop_triggered = false;

        if early_stop_config.enabled && val_acc_ratio >= early_stop_config.target_accuracy {
            epochs_at_target += 1;
            if epochs_at_target >= early_stop_config.patience {
                early_stop_triggered = true;
            }
        } else {
            epochs_at_target = 0; // Reset counter if we drop below target
        }

        // Build status string
        let mut status_parts = Vec::new();
        if is_best {
            status_parts.push(" (best)".green().to_string());
        }
        if early_stop_config.enabled && val_acc_ratio >= early_stop_config.target_accuracy {
            status_parts.push(
                format!(
                    " [target: {}/{}]",
                    epochs_at_target, early_stop_config.patience
                )
                .yellow()
                .to_string(),
            );
        }

        println!(
            "  {} Loss: {:.4} | Train Acc: {:.2}% | Val Acc: {:.2}%{}",
            "→".cyan(),
            avg_loss,
            train_acc,
            val_acc,
            status_parts.join("")
        );
        println!();

        // Early stopping
        if early_stop_triggered {
            println!(
                "{}",
                format!(
                    "🛑 Early stopping: reached {:.1}% validation accuracy for {} epochs",
                    early_stop_config.target_accuracy * 100.0,
                    early_stop_config.patience
                )
                .yellow()
                .bold()
            );
            println!("   Leaving room for SSL to improve the model!");
            println!();
            break;
        }
    }

    // Save the model with timestamp
    let artifact_dir = PathBuf::from(output_dir);
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let model_name = format!("plant_classifier_{}", timestamp);
    let checkpoint_path = artifact_dir.join(&model_name);

    println!("{}", "Saving Model...".cyan());
    std::fs::create_dir_all(&artifact_dir)?;

    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(&checkpoint_path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

    println!("  💾 Saved to: {:?}", checkpoint_path);
    println!();

    println!("{}", "Training Complete!".green().bold());
    println!("  🎉 Best validation accuracy: {:.2}%", best_val_acc);
    println!();

    println!("{}", "Next steps:".cyan().bold());
    println!(
        "  • Run inference: plantvillage_ssl infer --model {:?} --input <image>",
        checkpoint_path
    );
    println!(
        "  • Benchmark: plantvillage_ssl benchmark --model {:?}",
        checkpoint_path
    );

    Ok(())
}

/// Evaluate the model on a dataset
fn evaluate<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    dataset: &PlantVillageBurnDataset,
    _batcher: &PlantVillageBatcher<B>,
    batch_size: usize,
    image_size: usize,
) -> f64 {
    use burn::data::dataset::Dataset;
    use burn::tensor::backend::Backend;

    // Get the inner backend's device
    let device = <B::InnerBackend as Backend>::Device::default();

    // Create a batcher for the inner backend with correct image size
    let inner_batcher =
        PlantVillageBatcher::<B::InnerBackend>::with_image_size(device.clone(), image_size);

    let inner_model = model.clone().valid();
    let len = dataset.len();
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch = inner_batcher.batch(items, &device);
        let output = inner_model.forward(batch.images);
        let predictions = output.argmax(1).squeeze::<1>();

        let batch_correct: i64 = predictions
            .equal(batch.targets)
            .int()
            .sum()
            .into_scalar()
            .elem();

        correct += batch_correct as usize;
        total += end - start;
    }

    if total == 0 {
        0.0
    } else {
        100.0 * correct as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_training_compiles() {
        // This test just ensures the module compiles correctly
        // Actual training tests require a dataset
        assert!(true);
    }
}
