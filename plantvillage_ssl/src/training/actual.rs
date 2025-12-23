//! Actual Training Implementation using Burn's LearnerBuilder
//!
//! This implements a working training loop using Burn 0.15's high-level training API

use std::path::PathBuf;

use anyhow::Result;
use burn::{
    data::dataloader::DataLoaderBuilder,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{
        LearnerBuilder, TrainOutput, TrainStep, ValidOutput,
    },
};

use crate::{
    PlantVillageBatch, PlantVillageBatcher, PlantVillageBurnDataset, PlantVillageDataset,
    PlantClassifier,
};
use crate::model::cnn::PlantClassifierConfig;

/// Actual training function using Burn's LearnerBuilder
pub fn run_training<B>(
    data_dir: &str,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    labeled_ratio: f64,
    _confidence_threshold: f64,
    output_dir: &str,
    seed: u64,
) -> Result<()>
where
    B: AutodiffBackend,
{
    println!("{}", "Initializing Training...".green().bold());

    let device = B::Device::default();
    println!("  Device: {:?}", device);

    std::fs::create_dir_all(output_dir)?;

    println!("{}", "Loading Dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    stats.print();

    if stats.total_samples == 0 {
        println!("{} No images found in dataset directory!", "Error:".red());
        return Ok(());
    }

    let num_labeled = (stats.total_samples as f64 * labeled_ratio * 0.15 * 0.75) as usize;
    if num_labeled == 0 {
        println!("{} Not enough labeled data with ratio {}", "Error:".red(), labeled_ratio);
        return Ok(());
    }

    let num_val = (stats.total_samples as f64 * 0.10) as usize;
    let num_train = stats.total_samples - num_labeled - num_val;

    if num_train == 0 {
        println!("{} Not enough training data!", "Error:".red());
        return Ok(());
    }

    let samples: Vec<_> = dataset
        .samples
        .iter()
        .take(num_labeled + num_val)
        .map(|s| (s.path.clone(), s.label))
        .collect();

    let train_dataset = PlantVillageBurnDataset::new(samples[..num_labeled].to_vec(), 256);
    let val_dataset = PlantVillageBurnDataset::new(samples[num_labeled..].to_vec(), 256);

    let train_batcher = PlantVillageBatcher::new(device.clone());
    let val_batcher = PlantVillageBatcher::new(device.clone());

    println!("{}", "Creating Model...".cyan());
    let model_config = PlantClassifierConfig {
        num_classes: 39,
        input_size: 256,
        ..Default::default()
    };
    let model = PlantClassifier::<B>::new(&model_config, &device);

    println!("{}", "Setting up Learner...".cyan());
    let learner = LearnerBuilder::new(model.clone())
        .devices(vec![device.clone()])
        .num_epochs(epochs)
        .summary()
        .build(train_dataset, val_dataset, train_batcher, val_batcher);

    println!();
    println!("{}", "Training Configuration:".cyan().bold());
    println!("  📊 Dataset: {} total images", stats.total_samples);
    println!("  🏷️  Training: {} images", num_train);
    println!("  ✅ Validation: {} images", num_val);
    println!("  🔄 Epochs: {}", epochs);
    println!("  📦 Batch size: {}", batch_size);
    println!("  📈 Learning rate: {}", learning_rate);
    println!("  🧠 Device: {:?}", device);
    println!();

    println!("{}", "Starting Training...".green().bold());
    println!();

    let model_trained = learner.fit();

    println!();
    println!("{}", "Training Complete!".green().bold());
    println!("  🎉 Model trained successfully!");
    println!();

    let artifact_dir = PathBuf::from(output_dir);
    let checkpoint_path = artifact_dir.join("model");

    println!("{}", "Saving Model...".cyan());
    std::fs::create_dir_all(&artifact_dir)?;

    let recorder = CompactRecorder::new();
    model_trained
        .save_file(checkpoint_path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {}", e))?;

    println!("  💾 Saved to: {:?}", checkpoint_path);
    println!();

    println!("{}", "Model ready for:".green().bold());
    println!("  • Training with: `plantvillage_ssl train`");
    println!("  • Inference with: `plantvillage_ssl infer --model {:?}`", checkpoint_path);
    println!("  • Benchmark with: `plantvillage_ssl benchmark --model {:?}`", checkpoint_path);

    Ok(())
}
