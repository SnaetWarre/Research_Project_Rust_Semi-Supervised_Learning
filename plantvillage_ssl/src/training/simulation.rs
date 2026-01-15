//! Semi-Supervised Learning Simulation
//!
//! This module implements the stream simulation for demonstrating
//! semi-supervised learning with pseudo-labeling.

use std::path::{Path, PathBuf};

use anyhow::Result;
use chrono::Local;
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

use crate::dataset::burn_dataset::{PlantVillageBurnDataset, PlantVillageItem};
use crate::dataset::split::{DatasetSplits, HiddenLabelImage, SplitConfig};
use crate::model::cnn::PlantClassifierConfig;
use crate::training::pseudo_label::{Prediction, PseudoLabelConfig, PseudoLabeler, StreamSimulator};
use crate::{PlantClassifier, PlantVillageBatcher, PlantVillageDataset};

/// Configuration for the simulation
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub data_dir: String,
    pub model_path: String,
    pub days: usize,
    pub images_per_day: usize,
    pub confidence_threshold: f64,
    pub retrain_threshold: usize,
    pub labeled_ratio: f64,
    pub output_dir: String,
    pub seed: u64,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub retrain_epochs: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            data_dir: "data/plantvillage".to_string(),
            model_path: "output/models/plant_classifier".to_string(),
            days: 0,  // 0 = unlimited (use all available data)
            images_per_day: 100,
            confidence_threshold: 0.9,
            retrain_threshold: 200,
            labeled_ratio: 0.2,  // 20% for CNN, 60% for SSL stream, 10% val, 10% test
            output_dir: "output/simulation".to_string(),
            seed: 42,
            batch_size: 4,  // Small batch for Jetson's 8GB shared memory
            learning_rate: 0.0001,
            retrain_epochs: 5,
        }
    }
}

/// Run the semi-supervised learning simulation
pub fn run_simulation<B>(config: SimulationConfig) -> Result<SimulationResults>
where
    B: AutodiffBackend,
{
    println!("{}", "Starting Semi-Supervised Learning Simulation...".green().bold());
    println!();

    let device = B::Device::default();
    println!("  Device: {:?}", device);

    std::fs::create_dir_all(&config.output_dir)?;

    // Load the dataset
    println!("{}", "Loading Dataset...".cyan());
    let dataset = PlantVillageDataset::new(&config.data_dir)?;
    let stats = dataset.get_stats();

    if stats.total_samples == 0 {
        anyhow::bail!("No images found in dataset directory!");
    }

    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);

    // Create stratified splits using CLI-configured labeled_ratio
    println!("{}", "Creating Data Splits...".cyan());
    let all_images: Vec<(PathBuf, usize, String)> = dataset
        .samples
        .iter()
        .map(|s| (s.path.clone(), s.label, s.class_name.clone()))
        .collect();

    // Calculate stream fraction: remaining after test (10%), validation (10%), and labeled
    let stream_fraction = 1.0 - config.labeled_ratio - 0.10; // Reserve 10% for future pool
    
    let split_config = SplitConfig {
        test_fraction: 0.10,
        validation_fraction: 0.10,
        labeled_fraction: config.labeled_ratio,  // Use CLI-configured ratio!
        stream_fraction,                         // Maximize for SSL!
        seed: config.seed,
        stratified: true,
    };

    let splits = DatasetSplits::from_images(all_images, split_config)
        .map_err(|e| anyhow::anyhow!("Failed to create splits: {:?}", e))?;

    let split_stats = splits.stats();
    println!("  Test set: {} samples", split_stats.test_size);
    println!("  Validation set: {} samples", split_stats.validation_size);
    println!("  Labeled pool (CNN): {} samples", split_stats.labeled_pool_size);
    println!("  Stream pool (SSL): {} samples", split_stats.stream_pool_size);

    // Use stream pool directly for SSL (no separate future pool anymore)
    let unlabeled_pool = splits.stream_pool.clone();
    println!("  Total unlabeled for SSL: {} samples", unlabeled_pool.len());

    // Load or create model
    println!("{}", "Loading Model...".cyan());
    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,  // Moderate dropout
        in_channels: 3,
        base_filters: 32,   // Proper sized model
    };

    let mut model: PlantClassifier<B> = PlantClassifier::new(&model_config, &device);

    // Try to load checkpoint
    // CompactRecorder adds .mpk extension, so check both with and without
    let model_path = Path::new(&config.model_path);
    let model_path_mpk = PathBuf::from(format!("{}.mpk", config.model_path));
    
    let actual_model_path = if model_path_mpk.exists() {
        Some(model_path.to_path_buf()) // Use path without extension for load_file (Burn adds it)
    } else if model_path.exists() {
        Some(model_path.to_path_buf())
    } else {
        None
    };
    
    if let Some(load_path) = actual_model_path {
        println!("  Loading checkpoint from: {:?}", load_path);
        let recorder = CompactRecorder::new();
        model = model
            .load_file(&load_path, &recorder, &device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;
        println!("  ✅ Checkpoint loaded");
    } else {
        println!("  ⚠️  No checkpoint found at {:?}", model_path);
        println!("  Tip: Train a model first with: plantvillage_ssl train");
    }

    // Create optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    // Create stream simulator with combined unlabeled pool
    println!("{}", "Initializing Stream Simulator...".cyan());
    let mut stream = StreamSimulator::new(
        unlabeled_pool,
        config.seed,
        config.images_per_day,
    );
    println!("  Images per day: {}", config.images_per_day);
    println!("  Total unlabeled images: {}", stream.total());

    // Create pseudo-labeler
    let pseudo_config = PseudoLabelConfig {
        confidence_threshold: config.confidence_threshold,
        max_per_class: Some(500),
        retrain_threshold: config.retrain_threshold,
        curriculum_learning: false,
        ..Default::default()
    };
    let mut pseudo_labeler = PseudoLabeler::new(pseudo_config);

    // Create batcher for inference
    let batcher = PlantVillageBatcher::<B::InnerBackend>::with_image_size(
        <B::InnerBackend as burn::tensor::backend::Backend>::Device::default(),
        128,
    );

    // Prepare validation data
    let val_samples: Vec<(PathBuf, usize)> = splits
        .validation_set
        .iter()
        .map(|img| (img.image_path.clone(), img.label))
        .collect();
    let val_dataset = PlantVillageBurnDataset::new_cached(val_samples.clone(), 128)
        .expect("Failed to load validation dataset");

    // Track results
    let mut results = SimulationResults::new();

    // Initial validation accuracy
    let initial_acc = evaluate_model(&model, &val_dataset, &batcher, config.batch_size);
    println!();
    println!("{}", "Initial Model Performance:".yellow().bold());
    println!("  Validation accuracy: {:.2}%", initial_acc);
    results.accuracy_history.push((0, initial_acc));

    // Run simulation
    println!();
    println!("{}", "Running Simulation...".green().bold());
    println!();

    let mut day = 0;
    let mut total_pseudo_labels = 0;
    let mut retraining_count = 0;

    while let Some(daily_images) = stream.next_day() {
        day += 1;
        pseudo_labeler.set_day(day);

        println!(
            "{}",
            format!("Day {}: Processing {} images...", day, daily_images.len()).cyan()
        );

        // Run inference on daily images
        let predictions = run_inference_batch(&model, &daily_images, &batcher, &device, 128);

        // Process predictions for pseudo-labeling
        let new_pseudo_labels = pseudo_labeler.process_predictions(&predictions, &daily_images);

        let accepted = new_pseudo_labels.len();
        total_pseudo_labels += accepted;

        let stats = pseudo_labeler.stats();
        println!(
            "  → Accepted: {} new pseudo-labels (precision: {:.1}%)",
            accepted,
            stats.accuracy() * 100.0
        );

        // Check if we should retrain
        if pseudo_labeler.should_retrain() {
            retraining_count += 1;
            println!();
            println!(
                "{}",
                format!("Retraining #{} (accumulated {} pseudo-labels)", retraining_count, pseudo_labeler.num_pseudo_labels())
                    .yellow()
                    .bold()
            );

            // Get pseudo-labels and convert to training data
            let pseudo_labels = pseudo_labeler.get_and_clear_pseudo_labels();
            let pseudo_samples: Vec<(PathBuf, usize)> = pseudo_labels
                .iter()
                .map(|p| (p.image_path.clone(), p.predicted_label))
                .collect();

            // Combine with labeled pool
            let labeled_samples: Vec<(PathBuf, usize)> = splits
                .labeled_pool
                .iter()
                .map(|img| (img.image_path.clone(), img.label))
                .collect();

            let mut combined_samples = labeled_samples;
            combined_samples.extend(pseudo_samples);

            println!(
                "  Training on {} samples ({} labeled + {} pseudo)",
                combined_samples.len(),
                splits.labeled_pool.len(),
                pseudo_labels.len()
            );

            // Create combined dataset - use NON-cached to avoid OOM on Jetson
            // The validation set is already cached, so we can't cache training too
            let combined_dataset = PlantVillageBurnDataset::new(combined_samples, 128);

            // Retrain model
            model = retrain_model(
                model,
                &mut optimizer,
                &combined_dataset,
                config.retrain_epochs,
                config.batch_size,
                config.learning_rate,
                config.seed + day as u64,
            );

            // Evaluate after retraining
            let new_acc = evaluate_model(&model, &val_dataset, &batcher, config.batch_size);
            println!("  → New validation accuracy: {:.2}%", new_acc);
            results.accuracy_history.push((day, new_acc));

            println!();
        }

        // Stop if we've simulated enough days (0 = unlimited, process all data)
        if config.days > 0 && day >= config.days {
            break;
        }
    }

    // Final evaluation
    println!();
    println!("{}", "Simulation Complete!".green().bold());
    println!();

    let final_acc = evaluate_model(&model, &val_dataset, &batcher, config.batch_size);
    let stats = pseudo_labeler.stats();

    println!("{}", "Final Results:".cyan().bold());
    println!("  Days simulated: {}", day);
    println!("  Total images processed: {}", stats.total_processed);
    println!("  Total pseudo-labels generated: {}", total_pseudo_labels);
    println!("  Pseudo-label precision: {:.1}%", stats.accuracy() * 100.0);
    println!("  Retraining sessions: {}", retraining_count);
    println!();
    println!("  Initial accuracy: {:.2}%", initial_acc);
    println!("  Final accuracy: {:.2}%", final_acc);
    println!(
        "  Improvement: {:.2}%",
        final_acc - initial_acc
    );

    results.days_simulated = day;
    results.total_pseudo_labels = total_pseudo_labels;
    results.pseudo_label_precision = stats.accuracy();
    results.retraining_count = retraining_count;
    results.initial_accuracy = initial_acc;
    results.final_accuracy = final_acc;

    // Save final model with timestamp
    let timestamp = Local::now().format("%Y%m%d_%H%M%S");
    let final_model_path = PathBuf::from(&config.output_dir).join(format!("plant_classifier_ssl_{}", timestamp));
    println!();
    println!("  Saving model to: {:?}", final_model_path);
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(&final_model_path, &recorder)
        .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

    Ok(results)
}

/// Run inference on a batch of hidden-label images
/// 
/// Note: Processes in smaller batches to avoid cubecl-runtime memory management issues
fn run_inference_batch<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    images: &[HiddenLabelImage],
    batcher: &PlantVillageBatcher<B::InnerBackend>,
    _device: &B::Device,
    image_size: usize,
) -> Vec<Prediction> {
    const INFERENCE_BATCH_SIZE: usize = 32;
    let inner_model = model.clone().valid();
    let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
    let mut predictions = Vec::new();
    let num_classes = 38;

    // Process in smaller batches to avoid memory issues in cubecl-runtime
    for chunk_start in (0..images.len()).step_by(INFERENCE_BATCH_SIZE) {
        let chunk_end = (chunk_start + INFERENCE_BATCH_SIZE).min(images.len());
        let chunk_images = &images[chunk_start..chunk_end];

        // Load images for this chunk, keeping track of which ones succeeded
        let mut loaded_items: Vec<(usize, PlantVillageItem)> = Vec::new();
        for (idx, hidden) in chunk_images.iter().enumerate() {
            if let Ok(item) = PlantVillageItem::from_path(&hidden.image_path, hidden.hidden_label, image_size) {
                loaded_items.push((idx, item));
            }
        }

        if loaded_items.is_empty() {
            continue;
        }

        let items: Vec<PlantVillageItem> = loaded_items.iter().map(|(_, item)| item.clone()).collect();

        // Create batch and run inference
        let batch = batcher.batch(items, &inner_device);
        let output = inner_model.forward_softmax(batch.images);
        let output_data = output.into_data();
        let probs: Vec<f32> = output_data.to_vec().unwrap();

        // Extract predictions for this chunk
        for (i, (chunk_idx, _)) in loaded_items.iter().enumerate() {
            let hidden = &chunk_images[*chunk_idx];
            
            let start = i * num_classes;
            let end = start + num_classes;
            
            if end > probs.len() {
                break;
            }
            
            let item_probs: Vec<f32> = probs[start..end].to_vec();

            // Find max
            let (predicted_label, confidence) = item_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, &conf)| (idx, conf))
                .unwrap_or((0, 0.0));

            predictions.push(Prediction {
                image_path: hidden.image_path.clone(),
                predicted_label,
                confidence,
                probabilities: item_probs,
                image_id: hidden.image_id,
                ground_truth: Some(hidden.hidden_label),
            });
        }
    }

    predictions
}

/// Evaluate model on validation set
fn evaluate_model<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    dataset: &PlantVillageBurnDataset,
    batcher: &PlantVillageBatcher<B::InnerBackend>,
    batch_size: usize,
) -> f64 {
    use burn::data::dataset::Dataset;

    let inner_model = model.clone().valid();
    let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
    let len = dataset.len();
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch = batcher.batch(items, &inner_device);
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

/// Retrain model with combined labeled and pseudo-labeled data
fn retrain_model<B: AutodiffBackend>(
    mut model: PlantClassifier<B>,
    optimizer: &mut burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, PlantClassifier<B>, B>,
    dataset: &PlantVillageBurnDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
) -> PlantClassifier<B> {
    use burn::data::dataset::Dataset;

    let device = B::Device::default();
    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), 128);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for epoch in 0..epochs {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for start in (0..len).step_by(batch_size) {
            let end = (start + batch_size).min(len);
            let batch_indices = &indices[start..end];
            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);

            // Forward pass
            let output = model.forward(batch.images.clone());

            // Compute loss
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            batch_count += 1;

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);

            // Update model
            model = optimizer.step(learning_rate, model, grads);
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        println!("    Epoch {}/{}: avg_loss = {:.4}", epoch + 1, epochs, avg_loss);
    }

    model
}

/// Results from the simulation
#[derive(Debug, Clone)]
pub struct SimulationResults {
    pub days_simulated: usize,
    pub total_pseudo_labels: usize,
    pub pseudo_label_precision: f64,
    pub retraining_count: usize,
    pub initial_accuracy: f64,
    pub final_accuracy: f64,
    pub accuracy_history: Vec<(usize, f64)>,
}

impl SimulationResults {
    pub fn new() -> Self {
        Self {
            days_simulated: 0,
            total_pseudo_labels: 0,
            pseudo_label_precision: 0.0,
            retraining_count: 0,
            initial_accuracy: 0.0,
            final_accuracy: 0.0,
            accuracy_history: Vec::new(),
        }
    }

    pub fn improvement(&self) -> f64 {
        self.final_accuracy - self.initial_accuracy
    }
}

impl Default for SimulationResults {
    fn default() -> Self {
        Self::new()
    }
}

