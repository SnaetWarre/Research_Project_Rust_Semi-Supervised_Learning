//! Shared training routine for the incremental experiments.
//!
//! Trains a base model, adds new classes, fine-tunes, and measures
//! forgetting and new-class accuracy. Used by the class scaling and new
//! class position experiments.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use plantvillage_ssl::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageBurnDataset};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::training::evaluate_accuracy;
use plantvillage_ssl::PlantVillageDataset;

use super::experiment_verbose;

/// Result of adding new classes to a trained base model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ScalingResult {
    /// Number of base classes
    pub(crate) base_classes: usize,
    /// Number of classes after adding
    pub(crate) total_classes: usize,
    /// Accuracy on base classes before adding new class
    pub(crate) base_accuracy_before: f64,
    /// Accuracy on base classes after adding new class
    pub(crate) base_accuracy_after: f64,
    /// Accuracy on new class
    pub(crate) new_class_accuracy: f64,
    /// Overall accuracy after adding
    pub(crate) overall_accuracy: f64,
    /// Forgetting measure (accuracy drop on old classes)
    pub(crate) forgetting: f64,
    /// Training time in seconds
    pub(crate) training_time: f64,
}

/// Train a base model on `base_classes` classes, then add `new_classes`
/// classes and fine-tune on the combined data.
///
/// At most `max_labels_per_new_class` training images are used per new
/// class; pass `usize::MAX` to use the full 80% training split.
pub(crate) fn run_incremental_experiment<B: AutodiffBackend>(
    dataset: &PlantVillageDataset,
    base_classes: usize,
    new_classes: usize,
    max_labels_per_new_class: usize,
    epochs: usize,
    seed: u64,
) -> Result<ScalingResult> {
    let device = B::Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = Instant::now();

    // Group by class
    let mut by_class: HashMap<usize, Vec<&plantvillage_ssl::dataset::loader::ImageSample>> =
        HashMap::new();
    for sample in &dataset.samples {
        by_class.entry(sample.label).or_default().push(sample);
    }

    // Get class indices sorted
    let mut class_indices: Vec<usize> = by_class.keys().cloned().collect();
    class_indices.sort();

    let base_class_indices: Vec<usize> = class_indices.iter().take(base_classes).cloned().collect();
    let new_class_indices: Vec<usize> = class_indices
        .iter()
        .skip(base_classes)
        .take(new_classes)
        .cloned()
        .collect();

    // Create base training data (80% train, 20% val for each class)
    let mut base_train: Vec<(PathBuf, usize)> = Vec::new();
    let mut base_val: Vec<(PathBuf, usize)> = Vec::new();

    for &class_idx in &base_class_indices {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);
            let split = (samples.len() as f64 * 0.8) as usize;
            let new_label = base_class_indices
                .iter()
                .position(|&x| x == class_idx)
                .unwrap();
            for s in samples.iter().take(split) {
                base_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(split) {
                base_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Train base model
    let image_size = 128;
    let batch_size = 64;
    let learning_rate = 0.0001;

    let train_dataset = PlantVillageBurnDataset::new(base_train.clone(), image_size);
    let val_dataset = PlantVillageBurnDataset::new(base_val.clone(), image_size);

    let batcher = PlantVillageBatcher::with_image_size(image_size);

    let model_config = PlantClassifierConfig {
        num_classes: base_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    // Train base model
    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = indices.len().div_ceil(batch_size);

        let mut epoch_loss = 0.0f64;
        let mut train_batches = 0usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            train_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }

        if experiment_verbose() {
            let avg = if train_batches > 0 {
                epoch_loss / train_batches as f64
            } else {
                0.0
            };
            println!(
                "      [base model] epoch {}/{}  train_loss={:.4}  batches={}",
                epoch + 1,
                epochs,
                avg,
                train_batches
            );
            let _ = io::stdout().flush();
        }
    }

    // Evaluate base model on base classes
    let base_accuracy_before =
        evaluate_accuracy(&model.valid(), &val_dataset, batch_size, image_size);
    if experiment_verbose() {
        println!(
            "      [base model] validation accuracy: {:.2}%",
            base_accuracy_before
        );
        let _ = io::stdout().flush();
    }

    // Now add new class with LIMITED labels
    let total_classes = base_classes + new_classes;

    // Prepare combined data - but limit new class samples
    let mut combined_train: Vec<(PathBuf, usize)> = base_train.clone();
    let mut combined_val: Vec<(PathBuf, usize)> = base_val.clone();
    let mut new_class_val: Vec<(PathBuf, usize)> = Vec::new();

    for (i, &class_idx) in new_class_indices.iter().enumerate() {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);

            // Limit training samples to max_labels_per_new_class
            let available_train = (samples.len() as f64 * 0.8) as usize;
            let train_n = max_labels_per_new_class.min(available_train);

            // Use remaining as validation
            let val_start = train_n;
            let new_label = base_classes + i;

            for s in samples.iter().take(train_n) {
                combined_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(val_start) {
                combined_val.push((s.path.clone(), new_label));
                new_class_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Create new model with extended classes
    let new_model_config = PlantClassifierConfig {
        num_classes: total_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut new_model = PlantClassifier::<B>::new(&new_model_config, &device);
    let mut new_optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let combined_train_dataset = PlantVillageBurnDataset::new(combined_train, image_size);
    let combined_val_dataset = PlantVillageBurnDataset::new(combined_val.clone(), image_size);
    let new_batcher = PlantVillageBatcher::with_image_size(image_size);

    // Train on combined data
    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..combined_train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = indices.len().div_ceil(batch_size);

        let mut epoch_loss = 0.0f64;
        let mut train_batches = 0usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| combined_train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = new_batcher.batch(items, &device);
            let output = new_model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            train_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &new_model);
            new_model = new_optimizer.step(learning_rate, new_model, grads);
        }

        if experiment_verbose() {
            let avg = if train_batches > 0 {
                epoch_loss / train_batches as f64
            } else {
                0.0
            };
            println!(
                "      [expanded model] epoch {}/{}  train_loss={:.4}  batches={}",
                epoch + 1,
                epochs,
                avg,
                train_batches
            );
            let _ = io::stdout().flush();
        }
    }

    // Evaluate on base classes only (to measure forgetting)
    let base_val_dataset = PlantVillageBurnDataset::new(base_val, image_size);
    let base_accuracy_after = evaluate_accuracy(
        &new_model.valid(),
        &base_val_dataset,
        batch_size,
        image_size,
    );

    // Evaluate on new class only
    let new_class_accuracy = if !new_class_val.is_empty() {
        let new_class_val_dataset = PlantVillageBurnDataset::new(new_class_val, image_size);
        evaluate_accuracy(
            &new_model.valid(),
            &new_class_val_dataset,
            batch_size,
            image_size,
        )
    } else {
        0.0
    };

    // Overall accuracy
    let overall_accuracy = evaluate_accuracy(
        &new_model.valid(),
        &combined_val_dataset,
        batch_size,
        image_size,
    );

    let forgetting = base_accuracy_before - base_accuracy_after;
    let training_time = start.elapsed().as_secs_f64();

    if experiment_verbose() {
        println!(
            "      [expanded model] val: base_after={:.2}% new_cls={:.2}% overall={:.2}% forget={:.2}% time={:.1}s",
            base_accuracy_after,
            new_class_accuracy,
            overall_accuracy,
            forgetting,
            training_time
        );
        let _ = io::stdout().flush();
    }

    Ok(ScalingResult {
        base_classes,
        total_classes,
        base_accuracy_before,
        base_accuracy_after,
        new_class_accuracy,
        overall_accuracy,
        forgetting,
        training_time,
    })
}
