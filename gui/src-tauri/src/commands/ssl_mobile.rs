//! Mobile SSL Retraining Commands
//!
//! Commands for running lightweight SSL retraining on mobile devices
//! with adaptive configurations based on device capabilities.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::CompactRecorder;
use burn::tensor::ElementConversion;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use plantvillage_ssl::dataset::burn_dataset::{
    AugmentingBatcher, PlantVillageBurnDataset, RawPlantVillageDataset,
};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::PlantVillageBatcher;

use crate::backend::AdaptiveBackend;
use crate::device::{AdaptiveTrainingConfig, DeviceType};
use crate::state::AppState;

/// SSL Retraining configuration (lightweight for mobile)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSLRetrainingParams {
    pub model_path: String,
    pub labeled_data_dir: String,
    pub pseudo_labels: Vec<PseudoLabelItem>,
    pub output_path: String,
    /// If None, uses adaptive config based on device
    pub custom_config: Option<CustomRetrainingConfig>,
}

/// Custom retraining configuration (optional override)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRetrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
}

/// Pseudo-label item from frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLabelItem {
    pub image_path: String,
    pub predicted_label: usize,
    pub confidence: f64,
}

/// Retraining progress event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingProgress {
    pub epoch: usize,
    pub total_epochs: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub loss: f64,
    pub device_type: String,
}

/// Retraining result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingResult {
    pub model_path: String,
    pub epochs_completed: usize,
    pub samples_used: usize,
    pub pseudo_labels_used: usize,
    pub final_loss: f64,
    pub device_type: String,
    pub backend: String,
}

/// Start SSL retraining with adaptive configuration
#[tauri::command]
pub async fn start_ssl_retraining(
    params: SSLRetrainingParams,
    app: AppHandle,
    _state: State<'_, Arc<AppState>>,
) -> Result<RetrainingResult, String> {
    // Detect device and get adaptive config
    let device_type = DeviceType::detect();
    let mut training_config = AdaptiveTrainingConfig::for_ssl_retraining();

    // Apply custom config if provided (clone before move)
    if let Some(ref custom) = params.custom_config {
        training_config.batch_size = custom.batch_size;
        training_config.epochs = custom.epochs;
        training_config.learning_rate = custom.learning_rate;
    }

    tracing::info!(
        "Starting SSL retraining on {:?} with config: {:?}",
        device_type,
        training_config
    );

    // Emit start event (clone app before move)
    let _ = app.emit(
        "ssl:retraining:started",
        serde_json::json!({
            "device_type": format!("{:?}", device_type),
            "config": &training_config,
        }),
    );

    // Clone app for the closure
    let app_clone = app.clone();

    // Run retraining in blocking task
    let result = tokio::task::spawn_blocking(move || {
        run_ssl_retraining_inner(params, training_config, &app_clone)
    })
    .await
    .map_err(|e| format!("Retraining task failed: {:?}", e))?;

    // Emit completion event
    if let Ok(ref res) = result {
        let _ = app.emit("ssl:retraining:complete", res);
    }

    result
}

/// Inner retraining function
fn run_ssl_retraining_inner(
    params: SSLRetrainingParams,
    config: AdaptiveTrainingConfig,
    app: &AppHandle,
) -> Result<RetrainingResult, String> {
    let device = <AdaptiveBackend as burn::tensor::backend::Backend>::Device::default();

    tracing::info!(
        "Running on device: {:?} with backend: {}",
        device,
        crate::backend::backend_name()
    );

    // Load existing model
    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    let mut model: PlantClassifier<AdaptiveBackend> =
        PlantClassifier::new(&model_config, &device);

    // Load checkpoint
    let model_path = PathBuf::from(&params.model_path);
    if model_path.exists() {
        tracing::info!("Loading model from: {:?}", model_path);
        let recorder = CompactRecorder::new();
        model = model
            .load_file(&model_path, &recorder, &device)
            .map_err(|e| format!("Failed to load model: {:?}", e))?;
    } else {
        return Err(format!("Model not found at: {:?}", model_path));
    }

    // Load labeled data
    let labeled_dataset = PlantVillageBurnDataset::new_cached(
        vec![], // TODO: Load from labeled_data_dir
        128,
    )
    .map_err(|e| format!("Failed to load labeled dataset: {:?}", e))?;

    // Convert pseudo-labels to training samples
    let pseudo_samples: Vec<(PathBuf, usize)> = params
        .pseudo_labels
        .iter()
        .map(|pl| (PathBuf::from(&pl.image_path), pl.predicted_label))
        .collect();

    tracing::info!(
        "Retraining with {} pseudo-labels and {} labeled samples",
        pseudo_samples.len(),
        labeled_dataset.len()
    );

    // Create combined dataset using RawPlantVillageDataset for augmentation
    let combined_dataset = RawPlantVillageDataset::new_cached(pseudo_samples)
        .map_err(|e| format!("Failed to create combined dataset: {:?}", e))?;

    // Create optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    // Create augmenting batcher
    let aug_batcher = AugmentingBatcher::<AdaptiveBackend>::new(device.clone(), 128, 42);
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let num_batches =
        (combined_dataset.len() + config.batch_size - 1) / config.batch_size;

    tracing::info!(
        "Training for {} epochs with batch_size={}, {} batches per epoch",
        config.epochs,
        config.batch_size,
        num_batches
    );

    let mut final_loss = 0.0;

    // Training loop
    for epoch in 0..config.epochs {
        let len = combined_dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for (batch_idx, start) in (0..len).step_by(config.batch_size).enumerate() {
            let end = (start + config.batch_size).min(len);
            let batch_indices = &indices[start..end];
            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| combined_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            // Use augmenting batcher
            let batch = aug_batcher.batch(items, &device);

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
            model = optimizer.step(config.learning_rate, model, grads);

            // Emit progress
            let _ = app.emit(
                "ssl:retraining:progress",
                RetrainingProgress {
                    epoch: epoch + 1,
                    total_epochs: config.epochs,
                    batch: batch_idx + 1,
                    total_batches: num_batches,
                    loss: loss_value,
                    device_type: format!("{:?}", config.device_type),
                },
            );
        }

        final_loss = if batch_count > 0 {
            epoch_loss / batch_count as f64
        } else {
            0.0
        };

        tracing::info!(
            "Epoch {}/{}: avg_loss = {:.4}",
            epoch + 1,
            config.epochs,
            final_loss
        );
    }

    // Save retrained model
    let output_path = PathBuf::from(&params.output_path);
    std::fs::create_dir_all(output_path.parent().unwrap())
        .map_err(|e| format!("Failed to create output directory: {:?}", e))?;

    tracing::info!("Saving retrained model to: {:?}", output_path);
    let recorder = CompactRecorder::new();
    model
        .save_file(&output_path, &recorder)
        .map_err(|e| format!("Failed to save model: {:?}", e))?;

    Ok(RetrainingResult {
        model_path: format!("{}.mpk", output_path.display()),
        epochs_completed: config.epochs,
        samples_used: combined_dataset.len(),
        pseudo_labels_used: params.pseudo_labels.len(),
        final_loss,
        device_type: format!("{:?}", config.device_type),
        backend: crate::backend::backend_name().to_string(),
    })
}

/// Get recommended SSL retraining configuration for current device
#[tauri::command]
pub fn get_ssl_retraining_config() -> Result<AdaptiveTrainingConfig, String> {
    Ok(AdaptiveTrainingConfig::for_ssl_retraining())
}
