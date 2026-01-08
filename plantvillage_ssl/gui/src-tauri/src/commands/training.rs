//! Training Commands
//!
//! Commands for training models with real-time progress events.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;

use crate::state::{AppState, TrainingStatus};

/// Training configuration from frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub data_dir: String,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub labeled_ratio: f64,
    pub confidence_threshold: f64,
    pub output_dir: String,
    /// Use class-weighted loss for imbalanced data
    #[serde(default)]
    pub class_weighted: bool,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            data_dir: "data/plantvillage/balanced".to_string(),
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.0001,
            labeled_ratio: 0.2,
            confidence_threshold: 0.9,
            output_dir: "output/models".to_string(),
            class_weighted: false, // Not needed with balanced dataset
        }
    }
}

/// Event payload for epoch updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochUpdate {
    pub epoch: usize,
    pub total_epochs: usize,
    pub train_loss: f64,
    pub val_accuracy: f64,
    pub learning_rate: f64,
}

/// Event payload for batch updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchUpdate {
    pub epoch: usize,
    pub batch: usize,
    pub total_batches: usize,
    pub loss: f64,
}

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub final_accuracy: f64,
    pub epochs_completed: usize,
    pub model_path: String,
    pub loss_history: Vec<f64>,
    pub accuracy_history: Vec<f64>,
}

/// Get current training status
#[tauri::command]
pub async fn get_training_status(
    state: State<'_, Arc<AppState>>,
) -> Result<TrainingStatus, String> {
    let status = state.training_state.read().await;
    Ok(status.clone())
}

/// Start training (this will emit events during training)
#[tauri::command]
pub async fn start_training(
    params: TrainingParams,
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<TrainingResult, String> {
    use burn::backend::Autodiff;
    use burn_cuda::Cuda;

    // Update state to running
    {
        let mut status = state.training_state.write().await;
        *status = TrainingStatus::Running {
            epoch: 0,
            total_epochs: params.epochs,
            batch: 0,
            total_batches: 0,
            current_loss: 0.0,
            current_accuracy: 0.0,
        };
    }

    // Emit start event
    let _ = app.emit("training:started", &params);

    // Run training in a blocking task since it's CPU/GPU intensive
    let result = tokio::task::spawn_blocking(move || {
        run_training_inner::<Autodiff<Cuda>>(&params, &app)
    })
    .await
    .map_err(|e| format!("Training task failed: {:?}", e))?;

    // Update state based on result
    {
        let mut status = state.training_state.write().await;
        match &result {
            Ok(res) => {
                *status = TrainingStatus::Completed {
                    final_accuracy: res.final_accuracy,
                    total_epochs: res.epochs_completed,
                };
            }
            Err(e) => {
                *status = TrainingStatus::Error(e.clone());
            }
        }
    }

    result
}

/// Inner training function
fn run_training_inner<B>(
    params: &TrainingParams,
    app: &AppHandle,
) -> Result<TrainingResult, String>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    use std::path::PathBuf;
    use burn::data::dataset::Dataset;
    use burn::module::Module;
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::optim::{AdamConfig, GradientsParams, Optimizer};
    use burn::record::CompactRecorder;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use plantvillage_ssl::dataset::burn_dataset::PlantVillageBurnDataset;
    use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
    use plantvillage_ssl::{PlantVillageBatcher, PlantVillageDataset};

    let device = B::Device::default();

    // Load dataset
    let dataset = PlantVillageDataset::new(&params.data_dir)
        .map_err(|e| format!("Failed to load dataset: {:?}", e))?;

    let samples: Vec<(PathBuf, usize)> = dataset
        .samples
        .iter()
        .map(|s| (s.path.clone(), s.label))
        .collect();

    // Create train/val split
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut indices: Vec<usize> = (0..samples.len()).collect();
    indices.shuffle(&mut rng);

    let val_size = (samples.len() as f64 * 0.1) as usize;
    let val_indices = &indices[..val_size];
    let train_indices = &indices[val_size..];

    // Use only labeled_ratio of training data
    let labeled_size = (train_indices.len() as f64 * params.labeled_ratio) as usize;
    let train_indices = &train_indices[..labeled_size];

    let train_samples: Vec<(PathBuf, usize)> = train_indices
        .iter()
        .map(|&i| samples[i].clone())
        .collect();
    let val_samples: Vec<(PathBuf, usize)> = val_indices
        .iter()
        .map(|&i| samples[i].clone())
        .collect();

    // Compute class weights if enabled
    let class_weights: Option<Vec<f32>> = if params.class_weighted {
        let mut class_counts = vec![0usize; 38];
        for (_, label) in &train_samples {
            if *label < 38 {
                class_counts[*label] += 1;
            }
        }
        let total = train_samples.len() as f32;
        let num_classes = 38.0f32;
        let weights: Vec<f32> = class_counts
            .iter()
            .map(|&count| {
                if count > 0 {
                    total / (num_classes * count as f32)
                } else {
                    1.0
                }
            })
            .collect();
        Some(weights)
    } else {
        None
    };

    let train_dataset = PlantVillageBurnDataset::new_cached(train_samples, 128)
        .map_err(|e| format!("Failed to create train dataset: {:?}", e))?;
    let val_dataset = PlantVillageBurnDataset::new_cached(val_samples, 128)
        .map_err(|e| format!("Failed to create val dataset: {:?}", e))?;

    // Create model
    let config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model: PlantClassifier<B> = PlantClassifier::new(&config, &device);

    // Create optimizer
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), 128);

    let mut loss_history = Vec::new();
    let mut accuracy_history = Vec::new();

    let num_batches = (train_dataset.len() + params.batch_size - 1) / params.batch_size;

    // Training loop
    for epoch in 0..params.epochs {
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        // Training
        for (batch_idx, start) in (0..train_dataset.len()).step_by(params.batch_size).enumerate() {
            let end = (start + params.batch_size).min(train_dataset.len());
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

            // Compute loss (with optional class weights)
            let loss = if let Some(ref weights) = class_weights {
                weighted_cross_entropy(&output, &batch.targets, weights, &device)
            } else {
                CrossEntropyLossConfig::new()
                    .init(&output.device())
                    .forward(output, batch.targets)
            };

            use burn::tensor::ElementConversion;
            let loss_value: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            batch_count += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(params.learning_rate, model, grads);

            // Emit batch update
            let _ = app.emit("training:batch", BatchUpdate {
                epoch: epoch + 1,
                batch: batch_idx + 1,
                total_batches: num_batches,
                loss: loss_value,
            });
        }

        let avg_loss = if batch_count > 0 { epoch_loss / batch_count as f64 } else { 0.0 };
        loss_history.push(avg_loss);

        // Validation
        let val_batcher = PlantVillageBatcher::<B::InnerBackend>::with_image_size(
            <B::InnerBackend as burn::tensor::backend::Backend>::Device::default(),
            128,
        );
        let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
        let model_valid = model.clone().valid();

        let mut correct = 0usize;
        let mut total = 0usize;

        for start in (0..val_dataset.len()).step_by(params.batch_size) {
            let end = (start + params.batch_size).min(val_dataset.len());
            let items: Vec<_> = (start..end)
                .filter_map(|i| val_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = val_batcher.batch(items, &inner_device);
            let output = model_valid.forward(batch.images);
            let predictions = output.argmax(1).squeeze::<1>();

            use burn::tensor::ElementConversion;
            let batch_correct: i64 = predictions
                .equal(batch.targets)
                .int()
                .sum()
                .into_scalar()
                .elem();

            correct += batch_correct as usize;
            total += end - start;
        }

        let val_accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
        accuracy_history.push(val_accuracy);

        // Calculate learning rate (cosine annealing)
        let t = epoch as f64;
        let t_max = params.epochs as f64;
        let lr_min = params.learning_rate * 0.01;
        let lr_max = params.learning_rate;
        let current_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t / t_max).cos());

        // Emit epoch update
        let _ = app.emit("training:epoch", EpochUpdate {
            epoch: epoch + 1,
            total_epochs: params.epochs,
            train_loss: avg_loss,
            val_accuracy: val_accuracy * 100.0,
            learning_rate: current_lr,
        });
    }

    // Save model
    std::fs::create_dir_all(&params.output_dir)
        .map_err(|e| format!("Failed to create output directory: {:?}", e))?;

    let model_path = format!("{}/plant_classifier_gui", params.output_dir);
    let recorder = CompactRecorder::new();
    model
        .clone()
        .save_file(&model_path, &recorder)
        .map_err(|e| format!("Failed to save model: {:?}", e))?;

    let final_accuracy = accuracy_history.last().copied().unwrap_or(0.0) * 100.0;

    // Emit completion
    let result = TrainingResult {
        final_accuracy,
        epochs_completed: params.epochs,
        model_path: format!("{}.mpk", model_path),
        loss_history,
        accuracy_history: accuracy_history.iter().map(|a| a * 100.0).collect(),
    };

    let _ = app.emit("training:complete", &result);

    Ok(result)
}

/// Stop training
#[tauri::command]
pub async fn stop_training(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut status = state.training_state.write().await;
    if matches!(*status, TrainingStatus::Running { .. }) {
        *status = TrainingStatus::Idle;
    }
    Ok(())
}

/// Compute weighted cross entropy loss
///
/// Manually implements class-weighted cross entropy since Burn 0.20
/// doesn't provide built-in support for it.
fn weighted_cross_entropy<B: burn::tensor::backend::AutodiffBackend>(
    output: &burn::tensor::Tensor<B, 2>,
    targets: &burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    weights: &[f32],
    device: &B::Device,
) -> burn::tensor::Tensor<B, 1> {
    use burn::tensor::Tensor;
    use burn::tensor::activation::softmax;

    // Compute softmax probabilities
    let probs = softmax(output.clone(), 1);

    // Add small epsilon for numerical stability before log
    let epsilon = 1e-7f32;
    let log_probs = (probs + epsilon).log();

    // Get the log probability of the correct class for each sample
    let targets_expanded = targets.clone().unsqueeze_dim::<2>(1);
    let target_log_probs = log_probs.gather(1, targets_expanded).squeeze::<1>();

    // Create weight tensor for each sample based on its class
    let weights_tensor: Tensor<B, 1> = Tensor::from_floats(weights, device);

    // Gather weights for each target class
    let sample_weights = weights_tensor.gather(0, targets.clone());

    // Weighted negative log likelihood
    let weighted_nll = target_log_probs.neg().mul(sample_weights.clone());

    // Return mean weighted loss
    weighted_nll.sum() / sample_weights.sum()
}
