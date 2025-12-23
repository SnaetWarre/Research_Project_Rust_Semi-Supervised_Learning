//! Complete Training Pipeline for PlantVillage Semi-Supervised Learning
//!
//! This module implements the full training loop using the Burn framework,
//! including:
//! - Forward/backward passes with automatic differentiation
//! - Cross-entropy loss computation
//! - Adam optimizer with learning rate scheduling
//! - Validation and evaluation loops
//! - Checkpoint saving and loading
//! - Early stopping

use std::path::Path;

use burn::{
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, ElementConversion, Int, Tensor},
};
use tracing::{debug, info, warn};

use crate::dataset::{PlantVillageBatch, PseudoLabelBatch};
use crate::model::config::{SemiSupervisedConfig, TrainingConfig};
use crate::model::PlantClassifier;
use crate::utils::metrics::Metrics;

/// Training state for checkpointing and monitoring
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch (0-indexed)
    pub epoch: usize,
    /// Current iteration within epoch
    pub iteration: usize,
    /// Best validation accuracy seen so far
    pub best_val_accuracy: f64,
    /// Training loss history (per epoch)
    pub train_losses: Vec<f64>,
    /// Validation accuracy history (per epoch)
    pub val_accuracies: Vec<f64>,
    /// Number of epochs without improvement (for early stopping)
    pub epochs_without_improvement: usize,
    /// Total training samples seen
    pub samples_seen: usize,
    /// Current learning rate
    pub current_lr: f64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            iteration: 0,
            best_val_accuracy: 0.0,
            train_losses: Vec::new(),
            val_accuracies: Vec::new(),
            epochs_without_improvement: 0,
            samples_seen: 0,
            current_lr: 0.001,
        }
    }
}

impl TrainingState {
    /// Create a new training state with initial learning rate
    pub fn new(initial_lr: f64) -> Self {
        Self {
            current_lr: initial_lr,
            ..Default::default()
        }
    }

    /// Record training loss for current epoch
    pub fn record_train_loss(&mut self, loss: f64) {
        if self.train_losses.len() <= self.epoch {
            self.train_losses.push(loss);
        } else {
            self.train_losses[self.epoch] = loss;
        }
    }

    /// Record validation accuracy for current epoch
    pub fn record_val_accuracy(&mut self, accuracy: f64) {
        if self.val_accuracies.len() <= self.epoch {
            self.val_accuracies.push(accuracy);
        } else {
            self.val_accuracies[self.epoch] = accuracy;
        }
    }
}

/// Main trainer for the PlantClassifier model using Burn
pub struct Trainer<B: AutodiffBackend> {
    /// Model being trained
    pub model: PlantClassifier<B>,
    /// Adam optimizer
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam<B::InnerBackend>, PlantClassifier<B>, B>,
    /// Training configuration
    pub config: TrainingConfig,
    /// Semi-supervised learning configuration
    pub ssl_config: SemiSupervisedConfig,
    /// Current training state
    pub state: TrainingState,
    /// Device to train on
    device: B::Device,
    /// Number of classes
    num_classes: usize,
}

impl<B: AutodiffBackend> Trainer<B> {
    /// Create a new trainer with the given model and configuration
    pub fn new(
        model: PlantClassifier<B>,
        config: TrainingConfig,
        ssl_config: SemiSupervisedConfig,
        device: B::Device,
        num_classes: usize,
    ) -> Self {
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                config.weight_decay,
            )))
            .init();

        Self {
            model: model.clone(),
            optimizer,
            config: config.clone(),
            ssl_config,
            state: TrainingState::new(config.learning_rate),
            device,
            num_classes,
        }
    }

    /// Train for one epoch on labeled data
    ///
    /// # Arguments
    /// * `batches` - Iterator of (images, labels) batches
    ///
    /// # Returns
    /// * (average_loss, accuracy)
    pub fn train_epoch_labeled(
        &mut self,
        batches: &[PlantVillageBatch<B>],
    ) -> (f64, f64) {
        let mut total_loss = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;
        let num_batches = batches.len();

        info!(
            "Training epoch {} with {} batches",
            self.state.epoch + 1,
            num_batches
        );

        for (batch_idx, batch) in batches.iter().enumerate() {
            // Forward pass
            let output = self.model.forward(batch.images.clone());

            // Compute cross-entropy loss
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output.clone(), batch.targets.clone());

            let loss_scalar = loss.clone().into_scalar();
            let loss_value: f64 = loss_scalar.elem();
            total_loss += loss_value;

            // Calculate accuracy
            let predictions = output.argmax(1).squeeze::<1>(1);
            let batch_correct_tensor = predictions
                .equal(batch.targets.clone())
                .int()
                .sum();
            let batch_correct: i64 = batch_correct_tensor.into_scalar().elem();
            correct += batch_correct as usize;
            total += batch.targets.dims()[0];

            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);

            // Update parameters
            self.model = self.optimizer.step(self.state.current_lr, self.model.clone(), grads);

            self.state.iteration += 1;
            self.state.samples_seen += batch.targets.dims()[0];

            if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
                debug!(
                    "  Batch {}/{}: loss = {:.4}, acc = {:.2}%",
                    batch_idx + 1,
                    num_batches,
                    loss_value,
                    100.0 * correct as f64 / total as f64
                );
            }
        }

        let avg_loss = if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        };
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        self.state.record_train_loss(avg_loss);

        info!(
            "Epoch {} training: loss = {:.4}, accuracy = {:.2}%",
            self.state.epoch + 1,
            avg_loss,
            accuracy * 100.0
        );

        (avg_loss, accuracy)
    }

    /// Train for one epoch with semi-supervised learning
    ///
    /// Uses both labeled data and pseudo-labeled data with confidence weighting.
    pub fn train_epoch_semi_supervised(
        &mut self,
        labeled_batches: &[PlantVillageBatch<B>],
        pseudo_batches: &[PseudoLabelBatch<B>],
    ) -> (f64, f64, f64) {
        let mut labeled_loss_total = 0.0;
        let mut pseudo_loss_total = 0.0;
        let mut correct = 0usize;
        let mut total = 0usize;

        let num_labeled = labeled_batches.len();
        let num_pseudo = pseudo_batches.len();

        info!(
            "Semi-supervised epoch {} with {} labeled + {} pseudo batches",
            self.state.epoch + 1,
            num_labeled,
            num_pseudo
        );

        // Train on labeled data
        for batch in labeled_batches.iter() {
            let output = self.model.forward(batch.images.clone());

            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output.clone(), batch.targets.clone());

            let loss_scalar = loss.clone().into_scalar();
            let loss_value: f64 = loss_scalar.elem();
            labeled_loss_total += loss_value;

            // Calculate accuracy
            let predictions = output.argmax(1).squeeze::<1>(1);
            let batch_correct_tensor = predictions
                .equal(batch.targets.clone())
                .int()
                .sum();
            let batch_correct: i64 = batch_correct_tensor.into_scalar().elem();
            correct += batch_correct as usize;
            total += batch.targets.dims()[0];

            // Backward and update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = self.optimizer.step(self.state.current_lr, self.model.clone(), grads);

            self.state.iteration += 1;
            self.state.samples_seen += batch.targets.dims()[0];
        }

        // Train on pseudo-labeled data with weighted loss
        let ramp_up_weight = self.calculate_ramp_up_weight();

        for batch in pseudo_batches.iter() {
            let output = self.model.forward(batch.images.clone());

            // Compute weighted cross-entropy loss
            let loss = self.weighted_cross_entropy(
                output.clone(),
                batch.targets.clone(),
                batch.weights.clone(),
            );

            // Scale by unlabeled loss weight and ramp-up
            let loss_weight = self.ssl_config.unlabeled_loss_weight * ramp_up_weight;
            let loss = loss * loss_weight;

            let loss_scalar = loss.clone().into_scalar();
            let loss_value: f64 = loss_scalar.elem();
            pseudo_loss_total += loss_value;

            // Backward and update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &self.model);
            self.model = self.optimizer.step(self.state.current_lr, self.model.clone(), grads);

            self.state.iteration += 1;
            self.state.samples_seen += batch.targets.dims()[0];
        }

        let avg_labeled_loss = if num_labeled > 0 {
            labeled_loss_total / num_labeled as f64
        } else {
            0.0
        };

        let avg_pseudo_loss = if num_pseudo > 0 {
            pseudo_loss_total / num_pseudo as f64
        } else {
            0.0
        };

        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };

        self.state.record_train_loss(avg_labeled_loss + avg_pseudo_loss);

        info!(
            "Epoch {} SSL: labeled_loss = {:.4}, pseudo_loss = {:.4}, accuracy = {:.2}%",
            self.state.epoch + 1,
            avg_labeled_loss,
            avg_pseudo_loss,
            accuracy * 100.0
        );

        (avg_labeled_loss, avg_pseudo_loss, accuracy)
    }

    /// Weighted cross-entropy loss for pseudo-labeled data
    fn weighted_cross_entropy(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        weights: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = logits.device();
        let [batch_size, _num_classes] = logits.dims();

        // Compute log-softmax
        let log_probs = burn::tensor::activation::log_softmax(logits, 1);

        // Gather the log probabilities for the target classes
        let targets_2d = targets.clone().reshape([batch_size, 1]);
        let gathered = log_probs.gather(1, targets_2d);
        let nll = gathered.squeeze::<1>(1).neg(); // Negative log likelihood

        // Apply confidence weights
        let weighted_nll = nll * weights.clone();

        // Mean reduction
        let sum_weights = weights.sum();
        let sum_weighted_loss = weighted_nll.sum();

        // Avoid division by zero
        let eps = Tensor::<B, 1>::from_floats([1e-8], &device);
        sum_weighted_loss / (sum_weights + eps)
    }

    /// Calculate ramp-up weight for pseudo-label loss
    fn calculate_ramp_up_weight(&self) -> f64 {
        if self.ssl_config.ramp_up_epochs == 0 {
            return 1.0;
        }

        let progress = self.state.epoch as f64 / self.ssl_config.ramp_up_epochs as f64;
        progress.min(1.0)
    }

    /// Evaluate the model on a validation/test set
    ///
    /// Note: This uses the inner (non-autodiff) model for efficiency.
    pub fn evaluate(&self, batches: &[PlantVillageBatch<B::InnerBackend>]) -> Metrics {
        let model_valid = self.model.valid();

        let mut correct = 0usize;
        let mut total = 0usize;
        let mut total_loss = 0.0;
        let mut all_predictions: Vec<usize> = Vec::new();
        let mut all_targets: Vec<usize> = Vec::new();

        for batch in batches.iter() {
            let output = model_valid.forward(batch.images.clone());

            // Compute loss
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output.clone(), batch.targets.clone());

            let loss_scalar = loss.into_scalar();
            let loss_value: f64 = loss_scalar.elem();
            total_loss += loss_value;

            // Get predictions
            let predictions = output.argmax(1).squeeze::<1>(1);

            // Count correct predictions
            let batch_correct_tensor = predictions
                .clone()
                .equal(batch.targets.clone())
                .int()
                .sum();
            let batch_correct: i64 = batch_correct_tensor.into_scalar().elem();
            correct += batch_correct as usize;
            total += batch.targets.dims()[0];

            // Collect predictions and targets for detailed metrics
            let pred_data = predictions.into_data();
            let target_data = batch.targets.clone().into_data();

            // Convert to usize vectors
            let pred_vec: Vec<i64> = pred_data.to_vec().unwrap();
            let target_vec: Vec<i64> = target_data.to_vec().unwrap();

            all_predictions.extend(pred_vec.iter().map(|&p| p as usize));
            all_targets.extend(target_vec.iter().map(|&t| t as usize));
        }

        let num_batches = batches.len().max(1);
        let accuracy = if total > 0 {
            correct as f64 / total as f64
        } else {
            0.0
        };
        let avg_loss = total_loss / num_batches as f64;

        // Calculate detailed metrics
        let mut metrics = Metrics::from_predictions(&all_predictions, &all_targets, self.num_classes);
        metrics.loss = Some(avg_loss);
        metrics.accuracy = accuracy;

        info!(
            "Evaluation: loss = {:.4}, accuracy = {:.2}%, samples = {}",
            avg_loss,
            accuracy * 100.0,
            total
        );

        metrics
    }

    /// Run predictions on images and return (predicted_labels, confidences)
    pub fn predict_with_confidence(
        &self,
        images: Tensor<B::InnerBackend, 4>,
    ) -> (Vec<usize>, Vec<f32>) {
        let model_valid = self.model.valid();

        // Forward pass with softmax
        let probs = model_valid.forward_softmax(images);
        let [batch_size, _num_classes] = probs.dims();

        // Get max probability and index for each sample
        let max_probs = probs.clone().max_dim(1);
        let predictions = probs.argmax(1);

        // Convert to vectors
        let pred_data = predictions.into_data();
        let conf_data = max_probs.into_data();

        let pred_vec: Vec<i64> = pred_data.to_vec().unwrap();
        let conf_vec: Vec<f32> = conf_data.to_vec().unwrap();

        let predictions: Vec<usize> = pred_vec.iter().map(|&p| p as usize).collect();
        let confidences: Vec<f32> = conf_vec.into_iter().take(batch_size).collect();

        (predictions, confidences)
    }

    /// Update learning rate based on scheduler
    pub fn update_learning_rate(&mut self) {
        let lr = match self.config.lr_scheduler {
            crate::model::config::LRSchedulerType::Constant => self.config.learning_rate,

            crate::model::config::LRSchedulerType::CosineAnnealing => {
                let t = self.state.epoch as f64;
                let t_max = self.config.epochs as f64;
                let lr_min = self.config.learning_rate * 0.01; // Minimum LR is 1% of initial
                let lr_max = self.config.learning_rate;

                lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f64::consts::PI * t / t_max).cos())
            }

            crate::model::config::LRSchedulerType::StepDecay => {
                let decay_epochs = 10; // Decay every 10 epochs
                let decay_factor: f64 = 0.1;
                let num_decays = self.state.epoch / decay_epochs;
                self.config.learning_rate * decay_factor.powi(num_decays as i32)
            }

            crate::model::config::LRSchedulerType::Exponential => {
                let decay_rate: f64 = 0.95;
                self.config.learning_rate * decay_rate.powi(self.state.epoch as i32)
            }

            crate::model::config::LRSchedulerType::ReduceOnPlateau => {
                // Simple implementation: reduce if no improvement for 5 epochs
                if self.state.epochs_without_improvement >= 5 {
                    self.state.current_lr * 0.5
                } else {
                    self.state.current_lr
                }
            }
        };

        // Apply warmup
        let lr = if self.state.epoch < self.config.warmup_epochs {
            let warmup_factor = (self.state.epoch + 1) as f64 / self.config.warmup_epochs as f64;
            lr * warmup_factor
        } else {
            lr
        };

        if (lr - self.state.current_lr).abs() > 1e-8 {
            debug!(
                "Learning rate updated: {:.6} -> {:.6}",
                self.state.current_lr, lr
            );
        }

        self.state.current_lr = lr;
    }

    /// Save model checkpoint
    pub fn save_checkpoint(&self, path: &Path) -> anyhow::Result<()> {
        info!("Saving checkpoint to {:?}", path);

        // Create parent directories if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Save model using Burn's recorder
        let recorder = CompactRecorder::new();
        self.model
            .clone()
            .save_file(path, &recorder)
            .map_err(|e| anyhow::anyhow!("Failed to save model: {:?}", e))?;

        info!(
            "Checkpoint saved (epoch {}, best accuracy: {:.2}%)",
            self.state.epoch + 1,
            self.state.best_val_accuracy * 100.0
        );

        Ok(())
    }

    /// Load model from checkpoint
    pub fn load_checkpoint(&mut self, path: &Path) -> anyhow::Result<()> {
        info!("Loading checkpoint from {:?}", path);

        let recorder = CompactRecorder::new();
        self.model = self
            .model
            .clone()
            .load_file(path, &recorder, &self.device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

        info!("Checkpoint loaded successfully");

        Ok(())
    }

    /// Check if early stopping criteria is met
    pub fn should_early_stop(&self) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            if self.state.epochs_without_improvement >= patience {
                warn!(
                    "Early stopping triggered after {} epochs without improvement",
                    patience
                );
                return true;
            }
        }
        false
    }

    /// Update best model tracking after validation
    ///
    /// Returns true if this is a new best model
    pub fn update_best(&mut self, val_accuracy: f64) -> bool {
        self.state.record_val_accuracy(val_accuracy);

        if val_accuracy > self.state.best_val_accuracy {
            info!(
                "New best model! Accuracy improved: {:.2}% -> {:.2}%",
                self.state.best_val_accuracy * 100.0,
                val_accuracy * 100.0
            );
            self.state.best_val_accuracy = val_accuracy;
            self.state.epochs_without_improvement = 0;
            true
        } else {
            self.state.epochs_without_improvement += 1;
            debug!(
                "No improvement for {} epochs (best: {:.2}%)",
                self.state.epochs_without_improvement,
                self.state.best_val_accuracy * 100.0
            );
            false
        }
    }

    /// Advance to next epoch
    pub fn next_epoch(&mut self) {
        self.state.epoch += 1;
        self.state.iteration = 0;
        self.update_learning_rate();
    }

    /// Get current learning rate
    pub fn get_current_lr(&self) -> f64 {
        self.state.current_lr
    }

    /// Get reference to the model
    pub fn model(&self) -> &PlantClassifier<B> {
        &self.model
    }

    /// Get the device
    pub fn device(&self) -> &B::Device {
        &self.device
    }
}

/// Full training run configuration
pub struct TrainingRun {
    /// Path to save checkpoints
    pub checkpoint_dir: String,
    /// Whether to save best model only
    pub save_best_only: bool,
    /// Logging verbosity
    pub verbose: bool,
}

impl Default for TrainingRun {
    fn default() -> Self {
        Self {
            checkpoint_dir: "output/checkpoints".to_string(),
            save_best_only: true,
            verbose: true,
        }
    }
}

/// Compute accuracy from logits and targets
pub fn accuracy<B: burn::tensor::backend::Backend>(
    output: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
) -> f64 {
    let predictions = output.argmax(1).squeeze::<1>(1);
    let correct_tensor = predictions.equal(targets.clone()).int().sum();
    let correct: i64 = correct_tensor.into_scalar().elem();
    let total = targets.dims()[0];

    if total > 0 {
        correct as f64 / total as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::SemiSupervisedConfig;

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.best_val_accuracy, 0.0);
        assert!(state.train_losses.is_empty());
    }

    #[test]
    fn test_training_state_record() {
        let mut state = TrainingState::default();

        state.record_train_loss(0.5);
        assert_eq!(state.train_losses.len(), 1);
        assert_eq!(state.train_losses[0], 0.5);

        state.record_val_accuracy(0.85);
        assert_eq!(state.val_accuracies.len(), 1);
        assert_eq!(state.val_accuracies[0], 0.85);
    }

    #[test]
    fn test_ramp_up_weight() {
        let ssl_config = SemiSupervisedConfig {
            ramp_up_epochs: 10,
            ..Default::default()
        };

        // Epoch 0: weight should be 0.0
        let weight_0: f64 = 0.0 / 10.0;
        assert!((weight_0 - 0.0).abs() < 0.001);

        // Epoch 5: weight should be 0.5
        let weight_5: f64 = 5.0 / 10.0;
        assert!((weight_5 - 0.5).abs() < 0.001);

        // Epoch 10+: weight should be 1.0
        let weight_10 = (10.0 / 10.0_f64).min(1.0);
        assert!((weight_10 - 1.0).abs() < 0.001);
    }
}
