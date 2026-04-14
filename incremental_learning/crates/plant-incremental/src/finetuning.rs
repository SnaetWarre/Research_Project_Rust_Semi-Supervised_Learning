//! Fine-tuning based incremental learning
//!
//! This module implements simple fine-tuning strategies with optional
//! layer freezing to reduce catastrophic forgetting.

use crate::{IncrementalConfig, IncrementalLearner, TrainingMetrics};
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

/// Fine-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningConfig {
    /// Whether to freeze the backbone (feature extractor)
    pub freeze_backbone: bool,
    /// Number of layers to freeze from the input
    pub freeze_layers: usize,
    /// Learning rate for new classifier head
    pub head_lr: f64,
    /// Learning rate for unfrozen backbone layers
    pub backbone_lr: f64,
    /// Whether to use gradual unfreezing
    pub gradual_unfreeze: bool,
    /// Epochs per unfreezing step (if gradual_unfreeze is true)
    pub unfreeze_every: usize,
}

impl Default for FineTuningConfig {
    fn default() -> Self {
        Self {
            freeze_backbone: false,
            freeze_layers: 0,
            head_lr: 0.001,
            backbone_lr: 0.0001,
            gradual_unfreeze: false,
            unfreeze_every: 5,
        }
    }
}

/// Fine-tuning learner state
#[derive(Debug)]
pub struct FineTuningLearner<B: Backend> {
    /// Current step/task number
    current_step: usize,
    /// Total classes learned
    num_classes: usize,
    /// Configuration
    config: FineTuningConfig,
    /// Frozen layer indices
    frozen_layers: Vec<usize>,
    /// Phantom data for backend
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> FineTuningLearner<B> {
    /// Create a new fine-tuning learner
    pub fn new(config: FineTuningConfig) -> Self {
        Self {
            current_step: 0,
            num_classes: 0,
            config,
            frozen_layers: Vec::new(),
            _backend: std::marker::PhantomData,
        }
    }

    /// Freeze specified layers
    pub fn freeze_layers(&mut self, layer_indices: Vec<usize>) {
        self.frozen_layers = layer_indices;
    }

    /// Unfreeze specified layers
    pub fn unfreeze_layers(&mut self, layer_indices: &[usize]) {
        self.frozen_layers.retain(|idx| !layer_indices.contains(idx));
    }

    /// Unfreeze all layers
    pub fn unfreeze_all(&mut self) {
        self.frozen_layers.clear();
    }

    /// Get frozen layer indices
    pub fn frozen_layer_indices(&self) -> &[usize] {
        &self.frozen_layers
    }

    /// Check if gradual unfreezing should occur
    fn should_unfreeze(&self, epoch: usize) -> bool {
        self.config.gradual_unfreeze
            && epoch > 0
            && epoch % self.config.unfreeze_every == 0
            && !self.frozen_layers.is_empty()
    }

    /// Perform gradual unfreezing
    fn gradual_unfreeze(&mut self, epoch: usize) {
        if self.should_unfreeze(epoch) {
            // Unfreeze one layer from the end (closer to output)
            if let Some(last_frozen) = self.frozen_layers.pop() {
                println!("Unfreezing layer {} at epoch {}", last_frozen, epoch);
            }
        }
    }
}

impl<B: Backend> IncrementalLearner<B> for FineTuningLearner<B> {
    fn prepare_for_new_classes(&mut self, num_new_classes: usize) -> Result<()> {
        if num_new_classes == 0 {
            return Err(anyhow!("Cannot add zero classes"));
        }

        // Update class count
        self.num_classes += num_new_classes;

        // Apply layer freezing strategy
        if self.config.freeze_backbone && self.current_step > 0 {
            // Freeze early layers for incremental steps
            let layers_to_freeze: Vec<usize> = (0..self.config.freeze_layers).collect();
            self.freeze_layers(layers_to_freeze);
            println!(
                "Frozen {} backbone layers for incremental step {}",
                self.config.freeze_layers, self.current_step
            );
        }

        Ok(())
    }

    fn train_incremental(
        &mut self,
        train_data: &[(Vec<f32>, usize)],
        val_data: &[(Vec<f32>, usize)],
        _config: &IncrementalConfig,
    ) -> Result<TrainingMetrics> {
        if train_data.is_empty() {
            return Err(anyhow!("Training data is empty"));
        }
        if val_data.is_empty() {
            return Err(anyhow!("Validation data is empty"));
        }

        let mut metrics = TrainingMetrics::new();
        let start_time = std::time::Instant::now();

        // Simulate training epochs
        let num_epochs = 10; // This would come from a training config

        for epoch in 0..num_epochs {
            // Check for gradual unfreezing
            self.gradual_unfreeze(epoch);

            // Simulate training loss decrease
            let train_loss = 2.0 * (-0.1 * epoch as f32).exp();

            // Simulate validation accuracy increase
            let val_accuracy = 0.5 + 0.4 * (1.0 - (-0.15 * epoch as f32).exp());

            metrics.add_epoch(train_loss, val_accuracy);

            // Log progress
            if epoch % 5 == 0 {
                println!(
                    "Epoch {}/{}: loss={:.4}, acc={:.4}, frozen_layers={}",
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    val_accuracy,
                    self.frozen_layers.len()
                );
            }
        }

        metrics.set_training_time(start_time.elapsed().as_secs_f64());
        metrics.add_extra("frozen_layers", self.frozen_layers.len() as f32);

        Ok(metrics)
    }

    fn finalize_step(&mut self, step: usize) -> Result<()> {
        self.current_step = step;

        // Unfreeze all layers after each step if not using persistent freezing
        if !self.config.freeze_backbone {
            self.unfreeze_all();
        }

        Ok(())
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_finetuning_learner_creation() {
        let config = FineTuningConfig::default();
        let learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        assert_eq!(learner.current_step(), 0);
        assert_eq!(learner.num_classes(), 0);
        assert_eq!(learner.frozen_layer_indices().len(), 0);
    }

    #[test]
    fn test_layer_freezing() {
        let config = FineTuningConfig::default();
        let mut learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        learner.freeze_layers(vec![0, 1, 2]);
        assert_eq!(learner.frozen_layer_indices().len(), 3);

        learner.unfreeze_layers(&[1]);
        assert_eq!(learner.frozen_layer_indices().len(), 2);
        assert!(!learner.frozen_layer_indices().contains(&1));

        learner.unfreeze_all();
        assert_eq!(learner.frozen_layer_indices().len(), 0);
    }

    #[test]
    fn test_prepare_for_new_classes() {
        let config = FineTuningConfig {
            freeze_backbone: true,
            freeze_layers: 3,
            ..Default::default()
        };
        let mut learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        // Initial step shouldn't freeze
        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 5);

        // Move to next step
        learner.finalize_step(1).unwrap();

        // Second step should freeze layers
        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 10);
        assert_eq!(learner.frozen_layer_indices().len(), 3);
    }

    #[test]
    fn test_train_incremental() {
        let config = FineTuningConfig::default();
        let mut learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();

        // Create dummy data
        let train_data = vec![(vec![0.5; 100], 0); 10];
        let val_data = vec![(vec![0.3; 100], 0); 5];

        let inc_config = IncrementalConfig {
            initial_classes: 5,
            classes_per_step: 5,
            num_steps: 1,
            method: crate::IncrementalMethod::FineTuning {
                freeze_backbone: false,
                freeze_layers: 0,
            },
            seed: 42,
        };

        let metrics = learner.train_incremental(&train_data, &val_data, &inc_config).unwrap();

        assert!(metrics.train_loss.len() > 0);
        assert!(metrics.val_accuracy.len() > 0);
        assert!(metrics.training_time >= 0.0);
    }

    #[test]
    fn test_gradual_unfreezing() {
        let config = FineTuningConfig {
            gradual_unfreeze: true,
            unfreeze_every: 2,
            ..Default::default()
        };
        let mut learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        learner.freeze_layers(vec![0, 1, 2, 3]);
        assert_eq!(learner.frozen_layer_indices().len(), 4);

        // Should unfreeze at epoch 2, 4, 6, etc.
        learner.gradual_unfreeze(2);
        assert_eq!(learner.frozen_layer_indices().len(), 3);

        learner.gradual_unfreeze(4);
        assert_eq!(learner.frozen_layer_indices().len(), 2);
    }

    #[test]
    fn test_empty_data_error() {
        let config = FineTuningConfig::default();
        let mut learner: FineTuningLearner<TestBackend> = FineTuningLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();

        let train_data = vec![];
        let val_data = vec![(vec![0.3; 100], 0); 5];

        let inc_config = IncrementalConfig {
            initial_classes: 5,
            classes_per_step: 5,
            num_steps: 1,
            method: crate::IncrementalMethod::FineTuning {
                freeze_backbone: false,
                freeze_layers: 0,
            },
            seed: 42,
        };

        let result = learner.train_incremental(&train_data, &val_data, &inc_config);
        assert!(result.is_err());
    }
}
