//! Training infrastructure for plant disease classification models.
//!
//! This module provides:
//! - Training loop with epoch management
//! - Loss computation and backpropagation
//! - Metric tracking and logging
//! - Checkpointing and model saving
//! - Early stopping support

use plant_core::{Error, Result, TrainingConfig, TrainingMetrics};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn};

/// Training state for checkpointing and resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: usize,
    pub best_loss: f64,
    pub best_accuracy: f64,
    pub patience_counter: usize,
    pub training_history: TrainingMetrics,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            best_loss: f64::INFINITY,
            best_accuracy: 0.0,
            patience_counter: 0,
            training_history: TrainingMetrics::default(),
        }
    }
}

/// Configuration for the trainer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainerConfig {
    pub learning_rate: f64,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub early_stopping_patience: Option<usize>,
    pub checkpoint_dir: PathBuf,
    pub save_frequency: usize,
    pub gradient_clip: Option<f64>,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            num_epochs: 100,
            batch_size: 32,
            early_stopping_patience: Some(10),
            checkpoint_dir: PathBuf::from("checkpoints"),
            save_frequency: 5,
            gradient_clip: Some(1.0),
        }
    }
}

impl From<TrainingConfig> for TrainerConfig {
    fn from(config: TrainingConfig) -> Self {
        Self {
            learning_rate: config.training.learning_rate,
            num_epochs: config.training.num_epochs,
            batch_size: config.training.batch_size,
            early_stopping_patience: config.training.early_stopping_patience,
            checkpoint_dir: PathBuf::from("checkpoints"),
            save_frequency: 5,
            gradient_clip: config.training.grad_clip,
        }
    }
}

/// Metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub loss: f64,
    pub accuracy: f64,
}

/// Simple trainer interface
///
/// This is a simplified training interface that delegates actual training
/// to the Burn framework's training utilities. The actual training loop
/// would be implemented using Burn's `LearnerBuilder` and related APIs.
pub struct Trainer {
    config: TrainerConfig,
    state: TrainingState,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            state: TrainingState::default(),
        }
    }

    /// Get current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get mutable training state
    pub fn state_mut(&mut self) -> &mut TrainingState {
        &mut self.state
    }

    /// Load training state from checkpoint
    pub fn load_state(&mut self, state: TrainingState) {
        self.state = state;
    }

    /// Get the configuration
    pub fn config(&self) -> &TrainerConfig {
        &self.config
    }

    /// Update training metrics for current epoch
    pub fn update_epoch(
        &mut self,
        train_loss: f64,
        train_accuracy: f64,
        val_loss: f64,
        val_accuracy: f64,
    ) -> bool {
        self.state.epoch += 1;

        // Update history
        self.state.training_history.train_loss.push(train_loss);
        self.state.training_history.train_accuracy.push(train_accuracy);
        self.state.training_history.val_loss.push(val_loss);
        self.state.training_history.val_accuracy.push(val_accuracy);

        // Check for improvement
        let improved = val_loss < self.state.best_loss;
        if improved {
            info!(
                "Validation loss improved from {:.4} to {:.4}",
                self.state.best_loss, val_loss
            );
            self.state.best_loss = val_loss;
            self.state.best_accuracy = val_accuracy;
            self.state.patience_counter = 0;
        } else {
            self.state.patience_counter += 1;
            warn!(
                "No improvement. Patience: {}/{:?}",
                self.state.patience_counter, self.config.early_stopping_patience
            );
        }

        info!(
            "Epoch {}: train_loss={:.4}, train_acc={:.4}, val_loss={:.4}, val_acc={:.4}",
            self.state.epoch, train_loss, train_accuracy, val_loss, val_accuracy
        );

        improved
    }

    /// Check if early stopping should trigger
    pub fn should_stop(&self) -> bool {
        if let Some(patience) = self.config.early_stopping_patience {
            self.state.patience_counter >= patience
        } else {
            false
        }
    }

    /// Check if checkpoint should be saved
    pub fn should_save_checkpoint(&self) -> bool {
        self.state.epoch % self.config.save_frequency == 0
    }

    /// Get training history
    pub fn training_history(&self) -> &TrainingMetrics {
        &self.state.training_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_default() {
        let config = TrainerConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_state_default() {
        let state = TrainingState::default();
        assert_eq!(state.epoch, 0);
        assert_eq!(state.best_loss, f64::INFINITY);
        assert_eq!(state.best_accuracy, 0.0);
        assert_eq!(state.patience_counter, 0);
    }

    #[test]
    fn test_epoch_metrics() {
        let metrics = EpochMetrics {
            loss: 0.5,
            accuracy: 0.95,
        };
        assert_eq!(metrics.loss, 0.5);
        assert_eq!(metrics.accuracy, 0.95);
    }

    #[test]
    fn test_trainer_update_epoch() {
        let config = TrainerConfig::default();
        let mut trainer = Trainer::new(config);

        // First epoch with improvement
        let improved = trainer.update_epoch(1.0, 0.7, 0.9, 0.75);
        assert!(improved);
        assert_eq!(trainer.state().best_loss, 0.9);
        assert_eq!(trainer.state().patience_counter, 0);

        // Second epoch with improvement
        let improved = trainer.update_epoch(0.8, 0.75, 0.7, 0.80);
        assert!(improved);
        assert_eq!(trainer.state().best_loss, 0.7);
        assert_eq!(trainer.state().patience_counter, 0);

        // Third epoch without improvement
        let improved = trainer.update_epoch(0.6, 0.80, 0.75, 0.78);
        assert!(!improved);
        assert_eq!(trainer.state().best_loss, 0.7);
        assert_eq!(trainer.state().patience_counter, 1);
    }

    #[test]
    fn test_early_stopping() {
        let mut config = TrainerConfig::default();
        config.early_stopping_patience = Some(3);
        let mut trainer = Trainer::new(config);

        // No stopping initially
        assert!(!trainer.should_stop());

        // Simulate no improvement for patience epochs
        trainer.update_epoch(1.0, 0.7, 0.9, 0.75); // Improved
        assert!(!trainer.should_stop());

        trainer.update_epoch(0.9, 0.72, 0.95, 0.74); // No improvement (1)
        assert!(!trainer.should_stop());

        trainer.update_epoch(0.8, 0.74, 0.96, 0.73); // No improvement (2)
        assert!(!trainer.should_stop());

        trainer.update_epoch(0.7, 0.76, 0.97, 0.72); // No improvement (3)
        assert!(trainer.should_stop());
    }

    #[test]
    fn test_checkpoint_saving() {
        let mut config = TrainerConfig::default();
        config.save_frequency = 5;
        let mut trainer = Trainer::new(config);

        trainer.update_epoch(1.0, 0.7, 0.9, 0.75);
        assert!(!trainer.should_save_checkpoint());

        for _ in 0..4 {
            trainer.update_epoch(0.9, 0.75, 0.85, 0.78);
        }

        assert!(trainer.should_save_checkpoint());
    }
}
