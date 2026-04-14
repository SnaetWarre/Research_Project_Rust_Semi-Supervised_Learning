//! Incremental Learning Methods for Plant Disease Classification
//!
//! This crate implements various incremental learning strategies that allow
//! models to learn new plant disease classes without catastrophic forgetting.

pub mod finetuning;
pub mod lwf;
pub mod ewc;
pub mod rehearsal;
pub mod metrics;

use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for incremental learning experiments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Initial number of classes
    pub initial_classes: usize,
    /// Number of classes to add in each incremental step
    pub classes_per_step: usize,
    /// Total number of incremental steps
    pub num_steps: usize,
    /// Method to use for incremental learning
    pub method: IncrementalMethod,
    /// Random seed for reproducibility
    pub seed: u64,
}

/// Available incremental learning methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IncrementalMethod {
    /// Simple fine-tuning with optional layer freezing
    FineTuning {
        /// Whether to freeze early layers
        freeze_backbone: bool,
        /// Number of layers to freeze (from input)
        freeze_layers: usize,
    },
    /// Learning without Forgetting (distillation-based)
    LwF {
        /// Temperature for distillation
        temperature: f32,
        /// Weight for distillation loss
        lambda: f32,
    },
    /// Elastic Weight Consolidation
    EWC {
        /// Importance weight for EWC penalty
        lambda: f32,
        /// Number of samples for Fisher estimation
        fisher_samples: usize,
    },
    /// Rehearsal with exemplar memory
    Rehearsal {
        /// Memory size per class
        exemplars_per_class: usize,
        /// Exemplar selection strategy
        selection: ExemplarSelection,
    },
}

/// Strategy for selecting exemplars for rehearsal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExemplarSelection {
    /// Random selection
    Random,
    /// Herding (select most representative samples)
    Herding,
    /// Distance-based selection
    DistanceBased,
}

/// Trait for incremental learning methods
pub trait IncrementalLearner<B: Backend> {
    /// Prepare for learning new classes
    fn prepare_for_new_classes(&mut self, num_new_classes: usize) -> anyhow::Result<()>;

    /// Train on new classes
    fn train_incremental(
        &mut self,
        train_data: &[(Vec<f32>, usize)],
        val_data: &[(Vec<f32>, usize)],
        config: &IncrementalConfig,
    ) -> anyhow::Result<TrainingMetrics>;

    /// Finalize incremental step (e.g., update memory, compute Fisher)
    fn finalize_step(&mut self, step: usize) -> anyhow::Result<()>;

    /// Get current task/step number
    fn current_step(&self) -> usize;

    /// Get total number of classes learned so far
    fn num_classes(&self) -> usize;
}

/// Metrics collected during incremental training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss per epoch
    pub train_loss: Vec<f32>,
    /// Validation accuracy per epoch
    pub val_accuracy: Vec<f32>,
    /// Time taken for training (seconds)
    pub training_time: f64,
    /// Additional method-specific metrics
    pub extra: HashMap<String, f32>,
}

impl TrainingMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_accuracy: Vec::new(),
            training_time: 0.0,
            extra: HashMap::new(),
        }
    }

    /// Add an epoch's metrics
    pub fn add_epoch(&mut self, loss: f32, accuracy: f32) {
        self.train_loss.push(loss);
        self.val_accuracy.push(accuracy);
    }

    /// Set training time
    pub fn set_training_time(&mut self, time: f64) {
        self.training_time = time;
    }

    /// Add extra metric
    pub fn add_extra(&mut self, key: impl Into<String>, value: f32) {
        self.extra.insert(key.into(), value);
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an incremental learning experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalResult {
    /// Metrics for each incremental step
    pub step_metrics: Vec<StepMetrics>,
    /// Overall experiment metadata
    pub metadata: ExperimentMetadata,
}

/// Metrics for a single incremental step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    /// Step number (0 = initial training)
    pub step: usize,
    /// Training metrics
    pub training: TrainingMetrics,
    /// Accuracy on each task after this step
    pub task_accuracies: Vec<f32>,
    /// Average accuracy across all tasks
    pub average_accuracy: f32,
    /// Backward transfer (change in accuracy on old tasks)
    pub backward_transfer: Option<f32>,
    /// Forward transfer (accuracy on new task vs random)
    pub forward_transfer: Option<f32>,
}

/// Metadata about the experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentMetadata {
    /// Experiment name/ID
    pub name: String,
    /// Configuration used
    pub config: IncrementalConfig,
    /// Total experiment time (seconds)
    pub total_time: f64,
    /// Final average accuracy
    pub final_accuracy: f32,
    /// Average backward transfer
    pub avg_backward_transfer: f32,
    /// Average forward transfer
    pub avg_forward_transfer: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        assert_eq!(metrics.train_loss.len(), 0);

        metrics.add_epoch(0.5, 0.85);
        metrics.add_epoch(0.3, 0.90);
        assert_eq!(metrics.train_loss.len(), 2);
        assert_eq!(metrics.val_accuracy.len(), 2);

        metrics.set_training_time(123.45);
        assert_eq!(metrics.training_time, 123.45);

        metrics.add_extra("custom_metric", 42.0);
        assert_eq!(metrics.extra.get("custom_metric"), Some(&42.0));
    }

    #[test]
    fn test_incremental_config_serialization() {
        let config = IncrementalConfig {
            initial_classes: 5,
            classes_per_step: 5,
            num_steps: 5,
            method: IncrementalMethod::FineTuning {
                freeze_backbone: true,
                freeze_layers: 2,
            },
            seed: 42,
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: IncrementalConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.initial_classes, deserialized.initial_classes);
        assert_eq!(config.classes_per_step, deserialized.classes_per_step);
    }

    #[test]
    fn test_incremental_method_variants() {
        let lwf = IncrementalMethod::LwF {
            temperature: 2.0,
            lambda: 1.0,
        };

        let json = serde_json::to_string(&lwf).unwrap();
        assert!(json.contains("LwF"));
        assert!(json.contains("temperature"));

        let ewc = IncrementalMethod::EWC {
            lambda: 5000.0,
            fisher_samples: 200,
        };

        let json = serde_json::to_string(&ewc).unwrap();
        assert!(json.contains("EWC"));
        assert!(json.contains("fisher_samples"));
    }
}
