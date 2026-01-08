//! Training infrastructure for plant disease classification.
//!
//! This module provides:
//! - Model architectures (EfficientNet-B0, ResNet-18)
//! - Training loop with checkpointing
//! - Evaluation and metrics computation
//! - Learning rate scheduling
//! - Early stopping

pub mod model;
pub mod trainer;
pub mod evaluator;
pub mod lr_schedule;
pub mod checkpoint;

pub use model::{EfficientNetB0, ResNet18, ModelArchitecture, PlantClassifier};
pub use trainer::{Trainer, TrainerConfig, TrainingState};
pub use evaluator::{Evaluator, EvaluationResult};
pub use lr_schedule::{LearningRateScheduler, SchedulerType};
pub use checkpoint::{Checkpoint, CheckpointManager};

use burn::tensor::backend::Backend;
use plant_core::{Error, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use super::model::{EfficientNetB0, ResNet18, PlantClassifier};
    pub use super::trainer::{Trainer, TrainerConfig};
    pub use super::evaluator::{Evaluator, EvaluationResult};
    pub use super::lr_schedule::{LearningRateScheduler, SchedulerType};
    pub use super::checkpoint::{Checkpoint, CheckpointManager};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Ensure modules are accessible
        assert!(true);
    }
}
