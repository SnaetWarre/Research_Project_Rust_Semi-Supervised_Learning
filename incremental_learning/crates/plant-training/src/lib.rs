//! Training infrastructure for plant disease classification.
//!
//! This module provides:
//! - Model architectures (EfficientNet-B0, ResNet-18)
//! - Training loop with checkpointing
//! - Evaluation and metrics computation
//! - Learning rate scheduling
//! - Early stopping

pub mod checkpoint;
pub mod evaluator;
pub mod lr_schedule;
pub mod model;
pub mod trainer;

pub use checkpoint::{Checkpoint, CheckpointManager};
pub use evaluator::{EvaluationResult, Evaluator};
pub use lr_schedule::{LearningRateScheduler, SchedulerType};
pub use model::{EfficientNetB0, ModelArchitecture, PlantClassifier, ResNet18};
pub use trainer::{Trainer, TrainerConfig, TrainingState};

use burn::tensor::backend::Backend;
use plant_core::{Error, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use super::checkpoint::{Checkpoint, CheckpointManager};
    pub use super::evaluator::{EvaluationResult, Evaluator};
    pub use super::lr_schedule::{LearningRateScheduler, SchedulerType};
    pub use super::model::{EfficientNetB0, PlantClassifier, ResNet18};
    pub use super::trainer::{Trainer, TrainerConfig};
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
