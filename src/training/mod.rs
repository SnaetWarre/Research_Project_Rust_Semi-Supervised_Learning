//! Training module for semi-supervised learning
//!
//! This module provides:
//! - Main training loop with Burn framework
//! - Pseudo-labeling algorithm for semi-supervised learning
//! - Learning rate scheduling and optimization
//! - Training metrics and logging
//!
//! ## Semi-Supervised Learning Approach
//!
//! The training pipeline implements pseudo-labeling:
//! 1. Train initial model on labeled data
//! 2. Use model to predict labels for unlabeled data
//! 3. Add high-confidence predictions to training set
//! 4. Retrain model with augmented dataset
//! 5. Repeat until convergence or data exhaustion

pub mod supervised;
pub mod pseudo_label;
pub mod scheduler;
pub mod simulation;
pub mod ssl_incremental;
pub mod trainer;

// Re-export main types for convenience
pub use pseudo_label::{PseudoLabelConfig, PseudoLabeler};
pub use simulation::{run_simulation, SimulationConfig, SimulationResults};
pub use scheduler::LRScheduler as LearningRateScheduler;
pub use scheduler::LRScheduler as SchedulerType;
pub use ssl_incremental::{SSLIncrementalConfig, SSLIncrementalResults, run_ssl_incremental_experiment};
pub use trainer::{Trainer, TrainingState};

// Re-export TrainingConfig from model::config where it's defined
pub use crate::model::config::TrainingConfig;

/// Default number of training epochs
pub const DEFAULT_EPOCHS: usize = 50;

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 32;

/// Default learning rate
pub const DEFAULT_LEARNING_RATE: f64 = 0.001;

/// Default confidence threshold for pseudo-labeling
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.9;
