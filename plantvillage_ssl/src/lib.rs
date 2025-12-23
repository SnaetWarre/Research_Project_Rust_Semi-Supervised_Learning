//! # PlantVillage Semi-Supervised Learning
//!
//! A Rust library for semi-supervised plant disease classification using the Burn framework.
//! Designed for deployment on NVIDIA Jetson Orin Nano edge devices.
//!
//! ## Features
//!
//! - **Semi-supervised learning** with pseudo-labeling for efficient training on partially labeled data
//! - **Burn framework** for portable, efficient neural network training and inference
//! - **Edge deployment** optimized for NVIDIA Jetson Orin Nano
//! - **PlantVillage dataset** support with 39 disease classes
//!
//! ## Modules
//!
//! - `dataset`: Data loading, augmentation, and split strategies for semi-supervised learning
//! - `model`: CNN architecture built with Burn
//! - `training`: Training loops, pseudo-labeling, and learning rate scheduling
//! - `inference`: Prediction and benchmarking utilities
//! - `utils`: Logging, metrics, and helper functions
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use plantvillage_ssl::dataset::PlantVillageDataset;
//! use plantvillage_ssl::model::cnn::PlantClassifierConfig;
//!
//! // Load dataset
//! let dataset = PlantVillageDataset::new("data/plantvillage")?;
//!
//! // Create model
//! let config = PlantClassifierConfig::new();
//! // ... training and inference
//! ```

pub mod dataset;
pub mod inference;
pub mod model;
pub mod training;
pub mod utils;

// Re-export commonly used items for convenience
pub use dataset::loader::PlantVillageDataset;
pub use dataset::split::{DatasetSplits, SplitConfig};
pub use dataset::{
    PlantVillageBatch, PlantVillageBatcher, PlantVillageBurnDataset, PlantVillageItem,
    PseudoLabelBatch, PseudoLabelBatcher, PseudoLabelDataset, PseudoLabeledItem,
};
pub use inference::benchmark::BenchmarkResult;
pub use inference::predictor::Predictor;
pub use model::cnn::PlantClassifier;
pub use model::config::{ModelConfig, SemiSupervisedConfig};
pub use training::pseudo_label::{PseudoLabelConfig, PseudoLabeler};
pub use training::trainer::{Trainer, TrainingState};
pub use training::TrainingConfig;
pub use utils::error::{PlantVillageError, Result};
pub use utils::metrics::{ConfusionMatrix, Metrics};

/// PlantVillage disease classes (39 total)
pub const NUM_CLASSES: usize = 39;

/// Default image size for PlantVillage dataset
pub const IMAGE_SIZE: usize = 256;

/// Default confidence threshold for pseudo-labeling
pub const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.9;

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
