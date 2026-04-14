//! Model module for CNN architectures using the Burn framework
//!
//! This module provides:
//! - CNN model architectures for plant disease classification
//! - Model configuration and hyperparameters
//! - Model serialization and loading utilities
//!
//! ## Architecture
//!
//! The primary model is a convolutional neural network designed for:
//! - 39-class plant disease classification
//! - Efficient inference on edge devices (Jetson Orin Nano)
//! - Semi-supervised learning with pseudo-labeling

pub mod cnn;
pub mod config;

// Re-export main types for convenience
pub use cnn::PlantClassifier;
pub use config::ModelConfig;

/// Default dropout rate for regularization
pub const DEFAULT_DROPOUT: f64 = 0.5;

/// Default number of classes for PlantVillage
pub const DEFAULT_NUM_CLASSES: usize = 39;
