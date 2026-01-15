//! Application State Management
//!
//! Manages the global state for the Tauri application, including
//! model state, training state, and dataset information.

use std::path::PathBuf;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

use burn_cuda::Cuda;

/// Type alias for the backend we use
pub type AppBackend = Cuda;

/// Application state shared across commands
/// Note: Models are NOT stored here due to CUDA threading restrictions.
/// Instead, model paths are stored and models are loaded on-demand.
pub struct AppState {
    /// Path to currently loaded model (if any)
    pub model_path: RwLock<Option<PathBuf>>,
    /// Dataset information (if loaded)
    pub dataset_info: RwLock<Option<DatasetInfo>>,
    /// Training state
    pub training_state: RwLock<TrainingStatus>,
    /// Simulation state
    pub simulation_state: RwLock<SimulationStatus>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            model_path: RwLock::new(None),
            dataset_info: RwLock::new(None),
            training_state: RwLock::new(TrainingStatus::Idle),
            simulation_state: RwLock::new(SimulationStatus::Idle),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

/// Dataset information for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub path: String,
    pub total_samples: usize,
    pub num_classes: usize,
    pub class_names: Vec<String>,
    pub class_counts: Vec<usize>,
}

/// Training status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingStatus {
    Idle,
    Running {
        epoch: usize,
        total_epochs: usize,
        batch: usize,
        total_batches: usize,
        current_loss: f64,
        current_accuracy: f64,
    },
    Paused {
        epoch: usize,
        total_epochs: usize,
    },
    Completed {
        final_accuracy: f64,
        total_epochs: usize,
    },
    Error(String),
}

/// Simulation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SimulationStatus {
    Idle,
    Running {
        day: usize,
        total_days: usize,
        pseudo_labels: usize,
        current_accuracy: f64,
    },
    Completed {
        initial_accuracy: f64,
        final_accuracy: f64,
        total_pseudo_labels: usize,
    },
    Error(String),
}
