//! Model checkpointing and state management.
//!
//! This module provides:
//! - Model state saving and loading
//! - Training state persistence
//! - Checkpoint management (best model, periodic saves)
//! - Format conversion utilities

use plant_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Checkpoint containing model and training state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub epoch: usize,
    pub best_loss: f32,
    pub best_accuracy: f32,
    pub learning_rate: f64,
    pub optimizer_state: Option<Vec<u8>>,
    pub timestamp: String,
    pub metadata: CheckpointMetadata,
}

/// Metadata associated with a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub model_architecture: String,
    pub num_classes: usize,
    pub num_parameters: usize,
    pub training_samples: usize,
    pub validation_accuracy: f32,
    pub notes: Option<String>,
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self {
            model_architecture: String::from("unknown"),
            num_classes: 0,
            num_parameters: 0,
            training_samples: 0,
            validation_accuracy: 0.0,
            notes: None,
        }
    }
}

impl Checkpoint {
    /// Create a new checkpoint
    pub fn new(
        epoch: usize,
        best_loss: f32,
        best_accuracy: f32,
        learning_rate: f64,
        metadata: CheckpointMetadata,
    ) -> Self {
        use chrono::Utc;
        let timestamp = Utc::now().to_rfc3339();

        Self {
            epoch,
            best_loss,
            best_accuracy,
            learning_rate,
            optimizer_state: None,
            timestamp,
            metadata,
        }
    }

    /// Save checkpoint to file
    pub fn save(&self, path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Serialize checkpoint to JSON
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| Error::Serialization(format!("Failed to serialize checkpoint: {}", e)))?;

        // Write to file
        fs::write(path, json)?;

        info!("Checkpoint saved to {:?}", path);
        Ok(())
    }

    /// Load checkpoint from file
    pub fn load(path: &Path) -> Result<Self> {
        // Read file
        let json = fs::read_to_string(path)?;

        // Deserialize
        let checkpoint: Checkpoint = serde_json::from_str(&json)
            .map_err(|e| Error::Serialization(format!("Failed to deserialize checkpoint: {}", e)))?;

        info!("Checkpoint loaded from {:?}", path);
        Ok(checkpoint)
    }
}

/// Manager for handling multiple checkpoints
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    keep_best: bool,
    keep_last_n: Option<usize>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager
    pub fn new(checkpoint_dir: PathBuf) -> Self {
        Self {
            checkpoint_dir,
            keep_best: true,
            keep_last_n: Some(3),
        }
    }

    /// Configure whether to keep the best checkpoint
    pub fn keep_best(mut self, keep: bool) -> Self {
        self.keep_best = keep;
        self
    }

    /// Configure how many recent checkpoints to keep
    pub fn keep_last_n(mut self, n: Option<usize>) -> Self {
        self.keep_last_n = n;
        self
    }

    /// Save a checkpoint with automatic naming
    pub fn save_checkpoint(&self, checkpoint: &Checkpoint, is_best: bool) -> Result<()> {
        // Create checkpoint directory
        fs::create_dir_all(&self.checkpoint_dir)?;

        // Save regular checkpoint
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.json", checkpoint.epoch));
        checkpoint.save(&checkpoint_path)?;

        // Save best checkpoint if applicable
        if is_best && self.keep_best {
            let best_path = self.checkpoint_dir.join("best_model.json");
            checkpoint.save(&best_path)?;
            info!("Best model checkpoint saved");
        }

        // Save latest checkpoint
        let latest_path = self.checkpoint_dir.join("latest.json");
        checkpoint.save(&latest_path)?;

        // Clean up old checkpoints if configured
        if let Some(keep_n) = self.keep_last_n {
            self.cleanup_old_checkpoints(keep_n)?;
        }

        Ok(())
    }

    /// Load the latest checkpoint
    pub fn load_latest(&self) -> Result<Checkpoint> {
        let latest_path = self.checkpoint_dir.join("latest.json");
        Checkpoint::load(&latest_path)
    }

    /// Load the best checkpoint
    pub fn load_best(&self) -> Result<Checkpoint> {
        let best_path = self.checkpoint_dir.join("best_model.json");
        Checkpoint::load(&best_path)
    }

    /// Load a specific checkpoint by epoch
    pub fn load_epoch(&self, epoch: usize) -> Result<Checkpoint> {
        let checkpoint_path = self.checkpoint_dir.join(format!("checkpoint_epoch_{}.json", epoch));
        Checkpoint::load(&checkpoint_path)
    }

    /// List all available checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<PathBuf>> {
        if !self.checkpoint_dir.exists() {
            return Ok(Vec::new());
        }

        let mut checkpoints = Vec::new();

        let entries = fs::read_dir(&self.checkpoint_dir)?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s: &std::ffi::OsStr| s.to_str()) == Some("json") {
                // Skip special files
                let filename = path.file_name().and_then(|s: &std::ffi::OsStr| s.to_str()).unwrap_or("");
                if filename != "best_model.json" && filename != "latest.json" {
                    checkpoints.push(path);
                }
            }
        }

        // Sort by epoch number
        checkpoints.sort();

        Ok(checkpoints)
    }

    /// Clean up old checkpoints, keeping only the last N
    fn cleanup_old_checkpoints(&self, keep_n: usize) -> Result<()> {
        let mut checkpoints = self.list_checkpoints()?;

        if checkpoints.len() <= keep_n {
            return Ok(());
        }

        // Sort by epoch (oldest first)
        checkpoints.sort();

        // Remove oldest checkpoints
        let to_remove = checkpoints.len() - keep_n;
        for checkpoint_path in checkpoints.iter().take(to_remove) {
            if let Err(e) = fs::remove_file(checkpoint_path) {
                warn!("Failed to remove old checkpoint {:?}: {}", checkpoint_path, e);
            } else {
                info!("Removed old checkpoint: {:?}", checkpoint_path);
            }
        }

        Ok(())
    }

    /// Get checkpoint directory path
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }
}

/// Helper function to extract epoch number from checkpoint filename
pub fn extract_epoch_from_filename(filename: &str) -> Option<usize> {
    // Expected format: "checkpoint_epoch_123.json"
    filename
        .strip_prefix("checkpoint_epoch_")
        .and_then(|s| s.strip_suffix(".json"))
        .and_then(|s| s.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_creation() {
        let metadata = CheckpointMetadata {
            model_architecture: String::from("ResNet18"),
            num_classes: 10,
            num_parameters: 1000000,
            training_samples: 5000,
            validation_accuracy: 0.95,
            notes: Some(String::from("Test checkpoint")),
        };

        let checkpoint = Checkpoint::new(5, 0.123, 0.95, 0.001, metadata);

        assert_eq!(checkpoint.epoch, 5);
        assert_eq!(checkpoint.best_loss, 0.123);
        assert_eq!(checkpoint.best_accuracy, 0.95);
    }

    #[test]
    fn test_checkpoint_save_load() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_path = temp_dir.path().join("test_checkpoint.json");

        let metadata = CheckpointMetadata::default();
        let original = Checkpoint::new(10, 0.5, 0.9, 0.001, metadata);

        // Save
        original.save(&checkpoint_path)?;
        assert!(checkpoint_path.exists());

        // Load
        let loaded = Checkpoint::load(&checkpoint_path)?;

        assert_eq!(loaded.epoch, original.epoch);
        assert_eq!(loaded.best_loss, original.best_loss);
        assert_eq!(loaded.best_accuracy, original.best_accuracy);

        Ok(())
    }

    #[test]
    fn test_checkpoint_manager() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().to_path_buf();

        let manager = CheckpointManager::new(checkpoint_dir.clone())
            .keep_best(true)
            .keep_last_n(Some(2));

        // Create and save multiple checkpoints
        for epoch in 0..5 {
            let metadata = CheckpointMetadata::default();
            let checkpoint = Checkpoint::new(
                epoch,
                1.0 / (epoch + 1) as f32,
                0.5 + (epoch as f32 * 0.1),
                0.001,
                metadata,
            );

            let is_best = epoch == 4;
            manager.save_checkpoint(&checkpoint, is_best)?;
        }

        // Check that old checkpoints were cleaned up (keep_last_n = 2)
        let checkpoints = manager.list_checkpoints()?;
        assert!(checkpoints.len() <= 2);

        // Check that best checkpoint exists
        let best_path = checkpoint_dir.join("best_model.json");
        assert!(best_path.exists());

        // Check that latest checkpoint exists
        let latest_path = checkpoint_dir.join("latest.json");
        assert!(latest_path.exists());

        // Load latest
        let latest = manager.load_latest()?;
        assert_eq!(latest.epoch, 4);

        // Load best
        let best = manager.load_best()?;
        assert_eq!(best.epoch, 4);

        Ok(())
    }

    #[test]
    fn test_extract_epoch_from_filename() {
        assert_eq!(extract_epoch_from_filename("checkpoint_epoch_5.json"), Some(5));
        assert_eq!(extract_epoch_from_filename("checkpoint_epoch_123.json"), Some(123));
        assert_eq!(extract_epoch_from_filename("best_model.json"), None);
        assert_eq!(extract_epoch_from_filename("latest.json"), None);
        assert_eq!(extract_epoch_from_filename("invalid.json"), None);
    }

    #[test]
    fn test_checkpoint_manager_keep_all() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let checkpoint_dir = temp_dir.path().to_path_buf();

        let manager = CheckpointManager::new(checkpoint_dir.clone())
            .keep_last_n(None); // Keep all checkpoints

        // Create and save multiple checkpoints
        for epoch in 0..5 {
            let metadata = CheckpointMetadata::default();
            let checkpoint = Checkpoint::new(epoch, 0.5, 0.9, 0.001, metadata);
            manager.save_checkpoint(&checkpoint, false)?;
        }

        // All checkpoints should be present
        let checkpoints = manager.list_checkpoints()?;
        assert_eq!(checkpoints.len(), 5);

        Ok(())
    }
}
