//! Model Configuration Module
//!
//! Defines configuration structures for the CNN model architecture,
//! training hyperparameters, and optimization settings.

use serde::{Deserialize, Serialize};

/// Configuration for the CNN model architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Number of output classes (38 for PlantVillage)
    pub num_classes: usize,

    /// Input image size (width and height, assumed square)
    pub input_size: usize,

    /// Number of input channels (3 for RGB)
    pub input_channels: usize,

    /// Dropout rate for regularization (0.0 to 1.0)
    pub dropout_rate: f64,

    /// Number of filters in each convolutional layer
    pub conv_filters: Vec<usize>,

    /// Kernel size for convolutional layers
    pub kernel_size: usize,

    /// Number of units in fully connected layers
    pub fc_units: Vec<usize>,

    /// Whether to use batch normalization
    pub use_batch_norm: bool,

    /// Activation function to use
    pub activation: ActivationType,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_classes: 38,
            input_size: 128,
            input_channels: 3,
            dropout_rate: 0.5,
            conv_filters: vec![32, 64, 128, 256],
            kernel_size: 3,
            fc_units: vec![512, 256],
            use_batch_norm: true,
            activation: ActivationType::ReLU,
        }
    }
}

impl ModelConfig {
    /// Create a new model configuration with custom parameters
    pub fn new(num_classes: usize, input_size: usize) -> Self {
        Self {
            num_classes,
            input_size,
            ..Default::default()
        }
    }

    /// Create a lightweight model for edge deployment
    pub fn edge_optimized() -> Self {
        Self {
            num_classes: 38,
            input_size: 128, // Smaller input for faster inference
            input_channels: 3,
            dropout_rate: 0.3,
            conv_filters: vec![16, 32, 64, 128],
            kernel_size: 3,
            fc_units: vec![256],
            use_batch_norm: true,
            activation: ActivationType::ReLU,
        }
    }

    /// Create a larger model for maximum accuracy
    pub fn high_accuracy() -> Self {
        Self {
            num_classes: 38,
            input_size: 256,
            input_channels: 3,
            dropout_rate: 0.5,
            conv_filters: vec![64, 128, 256, 512],
            kernel_size: 3,
            fc_units: vec![1024, 512],
            use_batch_norm: true,
            activation: ActivationType::ReLU,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_classes == 0 {
            return Err("num_classes must be greater than 0".to_string());
        }

        if self.input_size == 0 || self.input_size % 32 != 0 {
            return Err("input_size must be a positive multiple of 32".to_string());
        }

        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err("dropout_rate must be in range [0.0, 1.0)".to_string());
        }

        if self.conv_filters.is_empty() {
            return Err("conv_filters must have at least one layer".to_string());
        }

        if self.kernel_size < 1 || self.kernel_size % 2 == 0 {
            return Err("kernel_size must be a positive odd number".to_string());
        }

        Ok(())
    }

    /// Calculate the expected output size after all conv layers
    pub fn calculate_conv_output_size(&self) -> usize {
        let mut size = self.input_size;

        // Each conv block: conv -> bn -> relu -> maxpool(2x2)
        for _ in &self.conv_filters {
            size = size / 2; // MaxPool2d with kernel_size=2, stride=2
        }

        // Final feature map size
        let final_filters = *self.conv_filters.last().unwrap_or(&64);
        size * size * final_filters
    }

    /// Save configuration to a JSON file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load configuration from a JSON file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Supported activation functions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ActivationType {
    ReLU,
    LeakyReLU,
    GELU,
    Swish,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Initial learning rate
    pub learning_rate: f64,

    /// Weight decay (L2 regularization)
    pub weight_decay: f64,

    /// Learning rate scheduler type
    pub lr_scheduler: LRSchedulerType,

    /// Warmup epochs for learning rate
    pub warmup_epochs: usize,

    /// Gradient clipping max norm (None to disable)
    pub gradient_clip: Option<f64>,

    /// Early stopping patience (epochs without improvement)
    pub early_stopping_patience: Option<usize>,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Number of data loading workers
    pub num_workers: usize,

    /// Save checkpoint every N epochs
    pub checkpoint_interval: usize,

    /// Path to save checkpoints
    pub checkpoint_dir: String,

    /// Enable mixed precision training
    pub mixed_precision: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 64,
            learning_rate: 0.001,
            weight_decay: 1e-4,
            lr_scheduler: LRSchedulerType::CosineAnnealing,
            warmup_epochs: 5,
            gradient_clip: Some(1.0),
            early_stopping_patience: Some(10),
            seed: 42,
            num_workers: 4,
            checkpoint_interval: 5,
            checkpoint_dir: "output/checkpoints".to_string(),
            mixed_precision: false,
        }
    }
}

impl TrainingConfig {
    /// Create a fast training config for debugging
    pub fn debug() -> Self {
        Self {
            epochs: 5,
            batch_size: 16,
            learning_rate: 0.001,
            weight_decay: 0.0,
            lr_scheduler: LRSchedulerType::Constant,
            warmup_epochs: 0,
            gradient_clip: None,
            early_stopping_patience: None,
            seed: 42,
            num_workers: 2,
            checkpoint_interval: 1,
            checkpoint_dir: "output/debug_checkpoints".to_string(),
            mixed_precision: false,
        }
    }

    /// Create config optimized for Jetson edge device
    pub fn jetson_optimized() -> Self {
        Self {
            epochs: 30,
            batch_size: 8, // Smaller batch for limited memory
            learning_rate: 0.0005,
            weight_decay: 1e-4,
            lr_scheduler: LRSchedulerType::CosineAnnealing,
            warmup_epochs: 3,
            gradient_clip: Some(1.0),
            early_stopping_patience: Some(5),
            seed: 42,
            num_workers: 2,
            checkpoint_interval: 5,
            checkpoint_dir: "output/jetson_checkpoints".to_string(),
            mixed_precision: true, // Use FP16 for speed
        }
    }
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LRSchedulerType {
    /// Constant learning rate
    Constant,
    /// Step decay at specified epochs
    StepDecay,
    /// Exponential decay
    Exponential,
    /// Cosine annealing
    CosineAnnealing,
    /// Reduce on plateau
    ReduceOnPlateau,
}

impl Default for LRSchedulerType {
    fn default() -> Self {
        Self::CosineAnnealing
    }
}

/// Semi-supervised learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemiSupervisedConfig {
    /// Confidence threshold for pseudo-labeling (0.0 to 1.0)
    pub confidence_threshold: f64,

    /// Weight for unlabeled loss in total loss
    pub unlabeled_loss_weight: f64,

    /// Ramp-up epochs for unlabeled loss weight
    pub ramp_up_epochs: usize,

    /// Maximum number of pseudo-labels per class (to prevent class imbalance)
    pub max_pseudo_per_class: Option<usize>,

    /// Retrain after accumulating this many pseudo-labels
    pub retrain_threshold: usize,

    /// Number of epochs per retraining cycle
    pub epochs_per_retrain: usize,

    /// Use exponential moving average for teacher model
    pub use_ema: bool,

    /// EMA decay rate (if use_ema is true)
    pub ema_decay: f64,
}

impl Default for SemiSupervisedConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            unlabeled_loss_weight: 1.0,
            ramp_up_epochs: 10,
            max_pseudo_per_class: Some(500),
            retrain_threshold: 200,
            epochs_per_retrain: 10,
            use_ema: false,
            ema_decay: 0.999,
        }
    }
}

impl SemiSupervisedConfig {
    /// Conservative settings (high threshold, less pseudo-labels)
    pub fn conservative() -> Self {
        Self {
            confidence_threshold: 0.95,
            unlabeled_loss_weight: 0.5,
            ramp_up_epochs: 15,
            max_pseudo_per_class: Some(200),
            retrain_threshold: 300,
            epochs_per_retrain: 15,
            use_ema: true,
            ema_decay: 0.999,
        }
    }

    /// Aggressive settings (lower threshold, more pseudo-labels)
    pub fn aggressive() -> Self {
        Self {
            confidence_threshold: 0.8,
            unlabeled_loss_weight: 1.5,
            ramp_up_epochs: 5,
            max_pseudo_per_class: Some(1000),
            retrain_threshold: 100,
            epochs_per_retrain: 5,
            use_ema: false,
            ema_decay: 0.999,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.num_classes, 39);
        assert_eq!(config.input_size, 256);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_model_config_validation() {
        let mut config = ModelConfig::default();
        config.num_classes = 0;
        assert!(config.validate().is_err());

        config = ModelConfig::default();
        config.input_size = 100; // Not a multiple of 32
        assert!(config.validate().is_err());

        config = ModelConfig::default();
        config.dropout_rate = 1.5; // Out of range
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_conv_output_size_calculation() {
        let config = ModelConfig {
            input_size: 256,
            conv_filters: vec![32, 64, 128, 256],
            ..Default::default()
        };

        // 256 -> 128 -> 64 -> 32 -> 16, final: 16 * 16 * 256 = 65536
        let expected = 16 * 16 * 256;
        assert_eq!(config.calculate_conv_output_size(), expected);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 50);
        assert_eq!(config.batch_size, 64);
    }

    #[test]
    fn test_semi_supervised_config() {
        let config = SemiSupervisedConfig::default();
        assert_eq!(config.confidence_threshold, 0.9);
        assert_eq!(config.retrain_threshold, 200);
    }
}
