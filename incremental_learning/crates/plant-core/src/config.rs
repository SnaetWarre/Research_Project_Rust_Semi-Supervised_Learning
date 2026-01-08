//! Configuration structures for the plant incremental learning project.

use crate::types::{DeviceType, ImageDimensions, IncrementalMethod, ModelArchitecture};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model configuration
    pub model: ModelConfig,
    /// Training hyperparameters
    pub training: TrainingParams,
    /// Data configuration
    pub data: DataConfig,
    /// Device configuration
    pub device: DeviceConfig,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingParams::default(),
            data: DataConfig::default(),
            device: DeviceConfig::default(),
            seed: 42,
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Architecture type
    pub architecture: ModelArchitecture,
    /// Number of output classes
    pub num_classes: usize,
    /// Whether to use pretrained weights
    pub pretrained: bool,
    /// Path to pretrained weights (optional)
    pub pretrained_path: Option<PathBuf>,
    /// Dropout rate
    pub dropout: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::EfficientNetB0,
            num_classes: 5,
            pretrained: true,
            pretrained_path: None,
            dropout: 0.2,
        }
    }
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Learning rate schedule
    pub lr_schedule: Option<LRSchedule>,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Gradient clipping value
    pub grad_clip: Option<f64>,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Number of workers for data loading
    pub num_workers: usize,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            num_epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: OptimizerType::Adam,
            lr_schedule: Some(LRSchedule::default()),
            weight_decay: 1e-4,
            grad_clip: Some(1.0),
            early_stopping_patience: Some(10),
            num_workers: 4,
        }
    }
}

/// Optimizer type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizerType {
    /// Adam optimizer
    Adam,
    /// SGD with momentum
    SGD,
    /// AdamW optimizer
    AdamW,
    /// RMSprop optimizer
    RMSprop,
}

impl std::fmt::Display for OptimizerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerType::Adam => write!(f, "adam"),
            OptimizerType::SGD => write!(f, "sgd"),
            OptimizerType::AdamW => write!(f, "adamw"),
            OptimizerType::RMSprop => write!(f, "rmsprop"),
        }
    }
}

/// Learning rate schedule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LRSchedule {
    /// Schedule type
    pub schedule_type: LRScheduleType,
    /// Step size for step decay
    pub step_size: Option<usize>,
    /// Decay factor
    pub gamma: f64,
    /// Minimum learning rate
    pub min_lr: f64,
}

impl Default for LRSchedule {
    fn default() -> Self {
        Self {
            schedule_type: LRScheduleType::StepLR,
            step_size: Some(10),
            gamma: 0.1,
            min_lr: 1e-6,
        }
    }
}

/// Learning rate schedule type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LRScheduleType {
    /// Step decay
    StepLR,
    /// Cosine annealing
    CosineAnnealing,
    /// Exponential decay
    Exponential,
    /// Reduce on plateau
    ReduceOnPlateau,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Path to data directory
    pub data_dir: PathBuf,
    /// Image dimensions
    pub image_size: ImageDimensions,
    /// Whether to apply data augmentation
    pub augmentation: bool,
    /// Augmentation parameters
    pub augmentation_params: Option<AugmentationConfig>,
    /// Train/val/test split ratios
    pub split_ratios: SplitRatios,
    /// Whether to shuffle training data
    pub shuffle: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data/processed"),
            image_size: ImageDimensions::imagenet(),
            augmentation: true,
            augmentation_params: Some(AugmentationConfig::default()),
            split_ratios: SplitRatios::default(),
            shuffle: true,
        }
    }
}

/// Data augmentation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationConfig {
    /// Random rotation range in degrees
    pub rotation_range: f32,
    /// Horizontal flip probability
    pub horizontal_flip: bool,
    /// Vertical flip probability
    pub vertical_flip: bool,
    /// Brightness adjustment range
    pub brightness_range: (f32, f32),
    /// Contrast adjustment range
    pub contrast_range: (f32, f32),
    /// Saturation adjustment range
    pub saturation_range: (f32, f32),
    /// Random crop probability
    pub random_crop: bool,
    /// Zoom range
    pub zoom_range: (f32, f32),
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            rotation_range: 20.0,
            horizontal_flip: true,
            vertical_flip: false,
            brightness_range: (0.8, 1.2),
            contrast_range: (0.8, 1.2),
            saturation_range: (0.8, 1.2),
            random_crop: true,
            zoom_range: (0.9, 1.1),
        }
    }
}

/// Train/validation/test split ratios
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplitRatios {
    /// Training data ratio
    pub train: f32,
    /// Validation data ratio
    pub validation: f32,
    /// Test data ratio
    pub test: f32,
}

impl Default for SplitRatios {
    fn default() -> Self {
        Self {
            train: 0.7,
            validation: 0.15,
            test: 0.15,
        }
    }
}

impl SplitRatios {
    /// Validates that ratios sum to 1.0
    pub fn validate(&self) -> Result<(), String> {
        let sum = self.train + self.validation + self.test;
        if (sum - 1.0).abs() > 1e-5 {
            return Err(format!(
                "Split ratios must sum to 1.0, got {}",
                sum
            ));
        }
        Ok(())
    }
}

/// Device configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    /// Device type
    pub device_type: DeviceType,
    /// CUDA device ID (if applicable)
    pub cuda_device_id: Option<usize>,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Cpu,
            cuda_device_id: None,
        }
    }
}

/// Incremental learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalConfig {
    /// Incremental learning method
    pub method: IncrementalMethod,
    /// Method-specific parameters
    pub method_params: MethodParams,
    /// Path to base model
    pub base_model_path: PathBuf,
    /// New class information
    pub new_class: NewClassConfig,
    /// Training configuration for incremental phase
    pub training: TrainingParams,
}

/// Method-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodParams {
    /// Learning without Forgetting parameters
    pub lwf: Option<LwFParams>,
    /// Elastic Weight Consolidation parameters
    pub ewc: Option<EWCParams>,
    /// Rehearsal parameters
    pub rehearsal: Option<RehearsalParams>,
    /// Fine-tuning parameters
    pub fine_tuning: Option<FineTuningParams>,
}

impl Default for MethodParams {
    fn default() -> Self {
        Self {
            lwf: None,
            ewc: None,
            rehearsal: None,
            fine_tuning: Some(FineTuningParams::default()),
        }
    }
}

/// Learning without Forgetting parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwFParams {
    /// Distillation temperature
    pub temperature: f64,
    /// Distillation loss weight
    pub distillation_weight: f64,
}

impl Default for LwFParams {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            distillation_weight: 0.5,
        }
    }
}

/// Elastic Weight Consolidation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCParams {
    /// Importance weight for EWC penalty
    pub importance_weight: f64,
    /// Number of samples for Fisher information estimation
    pub fisher_samples: usize,
}

impl Default for EWCParams {
    fn default() -> Self {
        Self {
            importance_weight: 1000.0,
            fisher_samples: 200,
        }
    }
}

/// Rehearsal/memory replay parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RehearsalParams {
    /// Number of samples to keep per old class
    pub samples_per_class: usize,
    /// Memory selection strategy
    pub selection_strategy: MemorySelectionStrategy,
}

impl Default for RehearsalParams {
    fn default() -> Self {
        Self {
            samples_per_class: 20,
            selection_strategy: MemorySelectionStrategy::Random,
        }
    }
}

/// Memory selection strategy for rehearsal
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemorySelectionStrategy {
    /// Random selection
    Random,
    /// Herding (closest to class mean)
    Herding,
    /// Highest confidence
    HighestConfidence,
}

/// Fine-tuning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuningParams {
    /// Fraction of layers to freeze (0.0 = none, 1.0 = all)
    pub freeze_ratio: f64,
    /// Whether to freeze batch norm layers
    pub freeze_bn: bool,
}

impl Default for FineTuningParams {
    fn default() -> Self {
        Self {
            freeze_ratio: 0.5,
            freeze_bn: false,
        }
    }
}

/// New class configuration for incremental learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewClassConfig {
    /// New class ID
    pub class_id: usize,
    /// New class name
    pub class_name: String,
    /// Number of training images (can be multiple for experiments)
    pub num_training_images: Vec<usize>,
    /// Path to new class data
    pub data_path: PathBuf,
}

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment name
    pub name: String,
    /// Experiment description
    pub description: Option<String>,
    /// Incremental learning configuration
    pub incremental: IncrementalConfig,
    /// Data configuration
    pub data: DataConfig,
    /// Device configuration
    pub device: DeviceConfig,
    /// Random seed
    pub seed: u64,
    /// Output directory for results
    pub output_dir: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.seed, 42);
        assert_eq!(config.model.num_classes, 5);
        assert_eq!(config.training.batch_size, 32);
    }

    #[test]
    fn test_split_ratios_validation() {
        let valid = SplitRatios::default();
        assert!(valid.validate().is_ok());

        let invalid = SplitRatios {
            train: 0.5,
            validation: 0.3,
            test: 0.1,
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_optimizer_display() {
        assert_eq!(OptimizerType::Adam.to_string(), "adam");
        assert_eq!(OptimizerType::SGD.to_string(), "sgd");
    }

    #[test]
    fn test_default_augmentation_config() {
        let config = AugmentationConfig::default();
        assert_eq!(config.rotation_range, 20.0);
        assert!(config.horizontal_flip);
        assert!(!config.vertical_flip);
    }

    #[test]
    fn test_default_method_params() {
        let params = MethodParams::default();
        assert!(params.fine_tuning.is_some());
        assert!(params.lwf.is_none());
    }

    #[test]
    fn test_lwf_params() {
        let params = LwFParams::default();
        assert_eq!(params.temperature, 2.0);
        assert_eq!(params.distillation_weight, 0.5);
    }
}
