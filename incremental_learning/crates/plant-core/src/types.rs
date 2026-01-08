//! Core type definitions for the plant incremental learning project.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Represents a plant class/category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct PlantClass {
    /// Unique identifier for the class
    pub id: usize,
    /// Human-readable name (e.g., "Tomato___Late_blight")
    pub name: String,
    /// Optional description
    pub description: Option<String>,
}

impl PlantClass {
    /// Creates a new plant class
    pub fn new(id: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: None,
        }
    }

    /// Creates a new plant class with description
    pub fn with_description(id: usize, name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            description: Some(description.into()),
        }
    }
}

/// Represents an image sample with its label
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSample {
    /// Path to the image file
    pub path: PathBuf,
    /// Class label (index)
    pub label: usize,
    /// Optional class name
    pub class_name: Option<String>,
}

impl ImageSample {
    /// Creates a new image sample
    pub fn new(path: PathBuf, label: usize) -> Self {
        Self {
            path,
            label,
            class_name: None,
        }
    }

    /// Creates a new image sample with class name
    pub fn with_class_name(path: PathBuf, label: usize, class_name: impl Into<String>) -> Self {
        Self {
            path,
            label,
            class_name: Some(class_name.into()),
        }
    }
}

/// Data split type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DataSplit {
    /// Training data
    Train,
    /// Validation data
    Validation,
    /// Test data
    Test,
}

impl std::fmt::Display for DataSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataSplit::Train => write!(f, "train"),
            DataSplit::Validation => write!(f, "validation"),
            DataSplit::Test => write!(f, "test"),
        }
    }
}

/// Model architecture type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// EfficientNet-B0
    EfficientNetB0,
    /// EfficientNet-B1
    EfficientNetB1,
    /// ResNet-18
    ResNet18,
    /// ResNet-34
    ResNet34,
    /// ResNet-50
    ResNet50,
    /// MobileNet-V2
    MobileNetV2,
    /// MobileNet-V3 Small
    MobileNetV3Small,
    /// MobileNet-V3 Large
    MobileNetV3Large,
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::EfficientNetB0 => write!(f, "efficientnet_b0"),
            ModelArchitecture::EfficientNetB1 => write!(f, "efficientnet_b1"),
            ModelArchitecture::ResNet18 => write!(f, "resnet18"),
            ModelArchitecture::ResNet34 => write!(f, "resnet34"),
            ModelArchitecture::ResNet50 => write!(f, "resnet50"),
            ModelArchitecture::MobileNetV2 => write!(f, "mobilenet_v2"),
            ModelArchitecture::MobileNetV3Small => write!(f, "mobilenet_v3_small"),
            ModelArchitecture::MobileNetV3Large => write!(f, "mobilenet_v3_large"),
        }
    }
}

/// Incremental learning method/strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IncrementalMethod {
    /// Simple fine-tuning (baseline)
    FineTuning,
    /// Learning without Forgetting
    LwF,
    /// Elastic Weight Consolidation
    EWC,
    /// Rehearsal/Memory replay
    Rehearsal,
    /// iCaRL (Incremental Classifier and Representation Learning)
    ICaRL,
}

impl std::fmt::Display for IncrementalMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncrementalMethod::FineTuning => write!(f, "fine_tuning"),
            IncrementalMethod::LwF => write!(f, "lwf"),
            IncrementalMethod::EWC => write!(f, "ewc"),
            IncrementalMethod::Rehearsal => write!(f, "rehearsal"),
            IncrementalMethod::ICaRL => write!(f, "icarl"),
        }
    }
}

/// Training phase
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Initial training on base classes
    BaseTraining,
    /// Incremental learning phase
    IncrementalTraining,
}

/// Device backend type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU backend
    Cpu,
    /// CUDA/GPU backend
    Cuda,
    /// WebGPU backend
    Wgpu,
    /// LibTorch backend
    Tch,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda => write!(f, "cuda"),
            DeviceType::Wgpu => write!(f, "wgpu"),
            DeviceType::Tch => write!(f, "tch"),
        }
    }
}

/// Image dimensions
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImageDimensions {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Number of channels (e.g., 3 for RGB)
    pub channels: u32,
}

impl ImageDimensions {
    /// Creates new image dimensions
    pub fn new(width: u32, height: u32, channels: u32) -> Self {
        Self {
            width,
            height,
            channels,
        }
    }

    /// Standard ImageNet dimensions (224x224x3)
    pub fn imagenet() -> Self {
        Self::new(224, 224, 3)
    }

    /// Total number of pixels
    pub fn total_pixels(&self) -> u32 {
        self.width * self.height * self.channels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plant_class_creation() {
        let class = PlantClass::new(0, "Tomato___Healthy");
        assert_eq!(class.id, 0);
        assert_eq!(class.name, "Tomato___Healthy");
        assert!(class.description.is_none());
    }

    #[test]
    fn test_plant_class_with_description() {
        let class = PlantClass::with_description(
            1,
            "Tomato___Late_blight",
            "Late blight disease in tomato plants"
        );
        assert_eq!(class.id, 1);
        assert!(class.description.is_some());
    }

    #[test]
    fn test_image_sample() {
        let sample = ImageSample::new(PathBuf::from("test.jpg"), 0);
        assert_eq!(sample.label, 0);
        assert!(sample.class_name.is_none());
    }

    #[test]
    fn test_data_split_display() {
        assert_eq!(DataSplit::Train.to_string(), "train");
        assert_eq!(DataSplit::Validation.to_string(), "validation");
        assert_eq!(DataSplit::Test.to_string(), "test");
    }

    #[test]
    fn test_model_architecture_display() {
        assert_eq!(ModelArchitecture::EfficientNetB0.to_string(), "efficientnet_b0");
        assert_eq!(ModelArchitecture::ResNet18.to_string(), "resnet18");
    }

    #[test]
    fn test_incremental_method_display() {
        assert_eq!(IncrementalMethod::FineTuning.to_string(), "fine_tuning");
        assert_eq!(IncrementalMethod::LwF.to_string(), "lwf");
        assert_eq!(IncrementalMethod::EWC.to_string(), "ewc");
    }

    #[test]
    fn test_image_dimensions() {
        let dims = ImageDimensions::imagenet();
        assert_eq!(dims.width, 224);
        assert_eq!(dims.height, 224);
        assert_eq!(dims.channels, 3);
        assert_eq!(dims.total_pixels(), 224 * 224 * 3);
    }
}
