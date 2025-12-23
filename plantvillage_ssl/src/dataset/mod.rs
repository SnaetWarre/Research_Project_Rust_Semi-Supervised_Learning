//! Dataset module for PlantVillage data handling
//!
//! This module provides functionality for:
//! - Loading the PlantVillage dataset from disk
//! - Data augmentation for training robustness
//! - Dataset splitting with semi-supervised learning simulation
//!
//! ## Semi-Supervised Split Strategy
//!
//! The dataset is split into several pools to simulate a real-world scenario:
//! 1. **Test Set**: 10% of data, held out for final evaluation (never seen during training)
//! 2. **Validation Set**: 10% of data, for hyperparameter tuning
//! 3. **Labeled Pool**: 20-30% of remaining data, simulating manually labeled images
//! 4. **Stream Pool**: Remaining data, simulating "incoming" camera captures
//! 5. **Pseudo-labeled Pool**: Images from stream pool that have been pseudo-labeled

pub mod burn_dataset;
pub mod loader;
pub mod split;

// Re-export main types for convenience
pub use burn_dataset::{
    CombinedDataset, PlantVillageBatch, PlantVillageBatcher, PlantVillageBurnDataset,
    PlantVillageItem, PseudoLabelBatch, PseudoLabelBatcher, PseudoLabelDataset, PseudoLabeledItem,
};
pub use loader::{DatasetStats, ImageSample, PlantVillageDataset};
pub use split::{DatasetSplits, HiddenLabelImage, LabeledImage, PseudoLabeledImage, SplitConfig};

/// Total number of classes in PlantVillage dataset
pub const NUM_CLASSES: usize = 39;

/// Default image dimensions (PlantVillage images are 256x256)
pub const DEFAULT_IMAGE_SIZE: usize = 256;

/// Class names for PlantVillage dataset (39 classes)
/// Format: "Plant___Disease" or "Plant___healthy"
pub const CLASS_NAMES: [&str; 39] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
    "Background_without_leaves", // Note: Some versions include this class
];

/// Get the class name for a given label index
pub fn class_name(label: usize) -> Option<&'static str> {
    CLASS_NAMES.get(label).copied()
}

/// Get the label index for a given class name
pub fn class_index(name: &str) -> Option<usize> {
    CLASS_NAMES.iter().position(|&n| n == name)
}

/// Check if a class represents a healthy plant (not diseased)
pub fn is_healthy_class(label: usize) -> bool {
    CLASS_NAMES
        .get(label)
        .map(|name| name.ends_with("healthy"))
        .unwrap_or(false)
}

/// Get the plant name from a class (e.g., "Tomato" from "Tomato___Bacterial_spot")
pub fn plant_name(label: usize) -> Option<&'static str> {
    CLASS_NAMES
        .get(label)
        .and_then(|name| name.split("___").next())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_class_name() {
        assert_eq!(class_name(0), Some("Apple___Apple_scab"));
        assert_eq!(class_name(38), Some("Background_without_leaves"));
        assert_eq!(class_name(100), None);
    }

    #[test]
    fn test_class_index() {
        assert_eq!(class_index("Apple___Apple_scab"), Some(0));
        assert_eq!(class_index("Tomato___healthy"), Some(37));
        assert_eq!(class_index("Unknown___class"), None);
    }

    #[test]
    fn test_is_healthy_class() {
        assert!(!is_healthy_class(0)); // Apple___Apple_scab
        assert!(is_healthy_class(3)); // Apple___healthy
        assert!(is_healthy_class(37)); // Tomato___healthy
    }

    #[test]
    fn test_plant_name() {
        assert_eq!(plant_name(0), Some("Apple"));
        assert_eq!(plant_name(28), Some("Tomato"));
    }
}
