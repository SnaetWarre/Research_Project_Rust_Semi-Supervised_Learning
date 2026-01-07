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
pub mod prepare;
pub mod split;

// Re-export main types for convenience
pub use burn_dataset::{
    CombinedDataset, PlantVillageBatch, PlantVillageBatcher, PlantVillageBurnDataset,
    PlantVillageItem, PseudoLabelBatch, PseudoLabelBatcher, PseudoLabelDataset, PseudoLabeledItem,
};
pub use loader::{DatasetStats, ImageSample, PlantVillageDataset};
pub use prepare::{compute_class_weights, prepare_balanced_dataset, PrepareConfig, PrepareStats};
pub use split::{DatasetSplits, HiddenLabelImage, LabeledImage, PseudoLabeledImage, SplitConfig};

/// Total number of classes in PlantVillage dataset
pub const NUM_CLASSES: usize = 38;

/// Default image dimensions (PlantVillage images are 256x256)
pub const DEFAULT_IMAGE_SIZE: usize = 256;

/// Class names for PlantVillage dataset (38 classes)
/// IMPORTANT: These MUST be in alphabetical order to match the dataset loader!
/// The loader assigns indices based on sorted directory names.
pub const CLASS_NAMES: [&str; 38] = [
    "Apple___Apple_scab",                                   // 0
    "Apple___Black_rot",                                    // 1
    "Apple___Cedar_apple_rust",                             // 2
    "Apple___healthy",                                      // 3
    "Blueberry___healthy",                                  // 4
    "Cherry_(including_sour)___healthy",                    // 5  (healthy before Powdery!)
    "Cherry_(including_sour)___Powdery_mildew",             // 6
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",   // 7
    "Corn_(maize)___Common_rust_",                          // 8
    "Corn_(maize)___healthy",                               // 9  (healthy before Northern!)
    "Corn_(maize)___Northern_Leaf_Blight",                  // 10
    "Grape___Black_rot",                                    // 11
    "Grape___Esca_(Black_Measles)",                         // 12
    "Grape___healthy",                                      // 13 (healthy before Leaf_blight!)
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",           // 14
    "Orange___Haunglongbing_(Citrus_greening)",             // 15
    "Peach___Bacterial_spot",                               // 16
    "Peach___healthy",                                      // 17
    "Pepper,_bell___Bacterial_spot",                        // 18
    "Pepper,_bell___healthy",                               // 19
    "Potato___Early_blight",                                // 20
    "Potato___healthy",                                     // 21 (healthy before Late!)
    "Potato___Late_blight",                                 // 22
    "Raspberry___healthy",                                  // 23
    "Soybean___healthy",                                    // 24
    "Squash___Powdery_mildew",                              // 25
    "Strawberry___healthy",                                 // 26 (healthy before Leaf_scorch!)
    "Strawberry___Leaf_scorch",                             // 27
    "Tomato___Bacterial_spot",                              // 28
    "Tomato___Early_blight",                                // 29
    "Tomato___healthy",                                     // 30 (healthy before Late!)
    "Tomato___Late_blight",                                 // 31
    "Tomato___Leaf_Mold",                                   // 32
    "Tomato___Septoria_leaf_spot",                          // 33
    "Tomato___Spider_mites Two-spotted_spider_mite",        // 34
    "Tomato___Target_Spot",                                 // 35
    "Tomato___Tomato_mosaic_virus",                         // 36 (mosaic before Yellow!)
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",               // 37
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
        assert_eq!(class_name(37), Some("Tomato___Tomato_Yellow_Leaf_Curl_Virus"));
        assert_eq!(class_name(100), None);
    }

    #[test]
    fn test_class_index() {
        assert_eq!(class_index("Apple___Apple_scab"), Some(0));
        assert_eq!(class_index("Tomato___healthy"), Some(30));
        assert_eq!(class_index("Unknown___class"), None);
    }

    #[test]
    fn test_is_healthy_class() {
        assert!(!is_healthy_class(0)); // Apple___Apple_scab
        assert!(is_healthy_class(3)); // Apple___healthy
        assert!(is_healthy_class(30)); // Tomato___healthy
    }

    #[test]
    fn test_plant_name() {
        assert_eq!(plant_name(0), Some("Apple"));
        assert_eq!(plant_name(28), Some("Tomato"));
    }
}
