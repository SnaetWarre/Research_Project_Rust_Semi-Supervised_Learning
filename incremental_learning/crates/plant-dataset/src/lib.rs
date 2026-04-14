//! Plant dataset loading and preprocessing library.
//!
//! This crate provides functionality for loading, preprocessing, and augmenting
//! plant disease images for training and evaluation.

pub mod augmentation;
pub mod dataset;
pub mod loader;
pub mod preprocess;
pub mod statistics;

pub use augmentation::AugmentationPipeline;
pub use dataset::{PlantDataset, PlantItem, PlantBatch};
pub use loader::ImageLoader;
pub use preprocess::{ImagePreprocessor, PreprocessConfig};
pub use statistics::DatasetStatistics;

use plant_core::{Error, Result};

/// Re-export commonly used types
pub mod prelude {
    pub use crate::augmentation::*;
    pub use crate::dataset::*;
    pub use crate::loader::*;
    pub use crate::preprocess::*;
    pub use crate::statistics::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_compiles() {
        // Basic smoke test
        assert!(true);
    }
}
