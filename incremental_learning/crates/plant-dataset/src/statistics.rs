//! Dataset statistics computation.

use plant_core::Result;
use std::collections::HashMap;

/// Dataset statistics
pub struct DatasetStatistics {
    pub num_samples: usize,
    pub num_classes: usize,
    pub class_distribution: HashMap<usize, usize>,
    pub mean_image_size: (u32, u32),
}

impl DatasetStatistics {
    pub fn new() -> Self {
        Self {
            num_samples: 0,
            num_classes: 0,
            class_distribution: HashMap::new(),
            mean_image_size: (0, 0),
        }
    }
}

impl Default for DatasetStatistics {
    fn default() -> Self {
        Self::new()
    }
}
