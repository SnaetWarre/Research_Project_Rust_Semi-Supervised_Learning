//! Dataset implementation for plant images.

use plant_core::{Result, ImageSample};

/// A single item from the plant dataset
pub struct PlantItem {
    pub image_tensor: Vec<f32>,
    pub label: usize,
}

/// A batch of plant items
pub struct PlantBatch<B> {
    pub images: B,
    pub labels: B,
}

/// Plant dataset structure
pub struct PlantDataset {
    samples: Vec<ImageSample>,
}

impl PlantDataset {
    pub fn new(samples: Vec<ImageSample>) -> Self {
        Self { samples }
    }
    
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}
