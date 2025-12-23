//! Minimal Data Augmentation Module
//!
//! Simplified augmentation functionality for PlantVillage training.

use image::{DynamicImage, GenericImageView, Rgb};

/// Simple augmenter with basic augmentations
pub struct Augmenter {
    image_size: u32,
}

impl Augmenter {
    pub fn new(image_size: u32) -> Self {
        Self { image_size }
    }

    /// Apply random horizontal flip
    pub fn flip(&self, mut img: DynamicImage) -> DynamicImage {
        img.fliph()
    }

    /// Resize image to target size
    pub fn resize(&self, img: DynamicImage) -> DynamicImage {
        img.resize_exact(self.image_size, self.image_size)
    }

    /// Normalize image to [0, 1] range
    pub fn normalize(&self, img: DynamicImage) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut data = Vec::with_capacity(3 * width as usize * height as usize);

        for pixel in rgb.pixels() {
            data.push(pixel[0] as f32 / 255.0);
            data.push(pixel[1] as f32 / 255.0);
            data.push(pixel[2] as f32 / 255.0);
        }

        data
    }
}
