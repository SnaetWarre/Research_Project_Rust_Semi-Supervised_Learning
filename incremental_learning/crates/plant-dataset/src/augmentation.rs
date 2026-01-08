//! Data augmentation functionality for plant images.
//!
//! This module provides various image augmentation techniques to improve
//! model generalization and handle limited training data.

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use plant_core::{AugmentationConfig, Error, Result};
use rand::Rng;

/// Augmentation pipeline for plant images
pub struct AugmentationPipeline {
    config: AugmentationConfig,
}

impl AugmentationPipeline {
    /// Creates a new augmentation pipeline with the given configuration
    pub fn new(config: AugmentationConfig) -> Self {
        Self {
            config,
        }
    }

    /// Creates a pipeline with default configuration
    pub fn default() -> Self {
        Self::new(AugmentationConfig::default())
    }

    /// Applies augmentation to an image
    pub fn augment(&mut self, image: &DynamicImage) -> Result<DynamicImage> {
        let mut augmented = image.clone();

        let mut rng = rand::thread_rng();

        // Random horizontal flip
        if self.config.horizontal_flip && rng.gen_bool(0.5) {
            augmented = self.flip_horizontal(&augmented);
        }

        // Random vertical flip
        if self.config.vertical_flip && rng.gen_bool(0.5) {
            augmented = self.flip_vertical(&augmented);
        }

        // Random rotation
        if self.config.rotation_range > 0.0 {
            let angle = rng.gen_range(-self.config.rotation_range..=self.config.rotation_range);
            if angle.abs() > 0.1 {
                augmented = self.rotate(&augmented, angle)?;
            }
        }

        // Random brightness adjustment
        if self.config.brightness_range != (1.0, 1.0) {
            let factor = rng.gen_range(self.config.brightness_range.0..=self.config.brightness_range.1);
            augmented = self.adjust_brightness(&augmented, factor);
        }

        // Random contrast adjustment
        if self.config.contrast_range != (1.0, 1.0) {
            let factor = rng.gen_range(self.config.contrast_range.0..=self.config.contrast_range.1);
            augmented = self.adjust_contrast(&augmented, factor);
        }

        // Random saturation adjustment
        if self.config.saturation_range != (1.0, 1.0) {
            let factor = rng.gen_range(self.config.saturation_range.0..=self.config.saturation_range.1);
            augmented = self.adjust_saturation(&augmented, factor);
        }

        // Random zoom/crop
        if self.config.random_crop && rng.gen_bool(0.3) {
            augmented = self.random_crop(&augmented, &mut rng)?;
        }

        Ok(augmented)
    }

    /// Flips an image horizontally
    fn flip_horizontal(&self, image: &DynamicImage) -> DynamicImage {
        DynamicImage::ImageRgb8(image::imageops::flip_horizontal(&image.to_rgb8()))
    }

    /// Flips an image vertically
    fn flip_vertical(&self, image: &DynamicImage) -> DynamicImage {
        DynamicImage::ImageRgb8(image::imageops::flip_vertical(&image.to_rgb8()))
    }

    /// Rotates an image by the given angle (in degrees)
    fn rotate(&self, image: &DynamicImage, angle: f32) -> Result<DynamicImage> {
        // For simplicity, we'll use 90-degree rotations
        // More sophisticated rotation would require additional dependencies
        let normalized_angle = ((angle % 360.0 + 360.0) % 360.0) as i32;

        let rotated = match normalized_angle {
            45..=135 => image::imageops::rotate90(&image.to_rgb8()),
            136..=225 => image::imageops::rotate180(&image.to_rgb8()),
            226..=315 => image::imageops::rotate270(&image.to_rgb8()),
            _ => image.to_rgb8(),
        };

        Ok(DynamicImage::ImageRgb8(rotated))
    }

    /// Adjusts image brightness
    fn adjust_brightness(&self, image: &DynamicImage, factor: f32) -> DynamicImage {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();

        let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = rgb.get_pixel(x, y);
            Rgb([
                (pixel[0] as f32 * factor).clamp(0.0, 255.0) as u8,
                (pixel[1] as f32 * factor).clamp(0.0, 255.0) as u8,
                (pixel[2] as f32 * factor).clamp(0.0, 255.0) as u8,
            ])
        });

        DynamicImage::ImageRgb8(adjusted)
    }

    /// Adjusts image contrast
    fn adjust_contrast(&self, image: &DynamicImage, factor: f32) -> DynamicImage {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();

        // Calculate mean intensity
        let mut sum = 0.0;
        let total_pixels = (width * height) as f32;

        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x, y);
                let intensity = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;
                sum += intensity;
            }
        }
        let mean = sum / total_pixels;

        let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = rgb.get_pixel(x, y);
            Rgb([
                (mean + factor * (pixel[0] as f32 - mean)).clamp(0.0, 255.0) as u8,
                (mean + factor * (pixel[1] as f32 - mean)).clamp(0.0, 255.0) as u8,
                (mean + factor * (pixel[2] as f32 - mean)).clamp(0.0, 255.0) as u8,
            ])
        });

        DynamicImage::ImageRgb8(adjusted)
    }

    /// Adjusts image saturation
    fn adjust_saturation(&self, image: &DynamicImage, factor: f32) -> DynamicImage {
        let rgb = image.to_rgb8();
        let (width, height) = rgb.dimensions();

        let adjusted = ImageBuffer::from_fn(width, height, |x, y| {
            let pixel = rgb.get_pixel(x, y);

            // Convert to grayscale (luminance)
            let gray = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;

            // Interpolate between grayscale and original based on factor
            Rgb([
                (gray as f32 + factor * (pixel[0] as f32 - gray as f32)).clamp(0.0, 255.0) as u8,
                (gray as f32 + factor * (pixel[1] as f32 - gray as f32)).clamp(0.0, 255.0) as u8,
                (gray as f32 + factor * (pixel[2] as f32 - gray as f32)).clamp(0.0, 255.0) as u8,
            ])
        });

        DynamicImage::ImageRgb8(adjusted)
    }

    /// Performs random crop
    fn random_crop(&self, image: &DynamicImage, rng: &mut rand::rngs::ThreadRng) -> Result<DynamicImage> {
        let (width, height) = image.dimensions();

        // Zoom factor between range
        let zoom_factor = rng.gen_range(self.config.zoom_range.0..=self.config.zoom_range.1);

        let crop_width = (width as f32 / zoom_factor) as u32;
        let crop_height = (height as f32 / zoom_factor) as u32;

        if crop_width == 0 || crop_height == 0 || crop_width > width || crop_height > height {
            return Ok(image.clone());
        }

        let x = rng.gen_range(0..=(width - crop_width));
        let y = rng.gen_range(0..=(height - crop_height));

        let cropped = image.crop_imm(x, y, crop_width, crop_height);

        // Resize back to original dimensions
        let resized = cropped.resize_exact(width, height, image::imageops::FilterType::Lanczos3);

        Ok(resized)
    }

    /// Applies augmentation multiple times to create variations
    pub fn augment_multiple(&mut self, image: &DynamicImage, count: usize) -> Result<Vec<DynamicImage>> {
        let mut augmented_images = Vec::with_capacity(count);

        for _ in 0..count {
            augmented_images.push(self.augment(image)?);
        }

        Ok(augmented_images)
    }
}

/// Simple augmentation presets
pub struct AugmentationPresets;

impl AugmentationPresets {
    /// Light augmentation (for clean datasets like PlantVillage)
    pub fn light() -> AugmentationConfig {
        AugmentationConfig {
            rotation_range: 10.0,
            horizontal_flip: true,
            vertical_flip: false,
            brightness_range: (0.9, 1.1),
            contrast_range: (0.9, 1.1),
            saturation_range: (0.95, 1.05),
            random_crop: true,
            zoom_range: (0.95, 1.05),
        }
    }

    /// Medium augmentation (balanced)
    pub fn medium() -> AugmentationConfig {
        AugmentationConfig {
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

    /// Heavy augmentation (for very small datasets or challenging conditions)
    pub fn heavy() -> AugmentationConfig {
        AugmentationConfig {
            rotation_range: 30.0,
            horizontal_flip: true,
            vertical_flip: true,
            brightness_range: (0.7, 1.3),
            contrast_range: (0.7, 1.3),
            saturation_range: (0.7, 1.3),
            random_crop: true,
            zoom_range: (0.85, 1.15),
        }
    }

    /// No augmentation (identity transform)
    pub fn none() -> AugmentationConfig {
        AugmentationConfig {
            rotation_range: 0.0,
            horizontal_flip: false,
            vertical_flip: false,
            brightness_range: (1.0, 1.0),
            contrast_range: (1.0, 1.0),
            saturation_range: (1.0, 1.0),
            random_crop: false,
            zoom_range: (1.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image() -> DynamicImage {
        let img = ImageBuffer::from_fn(100, 100, |x, y| {
            if x < 50 && y < 50 {
                Rgb([255u8, 0u8, 0u8])  // Red
            } else if x >= 50 && y < 50 {
                Rgb([0u8, 255u8, 0u8])  // Green
            } else if x < 50 && y >= 50 {
                Rgb([0u8, 0u8, 255u8])  // Blue
            } else {
                Rgb([255u8, 255u8, 0u8])  // Yellow
            }
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_pipeline_creation() {
        let config = AugmentationConfig::default();
        let _pipeline = AugmentationPipeline::new(config);
        assert!(true); // Just test creation
    }

    #[test]
    fn test_flip_horizontal() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();
        let flipped = pipeline.flip_horizontal(&image);

        assert_eq!(flipped.dimensions(), image.dimensions());
    }

    #[test]
    fn test_flip_vertical() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();
        let flipped = pipeline.flip_vertical(&image);

        assert_eq!(flipped.dimensions(), image.dimensions());
    }

    #[test]
    fn test_brightness_adjustment() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();

        let brighter = pipeline.adjust_brightness(&image, 1.5);
        assert_eq!(brighter.dimensions(), image.dimensions());

        let darker = pipeline.adjust_brightness(&image, 0.5);
        assert_eq!(darker.dimensions(), image.dimensions());
    }

    #[test]
    fn test_contrast_adjustment() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();

        let high_contrast = pipeline.adjust_contrast(&image, 1.5);
        assert_eq!(high_contrast.dimensions(), image.dimensions());
    }

    #[test]
    fn test_saturation_adjustment() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();

        let saturated = pipeline.adjust_saturation(&image, 1.5);
        assert_eq!(saturated.dimensions(), image.dimensions());

        let desaturated = pipeline.adjust_saturation(&image, 0.5);
        assert_eq!(desaturated.dimensions(), image.dimensions());
    }

    #[test]
    fn test_augment() {
        let mut pipeline = AugmentationPipeline::new(AugmentationPresets::light());
        let image = create_test_image();

        let result = pipeline.augment(&image);
        assert!(result.is_ok());

        let augmented = result.unwrap();
        assert_eq!(augmented.dimensions(), image.dimensions());
    }

    #[test]
    fn test_augment_multiple() {
        let mut pipeline = AugmentationPipeline::new(AugmentationPresets::light());
        let image = create_test_image();

        let result = pipeline.augment_multiple(&image, 5);
        assert!(result.is_ok());

        let augmented_images = result.unwrap();
        assert_eq!(augmented_images.len(), 5);

        for aug_image in &augmented_images {
            assert_eq!(aug_image.dimensions(), image.dimensions());
        }
    }

    #[test]
    fn test_presets() {
        let light = AugmentationPresets::light();
        assert_eq!(light.rotation_range, 10.0);

        let medium = AugmentationPresets::medium();
        assert_eq!(medium.rotation_range, 20.0);

        let heavy = AugmentationPresets::heavy();
        assert_eq!(heavy.rotation_range, 30.0);

        let none = AugmentationPresets::none();
        assert_eq!(none.rotation_range, 0.0);
    }

    #[test]
    fn test_rotate() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();

        let rotated = pipeline.rotate(&image, 90.0);
        assert!(rotated.is_ok());
    }

    #[test]
    fn test_random_crop() {
        let pipeline = AugmentationPipeline::default();
        let image = create_test_image();
        let mut rng = rand::thread_rng();

        let cropped = pipeline.random_crop(&image, &mut rng);
        assert!(cropped.is_ok());

        let result = cropped.unwrap();
        assert_eq!(result.dimensions(), image.dimensions());
    }
}
