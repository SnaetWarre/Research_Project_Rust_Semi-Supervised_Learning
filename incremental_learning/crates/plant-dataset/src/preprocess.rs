//! Image preprocessing functionality for plant dataset.
//!
//! This module provides image preprocessing operations including resizing,
//! normalization, and format conversions.

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};
use plant_core::{Error, ImageDimensions, Result};
use serde::{Deserialize, Serialize};

/// Configuration for image preprocessing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    /// Target image dimensions
    pub target_size: ImageDimensions,
    /// Whether to maintain aspect ratio
    pub maintain_aspect_ratio: bool,
    /// ImageNet normalization means [R, G, B]
    pub mean: [f32; 3],
    /// ImageNet normalization standard deviations [R, G, B]
    pub std: [f32; 3],
    /// Whether to convert to RGB (from RGBA, grayscale, etc.)
    pub force_rgb: bool,
}

impl Default for PreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: ImageDimensions::imagenet(),
            maintain_aspect_ratio: false,
            // ImageNet normalization values
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            force_rgb: true,
        }
    }
}

/// Image preprocessor for plant images
pub struct ImagePreprocessor {
    config: PreprocessConfig,
}

impl ImagePreprocessor {
    /// Creates a new image preprocessor with the given configuration
    pub fn new(config: PreprocessConfig) -> Self {
        Self { config }
    }

    /// Creates a preprocessor with default configuration
    pub fn default() -> Self {
        Self {
            config: PreprocessConfig::default(),
        }
    }

    /// Preprocesses an image for model input
    pub fn preprocess(&self, image: &DynamicImage) -> Result<Vec<f32>> {
        // Convert to RGB if needed
        let rgb_image = if self.config.force_rgb {
            image.to_rgb8()
        } else {
            match image {
                DynamicImage::ImageRgb8(img) => img.clone(),
                _ => image.to_rgb8(),
            }
        };

        // Resize image
        let resized = self.resize_image(&rgb_image)?;

        // Normalize
        let normalized = self.normalize_image(&resized)?;

        Ok(normalized)
    }

    /// Resizes an image to target dimensions
    fn resize_image(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let (width, height) = image.dimensions();
        let target_w = self.config.target_size.width;
        let target_h = self.config.target_size.height;

        if width == target_w && height == target_h {
            return Ok(image.clone());
        }

        let resized = if self.config.maintain_aspect_ratio {
            // Calculate scale to fit within target size
            let scale = (target_w as f32 / width as f32)
                .min(target_h as f32 / height as f32);

            let new_w = (width as f32 * scale) as u32;
            let new_h = (height as f32 * scale) as u32;

            let temp = image::imageops::resize(
                image,
                new_w,
                new_h,
                image::imageops::FilterType::Lanczos3,
            );

            // Center crop or pad to exact size
            self.center_crop_or_pad(&temp, target_w, target_h)
        } else {
            // Direct resize to target dimensions
            image::imageops::resize(
                image,
                target_w,
                target_h,
                image::imageops::FilterType::Lanczos3,
            )
        };

        Ok(resized)
    }

    /// Center crops or pads an image to exact dimensions
    fn center_crop_or_pad(
        &self,
        image: &ImageBuffer<Rgb<u8>, Vec<u8>>,
        target_w: u32,
        target_h: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let (width, height) = image.dimensions();

        if width == target_w && height == target_h {
            return image.clone();
        }

        let mut result = ImageBuffer::from_pixel(target_w, target_h, Rgb([0, 0, 0]));

        // Calculate offsets for centering
        let x_offset = if width > target_w {
            0
        } else {
            (target_w - width) / 2
        };

        let y_offset = if height > target_h {
            0
        } else {
            (target_h - height) / 2
        };

        // Copy image to center of result
        for y in 0..height.min(target_h) {
            for x in 0..width.min(target_w) {
                let src_x = if width > target_w { (width - target_w) / 2 + x } else { x };
                let src_y = if height > target_h { (height - target_h) / 2 + y } else { y };

                if src_x < width && src_y < height {
                    let pixel = image.get_pixel(src_x, src_y);
                    result.put_pixel(x_offset + x, y_offset + y, *pixel);
                }
            }
        }

        result
    }

    /// Normalizes an image using ImageNet statistics
    fn normalize_image(&self, image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> Result<Vec<f32>> {
        let (width, height) = image.dimensions();
        let num_pixels = (width * height) as usize;

        // Allocate output in CHW format (channels, height, width)
        let mut normalized = Vec::with_capacity(num_pixels * 3);

        // Process each channel separately (CHW format for Burn)
        for channel in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = image.get_pixel(x, y);
                    let value = pixel[channel] as f32 / 255.0;
                    let normalized_value = (value - self.config.mean[channel]) / self.config.std[channel];
                    normalized.push(normalized_value);
                }
            }
        }

        Ok(normalized)
    }

    /// Preprocesses an image from a file path
    pub fn preprocess_from_path(&self, path: &std::path::Path) -> Result<Vec<f32>> {
        let image = image::open(path)
            .map_err(|e| Error::Image(format!("Failed to load image: {}", e)))?;

        self.preprocess(&image)
    }

    /// Gets the expected output shape after preprocessing
    pub fn output_shape(&self) -> [usize; 3] {
        [
            self.config.target_size.channels as usize,
            self.config.target_size.height as usize,
            self.config.target_size.width as usize,
        ]
    }

    /// Denormalizes a tensor back to image space
    pub fn denormalize(&self, normalized: &[f32]) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let height = self.config.target_size.height as usize;
        let width = self.config.target_size.width as usize;

        if normalized.len() != height * width * 3 {
            return Err(Error::InvalidArgument(format!(
                "Expected {} values, got {}",
                height * width * 3,
                normalized.len()
            )));
        }

        let mut image = ImageBuffer::new(width as u32, height as u32);

        for y in 0..height {
            for x in 0..width {
                let mut pixel = Rgb([0u8; 3]);
                for c in 0..3 {
                    let idx = c * height * width + y * width + x;
                    let value = normalized[idx] * self.config.std[c] + self.config.mean[c];
                    pixel[c] = (value.clamp(0.0, 1.0) * 255.0) as u8;
                }
                image.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_config_default() {
        let config = PreprocessConfig::default();
        assert_eq!(config.target_size.width, 224);
        assert_eq!(config.target_size.height, 224);
        assert_eq!(config.target_size.channels, 3);
        assert!(config.force_rgb);
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = ImagePreprocessor::default();
        let shape = preprocessor.output_shape();
        assert_eq!(shape, [3, 224, 224]);
    }

    #[test]
    fn test_preprocess_small_image() {
        let preprocessor = ImagePreprocessor::default();

        // Create a simple 10x10 red image
        let img = ImageBuffer::from_fn(10, 10, |_, _| Rgb([255u8, 0u8, 0u8]));
        let dynamic = DynamicImage::ImageRgb8(img);

        let result = preprocessor.preprocess(&dynamic);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        assert_eq!(normalized.len(), 224 * 224 * 3);
    }

    #[test]
    fn test_resize_exact_match() {
        let config = PreprocessConfig::default();
        let preprocessor = ImagePreprocessor::new(config);

        // Create image that's already the right size
        let img = ImageBuffer::from_pixel(224, 224, Rgb([128u8, 128u8, 128u8]));

        let result = preprocessor.resize_image(&img);
        assert!(result.is_ok());

        let resized = result.unwrap();
        assert_eq!(resized.dimensions(), (224, 224));
    }

    #[test]
    fn test_normalization_values() {
        let config = PreprocessConfig::default();
        let preprocessor = ImagePreprocessor::new(config);

        // Create a simple image
        let img = ImageBuffer::from_pixel(224, 224, Rgb([128u8, 128u8, 128u8]));

        let result = preprocessor.normalize_image(&img);
        assert!(result.is_ok());

        let normalized = result.unwrap();
        // Check that values are reasonable (not NaN or infinite)
        assert!(normalized.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_denormalize_roundtrip() {
        let config = PreprocessConfig::default();
        let preprocessor = ImagePreprocessor::new(config);

        // Create a simple image
        let original = ImageBuffer::from_pixel(224, 224, Rgb([100u8, 150u8, 200u8]));

        // Normalize
        let normalized = preprocessor.normalize_image(&original).unwrap();

        // Denormalize
        let result = preprocessor.denormalize(&normalized);
        assert!(result.is_ok());

        let restored = result.unwrap();
        assert_eq!(restored.dimensions(), (224, 224));

        // Values should be close (within a few units due to float conversion)
        let original_pixel = original.get_pixel(112, 112);
        let restored_pixel = restored.get_pixel(112, 112);

        for c in 0..3 {
            let diff = (original_pixel[c] as i32 - restored_pixel[c] as i32).abs();
            assert!(diff < 5, "Channel {} differs by {}", c, diff);
        }
    }
}
