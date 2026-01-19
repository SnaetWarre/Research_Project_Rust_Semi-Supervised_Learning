//! Data Augmentation Module for Plant Disease Classification
//!
//! Provides on-the-fly image augmentations to improve model generalization.
//! Augmentations simulate real-world variations like different lighting,
//! camera angles, and image quality.
//!
//! # Augmentation Strategy
//!
//! - **Training**: Apply random augmentations to simulate real-world variation
//! - **Validation/Test**: No augmentations (clean evaluation)
//! - **SSL Inference**: No augmentations (consistent predictions for pseudo-labels)
//! - **SSL Retraining**: Same augmentations as training

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage};
use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Configuration for data augmentation
#[derive(Clone, Debug)]
pub struct AugmentationConfig {
    /// Probability of applying horizontal flip (0.0 - 1.0)
    pub horizontal_flip_prob: f32,
    /// Probability of applying vertical flip (0.0 - 1.0)
    pub vertical_flip_prob: f32,
    /// Maximum rotation angle in degrees (applies ±rotation_degrees)
    pub rotation_degrees: f32,
    /// Probability of applying rotation
    pub rotation_prob: f32,
    /// Brightness adjustment range (±brightness_delta)
    pub brightness_delta: f32,
    /// Probability of applying brightness adjustment
    pub brightness_prob: f32,
    /// Contrast adjustment range (1.0 ± contrast_delta)
    pub contrast_delta: f32,
    /// Probability of applying contrast adjustment
    pub contrast_prob: f32,
    /// Saturation adjustment range (1.0 ± saturation_delta)
    pub saturation_delta: f32,
    /// Probability of applying saturation adjustment
    pub saturation_prob: f32,
    /// Gaussian blur kernel size (odd number, 0 = disabled)
    pub blur_kernel_size: u32,
    /// Probability of applying blur
    pub blur_prob: f32,
    /// Gaussian noise standard deviation (0.0 = disabled)
    pub noise_std: f32,
    /// Probability of applying noise
    pub noise_prob: f32,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.2,
            rotation_degrees: 20.0,
            rotation_prob: 0.5,
            brightness_delta: 0.2,
            brightness_prob: 0.5,
            contrast_delta: 0.2,
            contrast_prob: 0.5,
            saturation_delta: 0.2,
            saturation_prob: 0.3,
            blur_kernel_size: 3,
            blur_prob: 0.1,
            noise_std: 0.02,
            noise_prob: 0.1,
        }
    }
}

impl AugmentationConfig {
    /// Create a "light" augmentation preset (less aggressive)
    pub fn light() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.0,
            rotation_degrees: 10.0,
            rotation_prob: 0.3,
            brightness_delta: 0.1,
            brightness_prob: 0.3,
            contrast_delta: 0.1,
            contrast_prob: 0.3,
            saturation_delta: 0.1,
            saturation_prob: 0.2,
            blur_kernel_size: 0,
            blur_prob: 0.0,
            noise_std: 0.0,
            noise_prob: 0.0,
        }
    }

    /// Create a "medium" augmentation preset (balanced)
    pub fn medium() -> Self {
        Self::default()
    }

    /// Create a "heavy" augmentation preset (aggressive, for maximum generalization)
    pub fn heavy() -> Self {
        Self {
            horizontal_flip_prob: 0.5,
            vertical_flip_prob: 0.3,
            rotation_degrees: 30.0,
            rotation_prob: 0.7,
            brightness_delta: 0.3,
            brightness_prob: 0.6,
            contrast_delta: 0.3,
            contrast_prob: 0.6,
            saturation_delta: 0.3,
            saturation_prob: 0.5,
            blur_kernel_size: 5,
            blur_prob: 0.15,
            noise_std: 0.03,
            noise_prob: 0.15,
        }
    }

    /// Disable all augmentations (for validation/inference)
    pub fn none() -> Self {
        Self {
            horizontal_flip_prob: 0.0,
            vertical_flip_prob: 0.0,
            rotation_degrees: 0.0,
            rotation_prob: 0.0,
            brightness_delta: 0.0,
            brightness_prob: 0.0,
            contrast_delta: 0.0,
            contrast_prob: 0.0,
            saturation_delta: 0.0,
            saturation_prob: 0.0,
            blur_kernel_size: 0,
            blur_prob: 0.0,
            noise_std: 0.0,
            noise_prob: 0.0,
        }
    }
}

/// Image augmenter that applies random transformations
#[derive(Clone)]
pub struct Augmenter {
    config: AugmentationConfig,
    image_size: u32,
}

impl Augmenter {
    /// Create a new augmenter with the given configuration
    pub fn new(config: AugmentationConfig, image_size: u32) -> Self {
        Self { config, image_size }
    }

    /// Create an augmenter with default (medium) augmentation
    pub fn with_defaults(image_size: u32) -> Self {
        Self::new(AugmentationConfig::default(), image_size)
    }

    /// Create an augmenter with no augmentation (for validation/inference)
    pub fn no_augmentation(image_size: u32) -> Self {
        Self::new(AugmentationConfig::none(), image_size)
    }

    /// Apply all configured augmentations randomly to an image
    ///
    /// # Arguments
    /// * `img` - The input image
    /// * `rng` - Random number generator for reproducibility
    ///
    /// # Returns
    /// The augmented image
    pub fn augment(&self, img: DynamicImage, rng: &mut ChaCha8Rng) -> DynamicImage {
        let mut result = img;

        // Apply horizontal flip
        if rng.gen::<f32>() < self.config.horizontal_flip_prob {
            result = result.fliph();
        }

        // Apply vertical flip
        if rng.gen::<f32>() < self.config.vertical_flip_prob {
            result = result.flipv();
        }

        // Apply rotation
        if self.config.rotation_prob > 0.0 && rng.gen::<f32>() < self.config.rotation_prob {
            let angle = rng.gen_range(-self.config.rotation_degrees..=self.config.rotation_degrees);
            result = self.rotate(&result, angle);
        }

        // Apply brightness adjustment
        if self.config.brightness_prob > 0.0 && rng.gen::<f32>() < self.config.brightness_prob {
            let delta = rng.gen_range(-self.config.brightness_delta..=self.config.brightness_delta);
            result = self.adjust_brightness(&result, delta);
        }

        // Apply contrast adjustment
        if self.config.contrast_prob > 0.0 && rng.gen::<f32>() < self.config.contrast_prob {
            let factor = 1.0 + rng.gen_range(-self.config.contrast_delta..=self.config.contrast_delta);
            result = self.adjust_contrast(&result, factor);
        }

        // Apply saturation adjustment
        if self.config.saturation_prob > 0.0 && rng.gen::<f32>() < self.config.saturation_prob {
            let factor = 1.0 + rng.gen_range(-self.config.saturation_delta..=self.config.saturation_delta);
            result = self.adjust_saturation(&result, factor);
        }

        // Apply Gaussian blur
        if self.config.blur_kernel_size > 0 && rng.gen::<f32>() < self.config.blur_prob {
            result = self.apply_blur(&result);
        }

        // Apply Gaussian noise
        if self.config.noise_std > 0.0 && rng.gen::<f32>() < self.config.noise_prob {
            result = self.apply_noise(&result, rng);
        }

        result
    }

    /// Rotate image by the given angle in degrees
    fn rotate(&self, img: &DynamicImage, angle_degrees: f32) -> DynamicImage {
        if angle_degrees.abs() < 0.1 {
            return img.clone();
        }

        let angle_rad = angle_degrees.to_radians();
        let (width, height) = img.dimensions();
        let rgb = img.to_rgb8();

        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;

        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        let mut output = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                // Rotate around center
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;

                let src_x = cx + dx * cos_a + dy * sin_a;
                let src_y = cy - dx * sin_a + dy * cos_a;

                // Bilinear interpolation
                let pixel = self.bilinear_sample(&rgb, src_x, src_y);
                output.put_pixel(x, y, pixel);
            }
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Sample a pixel using bilinear interpolation
    fn bilinear_sample(&self, img: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
        let (width, height) = img.dimensions();

        // Clamp to valid range
        if x < 0.0 || y < 0.0 || x >= width as f32 - 1.0 || y >= height as f32 - 1.0 {
            // Return black for out-of-bounds (or could use edge clamping)
            return Rgb([0, 0, 0]);
        }

        let x0 = x.floor() as u32;
        let y0 = y.floor() as u32;
        let x1 = (x0 + 1).min(width - 1);
        let y1 = (y0 + 1).min(height - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let p00 = img.get_pixel(x0, y0);
        let p10 = img.get_pixel(x1, y0);
        let p01 = img.get_pixel(x0, y1);
        let p11 = img.get_pixel(x1, y1);

        let mut result = [0u8; 3];
        for c in 0..3 {
            let v00 = p00[c] as f32;
            let v10 = p10[c] as f32;
            let v01 = p01[c] as f32;
            let v11 = p11[c] as f32;

            let v = v00 * (1.0 - fx) * (1.0 - fy)
                + v10 * fx * (1.0 - fy)
                + v01 * (1.0 - fx) * fy
                + v11 * fx * fy;

            result[c] = v.round().clamp(0.0, 255.0) as u8;
        }

        Rgb(result)
    }

    /// Adjust brightness by adding delta to all pixels
    fn adjust_brightness(&self, img: &DynamicImage, delta: f32) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let delta_u8 = (delta * 255.0) as i32;

        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = (pixel[0] as i32 + delta_u8).clamp(0, 255) as u8;
            let g = (pixel[1] as i32 + delta_u8).clamp(0, 255) as u8;
            let b = (pixel[2] as i32 + delta_u8).clamp(0, 255) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Adjust contrast by scaling pixel values around the mean
    fn adjust_contrast(&self, img: &DynamicImage, factor: f32) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();

        // Calculate mean luminance
        let mut sum = 0.0f64;
        let count = (width * height) as f64;
        for pixel in rgb.pixels() {
            let lum = 0.299 * pixel[0] as f64 + 0.587 * pixel[1] as f64 + 0.114 * pixel[2] as f64;
            sum += lum;
        }
        let mean = (sum / count) as f32;

        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            let r = (mean + factor * (pixel[0] as f32 - mean)).clamp(0.0, 255.0) as u8;
            let g = (mean + factor * (pixel[1] as f32 - mean)).clamp(0.0, 255.0) as u8;
            let b = (mean + factor * (pixel[2] as f32 - mean)).clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Adjust saturation by scaling the color intensity
    fn adjust_saturation(&self, img: &DynamicImage, factor: f32) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();

        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            // Convert to HSL-like: compute grayscale
            let gray = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;

            // Interpolate between grayscale and original based on factor
            let r = (gray + factor * (pixel[0] as f32 - gray)).clamp(0.0, 255.0) as u8;
            let g = (gray + factor * (pixel[1] as f32 - gray)).clamp(0.0, 255.0) as u8;
            let b = (gray + factor * (pixel[2] as f32 - gray)).clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Apply Gaussian blur using a simple box blur approximation
    fn apply_blur(&self, img: &DynamicImage) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let kernel_size = self.config.blur_kernel_size;
        let half_k = kernel_size as i32 / 2;

        let mut output = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let mut sum_r = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_b = 0.0f32;
                let mut count = 0.0f32;

                for ky in -half_k..=half_k {
                    for kx in -half_k..=half_k {
                        let px = x as i32 + kx;
                        let py = y as i32 + ky;

                        if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                            let pixel = rgb.get_pixel(px as u32, py as u32);
                            sum_r += pixel[0] as f32;
                            sum_g += pixel[1] as f32;
                            sum_b += pixel[2] as f32;
                            count += 1.0;
                        }
                    }
                }

                let r = (sum_r / count) as u8;
                let g = (sum_g / count) as u8;
                let b = (sum_b / count) as u8;
                output.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Apply Gaussian noise to the image
    fn apply_noise(&self, img: &DynamicImage, rng: &mut ChaCha8Rng) -> DynamicImage {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let std = self.config.noise_std * 255.0;

        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in rgb.enumerate_pixels() {
            // Box-Muller transform for Gaussian noise
            let u1: f32 = rng.gen();
            let u2: f32 = rng.gen();
            let noise = std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

            let r = (pixel[0] as f32 + noise).clamp(0.0, 255.0) as u8;
            let g = (pixel[1] as f32 + noise).clamp(0.0, 255.0) as u8;
            let b = (pixel[2] as f32 + noise).clamp(0.0, 255.0) as u8;
            output.put_pixel(x, y, Rgb([r, g, b]));
        }

        DynamicImage::ImageRgb8(output)
    }

    /// Resize image to target size (always applied, not random)
    pub fn resize(&self, img: DynamicImage) -> DynamicImage {
        img.resize_exact(
            self.image_size,
            self.image_size,
            image::imageops::FilterType::Triangle,
        )
    }

    /// Convert image to CHW float tensor data normalized to [0, 1]
    pub fn to_tensor_data(&self, img: &DynamicImage) -> Vec<f32> {
        let rgb = img.to_rgb8();
        let (width, height) = rgb.dimensions();
        let mut data = Vec::with_capacity(3 * height as usize * width as usize);

        // Convert to CHW format
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel = rgb.get_pixel(x, y);
                    data.push(pixel[c] as f32 / 255.0);
                }
            }
        }

        data
    }

    /// Full preprocessing pipeline: augment (optional), resize, convert to tensor
    pub fn preprocess(
        &self,
        img: DynamicImage,
        rng: Option<&mut ChaCha8Rng>,
    ) -> Vec<f32> {
        let mut result = img;

        // Apply augmentations if RNG is provided
        if let Some(rng) = rng {
            result = self.augment(result, rng);
        }

        // Resize to target size
        result = self.resize(result);

        // Convert to tensor data
        self.to_tensor_data(&result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn create_test_image() -> DynamicImage {
        let mut img = ImageBuffer::new(64, 64);
        for (x, y, pixel) in img.enumerate_pixels_mut() {
            *pixel = Rgb([(x * 4) as u8, (y * 4) as u8, 128]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_augmenter_creation() {
        let aug = Augmenter::with_defaults(128);
        assert_eq!(aug.image_size, 128);
        assert_eq!(aug.config.horizontal_flip_prob, 0.5);
    }

    #[test]
    fn test_no_augmentation() {
        let aug = Augmenter::no_augmentation(128);
        assert_eq!(aug.config.horizontal_flip_prob, 0.0);
        assert_eq!(aug.config.rotation_prob, 0.0);
    }

    #[test]
    fn test_augmentation_presets() {
        let light = AugmentationConfig::light();
        let medium = AugmentationConfig::medium();
        let heavy = AugmentationConfig::heavy();

        assert!(light.rotation_degrees < medium.rotation_degrees);
        assert!(medium.rotation_degrees < heavy.rotation_degrees);
    }

    #[test]
    fn test_augment_produces_valid_image() {
        let aug = Augmenter::with_defaults(64);
        let img = create_test_image();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = aug.augment(img, &mut rng);
        let (w, h) = result.dimensions();

        assert_eq!(w, 64);
        assert_eq!(h, 64);
    }

    #[test]
    fn test_resize() {
        let aug = Augmenter::with_defaults(32);
        let img = create_test_image();

        let result = aug.resize(img);
        let (w, h) = result.dimensions();

        assert_eq!(w, 32);
        assert_eq!(h, 32);
    }

    #[test]
    fn test_to_tensor_data() {
        let aug = Augmenter::with_defaults(64);
        let img = create_test_image();

        let data = aug.to_tensor_data(&img);

        // Should be CHW format: 3 * 64 * 64
        assert_eq!(data.len(), 3 * 64 * 64);

        // Values should be in [0, 1]
        for val in &data {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_preprocess_with_augmentation() {
        let aug = Augmenter::with_defaults(32);
        let img = create_test_image();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let data = aug.preprocess(img, Some(&mut rng));

        // Should be CHW format: 3 * 32 * 32
        assert_eq!(data.len(), 3 * 32 * 32);
    }

    #[test]
    fn test_preprocess_without_augmentation() {
        let aug = Augmenter::no_augmentation(32);
        let img = create_test_image();

        let data = aug.preprocess(img, None);

        // Should be CHW format: 3 * 32 * 32
        assert_eq!(data.len(), 3 * 32 * 32);
    }

    #[test]
    fn test_brightness_adjustment() {
        let aug = Augmenter::with_defaults(64);
        let img = create_test_image();

        let brighter = aug.adjust_brightness(&img, 0.2);
        let darker = aug.adjust_brightness(&img, -0.2);

        // Check that brightness actually changed
        let orig_rgb = img.to_rgb8();
        let bright_rgb = brighter.to_rgb8();
        let dark_rgb = darker.to_rgb8();

        let orig_pixel = orig_rgb.get_pixel(32, 32);
        let bright_pixel = bright_rgb.get_pixel(32, 32);
        let dark_pixel = dark_rgb.get_pixel(32, 32);

        // Brighter should have higher values (unless already at max)
        assert!(bright_pixel[0] >= orig_pixel[0] || orig_pixel[0] > 200);
        // Darker should have lower values (unless already at min)
        assert!(dark_pixel[0] <= orig_pixel[0] || orig_pixel[0] < 50);
    }
}
