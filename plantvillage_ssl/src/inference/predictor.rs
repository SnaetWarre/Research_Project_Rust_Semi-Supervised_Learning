//! Inference Predictor Module
//!
//! Provides functionality for running inference on images using trained models.
//! Designed for efficient prediction on both desktop and edge devices.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use image::{DynamicImage, imageops::FilterType};
use serde::{Deserialize, Serialize};

use crate::dataset::{class_name, NUM_CLASSES};

/// ImageNet normalization mean values (RGB)
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet normalization std values (RGB)
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Resize an image to the target dimensions
fn resize_image(image: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    image.resize_exact(width, height, FilterType::Lanczos3)
}

/// Normalize an image to a flat vector with ImageNet normalization
/// Returns CHW layout: [C, H, W] flattened
fn normalize_image(image: &DynamicImage) -> Vec<f32> {
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let num_pixels = (width * height) as usize;

    // Pre-allocate for CHW layout
    let mut normalized = vec![0.0f32; 3 * num_pixels];

    for (i, pixel) in rgb.pixels().enumerate() {
        let r = (pixel[0] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        let g = (pixel[1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        let b = (pixel[2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];

        // CHW layout: all R values, then all G values, then all B values
        normalized[i] = r;
        normalized[num_pixels + i] = g;
        normalized[2 * num_pixels + i] = b;
    }

    normalized
}

/// Result of a single prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Path to the input image (if applicable)
    pub image_path: Option<PathBuf>,

    /// Predicted class index
    pub predicted_class: usize,

    /// Predicted class name
    pub class_name: String,

    /// Confidence score (probability) for the predicted class
    pub confidence: f32,

    /// Full probability distribution over all classes
    pub probabilities: Vec<f32>,

    /// Top-k predictions with their probabilities
    pub top_k: Vec<(usize, String, f32)>,

    /// Inference time in milliseconds
    pub inference_time_ms: f64,
}

impl PredictionResult {
    /// Create a new prediction result
    pub fn new(
        probabilities: Vec<f32>,
        inference_time: Duration,
        image_path: Option<PathBuf>,
    ) -> Self {
        // Find predicted class (argmax)
        let (predicted_class, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap_or((0, &0.0));

        let class_name_str = class_name(predicted_class)
            .unwrap_or("Unknown")
            .to_string();

        // Get top-5 predictions
        let mut indexed: Vec<(usize, f32)> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let top_k: Vec<(usize, String, f32)> = indexed
            .iter()
            .take(5)
            .map(|&(idx, prob)| {
                let name = class_name(idx).unwrap_or("Unknown").to_string();
                (idx, name, prob)
            })
            .collect();

        Self {
            image_path,
            predicted_class,
            class_name: class_name_str,
            confidence,
            probabilities,
            top_k,
            inference_time_ms: inference_time.as_secs_f64() * 1000.0,
        }
    }

    /// Check if the prediction is high-confidence (suitable for pseudo-labeling)
    pub fn is_high_confidence(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }

    /// Get the entropy of the prediction (measure of uncertainty)
    pub fn entropy(&self) -> f32 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Get the margin between top-1 and top-2 predictions
    pub fn margin(&self) -> f32 {
        if self.top_k.len() >= 2 {
            self.top_k[0].2 - self.top_k[1].2
        } else {
            self.confidence
        }
    }

    /// Pretty print the prediction result
    pub fn display(&self) -> String {
        let mut output = String::new();

        if let Some(path) = &self.image_path {
            output.push_str(&format!("Image: {:?}\n", path));
        }

        output.push_str(&format!(
            "Prediction: {} (class {})\n",
            self.class_name, self.predicted_class
        ));
        output.push_str(&format!("Confidence: {:.2}%\n", self.confidence * 100.0));
        output.push_str(&format!("Inference time: {:.2} ms\n", self.inference_time_ms));

        output.push_str("\nTop-5 predictions:\n");
        for (i, (idx, name, prob)) in self.top_k.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} (class {}) - {:.2}%\n",
                i + 1,
                name,
                idx,
                prob * 100.0
            ));
        }

        output
    }
}

/// Predictor for running inference with a trained model
pub struct Predictor {
    /// Target image size for preprocessing
    pub image_size: u32,

    /// Number of classes
    pub num_classes: usize,

    /// Whether to use GPU acceleration
    pub use_gpu: bool,

    /// Batch size for batch inference
    pub batch_size: usize,
}

impl Default for Predictor {
    fn default() -> Self {
        Self {
            image_size: 256,
            num_classes: NUM_CLASSES,
            use_gpu: false,
            batch_size: 1,
        }
    }
}

impl Predictor {
    /// Create a new predictor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a predictor configured for edge deployment
    pub fn edge_optimized() -> Self {
        Self {
            image_size: 224, // Smaller for faster inference
            num_classes: NUM_CLASSES,
            use_gpu: true,
            batch_size: 1, // Single image for lowest latency
        }
    }

    /// Configure image size
    pub fn with_image_size(mut self, size: u32) -> Self {
        self.image_size = size;
        self
    }

    /// Configure GPU usage
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// Configure batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Preprocess an image for inference
    pub fn preprocess(&self, image: &DynamicImage) -> Vec<f32> {
        let resized = resize_image(image, self.image_size, self.image_size);
        normalize_image(&resized)
    }

    /// Load and preprocess an image from a file path
    pub fn load_and_preprocess(&self, path: &Path) -> anyhow::Result<Vec<f32>> {
        let image = image::open(path)?;
        Ok(self.preprocess(&image))
    }

    /// Predict on a preprocessed image tensor
    ///
    /// Note: This is a placeholder - actual implementation requires the model
    pub fn predict_tensor(&self, _tensor: &[f32]) -> PredictionResult {
        // TODO: Implement actual inference with Burn model
        // This is a placeholder that returns dummy results

        let start = Instant::now();

        // Placeholder: create uniform distribution
        let probabilities = vec![1.0 / self.num_classes as f32; self.num_classes];

        let inference_time = start.elapsed();

        PredictionResult::new(probabilities, inference_time, None)
    }

    /// Predict on an image from a file path
    pub fn predict_file(&self, path: &Path) -> anyhow::Result<PredictionResult> {
        let tensor = self.load_and_preprocess(path)?;
        let mut result = self.predict_tensor(&tensor);
        result.image_path = Some(path.to_path_buf());
        Ok(result)
    }

    /// Predict on multiple images (batch inference)
    pub fn predict_batch(&self, paths: &[PathBuf]) -> anyhow::Result<Vec<PredictionResult>> {
        // TODO: Implement true batched inference for efficiency
        // For now, process one at a time

        paths
            .iter()
            .map(|path| self.predict_file(path))
            .collect()
    }
}

/// Batch prediction statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchPredictionStats {
    /// Total number of images processed
    pub total_images: usize,

    /// Total inference time
    pub total_time_ms: f64,

    /// Average inference time per image
    pub avg_time_per_image_ms: f64,

    /// Minimum inference time
    pub min_time_ms: f64,

    /// Maximum inference time
    pub max_time_ms: f64,

    /// Throughput (images per second)
    pub throughput: f64,

    /// Number of high-confidence predictions
    pub high_confidence_count: usize,

    /// Confidence threshold used
    pub confidence_threshold: f32,
}

impl BatchPredictionStats {
    /// Calculate statistics from a batch of predictions
    pub fn from_predictions(predictions: &[PredictionResult], confidence_threshold: f32) -> Self {
        if predictions.is_empty() {
            return Self::default();
        }

        let times: Vec<f64> = predictions.iter().map(|p| p.inference_time_ms).collect();
        let total_time_ms: f64 = times.iter().sum();
        let total_images = predictions.len();

        let high_confidence_count = predictions
            .iter()
            .filter(|p| p.confidence >= confidence_threshold)
            .count();

        Self {
            total_images,
            total_time_ms,
            avg_time_per_image_ms: total_time_ms / total_images as f64,
            min_time_ms: times.iter().cloned().fold(f64::INFINITY, f64::min),
            max_time_ms: times.iter().cloned().fold(0.0, f64::max),
            throughput: total_images as f64 / (total_time_ms / 1000.0),
            high_confidence_count,
            confidence_threshold,
        }
    }
}

impl std::fmt::Display for BatchPredictionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Batch Prediction Statistics:")?;
        writeln!(f, "  Total images: {}", self.total_images)?;
        writeln!(f, "  Total time: {:.2} ms", self.total_time_ms)?;
        writeln!(f, "  Average time/image: {:.2} ms", self.avg_time_per_image_ms)?;
        writeln!(f, "  Min time: {:.2} ms", self.min_time_ms)?;
        writeln!(f, "  Max time: {:.2} ms", self.max_time_ms)?;
        writeln!(f, "  Throughput: {:.2} images/sec", self.throughput)?;
        writeln!(
            f,
            "  High confidence (>={:.0}%): {} ({:.1}%)",
            self.confidence_threshold * 100.0,
            self.high_confidence_count,
            100.0 * self.high_confidence_count as f64 / self.total_images as f64
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_result_new() {
        let mut probs = vec![0.0; 39];
        probs[5] = 0.8;
        probs[10] = 0.15;
        probs[3] = 0.05;

        let result = PredictionResult::new(probs, Duration::from_millis(50), None);

        assert_eq!(result.predicted_class, 5);
        assert_eq!(result.confidence, 0.8);
        assert_eq!(result.top_k.len(), 5);
        assert_eq!(result.top_k[0].0, 5);
    }

    #[test]
    fn test_prediction_entropy() {
        // Uniform distribution has high entropy
        let uniform_probs = vec![1.0 / 39.0; 39];
        let uniform_result =
            PredictionResult::new(uniform_probs, Duration::from_millis(10), None);

        // Confident prediction has low entropy
        let mut confident_probs = vec![0.001; 39];
        confident_probs[0] = 0.962; // Make it sum to ~1
        let confident_result =
            PredictionResult::new(confident_probs, Duration::from_millis(10), None);

        assert!(uniform_result.entropy() > confident_result.entropy());
    }

    #[test]
    fn test_predictor_default() {
        let predictor = Predictor::new();
        assert_eq!(predictor.image_size, 256);
        assert_eq!(predictor.num_classes, 39);
        assert!(!predictor.use_gpu);
    }

    #[test]
    fn test_batch_stats() {
        let predictions: Vec<PredictionResult> = (0..10)
            .map(|i| {
                let mut probs = vec![0.0; 39];
                probs[i % 39] = 0.9;
                PredictionResult::new(probs, Duration::from_millis(50 + i as u64), None)
            })
            .collect();

        let stats = BatchPredictionStats::from_predictions(&predictions, 0.8);

        assert_eq!(stats.total_images, 10);
        assert!(stats.avg_time_per_image_ms > 50.0);
        assert_eq!(stats.high_confidence_count, 10);
    }

    #[test]
    fn test_resize_and_normalize() {
        // Create a simple test image
        let img = DynamicImage::new_rgb8(100, 100);
        let resized = resize_image(&img, 256, 256);
        assert_eq!(resized.width(), 256);
        assert_eq!(resized.height(), 256);

        let normalized = normalize_image(&resized);
        // Should be CHW layout: 3 * 256 * 256
        assert_eq!(normalized.len(), 3 * 256 * 256);
    }
}
