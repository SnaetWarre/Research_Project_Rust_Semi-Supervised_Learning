//! Dataset split strategies for semi-supervised learning simulation
//!
//! This module implements a carefully designed split strategy that mimics real-world
//! agricultural scenarios where:
//! - A small portion of data is manually labeled (expensive)
//! - Most data arrives unlabeled from camera captures
//! - The model must learn to utilize both labeled and pseudo-labeled data
//!
//! ## Split Strategy
//!
//! The PlantVillage dataset is divided into:
//! 1. **Test Set (10%)** - Held out, NEVER seen during training (for honest evaluation)
//! 2. **Validation Set (10%)** - Used for hyperparameter tuning and early stopping
//! 3. **Labeled Pool (20%)** - Initial labeled data (simulating manual labeling)
//! 4. **Stream Pool (50%)** - "Incoming" unlabeled images (simulating camera captures)
//! 5. **Future Pool (10%)** - Reserved for future streaming simulation
//!
//! This split is deterministic and reproducible using a fixed random seed.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::utils::error::PlantVillageError;

/// Result type alias for this module
pub type Result<T> = std::result::Result<T, PlantVillageError>;

/// Configuration for dataset splitting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Fraction of data for test set (held out, never seen during training)
    pub test_fraction: f64,
    /// Fraction of data for validation set
    pub validation_fraction: f64,
    /// Fraction of remaining data to use as initial labeled pool
    pub labeled_fraction: f64,
    /// Fraction for stream simulation pool
    pub stream_fraction: f64,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Ensure class balance in splits
    pub stratified: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            test_fraction: 0.10,      // 10% for honest evaluation
            validation_fraction: 0.10, // 10% for validation
            labeled_fraction: 0.20,    // 20% labeled (of remaining 80%)
            stream_fraction: 0.60,     // 60% for streaming simulation (of remaining 80%)
            seed: 42,                  // Fixed seed for reproducibility
            stratified: true,          // Maintain class balance
        }
    }
}

impl SplitConfig {
    /// Create a new split configuration with custom fractions
    pub fn new(
        test_fraction: f64,
        validation_fraction: f64,
        labeled_fraction: f64,
        seed: u64,
    ) -> Result<Self> {
        // Validate fractions
        if test_fraction + validation_fraction >= 1.0 {
            return Err(PlantVillageError::Config(
                "Test + validation fractions must be less than 1.0".to_string(),
            ));
        }

        if test_fraction < 0.0 || test_fraction > 1.0 {
            return Err(PlantVillageError::Config(
                "Test fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        if validation_fraction < 0.0 || validation_fraction > 1.0 {
            return Err(PlantVillageError::Config(
                "Validation fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        if labeled_fraction < 0.0 || labeled_fraction > 1.0 {
            return Err(PlantVillageError::Config(
                "Labeled fraction must be between 0.0 and 1.0".to_string(),
            ));
        }

        let stream_fraction = 1.0 - labeled_fraction - 0.10; // Reserve 10% for future pool

        Ok(Self {
            test_fraction,
            validation_fraction,
            labeled_fraction,
            stream_fraction,
            seed,
            stratified: true,
        })
    }

    /// Create configuration for minimal labeled data scenario (realistic agricultural setting)
    pub fn minimal_labeled() -> Self {
        Self {
            test_fraction: 0.10,
            validation_fraction: 0.10,
            labeled_fraction: 0.10, // Only 10% labeled - very realistic
            stream_fraction: 0.70,
            seed: 42,
            stratified: true,
        }
    }
}

/// A labeled image with known ground truth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledImage {
    /// Path to the image file
    pub image_path: PathBuf,
    /// Ground truth class label
    pub label: usize,
    /// Class name (e.g., "Tomato___Early_blight")
    pub class_name: String,
    /// Unique identifier for tracking
    pub image_id: u64,
}

/// An image with hidden label (for stream simulation)
/// The model sees this as unlabeled, but we track the true label for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenLabelImage {
    /// Path to the image file
    pub image_path: PathBuf,
    /// Ground truth label (hidden from model, used for evaluation)
    pub hidden_label: usize,
    /// Class name (hidden from model)
    pub hidden_class_name: String,
    /// Unique identifier for tracking
    pub image_id: u64,
}

/// A pseudo-labeled image (predicted by the model with high confidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLabeledImage {
    /// Path to the image file
    pub image_path: PathBuf,
    /// Predicted label by the model
    pub predicted_label: usize,
    /// Confidence score of the prediction
    pub confidence: f32,
    /// Ground truth label (for evaluation purposes)
    pub ground_truth: usize,
    /// Whether the prediction matches ground truth
    pub is_correct: bool,
    /// Unique identifier
    pub image_id: u64,
    /// Timestamp when pseudo-label was assigned (simulation day)
    pub assigned_day: usize,
}

/// Complete dataset splits for semi-supervised learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetSplits {
    /// Test set - NEVER seen during training, for final evaluation
    pub test_set: Vec<LabeledImage>,

    /// Validation set - for hyperparameter tuning and early stopping
    pub validation_set: Vec<LabeledImage>,

    /// Initial labeled pool - starting training data
    pub labeled_pool: Vec<LabeledImage>,

    /// Stream pool - simulates incoming unlabeled camera captures
    pub stream_pool: Vec<HiddenLabelImage>,

    /// Future pool - reserved for extended simulation
    pub future_pool: Vec<HiddenLabelImage>,

    /// Pseudo-labeled images - populated during training
    pub pseudo_labeled: Vec<PseudoLabeledImage>,

    /// Configuration used to create these splits
    pub config: SplitConfig,

    /// Class names mapping (index -> name)
    pub class_names: Vec<String>,

    /// Total number of images
    pub total_images: usize,
}

impl DatasetSplits {
    /// Create dataset splits from a list of (image_path, class_index, class_name) tuples
    pub fn from_images(
        images: Vec<(PathBuf, usize, String)>,
        config: SplitConfig,
    ) -> Result<Self> {
        let total_images = images.len();

        if total_images == 0 {
            return Err(PlantVillageError::Dataset(
                "No images provided for splitting".to_string(),
            ));
        }

        // Extract unique class names
        let mut class_names: Vec<String> = images
            .iter()
            .map(|(_, _, name)| name.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        class_names.sort();

        // Create deterministic RNG
        let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

        // Assign unique IDs
        let mut images_with_ids: Vec<_> = images
            .into_iter()
            .enumerate()
            .map(|(id, (path, label, name))| (path, label, name, id as u64))
            .collect();

        // Shuffle deterministically
        images_with_ids.shuffle(&mut rng);

        let (test_set, validation_set, labeled_pool, stream_pool, future_pool) = if config.stratified
        {
            Self::stratified_split(&images_with_ids, &config, &mut rng)?
        } else {
            Self::random_split(&images_with_ids, &config)?
        };

        Ok(Self {
            test_set,
            validation_set,
            labeled_pool,
            stream_pool,
            future_pool,
            pseudo_labeled: Vec::new(),
            config,
            class_names,
            total_images,
        })
    }

    /// Perform stratified split maintaining class balance
    fn stratified_split(
        images: &[(PathBuf, usize, String, u64)],
        config: &SplitConfig,
        rng: &mut ChaCha8Rng,
    ) -> Result<(
        Vec<LabeledImage>,
        Vec<LabeledImage>,
        Vec<LabeledImage>,
        Vec<HiddenLabelImage>,
        Vec<HiddenLabelImage>,
    )> {
        // Group images by class
        let mut by_class: HashMap<usize, Vec<&(PathBuf, usize, String, u64)>> = HashMap::new();
        for img in images {
            by_class.entry(img.1).or_default().push(img);
        }

        let mut test_set = Vec::new();
        let mut validation_set = Vec::new();
        let mut labeled_pool = Vec::new();
        let mut stream_pool = Vec::new();
        let mut future_pool = Vec::new();

        // Split each class proportionally
        for (_, class_images) in by_class.iter_mut() {
            let n = class_images.len();
            let n_test = (n as f64 * config.test_fraction).ceil() as usize;
            let n_val = (n as f64 * config.validation_fraction).ceil() as usize;
            let remaining = n.saturating_sub(n_test + n_val);
            let n_labeled = (remaining as f64 * config.labeled_fraction).ceil() as usize;
            let n_stream = (remaining as f64 * config.stream_fraction).ceil() as usize;

            // Shuffle class images
            class_images.shuffle(rng);

            let mut idx = 0;

            // Test set
            for _ in 0..n_test.min(class_images.len() - idx) {
                let (path, label, name, id) = class_images[idx];
                test_set.push(LabeledImage {
                    image_path: path.clone(),
                    label: *label,
                    class_name: name.clone(),
                    image_id: *id,
                });
                idx += 1;
            }

            // Validation set
            for _ in 0..n_val.min(class_images.len() - idx) {
                let (path, label, name, id) = class_images[idx];
                validation_set.push(LabeledImage {
                    image_path: path.clone(),
                    label: *label,
                    class_name: name.clone(),
                    image_id: *id,
                });
                idx += 1;
            }

            // Labeled pool
            for _ in 0..n_labeled.min(class_images.len() - idx) {
                let (path, label, name, id) = class_images[idx];
                labeled_pool.push(LabeledImage {
                    image_path: path.clone(),
                    label: *label,
                    class_name: name.clone(),
                    image_id: *id,
                });
                idx += 1;
            }

            // Stream pool
            for _ in 0..n_stream.min(class_images.len() - idx) {
                let (path, label, name, id) = class_images[idx];
                stream_pool.push(HiddenLabelImage {
                    image_path: path.clone(),
                    hidden_label: *label,
                    hidden_class_name: name.clone(),
                    image_id: *id,
                });
                idx += 1;
            }

            // Future pool (remaining)
            while idx < class_images.len() {
                let (path, label, name, id) = class_images[idx];
                future_pool.push(HiddenLabelImage {
                    image_path: path.clone(),
                    hidden_label: *label,
                    hidden_class_name: name.clone(),
                    image_id: *id,
                });
                idx += 1;
            }
        }

        Ok((test_set, validation_set, labeled_pool, stream_pool, future_pool))
    }

    /// Perform simple random split (not stratified)
    fn random_split(
        images: &[(PathBuf, usize, String, u64)],
        config: &SplitConfig,
    ) -> Result<(
        Vec<LabeledImage>,
        Vec<LabeledImage>,
        Vec<LabeledImage>,
        Vec<HiddenLabelImage>,
        Vec<HiddenLabelImage>,
    )> {
        let n = images.len();
        let n_test = (n as f64 * config.test_fraction).ceil() as usize;
        let n_val = (n as f64 * config.validation_fraction).ceil() as usize;
        let remaining = n.saturating_sub(n_test + n_val);
        let n_labeled = (remaining as f64 * config.labeled_fraction).ceil() as usize;
        let n_stream = (remaining as f64 * config.stream_fraction).ceil() as usize;

        let mut idx = 0;

        let test_set: Vec<_> = images[idx..idx + n_test]
            .iter()
            .map(|(path, label, name, id)| LabeledImage {
                image_path: path.clone(),
                label: *label,
                class_name: name.clone(),
                image_id: *id,
            })
            .collect();
        idx += n_test;

        let validation_set: Vec<_> = images[idx..idx + n_val]
            .iter()
            .map(|(path, label, name, id)| LabeledImage {
                image_path: path.clone(),
                label: *label,
                class_name: name.clone(),
                image_id: *id,
            })
            .collect();
        idx += n_val;

        let labeled_pool: Vec<_> = images[idx..idx + n_labeled]
            .iter()
            .map(|(path, label, name, id)| LabeledImage {
                image_path: path.clone(),
                label: *label,
                class_name: name.clone(),
                image_id: *id,
            })
            .collect();
        idx += n_labeled;

        let stream_pool: Vec<_> = images[idx..idx + n_stream]
            .iter()
            .map(|(path, label, name, id)| HiddenLabelImage {
                image_path: path.clone(),
                hidden_label: *label,
                hidden_class_name: name.clone(),
                image_id: *id,
            })
            .collect();
        idx += n_stream;

        let future_pool: Vec<_> = images[idx..]
            .iter()
            .map(|(path, label, name, id)| HiddenLabelImage {
                image_path: path.clone(),
                hidden_label: *label,
                hidden_class_name: name.clone(),
                image_id: *id,
            })
            .collect();

        Ok((test_set, validation_set, labeled_pool, stream_pool, future_pool))
    }

    /// Add a pseudo-labeled image from the stream
    pub fn add_pseudo_label(
        &mut self,
        image: &HiddenLabelImage,
        predicted_label: usize,
        confidence: f32,
        simulation_day: usize,
    ) {
        let is_correct = predicted_label == image.hidden_label;

        self.pseudo_labeled.push(PseudoLabeledImage {
            image_path: image.image_path.clone(),
            predicted_label,
            confidence,
            ground_truth: image.hidden_label,
            is_correct,
            image_id: image.image_id,
            assigned_day: simulation_day,
        });
    }

    /// Get statistics about the splits
    pub fn stats(&self) -> SplitStats {
        let pseudo_correct = self.pseudo_labeled.iter().filter(|p| p.is_correct).count();

        SplitStats {
            total_images: self.total_images,
            test_size: self.test_set.len(),
            validation_size: self.validation_set.len(),
            labeled_pool_size: self.labeled_pool.len(),
            stream_pool_size: self.stream_pool.len(),
            future_pool_size: self.future_pool.len(),
            pseudo_labeled_size: self.pseudo_labeled.len(),
            pseudo_label_accuracy: if self.pseudo_labeled.is_empty() {
                0.0
            } else {
                pseudo_correct as f64 / self.pseudo_labeled.len() as f64
            },
            num_classes: self.class_names.len(),
        }
    }

    /// Save splits to a JSON file for reproducibility
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(|e| {
            PlantVillageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        })?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load splits from a JSON file
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let splits: Self = serde_json::from_str(&json).map_err(|e| {
            PlantVillageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        })?;
        Ok(splits)
    }
}

/// Statistics about dataset splits
#[derive(Debug, Clone)]
pub struct SplitStats {
    pub total_images: usize,
    pub test_size: usize,
    pub validation_size: usize,
    pub labeled_pool_size: usize,
    pub stream_pool_size: usize,
    pub future_pool_size: usize,
    pub pseudo_labeled_size: usize,
    pub pseudo_label_accuracy: f64,
    pub num_classes: usize,
}

impl std::fmt::Display for SplitStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dataset Split Statistics:")?;
        writeln!(f, "  Total images: {}", self.total_images)?;
        writeln!(f, "  Number of classes: {}", self.num_classes)?;
        writeln!(f, "  Test set: {} ({:.1}%)", self.test_size,
            100.0 * self.test_size as f64 / self.total_images as f64)?;
        writeln!(f, "  Validation set: {} ({:.1}%)", self.validation_size,
            100.0 * self.validation_size as f64 / self.total_images as f64)?;
        writeln!(f, "  Labeled pool: {} ({:.1}%)", self.labeled_pool_size,
            100.0 * self.labeled_pool_size as f64 / self.total_images as f64)?;
        writeln!(f, "  Stream pool: {} ({:.1}%)", self.stream_pool_size,
            100.0 * self.stream_pool_size as f64 / self.total_images as f64)?;
        writeln!(f, "  Future pool: {} ({:.1}%)", self.future_pool_size,
            100.0 * self.future_pool_size as f64 / self.total_images as f64)?;
        writeln!(f, "  Pseudo-labeled: {}", self.pseudo_labeled_size)?;
        if self.pseudo_labeled_size > 0 {
            writeln!(f, "  Pseudo-label accuracy: {:.1}%", self.pseudo_label_accuracy * 100.0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_images() -> Vec<(PathBuf, usize, String)> {
        let mut images = Vec::new();
        for class in 0..5 {
            for i in 0..100 {
                images.push((
                    PathBuf::from(format!("class_{}/image_{}.jpg", class, i)),
                    class,
                    format!("Class_{}", class),
                ));
            }
        }
        images
    }

    #[test]
    fn test_default_split() {
        let images = create_test_images();
        let config = SplitConfig::default();
        let splits = DatasetSplits::from_images(images, config).unwrap();

        let stats = splits.stats();
        assert_eq!(stats.total_images, 500);
        assert_eq!(stats.num_classes, 5);

        // Check that all images are accounted for
        let total_split = stats.test_size
            + stats.validation_size
            + stats.labeled_pool_size
            + stats.stream_pool_size
            + stats.future_pool_size;
        assert_eq!(total_split, 500);
    }

    #[test]
    fn test_stratified_maintains_class_balance() {
        let images = create_test_images();
        let config = SplitConfig::default();
        let splits = DatasetSplits::from_images(images, config).unwrap();

        // Check that each class is represented in test set
        let mut test_classes: HashMap<usize, usize> = HashMap::new();
        for img in &splits.test_set {
            *test_classes.entry(img.label).or_default() += 1;
        }

        // Each class should have some representation
        for class in 0..5 {
            assert!(test_classes.get(&class).unwrap_or(&0) > &0);
        }
    }

    #[test]
    fn test_reproducibility() {
        let images = create_test_images();
        let config = SplitConfig::default();

        let splits1 = DatasetSplits::from_images(images.clone(), config.clone()).unwrap();
        let splits2 = DatasetSplits::from_images(images, config).unwrap();

        // Same seed should produce same number of items in each split
        assert_eq!(splits1.test_set.len(), splits2.test_set.len());
        assert_eq!(splits1.validation_set.len(), splits2.validation_set.len());
        assert_eq!(splits1.labeled_pool.len(), splits2.labeled_pool.len());
        assert_eq!(splits1.stream_pool.len(), splits2.stream_pool.len());
    }

    #[test]
    fn test_add_pseudo_label() {
        let images = create_test_images();
        let config = SplitConfig::default();
        let mut splits = DatasetSplits::from_images(images, config).unwrap();

        if let Some(stream_img) = splits.stream_pool.first().cloned() {
            splits.add_pseudo_label(&stream_img, stream_img.hidden_label, 0.95, 1);

            assert_eq!(splits.pseudo_labeled.len(), 1);
            assert!(splits.pseudo_labeled[0].is_correct);
        }
    }
}
