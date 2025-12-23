//! Burn Dataset Integration for PlantVillage
//!
//! This module implements Burn's Dataset trait and Batcher for efficient
//! data loading and batching during training.

use std::path::PathBuf;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;
use image::imageops::FilterType;
use image::ImageReader;
use serde::{Deserialize, Serialize};

use crate::IMAGE_SIZE;

/// A single PlantVillage item ready for Burn
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlantVillageItem {
    /// Image data as flattened CHW float array [3 * H * W]
    pub image: Vec<f32>,
    /// Class label (0-38)
    pub label: usize,
    /// Image path (for debugging/logging)
    pub path: String,
}

impl PlantVillageItem {
    /// Create a new item by loading and preprocessing an image
    pub fn from_path(path: &PathBuf, label: usize, image_size: usize) -> anyhow::Result<Self> {
        let img = ImageReader::open(path)?
            .decode()?
            .resize_exact(
                image_size as u32,
                image_size as u32,
                FilterType::Triangle,
            )
            .to_rgb8();

        let (width, height) = (image_size, image_size);
        let mut image = vec![0.0f32; 3 * height * width];

        // Convert to CHW format and normalize to [0, 1]
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                image[0 * height * width + y * width + x] = pixel[0] as f32 / 255.0;
                image[1 * height * width + y * width + x] = pixel[1] as f32 / 255.0;
                image[2 * height * width + y * width + x] = pixel[2] as f32 / 255.0;
            }
        }

        Ok(Self {
            image,
            label,
            path: path.to_string_lossy().to_string(),
        })
    }

    /// Create from pre-loaded image data
    pub fn from_data(image: Vec<f32>, label: usize, path: String) -> Self {
        Self { image, label, path }
    }
}

/// PlantVillage Dataset implementing Burn's Dataset trait
///
/// This dataset lazily loads images on demand for memory efficiency.
#[derive(Debug, Clone)]
pub struct PlantVillageBurnDataset {
    /// List of (image_path, label) pairs
    samples: Vec<(PathBuf, usize)>,
    /// Target image size
    image_size: usize,
    /// Whether to cache loaded images in memory (reserved for future use)
    #[allow(dead_code)]
    cache_enabled: bool,
    /// Cached items (only used if cache_enabled)
    cached_items: Option<Vec<PlantVillageItem>>,
}

impl PlantVillageBurnDataset {
    /// Create a new dataset from a list of samples
    pub fn new(samples: Vec<(PathBuf, usize)>, image_size: usize) -> Self {
        Self {
            samples,
            image_size,
            cache_enabled: false,
            cached_items: None,
        }
    }

    /// Create a new dataset with caching enabled (loads all images into memory)
    pub fn new_cached(samples: Vec<(PathBuf, usize)>, image_size: usize) -> anyhow::Result<Self> {
        let cached_items: Result<Vec<_>, _> = samples
            .iter()
            .map(|(path, label)| PlantVillageItem::from_path(path, *label, image_size))
            .collect();

        Ok(Self {
            samples,
            image_size,
            cache_enabled: true,
            cached_items: Some(cached_items?),
        })
    }

    /// Create from PlantVillageDataset loader
    pub fn from_loader(loader: &super::loader::PlantVillageDataset) -> Self {
        let samples: Vec<_> = loader
            .samples
            .iter()
            .map(|s| (s.path.clone(), s.label))
            .collect();

        Self::new(samples, loader.image_size.0 as usize)
    }

    /// Get the number of classes in the dataset
    pub fn num_classes(&self) -> usize {
        self.samples
            .iter()
            .map(|(_, label)| *label)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0)
    }

    /// Get samples per class count
    pub fn class_distribution(&self) -> Vec<usize> {
        let num_classes = self.num_classes();
        let mut counts = vec![0usize; num_classes];
        for (_, label) in &self.samples {
            if *label < num_classes {
                counts[*label] += 1;
            }
        }
        counts
    }
}

impl Dataset<PlantVillageItem> for PlantVillageBurnDataset {
    fn get(&self, index: usize) -> Option<PlantVillageItem> {
        if index >= self.samples.len() {
            return None;
        }

        // Use cached item if available
        if let Some(ref cached) = self.cached_items {
            return cached.get(index).cloned();
        }

        // Otherwise load on demand
        let (path, label) = &self.samples[index];
        PlantVillageItem::from_path(path, *label, self.image_size).ok()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

/// A batch of PlantVillage images for training
#[derive(Clone, Debug)]
pub struct PlantVillageBatch<B: Backend> {
    /// Batch of images with shape [batch_size, 3, height, width]
    pub images: Tensor<B, 4>,
    /// Batch of labels with shape [batch_size]
    pub targets: Tensor<B, 1, Int>,
}

/// Batcher for creating PlantVillage training batches
#[derive(Clone, Debug)]
pub struct PlantVillageBatcher<B: Backend> {
    device: B::Device,
    image_size: usize,
}

impl<B: Backend> PlantVillageBatcher<B> {
    /// Create a new batcher for the given device
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            image_size: IMAGE_SIZE,
        }
    }

    /// Create a batcher with custom image size
    pub fn with_image_size(device: B::Device, image_size: usize) -> Self {
        Self { device, image_size }
    }
}

impl<B: Backend> Batcher<PlantVillageItem, PlantVillageBatch<B>> for PlantVillageBatcher<B> {
    fn batch(&self, items: Vec<PlantVillageItem>) -> PlantVillageBatch<B> {
        let batch_size = items.len();
        let channels = 3;
        let height = self.image_size;
        let width = self.image_size;

        // Flatten all images into a single vector
        let images_data: Vec<f32> = items.iter().flat_map(|item| item.image.clone()).collect();

        // Create image tensor with shape [batch_size, channels, height, width]
        let images = Tensor::<B, 4>::from_floats(
            TensorData::new(images_data, [batch_size, channels, height, width]),
            &self.device,
        );

        // Apply ImageNet normalization: (x - mean) / std
        // ImageNet mean: [0.485, 0.456, 0.406]
        // ImageNet std: [0.229, 0.224, 0.225]
        let mean = Tensor::<B, 4>::from_floats(
            TensorData::new(
                vec![0.485f32, 0.456, 0.406],
                [1, 3, 1, 1],
            ),
            &self.device,
        );
        let std = Tensor::<B, 4>::from_floats(
            TensorData::new(
                vec![0.229f32, 0.224, 0.225],
                [1, 3, 1, 1],
            ),
            &self.device,
        );

        let images = (images - mean) / std;

        // Create targets tensor
        let targets_data: Vec<i64> = items.iter().map(|item| item.label as i64).collect();
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::new(targets_data, [batch_size]),
            &self.device,
        );

        PlantVillageBatch { images, targets }
    }
}

/// A pseudo-labeled item with confidence score
#[derive(Clone, Debug)]
pub struct PseudoLabeledItem {
    /// The base item
    pub item: PlantVillageItem,
    /// Confidence score from the model (0.0 - 1.0)
    pub confidence: f32,
    /// Ground truth label (hidden, only for evaluation)
    pub ground_truth: Option<usize>,
}

impl PseudoLabeledItem {
    /// Create a new pseudo-labeled item
    pub fn new(item: PlantVillageItem, confidence: f32, ground_truth: Option<usize>) -> Self {
        Self {
            item,
            confidence,
            ground_truth,
        }
    }

    /// Check if the pseudo-label is correct (only if ground truth is available)
    pub fn is_correct(&self) -> Option<bool> {
        self.ground_truth.map(|gt| gt == self.item.label)
    }
}

/// A batch of pseudo-labeled images
#[derive(Clone, Debug)]
pub struct PseudoLabelBatch<B: Backend> {
    /// Batch of images with shape [batch_size, 3, height, width]
    pub images: Tensor<B, 4>,
    /// Batch of pseudo-labels with shape [batch_size]
    pub targets: Tensor<B, 1, Int>,
    /// Batch of confidence weights with shape [batch_size]
    pub weights: Tensor<B, 1>,
}

/// Batcher for pseudo-labeled data with confidence weights
#[derive(Clone, Debug)]
pub struct PseudoLabelBatcher<B: Backend> {
    device: B::Device,
    image_size: usize,
}

impl<B: Backend> PseudoLabelBatcher<B> {
    /// Create a new pseudo-label batcher
    pub fn new(device: B::Device) -> Self {
        Self {
            device,
            image_size: IMAGE_SIZE,
        }
    }
}

impl<B: Backend> Batcher<PseudoLabeledItem, PseudoLabelBatch<B>> for PseudoLabelBatcher<B> {
    fn batch(&self, items: Vec<PseudoLabeledItem>) -> PseudoLabelBatch<B> {
        let batch_size = items.len();
        let channels = 3;
        let height = self.image_size;
        let width = self.image_size;

        // Flatten all images
        let images_data: Vec<f32> = items
            .iter()
            .flat_map(|item| item.item.image.clone())
            .collect();

        // Create image tensor
        let images = Tensor::<B, 4>::from_floats(
            TensorData::new(images_data, [batch_size, channels, height, width]),
            &self.device,
        );

        // Apply normalization
        let mean = Tensor::<B, 4>::from_floats(
            TensorData::new(vec![0.485f32, 0.456, 0.406], [1, 3, 1, 1]),
            &self.device,
        );
        let std = Tensor::<B, 4>::from_floats(
            TensorData::new(vec![0.229f32, 0.224, 0.225], [1, 3, 1, 1]),
            &self.device,
        );
        let images = (images - mean) / std;

        // Create targets tensor
        let targets_data: Vec<i64> = items.iter().map(|item| item.item.label as i64).collect();
        let targets = Tensor::<B, 1, Int>::from_data(
            TensorData::new(targets_data, [batch_size]),
            &self.device,
        );

        // Create weights tensor (confidence scores)
        let weights_data: Vec<f32> = items.iter().map(|item| item.confidence).collect();
        let weights = Tensor::<B, 1>::from_floats(
            TensorData::new(weights_data, [batch_size]),
            &self.device,
        );

        PseudoLabelBatch {
            images,
            targets,
            weights,
        }
    }
}

/// Dataset for pseudo-labeled items
#[derive(Clone, Debug)]
pub struct PseudoLabelDataset {
    items: Vec<PseudoLabeledItem>,
}

impl PseudoLabelDataset {
    /// Create an empty pseudo-label dataset
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add a pseudo-labeled item
    pub fn add(&mut self, item: PseudoLabeledItem) {
        self.items.push(item);
    }

    /// Add multiple items
    pub fn extend(&mut self, items: impl IntoIterator<Item = PseudoLabeledItem>) {
        self.items.extend(items);
    }

    /// Get all items
    pub fn items(&self) -> &[PseudoLabeledItem] {
        &self.items
    }

    /// Clear all items
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Calculate pseudo-label precision (if ground truth is available)
    pub fn precision(&self) -> Option<f64> {
        let (correct, total) = self
            .items
            .iter()
            .filter_map(|item| item.is_correct())
            .fold((0, 0), |(c, t), is_correct| {
                (c + is_correct as usize, t + 1)
            });

        if total > 0 {
            Some(correct as f64 / total as f64)
        } else {
            None
        }
    }

    /// Get class distribution of pseudo-labels
    pub fn class_distribution(&self, num_classes: usize) -> Vec<usize> {
        let mut counts = vec![0usize; num_classes];
        for item in &self.items {
            if item.item.label < num_classes {
                counts[item.item.label] += 1;
            }
        }
        counts
    }

    /// Get average confidence
    pub fn average_confidence(&self) -> f64 {
        if self.items.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.items.iter().map(|item| item.confidence).sum();
        sum as f64 / self.items.len() as f64
    }
}

impl Default for PseudoLabelDataset {
    fn default() -> Self {
        Self::new()
    }
}

impl Dataset<PseudoLabeledItem> for PseudoLabelDataset {
    fn get(&self, index: usize) -> Option<PseudoLabeledItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// Combined dataset that merges labeled and pseudo-labeled data
#[derive(Clone, Debug)]
pub struct CombinedDataset {
    /// Original labeled items
    labeled_items: Vec<PlantVillageItem>,
    /// Pseudo-labeled items
    pseudo_items: Vec<PlantVillageItem>,
}

impl CombinedDataset {
    /// Create a new combined dataset
    pub fn new(labeled_items: Vec<PlantVillageItem>, pseudo_items: Vec<PlantVillageItem>) -> Self {
        Self {
            labeled_items,
            pseudo_items,
        }
    }

    /// Get the number of labeled items
    pub fn num_labeled(&self) -> usize {
        self.labeled_items.len()
    }

    /// Get the number of pseudo-labeled items
    pub fn num_pseudo(&self) -> usize {
        self.pseudo_items.len()
    }
}

impl Dataset<PlantVillageItem> for CombinedDataset {
    fn get(&self, index: usize) -> Option<PlantVillageItem> {
        if index < self.labeled_items.len() {
            self.labeled_items.get(index).cloned()
        } else {
            self.pseudo_items
                .get(index - self.labeled_items.len())
                .cloned()
        }
    }

    fn len(&self) -> usize {
        self.labeled_items.len() + self.pseudo_items.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plant_village_item_creation() {
        let image = vec![0.5f32; 3 * 256 * 256];
        let item = PlantVillageItem::from_data(image.clone(), 5, "test.jpg".to_string());

        assert_eq!(item.label, 5);
        assert_eq!(item.image.len(), 3 * 256 * 256);
        assert_eq!(item.path, "test.jpg");
    }

    #[test]
    fn test_pseudo_label_dataset() {
        let mut dataset = PseudoLabelDataset::new();

        let image = vec![0.5f32; 3 * 256 * 256];
        let item = PlantVillageItem::from_data(image, 3, "test.jpg".to_string());

        dataset.add(PseudoLabeledItem::new(item.clone(), 0.95, Some(3)));
        dataset.add(PseudoLabeledItem::new(
            PlantVillageItem::from_data(vec![0.5f32; 3 * 256 * 256], 5, "test2.jpg".to_string()),
            0.85,
            Some(3), // Wrong prediction
        ));

        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.precision(), Some(0.5)); // 1 correct out of 2
        assert!((dataset.average_confidence() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_combined_dataset() {
        let labeled = vec![
            PlantVillageItem::from_data(vec![0.0; 3 * 256 * 256], 0, "l1.jpg".to_string()),
            PlantVillageItem::from_data(vec![0.0; 3 * 256 * 256], 1, "l2.jpg".to_string()),
        ];
        let pseudo = vec![PlantVillageItem::from_data(
            vec![0.0; 3 * 256 * 256],
            2,
            "p1.jpg".to_string(),
        )];

        let combined = CombinedDataset::new(labeled, pseudo);

        assert_eq!(combined.len(), 3);
        assert_eq!(combined.num_labeled(), 2);
        assert_eq!(combined.num_pseudo(), 1);

        assert_eq!(combined.get(0).unwrap().label, 0);
        assert_eq!(combined.get(1).unwrap().label, 1);
        assert_eq!(combined.get(2).unwrap().label, 2);
        assert!(combined.get(3).is_none());
    }

    #[test]
    fn test_class_distribution() {
        let samples = vec![
            (PathBuf::from("a.jpg"), 0),
            (PathBuf::from("b.jpg"), 0),
            (PathBuf::from("c.jpg"), 1),
            (PathBuf::from("d.jpg"), 2),
        ];

        let dataset = PlantVillageBurnDataset::new(samples, 256);

        assert_eq!(dataset.num_classes(), 3);
        assert_eq!(dataset.class_distribution(), vec![2, 1, 1]);
    }
}
