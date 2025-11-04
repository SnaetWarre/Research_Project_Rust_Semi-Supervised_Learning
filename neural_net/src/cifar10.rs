//! CIFAR-10 dataset loader and utilities
//!
//! This module provides functionality to download, load, and process the CIFAR-10 dataset.
//! CIFAR-10 consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

use crate::Float;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

/// CIFAR-10 class names
pub const CLASS_NAMES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

/// Single CIFAR-10 image with label
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cifar10Image {
    /// Image data as flat RGB vector (3072 values: 32x32x3)
    pub data: Vec<u8>,
    /// Class label (0-9)
    pub label: usize,
    /// Whether this image was manually labeled or pseudo-labeled
    pub is_manually_labeled: bool,
    /// Confidence score (for pseudo-labeled images)
    pub confidence: Option<Float>,
}

impl Cifar10Image {
    /// Create new CIFAR-10 image
    pub fn new(data: Vec<u8>, label: usize) -> Self {
        Self {
            data,
            label,
            is_manually_labeled: true,
            confidence: None,
        }
    }

    /// Mark as pseudo-labeled with confidence
    pub fn as_pseudo_labeled(mut self, confidence: Float) -> Self {
        self.is_manually_labeled = false;
        self.confidence = Some(confidence);
        self
    }

    /// Get normalized image data (values in [0, 1])
    pub fn normalized_data(&self) -> Vec<Float> {
        self.data.iter().map(|&x| x as Float / 255.0).collect()
    }

    /// Get class name
    pub fn class_name(&self) -> &str {
        CLASS_NAMES[self.label]
    }

    /// Get image dimensions
    pub fn shape(&self) -> (usize, usize, usize) {
        (32, 32, 3) // CIFAR-10 is always 32x32 RGB
    }
}

/// CIFAR-10 dataset
#[derive(Clone, Debug)]
pub struct Cifar10Dataset {
    /// All images in the dataset
    pub images: Vec<Cifar10Image>,
    /// Dataset split (train/test)
    pub split: DatasetSplit,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DatasetSplit {
    Train,
    Test,
}

impl Cifar10Dataset {
    /// Create empty dataset
    pub fn new(split: DatasetSplit) -> Self {
        Self {
            images: Vec::new(),
            split,
        }
    }

    /// Load CIFAR-10 from directory (expects binary format)
    pub fn load_from_binary(data_dir: impl AsRef<Path>, split: DatasetSplit) -> Result<Self, String> {
        let data_dir = data_dir.as_ref();
        let mut dataset = Self::new(split.clone());

        match split {
            DatasetSplit::Train => {
                // Load all 5 training batches
                for i in 1..=5 {
                    let batch_file = data_dir.join(format!("data_batch_{}.bin", i));
                    let images = load_cifar_batch(&batch_file)?;
                    dataset.images.extend(images);
                }
            }
            DatasetSplit::Test => {
                // Load test batch
                let test_file = data_dir.join("test_batch.bin");
                dataset.images = load_cifar_batch(&test_file)?;
            }
        }

        Ok(dataset)
    }

    /// Get number of images
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Get images by indices
    pub fn get_subset(&self, indices: &[usize]) -> Vec<Cifar10Image> {
        indices.iter()
            .filter_map(|&i| self.images.get(i).cloned())
            .collect()
    }

    /// Split dataset into labeled and unlabeled portions
    pub fn split_labeled_unlabeled(&self, labeled_fraction: Float) -> (Vec<usize>, Vec<usize>) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let total = self.images.len();
        let num_labeled = (total as Float * labeled_fraction) as usize;

        let mut indices: Vec<usize> = (0..total).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);

        let labeled = indices[..num_labeled].to_vec();
        let unlabeled = indices[num_labeled..].to_vec();

        (labeled, unlabeled)
    }

    /// Split dataset ensuring balanced classes
    pub fn split_labeled_unlabeled_balanced(&self, labeled_per_class: usize) -> (Vec<usize>, Vec<usize>) {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut labeled = Vec::new();
        let mut unlabeled = Vec::new();
        let mut rng = thread_rng();

        // Group by class
        for class_id in 0..10 {
            let mut class_indices: Vec<usize> = self.images.iter()
                .enumerate()
                .filter(|(_, img)| img.label == class_id)
                .map(|(i, _)| i)
                .collect();

            class_indices.shuffle(&mut rng);

            let to_label = labeled_per_class.min(class_indices.len());
            labeled.extend_from_slice(&class_indices[..to_label]);
            unlabeled.extend_from_slice(&class_indices[to_label..]);
        }

        (labeled, unlabeled)
    }

    /// Get normalized image data for batch
    pub fn get_batch_data(&self, indices: &[usize]) -> Vec<Vec<Float>> {
        indices.iter()
            .filter_map(|&i| self.images.get(i))
            .map(|img| img.normalized_data())
            .collect()
    }

    /// Get one-hot encoded labels for batch
    pub fn get_batch_labels(&self, indices: &[usize]) -> Vec<Vec<Float>> {
        indices.iter()
            .filter_map(|&i| self.images.get(i))
            .map(|img| {
                let mut one_hot = vec![0.0; 10];
                one_hot[img.label] = 1.0;
                one_hot
            })
            .collect()
    }

    /// Get class distribution
    pub fn class_distribution(&self) -> [usize; 10] {
        let mut counts = [0; 10];
        for img in &self.images {
            counts[img.label] += 1;
        }
        counts
    }

    /// Save dataset to JSON
    pub fn save_to_json(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.images)
            .map_err(|e| format!("Failed to serialize: {}", e))?;

        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))
    }

    /// Load dataset from JSON
    pub fn load_from_json(path: impl AsRef<Path>, split: DatasetSplit) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let images: Vec<Cifar10Image> = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to deserialize: {}", e))?;

        Ok(Self { images, split })
    }
}

/// Load a single CIFAR-10 batch file
fn load_cifar_batch(path: &Path) -> Result<Vec<Cifar10Image>, String> {
    let mut file = File::open(path)
        .map_err(|e| format!("Failed to open file {:?}: {}", path, e))?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // CIFAR-10 binary format: [label (1 byte)][red (1024 bytes)][green (1024 bytes)][blue (1024 bytes)]
    const IMAGE_SIZE: usize = 3072; // 32 * 32 * 3
    const RECORD_SIZE: usize = 1 + IMAGE_SIZE;
    const NUM_IMAGES: usize = 10000;

    if buffer.len() != RECORD_SIZE * NUM_IMAGES {
        return Err(format!(
            "Invalid file size. Expected {}, got {}",
            RECORD_SIZE * NUM_IMAGES,
            buffer.len()
        ));
    }

    let mut images = Vec::with_capacity(NUM_IMAGES);

    for i in 0..NUM_IMAGES {
        let offset = i * RECORD_SIZE;
        let label = buffer[offset] as usize;

        // Read RGB data (stored as R...R, G...G, B...B)
        let mut data = vec![0u8; IMAGE_SIZE];
        for j in 0..1024 {
            data[j * 3] = buffer[offset + 1 + j];           // R
            data[j * 3 + 1] = buffer[offset + 1 + 1024 + j]; // G
            data[j * 3 + 2] = buffer[offset + 1 + 2048 + j]; // B
        }

        images.push(Cifar10Image::new(data, label));
    }

    Ok(images)
}

/// Download CIFAR-10 dataset
pub fn download_cifar10(data_dir: impl AsRef<Path>) -> Result<(), String> {
    let data_dir = data_dir.as_ref();
    fs::create_dir_all(data_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;

    let url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    let tar_gz_path = data_dir.join("cifar-10-binary.tar.gz");

    // Check if already downloaded
    if tar_gz_path.exists() {
        println!("CIFAR-10 archive already exists, skipping download");
    } else {
        println!("Downloading CIFAR-10 dataset from {}...", url);

        let response = reqwest::blocking::get(url)
            .map_err(|e| format!("Failed to download: {}", e))?;

        let bytes = response.bytes()
            .map_err(|e| format!("Failed to read response: {}", e))?;

        let mut file = File::create(&tar_gz_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        file.write_all(&bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        println!("Download complete!");
    }

    // Extract if not already extracted
    let extracted_dir = data_dir.join("cifar-10-batches-bin");
    if extracted_dir.exists() {
        println!("CIFAR-10 already extracted");
    } else {
        println!("Extracting CIFAR-10...");
        extract_tar_gz(&tar_gz_path, data_dir)?;
        println!("Extraction complete!");
    }

    Ok(())
}

/// Extract tar.gz file
fn extract_tar_gz(tar_gz_path: &Path, output_dir: &Path) -> Result<(), String> {
    let tar_gz = File::open(tar_gz_path)
        .map_err(|e| format!("Failed to open tar.gz: {}", e))?;

    let decompressor = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(decompressor);

    archive.unpack(output_dir)
        .map_err(|e| format!("Failed to extract: {}", e))?;

    Ok(())
}

/// Helper to create batches for training
pub struct DataLoader {
    indices: Vec<usize>,
    batch_size: usize,
    current: usize,
    shuffle: bool,
}

impl DataLoader {
    /// Create new data loader
    pub fn new(num_samples: usize, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..num_samples).collect();
        Self {
            indices,
            batch_size,
            current: 0,
            shuffle,
        }
    }

    /// Create from specific indices
    pub fn from_indices(indices: Vec<usize>, batch_size: usize, shuffle: bool) -> Self {
        Self {
            indices,
            batch_size,
            current: 0,
            shuffle,
        }
    }

    /// Reset and optionally shuffle
    pub fn reset(&mut self) {
        self.current = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            use rand::thread_rng;
            let mut rng = thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get next batch
    pub fn next_batch(&mut self) -> Option<Vec<usize>> {
        if self.current >= self.indices.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.indices.len());
        let batch = self.indices[self.current..end].to_vec();
        self.current = end;

        Some(batch)
    }

    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cifar10_image_creation() {
        let data = vec![0u8; 3072];
        let img = Cifar10Image::new(data, 5);

        assert_eq!(img.label, 5);
        assert_eq!(img.class_name(), "dog");
        assert_eq!(img.shape(), (32, 32, 3));
        assert!(img.is_manually_labeled);
        assert!(img.confidence.is_none());
    }

    #[test]
    fn test_pseudo_labeling() {
        let data = vec![0u8; 3072];
        let img = Cifar10Image::new(data, 3).as_pseudo_labeled(0.95);

        assert!(!img.is_manually_labeled);
        assert_eq!(img.confidence, Some(0.95));
    }

    #[test]
    fn test_data_loader() {
        let mut loader = DataLoader::new(100, 10, false);

        assert_eq!(loader.num_batches(), 10);

        let batch1 = loader.next_batch().unwrap();
        assert_eq!(batch1.len(), 10);
        assert_eq!(batch1[0], 0);
        assert_eq!(batch1[9], 9);

        // Get all batches
        let mut total = batch1.len();
        while let Some(batch) = loader.next_batch() {
            total += batch.len();
        }
        assert_eq!(total, 100);

        // Reset
        loader.reset();
        let batch1_again = loader.next_batch().unwrap();
        assert_eq!(batch1_again.len(), 10);
    }

    #[test]
    fn test_class_names() {
        assert_eq!(CLASS_NAMES.len(), 10);
        assert_eq!(CLASS_NAMES[0], "airplane");
        assert_eq!(CLASS_NAMES[9], "truck");
    }
}
