//! PlantVillage Dataset Loader
//!
//! This module handles loading the PlantVillage dataset from disk,
//! organizing it into labeled/unlabeled splits for semi-supervised learning.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use walkdir::WalkDir;

use crate::IMAGE_SIZE;

/// A single image sample with its label and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSample {
    /// Path to the image file
    pub path: PathBuf,
    /// Class label index (0-38)
    pub label: usize,
    /// Class name (e.g., "Tomato___Late_blight")
    pub class_name: String,
    /// Unique sample ID
    pub id: usize,
}

/// PlantVillage dataset with lazy loading
#[derive(Debug)]
pub struct PlantVillageDataset {
    /// Root directory of the dataset
    pub root_dir: PathBuf,
    /// All samples in the dataset
    pub samples: Vec<ImageSample>,
    /// Mapping from class name to label index
    pub class_to_idx: HashMap<String, usize>,
    /// Mapping from label index to class name
    pub idx_to_class: HashMap<usize, String>,
    /// Target image size (width, height)
    pub image_size: (u32, u32),
}

impl PlantVillageDataset {
    /// Create a new PlantVillage dataset from a directory
    ///
    /// The directory should be structured as:
    /// ```text
    /// root_dir/
    /// ├── Apple___Apple_scab/
    /// │   ├── image1.jpg
    /// │   └── image2.jpg
    /// ├── Apple___Black_rot/
    /// │   └── ...
    /// └── ...
    /// ```
    pub fn new<P: AsRef<Path>>(root_dir: P) -> Result<Self> {
        let root_dir = root_dir.as_ref().to_path_buf();
        info!("Loading PlantVillage dataset from: {:?}", root_dir);

        if !root_dir.exists() {
            anyhow::bail!("Dataset directory does not exist: {:?}", root_dir);
        }

        // Discover all class directories
        let mut class_dirs: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&root_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    class_dirs.push(name.to_string());
                }
            }
        }
        class_dirs.sort();

        info!("Found {} classes", class_dirs.len());

        // Create class mappings
        let class_to_idx: HashMap<String, usize> = class_dirs
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect();

        let idx_to_class: HashMap<usize, String> = class_dirs
            .iter()
            .enumerate()
            .map(|(idx, name)| (idx, name.clone()))
            .collect();

        // Load all samples
        let mut samples = Vec::new();
        let mut sample_id: usize = 0;

        for class_name in &class_dirs {
            let class_dir = root_dir.join(class_name);
            let label = class_to_idx[class_name];

            for entry in WalkDir::new(&class_dir)
                .min_depth(1)
                .max_depth(1)
                .into_iter()
                .filter_map(|e| e.ok())
            {
                let path = entry.path().to_path_buf();

                // Only include image files
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ["jpg", "jpeg", "png", "bmp"].contains(&ext.as_str()) {
                        samples.push(ImageSample {
                            path,
                            label,
                            class_name: class_name.clone(),
                            id: sample_id,
                        });
                        sample_id += 1;
                    }
                }
            }

            debug!(
                "Class '{}' (label {}): loaded samples",
                class_name, label
            );
        }

        info!("Loaded {} total samples", samples.len());

        Ok(Self {
            root_dir,
            samples,
            class_to_idx,
            idx_to_class,
            image_size: (IMAGE_SIZE as u32, IMAGE_SIZE as u32),
        })
    }

    /// Get the number of samples in the dataset
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get the number of classes
    pub fn num_classes(&self) -> usize {
        self.class_to_idx.len()
    }

    /// Load an image from disk and resize it
    pub fn load_image(&self, sample: &ImageSample) -> Result<DynamicImage> {
        let img = ImageReader::open(&sample.path)
            .with_context(|| format!("Failed to open image: {:?}", sample.path))?
            .decode()
            .with_context(|| format!("Failed to decode image: {:?}", sample.path))?;

        // Resize to target size
        let resized = img.resize_exact(
            self.image_size.0,
            self.image_size.1,
            image::imageops::FilterType::Triangle,
        );

        Ok(resized)
    }

    /// Load an image and convert to normalized float tensor data
    ///
    /// Returns a Vec<f32> with shape [3, height, width] in CHW format,
    /// normalized to [0, 1] range
    pub fn load_image_tensor(&self, sample: &ImageSample) -> Result<Vec<f32>> {
        let img = self.load_image(sample)?;
        let rgb = img.to_rgb8();

        let (width, height) = (self.image_size.0 as usize, self.image_size.1 as usize);
        let mut tensor = vec![0.0f32; 3 * height * width];

        // Convert to CHW format and normalize to [0, 1]
        for y in 0..height {
            for x in 0..width {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                tensor[0 * height * width + y * width + x] = pixel[0] as f32 / 255.0; // R
                tensor[1 * height * width + y * width + x] = pixel[1] as f32 / 255.0; // G
                tensor[2 * height * width + y * width + x] = pixel[2] as f32 / 255.0; // B
            }
        }

        Ok(tensor)
    }

    /// Shuffle the samples in place with a given seed
    pub fn shuffle(&mut self, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        self.samples.shuffle(&mut rng);
    }

    /// Get samples for a specific class
    pub fn get_samples_by_class(&self, class_idx: usize) -> Vec<&ImageSample> {
        self.samples
            .iter()
            .filter(|s| s.label == class_idx)
            .collect()
    }

    /// Get statistics about the dataset
    pub fn get_stats(&self) -> DatasetStats {
        let mut class_counts = vec![0usize; self.num_classes()];
        for sample in &self.samples {
            class_counts[sample.label] += 1;
        }

        DatasetStats {
            total_samples: self.samples.len(),
            num_classes: self.num_classes(),
            class_counts,
            class_names: self.idx_to_class.clone(),
        }
    }
}

/// Statistics about the dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_samples: usize,
    pub num_classes: usize,
    pub class_counts: Vec<usize>,
    pub class_names: HashMap<usize, String>,
}

impl DatasetStats {
    /// Print statistics to console
    pub fn print(&self) {
        println!("\n📊 Dataset Statistics:");
        println!("  Total samples: {}", self.total_samples);
        println!("  Number of classes: {}", self.num_classes);
        println!("\n  Samples per class:");

        let mut sorted: Vec<_> = self.class_names.iter().collect();
        sorted.sort_by_key(|(idx, _)| *idx);

        for (idx, name) in sorted {
            let count = self.class_counts[*idx];
            let bar_len = (count as f32 / self.total_samples as f32 * 40.0) as usize;
            let bar: String = "█".repeat(bar_len);
            println!("    {:3}. {:40} {:5} {}", idx, name, count, bar);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_sample_creation() {
        let sample = ImageSample {
            path: PathBuf::from("/test/image.jpg"),
            label: 5,
            class_name: "Tomato___Late_blight".to_string(),
            id: 42,
        };

        assert_eq!(sample.label, 5);
        assert_eq!(sample.id, 42);
    }
}
