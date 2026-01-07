//! Dataset Preparation Module
//!
//! Creates a balanced dataset from raw PlantVillage data.
//! Uses undersampling to ensure equal class representation.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

/// Configuration for dataset preparation
#[derive(Debug, Clone)]
pub struct PrepareConfig {
    /// Random seed for reproducibility
    pub seed: u64,
    /// Samples per class (None = use minimum class size for perfect balance)
    pub samples_per_class: Option<usize>,
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            samples_per_class: None,
        }
    }
}

/// Statistics about the prepared dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareStats {
    pub num_classes: usize,
    pub samples_per_class: usize,
    pub total_samples: usize,
    pub original_total: usize,
    pub original_min_class: usize,
    pub original_max_class: usize,
    pub imbalance_ratio: f64,
    pub class_stats: HashMap<String, ClassStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassStat {
    pub original: usize,
    pub used: usize,
}

/// Prepare a balanced dataset from raw PlantVillage data
///
/// This function:
/// 1. Scans the source directory for class folders
/// 2. Finds the minimum class size (or uses specified samples_per_class)
/// 3. Randomly samples that many images from each class
/// 4. Copies them to the output directory with consistent naming
pub fn prepare_balanced_dataset(
    source_dir: &Path,
    output_dir: &Path,
    config: &PrepareConfig,
) -> Result<PrepareStats> {
    println!("Preparing balanced dataset...");
    println!("  Source: {:?}", source_dir);
    println!("  Output: {:?}", output_dir);

    // Find source directory - check for nested structure
    let actual_source = find_source_directory(source_dir)?;
    println!("  Found images in: {:?}", actual_source);

    // Discover all classes and their images
    let class_images = discover_class_images(&actual_source)?;
    
    if class_images.is_empty() {
        anyhow::bail!("No class directories found in source");
    }

    println!("  Found {} classes", class_images.len());

    // Calculate statistics
    let original_total: usize = class_images.values().map(|v| v.len()).sum();
    let original_min = class_images.values().map(|v| v.len()).min().unwrap_or(0);
    let original_max = class_images.values().map(|v| v.len()).max().unwrap_or(0);
    let imbalance_ratio = if original_min > 0 {
        original_max as f64 / original_min as f64
    } else {
        f64::INFINITY
    };

    println!("  Original total: {} images", original_total);
    println!("  Smallest class: {} images", original_min);
    println!("  Largest class: {} images", original_max);
    println!("  Imbalance ratio: {:.1}:1", imbalance_ratio);

    // Determine samples per class
    let samples_per_class = config.samples_per_class.unwrap_or(original_min);
    
    if samples_per_class > original_min {
        println!(
            "  Warning: Requested {} samples but smallest class has {} - using {}",
            samples_per_class, original_min, original_min
        );
    }
    let samples_per_class = samples_per_class.min(original_min);

    println!("  Using {} samples per class", samples_per_class);

    // Clean and create output directory
    if output_dir.exists() {
        println!("  Removing existing output directory...");
        fs::remove_dir_all(output_dir)
            .context("Failed to remove existing output directory")?;
    }
    fs::create_dir_all(output_dir)
        .context("Failed to create output directory")?;

    // Create RNG
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    // Process each class
    let mut class_stats = HashMap::new();
    let mut total_copied = 0;

    for (class_name, mut images) in class_images {
        let original_count = images.len();

        // Shuffle and take samples
        images.shuffle(&mut rng);
        let selected: Vec<_> = images.into_iter().take(samples_per_class).collect();
        let used_count = selected.len();

        // Create class directory
        let class_dir = output_dir.join(&class_name);
        fs::create_dir_all(&class_dir)
            .with_context(|| format!("Failed to create class directory: {:?}", class_dir))?;

        // Copy images with consistent naming
        for (idx, src_path) in selected.iter().enumerate() {
            let ext = src_path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("jpg")
                .to_lowercase();
            
            let dst_name = format!("{}_{:04}.{}", class_name, idx, ext);
            let dst_path = class_dir.join(&dst_name);

            fs::copy(src_path, &dst_path)
                .with_context(|| format!("Failed to copy {:?} to {:?}", src_path, dst_path))?;
        }

        class_stats.insert(
            class_name.clone(),
            ClassStat {
                original: original_count,
                used: used_count,
            },
        );

        total_copied += used_count;
        println!(
            "  {} {}: {}/{} images",
            if used_count == original_count { "✓" } else { "↓" },
            class_name,
            used_count,
            original_count
        );
    }

    let stats = PrepareStats {
        num_classes: class_stats.len(),
        samples_per_class,
        total_samples: total_copied,
        original_total,
        original_min_class: original_min,
        original_max_class: original_max,
        imbalance_ratio,
        class_stats,
    };

    // Save config JSON
    let config_path = output_dir.join("dataset_config.json");
    let config_json = serde_json::to_string_pretty(&stats)?;
    fs::write(&config_path, config_json)?;
    println!("  Config saved to: {:?}", config_path);

    println!();
    println!("Dataset prepared successfully!");
    println!("  Total images: {}", total_copied);
    println!("  Classes: {}", stats.num_classes);
    println!("  Samples per class: {}", samples_per_class);
    println!("  New imbalance ratio: 1:1 (perfectly balanced)");

    Ok(stats)
}

/// Find the actual source directory containing class folders
fn find_source_directory(base: &Path) -> Result<PathBuf> {
    // Check if base itself contains class directories
    if has_class_directories(base) {
        return Ok(base.to_path_buf());
    }

    // Check common nested structures
    let candidates = [
        base.join("color"),                           // raw/plantvillage dataset/color/
        base.join("plantvillage dataset").join("color"),
        base.join("raw").join("plantvillage dataset").join("color"),
        base.join("organized"),
        base.join("balanced"),
    ];

    for candidate in &candidates {
        if candidate.exists() && has_class_directories(candidate) {
            return Ok(candidate.clone());
        }
    }

    // Search recursively for a directory with multiple subdirectories containing images
    for entry in WalkDir::new(base).max_depth(4).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_dir() && has_class_directories(entry.path()) {
            return Ok(entry.path().to_path_buf());
        }
    }

    anyhow::bail!(
        "Could not find class directories in {:?}. \
         Expected structure: <path>/<class_name>/*.jpg",
        base
    )
}

/// Check if a directory contains class subdirectories with images
fn has_class_directories(dir: &Path) -> bool {
    let mut class_count = 0;
    
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                // Check if this subdirectory contains images
                if let Ok(sub_entries) = fs::read_dir(entry.path()) {
                    let has_images = sub_entries.flatten().any(|e| {
                        e.path()
                            .extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| {
                                let ext = ext.to_lowercase();
                                ext == "jpg" || ext == "jpeg" || ext == "png"
                            })
                            .unwrap_or(false)
                    });
                    if has_images {
                        class_count += 1;
                    }
                }
            }
        }
    }

    class_count >= 2 // At least 2 class directories
}

/// Discover all class directories and their images
fn discover_class_images(source: &Path) -> Result<HashMap<String, Vec<PathBuf>>> {
    let mut class_images: HashMap<String, Vec<PathBuf>> = HashMap::new();

    for entry in fs::read_dir(source)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }

        let class_name = entry
            .file_name()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid class name"))?
            .to_string();

        // Skip hidden directories and config files
        if class_name.starts_with('.') || class_name.ends_with(".json") {
            continue;
        }

        let mut images = Vec::new();
        
        for img_entry in WalkDir::new(entry.path())
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = img_entry.path();
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                let ext = ext.to_lowercase();
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    images.push(path.to_path_buf());
                }
            }
        }

        if !images.is_empty() {
            class_images.insert(class_name, images);
        }
    }

    Ok(class_images)
}

/// Get class weights for weighted loss (inverse frequency)
/// 
/// Returns a vector of weights where weight[i] = total_samples / (num_classes * class_count[i])
pub fn compute_class_weights(data_dir: &Path) -> Result<Vec<f32>> {
    let class_images = discover_class_images(data_dir)?;
    
    let num_classes = class_images.len();
    let total_samples: usize = class_images.values().map(|v| v.len()).sum();
    
    // Sort class names to ensure consistent ordering
    let mut class_names: Vec<_> = class_images.keys().cloned().collect();
    class_names.sort();
    
    let weights: Vec<f32> = class_names
        .iter()
        .map(|name| {
            let count = class_images.get(name).map(|v| v.len()).unwrap_or(1);
            // Inverse frequency weighting: weight = N / (C * n_c)
            // where N = total samples, C = num classes, n_c = samples in class c
            total_samples as f32 / (num_classes as f32 * count as f32)
        })
        .collect();
    
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prepare_config_default() {
        let config = PrepareConfig::default();
        assert_eq!(config.seed, 42);
        assert!(config.samples_per_class.is_none());
    }
}
