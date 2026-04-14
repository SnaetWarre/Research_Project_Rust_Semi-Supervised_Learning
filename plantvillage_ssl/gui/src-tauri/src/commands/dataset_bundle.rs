//! Dataset Bundling for Mobile Deployment
//!
//! Utilities to prepare and bundle a subset of the dataset for mobile deployment.
//! Since we can't fit 87K images on a mobile device, this creates a smaller
//! representative subset for demonstration purposes.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// Configuration for dataset bundling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleConfig {
    /// Number of images per class to include
    pub images_per_class: usize,
    /// Source dataset directory
    pub source_dir: PathBuf,
    /// Output bundle directory
    pub output_dir: PathBuf,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for BundleConfig {
    fn default() -> Self {
        Self {
            images_per_class: 50, // 50 images × 38 classes = 1,900 images (~200MB)
            source_dir: PathBuf::from("data/plantvillage"),
            output_dir: PathBuf::from("mobile_dataset"),
            seed: 42,
        }
    }
}

/// Bundle metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMetadata {
    pub total_images: usize,
    pub num_classes: usize,
    pub images_per_class: HashMap<String, usize>,
    pub created_at: String,
}

/// Create a mobile dataset bundle
///
/// This samples a subset of images from each class and copies them to a new directory
/// suitable for embedding in a mobile app.
pub fn create_mobile_bundle(config: BundleConfig) -> Result<BundleMetadata> {
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    
    println!("Creating mobile dataset bundle...");
    println!("  Source: {:?}", config.source_dir);
    println!("  Output: {:?}", config.output_dir);
    println!("  Images per class: {}", config.images_per_class);

    // Create output directory
    fs::create_dir_all(&config.output_dir)
        .context("Failed to create output directory")?;

    let mut metadata = BundleMetadata {
        total_images: 0,
        num_classes: 0,
        images_per_class: HashMap::new(),
        created_at: chrono::Local::now().to_rfc3339(),
    };

    // Find all class directories
    let entries = fs::read_dir(&config.source_dir)
        .context("Failed to read source directory")?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_dir() {
            continue;
        }

        let class_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("Invalid class directory name")?
            .to_string();

        // Skip special directories
        if class_name.starts_with('.') {
            continue;
        }

        // Collect all images in this class
        let mut class_images: Vec<PathBuf> = Vec::new();
        for img_entry in fs::read_dir(&path)? {
            let img_entry = img_entry?;
            let img_path = img_entry.path();
            
            if img_path.is_file() {
                if let Some(ext) = img_path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if ext_str == "jpg" || ext_str == "jpeg" || ext_str == "png" {
                        class_images.push(img_path);
                    }
                }
            }
        }

        if class_images.is_empty() {
            continue;
        }

        // Randomly sample images
        class_images.shuffle(&mut rng);
        let sample_size = config.images_per_class.min(class_images.len());
        let sampled_images = &class_images[..sample_size];

        // Create class directory in output
        let output_class_dir = config.output_dir.join(&class_name);
        fs::create_dir_all(&output_class_dir)?;

        // Copy sampled images
        for img_path in sampled_images {
            if let Some(img_name) = img_path.file_name() {
                let output_path = output_class_dir.join(img_name);
                fs::copy(img_path, &output_path)
                    .with_context(|| format!("Failed to copy {:?}", img_path))?;
            }
        }

        metadata.images_per_class.insert(class_name.clone(), sample_size);
        metadata.total_images += sample_size;
        metadata.num_classes += 1;

        println!("  ✓ {} ({} images)", class_name, sample_size);
    }

    // Save metadata
    let metadata_path = config.output_dir.join("bundle_metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;

    println!();
    println!("Bundle created successfully!");
    println!("  Total images: {}", metadata.total_images);
    println!("  Classes: {}", metadata.num_classes);
    println!("  Metadata: {:?}", metadata_path);

    Ok(metadata)
}

/// Tauri command to create mobile dataset bundle
#[tauri::command]
pub async fn create_dataset_bundle(
    images_per_class: usize,
    source_dir: String,
    output_dir: String,
) -> Result<BundleMetadata, String> {
    let config = BundleConfig {
        images_per_class,
        source_dir: PathBuf::from(source_dir),
        output_dir: PathBuf::from(output_dir),
        seed: 42,
    };

    tokio::task::spawn_blocking(move || create_mobile_bundle(config))
        .await
        .map_err(|e| format!("Task failed: {}", e))?
        .map_err(|e| format!("Bundle creation failed: {}", e))
}

/// Tauri command to load bundle metadata
#[tauri::command]
pub async fn load_bundle_metadata(bundle_dir: String) -> Result<BundleMetadata, String> {
    let metadata_path = PathBuf::from(bundle_dir).join("bundle_metadata.json");
    
    let content = fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read metadata: {}", e))?;
    
    serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse metadata: {}", e))
}
