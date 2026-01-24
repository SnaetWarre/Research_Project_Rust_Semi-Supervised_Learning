//! Setup script for mobile demo
//!
//! This binary creates the farmer_demo folder with images that are
//! GUARANTEED to be from the stream_pool (never in labeled training data).
//!
//! Usage: cargo run --release --bin setup_mobile_demo -- --data-dir data/plantvillage --output-dir data/farmer_demo

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use plantvillage_ssl::dataset::split::{DatasetSplits, SplitConfig};
use plantvillage_ssl::PlantVillageDataset;

#[derive(Parser, Debug)]
#[command(name = "setup_mobile_demo")]
#[command(about = "Setup farmer_demo folder with images from stream_pool")]
struct Args {
    /// Path to PlantVillage dataset
    #[arg(short, long, default_value = "data/plantvillage")]
    data_dir: String,

    /// Output directory for farmer demo images
    #[arg(short, long, default_value = "data/farmer_demo")]
    output_dir: String,

    /// Number of images to copy to farmer_demo (0 = all stream images)
    #[arg(short, long, default_value = "500")]
    num_images: usize,

    /// Random seed for reproducibility
    #[arg(short, long, default_value = "42")]
    seed: u64,

    /// Labeled ratio (must match training config!)
    #[arg(short, long, default_value = "0.2")]
    labeled_ratio: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    println!("=== Mobile Demo Setup ===");
    println!("Data dir: {}", args.data_dir);
    println!("Output dir: {}", args.output_dir);
    println!("Labeled ratio: {:.1}%", args.labeled_ratio * 100.0);
    println!();

    // Load dataset
    println!("Loading dataset...");
    let dataset = PlantVillageDataset::new(&args.data_dir)?;
    let stats = dataset.get_stats();
    println!(
        "Found {} images across {} classes",
        stats.total_samples, stats.num_classes
    );

    // Create splits using SAME config as demo
    let stream_fraction = 1.0 - args.labeled_ratio;
    let split_config = SplitConfig {
        test_fraction: 0.10,
        validation_fraction: 0.10,
        labeled_fraction: args.labeled_ratio,
        stream_fraction,
        seed: args.seed,
        stratified: true,
    };

    let all_images: Vec<(PathBuf, usize, String)> = dataset
        .samples
        .iter()
        .map(|s| (s.path.clone(), s.label, s.class_name.clone()))
        .collect();

    println!("Creating splits...");
    let splits = DatasetSplits::from_images(all_images, split_config)?;

    println!("Split statistics:");
    println!("  Test set: {} images", splits.test_set.len());
    println!("  Validation set: {} images", splits.validation_set.len());
    println!("  Labeled pool: {} images", splits.labeled_pool.len());
    println!("  Stream pool: {} images", splits.stream_pool.len());
    println!();

    // Collect labeled pool paths for verification
    let labeled_paths: HashSet<PathBuf> = splits
        .labeled_pool
        .iter()
        .map(|img| img.image_path.clone())
        .collect();

    // Select images from stream pool
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed + 1000);
    let mut stream_images = splits.stream_pool.clone();
    stream_images.shuffle(&mut rng);

    let num_to_copy = if args.num_images == 0 {
        stream_images.len()
    } else {
        args.num_images.min(stream_images.len())
    };

    println!(
        "Selecting {} images from stream pool for farmer_demo...",
        num_to_copy
    );

    // Clear and create output directory
    let output_path = PathBuf::from(&args.output_dir);
    if output_path.exists() {
        println!("Clearing existing farmer_demo folder...");
        fs::remove_dir_all(&output_path)?;
    }
    fs::create_dir_all(&output_path)?;

    // Copy images organized by class
    let mut copied = 0;
    let mut class_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    for img in stream_images.iter().take(num_to_copy) {
        // Verify this image is NOT in labeled pool (should never happen, but double-check)
        if labeled_paths.contains(&img.image_path) {
            eprintln!(
                "WARNING: Image {:?} is in both stream and labeled pool! Skipping.",
                img.image_path
            );
            continue;
        }

        // Create class subdirectory
        let class_dir = output_path.join(&img.hidden_class_name);
        fs::create_dir_all(&class_dir)?;

        // Copy image
        let filename = img
            .image_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;
        let dest_path = class_dir.join(filename);

        fs::copy(&img.image_path, &dest_path)?;
        copied += 1;
        *class_counts
            .entry(img.hidden_class_name.clone())
            .or_insert(0) += 1;

        if copied % 100 == 0 {
            println!("  Copied {} images...", copied);
        }
    }

    println!();
    println!("=== Setup Complete ===");
    println!("Copied {} images to {}", copied, args.output_dir);
    println!();
    println!("Class distribution:");
    let mut classes: Vec<_> = class_counts.iter().collect();
    classes.sort_by_key(|(name, _)| name.as_str());
    for (name, count) in classes.iter().take(10) {
        println!("  {}: {}", name, count);
    }
    if classes.len() > 10 {
        println!("  ... and {} more classes", classes.len() - 10);
    }

    println!();
    println!("IMPORTANT: These images are from the STREAM POOL and are");
    println!("guaranteed to NOT be in the labeled training set!");
    println!();
    println!("Data split breakdown:");
    println!(
        "  - Test (10%): {} images - held out for evaluation",
        splits.test_set.len()
    );
    println!(
        "  - Validation (10%): {} images - for early stopping",
        splits.validation_set.len()
    );
    println!(
        "  - Labeled ({}%): {} images - initial training data",
        (args.labeled_ratio * 100.0) as usize,
        splits.labeled_pool.len()
    );
    println!(
        "  - Stream ({}%): {} images - for SSL pseudo-labeling",
        ((1.0 - args.labeled_ratio) * 0.8 * 100.0) as usize,
        splits.stream_pool.len()
    );
    println!();
    println!(
        "The farmer_demo folder contains {} images from the stream pool.",
        copied
    );

    Ok(())
}
