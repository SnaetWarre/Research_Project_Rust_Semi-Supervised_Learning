//! Preprocessing tool for PlantVillage dataset.
//!
//! This tool provides various preprocessing operations:
//! - Image resizing and normalization
//! - Dataset splitting (train/val/test)
//! - Data augmentation
//! - Dataset statistics

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use plant_core::{AugmentationConfig, ImageDimensions, SplitRatios};
use plant_dataset::{AugmentationPipeline, ImageLoader, ImagePreprocessor, PreprocessConfig};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "preprocess")]
#[command(about = "Preprocessing tool for PlantVillage dataset", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Preprocess images (resize and normalize)
    Process {
        /// Input directory containing raw images
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory for processed images
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Target image size (default: 224)
        #[arg(short, long, default_value = "224")]
        size: u32,

        /// Number of parallel workers (default: num_cpus)
        #[arg(short, long)]
        workers: Option<usize>,
    },

    /// Split dataset into train/val/test sets
    Split {
        /// Data directory containing class folders
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Output directory for splits
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Training set ratio (default: 0.7)
        #[arg(long, default_value = "0.7")]
        train_ratio: f32,

        /// Validation set ratio (default: 0.15)
        #[arg(long, default_value = "0.15")]
        val_ratio: f32,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Augment images to increase dataset size
    Augment {
        /// Input directory containing images
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory for augmented images
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Number of augmented versions per image
        #[arg(short, long, default_value = "5")]
        multiplier: usize,

        /// Augmentation preset: light, medium, heavy
        #[arg(short, long, default_value = "medium")]
        preset: String,
    },

    /// Analyze dataset and generate statistics
    Analyze {
        /// Data directory to analyze
        #[arg(short, long)]
        data_dir: PathBuf,

        /// Output file for statistics (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Select specific classes from dataset
    Select {
        /// Input directory containing all classes
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory for selected classes
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Class names to select (comma-separated or provide multiple times)
        #[arg(short, long, value_delimiter = ',')]
        classes: Vec<String>,
    },
}

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Process {
            input_dir,
            output_dir,
            size,
            workers,
        } => process_images(&input_dir, &output_dir, size, workers)?,

        Commands::Split {
            data_dir,
            output_dir,
            train_ratio,
            val_ratio,
            seed,
        } => split_dataset(&data_dir, &output_dir, train_ratio, val_ratio, seed)?,

        Commands::Augment {
            input_dir,
            output_dir,
            multiplier,
            preset,
        } => augment_dataset(&input_dir, &output_dir, multiplier, &preset)?,

        Commands::Analyze {
            data_dir,
            output,
        } => analyze_dataset(&data_dir, output.as_deref())?,

        Commands::Select {
            input_dir,
            output_dir,
            classes,
        } => select_classes(&input_dir, &output_dir, &classes)?,
    }

    Ok(())
}

/// Process images: resize and normalize
fn process_images(
    input_dir: &Path,
    output_dir: &Path,
    size: u32,
    workers: Option<usize>,
) -> Result<()> {
    info!("Processing images from {:?} to {:?}", input_dir, output_dir);
    info!("Target size: {}x{}", size, size);

    // Set up preprocessing config
    let config = PreprocessConfig {
        target_size: ImageDimensions::new(size, size, 3),
        maintain_aspect_ratio: false,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
        force_rgb: true,
    };

    let preprocessor = ImagePreprocessor::new(config);

    // Set up thread pool
    if let Some(n) = workers {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
            .context("Failed to set thread pool size")?;
    }

    // Scan for class directories
    let class_dirs = fs::read_dir(input_dir)
        .context("Failed to read input directory")?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    info!("Found {} class directories", class_dirs.len());

    // Process each class
    for class_entry in class_dirs {
        let class_path = class_entry.path();
        let class_name = class_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        info!("Processing class: {}", class_name);

        let output_class_dir = output_dir.join(&class_name);
        fs::create_dir_all(&output_class_dir)
            .context("Failed to create output directory")?;

        // Load all images in this class
        let loader = ImageLoader::new(&class_path);
        let images = loader
            .scan_directory(Path::new(""))
            .context("Failed to scan directory")?;

        info!("  Found {} images", images.len());

        // Progress bar
        let pb = ProgressBar::new(images.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        // Process images in parallel
        images.par_iter().for_each(|img_path| {
            let file_name = img_path.file_name().unwrap();
            let output_path = output_class_dir.join(file_name);

            // Load and preprocess
            match image::open(img_path) {
                Ok(img) => {
                    // Resize and save (we don't normalize for storage, just resize)
                    let rgb = img.to_rgb8();
                    let resized = image::imageops::resize(
                        &rgb,
                        size,
                        size,
                        image::imageops::FilterType::Lanczos3,
                    );

                    if let Err(e) = resized.save(&output_path) {
                        warn!("Failed to save {}: {}", output_path.display(), e);
                    }
                }
                Err(e) => {
                    warn!("Failed to load {}: {}", img_path.display(), e);
                }
            }

            pb.inc(1);
        });

        pb.finish_with_message("Done");
    }

    info!("✓ Processing complete!");
    Ok(())
}

/// Split dataset into train/val/test sets
fn split_dataset(
    data_dir: &Path,
    output_dir: &Path,
    train_ratio: f32,
    val_ratio: f32,
    seed: u64,
) -> Result<()> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    info!("Splitting dataset from {:?}", data_dir);
    info!(
        "Ratios - Train: {:.0}%, Val: {:.0}%, Test: {:.0}%",
        train_ratio * 100.0,
        val_ratio * 100.0,
        (1.0 - train_ratio - val_ratio) * 100.0
    );

    // Validate ratios
    let test_ratio = 1.0 - train_ratio - val_ratio;
    if test_ratio < 0.0 || test_ratio > 1.0 {
        anyhow::bail!("Invalid split ratios: must sum to <= 1.0");
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Create output directories
    let train_dir = output_dir.join("train");
    let val_dir = output_dir.join("val");
    let test_dir = output_dir.join("test");

    fs::create_dir_all(&train_dir)?;
    fs::create_dir_all(&val_dir)?;
    fs::create_dir_all(&test_dir)?;

    // Process each class
    let class_dirs = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    info!("Found {} classes", class_dirs.len());

    let mut total_train = 0;
    let mut total_val = 0;
    let mut total_test = 0;

    for class_entry in class_dirs {
        let class_path = class_entry.path();
        let class_name = class_path.file_name().unwrap().to_string_lossy().to_string();

        // Get all images
        let loader = ImageLoader::new(&class_path);
        let mut images = loader.scan_directory(Path::new(""))?;

        // Shuffle
        images.shuffle(&mut rng);

        let total = images.len();
        let train_count = (total as f32 * train_ratio) as usize;
        let val_count = (total as f32 * val_ratio) as usize;

        // Create class subdirectories
        fs::create_dir_all(train_dir.join(&class_name))?;
        fs::create_dir_all(val_dir.join(&class_name))?;
        fs::create_dir_all(test_dir.join(&class_name))?;

        // Copy files to appropriate splits
        for (idx, img_path) in images.iter().enumerate() {
            let file_name = img_path.file_name().unwrap();
            let dest = if idx < train_count {
                total_train += 1;
                train_dir.join(&class_name).join(file_name)
            } else if idx < train_count + val_count {
                total_val += 1;
                val_dir.join(&class_name).join(file_name)
            } else {
                total_test += 1;
                test_dir.join(&class_name).join(file_name)
            };

            fs::copy(img_path, dest)?;
        }

        info!(
            "  {}: {} total (train: {}, val: {}, test: {})",
            class_name,
            total,
            train_count,
            val_count,
            total - train_count - val_count
        );
    }

    info!("\n✓ Split complete!");
    info!("  Total - Train: {}, Val: {}, Test: {}", total_train, total_val, total_test);

    // Save split metadata
    let metadata = serde_json::json!({
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_count": total_train,
        "val_count": total_val,
        "test_count": total_test,
    });

    let metadata_path = output_dir.join("split_info.json");
    fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;
    info!("  Metadata saved to {:?}", metadata_path);

    Ok(())
}

/// Augment dataset
fn augment_dataset(
    input_dir: &Path,
    output_dir: &Path,
    multiplier: usize,
    preset: &str,
) -> Result<()> {
    info!("Augmenting dataset with preset: {}", preset);
    info!("Multiplier: {}x", multiplier);

    // Select augmentation config
    let aug_config = match preset.to_lowercase().as_str() {
        "light" => plant_core::AugmentationConfig {
            rotation_range: 10.0,
            horizontal_flip: true,
            vertical_flip: false,
            brightness_range: (0.9, 1.1),
            contrast_range: (0.9, 1.1),
            saturation_range: (0.95, 1.05),
            random_crop: true,
            zoom_range: (0.95, 1.05),
        },
        "medium" => plant_core::AugmentationConfig::default(),
        "heavy" => plant_core::AugmentationConfig {
            rotation_range: 30.0,
            horizontal_flip: true,
            vertical_flip: true,
            brightness_range: (0.7, 1.3),
            contrast_range: (0.7, 1.3),
            saturation_range: (0.7, 1.3),
            random_crop: true,
            zoom_range: (0.85, 1.15),
        },
        _ => anyhow::bail!("Unknown preset: {}. Use 'light', 'medium', or 'heavy'", preset),
    };

    let mut pipeline = AugmentationPipeline::new(aug_config);

    // Process each class
    let class_dirs = fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    for class_entry in class_dirs {
        let class_path = class_entry.path();
        let class_name = class_path.file_name().unwrap().to_string_lossy().to_string();

        info!("Augmenting class: {}", class_name);

        let output_class_dir = output_dir.join(&class_name);
        fs::create_dir_all(&output_class_dir)?;

        let loader = ImageLoader::new(&class_path);
        let images = loader.scan_directory(Path::new(""))?;

        let pb = ProgressBar::new(images.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        for img_path in &images {
            let img = image::open(img_path)?;
            let stem = img_path.file_stem().unwrap().to_string_lossy();

            // Save original
            let original_name = format!("{}_original.jpg", stem);
            img.save(output_class_dir.join(original_name))?;

            // Generate augmented versions
            for i in 0..multiplier {
                let augmented = pipeline.augment(&img)?;
                let aug_name = format!("{}_aug{}.jpg", stem, i);
                augmented.save(output_class_dir.join(aug_name))?;
            }

            pb.inc(1);
        }

        pb.finish_with_message("Done");
    }

    info!("✓ Augmentation complete!");
    Ok(())
}

/// Analyze dataset statistics
fn analyze_dataset(data_dir: &Path, output: Option<&Path>) -> Result<()> {
    info!("Analyzing dataset at {:?}", data_dir);

    let mut stats = HashMap::new();
    let mut total_images = 0;
    let mut total_size: u64 = 0;

    let class_dirs = fs::read_dir(data_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect::<Vec<_>>();

    println!("\n📊 Dataset Statistics\n");
    println!("{:<40} {:>10} {:>15}", "Class", "Images", "Avg Size");
    println!("{}", "=".repeat(70));

    for class_entry in class_dirs {
        let class_path = class_entry.path();
        let class_name = class_path.file_name().unwrap().to_string_lossy().to_string();

        let loader = ImageLoader::new(&class_path);
        let images = loader.scan_directory(Path::new(""))?;

        let count = images.len();
        total_images += count;

        // Calculate average file size
        let class_size: u64 = images
            .iter()
            .filter_map(|p| fs::metadata(p).ok())
            .map(|m| m.len())
            .sum();
        total_size += class_size;

        let avg_size = if count > 0 {
            class_size / count as u64
        } else {
            0
        };

        println!(
            "{:<40} {:>10} {:>12} KB",
            class_name,
            count,
            avg_size / 1024
        );

        stats.insert(
            class_name,
            serde_json::json!({
                "count": count,
                "total_size_bytes": class_size,
                "avg_size_bytes": avg_size,
            }),
        );
    }

    println!("{}", "=".repeat(70));
    println!(
        "{:<40} {:>10} {:>12} MB",
        "TOTAL",
        total_images,
        total_size / (1024 * 1024)
    );

    // Summary statistics
    let num_classes = stats.len();
    let avg_per_class = if num_classes > 0 {
        total_images / num_classes
    } else {
        0
    };

    println!("\n📈 Summary:");
    println!("  Total classes:       {}", num_classes);
    println!("  Total images:        {}", total_images);
    println!("  Avg images/class:    {}", avg_per_class);
    println!("  Total size:          {:.2} MB", total_size as f64 / (1024.0 * 1024.0));

    // Save to file if requested
    if let Some(output_path) = output {
        let full_stats = serde_json::json!({
            "total_classes": num_classes,
            "total_images": total_images,
            "total_size_bytes": total_size,
            "avg_images_per_class": avg_per_class,
            "classes": stats,
        });

        fs::write(output_path, serde_json::to_string_pretty(&full_stats)?)?;
        info!("\n✓ Statistics saved to {:?}", output_path);
    }

    Ok(())
}

/// Select specific classes from dataset
fn select_classes(input_dir: &Path, output_dir: &Path, classes: &[String]) -> Result<()> {
    info!("Selecting {} classes", classes.len());

    fs::create_dir_all(output_dir)?;

    for class_name in classes {
        let class_path = input_dir.join(class_name);

        if !class_path.exists() {
            warn!("Class '{}' not found, skipping", class_name);
            continue;
        }

        info!("Copying class: {}", class_name);

        let output_class_dir = output_dir.join(class_name);
        fs::create_dir_all(&output_class_dir)?;

        // Copy all images
        let loader = ImageLoader::new(&class_path);
        let images = loader.scan_directory(Path::new(""))?;

        for img_path in &images {
            let file_name = img_path.file_name().unwrap();
            let dest = output_class_dir.join(file_name);
            fs::copy(img_path, dest)?;
        }

        info!("  Copied {} images", images.len());
    }

    info!("✓ Class selection complete!");
    Ok(())
}
