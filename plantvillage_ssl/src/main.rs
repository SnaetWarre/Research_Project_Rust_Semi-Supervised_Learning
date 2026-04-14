//! PlantVillage Semi-Supervised Learning CLI
//!
//! This is the main entry point for the PlantVillage plant disease classification
//! system using semi-supervised learning with the Burn framework.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing::info;

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{activation::softmax, Int, Tensor};
use plantvillage_ssl::backend::TrainingBackend;
use plantvillage_ssl::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageBurnDataset};
use plantvillage_ssl::dataset::split::{DatasetSplits, SplitConfig};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::utils::logging::{init_logging, LogConfig};
use plantvillage_ssl::PlantVillageDataset;

/// PlantVillage Semi-Supervised Plant Disease Classification
///
/// A Rust-based semi-supervised learning system for plant disease classification
/// using the Burn framework with CUDA GPU acceleration.
#[derive(Parser, Debug)]
#[command(name = "plantvillage_ssl")]
#[command(author = "Warre Snaet")]
#[command(version = "0.1.0")]
#[command(about = "Semi-supervised plant disease classification with Burn", long_about = None)]
struct Cli {
    /// Enable verbose logging
    #[arg(short, long, default_value = "false")]
    verbose: bool,

    /// Subcommand to execute
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download and prepare the PlantVillage dataset
    Download {
        /// Output directory for the dataset
        #[arg(short, long, default_value = "data/plantvillage")]
        output_dir: String,
    },

    /// Train the model with semi-supervised learning
    Train {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage")]
        data_dir: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "50")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "64")]
        batch_size: usize,

        /// Learning rate
        #[arg(short, long, default_value = "0.0001")]
        learning_rate: f64,

        /// Percentage of labeled data (0.0-1.0). Use 0.2 for SSL workflows!
        #[arg(long, default_value = "0.2")]
        labeled_ratio: f64,

        /// Confidence threshold for pseudo-labeling (0.0-1.0)
        #[arg(long, default_value = "0.9")]
        confidence_threshold: f64,

        /// Output directory for model checkpoints
        #[arg(short, long, default_value = "output/models")]
        output_dir: String,

        /// Use CUDA backend
        #[arg(long, default_value = "false")]
        cuda: bool,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Quick test mode - use only 500 samples for fast verification
        #[arg(long, default_value = "false")]
        quick: bool,

        /// Enable data augmentation during training (improves generalization)
        #[arg(long, default_value = "false")]
        augmentation: bool,

        /// Disable early stopping at target validation accuracy
        #[arg(long, default_value = "false")]
        no_early_stop: bool,

        /// Target validation accuracy for early stopping (0.0-1.0)
        #[arg(long, default_value = "0.88")]
        target_accuracy: f64,

        /// Patience epochs for early stopping (stop after N epochs at target)
        #[arg(long, default_value = "3")]
        early_stop_patience: usize,
    },

    /// Run inference on a single image or directory
    Infer {
        /// Path to input image or directory
        #[arg(short, long)]
        input: String,

        /// Path to trained model
        #[arg(short, long)]
        model: String,

        /// Use CUDA backend
        #[arg(long, default_value = "false")]
        cuda: bool,
    },

    /// Benchmark inference performance
    Benchmark {
        /// Path to trained model (optional - uses random weights if not specified)
        #[arg(short, long)]
        model: Option<String>,

        /// Number of inference iterations for timing
        #[arg(short, long, default_value = "100")]
        iterations: usize,

        /// Number of warmup iterations
        #[arg(long, default_value = "10")]
        warmup: usize,

        /// Batch size for inference
        #[arg(short, long, default_value = "1")]
        batch_size: usize,

        /// Image size (square)
        #[arg(long, default_value = "128")]
        image_size: usize,

        /// Output JSON file for benchmark results
        #[arg(short, long)]
        output: Option<String>,

        /// Verbose output
        #[arg(long, default_value = "false")]
        verbose: bool,
    },

    /// Simulate streaming data for semi-supervised learning demo
    Simulate {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage")]
        data_dir: String,

        /// Path to trained model (for starting point)
        #[arg(short, long)]
        model: String,

        /// Number of simulated days (use 0 for unlimited - process all available data)
        #[arg(long, default_value = "0")]
        days: usize,

        /// Images per day
        #[arg(long, default_value = "100")]
        images_per_day: usize,

        /// Confidence threshold for pseudo-labeling
        #[arg(long, default_value = "0.9")]
        confidence_threshold: f64,

        /// Retrain after this many pseudo-labeled images
        #[arg(long, default_value = "200")]
        retrain_threshold: usize,

        /// Labeled ratio for data split (0.0-1.0). Should match training!
        #[arg(long, default_value = "0.2")]
        labeled_ratio: f64,

        /// Output directory for logs and metrics
        #[arg(short, long, default_value = "output/simulation")]
        output_dir: String,

        /// Use CUDA backend
        #[arg(long, default_value = "false")]
        cuda: bool,
    },

    /// Export metrics and results
    Export {
        /// Path to metrics directory
        #[arg(short, long, default_value = "output")]
        input_dir: String,

        /// Output format (csv, json)
        #[arg(short, long, default_value = "csv")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: String,
    },

    /// Evaluate model accuracy and macro F1 on test split (fallback: validation split)
    Eval {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage")]
        data_dir: String,

        /// Model checkpoint path (optional). If omitted, latest .mpk from output/simulation is used.
        #[arg(short, long)]
        model: Option<String>,

        /// Labeled ratio used in split simulation
        #[arg(long, default_value = "0.2")]
        labeled_ratio: f64,

        /// Batch size for evaluation
        #[arg(short, long, default_value = "64")]
        batch_size: usize,

        /// Output directory used to auto-discover latest simulation model
        #[arg(long, default_value = "output/simulation")]
        simulation_output_dir: String,
    },

    /// Show dataset statistics
    Stats {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage")]
        data_dir: String,

        /// Show split simulation info
        #[arg(long, default_value = "false")]
        show_splits: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_config = if cli.verbose {
        LogConfig::verbose()
    } else {
        LogConfig::default()
    };

    let _ = init_logging(&log_config);

    print_banner();

    match cli.command {
        Commands::Download { output_dir } => {
            cmd_download(&output_dir)?;
        }

        Commands::Train {
            data_dir,
            epochs,
            batch_size,
            learning_rate,
            labeled_ratio,
            confidence_threshold,
            output_dir,
            cuda,
            seed,
            quick,
            augmentation,
            no_early_stop,
            target_accuracy,
            early_stop_patience,
        } => {
            // Always use CUDA - this project targets GPU
            let _ = cuda; // Ignore flag, always GPU

            let max_samples = if quick {
                println!(
                    "{}",
                    "🚀 Quick test mode: using only 500 samples".yellow().bold()
                );
                Some(500usize)
            } else {
                None
            };

            // Configure early stopping
            let early_stopping = if no_early_stop {
                None
            } else {
                Some(
                    plantvillage_ssl::training::supervised::EarlyStoppingConfig {
                        target_accuracy,
                        patience: early_stop_patience,
                        enabled: true,
                    },
                )
            };

            plantvillage_ssl::training::supervised::run_training::<TrainingBackend>(
                &data_dir,
                epochs,
                batch_size,
                learning_rate,
                labeled_ratio,
                confidence_threshold,
                &output_dir,
                seed,
                max_samples,
                augmentation,
                early_stopping,
            )?;
        }

        Commands::Infer {
            input,
            model,
            cuda: _cuda,
        } => {
            cmd_infer(&input, &model, true)?;
        }

        Commands::Benchmark {
            model,
            iterations,
            warmup,
            batch_size,
            image_size,
            output,
            verbose,
        } => {
            cmd_benchmark(
                model.as_deref(),
                iterations,
                warmup,
                batch_size,
                image_size,
                output.as_deref(),
                verbose,
            )?;
        }

        Commands::Simulate {
            data_dir,
            model,
            days,
            images_per_day,
            confidence_threshold,
            retrain_threshold,
            labeled_ratio,
            output_dir,
            cuda: _cuda,
        } => {
            cmd_simulate(
                &data_dir,
                &model,
                days,
                images_per_day,
                confidence_threshold,
                retrain_threshold,
                labeled_ratio,
                &output_dir,
                true,
            )?;
        }

        Commands::Export {
            input_dir,
            format: _format,
            output,
        } => {
            cmd_export(&input_dir, &_format, &output)?;
        }

        Commands::Eval {
            data_dir,
            model,
            labeled_ratio,
            batch_size,
            simulation_output_dir,
        } => {
            cmd_eval(
                &data_dir,
                model.as_deref(),
                labeled_ratio,
                batch_size,
                &simulation_output_dir,
            )?;
        }

        Commands::Stats {
            data_dir,
            show_splits,
        } => {
            cmd_stats(&data_dir, show_splits)?;
        }
    }

    Ok(())
}

fn print_banner() {
    println!(
        "{}",
        r#"
 ╔══════════════════════════════════════════════════════════════════╗
 ║   🌱 PlantVillage Semi-Supervised Learning                           ║
 ║   Plant Disease Classification with Burn + Rust                      ║
 ║   Designed for GPU Training with CUDA                               ║
 ╚════════════════════════════════════════════════════════════════╝
  "#
        .green()
    );
}

fn cmd_download(output_dir: &str) -> Result<()> {
    info!("Downloading New Plant Diseases Dataset to: {}", output_dir);

    println!(
        "{} Dataset download requires Kaggle CLI. Please download manually or use the provided script.",
        "Note:".yellow()
    );
    println!();
    println!("{}", "Option 1: Use the download script".cyan());
    println!("  ./scripts/download_dataset.sh");
    println!();
    println!("{}", "Option 2: Download manually from Kaggle".cyan());
    println!("  Dataset: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset");
    println!("  Extract train/ and valid/ folders to: {}", output_dir);
    println!();
    println!("{}", "Expected structure:".yellow());
    println!("  {}/", output_dir);
    println!("  ├── train/");
    println!("  │   ├── Apple___Apple_scab/");
    println!("  │   └── ...");
    println!("  └── valid/");
    println!("      ├── Apple___Apple_scab/");
    println!("      └── ...");
    println!();
    println!("{}", "After download, run:".green());
    println!("  plantvillage_ssl train --epochs 50 --cuda");

    Ok(())
}

fn cmd_stats(data_dir: &str, show_splits: bool) -> Result<()> {
    info!("Computing dataset statistics for: {}", data_dir);

    if !Path::new(data_dir).exists() {
        println!(
            "{} Dataset directory not found: {}",
            "Error:".red(),
            data_dir
        );
        println!();
        println!("Please download the dataset first:");
        println!("  plantvillage_ssl download --output-dir {}", data_dir);
        return Ok(());
    }

    // Load the dataset and show statistics
    match plantvillage_ssl::PlantVillageDataset::new(data_dir) {
        Ok(dataset) => {
            let stats = dataset.get_stats();

            println!("{}", "Dataset Statistics:".cyan().bold());
            println!("  📊 Total samples: {}", stats.total_samples);
            println!("  🏷️  Number of classes: {}", stats.num_classes);
            println!();

            if show_splits {
                println!("{}", "Simulated Split Configuration:".yellow().bold());
                let total = stats.total_samples;
                let test_size = (total as f64 * 0.10) as usize;
                let val_size = (total as f64 * 0.10) as usize;
                let remaining = total - test_size - val_size;
                let labeled_size = (remaining as f64 * 0.20) as usize;
                let stream_size = remaining - labeled_size;

                println!(
                    "  🧪 Test set:        {} ({:.1}%)",
                    test_size,
                    100.0 * test_size as f64 / total as f64
                );
                println!(
                    "  ✅ Validation set:  {} ({:.1}%)",
                    val_size,
                    100.0 * val_size as f64 / total as f64
                );
                println!(
                    "  🏷️  Labeled pool:    {} ({:.1}%)",
                    labeled_size,
                    100.0 * labeled_size as f64 / total as f64
                );
                println!(
                    "  📷 Stream pool:     {} ({:.1}%)",
                    stream_size,
                    100.0 * stream_size as f64 / total as f64
                );
                println!();
            }

            println!("{}", "Class Distribution:".cyan().bold());
            for (idx, count) in stats.class_counts.iter().enumerate() {
                let class_name = stats
                    .class_names
                    .get(&idx)
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown");
                let pct = 100.0 * *count as f64 / stats.total_samples as f64;
                println!("  {:40} {:>5} ({:>5.1}%)", class_name, count, pct);
            }
        }
        Err(e) => {
            println!("{} Failed to load dataset: {}", "Error:".red(), e);
        }
    }

    Ok(())
}

fn cmd_infer(input: &str, model: &str, _cuda: bool) -> Result<()> {
    use burn::module::Module;
    use burn::record::CompactRecorder;
    use burn::tensor::activation::softmax;
    use burn::tensor::Tensor;
    use image::imageops::FilterType;
    use plantvillage_ssl::backend::{backend_name, default_device, DefaultBackend};
    use plantvillage_ssl::dataset::CLASS_NAMES;
    use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

    info!("Running inference");
    info!("  Input: {}", input);
    info!("  Model: {}", model);

    println!("{}", "Inference Configuration:".cyan().bold());
    println!("  📷 Input:  {}", input);
    println!("  🧠 Model:  {}", model);
    println!("  🖥️  Backend: {}", backend_name());
    println!();

    if !Path::new(input).exists() {
        println!("{} Input path not found: {}", "Error:".red(), input);
        return Ok(());
    }

    if !Path::new(model).exists() {
        println!("{} Model path not found: {}", "Error:".red(), model);
        return Ok(());
    }

    // Load model
    println!("{}", "Loading model...".cyan());
    let device = default_device();
    let config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let model_instance: PlantClassifier<DefaultBackend> = PlantClassifier::new(&config, &device);
    let recorder = CompactRecorder::new();
    let model_instance = model_instance
        .load_file(model, &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

    // Process input (single file or directory)
    let input_path = Path::new(input);
    let files: Vec<_> = if input_path.is_dir() {
        std::fs::read_dir(input_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| ["jpg", "jpeg", "png"].contains(&e.to_lowercase().as_str()))
                    .unwrap_or(false)
            })
            .take(10) // Limit to 10 images
            .collect()
    } else {
        vec![input_path.to_path_buf()]
    };

    println!("{}", "Running inference...".cyan());
    println!();

    for file_path in &files {
        // Load and preprocess image
        let img = image::open(file_path)?;
        let img = img.resize_exact(128, 128, FilterType::Lanczos3);
        let rgb = img.to_rgb8();

        // Normalize to tensor (CHW format)
        let mut data = vec![0.0f32; 3 * 128 * 128];
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        for (i, pixel) in rgb.pixels().enumerate() {
            data[i] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
            data[128 * 128 + i] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
            data[2 * 128 * 128 + i] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
        }

        // Create tensor [1, 3, 128, 128]
        let tensor: Tensor<DefaultBackend, 1> = Tensor::from_floats(&data[..], &device);
        let tensor: Tensor<DefaultBackend, 4> = tensor.reshape([1, 3, 128, 128]);

        // Run inference
        let start = std::time::Instant::now();
        let output = model_instance.forward(tensor);
        let probs = softmax(output, 1);
        let inference_time = start.elapsed();

        // Get predictions
        let probs_vec: Vec<f32> = probs.into_data().to_vec().unwrap();

        // Find top-5
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Get actual class from filename
        let actual_class = file_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown");

        let predicted_class = CLASS_NAMES.get(indexed[0].0).unwrap_or(&"Unknown");
        let is_correct = actual_class == *predicted_class;

        println!(
            "📷 {}",
            file_path.file_name().unwrap_or_default().to_string_lossy()
        );
        println!("  Actual:    {}", actual_class.yellow());
        println!(
            "  Predicted: {} {}",
            predicted_class,
            if is_correct {
                "✅".green()
            } else {
                "❌".red()
            }
        );
        println!("  Confidence: {:.1}%", indexed[0].1 * 100.0);
        println!("  Time: {:.2}ms", inference_time.as_secs_f64() * 1000.0);
        println!("  Top-5:");
        for (i, (idx, prob)) in indexed.iter().take(5).enumerate() {
            let name = CLASS_NAMES.get(*idx).unwrap_or(&"Unknown");
            println!("    {}. {} ({:.1}%)", i + 1, name, prob * 100.0);
        }
        println!();
    }

    Ok(())
}

fn cmd_benchmark(
    model: Option<&str>,
    iterations: usize,
    warmup: usize,
    batch_size: usize,
    image_size: usize,
    output: Option<&str>,
    verbose: bool,
) -> Result<()> {
    use plantvillage_ssl::backend::{default_device, DefaultBackend};
    use plantvillage_ssl::inference::{run_benchmark, BenchmarkConfig};
    use std::path::PathBuf;

    info!("Running benchmark");
    if let Some(m) = model {
        info!("  Model: {}", m);
    }
    info!("  Iterations: {}", iterations);
    info!("  Warmup: {}", warmup);
    info!("  Batch size: {}", batch_size);
    info!("  Image size: {}", image_size);

    let config = BenchmarkConfig {
        warmup_iterations: warmup,
        iterations,
        batch_size,
        measure_memory: true,
        verbose,
        output_path: output.map(PathBuf::from),
    };

    let device = default_device();
    let model_path = model.map(std::path::Path::new);

    let _result = run_benchmark::<DefaultBackend>(config, model_path, image_size, &device)?;

    // Print JSON output for easy parsing
    if output.is_some() {
        println!();
        println!("{}", "Benchmark complete!".green().bold());
    }

    Ok(())
}

fn cmd_simulate(
    data_dir: &str,
    model: &str,
    days: usize,
    images_per_day: usize,
    confidence_threshold: f64,
    retrain_threshold: usize,
    labeled_ratio: f64,
    output_dir: &str,
    _cuda: bool,
) -> Result<()> {
    use plantvillage_ssl::backend::{backend_name, TrainingBackend};
    use plantvillage_ssl::training::{run_simulation, SimulationConfig};

    info!("Starting stream simulation");
    info!("  Days: {} (0 = unlimited)", days);
    info!("  Images per day: {}", images_per_day);
    info!("  Confidence threshold: {}", confidence_threshold);
    info!("  Retrain threshold: {} images", retrain_threshold);
    info!("  Labeled ratio: {:.0}%", labeled_ratio * 100.0);

    println!("{}", "Simulation Configuration:".cyan().bold());
    println!("  📁 Data directory:    {}", data_dir);
    println!("  🧠 Initial model:     {}", model);
    println!("  📅 Simulated days:   {} (0 = unlimited)", days);
    println!("  📷 Images per day:   {}", images_per_day);
    println!("  🎯 Confidence threshold: {}", confidence_threshold);
    println!("  🔄 Retrain threshold:  {} images", retrain_threshold);
    println!(
        "  🏷️  Labeled ratio:     {:.0}% (SSL stream: {:.0}%)",
        labeled_ratio * 100.0,
        (1.0 - labeled_ratio) * 100.0
    );
    println!("  💾 Output directory:   {}", output_dir);
    println!("  🖥️  Backend:          {}", backend_name());
    println!();

    let config = SimulationConfig {
        data_dir: data_dir.to_string(),
        model_path: model.to_string(),
        days,
        images_per_day,
        confidence_threshold,
        retrain_threshold,
        labeled_ratio,
        output_dir: output_dir.to_string(),
        seed: 42,
        batch_size: 64, // Standard batch size for GPU training
        learning_rate: 0.0001,
        retrain_epochs: 5,
    };

    let results = run_simulation::<TrainingBackend>(config)?;

    println!();
    println!("{}", "Simulation Summary:".green().bold());
    println!("  SSL improvement: {:.2}%", results.improvement());

    Ok(())
}

fn find_latest_checkpoint(dir: &Path) -> Result<PathBuf> {
    if !dir.exists() {
        return Err(anyhow!(
            "Simulation output directory does not exist: {}",
            dir.display()
        ));
    }

    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.eq_ignore_ascii_case("mpk"))
                    .unwrap_or(false)
        })
        .filter_map(|p| {
            let modified = std::fs::metadata(&p).ok()?.modified().ok()?;
            Some((modified, p))
        })
        .collect();

    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    candidates
        .into_iter()
        .next()
        .map(|(_, p)| p)
        .ok_or_else(|| anyhow!("No .mpk checkpoint found in {}", dir.display()))
}

fn evaluate_with_metrics<B: Backend>(
    model: &PlantClassifier<B>,
    dataset: &PlantVillageBurnDataset,
    batch_size: usize,
    image_size: usize,
    num_classes: usize,
) -> (f64, f64) {
    let device = <B as Backend>::Device::default();
    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);
    let len = dataset.len();

    if len == 0 || num_classes == 0 {
        return (0.0, 0.0);
    }

    let mut correct = 0usize;
    let mut total = 0usize;

    let mut tp = vec![0usize; num_classes];
    let mut fp = vec![0usize; num_classes];
    let mut fn_ = vec![0usize; num_classes];

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch = batcher.batch(items, &device);
        let output = model.forward(batch.images);
        let probs = softmax(output, 1);
        let preds = probs.argmax(1);
        let [batch_dim, _] = preds.dims();

        let preds_flat: Tensor<B, 1, Int> = preds.reshape([batch_dim]);
        let pred_vec: Vec<i32> = preds_flat.into_data().to_vec::<i32>().unwrap_or_default();

        let target_vec: Vec<i32> = batch
            .targets
            .into_data()
            .to_vec::<i32>()
            .unwrap_or_default();

        for (&p_val, &t_val) in pred_vec.iter().zip(target_vec.iter()) {
            let p = p_val as usize;
            let t = t_val as usize;

            if p == t {
                correct += 1;
                if t < num_classes {
                    tp[t] += 1;
                }
            } else {
                if p < num_classes {
                    fp[p] += 1;
                }
                if t < num_classes {
                    fn_[t] += 1;
                }
            }
            total += 1;
        }
    }

    let accuracy = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };

    let mut f1_sum = 0.0f64;
    let mut counted = 0usize;
    for c in 0..num_classes {
        let tp_c = tp[c] as f64;
        let fp_c = fp[c] as f64;
        let fn_c = fn_[c] as f64;

        let precision = if (tp_c + fp_c) > 0.0 {
            tp_c / (tp_c + fp_c)
        } else {
            0.0
        };
        let recall = if (tp_c + fn_c) > 0.0 {
            tp_c / (tp_c + fn_c)
        } else {
            0.0
        };
        let f1 = if (precision + recall) > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        f1_sum += f1;
        counted += 1;
    }

    let macro_f1 = if counted > 0 {
        100.0 * (f1_sum / counted as f64)
    } else {
        0.0
    };

    (accuracy, macro_f1)
}

fn cmd_eval(
    data_dir: &str,
    model: Option<&str>,
    labeled_ratio: f64,
    batch_size: usize,
    simulation_output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::backend::DefaultBackend;

    info!("Evaluating model");
    info!("  Data dir: {}", data_dir);
    info!("  Batch size: {}", batch_size);
    info!("  Labeled ratio: {}", labeled_ratio);

    if !Path::new(data_dir).exists() {
        return Err(anyhow!("Dataset directory not found: {}", data_dir));
    }

    let model_path = if let Some(path) = model {
        PathBuf::from(path)
    } else {
        find_latest_checkpoint(Path::new(simulation_output_dir))?
    };

    if !model_path.exists() {
        return Err(anyhow!(
            "Model checkpoint not found: {}",
            model_path.display()
        ));
    }

    println!("{}", "Evaluation Configuration:".cyan().bold());
    println!("  📁 Data directory: {}", data_dir);
    println!("  🧠 Model: {}", model_path.display());
    println!("  📦 Batch size: {}", batch_size);
    println!("  🏷️  Labeled ratio: {:.0}%", labeled_ratio * 100.0);
    println!();

    let dataset = PlantVillageDataset::new(data_dir)?;
    let images: Vec<(PathBuf, usize, String)> = dataset
        .samples
        .iter()
        .map(|s| {
            let class_name = dataset
                .idx_to_class
                .get(&s.label)
                .cloned()
                .unwrap_or_else(|| format!("class_{}", s.label));
            (s.path.clone(), s.label, class_name)
        })
        .collect();

    let split_config = SplitConfig::new(0.10, 0.10, labeled_ratio / 0.80, 42)?;
    let splits = DatasetSplits::from_images(images, split_config)?;

    let eval_samples: Vec<(PathBuf, usize)> = if !splits.test_set.is_empty() {
        println!("Using {} samples from test split.", splits.test_set.len());
        splits
            .test_set
            .iter()
            .map(|x| (x.image_path.clone(), x.label))
            .collect()
    } else {
        println!("Test split empty, falling back to validation split.");
        splits
            .validation_set
            .iter()
            .map(|x| (x.image_path.clone(), x.label))
            .collect()
    };

    let eval_dataset = PlantVillageBurnDataset::new_cached(eval_samples, 128)?;
    let device = Default::default();

    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    let model_instance: PlantClassifier<DefaultBackend> =
        PlantClassifier::new(&model_config, &device)
            .load_file(&model_path, &CompactRecorder::new(), &device)
            .map_err(|e| anyhow!("Failed to load model {}: {:?}", model_path.display(), e))?;

    let (accuracy, macro_f1) = evaluate_with_metrics::<DefaultBackend>(
        &model_instance,
        &eval_dataset,
        batch_size,
        128,
        38,
    );

    println!();
    println!("{}", "Evaluation Results:".green().bold());
    println!("  ✅ Accuracy (Top-1): {:.2}%", accuracy);
    println!("  ✅ Macro F1:          {:.2}%", macro_f1);

    Ok(())
}

fn cmd_export(_input_dir: &str, _format: &str, _output: &str) -> Result<()> {
    info!("Exporting metrics");
    info!("  Input directory: {}", _input_dir);
    info!("  Format: {}", _format);
    info!("  Output: {}", _output);

    println!("{}", "Export Configuration:".cyan().bold());
    println!("  📁 Input directory: {}", _input_dir);
    println!("  📄 Format:          {}", _format);
    println!("  💾 Output:          {}", _output);
    println!();

    println!("{} Export implementation pending.", "Note:".yellow());
    println!("  Metrics can be exported from:");
    println!("    • Training logs (loss, accuracy per epoch)");
    println!("    • Confusion matrices");
    println!("    • Pseudo-label precision over time");
    println!("    • Benchmark results");

    Ok(())
}
