//! PlantVillage Semi-Supervised Learning CLI
//!
//! This is the main entry point for the PlantVillage plant disease classification
//! system using semi-supervised learning with the Burn framework.

use std::path::Path;

use anyhow::Result;
use burn::backend::Autodiff;
use clap::{Parser, Subcommand};
use colored::Colorize;
use tracing::info;

use plantvillage_ssl::backend::{DefaultBackend, TrainingBackend, backend_name};
use plantvillage_ssl::utils::logging::{init_logging, LogConfig};

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

    /// Prepare a balanced dataset from raw PlantVillage data
    Prepare {
        /// Source directory containing raw data (will auto-detect nested structure)
        #[arg(short, long, default_value = "data/plantvillage/raw")]
        source_dir: String,

        /// Output directory for balanced dataset
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        output_dir: String,

        /// Samples per class (default: use minimum class size for perfect balance)
        #[arg(short, long)]
        samples_per_class: Option<usize>,

        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Train the model with semi-supervised learning
    Train {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "50")]
        epochs: usize,

        /// Batch size for training
        #[arg(short, long, default_value = "32")]
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

        /// Use class-weighted loss (inverse frequency weighting for imbalanced data)
        #[arg(long, default_value = "false")]
        class_weighted: bool,
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

        Commands::Prepare {
            source_dir,
            output_dir,
            samples_per_class,
            seed,
        } => {
            cmd_prepare(&source_dir, &output_dir, samples_per_class, seed)?;
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
            class_weighted,
        } => {
            // Always use CUDA - this project targets GPU
            let _ = cuda; // Ignore flag, always GPU

            let max_samples = if quick {
                println!("{}", "🚀 Quick test mode: using only 500 samples".yellow().bold());
                Some(500usize)
            } else {
                None
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
                class_weighted,
            )?;
        }

        Commands::Infer { input, model, cuda: _cuda } => {
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
            cmd_benchmark(model.as_deref(), iterations, warmup, batch_size, image_size, output.as_deref(), verbose)?;
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
    info!("Downloading PlantVillage dataset to: {}", output_dir);

    println!(
        "{} Dataset download not yet implemented. Please download manually from Kaggle.",
        "Note:".yellow()
    );
    println!("  Dataset URL: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset");
    println!("  Extract to: {}", output_dir);
    println!();
    println!("{}", "Steps to download manually:".cyan());
    println!("  1. Visit the Kaggle URL above");
    println!("  2. Download and extract the dataset");
    println!("  3. Run: plantvillage_ssl prepare --source-dir {}/raw --output-dir {}/balanced", output_dir, output_dir);

    Ok(())
}

fn cmd_prepare(source_dir: &str, output_dir: &str, samples_per_class: Option<usize>, seed: u64) -> Result<()> {
    use plantvillage_ssl::dataset::prepare::{prepare_balanced_dataset, PrepareConfig};
    use std::path::Path;

    info!("Preparing balanced dataset");
    info!("  Source: {}", source_dir);
    info!("  Output: {}", output_dir);

    println!("{}", "Dataset Preparation".cyan().bold());
    println!("  📁 Source: {}", source_dir);
    println!("  📁 Output: {}", output_dir);
    println!("  🎲 Seed: {}", seed);
    if let Some(n) = samples_per_class {
        println!("  📊 Samples per class: {}", n);
    } else {
        println!("  📊 Samples per class: auto (minimum class size)");
    }
    println!();

    let config = PrepareConfig {
        seed,
        samples_per_class,
    };

    let stats = prepare_balanced_dataset(
        Path::new(source_dir),
        Path::new(output_dir),
        &config,
    )?;

    println!();
    println!("{}", "Next steps:".cyan().bold());
    println!("  • Train with balanced data:");
    println!("    plantvillage_ssl train --data-dir {} --epochs 50", output_dir);
    println!();
    println!("  • Or use class-weighted loss with original data:");
    println!("    plantvillage_ssl train --data-dir data/plantvillage/raw --class-weighted");

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

                println!("  🧪 Test set:        {} ({:.1}%)", test_size, 100.0 * test_size as f64 / total as f64);
                println!("  ✅ Validation set:  {} ({:.1}%)", val_size, 100.0 * val_size as f64 / total as f64);
                println!("  🏷️  Labeled pool:    {} ({:.1}%)", labeled_size, 100.0 * labeled_size as f64 / total as f64);
                println!("  📷 Stream pool:     {} ({:.1}%)", stream_size, 100.0 * stream_size as f64 / total as f64);
                println!();
            }

            println!("{}", "Class Distribution:".cyan().bold());
            for (idx, count) in stats.class_counts.iter().enumerate() {
                let class_name = stats.class_names.get(&idx)
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
    use burn::tensor::Tensor;
    use burn::tensor::activation::softmax;
    use image::imageops::FilterType;
    use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
    use plantvillage_ssl::dataset::CLASS_NAMES;
    use plantvillage_ssl::backend::{DefaultBackend, default_device, backend_name};

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
        let mut indexed: Vec<(usize, f32)> = probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Get actual class from filename
        let actual_class = file_path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown");
        
        let predicted_class = CLASS_NAMES.get(indexed[0].0).unwrap_or(&"Unknown");
        let is_correct = actual_class == *predicted_class;
        
        println!("📷 {}", file_path.file_name().unwrap_or_default().to_string_lossy());
        println!("  Actual:    {}", actual_class.yellow());
        println!("  Predicted: {} {}", 
            predicted_class,
            if is_correct { "✅".green() } else { "❌".red() }
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
    use plantvillage_ssl::inference::{BenchmarkConfig, run_benchmark};
    use plantvillage_ssl::backend::{DefaultBackend, default_device};
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
    use plantvillage_ssl::training::{run_simulation, SimulationConfig};
    use plantvillage_ssl::backend::{TrainingBackend, backend_name};

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
    println!("  🏷️  Labeled ratio:     {:.0}% (SSL stream: {:.0}%)", labeled_ratio * 100.0, (1.0 - labeled_ratio - 0.20) * 100.0);
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
        batch_size: 32,  // Standard batch size for GPU training
        learning_rate: 0.0001,
        retrain_epochs: 5,
    };

    let results = run_simulation::<TrainingBackend>(config)?;

    println!();
    println!("{}", "Simulation Summary:".green().bold());
    println!("  SSL improvement: {:.2}%", results.improvement());

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
