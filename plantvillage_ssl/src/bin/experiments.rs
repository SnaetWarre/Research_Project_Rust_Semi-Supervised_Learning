//! Experiment Runner for Research Project
//!
//! This binary runs the key experiments for the research project:
//! 1. Label Efficiency Curve: How many images per class are needed?
//! 2. Class Scaling: Is 5→6 harder than 30→31?
//!
//! Results are saved to output/experiments/ with conclusions in a .txt file.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use burn::backend::Autodiff;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use burn_cuda::Cuda;
use chrono::Local;
use clap::{Parser, Subcommand};
use colored::Colorize;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use plantvillage_ssl::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageBurnDataset};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::PlantVillageDataset;

type Backend = Autodiff<Cuda>;

/// Experiment Runner CLI
#[derive(Parser, Debug)]
#[command(name = "experiments")]
#[command(about = "Run research experiments for plant disease classification")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run label efficiency experiment: How many images per new class are needed?
    LabelEfficiency {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments/label_efficiency")]
        output_dir: String,

        /// Number of epochs per experiment
        #[arg(short, long, default_value = "30")]
        epochs: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run class scaling experiment: Is 5→6 harder than 30→31?
    ClassScaling {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments/class_scaling")]
        output_dir: String,

        /// Number of epochs per experiment
        #[arg(short, long, default_value = "30")]
        epochs: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run all experiments
    All {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments")]
        output_dir: String,

        /// Number of epochs per experiment
        #[arg(short, long, default_value = "30")]
        epochs: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run SSL + Incremental Learning combined experiment
    SslIncremental {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments/ssl_incremental")]
        output_dir: String,

        /// Number of base classes
        #[arg(long, default_value = "30")]
        base_classes: usize,

        /// Labeled samples per new class (simulating limited labels)
        #[arg(long, default_value = "10")]
        labeled_samples: usize,

        /// Confidence threshold for pseudo-labeling
        #[arg(long, default_value = "0.8")]
        confidence_threshold: f64,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run new class position experiment: Does 6th vs 31st class need different amounts of labels?
    NewClassPosition {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage/balanced")]
        data_dir: String,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments/new_class_position")]
        output_dir: String,

        /// Number of epochs per experiment
        #[arg(short, long, default_value = "30")]
        epochs: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },

    /// Run inference benchmark (with Jetson power monitoring if available)
    Benchmark {
        /// Path to model checkpoint (optional, uses fresh model if not specified)
        #[arg(short, long)]
        model_path: Option<String>,

        /// Output directory for results
        #[arg(short, long, default_value = "output/experiments/benchmark")]
        output_dir: String,

        /// Number of warmup iterations
        #[arg(long, default_value = "10")]
        warmup: usize,

        /// Number of benchmark iterations
        #[arg(long, default_value = "100")]
        iterations: usize,

        /// Batch size
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Image size
        #[arg(long, default_value = "128")]
        image_size: usize,
    },
}

/// Results from label efficiency experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabelEfficiencyResults {
    /// Images per class tested
    images_per_class: Vec<usize>,
    /// Accuracy for each configuration
    accuracies: Vec<f64>,
    /// Training time in seconds
    training_times: Vec<f64>,
    /// Best accuracy achieved
    best_accuracy: f64,
    /// Images per class that achieved best accuracy
    best_images_per_class: usize,
    /// Minimum images for acceptable accuracy (>80%)
    min_acceptable_images: Option<usize>,
}

/// Results from class scaling experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassScalingResults {
    /// Results for small base (5 classes + 1)
    small_base: ScalingResult,
    /// Results for large base (30 classes + 1)
    large_base: ScalingResult,
    /// Relative difficulty (large_base forgetting / small_base forgetting)
    relative_difficulty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalingResult {
    /// Number of base classes
    base_classes: usize,
    /// Number of classes after adding
    total_classes: usize,
    /// Accuracy on base classes before adding new class
    base_accuracy_before: f64,
    /// Accuracy on base classes after adding new class
    base_accuracy_after: f64,
    /// Accuracy on new class
    new_class_accuracy: f64,
    /// Overall accuracy after adding
    overall_accuracy: f64,
    /// Forgetting measure (accuracy drop on old classes)
    forgetting: f64,
    /// Training time in seconds
    training_time: f64,
}

/// Results from new class position experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewClassPositionResults {
    /// Results for small base (5 classes) with varying labeled samples
    small_base_results: Vec<PositionLabelResult>,
    /// Results for large base (30 classes) with varying labeled samples
    large_base_results: Vec<PositionLabelResult>,
    /// Summary statistics
    summary: PositionSummary,
}

/// Result for a single combination of base size and labeled samples
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionLabelResult {
    /// Number of base classes
    base_classes: usize,
    /// Number of labeled samples for the new class
    labeled_samples: usize,
    /// Accuracy on the new class
    new_class_accuracy: f64,
    /// Accuracy on base classes after adding
    base_accuracy_after: f64,
    /// Forgetting (accuracy drop on old classes)
    forgetting: f64,
    /// Overall accuracy
    overall_accuracy: f64,
    /// Training time in seconds
    training_time: f64,
}

/// Summary of the position experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionSummary {
    /// Minimum samples for 70% accuracy on new class (small base)
    min_samples_small_70pct: Option<usize>,
    /// Minimum samples for 70% accuracy on new class (large base)
    min_samples_large_70pct: Option<usize>,
    /// Minimum samples for 80% accuracy on new class (small base)
    min_samples_small_80pct: Option<usize>,
    /// Minimum samples for 80% accuracy on new class (large base)
    min_samples_large_80pct: Option<usize>,
    /// Average forgetting difference (large - small)
    avg_forgetting_difference: f64,
    /// Is learning harder as 31st class?
    harder_as_31st: bool,
    /// Samples needed ratio (large/small for equivalent accuracy)
    samples_ratio: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    print_banner();

    match cli.command {
        Commands::LabelEfficiency {
            data_dir,
            output_dir,
            epochs,
            seed,
        } => {
            run_label_efficiency(&data_dir, &output_dir, epochs, seed)?;
        }
        Commands::ClassScaling {
            data_dir,
            output_dir,
            epochs,
            seed,
        } => {
            run_class_scaling(&data_dir, &output_dir, epochs, seed)?;
        }
        Commands::All {
            data_dir,
            output_dir,
            epochs,
            seed,
        } => {
            println!("{}", "Running ALL experiments...".green().bold());
            println!();

            let le_output = format!("{}/label_efficiency", output_dir);
            run_label_efficiency(&data_dir, &le_output, epochs, seed)?;

            println!();
            println!("{}", "=".repeat(80).cyan());
            println!();

            let cs_output = format!("{}/class_scaling", output_dir);
            run_class_scaling(&data_dir, &cs_output, epochs, seed)?;

            println!();
            println!("{}", "=".repeat(80).cyan());
            println!();

            let ncp_output = format!("{}/new_class_position", output_dir);
            run_new_class_position(&data_dir, &ncp_output, epochs, seed)?;
        }
        Commands::SslIncremental {
            data_dir,
            output_dir,
            base_classes,
            labeled_samples,
            confidence_threshold,
            seed,
        } => {
            run_ssl_incremental(&data_dir, &output_dir, base_classes, labeled_samples, confidence_threshold, seed)?;
        }
        Commands::NewClassPosition {
            data_dir,
            output_dir,
            epochs,
            seed,
        } => {
            run_new_class_position(&data_dir, &output_dir, epochs, seed)?;
        }
        Commands::Benchmark {
            model_path,
            output_dir,
            warmup,
            iterations,
            batch_size,
            image_size,
        } => {
            run_inference_benchmark(model_path.as_deref(), &output_dir, warmup, iterations, batch_size, image_size)?;
        }
    }

    Ok(())
}

fn print_banner() {
    println!(
        "{}",
        r#"
╔══════════════════════════════════════════════════════════════════╗
║   🔬 Plant Disease Classification - Experiment Runner               ║
║   Research Project: Semi-Supervised Learning on Edge Devices        ║
╚══════════════════════════════════════════════════════════════════╝
  "#
        .cyan()
    );
}

/// Run label efficiency experiment
fn run_label_efficiency(data_dir: &str, output_dir: &str, epochs: usize, seed: u64) -> Result<()> {
    println!("{}", "EXPERIMENT 1: Label Efficiency Curve".yellow().bold());
    println!("Question: How many labeled images per new class are needed for good accuracy?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Test different numbers of images per class
    let images_per_class_tests = vec![5, 10, 25, 50, 100, 152]; // 152 is max in balanced dataset
    let mut results = LabelEfficiencyResults {
        images_per_class: Vec::new(),
        accuracies: Vec::new(),
        training_times: Vec::new(),
        best_accuracy: 0.0,
        best_images_per_class: 0,
        min_acceptable_images: None,
    };

    for &n_images in &images_per_class_tests {
        println!(
            "{}",
            format!("Testing with {} images per class...", n_images)
                .yellow()
                .bold()
        );

        let start = Instant::now();
        let accuracy = train_with_n_images_per_class::<Backend>(&dataset, n_images, epochs, seed)?;
        let training_time = start.elapsed().as_secs_f64();

        results.images_per_class.push(n_images);
        results.accuracies.push(accuracy);
        results.training_times.push(training_time);

        println!(
            "  {} images/class → {:.2}% accuracy ({:.1}s)",
            n_images, accuracy, training_time
        );

        if accuracy > results.best_accuracy {
            results.best_accuracy = accuracy;
            results.best_images_per_class = n_images;
        }

        if results.min_acceptable_images.is_none() && accuracy >= 80.0 {
            results.min_acceptable_images = Some(n_images);
        }

        println!();
    }

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_label_efficiency_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG chart
    generate_label_efficiency_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Run class scaling experiment
fn run_class_scaling(data_dir: &str, output_dir: &str, epochs: usize, seed: u64) -> Result<()> {
    println!("{}", "EXPERIMENT 2: Class Scaling Effect".yellow().bold());
    println!("Question: Is adding a class to 5 classes harder than adding to 30 classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Experiment A: 5 → 6 classes
    println!("{}", "Part A: 5 → 6 classes".yellow().bold());
    let small_base = run_incremental_experiment::<Backend>(&dataset, 5, 1, epochs, seed)?;
    println!(
        "  Base accuracy: {:.2}% → {:.2}% (forgetting: {:.2}%)",
        small_base.base_accuracy_before,
        small_base.base_accuracy_after,
        small_base.forgetting
    );
    println!("  New class accuracy: {:.2}%", small_base.new_class_accuracy);
    println!("  Overall accuracy: {:.2}%", small_base.overall_accuracy);
    println!();

    // Experiment B: 30 → 31 classes
    println!("{}", "Part B: 30 → 31 classes".yellow().bold());
    let large_base = run_incremental_experiment::<Backend>(&dataset, 30, 1, epochs, seed)?;
    println!(
        "  Base accuracy: {:.2}% → {:.2}% (forgetting: {:.2}%)",
        large_base.base_accuracy_before,
        large_base.base_accuracy_after,
        large_base.forgetting
    );
    println!("  New class accuracy: {:.2}%", large_base.new_class_accuracy);
    println!("  Overall accuracy: {:.2}%", large_base.overall_accuracy);
    println!();

    let relative_difficulty = if small_base.forgetting > 0.0 {
        large_base.forgetting / small_base.forgetting
    } else {
        1.0
    };

    let results = ClassScalingResults {
        small_base,
        large_base,
        relative_difficulty,
    };

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_class_scaling_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG chart
    generate_class_scaling_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Run new class position experiment: Does adding a class as 6th vs 31st need different amounts of labels?
fn run_new_class_position(data_dir: &str, output_dir: &str, epochs: usize, seed: u64) -> Result<()> {
    println!("{}", "EXPERIMENT 4: New Class Position Effect".yellow().bold());
    println!("Question: Does adding a class as 6th vs 31st require different amounts of labeled samples?");
    println!("Sub-question: How does the number of existing classes affect label efficiency for new classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Test different numbers of labeled samples for the new class
    let labeled_samples_tests = vec![5, 10, 25, 50, 100];
    
    let mut small_base_results: Vec<PositionLabelResult> = Vec::new();
    let mut large_base_results: Vec<PositionLabelResult> = Vec::new();

    // Part A: 5 → 6 classes with varying labeled samples
    println!("{}", "Part A: Adding 6th class (5 base classes)".yellow().bold());
    println!();
    
    for &n_samples in &labeled_samples_tests {
        println!(
            "  Testing with {} labeled samples for new class...",
            n_samples
        );
        let result = run_incremental_with_limited_labels::<Backend>(
            &dataset, 5, 1, n_samples, epochs, seed
        )?;
        println!(
            "    New class: {:.2}%, Base after: {:.2}%, Forgetting: {:.2}%",
            result.new_class_accuracy, result.base_accuracy_after, result.forgetting
        );
        small_base_results.push(PositionLabelResult {
            base_classes: 5,
            labeled_samples: n_samples,
            new_class_accuracy: result.new_class_accuracy,
            base_accuracy_after: result.base_accuracy_after,
            forgetting: result.forgetting,
            overall_accuracy: result.overall_accuracy,
            training_time: result.training_time,
        });
    }
    println!();

    // Part B: 30 → 31 classes with varying labeled samples
    println!("{}", "Part B: Adding 31st class (30 base classes)".yellow().bold());
    println!();
    
    for &n_samples in &labeled_samples_tests {
        println!(
            "  Testing with {} labeled samples for new class...",
            n_samples
        );
        let result = run_incremental_with_limited_labels::<Backend>(
            &dataset, 30, 1, n_samples, epochs, seed
        )?;
        println!(
            "    New class: {:.2}%, Base after: {:.2}%, Forgetting: {:.2}%",
            result.new_class_accuracy, result.base_accuracy_after, result.forgetting
        );
        large_base_results.push(PositionLabelResult {
            base_classes: 30,
            labeled_samples: n_samples,
            new_class_accuracy: result.new_class_accuracy,
            base_accuracy_after: result.base_accuracy_after,
            forgetting: result.forgetting,
            overall_accuracy: result.overall_accuracy,
            training_time: result.training_time,
        });
    }
    println!();

    // Calculate summary statistics
    let summary = calculate_position_summary(&small_base_results, &large_base_results);

    let results = NewClassPositionResults {
        small_base_results,
        large_base_results,
        summary,
    };

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_new_class_position_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG charts
    generate_new_class_position_charts(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Run incremental experiment with limited labels for the new class
fn run_incremental_with_limited_labels<B: AutodiffBackend>(
    dataset: &PlantVillageDataset,
    base_classes: usize,
    new_classes: usize,
    max_labels_per_new_class: usize,
    epochs: usize,
    seed: u64,
) -> Result<ScalingResult> {
    let device = B::Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = Instant::now();

    // Group by class
    let mut by_class: HashMap<usize, Vec<&plantvillage_ssl::dataset::loader::ImageSample>> =
        HashMap::new();
    for sample in &dataset.samples {
        by_class.entry(sample.label).or_default().push(sample);
    }

    // Get class indices sorted
    let mut class_indices: Vec<usize> = by_class.keys().cloned().collect();
    class_indices.sort();

    let base_class_indices: Vec<usize> = class_indices.iter().take(base_classes).cloned().collect();
    let new_class_indices: Vec<usize> = class_indices
        .iter()
        .skip(base_classes)
        .take(new_classes)
        .cloned()
        .collect();

    // Create base training data (80% train, 20% val for each class)
    let mut base_train: Vec<(PathBuf, usize)> = Vec::new();
    let mut base_val: Vec<(PathBuf, usize)> = Vec::new();

    for &class_idx in &base_class_indices {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);
            let split = (samples.len() as f64 * 0.8) as usize;
            let new_label = base_class_indices.iter().position(|&x| x == class_idx).unwrap();
            for s in samples.iter().take(split) {
                base_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(split) {
                base_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Train base model
    let image_size = 128;
    let batch_size = 32;
    let learning_rate = 0.0001;

    let train_dataset =
        PlantVillageBurnDataset::new_cached(base_train.clone(), image_size).expect("Failed");
    let val_dataset =
        PlantVillageBurnDataset::new_cached(base_val.clone(), image_size).expect("Failed");

    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);

    let model_config = PlantClassifierConfig {
        num_classes: base_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    // Train base model
    for _epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = (indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }
    }

    // Evaluate base model on base classes
    let base_accuracy_before = evaluate::<B>(&model, &val_dataset, &batcher, batch_size, image_size);

    // Now add new class with LIMITED labels
    let total_classes = base_classes + new_classes;

    // Prepare combined data - but limit new class samples
    let mut combined_train: Vec<(PathBuf, usize)> = base_train.clone();
    let mut combined_val: Vec<(PathBuf, usize)> = base_val.clone();
    let mut new_class_val: Vec<(PathBuf, usize)> = Vec::new();

    for (i, &class_idx) in new_class_indices.iter().enumerate() {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);
            
            // Limit training samples to max_labels_per_new_class
            let available_train = (samples.len() as f64 * 0.8) as usize;
            let train_n = max_labels_per_new_class.min(available_train);
            
            // Use remaining as validation
            let val_start = train_n;
            let new_label = base_classes + i;
            
            for s in samples.iter().take(train_n) {
                combined_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(val_start) {
                combined_val.push((s.path.clone(), new_label));
                new_class_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Create new model with extended classes
    let new_model_config = PlantClassifierConfig {
        num_classes: total_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut new_model = PlantClassifier::<B>::new(&new_model_config, &device);
    let mut new_optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let combined_train_dataset =
        PlantVillageBurnDataset::new_cached(combined_train, image_size).expect("Failed");
    let combined_val_dataset =
        PlantVillageBurnDataset::new_cached(combined_val.clone(), image_size).expect("Failed");
    let new_batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);

    // Train on combined data
    for _epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..combined_train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = (indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| combined_train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = new_batcher.batch(items, &device);
            let output = new_model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &new_model);
            new_model = new_optimizer.step(learning_rate, new_model, grads);
        }
    }

    // Evaluate on base classes only (to measure forgetting)
    let base_val_dataset =
        PlantVillageBurnDataset::new_cached(base_val, image_size).expect("Failed");
    let base_accuracy_after =
        evaluate::<B>(&new_model, &base_val_dataset, &new_batcher, batch_size, image_size);

    // Evaluate on new class only
    let new_class_accuracy = if !new_class_val.is_empty() {
        let new_class_val_dataset =
            PlantVillageBurnDataset::new_cached(new_class_val, image_size).expect("Failed");
        evaluate::<B>(&new_model, &new_class_val_dataset, &new_batcher, batch_size, image_size)
    } else {
        0.0
    };

    // Overall accuracy
    let overall_accuracy =
        evaluate::<B>(&new_model, &combined_val_dataset, &new_batcher, batch_size, image_size);

    let forgetting = base_accuracy_before - base_accuracy_after;
    let training_time = start.elapsed().as_secs_f64();

    Ok(ScalingResult {
        base_classes,
        total_classes,
        base_accuracy_before,
        base_accuracy_after,
        new_class_accuracy,
        overall_accuracy,
        forgetting,
        training_time,
    })
}

/// Calculate summary statistics for the position experiment
fn calculate_position_summary(
    small_results: &[PositionLabelResult],
    large_results: &[PositionLabelResult],
) -> PositionSummary {
    // Find minimum samples for 70% accuracy
    let min_samples_small_70pct = small_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 70.0)
        .map(|r| r.labeled_samples)
        .min();
    
    let min_samples_large_70pct = large_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 70.0)
        .map(|r| r.labeled_samples)
        .min();

    // Find minimum samples for 80% accuracy
    let min_samples_small_80pct = small_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 80.0)
        .map(|r| r.labeled_samples)
        .min();
    
    let min_samples_large_80pct = large_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 80.0)
        .map(|r| r.labeled_samples)
        .min();

    // Calculate average forgetting difference
    let avg_small_forgetting: f64 = if !small_results.is_empty() {
        small_results.iter().map(|r| r.forgetting).sum::<f64>() / small_results.len() as f64
    } else {
        0.0
    };
    
    let avg_large_forgetting: f64 = if !large_results.is_empty() {
        large_results.iter().map(|r| r.forgetting).sum::<f64>() / large_results.len() as f64
    } else {
        0.0
    };
    
    let avg_forgetting_difference = avg_large_forgetting - avg_small_forgetting;

    // Compare equivalent accuracy levels to determine if harder as 31st
    let harder_as_31st = match (min_samples_small_70pct, min_samples_large_70pct) {
        (Some(small), Some(large)) => large > small,
        (Some(_), None) => true, // Large never achieved 70%
        _ => false,
    };

    // Calculate samples ratio for equivalent accuracy
    let samples_ratio = match (min_samples_small_70pct, min_samples_large_70pct) {
        (Some(small), Some(large)) if small > 0 => large as f64 / small as f64,
        _ => 1.0,
    };

    PositionSummary {
        min_samples_small_70pct,
        min_samples_large_70pct,
        min_samples_small_80pct,
        min_samples_large_80pct,
        avg_forgetting_difference,
        harder_as_31st,
        samples_ratio,
    }
}

/// Train model with n images per class and return validation accuracy
fn train_with_n_images_per_class<B: AutodiffBackend>(
    dataset: &PlantVillageDataset,
    n_images: usize,
    epochs: usize,
    seed: u64,
) -> Result<f64> {
    let device = B::Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sample n images per class
    let mut by_class: HashMap<usize, Vec<&plantvillage_ssl::dataset::loader::ImageSample>> =
        HashMap::new();
    for sample in &dataset.samples {
        by_class.entry(sample.label).or_default().push(sample);
    }

    let mut train_samples: Vec<(PathBuf, usize)> = Vec::new();
    let mut val_samples: Vec<(PathBuf, usize)> = Vec::new();

    for (label, samples) in by_class.iter_mut() {
        samples.shuffle(&mut rng);
        let take_n = n_images.min(samples.len());
        let val_n = (take_n as f64 * 0.2).ceil() as usize;
        let train_n = take_n - val_n;

        for s in samples.iter().take(train_n) {
            train_samples.push((s.path.clone(), *label));
        }
        for s in samples.iter().skip(train_n).take(val_n) {
            val_samples.push((s.path.clone(), *label));
        }
    }

    // Train
    let image_size = 128;
    let batch_size = 32;
    let learning_rate = 0.0001;

    let train_dataset =
        PlantVillageBurnDataset::new_cached(train_samples, image_size).expect("Failed to load");
    let val_dataset =
        PlantVillageBurnDataset::new_cached(val_samples, image_size).expect("Failed to load");

    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);

    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    for _epoch in 0..epochs {
        // Training
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = (indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }
    }

    // Evaluate
    let val_acc = evaluate::<B>(&model, &val_dataset, &batcher, batch_size, image_size);

    Ok(val_acc)
}

/// Run incremental learning experiment: train on base_classes, then add new_classes
fn run_incremental_experiment<B: AutodiffBackend>(
    dataset: &PlantVillageDataset,
    base_classes: usize,
    new_classes: usize,
    epochs: usize,
    seed: u64,
) -> Result<ScalingResult> {
    let device = B::Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let start = Instant::now();

    // Group by class
    let mut by_class: HashMap<usize, Vec<&plantvillage_ssl::dataset::loader::ImageSample>> =
        HashMap::new();
    for sample in &dataset.samples {
        by_class.entry(sample.label).or_default().push(sample);
    }

    // Get class indices sorted
    let mut class_indices: Vec<usize> = by_class.keys().cloned().collect();
    class_indices.sort();

    let base_class_indices: Vec<usize> = class_indices.iter().take(base_classes).cloned().collect();
    let new_class_indices: Vec<usize> = class_indices
        .iter()
        .skip(base_classes)
        .take(new_classes)
        .cloned()
        .collect();

    // Create base training data (80% train, 20% val for each class)
    let mut base_train: Vec<(PathBuf, usize)> = Vec::new();
    let mut base_val: Vec<(PathBuf, usize)> = Vec::new();

    for &class_idx in &base_class_indices {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);
            let split = (samples.len() as f64 * 0.8) as usize;
            // Remap to 0..base_classes
            let new_label = base_class_indices.iter().position(|&x| x == class_idx).unwrap();
            for s in samples.iter().take(split) {
                base_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(split) {
                base_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Train base model
    let image_size = 128;
    let batch_size = 32;
    let learning_rate = 0.0001;

    let train_dataset =
        PlantVillageBurnDataset::new_cached(base_train.clone(), image_size).expect("Failed");
    let val_dataset =
        PlantVillageBurnDataset::new_cached(base_val.clone(), image_size).expect("Failed");

    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);

    let model_config = PlantClassifierConfig {
        num_classes: base_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    // Train base model
    for _epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = (indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }
    }

    // Evaluate base model on base classes
    let base_accuracy_before = evaluate::<B>(&model, &val_dataset, &batcher, batch_size, image_size);

    // Now add new class and retrain with simple fine-tuning
    // For simplicity, we'll create a new model with more classes and train on all data
    let total_classes = base_classes + new_classes;

    // Prepare combined data
    let mut combined_train: Vec<(PathBuf, usize)> = base_train.clone();
    let mut combined_val: Vec<(PathBuf, usize)> = base_val.clone();
    let mut new_class_val: Vec<(PathBuf, usize)> = Vec::new();

    for (i, &class_idx) in new_class_indices.iter().enumerate() {
        if let Some(samples) = by_class.get_mut(&class_idx) {
            samples.shuffle(&mut rng);
            let split = (samples.len() as f64 * 0.8) as usize;
            let new_label = base_classes + i;
            for s in samples.iter().take(split) {
                combined_train.push((s.path.clone(), new_label));
            }
            for s in samples.iter().skip(split) {
                combined_val.push((s.path.clone(), new_label));
                new_class_val.push((s.path.clone(), new_label));
            }
        }
    }

    // Create new model with extended classes
    let new_model_config = PlantClassifierConfig {
        num_classes: total_classes,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut new_model = PlantClassifier::<B>::new(&new_model_config, &device);
    let mut new_optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let combined_train_dataset =
        PlantVillageBurnDataset::new_cached(combined_train, image_size).expect("Failed");
    let combined_val_dataset =
        PlantVillageBurnDataset::new_cached(combined_val.clone(), image_size).expect("Failed");
    let new_batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);

    // Train on combined data (simple fine-tuning approach)
    for _epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..combined_train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = (indices.len() + batch_size - 1) / batch_size;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| combined_train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = new_batcher.batch(items, &device);
            let output = new_model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &new_model);
            new_model = new_optimizer.step(learning_rate, new_model, grads);
        }
    }

    // Evaluate on base classes only (to measure forgetting)
    let base_val_dataset =
        PlantVillageBurnDataset::new_cached(base_val, image_size).expect("Failed");
    let base_accuracy_after =
        evaluate::<B>(&new_model, &base_val_dataset, &new_batcher, batch_size, image_size);

    // Evaluate on new class only
    let new_class_val_dataset =
        PlantVillageBurnDataset::new_cached(new_class_val, image_size).expect("Failed");
    let new_class_accuracy =
        evaluate::<B>(&new_model, &new_class_val_dataset, &new_batcher, batch_size, image_size);

    // Overall accuracy
    let overall_accuracy =
        evaluate::<B>(&new_model, &combined_val_dataset, &new_batcher, batch_size, image_size);

    let forgetting = base_accuracy_before - base_accuracy_after;
    let training_time = start.elapsed().as_secs_f64();

    Ok(ScalingResult {
        base_classes,
        total_classes,
        base_accuracy_before,
        base_accuracy_after,
        new_class_accuracy,
        overall_accuracy,
        forgetting,
        training_time,
    })
}

/// Evaluate model on dataset
fn evaluate<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    dataset: &PlantVillageBurnDataset,
    _batcher: &PlantVillageBatcher<B>,
    batch_size: usize,
    image_size: usize,
) -> f64 {
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

    let device = <B::InnerBackend as Backend>::Device::default();
    let inner_batcher = PlantVillageBatcher::<B::InnerBackend>::with_image_size(device.clone(), image_size);

    let inner_model = model.clone().valid();
    let len = dataset.len();
    
    if len == 0 {
        return 0.0;
    }
    
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch = inner_batcher.batch(items, &device);
        let output = inner_model.forward(batch.images);
        
        // Get predictions - handle both single sample and batch cases
        let predictions = output.argmax(1);
        let [batch_dim, _] = predictions.dims();
        
        // Flatten predictions to 1D
        let predictions_flat: Tensor<B::InnerBackend, 1, burn::tensor::Int> = predictions.reshape([batch_dim]);
        
        let batch_correct: i64 = predictions_flat
            .equal(batch.targets)
            .int()
            .sum()
            .into_scalar()
            .elem();

        correct += batch_correct as usize;
        total += end - start;
    }

    if total == 0 {
        0.0
    } else {
        100.0 * correct as f64 / total as f64
    }
}

/// Generate conclusions for label efficiency experiment
fn generate_label_efficiency_conclusions(results: &LabelEfficiencyResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str(&format!(
        "========================================================================\n"
    ));
    text.push_str(&format!(
        "EXPERIMENT 1: Label Efficiency Curve - Conclusions\n"
    ));
    text.push_str(&format!(
        "Generated: {}\n",
        timestamp
    ));
    text.push_str(&format!(
        "========================================================================\n\n"
    ));

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str("How many labeled images per class are needed for acceptable accuracy?\n\n");

    text.push_str("RESULTS:\n");
    text.push_str(&format!("{:>12} | {:>12} | {:>12}\n", "Images/Class", "Accuracy (%)", "Time (s)"));
    text.push_str(&format!("{}\n", "-".repeat(42)));

    for i in 0..results.images_per_class.len() {
        text.push_str(&format!(
            "{:>12} | {:>12.2} | {:>12.1}\n",
            results.images_per_class[i], results.accuracies[i], results.training_times[i]
        ));
    }

    text.push_str("\nKEY FINDINGS:\n");
    text.push_str(&format!(
        "1. Best accuracy: {:.2}% with {} images per class\n",
        results.best_accuracy, results.best_images_per_class
    ));

    if let Some(min) = results.min_acceptable_images {
        text.push_str(&format!(
            "2. Minimum for >80% accuracy: {} images per class\n",
            min
        ));
    } else {
        text.push_str("2. No configuration achieved >80% accuracy\n");
    }

    // Calculate efficiency gain
    if results.accuracies.len() >= 2 {
        let first_acc = results.accuracies[0];
        let last_acc = results.accuracies[results.accuracies.len() - 1];
        let improvement = last_acc - first_acc;
        text.push_str(&format!(
            "3. Improvement from {} to {} images: {:.2}% points\n",
            results.images_per_class[0],
            results.images_per_class[results.images_per_class.len() - 1],
            improvement
        ));
    }

    text.push_str("\nCONCLUSIONS:\n");
    
    if let Some(min) = results.min_acceptable_images {
        if min <= 25 {
            text.push_str("• The model achieves acceptable accuracy with very few labeled samples.\n");
            text.push_str("• This indicates good potential for semi-supervised learning scenarios.\n");
        } else if min <= 50 {
            text.push_str("• A moderate number of labeled samples is needed for acceptable accuracy.\n");
            text.push_str("• Semi-supervised learning can help reduce this requirement.\n");
        } else {
            text.push_str("• Many labeled samples are needed for acceptable accuracy.\n");
            text.push_str("• SSL methods like pseudo-labeling are crucial for practical deployment.\n");
        }
    }

    text.push_str("\nPRACTICAL RECOMMENDATION:\n");
    if let Some(min) = results.min_acceptable_images {
        text.push_str(&format!(
            "For production deployment, collect at least {} labeled images per disease class.\n",
            min
        ));
    }
    text.push_str("Use semi-supervised learning to leverage unlabeled data for improved accuracy.\n");

    text
}

/// Generate conclusions for class scaling experiment
fn generate_class_scaling_conclusions(results: &ClassScalingResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str(&format!(
        "========================================================================\n"
    ));
    text.push_str(&format!(
        "EXPERIMENT 2: Class Scaling Effect - Conclusions\n"
    ));
    text.push_str(&format!(
        "Generated: {}\n",
        timestamp
    ));
    text.push_str(&format!(
        "========================================================================\n\n"
    ));

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str("Is adding a class to 5 classes harder than adding to 30 classes?\n");
    text.push_str("Does the model become more biased toward existing classes with a larger base?\n\n");

    text.push_str("RESULTS:\n\n");

    text.push_str(&format!("Scenario A: {} → {} classes\n", 
        results.small_base.base_classes, results.small_base.total_classes));
    text.push_str(&format!("  Base accuracy:      {:.2}% → {:.2}%\n", 
        results.small_base.base_accuracy_before, results.small_base.base_accuracy_after));
    text.push_str(&format!("  New class accuracy: {:.2}%\n", results.small_base.new_class_accuracy));
    text.push_str(&format!("  Overall accuracy:   {:.2}%\n", results.small_base.overall_accuracy));
    text.push_str(&format!("  Forgetting:         {:.2}% points\n", results.small_base.forgetting));
    text.push_str("\n");

    text.push_str(&format!("Scenario B: {} → {} classes\n", 
        results.large_base.base_classes, results.large_base.total_classes));
    text.push_str(&format!("  Base accuracy:      {:.2}% → {:.2}%\n", 
        results.large_base.base_accuracy_before, results.large_base.base_accuracy_after));
    text.push_str(&format!("  New class accuracy: {:.2}%\n", results.large_base.new_class_accuracy));
    text.push_str(&format!("  Overall accuracy:   {:.2}%\n", results.large_base.overall_accuracy));
    text.push_str(&format!("  Forgetting:         {:.2}% points\n", results.large_base.forgetting));
    text.push_str("\n");

    text.push_str("COMPARATIVE ANALYSIS:\n");
    text.push_str(&format!("  Relative difficulty (large/small forgetting): {:.2}x\n", 
        results.relative_difficulty));
    
    let new_class_diff = results.large_base.new_class_accuracy - results.small_base.new_class_accuracy;
    text.push_str(&format!("  New class accuracy difference: {:.2}% points\n", new_class_diff));

    text.push_str("\nKEY FINDINGS:\n");

    // Analyze forgetting
    if results.large_base.forgetting > results.small_base.forgetting {
        text.push_str(&format!(
            "1. Larger base (30 classes) shows MORE forgetting ({:.2}% vs {:.2}%)\n",
            results.large_base.forgetting, results.small_base.forgetting
        ));
        text.push_str("   → The model is more biased toward existing classes with a larger base.\n");
    } else if results.large_base.forgetting < results.small_base.forgetting {
        text.push_str(&format!(
            "1. Larger base (30 classes) shows LESS forgetting ({:.2}% vs {:.2}%)\n",
            results.large_base.forgetting, results.small_base.forgetting
        ));
        text.push_str("   → More diverse training may create more robust representations.\n");
    } else {
        text.push_str("1. Forgetting is similar regardless of base size.\n");
    }

    // Analyze new class learning
    if results.large_base.new_class_accuracy > results.small_base.new_class_accuracy {
        text.push_str(&format!(
            "2. New class learning is EASIER with larger base ({:.2}% vs {:.2}%)\n",
            results.large_base.new_class_accuracy, results.small_base.new_class_accuracy
        ));
        text.push_str("   → Transfer learning benefits from diverse prior knowledge.\n");
    } else {
        text.push_str(&format!(
            "2. New class learning is HARDER with larger base ({:.2}% vs {:.2}%)\n",
            results.large_base.new_class_accuracy, results.small_base.new_class_accuracy
        ));
        text.push_str("   → Class competition increases with more existing classes.\n");
    }

    text.push_str("\nCONCLUSIONS:\n");
    
    if results.relative_difficulty > 1.5 {
        text.push_str("• Adding classes to larger models requires careful management.\n");
        text.push_str("• Consider using incremental learning methods (LwF, EWC, Rehearsal).\n");
    } else if results.relative_difficulty < 0.7 {
        text.push_str("• Larger models may actually be more stable when adding classes.\n");
        text.push_str("• The diverse feature space helps accommodate new classes.\n");
    } else {
        text.push_str("• Class scaling has moderate impact on forgetting.\n");
        text.push_str("• Standard fine-tuning may be sufficient for incremental updates.\n");
    }

    text.push_str("\nPRACTICAL RECOMMENDATIONS:\n");
    text.push_str("• For production systems, start with a comprehensive base model.\n");
    text.push_str("• Use incremental learning methods when adding new disease classes.\n");
    text.push_str("• Monitor accuracy on existing classes after each update.\n");
    text.push_str("• Consider rehearsal-based methods to maintain performance on old classes.\n");

    text
}

/// Generate conclusions for new class position experiment
fn generate_new_class_position_conclusions(results: &NewClassPositionResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str(&format!(
        "========================================================================\n"
    ));
    text.push_str(&format!(
        "EXPERIMENT 4: New Class Position Effect - Conclusions\n"
    ));
    text.push_str(&format!(
        "Generated: {}\n",
        timestamp
    ));
    text.push_str(&format!(
        "========================================================================\n\n"
    ));

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str("Does adding a class as the 6th class (small base) require different amounts of\n");
    text.push_str("labeled samples compared to adding as the 31st class (large base)?\n\n");

    text.push_str("RESULTS:\n\n");

    // Table header
    text.push_str(&format!("{:>8} | {:>12} | {:>12} | {:>12} | {:>12}\n", 
        "Labels", "6th Class", "31st Class", "Difference", "Ratio"));
    text.push_str(&format!("{}\n", "-".repeat(68)));

    // Compare results at each label count
    for (small, large) in results.small_base_results.iter().zip(results.large_base_results.iter()) {
        let diff = large.new_class_accuracy - small.new_class_accuracy;
        let ratio = if small.new_class_accuracy > 0.0 {
            large.new_class_accuracy / small.new_class_accuracy
        } else {
            0.0
        };
        text.push_str(&format!(
            "{:>8} | {:>11.2}% | {:>11.2}% | {:>+11.2}% | {:>11.2}x\n",
            small.labeled_samples,
            small.new_class_accuracy,
            large.new_class_accuracy,
            diff,
            ratio
        ));
    }

    text.push_str("\nFORGETTING ANALYSIS:\n\n");
    text.push_str(&format!("{:>8} | {:>12} | {:>12} | {:>12}\n", 
        "Labels", "5→6 Forget", "30→31 Forget", "Difference"));
    text.push_str(&format!("{}\n", "-".repeat(56)));

    for (small, large) in results.small_base_results.iter().zip(results.large_base_results.iter()) {
        let diff = large.forgetting - small.forgetting;
        text.push_str(&format!(
            "{:>8} | {:>11.2}% | {:>11.2}% | {:>+11.2}%\n",
            small.labeled_samples,
            small.forgetting,
            large.forgetting,
            diff
        ));
    }

    text.push_str("\nKEY FINDINGS:\n");

    // Finding 1: Minimum samples needed
    if let Some(min_small) = results.summary.min_samples_small_70pct {
        text.push_str(&format!(
            "1. Minimum samples for 70% accuracy (6th class): {} images\n",
            min_small
        ));
    } else {
        text.push_str("1. 70% accuracy not achieved for 6th class with tested sample counts\n");
    }

    if let Some(min_large) = results.summary.min_samples_large_70pct {
        text.push_str(&format!(
            "   Minimum samples for 70% accuracy (31st class): {} images\n",
            min_large
        ));
    } else {
        text.push_str("   70% accuracy not achieved for 31st class with tested sample counts\n");
    }

    // Finding 2: Is it harder?
    if results.summary.harder_as_31st {
        text.push_str(&format!(
            "2. Learning a new class is HARDER as the 31st class\n"
        ));
        text.push_str(&format!(
            "   Samples ratio: {:.2}x more samples needed for equivalent accuracy\n",
            results.summary.samples_ratio
        ));
    } else {
        text.push_str("2. Learning a new class is NOT harder as the 31st class\n");
        text.push_str("   The larger feature space may actually help generalization\n");
    }

    // Finding 3: Forgetting
    if results.summary.avg_forgetting_difference > 1.0 {
        text.push_str(&format!(
            "3. Larger base shows MORE forgetting ({:+.2}% difference on average)\n",
            results.summary.avg_forgetting_difference
        ));
        text.push_str("   This confirms class competition increases with more classes\n");
    } else if results.summary.avg_forgetting_difference < -1.0 {
        text.push_str(&format!(
            "3. Larger base shows LESS forgetting ({:+.2}% difference on average)\n",
            results.summary.avg_forgetting_difference
        ));
        text.push_str("   More diverse representations may be more stable\n");
    } else {
        text.push_str("3. Forgetting is similar regardless of base size\n");
    }

    text.push_str("\nCONCLUSIONS:\n");

    if results.summary.harder_as_31st && results.summary.samples_ratio > 1.5 {
        text.push_str("• As models accumulate more classes, new class learning becomes harder\n");
        text.push_str("• Consider collecting MORE labeled samples when adding to large models\n");
        text.push_str(&format!(
            "• Recommendation: Collect ~{:.0}x more samples for mature models\n",
            results.summary.samples_ratio.ceil()
        ));
    } else {
        text.push_str("• Class position has limited impact on learning difficulty\n");
        text.push_str("• The same labeling effort works for both early and late classes\n");
    }

    if results.summary.avg_forgetting_difference > 2.0 {
        text.push_str("• Use incremental learning methods (LwF, EWC) for larger models\n");
        text.push_str("• Rehearsal is especially important when base > 20 classes\n");
    }

    text.push_str("\nPRACTICAL RECOMMENDATIONS:\n");
    
    if let Some(min_70) = results.summary.min_samples_small_70pct {
        text.push_str(&format!(
            "• For new deployments (few classes): Collect at least {} labeled samples\n",
            min_70
        ));
    }
    
    if let Some(min_70) = results.summary.min_samples_large_70pct {
        text.push_str(&format!(
            "• For mature systems (many classes): Collect at least {} labeled samples\n",
            min_70
        ));
    }
    
    text.push_str("• Use SSL pseudo-labeling to augment limited labeled data\n");
    text.push_str("• Monitor forgetting on existing classes after each update\n");

    text
}

/// Generate SVG chart for label efficiency results
fn generate_label_efficiency_chart(results: &LabelEfficiencyResults, output_dir: &str) -> Result<()> {
    use plantvillage_ssl::utils::charts::{DataPoint, DataSeries, generate_line_chart, generate_bar_chart, BarData};

    // Line chart: accuracy vs images per class
    let series = vec![DataSeries {
        name: "Accuracy".to_string(),
        points: results
            .images_per_class
            .iter()
            .zip(results.accuracies.iter())
            .map(|(&x, &y)| DataPoint {
                x: x as f64,
                y,
                label: Some(format!("{:.1}%", y)),
            })
            .collect(),
        color: "#3498db".to_string(),
    }];

    let chart_path = Path::new(output_dir).join("label_efficiency_curve.svg");
    generate_line_chart(
        "Label Efficiency: Accuracy vs Images per Class",
        "Images per Class",
        "Validation Accuracy (%)",
        &series,
        &chart_path,
    )?;
    println!("Chart saved to: {:?}", chart_path);

    // Bar chart version
    let bars: Vec<BarData> = results
        .images_per_class
        .iter()
        .zip(results.accuracies.iter())
        .map(|(&images, &acc)| BarData {
            label: format!("{}", images),
            value: acc,
            color: if acc >= 80.0 { "#2ecc71".to_string() } else { "#3498db".to_string() },
        })
        .collect();

    let bar_chart_path = Path::new(output_dir).join("label_efficiency_bars.svg");
    generate_bar_chart(
        "Accuracy by Number of Labeled Images",
        "Validation Accuracy (%)",
        &bars,
        &bar_chart_path,
    )?;
    println!("Bar chart saved to: {:?}", bar_chart_path);

    Ok(())
}

/// Generate SVG chart for class scaling results
fn generate_class_scaling_chart(results: &ClassScalingResults, output_dir: &str) -> Result<()> {
    use plantvillage_ssl::utils::charts::generate_comparison_chart;

    let groups = vec![
        (
            "5 → 6 Classes",
            vec![
                ("Base Before", results.small_base.base_accuracy_before, "#3498db"),
                ("Base After", results.small_base.base_accuracy_after, "#2ecc71"),
                ("New Class", results.small_base.new_class_accuracy, "#9b59b6"),
            ],
        ),
        (
            "30 → 31 Classes",
            vec![
                ("Base Before", results.large_base.base_accuracy_before, "#3498db"),
                ("Base After", results.large_base.base_accuracy_after, "#2ecc71"),
                ("New Class", results.large_base.new_class_accuracy, "#9b59b6"),
            ],
        ),
    ];

    let chart_path = Path::new(output_dir).join("class_scaling_comparison.svg");
    generate_comparison_chart(
        "Class Scaling: 5→6 vs 30→31 Classes",
        &groups,
        &chart_path,
    )?;
    println!("Chart saved to: {:?}", chart_path);

    Ok(())
}

/// Generate SVG charts for new class position experiment
fn generate_new_class_position_charts(results: &NewClassPositionResults, output_dir: &str) -> Result<()> {
    use plantvillage_ssl::utils::charts::{DataPoint, DataSeries, generate_line_chart, BarData};

    // Line chart: New class accuracy vs labeled samples for both positions
    let series = vec![
        DataSeries {
            name: "6th Class (5 base)".to_string(),
            points: results
                .small_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.new_class_accuracy,
                    label: Some(format!("{:.1}%", r.new_class_accuracy)),
                })
                .collect(),
            color: "#3498db".to_string(),
        },
        DataSeries {
            name: "31st Class (30 base)".to_string(),
            points: results
                .large_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.new_class_accuracy,
                    label: Some(format!("{:.1}%", r.new_class_accuracy)),
                })
                .collect(),
            color: "#e74c3c".to_string(),
        },
    ];

    let chart_path = Path::new(output_dir).join("new_class_accuracy_curve.svg");
    generate_line_chart(
        "New Class Accuracy: 6th vs 31st Class Position",
        "Labeled Samples",
        "New Class Accuracy (%)",
        &series,
        &chart_path,
    )?;
    println!("New class accuracy chart saved to: {:?}", chart_path);

    // Line chart: Forgetting vs labeled samples
    let forgetting_series = vec![
        DataSeries {
            name: "5→6 Forgetting".to_string(),
            points: results
                .small_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.forgetting,
                    label: Some(format!("{:.1}%", r.forgetting)),
                })
                .collect(),
            color: "#2ecc71".to_string(),
        },
        DataSeries {
            name: "30→31 Forgetting".to_string(),
            points: results
                .large_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.forgetting,
                    label: Some(format!("{:.1}%", r.forgetting)),
                })
                .collect(),
            color: "#9b59b6".to_string(),
        },
    ];

    let forgetting_path = Path::new(output_dir).join("forgetting_curve.svg");
    generate_line_chart(
        "Forgetting: 5→6 vs 30→31 Classes",
        "Labeled Samples",
        "Forgetting (%)",
        &forgetting_series,
        &forgetting_path,
    )?;
    println!("Forgetting chart saved to: {:?}", forgetting_path);

    // Generate comparison bar chart at 50 samples
    let fifty_sample_small = results.small_base_results.iter().find(|r| r.labeled_samples == 50);
    let fifty_sample_large = results.large_base_results.iter().find(|r| r.labeled_samples == 50);

    if let (Some(small), Some(large)) = (fifty_sample_small, fifty_sample_large) {
        let comparison_path = Path::new(output_dir).join("position_comparison_50.svg");
        generate_position_comparison_chart(
            "New Class Position Effect (50 labeled samples)",
            small,
            large,
            &comparison_path,
        )?;
        println!("Position comparison chart saved to: {:?}", comparison_path);
    }

    Ok(())
}

/// Generate a comparison chart for position experiment
fn generate_position_comparison_chart(
    title: &str,
    small: &PositionLabelResult,
    large: &PositionLabelResult,
    output_path: &Path,
) -> std::io::Result<()> {
    const CHART_WIDTH: f64 = 900.0;
    const CHART_HEIGHT: f64 = 500.0;
    const MARGIN_TOP: f64 = 70.0;
    const MARGIN_RIGHT: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 100.0;
    const MARGIN_LEFT: f64 = 80.0;
    const COLOR_GRID: &str = "#ecf0f1";
    const COLOR_TEXT: &str = "#2c3e50";

    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let metrics = vec![
        ("New Class Acc", small.new_class_accuracy, large.new_class_accuracy, "#3498db", "#e74c3c"),
        ("Base After", small.base_accuracy_after, large.base_accuracy_after, "#2ecc71", "#27ae60"),
        ("Overall", small.overall_accuracy, large.overall_accuracy, "#9b59b6", "#8e44ad"),
        ("Forgetting", small.forgetting, large.forgetting, "#f39c12", "#d35400"),
    ];

    let y_max = 100.0;
    let group_width = plot_width / metrics.len() as f64;
    let bar_width = group_width * 0.35;

    let mut svg = String::new();

    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, title
    ));

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = (i as f64 / 5.0) * y_max;

        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.0}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Bars
    for (i, (name, small_val, large_val, small_color, large_color)) in metrics.iter().enumerate() {
        let group_x = MARGIN_LEFT + (i as f64 * group_width) + group_width * 0.1;

        // Small base bar
        let bar_height = (small_val.abs() / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            group_x, y, bar_width, bar_height.max(2.0), small_color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{}">{:.1}%</text>"#,
            group_x + bar_width / 2.0, y - 5.0, COLOR_TEXT, small_val
        ));

        // Large base bar
        let bar_height = (large_val.abs() / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            group_x + bar_width + 5.0, y, bar_width, bar_height.max(2.0), large_color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{}">{:.1}%</text>"#,
            group_x + bar_width * 1.5 + 5.0, y - 5.0, COLOR_TEXT, large_val
        ));

        // Label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
            group_x + bar_width + 2.5, CHART_HEIGHT - 60.0, COLOR_TEXT, name
        ));
    }

    // Legend
    let legend_y = CHART_HEIGHT - 30.0;
    svg.push_str(&format!(
        r##"<rect x="{}" y="{}" width="15" height="15" fill="#3498db" rx="2"/>"##,
        CHART_WIDTH / 2.0 - 120.0, legend_y - 12.0
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">6th Class (5 base)</text>"#,
        CHART_WIDTH / 2.0 - 100.0, legend_y, COLOR_TEXT
    ));
    svg.push_str(&format!(
        r##"<rect x="{}" y="{}" width="15" height="15" fill="#e74c3c" rx="2"/>"##,
        CHART_WIDTH / 2.0 + 50.0, legend_y - 12.0
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">31st Class (30 base)</text>"#,
        CHART_WIDTH / 2.0 + 70.0, legend_y, COLOR_TEXT
    ));

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

/// Run SSL + Incremental Learning combined experiment
fn run_ssl_incremental(
    data_dir: &str,
    output_dir: &str,
    base_classes: usize,
    labeled_samples: usize,
    confidence_threshold: f64,
    seed: u64,
) -> Result<()> {
    use plantvillage_ssl::training::ssl_incremental::{
        SSLIncrementalConfig, run_ssl_incremental_experiment, generate_ssl_incremental_conclusions
    };

    println!("{}", "EXPERIMENT 3: SSL + Incremental Learning".yellow().bold());
    println!("Question: Can pseudo-labeling reduce labeled samples needed for new classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Group samples by class
    let mut samples_by_class: HashMap<usize, Vec<(PathBuf, usize)>> = HashMap::new();
    for sample in &dataset.samples {
        samples_by_class
            .entry(sample.label)
            .or_default()
            .push((sample.path.clone(), sample.label));
    }

    let config = SSLIncrementalConfig {
        base_classes,
        labeled_samples_per_new_class: labeled_samples,
        confidence_threshold,
        max_pseudo_labels_per_class: 100,
        base_epochs: 30,
        incremental_epochs: 20,
        batch_size: 32,
        learning_rate: 0.0001,
        seed,
        use_distillation: true,
        distillation_temperature: 2.0,
        distillation_lambda: 1.0,
    };

    println!("{}", "Configuration:".cyan());
    println!("  Base classes: {}", config.base_classes);
    println!("  Labeled samples per new class: {}", config.labeled_samples_per_new_class);
    println!("  Confidence threshold: {}", config.confidence_threshold);
    println!();

    println!("{}", "Running SSL+IL experiment...".yellow().bold());
    let results = run_ssl_incremental_experiment::<Backend>(samples_by_class, config)?;

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_ssl_incremental_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG charts
    generate_ssl_incremental_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Generate SVG charts for SSL+IL experiment results
fn generate_ssl_incremental_chart(
    results: &plantvillage_ssl::training::ssl_incremental::SSLIncrementalResults,
    output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::utils::charts::{generate_comparison_chart, BarData};

    // Comparison chart: Without SSL vs With SSL
    let groups = vec![
        (
            "Without SSL",
            vec![
                ("Old Classes", results.without_ssl.old_class_accuracy, "#3498db"),
                ("New Class", results.without_ssl.new_class_accuracy, "#e74c3c"),
                ("Overall", results.without_ssl.overall_accuracy, "#9b59b6"),
            ],
        ),
        (
            "With SSL",
            vec![
                ("Old Classes", results.with_ssl.old_class_accuracy, "#3498db"),
                ("New Class", results.with_ssl.new_class_accuracy, "#2ecc71"),
                ("Overall", results.with_ssl.overall_accuracy, "#9b59b6"),
            ],
        ),
    ];

    let comparison_path = Path::new(output_dir).join("ssl_incremental_comparison.svg");
    generate_comparison_chart(
        "SSL Impact: Incremental Learning Accuracy",
        &groups,
        &comparison_path,
    )?;
    println!("Comparison chart saved to: {:?}", comparison_path);

    // Improvement bar chart
    let improvement_bars = vec![
        BarData {
            label: "Old Class".to_string(),
            value: results.with_ssl.old_class_accuracy - results.without_ssl.old_class_accuracy,
            color: if results.with_ssl.old_class_accuracy >= results.without_ssl.old_class_accuracy {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
        BarData {
            label: "New Class".to_string(),
            value: results.ssl_improvement,
            color: if results.ssl_improvement >= 0.0 {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
        BarData {
            label: "Overall".to_string(),
            value: results.with_ssl.overall_accuracy - results.without_ssl.overall_accuracy,
            color: if results.with_ssl.overall_accuracy >= results.without_ssl.overall_accuracy {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
    ];

    // Generate improvement chart
    let improvement_path = Path::new(output_dir).join("ssl_improvement.svg");
    generate_improvement_chart(
        "SSL Improvement over Baseline (% points)",
        &improvement_bars,
        &improvement_path,
    )?;
    println!("Improvement chart saved to: {:?}", improvement_path);

    Ok(())
}

/// Generate a bar chart for showing improvements (can handle any positive values)
fn generate_improvement_chart(
    title: &str,
    bars: &[plantvillage_ssl::utils::charts::BarData],
    output_path: &Path,
) -> std::io::Result<()> {
    const CHART_WIDTH: f64 = 800.0;
    const CHART_HEIGHT: f64 = 500.0;
    const MARGIN_TOP: f64 = 60.0;
    const MARGIN_RIGHT: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 80.0;
    const MARGIN_LEFT: f64 = 80.0;
    const COLOR_GRID: &str = "#ecf0f1";
    const COLOR_AXIS: &str = "#2c3e50";
    const COLOR_TEXT: &str = "#2c3e50";

    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let y_min = bars.iter().map(|b| b.value).fold(0.0f64, f64::min).min(0.0);
    let y_max = bars.iter().map(|b| b.value).fold(0.0f64, f64::max).max(10.0);
    let y_range = (y_max - y_min).max(1.0);
    let y_min_padded = y_min - y_range * 0.1;
    let y_max_padded = y_max + y_range * 0.1;

    let bar_width = (plot_width / bars.len() as f64) * 0.7;
    let bar_gap = (plot_width / bars.len() as f64) * 0.3;

    let mut svg = String::new();

    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, title
    ));

    let zero_y = MARGIN_TOP + plot_height - ((0.0 - y_min_padded) / (y_max_padded - y_min_padded)) * plot_height;

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = y_min_padded + (i as f64 / 5.0) * (y_max_padded - y_min_padded);

        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:+.1}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Zero line
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, zero_y, MARGIN_LEFT + plot_width, zero_y, COLOR_AXIS
    ));

    // Bars
    for (i, bar) in bars.iter().enumerate() {
        let x = MARGIN_LEFT + (i as f64 * (bar_width + bar_gap)) + bar_gap / 2.0;
        let bar_height = (bar.value.abs() / (y_max_padded - y_min_padded)) * plot_height;

        let y = if bar.value >= 0.0 {
            zero_y - bar_height
        } else {
            zero_y
        };

        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            x, y, bar_width, bar_height.max(2.0), bar.color
        ));

        let label_y = if bar.value >= 0.0 { y - 8.0 } else { y + bar_height + 18.0 };
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{}">{:+.1}%</text>"#,
            x + bar_width / 2.0, label_y, COLOR_TEXT, bar.value
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
            x + bar_width / 2.0, CHART_HEIGHT - 25.0, COLOR_TEXT, bar.label
        ));
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

/// Run inference benchmark with optional Jetson power monitoring
fn run_inference_benchmark(
    model_path: Option<&str>,
    output_dir: &str,
    warmup: usize,
    iterations: usize,
    batch_size: usize,
    image_size: usize,
) -> Result<()> {
    use plantvillage_ssl::inference::{BenchmarkConfig, run_benchmark, is_jetson};
    use plantvillage_ssl::inference::jetson::JetsonDeviceInfo;
    use burn::tensor::Tensor;
    use burn_cuda::Cuda;

    println!("{}", "BENCHMARK: Inference Performance".yellow().bold());
    println!();

    fs::create_dir_all(output_dir)?;

    // Check if running on Jetson
    let on_jetson = is_jetson();
    if on_jetson {
        println!("{}", "Detected Jetson device - enabling power monitoring".green());
        if let Some(info) = JetsonDeviceInfo::detect() {
            println!("  Model: {}", info.jetson_model);
            if let Some(ref mode) = info.power_mode {
                println!("  Power Mode: {}", mode);
            }
            if let Some(freq) = info.gpu_freq_mhz {
                println!("  GPU Frequency: {} MHz", freq);
            }
        }
    } else {
        println!("{}", "Not running on Jetson - power monitoring disabled".yellow());
    }
    println!();

    let config = BenchmarkConfig {
        warmup_iterations: warmup,
        iterations,
        batch_size,
        measure_memory: true,
        verbose: false,
        output_path: Some(Path::new(output_dir).join("benchmark_results.json")),
    };

    let device = <Cuda as burn::tensor::backend::Backend>::Device::default();
    let model_path_ref = model_path.map(Path::new);

    let result = run_benchmark::<Cuda>(config.clone(), model_path_ref, image_size, &device)?;

    // Save standard results
    let json = serde_json::to_string_pretty(&result)?;
    let results_path = Path::new(output_dir).join("benchmark_results.json");
    fs::write(&results_path, &json)?;
    println!("\nResults saved to: {:?}", results_path);

    // Generate summary for research paper
    let summary = generate_benchmark_summary(&result, on_jetson);
    let summary_path = Path::new(output_dir).join("benchmark_summary.txt");
    fs::write(&summary_path, &summary)?;
    println!("Summary saved to: {:?}", summary_path);

    // Generate chart
    generate_benchmark_chart(&result, output_dir)?;

    println!();
    println!("{}", "BENCHMARK COMPLETE".green().bold());

    Ok(())
}

/// Generate benchmark summary for research paper
fn generate_benchmark_summary(result: &plantvillage_ssl::inference::BenchmarkOutput, on_jetson: bool) -> String {
    let mut summary = String::new();

    summary.push_str("========================================================================\n");
    summary.push_str("INFERENCE BENCHMARK RESULTS\n");
    summary.push_str("========================================================================\n\n");

    summary.push_str(&format!("Framework: {}\n", result.framework));
    summary.push_str(&format!("Device: {}\n", result.device));
    summary.push_str(&format!("Timestamp: {}\n", result.timestamp));
    if on_jetson {
        summary.push_str("Platform: NVIDIA Jetson\n");
    }
    summary.push_str("\n");

    summary.push_str("CONFIGURATION:\n");
    summary.push_str(&format!("  Image Size: {}x{}\n", result.image_size, result.image_size));
    summary.push_str(&format!("  Batch Size: {}\n", result.batch_size));
    summary.push_str(&format!("  Warmup Iterations: {}\n", result.warmup_iterations));
    summary.push_str(&format!("  Benchmark Iterations: {}\n", result.num_iterations));
    summary.push_str("\n");

    summary.push_str("LATENCY METRICS:\n");
    summary.push_str(&format!("  Mean:   {:8.2} ms\n", result.mean_ms));
    summary.push_str(&format!("  Std:    {:8.2} ms\n", result.std_ms));
    summary.push_str(&format!("  Min:    {:8.2} ms\n", result.min_ms));
    summary.push_str(&format!("  Max:    {:8.2} ms\n", result.max_ms));
    summary.push_str(&format!("  P50:    {:8.2} ms\n", result.p50_ms));
    summary.push_str(&format!("  P95:    {:8.2} ms\n", result.p95_ms));
    summary.push_str(&format!("  P99:    {:8.2} ms\n", result.p99_ms));
    summary.push_str("\n");

    summary.push_str("THROUGHPUT:\n");
    summary.push_str(&format!("  {:8.1} images/second\n", result.throughput_fps));
    summary.push_str("\n");

    if result.model_size_mb > 0.0 {
        summary.push_str("MODEL:\n");
        summary.push_str(&format!("  Size: {:.2} MB\n", result.model_size_mb));
        summary.push_str("\n");
    }

    // Check against targets
    summary.push_str("TARGET COMPLIANCE:\n");
    let target_200ms = result.mean_ms <= 200.0;
    let target_500ms = result.mean_ms <= 500.0;
    summary.push_str(&format!("  200ms target: {}\n", if target_200ms { "PASS" } else { "FAIL" }));
    summary.push_str(&format!("  500ms target: {}\n", if target_500ms { "PASS" } else { "FAIL" }));

    summary
}

/// Generate benchmark visualization chart
fn generate_benchmark_chart(result: &plantvillage_ssl::inference::BenchmarkOutput, output_dir: &str) -> Result<()> {
    use plantvillage_ssl::utils::charts::{generate_bar_chart, BarData};

    let bars = vec![
        BarData {
            label: "Mean".to_string(),
            value: result.mean_ms,
            color: "#3498db".to_string(),
        },
        BarData {
            label: "P50".to_string(),
            value: result.p50_ms,
            color: "#2ecc71".to_string(),
        },
        BarData {
            label: "P95".to_string(),
            value: result.p95_ms,
            color: "#f39c12".to_string(),
        },
        BarData {
            label: "P99".to_string(),
            value: result.p99_ms,
            color: "#e74c3c".to_string(),
        },
    ];

    // We need a custom chart since this isn't percentages
    let chart_path = Path::new(output_dir).join("latency_distribution.svg");
    generate_latency_chart(
        &format!("Inference Latency ({} iterations)", result.num_iterations),
        "Latency (ms)",
        &bars,
        &chart_path,
    )?;
    println!("Latency chart saved to: {:?}", chart_path);

    Ok(())
}

/// Generate a bar chart for latency metrics (in ms, not %)
fn generate_latency_chart(
    title: &str,
    y_label: &str,
    bars: &[plantvillage_ssl::utils::charts::BarData],
    output_path: &Path,
) -> std::io::Result<()> {
    const CHART_WIDTH: f64 = 800.0;
    const CHART_HEIGHT: f64 = 500.0;
    const MARGIN_TOP: f64 = 60.0;
    const MARGIN_RIGHT: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 80.0;
    const MARGIN_LEFT: f64 = 80.0;
    const COLOR_GRID: &str = "#ecf0f1";
    const COLOR_AXIS: &str = "#2c3e50";
    const COLOR_TEXT: &str = "#2c3e50";

    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let y_max = bars.iter().map(|b| b.value).fold(0.0f64, f64::max) * 1.2;

    let bar_width = (plot_width / bars.len() as f64) * 0.7;
    let bar_gap = (plot_width / bars.len() as f64) * 0.3;

    let mut svg = String::new();

    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, title
    ));

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = (i as f64 / 5.0) * y_max;

        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.1} ms</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, MARGIN_TOP + plot_height, MARGIN_LEFT + plot_width, MARGIN_TOP + plot_height, COLOR_AXIS
    ));

    // Y-axis label
    svg.push_str(&format!(
        r#"<text x="20" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}" transform="rotate(-90 20 {})">{}</text>"#,
        CHART_HEIGHT / 2.0, COLOR_TEXT, CHART_HEIGHT / 2.0, y_label
    ));

    // Bars
    for (i, bar) in bars.iter().enumerate() {
        let x = MARGIN_LEFT + (i as f64 * (bar_width + bar_gap)) + bar_gap / 2.0;
        let bar_height = (bar.value / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;

        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            x, y, bar_width, bar_height, bar.color
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="{}">{:.1} ms</text>"#,
            x + bar_width / 2.0, y - 8.0, COLOR_TEXT, bar.value
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{}">{}</text>"#,
            x + bar_width / 2.0, MARGIN_TOP + plot_height + 25.0, COLOR_TEXT, bar.label
        ));
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}
