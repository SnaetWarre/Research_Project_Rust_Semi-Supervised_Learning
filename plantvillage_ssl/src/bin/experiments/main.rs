//! Experiment Runner for Research Project
//!
//! This binary runs the key experiments for the research project:
//! - `label_efficiency`: how many images per class are needed?
//! - `class_scaling`: is adding a 6th class harder than adding a 31st?
//! - `new_class_position`: the same, with a varying number of labeled samples
//! - `ssl_incremental`: pseudo-labeling combined with adding a new class
//! - `benchmark`: inference latency and throughput
//!
//! `incremental` holds the training routine shared by the class experiments.
//! Results are saved to output/experiments/ with a summary in conclusions.txt.

mod benchmark;
mod class_scaling;
mod incremental;
mod label_efficiency;
mod new_class_position;
mod ssl_incremental;

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;

use plantvillage_ssl::backend::TrainingBackend;

use benchmark::run_inference_benchmark;
use class_scaling::run_class_scaling;
use label_efficiency::run_label_efficiency;
use new_class_position::run_new_class_position;
use ssl_incremental::run_ssl_incremental;

type Backend = TrainingBackend;

static EXPERIMENT_VERBOSE: AtomicBool = AtomicBool::new(false);

#[inline]
fn experiment_verbose() -> bool {
    EXPERIMENT_VERBOSE.load(Ordering::Relaxed)
}

/// Experiment Runner CLI
#[derive(Parser, Debug)]
#[command(name = "experiments")]
#[command(about = "Run research experiments for plant disease classification")]
struct Cli {
    /// Log each training epoch (loss, batch counts) while experiments run
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run label efficiency experiment: How many images per new class are needed?
    LabelEfficiency {
        /// Path to the dataset directory
        #[arg(short, long, default_value = "data/plantvillage")]
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
        #[arg(short, long, default_value = "data/plantvillage")]
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
        #[arg(short, long, default_value = "data/plantvillage")]
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
        #[arg(short, long, default_value = "data/plantvillage")]
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
        #[arg(short, long, default_value = "data/plantvillage")]
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

    /// Run inference benchmark
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    EXPERIMENT_VERBOSE.store(cli.verbose, Ordering::Relaxed);

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
            run_ssl_incremental(
                &data_dir,
                &output_dir,
                base_classes,
                labeled_samples,
                confidence_threshold,
                seed,
            )?;
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
            run_inference_benchmark(
                model_path.as_deref(),
                &output_dir,
                warmup,
                iterations,
                batch_size,
                image_size,
            )?;
        }
    }

    Ok(())
}

fn print_banner() {
    println!(
        "{}",
        r#"
╔══════════════════════════════════════════════════════════════════╗
║    Plant Disease Classification - Experiment Runner               ║
║   Research Project: Semi-Supervised Learning on Edge Devices        ║
╚══════════════════════════════════════════════════════════════════╝
  "#
        .cyan()
    );
}
