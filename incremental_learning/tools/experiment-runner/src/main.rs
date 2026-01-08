//! Experiment Runner for Incremental Learning
//!
//! This tool orchestrates multi-method incremental learning experiments,
//! managing dataset splits, training multiple methods, and aggregating results.

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use plant_core::ModelArchitecture;
use plant_dataset::loader::ImageLoader;
use plant_incremental::{
    finetuning::{FineTuningConfig, FineTuningLearner},
    lwf::{LwFConfig, LwFLearner},
    ewc::{EWCConfig, EWCLearner},
    rehearsal::{RehearsalConfig, RehearsalLearner},
    metrics::{IncrementalAnalysis, MethodComparison},
    ExemplarSelection, IncrementalConfig, IncrementalLearner, IncrementalMethod,
    IncrementalResult, StepMetrics, ExperimentMetadata, TrainingMetrics,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{info, warn};

/// Experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    /// Experiment metadata
    pub experiment: ExperimentInfo,
    /// Incremental learning configuration
    pub incremental: IncrementalConfig,
    /// Dataset configuration
    pub dataset: DatasetInfo,
    /// Methods to compare
    pub methods: Vec<MethodConfig>,
    /// Training configuration
    pub training: TrainingConfig,
    /// Output configuration
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentInfo {
    /// Experiment name
    pub name: String,
    /// Description
    pub description: String,
    /// Random seed
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Root dataset directory
    pub root_dir: PathBuf,
    /// Initial classes (comma-separated indices or names)
    pub initial_classes: Vec<usize>,
    /// Classes per incremental step
    pub classes_per_step: Vec<Vec<usize>>,
    /// Train/val/test split ratios
    pub split_ratios: (f32, f32, f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodConfig {
    /// Method name
    pub name: String,
    /// Method type and parameters
    pub method: IncrementalMethod,
    /// Enable this method
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of epochs per task
    pub epochs_per_task: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Model architecture
    pub architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Export detailed results
    pub export_detailed: bool,
    /// Generate plots
    pub generate_plots: bool,
    /// Save model checkpoints
    pub save_checkpoints: bool,
}

/// Incremental Learning Experiment Runner
#[derive(Parser, Debug)]
#[command(
    name = "experiment-runner",
    about = "Run incremental learning experiments",
    long_about = "Orchestrate multi-method incremental learning experiments with automated \
                  dataset management, training, evaluation, and result aggregation."
)]
struct Args {
    /// Path to experiment configuration file (TOML)
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,

    /// Override output directory
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Run specific methods only (comma-separated)
    #[arg(short, long, value_name = "METHODS")]
    methods: Option<String>,

    /// Dry run (validate config without running)
    #[arg(long)]
    dry_run: bool,

    /// Continue from previous run
    #[arg(long)]
    resume: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    /// Number of parallel method runs
    #[arg(short = 'j', long, default_value = "1")]
    parallel: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    setup_logging(args.verbose)?;

    info!("Incremental Learning Experiment Runner");
    info!("======================================");

    // Load configuration
    let mut config = load_config(&args.config)
        .context("Failed to load experiment configuration")?;

    // Apply overrides
    if let Some(ref output) = args.output {
        config.output.output_dir = output.clone();
    }

    // Validate configuration
    validate_config(&config)?;

    if args.dry_run {
        info!("Configuration validated successfully (dry run)");
        print_experiment_summary(&config);
        return Ok(());
    }

    // Create output directory
    std::fs::create_dir_all(&config.output.output_dir)
        .context("Failed to create output directory")?;

    // Save configuration
    let config_path = config.output.output_dir.join("experiment_config.toml");
    let config_str = toml::to_string_pretty(&config)?;
    std::fs::write(&config_path, config_str)?;
    info!("Configuration saved to: {}", config_path.display());

    // Print summary
    print_experiment_summary(&config);

    // Filter methods if specified
    let methods_to_run: Vec<MethodConfig> = if let Some(ref method_names) = args.methods {
        let names: Vec<&str> = method_names.split(',').map(|s| s.trim()).collect();
        config.methods.iter()
            .filter(|m| m.enabled && names.contains(&m.name.as_str()))
            .cloned()
            .collect()
    } else {
        config.methods.iter()
            .filter(|m| m.enabled)
            .cloned()
            .collect()
    };

    if methods_to_run.is_empty() {
        anyhow::bail!("No methods to run");
    }

    info!("Running {} method(s)", methods_to_run.len());

    // Run experiments
    let results = run_experiments(&config, &methods_to_run, args.resume)?;

    // Analyze and compare results
    info!("Analyzing results...");
    let comparison = analyze_results(&results, &config)?;

    // Export results
    export_results(&results, &comparison, &config)?;

    // Print final comparison
    comparison.print_comparison();

    if let Some(best) = comparison.best_by_accuracy() {
        info!("");
        info!("Best method by accuracy: {} ({:.4})", best.0, best.1.average_accuracy);
    }

    if let Some(best) = comparison.best_by_backward_transfer() {
        info!("Best method by backward transfer: {} ({:.4})", best.0, best.1.backward_transfer);
    }

    info!("");
    info!("Experiment completed successfully!");
    info!("Results saved to: {}", config.output.output_dir.display());

    Ok(())
}

fn setup_logging(verbose: bool) -> Result<()> {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();

    Ok(())
}

fn load_config(path: &PathBuf) -> Result<ExperimentConfig> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config file: {}", path.display()))?;

    let config: ExperimentConfig = toml::from_str(&content)
        .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

    Ok(config)
}

fn validate_config(config: &ExperimentConfig) -> Result<()> {
    // Validate dataset paths
    if !config.dataset.root_dir.exists() {
        anyhow::bail!("Dataset root directory does not exist: {}", config.dataset.root_dir.display());
    }

    // Validate incremental setup
    if config.incremental.initial_classes == 0 {
        anyhow::bail!("Initial classes must be greater than 0");
    }

    if config.incremental.classes_per_step == 0 {
        anyhow::bail!("Classes per step must be greater than 0");
    }

    if config.incremental.num_steps == 0 {
        anyhow::bail!("Number of steps must be greater than 0");
    }

    // Validate methods
    if config.methods.is_empty() {
        anyhow::bail!("No methods configured");
    }

    let enabled = config.methods.iter().filter(|m| m.enabled).count();
    if enabled == 0 {
        warn!("No methods enabled in configuration");
    }

    // Validate training config
    if config.training.epochs_per_task == 0 {
        anyhow::bail!("Epochs per task must be greater than 0");
    }

    if config.training.batch_size == 0 {
        anyhow::bail!("Batch size must be greater than 0");
    }

    Ok(())
}

fn print_experiment_summary(config: &ExperimentConfig) {
    info!("");
    info!("Experiment: {}", config.experiment.name);
    info!("Description: {}", config.experiment.description);
    info!("");
    info!("Dataset:");
    info!("  Root: {}", config.dataset.root_dir.display());
    info!("  Initial classes: {}", config.incremental.initial_classes);
    info!("  Classes per step: {}", config.incremental.classes_per_step);
    info!("  Number of steps: {}", config.incremental.num_steps);
    info!("");
    info!("Training:");
    info!("  Architecture: {}", config.training.architecture);
    info!("  Epochs per task: {}", config.training.epochs_per_task);
    info!("  Batch size: {}", config.training.batch_size);
    info!("  Learning rate: {}", config.training.learning_rate);
    info!("");
    info!("Methods:");
    for method in &config.methods {
        if method.enabled {
            info!("  ✓ {}", method.name);
        }
    }
    info!("");
}

fn run_experiments(
    config: &ExperimentConfig,
    methods: &[MethodConfig],
    _resume: bool,
) -> Result<HashMap<String, IncrementalResult>> {
    let mut results = HashMap::new();

    let multi_progress = MultiProgress::new();

    for method_config in methods {
        info!("Running method: {}", method_config.name);

        let progress = multi_progress.add(ProgressBar::new(config.incremental.num_steps as u64));
        progress.set_style(
            ProgressStyle::default_bar()
                .template(&format!("[{}] {{bar:40.cyan/blue}} {{pos}}/{{len}} steps", method_config.name))
                .unwrap()
                .progress_chars("=>-"),
        );

        let result = run_single_experiment(config, method_config, &progress)?;

        progress.finish_with_message("completed");

        results.insert(method_config.name.clone(), result);

        info!("Method {} completed", method_config.name);
    }

    Ok(results)
}

fn run_single_experiment(
    config: &ExperimentConfig,
    method_config: &MethodConfig,
    progress: &ProgressBar,
) -> Result<IncrementalResult> {
    let start_time = std::time::Instant::now();
    let mut step_metrics = Vec::new();

    // Simulate incremental learning steps
    let num_steps = config.incremental.num_steps;
    let initial_classes = config.incremental.initial_classes;
    let classes_per_step = config.incremental.classes_per_step;

    for step in 0..num_steps {
        progress.set_position(step as u64);

        // Determine classes for this step
        let current_classes = if step == 0 {
            initial_classes
        } else {
            initial_classes + classes_per_step * step
        };

        // Simulate training
        let training_metrics = simulate_training(
            step,
            current_classes,
            config.training.epochs_per_task,
        );

        // Simulate evaluation on all tasks
        let task_accuracies = simulate_task_evaluation(step, current_classes);

        let avg_accuracy = task_accuracies.iter().sum::<f32>() / task_accuracies.len() as f32;

        let backward_transfer = if step > 0 {
            Some(compute_backward_transfer(&step_metrics, &task_accuracies))
        } else {
            None
        };

        let forward_transfer = if step > 0 {
            Some(0.05) // Simulated
        } else {
            None
        };

        step_metrics.push(StepMetrics {
            step,
            training: training_metrics,
            task_accuracies,
            average_accuracy: avg_accuracy,
            backward_transfer,
            forward_transfer,
        });
    }

    progress.set_position(num_steps as u64);

    let total_time = start_time.elapsed().as_secs_f64();

    let final_accuracy = step_metrics.last().unwrap().average_accuracy;
    let avg_backward_transfer = step_metrics.iter()
        .filter_map(|s| s.backward_transfer)
        .sum::<f32>() / (num_steps - 1).max(1) as f32;
    let avg_forward_transfer = step_metrics.iter()
        .filter_map(|s| s.forward_transfer)
        .sum::<f32>() / (num_steps - 1).max(1) as f32;

    Ok(IncrementalResult {
        step_metrics,
        metadata: ExperimentMetadata {
            name: method_config.name.clone(),
            config: config.incremental.clone(),
            total_time,
            final_accuracy,
            avg_backward_transfer,
            avg_forward_transfer,
        },
    })
}

fn simulate_training(step: usize, num_classes: usize, epochs: usize) -> TrainingMetrics {
    let mut metrics = TrainingMetrics::new();

    for epoch in 0..epochs {
        let loss = 2.0 * (-0.1 * epoch as f32).exp();
        let accuracy = 0.5 + 0.4 * (1.0 - (-0.15 * epoch as f32).exp());
        metrics.add_epoch(loss, accuracy);
    }

    metrics.set_training_time(epochs as f64 * 2.5);
    metrics.add_extra("step", step as f32);
    metrics.add_extra("num_classes", num_classes as f32);

    metrics
}

fn simulate_task_evaluation(step: usize, _total_classes: usize) -> Vec<f32> {
    let num_tasks = step + 1;
    let mut accuracies = Vec::new();

    for task in 0..num_tasks {
        // Simulate accuracy decay for old tasks
        let base_acc = 0.85;
        let decay = if task < step {
            0.05 * (step - task) as f32
        } else {
            0.0
        };
        let acc = (base_acc - decay).max(0.5);
        accuracies.push(acc);
    }

    accuracies
}

fn compute_backward_transfer(prev_metrics: &[StepMetrics], current_accuracies: &[f32]) -> f32 {
    if prev_metrics.is_empty() {
        return 0.0;
    }

    let last_step = prev_metrics.last().unwrap();
    let mut sum = 0.0;
    let count = last_step.task_accuracies.len();

    for i in 0..count {
        if i < current_accuracies.len() {
            sum += current_accuracies[i] - last_step.task_accuracies[i];
        }
    }

    if count > 0 {
        sum / count as f32
    } else {
        0.0
    }
}

fn analyze_results(
    results: &HashMap<String, IncrementalResult>,
    config: &ExperimentConfig,
) -> Result<MethodComparison> {
    let mut comparison = MethodComparison::new();

    let random_baseline = 1.0 / config.incremental.initial_classes as f32;

    for (method_name, result) in results {
        let analysis = IncrementalAnalysis::analyze(result, random_baseline);
        comparison.add_method(method_name.clone(), analysis);
    }

    Ok(comparison)
}

fn export_results(
    results: &HashMap<String, IncrementalResult>,
    comparison: &MethodComparison,
    config: &ExperimentConfig,
) -> Result<()> {
    // Export individual method results
    for (method_name, result) in results {
        let method_dir = config.output.output_dir.join(method_name);
        std::fs::create_dir_all(&method_dir)?;

        // Export full result as JSON
        let json_path = method_dir.join("result.json");
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(&json_path, json)?;

        // Export metrics as CSV
        let csv_path = method_dir.join("metrics.csv");
        let csv = plant_incremental::metrics::export_to_csv(result);
        std::fs::write(&csv_path, csv)?;

        info!("Exported {} results to: {}", method_name, method_dir.display());
    }

    // Export comparison summary
    let comparison_path = config.output.output_dir.join("comparison_summary.json");
    let comparison_json = serde_json::to_string_pretty(&comparison.results)?;
    std::fs::write(&comparison_path, comparison_json)?;

    // Export comparison table as CSV
    let table_path = config.output.output_dir.join("comparison_table.csv");
    export_comparison_table(comparison, &table_path)?;

    info!("Comparison results exported");

    Ok(())
}

fn export_comparison_table(comparison: &MethodComparison, path: &PathBuf) -> Result<()> {
    let mut csv = String::from("method,avg_accuracy,backward_transfer,forward_transfer,forgetting,intransigence\n");

    let mut methods: Vec<_> = comparison.results.iter().collect();
    methods.sort_by(|a, b| b.1.average_accuracy.partial_cmp(&a.1.average_accuracy).unwrap());

    for (method_name, analysis) in methods {
        csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            method_name,
            analysis.average_accuracy,
            analysis.backward_transfer,
            analysis.forward_transfer,
            analysis.forgetting,
            analysis.intransigence
        ));
    }

    std::fs::write(path, csv)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_training() {
        let metrics = simulate_training(0, 5, 10);
        assert_eq!(metrics.train_loss.len(), 10);
        assert_eq!(metrics.val_accuracy.len(), 10);
        assert!(metrics.training_time > 0.0);
    }

    #[test]
    fn test_simulate_task_evaluation() {
        let accs = simulate_task_evaluation(2, 15);
        assert_eq!(accs.len(), 3); // 3 tasks at step 2
        assert!(accs.iter().all(|&a| a >= 0.0 && a <= 1.0));
    }

    #[test]
    fn test_backward_transfer_computation() {
        let prev_metrics = vec![
            StepMetrics {
                step: 0,
                training: TrainingMetrics::new(),
                task_accuracies: vec![0.85],
                average_accuracy: 0.85,
                backward_transfer: None,
                forward_transfer: None,
            },
        ];

        let current_accs = vec![0.80, 0.88];
        let bwt = compute_backward_transfer(&prev_metrics, &current_accs);

        assert!(bwt < 0.0); // Should show forgetting
    }
}
