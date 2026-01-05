//! Benchmark Runner Module
//!
//! Provides functions to run inference benchmarks on trained models.
//! Outputs results in JSON format for easy comparison with PyTorch.

use std::path::Path;


use anyhow::Result;
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
};
use colored::Colorize;
use serde::{Deserialize, Serialize};

use crate::model::cnn::{PlantClassifier, PlantClassifierConfig};
use super::benchmark::{BenchmarkConfig, BenchmarkResult, LatencyStats, DeviceInfo, Timer};

/// Benchmark results in a format compatible with the comparison script
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkOutput {
    pub framework: String,
    pub device: String,
    pub batch_size: usize,
    pub image_size: usize,
    pub num_iterations: usize,
    pub warmup_iterations: usize,

    // Latency metrics (ms)
    pub mean_ms: f64,
    pub std_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,

    // Throughput
    pub throughput_fps: f64,

    // Model info
    pub model_size_mb: f64,

    // Timestamp
    pub timestamp: String,
}

impl From<BenchmarkResult> for BenchmarkOutput {
    fn from(result: BenchmarkResult) -> Self {
        Self {
            framework: "Burn (Rust)".to_string(),
            device: result.device_info.device_type.clone(),
            batch_size: result.config.batch_size,
            image_size: 128, // Default, can be parameterized
            num_iterations: result.config.iterations,
            warmup_iterations: result.config.warmup_iterations,
            mean_ms: result.latency.mean_ms,
            std_ms: result.latency.std_ms,
            min_ms: result.latency.min_ms,
            max_ms: result.latency.max_ms,
            p50_ms: result.latency.p50_ms,
            p95_ms: result.latency.p95_ms,
            p99_ms: result.latency.p99_ms,
            throughput_fps: result.throughput,
            model_size_mb: 0.0, // Filled in separately
            timestamp: result.timestamp,
        }
    }
}

/// Run a benchmark on a model with random input
///
/// This function creates a model, runs warmup iterations, then measures
/// inference latency over multiple iterations.
pub fn run_benchmark<B: Backend>(
    config: BenchmarkConfig,
    model_path: Option<&Path>,
    image_size: usize,
    device: &B::Device,
) -> Result<BenchmarkOutput> {
    println!("{}", "Initializing Benchmark...".green().bold());
    println!("  Device: {:?}", device);
    println!("  Batch size: {}", config.batch_size);
    println!("  Image size: {}x{}", image_size, image_size);
    println!("  Warmup iterations: {}", config.warmup_iterations);
    println!("  Benchmark iterations: {}", config.iterations);
    println!();

    // Create or load model
    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: image_size,
        dropout_rate: 0.5,
        in_channels: 3,
        base_filters: 32,
    };

    let model: PlantClassifier<B> = if let Some(path) = model_path {
        println!("{}", format!("Loading model from {:?}...", path).cyan());
        let recorder = CompactRecorder::new();
        PlantClassifier::new(&model_config, device)
            .load_file(path, &recorder, device)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?
    } else {
        println!("{}", "Creating model (no checkpoint specified)...".cyan());
        PlantClassifier::new(&model_config, device)
    };

    // Create random input tensor
    let input = Tensor::<B, 4>::random(
        [config.batch_size, 3, image_size, image_size],
        burn::tensor::Distribution::Uniform(-1.0, 1.0),
        device,
    );

    // Warmup phase
    println!();
    println!("{}", "Running warmup...".yellow());
    for i in 0..config.warmup_iterations {
        let _ = model.forward(input.clone());
        if config.verbose && (i + 1) % 5 == 0 {
            println!("  Warmup iteration {}/{}", i + 1, config.warmup_iterations);
        }
    }

    // Benchmark phase
    println!();
    println!("{}", "Running benchmark...".green().bold());
    let mut timer = Timer::new();

    for i in 0..config.iterations {
        timer.start();
        let _ = model.forward(input.clone());
        timer.stop();

        if config.verbose && (i + 1) % 20 == 0 {
            println!("  Iteration {}/{}", i + 1, config.iterations);
        }
    }

    let times = timer.times();
    let latency = LatencyStats::from_durations(&times);

    // Calculate throughput
    let throughput = if latency.mean_ms > 0.0 {
        config.batch_size as f64 / (latency.mean_ms / 1000.0)
    } else {
        0.0
    };

    // Get model size if path provided
    let model_size_mb = if let Some(path) = model_path {
        std::fs::metadata(path)
            .map(|m| m.len() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    } else {
        0.0
    };

    // Detect device info
    let device_info = DeviceInfo::detect();

    // Build result
    let output = BenchmarkOutput {
        framework: "Burn (Rust)".to_string(),
        device: device_info.device_type.clone(),
        batch_size: config.batch_size,
        image_size,
        num_iterations: config.iterations,
        warmup_iterations: config.warmup_iterations,
        mean_ms: latency.mean_ms,
        std_ms: latency.std_ms,
        min_ms: latency.min_ms,
        max_ms: latency.max_ms,
        p50_ms: latency.p50_ms,
        p95_ms: latency.p95_ms,
        p99_ms: latency.p99_ms,
        throughput_fps: throughput,
        model_size_mb,
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    // Print results
    println!();
    println!("{}", "Benchmark Results:".cyan().bold());
    println!("  Device: {}", output.device);
    println!("  Batch size: {}", output.batch_size);
    println!("  Image size: {}x{}", output.image_size, output.image_size);
    println!();
    println!("  {} {} ± {} ms", "Mean latency:".green(),
        format!("{:.2}", output.mean_ms).bold(),
        format!("{:.2}", output.std_ms));
    println!("  P50/P95/P99: {:.2}/{:.2}/{:.2} ms",
        output.p50_ms, output.p95_ms, output.p99_ms);
    println!("  Min/Max: {:.2}/{:.2} ms", output.min_ms, output.max_ms);
    println!();
    println!("  {} {} FPS", "Throughput:".green(),
        format!("{:.1}", output.throughput_fps).bold());

    if output.model_size_mb > 0.0 {
        println!("  Model size: {:.2} MB", output.model_size_mb);
    }

    // Check against target
    let target_latency_ms = 200.0;
    if output.mean_ms <= target_latency_ms {
        println!();
        println!("{} Meets target latency of {} ms!",
            "✓".green().bold(), target_latency_ms);
    } else {
        println!();
        println!("{} Exceeds target latency of {} ms by {:.1} ms",
            "⚠".yellow().bold(), target_latency_ms, output.mean_ms - target_latency_ms);
    }

    // Save to file if output path specified
    if let Some(output_path) = &config.output_path {
        let json = serde_json::to_string_pretty(&output)?;
        std::fs::write(output_path, &json)?;
        println!();
        println!("  Saved results to: {:?}", output_path);
    }

    Ok(output)
}

/// Run a quick benchmark with default settings
pub fn run_quick_benchmark<B: Backend>(device: &B::Device) -> Result<BenchmarkOutput> {
    let config = BenchmarkConfig::quick();
    run_benchmark::<B>(config, None, 128, device)
}

/// Run a thorough benchmark for final performance numbers
pub fn run_thorough_benchmark<B: Backend>(
    model_path: &Path,
    output_path: &Path,
    device: &B::Device,
) -> Result<BenchmarkOutput> {
    let mut config = BenchmarkConfig::thorough();
    config.output_path = Some(output_path.to_path_buf());
    run_benchmark::<B>(config, Some(model_path), 128, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cuda::Cuda;

    #[test]
    fn test_quick_benchmark() {
        let device = Default::default();
        let result = run_quick_benchmark::<Cuda>(&device);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(output.mean_ms > 0.0);
        assert!(output.throughput_fps > 0.0);
    }
}
