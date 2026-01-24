//! Benchmark Commands
//!
//! Commands for running inference benchmarks and displaying results.

use serde::{Deserialize, Serialize};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkParams {
    pub model_path: Option<String>,
    pub iterations: usize,
    pub warmup: usize,
    pub batch_size: usize,
    pub image_size: usize,
}

impl Default for BenchmarkParams {
    fn default() -> Self {
        Self {
            model_path: None,
            iterations: 100,
            warmup: 10,
            batch_size: 1,
            image_size: 128,
        }
    }
}

/// Benchmark results for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub mean_latency_ms: f64,
    pub std_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_fps: f64,
    pub gpu_memory_mb: Option<f64>,
    pub device_name: String,
    pub iterations: usize,
    pub batch_size: usize,
    pub image_size: usize,
}

/// Run benchmark
#[tauri::command]
pub async fn run_benchmark(
    params: BenchmarkParams,
) -> Result<BenchmarkResults, String> {
    use crate::backend::AdaptiveBackend;
    use plantvillage_ssl::inference::{BenchmarkConfig, run_benchmark as run_bench};
    
    // Determine model path
    let model_path = params.model_path.as_ref().map(|p| std::path::Path::new(p));

    let config = BenchmarkConfig {
        warmup_iterations: params.warmup,
        iterations: params.iterations,
        batch_size: params.batch_size,
        measure_memory: true,
        verbose: false,
        output_path: None,
    };

    let device = <AdaptiveBackend as burn::tensor::backend::Backend>::Device::default();
    
    let result = run_bench::<AdaptiveBackend>(config, model_path, params.image_size, &device)
        .map_err(|e| format!("Benchmark failed: {:?}", e))?;

    Ok(BenchmarkResults {
        mean_latency_ms: result.mean_ms,
        std_latency_ms: result.std_ms,
        min_latency_ms: result.min_ms,
        max_latency_ms: result.max_ms,
        p50_latency_ms: result.p50_ms,
        p95_latency_ms: result.p95_ms,
        p99_latency_ms: result.p99_ms,
        throughput_fps: result.throughput_fps,
        gpu_memory_mb: None, // BenchmarkOutput doesn't track this
        device_name: result.device.clone(),
        iterations: params.iterations,
        batch_size: params.batch_size,
        image_size: params.image_size,
    })
}

/// Load benchmark from file
#[tauri::command]
pub async fn load_benchmark_results(
    path: String,
) -> Result<BenchmarkResults, String> {
    use plantvillage_ssl::inference::runner::BenchmarkOutput;
    
    let json = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read file: {:?}", e))?;
    let result: BenchmarkOutput = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse benchmark results: {:?}", e))?;

    Ok(BenchmarkResults {
        mean_latency_ms: result.mean_ms,
        std_latency_ms: result.std_ms,
        min_latency_ms: result.min_ms,
        max_latency_ms: result.max_ms,
        p50_latency_ms: result.p50_ms,
        p95_latency_ms: result.p95_ms,
        p99_latency_ms: result.p99_ms,
        throughput_fps: result.throughput_fps,
        gpu_memory_mb: None,
        device_name: result.device,
        iterations: result.num_iterations,
        batch_size: result.batch_size,
        image_size: result.image_size,
    })
}
