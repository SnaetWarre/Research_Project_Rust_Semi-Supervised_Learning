//! Benchmark Module for Inference Latency Testing
//!
//! This module provides utilities for measuring and analyzing inference
//! performance on edge devices like the NVIDIA Jetson Orin Nano.
//!
//! ## Key Metrics
//!
//! - **Latency**: Time per inference (ms)
//! - **Throughput**: Images processed per second
//! - **Memory Usage**: GPU/CPU memory consumption
//! - **Power Consumption**: Watts used during inference (on supported devices)

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Configuration for benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations (excluded from measurements)
    pub warmup_iterations: usize,

    /// Number of benchmark iterations
    pub iterations: usize,

    /// Batch size for inference
    pub batch_size: usize,

    /// Whether to measure GPU memory usage
    pub measure_memory: bool,

    /// Whether to log individual iteration times
    pub verbose: bool,

    /// Output file for results (optional)
    pub output_path: Option<PathBuf>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            iterations: 100,
            batch_size: 1,
            measure_memory: true,
            verbose: false,
            output_path: None,
        }
    }
}

impl BenchmarkConfig {
    /// Create a quick benchmark config for testing
    pub fn quick() -> Self {
        Self {
            warmup_iterations: 5,
            iterations: 20,
            batch_size: 1,
            measure_memory: false,
            verbose: false,
            output_path: None,
        }
    }

    /// Create a thorough benchmark config for production testing
    pub fn thorough() -> Self {
        Self {
            warmup_iterations: 20,
            iterations: 500,
            batch_size: 1,
            measure_memory: true,
            verbose: true,
            output_path: None,
        }
    }

    /// Create a config optimized for Jetson testing
    pub fn jetson() -> Self {
        Self {
            warmup_iterations: 10,
            iterations: 100,
            batch_size: 1, // Single image, common for edge inference
            measure_memory: true,
            verbose: false,
            output_path: Some(PathBuf::from("output/jetson_benchmark.json")),
        }
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Latency statistics
    pub latency: LatencyStats,

    /// Throughput (images per second)
    pub throughput: f64,

    /// Memory usage statistics
    pub memory: Option<MemoryStats>,

    /// Individual iteration times (if verbose)
    pub iteration_times_ms: Vec<f64>,

    /// Configuration used for this benchmark
    pub config: BenchmarkConfig,

    /// Timestamp of when benchmark was run
    pub timestamp: String,

    /// Device information
    pub device_info: DeviceInfo,
}

impl BenchmarkResult {
    /// Create a new benchmark result from timing data
    pub fn from_timings(
        timings: Vec<Duration>,
        config: BenchmarkConfig,
        memory: Option<MemoryStats>,
        device_info: DeviceInfo,
    ) -> Self {
        let latency = LatencyStats::from_durations(&timings);
        let throughput = 1000.0 / latency.mean_ms * config.batch_size as f64;

        let iteration_times_ms = if config.verbose {
            timings.iter().map(|d| d.as_secs_f64() * 1000.0).collect()
        } else {
            Vec::new()
        };

        let timestamp = chrono::Utc::now().to_rfc3339();

        Self {
            latency,
            throughput,
            memory,
            iteration_times_ms,
            config,
            timestamp,
            device_info,
        }
    }

    /// Save results to a JSON file
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, json)
    }

    /// Load results from a JSON file
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Check if latency meets the target (in milliseconds)
    pub fn meets_latency_target(&self, target_ms: f64) -> bool {
        self.latency.p95_ms <= target_ms
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Latency: {:.2}ms (mean), {:.2}ms (p95), {:.2}ms (p99) | Throughput: {:.1} img/s",
            self.latency.mean_ms, self.latency.p95_ms, self.latency.p99_ms, self.throughput
        )
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║              Benchmark Results                               ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Device: {:54} ║", self.device_info.name)?;
        writeln!(f, "║ Timestamp: {:51} ║", &self.timestamp[..19])?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Latency Statistics                                           ║")?;
        writeln!(f, "║   Mean:     {:8.2} ms                                      ║", self.latency.mean_ms)?;
        writeln!(f, "║   Std Dev:  {:8.2} ms                                      ║", self.latency.std_ms)?;
        writeln!(f, "║   Min:      {:8.2} ms                                      ║", self.latency.min_ms)?;
        writeln!(f, "║   Max:      {:8.2} ms                                      ║", self.latency.max_ms)?;
        writeln!(f, "║   P50:      {:8.2} ms                                      ║", self.latency.p50_ms)?;
        writeln!(f, "║   P95:      {:8.2} ms                                      ║", self.latency.p95_ms)?;
        writeln!(f, "║   P99:      {:8.2} ms                                      ║", self.latency.p99_ms)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Throughput: {:8.1} images/second                          ║", self.throughput)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Configuration                                                ║")?;
        writeln!(f, "║   Batch Size:   {:5}                                        ║", self.config.batch_size)?;
        writeln!(f, "║   Iterations:   {:5}                                        ║", self.config.iterations)?;
        writeln!(f, "║   Warmup:       {:5}                                        ║", self.config.warmup_iterations)?;

        if let Some(ref mem) = self.memory {
            writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
            writeln!(f, "║ Memory Usage                                                 ║")?;
            writeln!(f, "║   GPU Used:     {:6.1} MB                                    ║", mem.gpu_used_mb)?;
            writeln!(f, "║   GPU Total:    {:6.1} MB                                    ║", mem.gpu_total_mb)?;
            writeln!(f, "║   CPU RSS:      {:6.1} MB                                    ║", mem.cpu_rss_mb)?;
        }

        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Latency statistics from benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Mean latency in milliseconds
    pub mean_ms: f64,
    /// Standard deviation in milliseconds
    pub std_ms: f64,
    /// Minimum latency
    pub min_ms: f64,
    /// Maximum latency
    pub max_ms: f64,
    /// Median (50th percentile)
    pub p50_ms: f64,
    /// 95th percentile
    pub p95_ms: f64,
    /// 99th percentile
    pub p99_ms: f64,
}

impl LatencyStats {
    /// Calculate statistics from a list of durations
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self::default();
        }

        let mut times_ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = times_ms.len();
        let sum: f64 = times_ms.iter().sum();
        let mean = sum / n as f64;

        let variance: f64 = times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        Self {
            mean_ms: mean,
            std_ms: std,
            min_ms: times_ms[0],
            max_ms: times_ms[n - 1],
            p50_ms: percentile(&times_ms, 50.0),
            p95_ms: percentile(&times_ms, 95.0),
            p99_ms: percentile(&times_ms, 99.0),
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            mean_ms: 0.0,
            std_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
        }
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let idx = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[idx.min(sorted_data.len() - 1)]
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// GPU memory used in MB
    pub gpu_used_mb: f64,
    /// GPU total memory in MB
    pub gpu_total_mb: f64,
    /// CPU resident set size in MB
    pub cpu_rss_mb: f64,
}

impl MemoryStats {
    /// Get current memory usage
    pub fn current() -> Self {
        // GPU memory: try to read from nvidia-smi or sysfs
        let (gpu_used, gpu_total) = get_gpu_memory().unwrap_or((0.0, 0.0));

        // CPU memory: read from /proc/self/status
        let cpu_rss = get_cpu_memory().unwrap_or(0.0);

        Self {
            gpu_used_mb: gpu_used,
            gpu_total_mb: gpu_total,
            cpu_rss_mb: cpu_rss,
        }
    }
}

/// Get GPU memory usage (returns (used_mb, total_mb))
fn get_gpu_memory() -> Option<(f64, f64)> {
    // Try nvidia-smi first
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = stdout.trim().split(',').collect();

        if parts.len() >= 2 {
            let used: f64 = parts[0].trim().parse().ok()?;
            let total: f64 = parts[1].trim().parse().ok()?;
            return Some((used, total));
        }
    }

    None
}

/// Get CPU memory usage in MB
fn get_cpu_memory() -> Option<f64> {
    // Read from /proc/self/status
    let status = std::fs::read_to_string("/proc/self/status").ok()?;

    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let kb: f64 = parts[1].parse().ok()?;
                return Some(kb / 1024.0);
            }
        }
    }

    None
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Device type (CPU, CUDA, etc.)
    pub device_type: String,
    /// CUDA version (if applicable)
    pub cuda_version: Option<String>,
    /// Compute capability (for GPUs)
    pub compute_capability: Option<String>,
}

impl DeviceInfo {
    /// Detect current device information
    pub fn detect() -> Self {
        let mut info = Self {
            name: "Unknown".to_string(),
            device_type: "CPU".to_string(),
            cuda_version: None,
            compute_capability: None,
        };

        // Try to get GPU info from nvidia-smi
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
        {
            if output.status.success() {
                let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !name.is_empty() {
                    info.name = name;
                    info.device_type = "CUDA".to_string();
                }
            }
        }

        // Check if running on Jetson
        if std::path::Path::new("/etc/nv_tegra_release").exists() {
            info.name = format!("{} (Jetson)", info.name);
        }

        // Get CUDA version
        if let Ok(output) = std::process::Command::new("nvcc")
            .args(["--version"])
            .output()
        {
            if output.status.success() {
                let version_info = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = version_info.lines().find(|l| l.contains("release")) {
                    if let Some(version) = line.split("release").nth(1) {
                        info.cuda_version = Some(version.split(',').next().unwrap_or("").trim().to_string());
                    }
                }
            }
        }

        info
    }
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self::detect()
    }
}

/// Timer utility for benchmarking
pub struct Timer {
    start: Instant,
    times: Vec<Duration>,
}

impl Timer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            times: Vec::new(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Stop timing and record the duration
    pub fn stop(&mut self) -> Duration {
        let elapsed = self.start.elapsed();
        self.times.push(elapsed);
        elapsed
    }

    /// Get all recorded times
    pub fn times(&self) -> &[Duration] {
        &self.times
    }

    /// Clear recorded times
    pub fn clear(&mut self) {
        self.times.clear();
    }

    /// Get statistics from recorded times
    pub fn stats(&self) -> LatencyStats {
        LatencyStats::from_durations(&self.times)
    }
}

impl Default for Timer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats() {
        let durations: Vec<Duration> = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(15),
            Duration::from_millis(9),
        ];

        let stats = LatencyStats::from_durations(&durations);

        assert!((stats.mean_ms - 11.4).abs() < 0.1);
        assert_eq!(stats.min_ms, 9.0);
        assert_eq!(stats.max_ms, 15.0);
    }

    #[test]
    fn test_benchmark_config_defaults() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.iterations, 100);
        assert_eq!(config.batch_size, 1);
    }

    #[test]
    fn test_timer() {
        let mut timer = Timer::new();

        for _ in 0..5 {
            timer.start();
            std::thread::sleep(Duration::from_millis(1));
            timer.stop();
        }

        assert_eq!(timer.times().len(), 5);
        assert!(timer.stats().mean_ms >= 1.0);
    }

    #[test]
    fn test_meets_latency_target() {
        let timings = vec![
            Duration::from_millis(100),
            Duration::from_millis(120),
            Duration::from_millis(110),
        ];

        let result = BenchmarkResult::from_timings(
            timings,
            BenchmarkConfig::default(),
            None,
            DeviceInfo::default(),
        );

        assert!(result.meets_latency_target(500.0));
        assert!(!result.meets_latency_target(50.0));
    }
}
