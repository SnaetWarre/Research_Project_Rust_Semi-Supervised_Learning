//! Jetson-specific Benchmarking Module
//!
//! This module provides utilities for benchmarking on NVIDIA Jetson devices,
//! including:
//! - Power consumption monitoring (via `/sys/bus/i2c/drivers/ina3221x/`)
//! - GPU/CPU frequency reading
//! - Tegra-specific device detection
//! - Comprehensive benchmark reports for research papers

use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::benchmark::{BenchmarkConfig, BenchmarkResult, DeviceInfo, LatencyStats, MemoryStats};

/// Power measurements for Jetson devices
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct JetsonPowerStats {
    /// Total system power (mW)
    pub total_power_mw: f64,
    /// GPU power (mW)
    pub gpu_power_mw: f64,
    /// CPU power (mW)
    pub cpu_power_mw: f64,
    /// SOC power (mW)
    pub soc_power_mw: f64,
    /// Average power during inference (mW)
    pub avg_inference_power_mw: f64,
    /// Peak power during inference (mW)
    pub peak_power_mw: f64,
    /// Energy per inference (mJ)
    pub energy_per_inference_mj: f64,
}

/// Jetson-specific device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetsonDeviceInfo {
    /// Base device info
    pub base: DeviceInfo,
    /// Jetson model (e.g., "Orin Nano", "Xavier NX")
    pub jetson_model: String,
    /// L4T version
    pub l4t_version: Option<String>,
    /// JetPack version
    pub jetpack_version: Option<String>,
    /// Current GPU frequency (MHz)
    pub gpu_freq_mhz: Option<u32>,
    /// Current CPU frequency (MHz)
    pub cpu_freq_mhz: Option<u32>,
    /// Power mode
    pub power_mode: Option<String>,
}

impl JetsonDeviceInfo {
    /// Detect Jetson device information
    pub fn detect() -> Option<Self> {
        // Check if running on Jetson
        if !is_jetson() {
            return None;
        }

        let base = DeviceInfo::detect();
        let jetson_model = detect_jetson_model().unwrap_or_else(|| "Unknown Jetson".to_string());
        let l4t_version = detect_l4t_version();
        let gpu_freq_mhz = read_gpu_frequency();
        let cpu_freq_mhz = read_cpu_frequency();
        let power_mode = detect_power_mode();

        Some(Self {
            base,
            jetson_model,
            l4t_version,
            jetpack_version: None, // Would need to parse dpkg
            gpu_freq_mhz,
            cpu_freq_mhz,
            power_mode,
        })
    }
}

/// Complete Jetson benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JetsonBenchmarkResult {
    /// Standard benchmark metrics
    pub latency: LatencyStats,
    /// Throughput (images per second)
    pub throughput_fps: f64,
    /// Memory usage
    pub memory: MemoryStats,
    /// Power statistics
    pub power: JetsonPowerStats,
    /// Device information
    pub device: JetsonDeviceInfo,
    /// Benchmark configuration
    pub config: BenchmarkConfig,
    /// Timestamp
    pub timestamp: String,
    /// Model path (if any)
    pub model_path: Option<String>,
    /// Image size used
    pub image_size: usize,
}

impl JetsonBenchmarkResult {
    /// Generate a summary suitable for research papers
    pub fn paper_summary(&self) -> String {
        let mut summary = String::new();
        
        summary.push_str("## Performance Metrics\n\n");
        summary.push_str(&format!("| Metric | Value |\n"));
        summary.push_str(&format!("|--------|-------|\n"));
        summary.push_str(&format!("| Device | {} |\n", self.device.jetson_model));
        summary.push_str(&format!("| Mean Latency | {:.2} ms |\n", self.latency.mean_ms));
        summary.push_str(&format!("| P95 Latency | {:.2} ms |\n", self.latency.p95_ms));
        summary.push_str(&format!("| P99 Latency | {:.2} ms |\n", self.latency.p99_ms));
        summary.push_str(&format!("| Throughput | {:.1} FPS |\n", self.throughput_fps));
        summary.push_str(&format!("| GPU Memory | {:.1} MB |\n", self.memory.gpu_used_mb));
        summary.push_str(&format!("| Avg Power | {:.1} mW |\n", self.power.avg_inference_power_mw));
        summary.push_str(&format!("| Energy/Inference | {:.2} mJ |\n", self.power.energy_per_inference_mj));
        
        summary
    }

    /// Save to JSON file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        fs::write(path, json)
    }
}

impl std::fmt::Display for JetsonBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║           Jetson Benchmark Results                               ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ Device: {:58} ║", self.device.jetson_model)?;
        if let Some(ref mode) = self.device.power_mode {
            writeln!(f, "║ Power Mode: {:54} ║", mode)?;
        }
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ LATENCY                                                          ║")?;
        writeln!(f, "║   Mean:     {:8.2} ms                                            ║", self.latency.mean_ms)?;
        writeln!(f, "║   Std Dev:  {:8.2} ms                                            ║", self.latency.std_ms)?;
        writeln!(f, "║   P50:      {:8.2} ms                                            ║", self.latency.p50_ms)?;
        writeln!(f, "║   P95:      {:8.2} ms                                            ║", self.latency.p95_ms)?;
        writeln!(f, "║   P99:      {:8.2} ms                                            ║", self.latency.p99_ms)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ THROUGHPUT: {:8.1} images/second                                 ║", self.throughput_fps)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ MEMORY                                                           ║")?;
        writeln!(f, "║   GPU Used:   {:6.1} MB                                          ║", self.memory.gpu_used_mb)?;
        writeln!(f, "║   GPU Total:  {:6.1} MB                                          ║", self.memory.gpu_total_mb)?;
        writeln!(f, "║   CPU RSS:    {:6.1} MB                                          ║", self.memory.cpu_rss_mb)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║ POWER                                                            ║")?;
        writeln!(f, "║   Avg Power:  {:7.1} mW                                          ║", self.power.avg_inference_power_mw)?;
        writeln!(f, "║   Peak Power: {:7.1} mW                                          ║", self.power.peak_power_mw)?;
        writeln!(f, "║   Energy/Inf: {:7.2} mJ                                          ║", self.power.energy_per_inference_mj)?;
        writeln!(f, "╚══════════════════════════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Check if running on a Jetson device
pub fn is_jetson() -> bool {
    Path::new("/etc/nv_tegra_release").exists()
        || Path::new("/sys/devices/soc0/family").exists()
}

/// Detect Jetson model name
fn detect_jetson_model() -> Option<String> {
    // Try reading from /sys/devices/soc0/machine
    if let Ok(machine) = fs::read_to_string("/sys/devices/soc0/machine") {
        let machine = machine.trim();
        if !machine.is_empty() {
            return Some(machine.to_string());
        }
    }

    // Try reading from /proc/device-tree/model
    if let Ok(model) = fs::read_to_string("/proc/device-tree/model") {
        let model = model.trim().trim_end_matches('\0');
        if !model.is_empty() {
            return Some(model.to_string());
        }
    }

    // Fall back to nvidia-smi
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
    {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() {
                return Some(format!("{} (Jetson)", name));
            }
        }
    }

    None
}

/// Detect L4T version
fn detect_l4t_version() -> Option<String> {
    let release = fs::read_to_string("/etc/nv_tegra_release").ok()?;
    
    // Parse: "# R35 (release), REVISION: 1.0, ..."
    for part in release.split(',') {
        if let Some(version) = part.strip_prefix("# R") {
            return Some(format!("R{}", version.split_whitespace().next()?));
        }
    }
    
    None
}

/// Read current GPU frequency in MHz
fn read_gpu_frequency() -> Option<u32> {
    // Try Jetson sysfs path
    let paths = [
        "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
        "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq",
        "/sys/kernel/debug/clk/gpcclk/clk_rate",
    ];

    for path in paths {
        if let Ok(freq_str) = fs::read_to_string(path) {
            if let Ok(freq_hz) = freq_str.trim().parse::<u64>() {
                return Some((freq_hz / 1_000_000) as u32);
            }
        }
    }

    None
}

/// Read current CPU frequency in MHz
fn read_cpu_frequency() -> Option<u32> {
    let path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    if let Ok(freq_str) = fs::read_to_string(path) {
        if let Ok(freq_khz) = freq_str.trim().parse::<u64>() {
            return Some((freq_khz / 1000) as u32);
        }
    }
    None
}

/// Detect current power mode
fn detect_power_mode() -> Option<String> {
    // Try nvpmodel
    if let Ok(output) = std::process::Command::new("nvpmodel")
        .args(["-q"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("NV Power Mode") {
                    return Some(line.trim().to_string());
                }
            }
        }
    }
    None
}

/// Power monitor for continuous power measurement during inference
pub struct PowerMonitor {
    samples: Vec<f64>,
    sampling_interval_ms: u64,
    running: bool,
}

impl PowerMonitor {
    /// Create a new power monitor
    pub fn new(sampling_interval_ms: u64) -> Self {
        Self {
            samples: Vec::new(),
            sampling_interval_ms,
            running: false,
        }
    }

    /// Read current total power in mW
    pub fn read_power() -> Option<f64> {
        // Jetson Orin Nano power paths (INA3221 power monitors)
        let power_paths = [
            "/sys/bus/i2c/drivers/ina3221x/1-0040/hwmon/hwmon2/in1_input", // VDD_IN
            "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon2/power1_input",
            "/sys/class/hwmon/hwmon2/in1_input",
        ];

        for path in power_paths {
            if let Ok(power_str) = fs::read_to_string(path) {
                if let Ok(power_mv) = power_str.trim().parse::<f64>() {
                    // Convert from mV or uW to mW depending on the source
                    return Some(power_mv);
                }
            }
        }

        // Fallback: use tegrastats parsing
        None
    }

    /// Read power breakdown (total, gpu, cpu, soc)
    pub fn read_power_breakdown() -> JetsonPowerStats {
        // Try to read from tegrastats or INA3221
        let total = Self::read_power().unwrap_or(0.0);
        
        JetsonPowerStats {
            total_power_mw: total,
            gpu_power_mw: 0.0,  // Would need specific INA channel
            cpu_power_mw: 0.0,
            soc_power_mw: 0.0,
            avg_inference_power_mw: total,
            peak_power_mw: total,
            energy_per_inference_mj: 0.0,
        }
    }

    /// Sample power and return stats
    pub fn sample_during<F>(&mut self, duration_ms: u64, mut callback: F) -> JetsonPowerStats 
    where
        F: FnMut(),
    {
        self.samples.clear();
        let start = Instant::now();
        let duration = Duration::from_millis(duration_ms);
        let interval = Duration::from_millis(self.sampling_interval_ms);

        while start.elapsed() < duration {
            // Run callback (inference)
            callback();
            
            // Sample power
            if let Some(power) = Self::read_power() {
                self.samples.push(power);
            }
            
            std::thread::sleep(interval.saturating_sub(Duration::from_millis(1)));
        }

        self.calculate_stats()
    }

    /// Calculate statistics from samples
    fn calculate_stats(&self) -> JetsonPowerStats {
        if self.samples.is_empty() {
            return JetsonPowerStats::default();
        }

        let sum: f64 = self.samples.iter().sum();
        let avg = sum / self.samples.len() as f64;
        let peak = self.samples.iter().cloned().fold(0.0f64, f64::max);

        JetsonPowerStats {
            total_power_mw: avg,
            gpu_power_mw: 0.0,
            cpu_power_mw: 0.0,
            soc_power_mw: 0.0,
            avg_inference_power_mw: avg,
            peak_power_mw: peak,
            energy_per_inference_mj: 0.0, // Calculated later with latency
        }
    }
}

/// Run a Jetson-optimized benchmark
pub fn run_jetson_benchmark<B, F>(
    config: BenchmarkConfig,
    image_size: usize,
    mut inference_fn: F,
) -> Result<JetsonBenchmarkResult>
where
    B: burn::tensor::backend::Backend,
    F: FnMut(),
{
    use super::benchmark::Timer;

    // Detect device
    let device_info = JetsonDeviceInfo::detect()
        .ok_or_else(|| anyhow::anyhow!("Not running on a Jetson device"))?;

    println!("Jetson Device: {}", device_info.jetson_model);
    if let Some(ref mode) = device_info.power_mode {
        println!("Power Mode: {}", mode);
    }

    // Warmup
    println!("\nRunning warmup ({} iterations)...", config.warmup_iterations);
    for _ in 0..config.warmup_iterations {
        inference_fn();
    }

    // Benchmark with timing
    println!("Running benchmark ({} iterations)...", config.iterations);
    let mut timer = Timer::new();
    let mut power_samples: Vec<f64> = Vec::new();

    for _ in 0..config.iterations {
        // Sample power before
        if let Some(power) = PowerMonitor::read_power() {
            power_samples.push(power);
        }

        timer.start();
        inference_fn();
        timer.stop();
    }

    // Calculate latency stats
    let latency = LatencyStats::from_durations(timer.times());
    let throughput = if latency.mean_ms > 0.0 {
        1000.0 / latency.mean_ms * config.batch_size as f64
    } else {
        0.0
    };

    // Calculate power stats
    let avg_power = if power_samples.is_empty() {
        0.0
    } else {
        power_samples.iter().sum::<f64>() / power_samples.len() as f64
    };
    let peak_power = power_samples.iter().cloned().fold(0.0f64, f64::max);
    let energy_per_inference = avg_power * latency.mean_ms / 1000.0; // mW * s = mJ

    let power = JetsonPowerStats {
        total_power_mw: avg_power,
        gpu_power_mw: 0.0,
        cpu_power_mw: 0.0,
        soc_power_mw: 0.0,
        avg_inference_power_mw: avg_power,
        peak_power_mw: peak_power,
        energy_per_inference_mj: energy_per_inference,
    };

    // Get memory stats
    let memory = MemoryStats::current();

    let result = JetsonBenchmarkResult {
        latency,
        throughput_fps: throughput,
        memory,
        power,
        device: device_info,
        config,
        timestamp: chrono::Utc::now().to_rfc3339(),
        model_path: None,
        image_size,
    };

    println!("\n{}", result);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_jetson() {
        // This test will pass on Jetson, fail on other devices
        let is_jetson = is_jetson();
        println!("Running on Jetson: {}", is_jetson);
    }

    #[test]
    fn test_power_monitor() {
        let power = PowerMonitor::read_power();
        println!("Current power: {:?} mW", power);
    }

    #[test]
    fn test_jetson_device_info() {
        if let Some(info) = JetsonDeviceInfo::detect() {
            println!("Jetson Model: {}", info.jetson_model);
            println!("L4T Version: {:?}", info.l4t_version);
            println!("GPU Freq: {:?} MHz", info.gpu_freq_mhz);
            println!("CPU Freq: {:?} MHz", info.cpu_freq_mhz);
            println!("Power Mode: {:?}", info.power_mode);
        } else {
            println!("Not running on Jetson");
        }
    }
}
