//! Inference module for model prediction and benchmarking
//!
//! This module provides:
//! - Single image and batch prediction
//! - Latency benchmarking for deployment
//! - Model optimization utilities
//!
//! ## Deployment
//!
//! The inference pipeline is optimized for GPU with:
//! - Batched inference for throughput
//! - Memory-efficient processing
//! - Latency profiling and monitoring

pub mod predictor;
pub mod benchmark;
pub mod runner;

// Re-export main types for convenience
pub use predictor::{Predictor, PredictionResult};
pub use benchmark::{BenchmarkConfig, BenchmarkResult, LatencyStats, DeviceInfo, Timer};
pub use runner::{BenchmarkOutput, run_benchmark, run_quick_benchmark, run_thorough_benchmark};

/// Target latency for deployment (milliseconds)
pub const TARGET_LATENCY_MS: f64 = 200.0;

/// Maximum acceptable latency (milliseconds)
pub const MAX_LATENCY_MS: f64 = 500.0;

/// Default number of warmup iterations for benchmarking
pub const WARMUP_ITERATIONS: usize = 10;

/// Default number of benchmark iterations
pub const BENCHMARK_ITERATIONS: usize = 100;
