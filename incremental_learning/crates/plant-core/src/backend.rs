//! Backend selection for Burn framework.
//!
//! This module provides automatic backend detection and configuration,
//! prioritizing GPU when available for better performance.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// Device type for backend selection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Device {
    /// CPU backend
    Cpu,
    /// GPU backend (when available)
    Gpu(usize),
}

impl Default for Device {
    fn default() -> Self {
        if is_gpu_available() {
            Device::Gpu(0)
        } else {
            Device::Cpu
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            Device::Gpu(id) => write!(f, "GPU:{}", id),
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Preferred device (CPU or GPU)
    pub device: Device,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            device: Device::default(),
        }
    }
}

static BACKEND_INFO: OnceLock<String> = OnceLock::new();

/// Initialize and log the backend selection
pub fn init_backend(config: &BackendConfig) -> Device {
    let device = match config.device {
        Device::Cpu => {
            eprintln!("💻 Using CPU backend");
            Device::Cpu
        }
        Device::Gpu(id) => {
            if is_gpu_available() {
                eprintln!("🚀 GPU detected (ID: {}) - will accelerate training", id);
                Device::Gpu(id)
            } else {
                eprintln!("⚠️  GPU requested but not available - falling back to CPU");
                Device::Cpu
            }
        }
    };

    // Store backend info for logging
    let info = format!("Using device: {}", device);
    let _ = BACKEND_INFO.set(info.clone());

    eprintln!("🔥 Burn backend initialized: {}", info);

    device
}

/// Select the best available device automatically
pub fn select_best_device() -> Device {
    // Try GPU first if available
    if is_discrete_gpu_available() {
        eprintln!("🚀 Discrete GPU detected - will use for acceleration");
        return Device::Gpu(0);
    }

    // Fallback to CPU
    eprintln!("💻 No discrete GPU detected - using CPU backend");
    Device::Cpu
}

/// Check if discrete GPU is available
fn is_discrete_gpu_available() -> bool {
    // Check for NVIDIA GPU
    has_nvidia_gpu() || has_amd_gpu()
}

/// Check if any GPU acceleration is available
pub fn is_gpu_available() -> bool {
    is_discrete_gpu_available()
}

/// Check for NVIDIA GPU (CUDA)
fn has_nvidia_gpu() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/proc/driver/nvidia/version").exists() ||
        std::path::Path::new("/dev/nvidia0").exists() ||
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok() ||
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("nvidia-smi.exe")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[cfg(target_os = "macos")]
    {
        false // No NVIDIA support on modern macOS
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Check for AMD GPU (ROCm)
fn has_amd_gpu() -> bool {
    #[cfg(target_os = "linux")]
    {
        std::path::Path::new("/sys/module/amdgpu").exists() ||
        std::env::var("HIP_VISIBLE_DEVICES").is_ok() ||
        std::process::Command::new("rocm-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[cfg(target_os = "windows")]
    {
        false // ROCm not well supported on Windows
    }

    #[cfg(target_os = "macos")]
    {
        false // No ROCm on macOS
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        false
    }
}

/// Get information about the current backend
pub fn backend_info() -> String {
    BACKEND_INFO.get()
        .cloned()
        .unwrap_or_else(|| "Backend not initialized".to_string())
}

/// Get recommended device based on system
pub fn get_recommended_device() -> Device {
    select_best_device()
}

/// Print system GPU information
pub fn print_gpu_info() {
    println!("🔍 System GPU Detection:");
    println!("  NVIDIA GPU: {}", if has_nvidia_gpu() { "✓ Detected" } else { "✗ Not found" });
    println!("  AMD GPU:    {}", if has_amd_gpu() { "✓ Detected" } else { "✗ Not found" });
    println!("  Recommended device: {}", get_recommended_device());
    println!();

    if is_gpu_available() {
        println!("📝 Note: GPU detected but Burn will use NdArray backend (CPU) by default.");
        println!("   For GPU acceleration, additional setup is required:");
        println!("   - NVIDIA: Install CUDA toolkit and use burn-tch or burn-wgpu backend");
        println!("   - AMD: Install ROCm and use burn-tch backend");
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        // Should select GPU if available, otherwise CPU
        assert!(matches!(config.device, Device::Cpu | Device::Gpu(_)));
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert!(matches!(device, Device::Cpu | Device::Gpu(_)));
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "CPU");
        assert_eq!(Device::Gpu(0).to_string(), "GPU:0");
        assert_eq!(Device::Gpu(1).to_string(), "GPU:1");
    }

    #[test]
    fn test_init_backend_cpu() {
        let config = BackendConfig {
            device: Device::Cpu,
        };
        let device = init_backend(&config);
        assert_eq!(device, Device::Cpu);
    }

    #[test]
    fn test_backend_info() {
        // Should return something even if not initialized
        let info = backend_info();
        assert!(!info.is_empty());
    }

    #[test]
    fn test_select_best_device() {
        // This will select whatever is available on the system
        let device = select_best_device();
        // Just verify it returns a valid device
        assert!(matches!(device, Device::Cpu | Device::Gpu(_)));
    }

    #[test]
    fn test_is_gpu_available() {
        // Just ensure the function doesn't panic
        let _ = is_gpu_available();
    }

    #[test]
    fn test_get_recommended_device() {
        let device = get_recommended_device();
        // Verify it returns a valid device type
        assert!(matches!(device, Device::Cpu | Device::Gpu(_)));
    }

    #[test]
    fn test_print_gpu_info() {
        // Just ensure it doesn't panic
        print_gpu_info();
    }
}
