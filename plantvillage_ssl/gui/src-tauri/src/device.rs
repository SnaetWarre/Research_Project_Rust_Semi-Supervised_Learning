//! Device Detection and Adaptive Configuration
//!
//! This module automatically detects the device type (Desktop vs Mobile)
//! and provides appropriate training configurations for each platform.

use serde::{Deserialize, Serialize};

/// Device type detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    /// Desktop/Laptop with GPU (CUDA)
    Desktop,
    /// Mobile device (iOS/Android) with CPU only
    Mobile,
}

/// Device capabilities and recommended settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub device_type: DeviceType,
    pub has_gpu: bool,
    pub backend_name: String,
    pub recommended_batch_size: usize,
    pub recommended_epochs: usize,
    pub max_memory_mb: usize,
}

impl DeviceType {
    /// Detect the current device type
    pub fn detect() -> Self {
        // Check if we're running on mobile platform
        #[cfg(target_os = "android")]
        {
            return DeviceType::Mobile;
        }

        #[cfg(target_os = "ios")]
        {
            return DeviceType::Mobile;
        }

        // Desktop platforms (Windows, Linux, macOS)
        DeviceType::Desktop
    }

    /// Check if GPU acceleration is available for this device
    pub fn has_gpu_available(&self) -> bool {
        match self {
            DeviceType::Desktop => {
                // Check if CUDA is compiled in
                #[cfg(feature = "cuda")]
                {
                    true
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            }
            DeviceType::Mobile => {
                // TODO: Add WGPU support for mobile GPU acceleration
                false
            }
        }
    }

    /// Get the backend name for this device
    pub fn backend_name(&self) -> &'static str {
        match self {
            DeviceType::Desktop => {
                #[cfg(feature = "cuda")]
                {
                    "CUDA"
                }
                #[cfg(not(feature = "cuda"))]
                {
                    "NdArray (CPU)"
                }
            }
            DeviceType::Mobile => "NdArray (CPU)",
        }
    }

    /// Get device capabilities with recommended settings
    pub fn capabilities(&self) -> DeviceCapabilities {
        match self {
            DeviceType::Desktop => DeviceCapabilities {
                device_type: *self,
                has_gpu: self.has_gpu_available(),
                backend_name: self.backend_name().to_string(),
                recommended_batch_size: 32, // Standard desktop batch size
                recommended_epochs: 50,     // Full training epochs
                max_memory_mb: 8192,        // 8GB typical GPU
            },
            DeviceType::Mobile => DeviceCapabilities {
                device_type: *self,
                has_gpu: false,
                backend_name: self.backend_name().to_string(),
                recommended_batch_size: 8, // Small batches for mobile
                recommended_epochs: 2,     // Quick retraining only
                max_memory_mb: 512,        // Conservative mobile RAM
            },
        }
    }
}

/// Adaptive training configuration based on device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveTrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub use_augmentation: bool,
    pub device_type: DeviceType,
}

impl AdaptiveTrainingConfig {
    /// Create an adaptive configuration for the current device
    pub fn for_current_device() -> Self {
        let device = DeviceType::detect();
        Self::for_device(device)
    }

    /// Create configuration for a specific device type
    pub fn for_device(device: DeviceType) -> Self {
        match device {
            DeviceType::Desktop => Self {
                batch_size: 32,
                epochs: 50,
                learning_rate: 0.0001,
                use_augmentation: true,
                device_type: device,
            },
            DeviceType::Mobile => Self {
                batch_size: 8,
                epochs: 2,
                learning_rate: 0.0001,
                use_augmentation: true, // Still use augmentation to prevent overfitting
                device_type: device,
            },
        }
    }

    /// Configuration optimized for SSL retraining (lighter than full training)
    pub fn for_ssl_retraining() -> Self {
        let device = DeviceType::detect();
        match device {
            DeviceType::Desktop => Self {
                batch_size: 64, // Larger batches on GPU
                epochs: 5,      // Quick retraining
                learning_rate: 0.0001,
                use_augmentation: true,
                device_type: device,
            },
            DeviceType::Mobile => Self {
                batch_size: 8, // Keep small for memory
                epochs: 2,     // Minimal epochs on mobile
                learning_rate: 0.0001,
                use_augmentation: true,
                device_type: device,
            },
        }
    }

    /// Scale configuration for a specific number of samples
    /// This helps when working with smaller datasets or pseudo-label batches
    pub fn scale_for_samples(&mut self, num_samples: usize) {
        // If we have fewer samples than batch_size, reduce it
        if num_samples < self.batch_size {
            self.batch_size = (num_samples / 2).max(4); // At least 4, at most half the samples
        }

        // Adjust epochs based on total iterations
        let iterations_per_epoch = (num_samples + self.batch_size - 1) / self.batch_size;

        // If very few iterations, might want to increase epochs slightly
        if iterations_per_epoch < 10 && self.epochs < 5 {
            self.epochs = 3; // Do at least 3 epochs for tiny datasets
        }
    }
}

/// Get device information as a JSON string for the frontend
pub fn get_device_info() -> String {
    let device = DeviceType::detect();
    let capabilities = device.capabilities();
    serde_json::to_string_pretty(&capabilities).unwrap_or_else(|_| "{}".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_detection() {
        let device = DeviceType::detect();
        // Should work on any platform
        assert!(matches!(device, DeviceType::Desktop | DeviceType::Mobile));
    }

    #[test]
    fn test_adaptive_config_scaling() {
        let mut config = AdaptiveTrainingConfig::for_ssl_retraining();

        // Test scaling down for small dataset
        config.scale_for_samples(50);
        assert!(config.batch_size <= 25); // Should be at most half
        assert!(config.batch_size >= 4); // But at least 4
    }

    #[test]
    fn test_mobile_config_is_lighter() {
        let desktop = AdaptiveTrainingConfig::for_device(DeviceType::Desktop);
        let mobile = AdaptiveTrainingConfig::for_device(DeviceType::Mobile);

        assert!(mobile.batch_size < desktop.batch_size);
        assert!(mobile.epochs <= desktop.epochs);
    }
}
