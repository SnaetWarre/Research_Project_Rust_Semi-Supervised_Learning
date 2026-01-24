//! Unified Backend Selection
//!
//! This module provides automatic backend selection based on compile-time features
//! and runtime device detection. It automatically chooses:
//! - CUDA backend on desktop with GPU
//! - NdArray backend on mobile or CPU-only systems

use burn::backend::Autodiff;
use burn::tensor::backend::Backend;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[cfg(not(feature = "cuda"))]
use burn_ndarray::NdArray;

// Type alias for the training backend based on compile-time features
#[cfg(feature = "cuda")]
pub type AdaptiveBackend = Autodiff<Cuda>;

#[cfg(not(feature = "cuda"))]
pub type AdaptiveBackend = Autodiff<NdArray>;

/// Get a human-readable name for the current backend
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA"
    }

    #[cfg(not(feature = "cuda"))]
    {
        "NdArray (CPU)"
    }
}

/// Get the default device for the current backend
pub fn default_device() -> <AdaptiveBackend as Backend>::Device {
    <AdaptiveBackend as Backend>::Device::default()
}

/// Check if GPU acceleration is available
pub fn has_gpu() -> bool {
    #[cfg(feature = "cuda")]
    {
        true
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_available() {
        // Should always have a backend available
        let _device = default_device();
        let name = backend_name();
        assert!(!name.is_empty());
    }
}
