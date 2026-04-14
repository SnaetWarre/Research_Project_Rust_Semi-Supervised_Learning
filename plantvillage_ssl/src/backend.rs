//! Backend abstraction - Multi-backend support
//!
//! Supports both CUDA (GPU) and NdArray (CPU) backends with automatic selection.

use burn::backend::Autodiff;

// --------------------------------------------------------------------------------
// BACKEND SELECTION: CUDA (preferred) or NdArray (fallback)
// --------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub type DefaultBackend = burn_cuda::Cuda;

#[cfg(all(not(feature = "cuda"), any(feature = "ndarray", feature = "cpu")))]
pub type DefaultBackend = burn_ndarray::NdArray;

#[cfg(all(not(feature = "cuda"), not(feature = "ndarray"), not(feature = "cpu")))]
compile_error!("At least one backend (cuda, ndarray, or cpu) must be enabled!");

/// The default autodiff backend for training
pub type TrainingBackend = Autodiff<DefaultBackend>;

/// Get the default device
pub fn default_device() -> <DefaultBackend as burn::tensor::backend::Backend>::Device {
    <DefaultBackend as burn::tensor::backend::Backend>::Device::default()
}

/// Get a human-readable name for the current backend
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "CUDA (GPU)"
    }

    #[cfg(all(not(feature = "cuda"), any(feature = "ndarray", feature = "cpu")))]
    {
        "NdArray (CPU)"
    }
}
