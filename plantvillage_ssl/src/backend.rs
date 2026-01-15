//! Backend abstraction - CUDA GPU backend
//!
//! This module strictly enforces usage of the CUDA backend.
//! CPU fallbacks have been removed to ensure high-performance GPU execution.

use burn::backend::Autodiff;

// --------------------------------------------------------------------------------
// BACKEND SELECTION: STRICTLY CUDA
// --------------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub type DefaultBackend = burn_cuda::Cuda;

// Compile-time error if CUDA is not enabled!
#[cfg(not(feature = "cuda"))]
compile_error!("CUDA feature is required! CPU fallback has been disabled.");

/// The default autodiff backend for training
pub type TrainingBackend = Autodiff<DefaultBackend>;

/// Get the default device (CUDA)
pub fn default_device() -> <DefaultBackend as burn::tensor::backend::Backend>::Device {
    #[cfg(feature = "cuda")]
    {
        // Default to the first GPU
        burn_cuda::CudaDevice::default()
    }
}

/// Get a human-readable name for the current backend
pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    { "CUDA (GPU)" }
}
