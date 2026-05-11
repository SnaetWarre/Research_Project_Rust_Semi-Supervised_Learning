//! Backend abstraction for CPU and CUDA builds.
//!
//! The CLI chooses between the compiled backends at runtime.

use burn::backend::Autodiff;

#[cfg(any(feature = "cpu", feature = "ndarray"))]
pub type CpuBackend = burn_ndarray::NdArray;

#[cfg(feature = "cuda")]
pub type CudaBackend = burn_cuda::Cuda;

#[cfg(any(feature = "cpu", feature = "ndarray"))]
pub type DefaultBackend = CpuBackend;

#[cfg(all(not(feature = "cpu"), not(feature = "ndarray"), feature = "cuda"))]
pub type DefaultBackend = CudaBackend;

#[cfg(all(not(feature = "cpu"), not(feature = "ndarray"), not(feature = "cuda")))]
compile_error!("At least one backend (cuda, ndarray, or cpu) must be enabled!");

/// The default autodiff backend for training
pub type TrainingBackend = Autodiff<DefaultBackend>;

/// Get the default device
pub fn default_device() -> <DefaultBackend as burn::tensor::backend::Backend>::Device {
    <DefaultBackend as burn::tensor::backend::Backend>::Device::default()
}

/// Get a human-readable name for the current backend
pub fn backend_name() -> &'static str {
    default_backend_name()
}

pub fn default_backend_name() -> &'static str {
    #[cfg(any(feature = "cpu", feature = "ndarray"))]
    return cpu_backend_name();

    #[cfg(all(not(feature = "cpu"), not(feature = "ndarray"), feature = "cuda"))]
    return cuda_backend_name();
}

#[cfg(any(feature = "cpu", feature = "ndarray"))]
pub fn cpu_backend_name() -> &'static str {
    "NdArray (CPU)"
}

#[cfg(feature = "cuda")]
pub fn cuda_backend_name() -> &'static str {
    "CUDA (GPU)"
}
