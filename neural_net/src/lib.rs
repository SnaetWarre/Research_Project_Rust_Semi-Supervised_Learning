//! # GPU Neural Network Library
//!
//! A GPU-only neural network library built from scratch using CUDA.
//! All operations run on GPU with minimal CPU transfers.

#[cfg(feature = "cuda")]
pub mod cifar10;
#[cfg(feature = "cuda")]
pub mod gpu_tensor;
#[cfg(feature = "cuda")]
pub mod gpu_layer;
#[cfg(feature = "cuda")]
pub mod pseudo_label;
#[cfg(feature = "cuda")]
pub mod model;

// Re-export commonly used types
#[cfg(feature = "cuda")]
pub use cifar10::{Cifar10Dataset, Cifar10Image, DataLoader, DatasetSplit, CLASS_NAMES};
#[cfg(feature = "cuda")]
pub use gpu_tensor::{GpuTensor, GpuContext};
#[cfg(feature = "cuda")]
pub use gpu_layer::{GpuLayer, GpuDense, GpuBatchNorm, GpuDropout, GpuNetwork, GpuSGD, GpuSoftmaxCrossEntropy, ActivationType, GpuConv2D, GpuMaxPool2D, GpuFlatten};
#[cfg(feature = "cuda")]
pub use pseudo_label::{PseudoLabeler, PseudoLabelConfig, PseudoLabelResult, PseudoLabelingHistory};
#[cfg(feature = "cuda")]
pub use model::{save_model, load_model, ModelMetadata, LayerConfig};

/// Common type alias for our neural network computations
pub type Float = f64;

#[cfg(test)]
mod tests {
    #[test]
    fn library_loads() {
        // Basic smoke test to ensure the library compiles
        assert_eq!(2 + 2, 4);
    }
}
