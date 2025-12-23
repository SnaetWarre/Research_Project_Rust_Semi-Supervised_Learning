//! GPU Tensor - All operations stay on GPU
//!
//! This is a complete GPU tensor implementation that keeps data on GPU
//! and only transfers to CPU when explicitly requested.

use crate::Float;
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::cublas::{CudaBlas, Gemm};
use std::sync::Arc;

/// Cache for loaded CUDA kernels
struct KernelCache {
    elementwise_loaded: bool,
    activations_loaded: bool,
    matrix_ops_loaded: bool,
}

impl KernelCache {
    fn new() -> Self {
        Self {
            elementwise_loaded: false,
            activations_loaded: false,
            matrix_ops_loaded: false,
        }
    }
}

thread_local! {
    static KERNEL_CACHE: std::cell::RefCell<KernelCache> = std::cell::RefCell::new(KernelCache::new());
}

/// Load and compile CUDA kernel
fn load_kernel(device: &Arc<CudaDevice>, module_name: &str, kernel_code: &str, functions: &[&'static str]) -> Result<(), String> {
    use cudarc::nvrtc::compile_ptx;
    
    let ptx = compile_ptx(kernel_code)
        .map_err(|e| format!("Failed to compile CUDA kernel {}: {:?}", module_name, e))?;
    
    device.load_ptx(ptx, module_name, functions)
        .map_err(|e| format!("Failed to load PTX module {}: {:?}", module_name, e))?;
    
    Ok(())
}

/// Ensure kernels are loaded (public for use by layers)
pub fn ensure_kernels_loaded(device: &Arc<CudaDevice>, module: &str) -> Result<(), String> {
    KERNEL_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        match module {
            "elementwise" if !cache.elementwise_loaded => {
                let kernel_code = include_str!("kernels/elementwise.cu");
                load_kernel(device, "elementwise", kernel_code, &[
                    "add", "sub", "mul", "mul_scalar", "add_scalar",
                    "exp_elementwise", "log_elementwise", "sqrt_elementwise"
                ])?;
                cache.elementwise_loaded = true;
            }
            "activations" if !cache.activations_loaded => {
                let kernel_code = include_str!("kernels/activations.cu");
                load_kernel(device, "activations", kernel_code, &[
                    "relu_forward", "sigmoid_forward", "tanh_forward", "softmax_forward"
                ])?;
                cache.activations_loaded = true;
            }
            "matrix_ops" if !cache.matrix_ops_loaded => {
                // Note: matrix_ops.cu uses float, but we need double - will handle conversion
                let kernel_code = include_str!("kernels/matrix_ops.cu");
                load_kernel(device, "matrix_ops", kernel_code, &[
                    "argmax_rows", "sum_cols", "add_row_vector"
                ])?;
                cache.matrix_ops_loaded = true;
            }
            "data_prep" => {
                let kernel_code = include_str!("kernels/data_prep.cu");
                load_kernel(device, "data_prep", kernel_code, &[
                    "hwc_to_chw_norm_augment",
                    "cutout_apply",
                    "color_jitter"
                ])?;
            }
            // CNN layers use im2col (matrix multiplication) - no special kernels needed
            _ => {}
        }
        Ok(())
    })
}

/// A tensor that lives entirely on the GPU
#[derive(Clone)]
pub struct GpuTensor {
    /// GPU data
    data: CudaSlice<Float>,
    /// Shape (rows, cols)
    shape: (usize, usize),
    /// CUDA device
    device: Arc<CudaDevice>,
}

impl GpuTensor {
    /// Create a new GPU tensor from CPU data
    pub fn from_slice(data: &[Float], rows: usize, cols: usize, device: Arc<CudaDevice>) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err(format!("Data length {} doesn't match shape {}x{}", data.len(), rows, cols));
        }

        let gpu_data = device.htod_copy(data.to_vec())
            .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

        Ok(Self {
            data: gpu_data,
            shape: (rows, cols),
            device,
        })
    }

    /// Create zeros on GPU
    pub fn zeros(rows: usize, cols: usize, device: Arc<CudaDevice>) -> Result<Self, String> {
        let size = rows * cols;
        let gpu_data = device.alloc_zeros::<Float>(size)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        Ok(Self {
            data: gpu_data,
            shape: (rows, cols),
            device,
        })
    }

    /// Create ones on GPU
    pub fn ones(rows: usize, cols: usize, device: Arc<CudaDevice>) -> Result<Self, String> {
        let size = rows * cols;
        let ones_vec = vec![1.0; size];
        let gpu_data = device.htod_copy(ones_vec)
            .map_err(|e| format!("Failed to copy to GPU: {:?}", e))?;

        Ok(Self {
            data: gpu_data,
            shape: (rows, cols),
            device,
        })
    }

    /// Create random uniform tensor on GPU
    pub fn random_uniform(rows: usize, cols: usize, min: Float, max: Float, device: Arc<CudaDevice>) -> Result<Self, String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = rows * cols;
        let data: Vec<Float> = (0..size)
            .map(|_| rng.gen::<Float>() * (max - min) + min)
            .collect();

        Self::from_slice(&data, rows, cols, device)
    }

    /// Get shape
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get rows
    pub fn rows(&self) -> usize {
        self.shape.0
    }

    /// Get cols
    pub fn cols(&self) -> usize {
        self.shape.1
    }

    /// Get total elements
    pub fn len(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get raw GPU data
    pub fn data(&self) -> &CudaSlice<Float> {
        &self.data
    }

    /// Get mutable raw GPU data
    pub fn data_mut(&mut self) -> &mut CudaSlice<Float> {
        &mut self.data
    }

    /// Copy data back to CPU
    pub fn to_vec(&self) -> Result<Vec<Float>, String> {
        self.device.dtoh_sync_copy(&self.data)
            .map_err(|e| format!("Failed to copy from GPU: {:?}", e))
    }

    /// Matrix multiplication (stays on GPU!)
    pub fn matmul(&self, other: &GpuTensor) -> Result<GpuTensor, String> {
        use std::os::raw::c_int;
        use cudarc::cublas::sys::cublasOperation_t;

        let m = self.rows() as c_int;
        let k = self.cols() as c_int;
        let n = other.cols() as c_int;

        if self.cols() != other.rows() {
            return Err(format!("Matrix dimension mismatch: {}x{} × {}x{}",
                self.rows(), self.cols(), other.rows(), other.cols()));
        }

        // Create cuBLAS handle for this operation
        let blas = CudaBlas::new(self.device.clone())
            .map_err(|e| format!("Failed to create cuBLAS handle: {:?}", e))?;

        // Allocate output on GPU
        let mut result = GpuTensor::zeros(self.rows(), other.cols(), self.device.clone())?;

        // cuBLAS GEMM: C = alpha * A * B + beta * C
        // Handle row-major to column-major by computing C^T = B^T * A^T
        let config = cudarc::cublas::GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n,
            n: m,
            k,
            alpha: 1.0,
            lda: n,
            ldb: k,
            beta: 0.0,
            ldc: n,
        };

        unsafe {
            blas.gemm(config, &other.data, &self.data, &mut result.data)
                .map_err(|e| format!("cuBLAS gemm failed: {:?}", e))?;
        }

        // cuBLAS handles the row-major to column-major conversion, no extra transpose needed
        Ok(result)
    }

    /// Transpose in place (swap interpretation, actual transpose done in kernel later)
    fn transpose_inplace(&mut self) -> Result<(), String> {
        // For now, we do actual data reordering
        let (rows, cols) = self.shape;
        let vec = self.to_vec()?;

        let mut transposed = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = vec[i * cols + j];
            }
        }

        self.data = self.device.htod_copy(transposed)
            .map_err(|e| format!("Failed to copy transposed data: {:?}", e))?;
        self.shape = (cols, rows);

        Ok(())
    }

    /// Element-wise addition (stays on GPU!)
    pub fn add(&self, other: &GpuTensor) -> Result<GpuTensor, String> {
        if self.shape != other.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }

        ensure_kernels_loaded(&self.device, "elementwise")?;
        let kernel = self.device.get_func("elementwise", "add")
            .ok_or("Failed to get add kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let other_slice = other.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &other_slice,
                &mut result_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Element-wise subtraction (stays on GPU!)
    pub fn sub(&self, other: &GpuTensor) -> Result<GpuTensor, String> {
        if self.shape != other.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }

        ensure_kernels_loaded(&self.device, "elementwise")?;
        let kernel = self.device.get_func("elementwise", "sub")
            .ok_or("Failed to get sub kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let other_slice = other.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &other_slice,
                &mut result_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Element-wise multiplication (stays on GPU!)
    pub fn mul(&self, other: &GpuTensor) -> Result<GpuTensor, String> {
        if self.shape != other.shape {
            return Err(format!("Shape mismatch: {:?} vs {:?}", self.shape, other.shape));
        }

        ensure_kernels_loaded(&self.device, "elementwise")?;
        let kernel = self.device.get_func("elementwise", "mul")
            .ok_or("Failed to get mul kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let other_slice = other.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &other_slice,
                &mut result_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: Float) -> Result<GpuTensor, String> {
        ensure_kernels_loaded(&self.device, "elementwise")?;
        let kernel = self.device.get_func("elementwise", "mul_scalar")
            .ok_or("Failed to get mul_scalar kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                scalar,
                &mut result_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// ReLU activation (in place)
    pub fn relu_inplace(&mut self) -> Result<(), String> {
        ensure_kernels_loaded(&self.device, "activations")?;
        let kernel = self.device.get_func("activations", "relu_forward")
            .ok_or("Failed to get relu_forward kernel")?;

        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Create temporary copy for input (kernel expects separate input/output)
        let input_copy = self.clone();

        unsafe {
            let input_slice = input_copy.data.slice(..);
            let mut output_slice = self.data.slice_mut(..);
            kernel.launch(cfg, (
                &input_slice,
                &mut output_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(())
    }

    /// ReLU (returns new tensor)
    pub fn relu(&self) -> Result<GpuTensor, String> {
        let mut result = self.clone();
        result.relu_inplace()?;
        Ok(result)
    }

    /// Sigmoid activation (in place)
    pub fn sigmoid_inplace(&mut self) -> Result<(), String> {
        ensure_kernels_loaded(&self.device, "activations")?;
        let kernel = self.device.get_func("activations", "sigmoid_forward")
            .ok_or("Failed to get sigmoid_forward kernel")?;

        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Create temporary copy for input (kernel expects separate input/output)
        let input_copy = self.clone();

        unsafe {
            let input_slice = input_copy.data.slice(..);
            let mut output_slice = self.data.slice_mut(..);
            kernel.launch(cfg, (
                &input_slice,
                &mut output_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(())
    }

    /// Sigmoid (returns new tensor)
    pub fn sigmoid(&self) -> Result<GpuTensor, String> {
        let mut result = self.clone();
        result.sigmoid_inplace()?;
        Ok(result)
    }

    /// Tanh activation (in place)
    pub fn tanh_inplace(&mut self) -> Result<(), String> {
        ensure_kernels_loaded(&self.device, "activations")?;
        let kernel = self.device.get_func("activations", "tanh_forward")
            .ok_or("Failed to get tanh_forward kernel")?;

        let n = self.len() as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Create temporary copy for input (kernel expects separate input/output)
        let input_copy = self.clone();

        unsafe {
            let input_slice = input_copy.data.slice(..);
            let mut output_slice = self.data.slice_mut(..);
            kernel.launch(cfg, (
                &input_slice,
                &mut output_slice,
                n as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(())
    }

    /// Tanh (returns new tensor)
    pub fn tanh(&self) -> Result<GpuTensor, String> {
        let mut result = self.clone();
        result.tanh_inplace()?;
        Ok(result)
    }

    /// Softmax across rows (for batched predictions)
    pub fn softmax(&self) -> Result<GpuTensor, String> {
        ensure_kernels_loaded(&self.device, "activations")?;
        let kernel = self.device.get_func("activations", "softmax_forward")
            .ok_or("Failed to get softmax_forward kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let rows = self.rows() as u32;
        let cols = self.cols() as u32;
        // Use power-of-two thread count for stable shared-memory reductions in CUDA kernel
        let mut pow2 = 1u32;
        while pow2 < cols { pow2 <<= 1; }
        let threads_per_block = pow2.min(256);

        let cfg = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: (threads_per_block * 2) as u32 * std::mem::size_of::<Float>() as u32,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &mut result_slice,
                rows as i32,
                cols as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Sum all elements (returns scalar on CPU)
    pub fn sum(&self) -> Result<Float, String> {
        let vec = self.to_vec()?;
        Ok(vec.iter().sum())
    }

    /// Mean of all elements
    pub fn mean(&self) -> Result<Float, String> {
        Ok(self.sum()? / self.len() as Float)
    }

    /// Sum along axis 0 (sum columns, return row vector)
    pub fn sum_axis_0(&self) -> Result<GpuTensor, String> {
        ensure_kernels_loaded(&self.device, "matrix_ops")?;
        let kernel = self.device.get_func("matrix_ops", "sum_cols")
            .ok_or("Failed to get sum_cols kernel")?;

        let mut result = GpuTensor::zeros(1, self.cols(), self.device.clone())?;
        let rows = self.rows() as u32;
        let cols = self.cols() as u32;
        let threads_per_block = 256;
        let blocks = (cols + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &mut result_slice,
                rows as i32,
                cols as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Broadcast add row vector to each row
    pub fn add_row_vector(&self, vector: &GpuTensor) -> Result<GpuTensor, String> {
        if vector.rows() != 1 || vector.cols() != self.cols() {
            return Err(format!("Vector shape mismatch: expected (1, {}), got {:?}",
                self.cols(), vector.shape()));
        }

        ensure_kernels_loaded(&self.device, "matrix_ops")?;
        let kernel = self.device.get_func("matrix_ops", "add_row_vector")
            .ok_or("Failed to get add_row_vector kernel")?;

        let mut result = GpuTensor::zeros(self.rows(), self.cols(), self.device.clone())?;
        let rows = self.rows() as u32;
        let cols = self.cols() as u32;
        let n = (rows * cols) as u32;
        let threads_per_block = 256;
        let blocks = (n + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let vector_slice = vector.data.slice(..);
            let mut result_slice = result.data.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &vector_slice,
                &mut result_slice,
                rows as i32,
                cols as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        Ok(result)
    }

    /// Transpose (returns new tensor)
    pub fn transpose(&self) -> Result<GpuTensor, String> {
        let vec = self.to_vec()?;
        let (rows, cols) = self.shape;
        let mut transposed = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = vec[i * cols + j];
            }
        }

        GpuTensor::from_slice(&transposed, cols, rows, self.device.clone())
    }

    /// Argmax along rows (returns indices on GPU, then copy to CPU)
    pub fn argmax_rows(&self) -> Result<Vec<i32>, String> {
        ensure_kernels_loaded(&self.device, "matrix_ops")?;
        let kernel = self.device.get_func("matrix_ops", "argmax_rows")
            .ok_or("Failed to get argmax_rows kernel")?;

        let rows = self.rows() as u32;
        let cols = self.cols() as u32;
        let threads_per_block = 256;
        let blocks = (rows + threads_per_block - 1) / threads_per_block;

        // Allocate GPU memory for output indices
        let mut indices_gpu = self.device.alloc_zeros::<i32>(rows as usize)
            .map_err(|e| format!("Failed to allocate GPU memory: {:?}", e))?;

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let self_slice = self.data.slice(..);
            let mut indices_slice = indices_gpu.slice_mut(..);
            kernel.launch(cfg, (
                &self_slice,
                &mut indices_slice,
                rows as i32,
                cols as i32,
            )).map_err(|e| format!("Kernel launch failed: {:?}", e))?;
        }

        self.device.synchronize()
            .map_err(|e| format!("Device sync failed: {:?}", e))?;

        // Copy results back to CPU
        let indices = self.device.dtoh_sync_copy(&indices_gpu)
            .map_err(|e| format!("Failed to copy indices from GPU: {:?}", e))?;

        Ok(indices)
    }
}

/// Context for GPU operations
pub struct GpuContext {
    device: Arc<CudaDevice>,
}

impl GpuContext {
    /// Create a new GPU context
    pub fn new(device_id: usize) -> Result<Self, String> {
        let device = CudaDevice::new(device_id)
            .map_err(|e| format!("Failed to create CUDA device: {:?}", e))?;

        Ok(Self { device })
    }

    /// Get device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Create zeros tensor
    pub fn zeros(&self, rows: usize, cols: usize) -> Result<GpuTensor, String> {
        GpuTensor::zeros(rows, cols, self.device.clone())
    }

    /// Create ones tensor
    pub fn ones(&self, rows: usize, cols: usize) -> Result<GpuTensor, String> {
        GpuTensor::ones(rows, cols, self.device.clone())
    }

    /// Create random tensor
    pub fn random_uniform(&self, rows: usize, cols: usize, min: Float, max: Float) -> Result<GpuTensor, String> {
        GpuTensor::random_uniform(rows, cols, min, max, self.device.clone())
    }

    /// Create tensor from CPU data
    pub fn from_slice(&self, data: &[Float], rows: usize, cols: usize) -> Result<GpuTensor, String> {
        GpuTensor::from_slice(data, rows, cols, self.device.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tensor_creation() {
        let ctx = GpuContext::new(0).expect("Failed to create GPU context");

        let t = ctx.zeros(10, 20).expect("Failed to create zeros");
        assert_eq!(t.shape(), (10, 20));

        let vec = t.to_vec().expect("Failed to copy to CPU");
        assert_eq!(vec.len(), 200);
        assert!(vec.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gpu_matmul() {
        let ctx = GpuContext::new(0).expect("Failed to create GPU context");

        let a = ctx.from_slice(&[1.0, 2.0, 3.0, 4.0], 2, 2).expect("Failed to create A");
        let b = ctx.from_slice(&[5.0, 6.0, 7.0, 8.0], 2, 2).expect("Failed to create B");

        let c = a.matmul(&b).expect("Failed to multiply");
        let result = c.to_vec().expect("Failed to copy result");

        // A * B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //       = [[19, 22], [43, 50]]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 19.0).abs() < 1e-6);
        assert!((result[1] - 22.0).abs() < 1e-6);
        assert!((result[2] - 43.0).abs() < 1e-6);
        assert!((result[3] - 50.0).abs() < 1e-6);
    }
}
