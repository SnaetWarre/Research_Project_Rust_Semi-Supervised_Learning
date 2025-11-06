//! CNN Layers for GPU - Conv2D, MaxPool2D, Flatten
//!
//! Using im2col approach: convert convolution to matrix multiplication
//! This allows us to use our existing GPU matmul which is highly optimized

use crate::{Float, gpu_tensor::GpuTensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use crate::gpu_layer::GpuLayer;
use std::any::Any;

/// GPU Conv2D Layer using im2col (image-to-column) approach
/// Converts convolution to matrix multiplication for GPU efficiency
pub struct GpuConv2D {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    /// Weights (out_channels, in_channels * kernel_size * kernel_size)
    weights: GpuTensor,
    /// Bias (out_channels)
    bias: GpuTensor,
    /// Weight gradients
    weights_grad: GpuTensor,
    /// Bias gradients
    bias_grad: GpuTensor,
    /// Cached input for backward
    cached_input: Option<GpuTensor>,
    cached_input_shape: Option<(usize, usize, usize, usize)>, // (batch, channels, height, width)
    cached_output_shape: Option<(usize, usize, usize, usize)>,
    cached_im2col: Option<GpuTensor>, // Cached im2col matrix
    cached_output: Option<GpuTensor>, // Cached output before activation
    use_relu: bool, // Whether to apply ReLU activation
    /// Device
    device: Arc<CudaDevice>,
    /// Layer name
    name: String,
}

impl GpuConv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        device: Arc<CudaDevice>,
    ) -> Result<Self, String> {
        Self::new_with_activation(in_channels, out_channels, kernel_size, stride, padding, true, device)
    }

    pub fn new_with_activation(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        use_relu: bool,
        device: Arc<CudaDevice>,
    ) -> Result<Self, String> {
        // He initialization for weights (good for ReLU)
        let fan_in = in_channels * kernel_size * kernel_size;
        let std_dev = (2.0 / fan_in as Float).sqrt();
        
        let weight_size = in_channels * kernel_size * kernel_size;
        let weights = GpuTensor::random_uniform(
            out_channels,
            weight_size,
            -std_dev,
            std_dev,
            device.clone(),
        )?;
        
        let bias = GpuTensor::zeros(1, out_channels, device.clone())?;
        let weights_grad = GpuTensor::zeros(out_channels, weight_size, device.clone())?;
        let bias_grad = GpuTensor::zeros(1, out_channels, device.clone())?;

        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            weights_grad,
            bias_grad,
            cached_input: None,
            cached_input_shape: None,
            cached_output_shape: None,
            cached_im2col: None,
            cached_output: None,
            use_relu,
            device,
            name: format!("GpuConv2D({}x{}x{} -> {}x{}x{})", in_channels, kernel_size, kernel_size, out_channels, kernel_size, kernel_size),
        })
    }

    /// Calculate output dimensions
    fn output_dims(&self, in_height: usize, in_width: usize) -> (usize, usize) {
        let out_height = (in_height + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_width = (in_width + 2 * self.padding - self.kernel_size) / self.stride + 1;
        (out_height, out_width)
    }

    /// Convert image to column matrix (im2col) for convolution via matrix multiplication
    /// Input: (batch, channels*height*width) flattened
    /// Output: (batch*out_h*out_w, in_channels*kernel_size*kernel_size)
    fn im2col(&self, input: &GpuTensor, batch: usize, channels: usize, height: usize, width: usize) -> Result<GpuTensor, String> {
        let (out_height, out_width) = self.output_dims(height, width);
        let out_spatial = out_height * out_width;
        let kernel_elements = channels * self.kernel_size * self.kernel_size;
        
        // Get input data
        let input_vec = input.to_vec()?;
        
        // Create im2col matrix
        let mut im2col_data = vec![0.0; batch * out_spatial * kernel_elements];
        
        for n in 0..batch {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = n * out_spatial + oh * out_width + ow;
                    
                    for c in 0..channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = (oh * self.stride).saturating_add(kh).saturating_sub(self.padding);
                                let iw = (ow * self.stride).saturating_add(kw).saturating_sub(self.padding);
                                
                                let col_idx = c * (self.kernel_size * self.kernel_size) + kh * self.kernel_size + kw;
                                let im2col_idx = out_idx * kernel_elements + col_idx;
                                
                                if ih < height && iw < width {
                                    let input_idx = n * (channels * height * width) +
                                                   c * (height * width) +
                                                   ih * width + iw;
                                    im2col_data[im2col_idx] = input_vec[input_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        GpuTensor::from_slice(&im2col_data, batch * out_spatial, kernel_elements, self.device.clone())
    }

    /// Convert column matrix back to image (col2im) for backward pass
    fn col2im(&self, col: &GpuTensor, batch: usize, channels: usize, height: usize, width: usize) -> Result<GpuTensor, String> {
        let (out_height, out_width) = self.output_dims(height, width);
        let out_spatial = out_height * out_width;
        let kernel_elements = channels * self.kernel_size * self.kernel_size;
        
        let col_vec = col.to_vec()?;
        let mut im_data = vec![0.0; batch * channels * height * width];
        
        for n in 0..batch {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let out_idx = n * out_spatial + oh * out_width + ow;
                    
                    for c in 0..channels {
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = (oh * self.stride).saturating_add(kh).saturating_sub(self.padding);
                                let iw = (ow * self.stride).saturating_add(kw).saturating_sub(self.padding);
                                
                                if ih < height && iw < width {
                                    let col_idx = c * (self.kernel_size * self.kernel_size) + kh * self.kernel_size + kw;
                                    let col_val = col_vec[out_idx * kernel_elements + col_idx];
                                    
                                    let im_idx = n * (channels * height * width) +
                                                c * (height * width) +
                                                ih * width + iw;
                                    im_data[im_idx] += col_val;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        GpuTensor::from_slice(&im_data, batch, channels * height * width, self.device.clone())
    }
}

impl GpuLayer for GpuConv2D {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        // Input is flattened: (batch, channels*height*width)
        let batch_size = input.rows();
        let flattened_size = input.cols();
        
        // Infer spatial dimensions
        if flattened_size % self.in_channels != 0 {
            return Err(format!("Input size {} not divisible by in_channels {}", flattened_size, self.in_channels));
        }
        let spatial_size = flattened_size / self.in_channels;
        let height = (spatial_size as f64).sqrt() as usize;
        let width = height;
        
        if height * width != spatial_size {
            return Err(format!("Cannot infer square spatial dimensions from size {}", spatial_size));
        }
        
        let (out_height, out_width) = self.output_dims(height, width);
        let out_flattened = self.out_channels * out_height * out_width;
        
        // Cache shapes
        self.cached_input = Some(input.clone());
        self.cached_input_shape = Some((batch_size, self.in_channels, height, width));
        self.cached_output_shape = Some((batch_size, self.out_channels, out_height, out_width));
        
        // Convert to im2col
        let im2col = self.im2col(input, batch_size, self.in_channels, height, width)?;
        self.cached_im2col = Some(im2col.clone());
        
        // Convolution = im2col * weights^T
        // im2col: (batch*out_h*out_w, in_c*k*k)
        // weights: (out_channels, in_c*k*k)
        // output: (batch*out_h*out_w, out_channels)
        let conv_output = im2col.matmul(&self.weights.transpose()?)?;
        
        // Reshape: (batch*out_h*out_w, out_channels) -> (batch, out_channels*out_h*out_w)
        let conv_vec = conv_output.to_vec()?;
        let mut output_data = vec![0.0; batch_size * out_flattened];
        
        for n in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_spatial = out_height * out_width;
                        let out_idx = n * out_flattened + oc * out_spatial + oh * out_width + ow;
                        let conv_idx = (n * out_spatial + oh * out_width + ow) * self.out_channels + oc;
                        output_data[out_idx] = conv_vec[conv_idx];
                    }
                }
            }
        }
        
        let output = GpuTensor::from_slice(&output_data, batch_size, out_flattened, self.device.clone())?;
        
        // Add bias (broadcast to each spatial position)
        // For each channel, add its bias to all spatial positions
        let bias_vec = self.bias.to_vec()?;
        let output_vec = output.to_vec()?;
        let mut output_with_bias = vec![0.0; batch_size * out_flattened];
        
        for n in 0..batch_size {
            for oc in 0..self.out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let out_spatial = out_height * out_width;
                        let out_idx = n * out_flattened + oc * out_spatial + oh * out_width + ow;
                        output_with_bias[out_idx] = output_vec[out_idx] + bias_vec[oc];
                    }
                }
            }
        }
        
        let output_final = GpuTensor::from_slice(&output_with_bias, batch_size, out_flattened, self.device.clone())?;
        
        // Cache pre-activation output for backward pass
        self.cached_output = Some(output_final.clone());
        
        // Apply ReLU if needed
        if self.use_relu {
            let mut output_relu = output_final.clone();
            output_relu.relu_inplace()?;
            Ok(output_relu)
        } else {
            Ok(output_final)
        }
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        let im2col = self.cached_im2col.as_ref()
            .ok_or("Missing cached im2col matrix")?;
        let (batch, in_c, in_h, in_w) = self.cached_input_shape
            .ok_or("Missing cached input shape")?;
        let (_, out_c, out_h, out_w) = self.cached_output_shape
            .ok_or("Missing cached output shape")?;
        
        // Apply ReLU derivative if needed
        let grad_output_after_relu = if self.use_relu {
            let cached_out = self.cached_output.as_ref()
                .ok_or("Missing cached output for ReLU backward")?;
            // ReLU derivative: 1 if x > 0, else 0
            let cached_vec = cached_out.to_vec()?;
            let grad_vec = grad_output.to_vec()?;
            let mut relu_grad = vec![0.0; grad_vec.len()];
            for i in 0..grad_vec.len() {
                relu_grad[i] = if cached_vec[i] > 0.0 { grad_vec[i] } else { 0.0 };
            }
            GpuTensor::from_slice(&relu_grad, grad_output.rows(), grad_output.cols(), self.device.clone())?
        } else {
            grad_output.clone()
        };
        
        // Reshape grad_output: (batch, out_c*out_h*out_w) -> (batch*out_h*out_w, out_c)
        let grad_out_vec = grad_output_after_relu.to_vec()?;
        let out_spatial = out_h * out_w;
        let mut grad_out_reshaped = vec![0.0; batch * out_spatial * out_c];
        
        for n in 0..batch {
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let orig_idx = n * (out_c * out_spatial) + oc * out_spatial + oh * out_w + ow;
                        let new_idx = (n * out_spatial + oh * out_w + ow) * out_c + oc;
                        grad_out_reshaped[new_idx] = grad_out_vec[orig_idx];
                    }
                }
            }
        }
        
        let grad_out_mat = GpuTensor::from_slice(&grad_out_reshaped, batch * out_spatial, out_c, self.device.clone())?;
        
        // Weight gradient: grad_output^T * im2col
        // grad_output: (batch*out_h*out_w, out_c)
        // im2col: (batch*out_h*out_w, in_c*k*k)
        // weights_grad: (out_c, in_c*k*k)
        self.weights_grad = grad_out_mat.transpose()?.matmul(im2col)?;
        
        // Bias gradient: sum grad_output over batch and spatial dimensions
        let bias_grad_vec = grad_out_mat.to_vec()?;
        let mut bias_grad_data = vec![0.0; out_c];
        for n in 0..batch {
            for sp in 0..out_spatial {
                for oc in 0..out_c {
                    bias_grad_data[oc] += bias_grad_vec[(n * out_spatial + sp) * out_c + oc];
                }
            }
        }
        self.bias_grad = GpuTensor::from_slice(&bias_grad_data, 1, out_c, self.device.clone())?;
        
        // Input gradient: grad_output * weights, then col2im
        // grad_output: (batch*out_h*out_w, out_c)
        // weights: (out_c, in_c*k*k)
        // grad_im2col: (batch*out_h*out_w, in_c*k*k)
        let grad_im2col = grad_out_mat.matmul(&self.weights)?;
        
        // Convert back from im2col to image format
        self.col2im(&grad_im2col, batch, in_c, in_h, in_w)
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.weights, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn gradients(&self) -> Vec<&GpuTensor> {
        vec![&self.weights_grad, &self.bias_grad]
    }

    fn set_training(&mut self, _training: bool) {
        // Conv2D doesn't need training mode distinction
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// GPU MaxPool2D Layer
pub struct GpuMaxPool2D {
    pool_size: usize,
    stride: usize,
    cached_input_shape: Option<(usize, usize, usize, usize)>,
    cached_output_shape: Option<(usize, usize, usize, usize)>,
    cached_indices: Option<Vec<usize>>, // Store indices on CPU for simplicity
    device: Arc<CudaDevice>,
    name: String,
}

impl GpuMaxPool2D {
    pub fn new(pool_size: usize, stride: usize, device: Arc<CudaDevice>) -> Result<Self, String> {
        Ok(Self {
            pool_size,
            stride,
            cached_input_shape: None,
            cached_output_shape: None,
            cached_indices: None,
            device,
            name: format!("GpuMaxPool2D({}x{})", pool_size, pool_size),
        })
    }

    fn output_dims(&self, in_height: usize, in_width: usize) -> (usize, usize) {
        let out_height = (in_height - self.pool_size) / self.stride + 1;
        let out_width = (in_width - self.pool_size) / self.stride + 1;
        (out_height, out_width)
    }
}

impl GpuLayer for GpuMaxPool2D {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        let batch_size = input.rows();
        let flattened_size = input.cols();
        
        // Infer channels and spatial dimensions from input
        // Try common channel counts in order of preference, checking that spatial size is a perfect square
        let channels = if flattened_size % 3 == 0 && (flattened_size / 3) == 1024 {
            3 // Input layer: 3 channels * 32*32 = 3072
        } else {
            // Try channel counts in order: 32, 64, 128, 256, etc.
            // For each, check if the resulting spatial size is a perfect square
            let channel_candidates = [32, 64, 128, 256];
            let mut inferred_channels = 0;
            
            for &ch in &channel_candidates {
                if flattened_size % ch == 0 {
                    let spatial_size = flattened_size / ch;
                    let height = (spatial_size as f64).sqrt() as usize;
                    if height * height == spatial_size {
                        inferred_channels = ch;
                        break;
                    }
                }
            }
            
            // Fallback: try to infer from known spatial sizes
            if inferred_channels == 0 {
                let spatial_candidates = [1024, 256, 64, 16, 4]; // 32*32, 16*16, 8*8, 4*4, 2*2
                for &spatial in &spatial_candidates {
                    if flattened_size % spatial == 0 {
                        let candidate_channels = flattened_size / spatial;
                        // Check if it's a reasonable channel count
                        if candidate_channels <= 512 && candidate_channels > 0 {
                            inferred_channels = candidate_channels;
                            break;
                        }
                    }
                }
            }
            
            if inferred_channels == 0 {
                return Err(format!("Cannot infer channels from input size {}. Expected size divisible by a channel count (3, 32, 64, 128, etc.) with perfect square spatial dimensions.", flattened_size));
            }
            inferred_channels
        };
        
        let spatial_size = flattened_size / channels;
        let height = (spatial_size as f64).sqrt() as usize;
        let width = height;
        
        if height * width != spatial_size {
            return Err(format!("Cannot infer square spatial dimensions. Channels: {}, spatial_size: {}, height: {}, width: {}", channels, spatial_size, height, width));
        }
        
        let (out_height, out_width) = self.output_dims(height, width);
        let out_flattened = channels * out_height * out_width;
        
        self.cached_input_shape = Some((batch_size, channels, height, width));
        self.cached_output_shape = Some((batch_size, channels, out_height, out_width));
        
        // MaxPool on CPU for now (will optimize later)
        let input_vec = input.to_vec()?;
        let mut output_data = vec![0.0; batch_size * out_flattened];
        let mut indices = Vec::new();
        
        for n in 0..batch_size {
            for c in 0..channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0;
                        
                        for ph in 0..self.pool_size {
                            for pw in 0..self.pool_size {
                                let ih = oh * self.stride + ph;
                                let iw = ow * self.stride + pw;
                                
                                if ih < height && iw < width {
                                    let input_idx = n * (channels * height * width) +
                                                   c * (height * width) +
                                                   ih * width + iw;
                                    let val = input_vec[input_idx];
                                    if val > max_val {
                                        max_val = val;
                                        max_idx = input_idx;
                                    }
                                }
                            }
                        }
                        
                        let output_idx = n * out_flattened + c * (out_height * out_width) + oh * out_width + ow;
                        output_data[output_idx] = max_val;
                        indices.push(max_idx);
                    }
                }
            }
        }
        
        self.cached_indices = Some(indices);
        GpuTensor::from_slice(&output_data, batch_size, out_flattened, self.device.clone())
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        let indices = self.cached_indices.as_ref()
            .ok_or("Forward must be called before backward")?;
        let (batch, channels, in_h, in_w) = self.cached_input_shape
            .ok_or("Missing cached input shape")?;
        
        let grad_out_vec = grad_output.to_vec()?;
        let mut grad_input_data = vec![0.0; batch * channels * in_h * in_w];
        
        let (_, _, out_h, out_w) = self.cached_output_shape
            .ok_or("Missing cached output shape")?;
        
        let mut idx = 0;
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let grad_out_idx = n * (channels * out_h * out_w) +
                                         c * (out_h * out_w) +
                                         oh * out_w + ow;
                        let input_idx = indices[idx];
                        grad_input_data[input_idx] += grad_out_vec[grad_out_idx];
                        idx += 1;
                    }
                }
            }
        }
        
        GpuTensor::from_slice(&grad_input_data, batch, channels * in_h * in_w, self.device.clone())
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // MaxPool doesn't need training mode
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// GPU Flatten Layer - converts CNN output to dense input
pub struct GpuFlatten {
    name: String,
}

impl GpuFlatten {
    pub fn new() -> Self {
        Self {
            name: "GpuFlatten".to_string(),
        }
    }
}

impl GpuLayer for GpuFlatten {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        // Flatten is just a reshape - no data movement needed
        Ok(input.clone())
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        // Flatten backward is identity
        Ok(grad_output.clone())
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![]
    }

    fn gradients(&self) -> Vec<&GpuTensor> {
        vec![]
    }

    fn set_training(&mut self, _training: bool) {
        // Flatten doesn't need training mode
    }

    fn name(&self) -> &str {
        &self.name
    }
}
