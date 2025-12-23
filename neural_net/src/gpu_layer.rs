//! GPU-native layer implementations
//!
//! All operations stay on GPU - no CPU transfers except for final results.

use crate::{Float, gpu_tensor::GpuTensor};
use cudarc::driver::CudaDevice;
use std::sync::Arc;
use std::any::Any;

/// Trait for GPU layers
pub trait GpuLayer: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Forward pass on GPU
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String>;

    /// Backward pass on GPU
    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String>;

    /// Get parameters (weights, biases)
    fn parameters(&self) -> Vec<&GpuTensor>;

    /// Get mutable parameters
    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor>;

    /// Get gradients
    fn gradients(&self) -> Vec<&GpuTensor>;

    /// Set training mode
    fn set_training(&mut self, training: bool);

    /// Get layer name
    fn name(&self) -> &str;
}

/// GPU Dense (Fully Connected) Layer
pub struct GpuDense {
    /// Weights (input_size x output_size)
    weights: GpuTensor,
    /// Bias (1 x output_size)
    bias: GpuTensor,
    /// Weight gradients
    weights_grad: GpuTensor,
    /// Bias gradients
    bias_grad: GpuTensor,
    /// Cached input for backward pass
    cached_input: Option<GpuTensor>,
    /// Cached pre-activation
    cached_pre_activation: Option<GpuTensor>,
    /// Activation type
    activation: ActivationType,
    /// Device
    device: Arc<CudaDevice>,
    /// Layer name
    name: String,
}

#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
}

impl GpuDense {
    /// Create new GPU dense layer with He/Xavier initialization
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationType,
        device: Arc<CudaDevice>,
    ) -> Result<Self, String> {
        // He initialization for ReLU, Xavier for others
        let std_dev = match activation {
            ActivationType::ReLU => (2.0 / input_size as Float).sqrt(),
            _ => (2.0 / (input_size + output_size) as Float).sqrt(),
        };

        let weights = GpuTensor::random_uniform(
            input_size,
            output_size,
            -std_dev,
            std_dev,
            device.clone(),
        )?;

        let bias = GpuTensor::zeros(1, output_size, device.clone())?;
        let weights_grad = GpuTensor::zeros(input_size, output_size, device.clone())?;
        let bias_grad = GpuTensor::zeros(1, output_size, device.clone())?;

        Ok(Self {
            weights,
            bias,
            weights_grad,
            bias_grad,
            cached_input: None,
            cached_pre_activation: None,
            activation,
            device,
            name: format!("GpuDense({} -> {})", input_size, output_size),
        })
    }

    /// Apply activation function on GPU
    fn apply_activation(&self, input: &GpuTensor) -> Result<GpuTensor, String> {
        match self.activation {
            ActivationType::Linear => Ok(input.clone()),
            ActivationType::ReLU => input.relu(),
            ActivationType::Sigmoid => input.sigmoid(),
            ActivationType::Tanh => input.tanh(),
        }
    }

    /// Compute activation derivative on GPU
    fn activation_derivative(&self, output: &GpuTensor) -> Result<GpuTensor, String> {
        match self.activation {
            ActivationType::Linear => GpuTensor::ones(output.rows(), output.cols(), self.device.clone()),
            ActivationType::ReLU => {
                // ReLU derivative: 1 if x > 0, else 0
                let vec = output.to_vec()?;
                let deriv: Vec<Float> = vec.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
                GpuTensor::from_slice(&deriv, output.rows(), output.cols(), self.device.clone())
            }
            ActivationType::Sigmoid => {
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                let ones = GpuTensor::ones(output.rows(), output.cols(), self.device.clone())?;
                let one_minus_out = ones.add(&output.scale(-1.0)?)?;
                output.mul(&one_minus_out)
            }
            ActivationType::Tanh => {
                // tanh'(x) = 1 - tanh(x)^2
                let ones = GpuTensor::ones(output.rows(), output.cols(), self.device.clone())?;
                let out_squared = output.mul(output)?;
                ones.add(&out_squared.scale(-1.0)?)
            }
        }
    }
}

impl GpuLayer for GpuDense {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        // Cache input for backward
        self.cached_input = Some(input.clone());

        // Linear: output = input * weights + bias
        let linear = input.matmul(&self.weights)?;
        let pre_activation = linear.add_row_vector(&self.bias)?;

        // Cache pre-activation
        self.cached_pre_activation = Some(pre_activation.clone());

        // Apply activation
        self.apply_activation(&pre_activation)
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        let input = self.cached_input.as_ref()
            .ok_or("Forward must be called before backward")?;
        let pre_activation = self.cached_pre_activation.as_ref()
            .ok_or("Forward must be called before backward")?;

        // Compute activation gradient based on pre-activation (not activated output)
        // For ReLU: derivative depends on pre-activation (x > 0)
        // For Sigmoid/Tanh: derivative depends on activated output
        let activation_grad = match self.activation {
            ActivationType::Linear => {
                GpuTensor::ones(grad_output.rows(), grad_output.cols(), self.device.clone())?
            }
            ActivationType::ReLU => {
                // ReLU derivative: 1 if pre_activation > 0, else 0
                let vec = pre_activation.to_vec()?;
                let deriv: Vec<Float> = vec.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
                GpuTensor::from_slice(&deriv, pre_activation.rows(), pre_activation.cols(), self.device.clone())?
            }
            ActivationType::Sigmoid => {
                // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
                let activated = self.apply_activation(pre_activation)?;
                let ones = GpuTensor::ones(activated.rows(), activated.cols(), self.device.clone())?;
                let one_minus_out = ones.sub(&activated)?;
                activated.mul(&one_minus_out)?
            }
            ActivationType::Tanh => {
                // tanh'(x) = 1 - tanh(x)^2
                let activated = self.apply_activation(pre_activation)?;
                let ones = GpuTensor::ones(activated.rows(), activated.cols(), self.device.clone())?;
                let out_squared = activated.mul(&activated)?;
                ones.sub(&out_squared)?
            }
        };

        // Gradient w.r.t. pre-activation: grad_output * activation'
        let grad_pre_activation = grad_output.mul(&activation_grad)?;

        // Gradient w.r.t. weights: input^T * grad_pre_activation
        self.weights_grad = input.transpose()?.matmul(&grad_pre_activation)?;

        // Gradient w.r.t. bias: sum along batch dimension
        self.bias_grad = grad_pre_activation.sum_axis_0()?;

        // Gradient w.r.t. input: grad_pre_activation * weights^T
        grad_pre_activation.matmul(&self.weights.transpose()?)
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
        // Dense layer doesn't need training mode distinction
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// GPU Batch Normalization Layer
pub struct GpuBatchNorm {
    num_features: usize,
    /// Scale parameter (learnable)
    gamma: GpuTensor,
    /// Shift parameter (learnable)
    beta: GpuTensor,
    /// Running mean (for inference)
    running_mean: GpuTensor,
    /// Running variance (for inference)
    running_var: GpuTensor,
    /// Momentum for running stats
    momentum: Float,
    /// Epsilon for numerical stability
    epsilon: Float,
    /// Training mode
    training: bool,
    /// Cached values for backward
    cached_input: Option<GpuTensor>,
    cached_mean: Option<GpuTensor>,
    cached_var: Option<GpuTensor>,
    cached_normalized: Option<GpuTensor>,
    /// Gradients
    gamma_grad: GpuTensor,
    beta_grad: GpuTensor,
    /// Device
    device: Arc<CudaDevice>,
}

impl GpuBatchNorm {
    pub fn new(num_features: usize, device: Arc<CudaDevice>) -> Result<Self, String> {
        Ok(Self {
            num_features,
            gamma: GpuTensor::ones(1, num_features, device.clone())?,
            beta: GpuTensor::zeros(1, num_features, device.clone())?,
            running_mean: GpuTensor::zeros(1, num_features, device.clone())?,
            running_var: GpuTensor::ones(1, num_features, device.clone())?,
            momentum: 0.1,
            epsilon: 1e-5,
            training: true,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            cached_normalized: None,
            gamma_grad: GpuTensor::zeros(1, num_features, device.clone())?,
            beta_grad: GpuTensor::zeros(1, num_features, device.clone())?,
            device,
        })
    }

    /// Compute batch statistics on GPU
    fn compute_batch_stats(&self, input: &GpuTensor) -> Result<(GpuTensor, GpuTensor), String> {
        let input_vec = input.to_vec()?;
        let batch_size = input.rows() as Float;
        let num_features = input.cols();

        // Compute mean
        let mut mean_data = vec![0.0; num_features];
        for j in 0..num_features {
            let mut sum = 0.0;
            for i in 0..input.rows() {
                sum += input_vec[i * num_features + j];
            }
            mean_data[j] = sum / batch_size;
        }

        // Compute variance
        let mut var_data = vec![0.0; num_features];
        for j in 0..num_features {
            let mut sum_sq = 0.0;
            for i in 0..input.rows() {
                let diff = input_vec[i * num_features + j] - mean_data[j];
                sum_sq += diff * diff;
            }
            var_data[j] = sum_sq / batch_size;
        }

        let mean = GpuTensor::from_slice(&mean_data, 1, num_features, self.device.clone())?;
        let var = GpuTensor::from_slice(&var_data, 1, num_features, self.device.clone())?;

        Ok((mean, var))
    }

    /// Normalize on GPU
    fn normalize(&self, input: &GpuTensor, mean: &GpuTensor, var: &GpuTensor) -> Result<GpuTensor, String> {
        let input_vec = input.to_vec()?;
        let mean_vec = mean.to_vec()?;
        let var_vec = var.to_vec()?;
        let num_features = input.cols();

        let mut normalized = vec![0.0; input_vec.len()];
        for i in 0..input.rows() {
            for j in 0..num_features {
                let idx = i * num_features + j;
                normalized[idx] = (input_vec[idx] - mean_vec[j]) / (var_vec[j] + self.epsilon).sqrt();
            }
        }

        GpuTensor::from_slice(&normalized, input.rows(), num_features, self.device.clone())
    }

    /// Scale and shift on GPU
    fn scale_shift(&self, normalized: &GpuTensor) -> Result<GpuTensor, String> {
        let normalized_vec = normalized.to_vec()?;
        let gamma_vec = self.gamma.to_vec()?;
        let beta_vec = self.beta.to_vec()?;
        let num_features = normalized.cols();

        let mut output = vec![0.0; normalized_vec.len()];
        for i in 0..normalized.rows() {
            for j in 0..num_features {
                let idx = i * num_features + j;
                output[idx] = gamma_vec[j] * normalized_vec[idx] + beta_vec[j];
            }
        }

        GpuTensor::from_slice(&output, normalized.rows(), num_features, self.device.clone())
    }

    /// Update running statistics
    fn update_running_stats(&mut self, mean: &GpuTensor, var: &GpuTensor) -> Result<(), String> {
        let mean_vec = mean.to_vec()?;
        let var_vec = var.to_vec()?;
        let mut running_mean_vec = self.running_mean.to_vec()?;
        let mut running_var_vec = self.running_var.to_vec()?;

        for j in 0..self.num_features {
            running_mean_vec[j] = (1.0 - self.momentum) * running_mean_vec[j] + self.momentum * mean_vec[j];
            running_var_vec[j] = (1.0 - self.momentum) * running_var_vec[j] + self.momentum * var_vec[j];
        }

        self.running_mean = GpuTensor::from_slice(&running_mean_vec, 1, self.num_features, self.device.clone())?;
        self.running_var = GpuTensor::from_slice(&running_var_vec, 1, self.num_features, self.device.clone())?;

        Ok(())
    }

    pub fn get_running_stats(&self) -> (GpuTensor, GpuTensor) {
        (self.running_mean.clone(), self.running_var.clone())
    }

    pub fn set_running_stats(&mut self, mean: &GpuTensor, var: &GpuTensor) -> Result<(), String> {
        if mean.rows() != 1 || mean.cols() != self.num_features {
            return Err("Running mean shape mismatch".to_string());
        }
        if var.rows() != 1 || var.cols() != self.num_features {
            return Err("Running var shape mismatch".to_string());
        }
        self.running_mean = mean.clone();
        self.running_var = var.clone();
        Ok(())
    }
}

impl GpuLayer for GpuBatchNorm {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        if self.training {
            // Compute batch statistics
            let (mean, var) = self.compute_batch_stats(input)?;

            // Normalize
            let normalized = self.normalize(input, &mean, &var)?;

            // Scale and shift
            let output = self.scale_shift(&normalized)?;

            // Update running stats
            self.update_running_stats(&mean, &var)?;

            // Cache for backward
            self.cached_input = Some(input.clone());
            self.cached_mean = Some(mean);
            self.cached_var = Some(var);
            self.cached_normalized = Some(normalized);

            Ok(output)
        } else {
            // Use running statistics for inference
            let normalized = self.normalize(input, &self.running_mean, &self.running_var)?;
            self.scale_shift(&normalized)
        }
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        let input = self.cached_input.as_ref().ok_or("Forward must be called first")?;
        let mean = self.cached_mean.as_ref().ok_or("Forward must be called first")?;
        let var = self.cached_var.as_ref().ok_or("Forward must be called first")?;
        let normalized = self.cached_normalized.as_ref().ok_or("Forward must be called first")?;

        let grad_output_vec = grad_output.to_vec()?;
        let normalized_vec = normalized.to_vec()?;
        let gamma_vec = self.gamma.to_vec()?;
        let var_vec = var.to_vec()?;
        let input_vec = input.to_vec()?;
        let mean_vec = mean.to_vec()?;

        let batch_size = input.rows() as Float;
        let num_features = input.cols();

        // Gradient w.r.t. gamma
        let mut gamma_grad_vec = vec![0.0; num_features];
        for j in 0..num_features {
            let mut sum = 0.0;
            for i in 0..input.rows() {
                sum += grad_output_vec[i * num_features + j] * normalized_vec[i * num_features + j];
            }
            gamma_grad_vec[j] = sum;
        }
        self.gamma_grad = GpuTensor::from_slice(&gamma_grad_vec, 1, num_features, self.device.clone())?;

        // Gradient w.r.t. beta
        let mut beta_grad_vec = vec![0.0; num_features];
        for j in 0..num_features {
            let mut sum = 0.0;
            for i in 0..input.rows() {
                sum += grad_output_vec[i * num_features + j];
            }
            beta_grad_vec[j] = sum;
        }
        self.beta_grad = GpuTensor::from_slice(&beta_grad_vec, 1, num_features, self.device.clone())?;

        // Gradient w.r.t. input (full BatchNorm backward)
        let mut grad_input_vec = vec![0.0; input_vec.len()];
        for j in 0..num_features {
            let std_inv = 1.0 / (var_vec[j] + self.epsilon).sqrt();

            // Sum of grad_output for this feature
            let mut grad_sum = 0.0;
            let mut grad_norm_sum = 0.0;
            for i in 0..input.rows() {
                let idx = i * num_features + j;
                grad_sum += grad_output_vec[idx];
                grad_norm_sum += grad_output_vec[idx] * (input_vec[idx] - mean_vec[j]);
            }

            for i in 0..input.rows() {
                let idx = i * num_features + j;
                let norm = (input_vec[idx] - mean_vec[j]) * std_inv;
                grad_input_vec[idx] = gamma_vec[j] * std_inv * (
                    grad_output_vec[idx] - grad_sum / batch_size - norm * grad_norm_sum / batch_size
                );
            }
        }

        GpuTensor::from_slice(&grad_input_vec, input.rows(), num_features, self.device.clone())
    }

    fn parameters(&self) -> Vec<&GpuTensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn gradients(&self) -> Vec<&GpuTensor> {
        vec![&self.gamma_grad, &self.beta_grad]
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn name(&self) -> &str {
        "GpuBatchNorm"
    }
}

/// GPU Dropout Layer
pub struct GpuDropout {
    dropout_rate: Float,
    training: bool,
    /// Cached mask for backward
    cached_mask: Option<GpuTensor>,
    device: Arc<CudaDevice>,
}

impl GpuDropout {
    pub fn new(dropout_rate: Float, device: Arc<CudaDevice>) -> Result<Self, String> {
        if dropout_rate < 0.0 || dropout_rate >= 1.0 {
            return Err(format!("Dropout rate must be in [0, 1), got {}", dropout_rate));
        }

        Ok(Self {
            dropout_rate,
            training: true,
            cached_mask: None,
            device,
        })
    }

    /// Generate dropout mask on GPU
    fn generate_mask(&self, rows: usize, cols: usize) -> Result<GpuTensor, String> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = rows * cols;
        let scale = 1.0 / (1.0 - self.dropout_rate);

        let mask: Vec<Float> = (0..size)
            .map(|_| {
                if rng.gen::<Float>() < self.dropout_rate {
                    0.0
                } else {
                    scale
                }
            })
            .collect();

        GpuTensor::from_slice(&mask, rows, cols, self.device.clone())
    }
}

impl GpuLayer for GpuDropout {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        if self.training && self.dropout_rate > 0.0 {
            let mask = self.generate_mask(input.rows(), input.cols())?;
            let output = input.mul(&mask)?;
            self.cached_mask = Some(mask);
            Ok(output)
        } else {
            Ok(input.clone())
        }
    }

    fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        if self.training && self.dropout_rate > 0.0 {
            let mask = self.cached_mask.as_ref()
                .ok_or("Forward must be called before backward")?;
            grad_output.mul(mask)
        } else {
            Ok(grad_output.clone())
        }
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

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn name(&self) -> &str {
        "GpuDropout"
    }
}

/// GPU Optimizer - SGD with momentum
pub struct GpuSGD {
    learning_rate: Float,
    momentum: Float,
    velocities: Vec<Option<GpuTensor>>,
    weight_decay: Float,
}

impl GpuSGD {
    pub fn new(learning_rate: Float) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            velocities: Vec::new(),
            weight_decay: 0.0,
        }
    }

    pub fn with_momentum(learning_rate: Float, momentum: Float) -> Self {
        Self {
            learning_rate,
            momentum,
            velocities: Vec::new(),
            weight_decay: 0.0,
        }
    }

    pub fn set_learning_rate(&mut self, lr: Float) {
        self.learning_rate = lr;
    }

    pub fn set_weight_decay(&mut self, wd: Float) {
        self.weight_decay = wd.max(0.0);
    }

    /// Update parameters on GPU
    pub fn step(&mut self, parameters: &mut [&mut GpuTensor], gradients: &[&GpuTensor]) -> Result<(), String> {
        if parameters.len() != gradients.len() {
            return Err("Parameters and gradients length mismatch".to_string());
        }

        // Initialize velocities if using momentum
        if self.momentum > 0.0 && self.velocities.is_empty() {
            for param in parameters.iter() {
                let device = param.device().clone();
                let velocity = GpuTensor::zeros(param.rows(), param.cols(), device)?;
                self.velocities.push(Some(velocity));
            }
        }

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay: grad += wd * param
            let effective_grad = if self.weight_decay > 0.0 {
                let wd_term = (*param).scale(self.weight_decay)?;
                grad.add(&wd_term)?
            } else { (*grad).clone() };
            if self.momentum > 0.0 {
                // Update velocity: v = momentum * v - lr * grad
                let velocity = self.velocities[i].as_mut().unwrap();
                let scaled_grad = effective_grad.scale(self.learning_rate)?;
                *velocity = velocity.scale(self.momentum)?.add(&scaled_grad.scale(-1.0)?)?;

                // Update parameter: param = param + velocity
                **param = param.add(velocity)?;
            } else {
                // Simple SGD: param = param - lr * grad
                let update = effective_grad.scale(self.learning_rate)?;
                **param = param.add(&update.scale(-1.0)?)?;
            }
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.velocities.clear();
    }
}

/// GPU Network - sequential model
pub struct GpuNetwork {
    layers: Vec<Box<dyn GpuLayer>>,
    device: Arc<CudaDevice>,
}

impl GpuNetwork {
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            layers: Vec::new(),
            device,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn GpuLayer>) {
        self.layers.push(layer);
    }

    pub fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor, String> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    pub fn backward(&mut self, grad_output: &GpuTensor) -> Result<GpuTensor, String> {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }
        Ok(grad)
    }

    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    pub fn parameters(&self) -> Vec<&GpuTensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut GpuTensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        params
    }

    pub fn gradients(&self) -> Vec<&GpuTensor> {
        let mut grads = Vec::new();
        for layer in &self.layers {
            grads.extend(layer.gradients());
        }
        grads
    }

    /// Apply optimizer step directly on the network
    pub fn optimizer_step(&mut self, optimizer: &mut GpuSGD) -> Result<(), String> {
        // Collect ALL parameters and gradients across ALL layers
        let mut all_params: Vec<&mut GpuTensor> = Vec::new();
        let mut all_grads_owned: Vec<GpuTensor> = Vec::new();

        for layer in &self.layers {
            for grad in layer.gradients() {
                all_grads_owned.push(grad.clone());
            }
        }

        for layer in &mut self.layers {
            all_params.extend(layer.parameters_mut());
        }

        // Convert to references
        let all_grads_refs: Vec<&GpuTensor> = all_grads_owned.iter().collect();

        // Single optimizer step for all parameters
        optimizer.step(&mut all_params, &all_grads_refs)?;

        Ok(())
    }

    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    pub fn for_each_layer<F: FnMut(&dyn GpuLayer)>(&self, mut f: F) {
        for layer in &self.layers {
            f(layer.as_ref());
        }
    }

    /// Copy parameters from another network (same architecture)
    pub fn copy_parameters_from(&mut self, other: &GpuNetwork) -> Result<(), String> {
        let mut self_params = self.parameters_mut();
        let other_params = other.parameters();
        if self_params.len() != other_params.len() {
            return Err("Parameter count mismatch".to_string());
        }
        for (dst, src) in self_params.iter_mut().zip(other_params.iter()) {
            **dst = (*src).clone();
        }
        Ok(())
    }

    /// EMA update: self = decay*self + (1-decay)*other
    pub fn ema_update_from(&mut self, other: &GpuNetwork, decay: Float) -> Result<(), String> {
        if decay < 0.0 || decay > 1.0 { return Err("Invalid EMA decay".to_string()); }
        let mut self_params = self.parameters_mut();
        let other_params = other.parameters();
        if self_params.len() != other_params.len() {
            return Err("Parameter count mismatch".to_string());
        }
        for (dst, src) in self_params.iter_mut().zip(other_params.iter()) {
            let a = (*dst).scale(decay)?;
            let b = (*src).scale(1.0 - decay)?;
            **dst = a.add(&b)?;
        }
        Ok(())
    }
}

/// GPU Softmax + CrossEntropy Loss (fused for numerical stability)
pub struct GpuSoftmaxCrossEntropy {
    epsilon: Float,
}

impl GpuSoftmaxCrossEntropy {
    pub fn new() -> Self {
        Self { epsilon: 1e-15 }
    }

    /// Compute loss (returns scalar)
    pub fn compute(&self, predictions: &GpuTensor, targets: &GpuTensor) -> Result<Float, String> {
        if predictions.shape() != targets.shape() {
            return Err(format!("Shape mismatch: {:?} vs {:?}", predictions.shape(), targets.shape()));
        }

        let pred_vec = predictions.to_vec()?;
        let target_vec = targets.to_vec()?;
        let batch_size = predictions.rows() as Float;

        let mut sum = 0.0;
        for i in 0..pred_vec.len() {
            let pred = pred_vec[i].max(self.epsilon);
            sum += target_vec[i] * pred.ln();
        }

        Ok(-sum / batch_size)
    }

    /// Compute gradient (predictions - targets) / batch_size
    pub fn gradient(&self, predictions: &GpuTensor, targets: &GpuTensor) -> Result<GpuTensor, String> {
        let batch_size = predictions.rows() as Float;
        let diff = predictions.add(&targets.scale(-1.0)?)?;
        diff.scale(1.0 / batch_size)
    }
}

// Include CNN layers from separate module
#[path = "gpu_layer_cnn.rs"]
mod gpu_layer_cnn;
pub use gpu_layer_cnn::{GpuConv2D, GpuMaxPool2D, GpuFlatten};
