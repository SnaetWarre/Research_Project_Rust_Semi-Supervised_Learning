//! Model serialization and loading
//!
//! This module provides functionality to save and load GPU neural network models.
//! Models are saved as JSON metadata + binary weights.

use crate::{Float, GpuNetwork, GpuTensor, ActivationType};
use crate::gpu_layer::GpuLayer;
use cudarc::driver::CudaDevice;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Model architecture metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Input size
    pub input_size: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Layer configurations
    pub layers: Vec<LayerConfig>,
    /// Dropout rate
    pub dropout_rate: Float,
}

/// Layer configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: String,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
    /// Activation type (for dense layers)
    pub activation: Option<String>,
}

/// Model weights (raw binary data)
#[derive(Clone, Debug)]
pub struct ModelWeights {
    /// Weight tensors (row-major f64)
    pub weights: Vec<Vec<Float>>,
    /// Bias tensors
    pub biases: Vec<Vec<Float>>,
}

/// Save GPU model to disk
pub fn save_model(
    network: &GpuNetwork,
    metadata: ModelMetadata,
    model_dir: impl AsRef<Path>,
) -> Result<(), String> {
    let model_dir = model_dir.as_ref();
    std::fs::create_dir_all(model_dir)
        .map_err(|e| format!("Failed to create directory: {}", e))?;

    // Save metadata as JSON
    let metadata_path = model_dir.join("model.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
    std::fs::write(&metadata_path, metadata_json)
        .map_err(|e| format!("Failed to write metadata: {}", e))?;

    // Extract weights from GPU
    let mut weights_data = Vec::new();
    let mut biases_data = Vec::new();

    let params = network.parameters();
    let mut param_iter = params.iter();

    // Alternate between weights and biases (weights come first, then biases)
    while let Some(param) = param_iter.next() {
        let param_vec = param.to_vec()
            .map_err(|e| format!("Failed to copy parameter from GPU: {}", e))?;
        
        // Distinguish between weights and biases by shape
        if param.rows() > 1 {
            weights_data.push(param_vec);
        } else {
            biases_data.push(param_vec);
        }
    }

    // Save weights as binary
    let weights_path = model_dir.join("weights.bin");
    let mut weights_file = File::create(&weights_path)
        .map_err(|e| format!("Failed to create weights file: {}", e))?;

    // Write number of weight tensors
    let num_weights = weights_data.len() as u64;
    weights_file.write_all(&num_weights.to_le_bytes())
        .map_err(|e| format!("Failed to write weights count: {}", e))?;

    // Write each weight tensor
    for weight_tensor in &weights_data {
        let len = weight_tensor.len() as u64;
        weights_file.write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write weight tensor length: {}", e))?;
        
        let bytes: Vec<u8> = weight_tensor.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        weights_file.write_all(&bytes)
            .map_err(|e| format!("Failed to write weight tensor: {}", e))?;
    }

    // Write number of bias tensors
    let num_biases = biases_data.len() as u64;
    weights_file.write_all(&num_biases.to_le_bytes())
        .map_err(|e| format!("Failed to write biases count: {}", e))?;

    // Write each bias tensor
    for bias_tensor in &biases_data {
        let len = bias_tensor.len() as u64;
        weights_file.write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write bias tensor length: {}", e))?;
        
        let bytes: Vec<u8> = bias_tensor.iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        weights_file.write_all(&bytes)
            .map_err(|e| format!("Failed to write bias tensor: {}", e))?;
    }

    // Persist BatchNorm running stats (means then vars) for stable inference
    let mut bn_means: Vec<Vec<Float>> = Vec::new();
    let mut bn_vars: Vec<Vec<Float>> = Vec::new();
    network.for_each_layer(|layer| {
        if layer.name() == "GpuBatchNorm" {
            if let Some(bn) = layer.as_any().downcast_ref::<crate::gpu_layer::GpuBatchNorm>() {
                let (mean, var) = bn.get_running_stats();
                if let (Ok(mv), Ok(vv)) = (mean.to_vec(), var.to_vec()) {
                    bn_means.push(mv);
                    bn_vars.push(vv);
                }
            }
        }
    });

    // Write BN means
    let num_bn = bn_means.len() as u64;
    weights_file.write_all(&num_bn.to_le_bytes())
        .map_err(|e| format!("Failed to write BN means count: {}", e))?;
    for mean in &bn_means {
        let len = mean.len() as u64;
        weights_file.write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write BN mean length: {}", e))?;
        let bytes: Vec<u8> = mean.iter().flat_map(|&f| f.to_le_bytes()).collect();
        weights_file.write_all(&bytes)
            .map_err(|e| format!("Failed to write BN mean: {}", e))?;
    }

    // Write BN vars
    let num_bn_vars = bn_vars.len() as u64;
    weights_file.write_all(&num_bn_vars.to_le_bytes())
        .map_err(|e| format!("Failed to write BN vars count: {}", e))?;
    for var in &bn_vars {
        let len = var.len() as u64;
        weights_file.write_all(&len.to_le_bytes())
            .map_err(|e| format!("Failed to write BN var length: {}", e))?;
        let bytes: Vec<u8> = var.iter().flat_map(|&f| f.to_le_bytes()).collect();
        weights_file.write_all(&bytes)
            .map_err(|e| format!("Failed to write BN var: {}", e))?;
    }

    Ok(())
}

/// Load GPU model from disk
pub fn load_model(
    model_dir: impl AsRef<Path>,
    device: Arc<CudaDevice>,
) -> Result<(GpuNetwork, ModelMetadata), String> {
    let model_dir = model_dir.as_ref();

    // Load metadata
    let metadata_path = model_dir.join("model.json");
    let metadata_json = std::fs::read_to_string(&metadata_path)
        .map_err(|e| format!("Failed to read metadata: {}", e))?;
    let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
        .map_err(|e| format!("Failed to parse metadata: {}", e))?;

    // Load weights
    let weights_path = model_dir.join("weights.bin");
    let mut weights_file = File::open(&weights_path)
        .map_err(|e| format!("Failed to open weights file: {}", e))?;

    // Read weights
    let mut weights_data = Vec::new();
    let mut buf = [0u8; 8];
    
    weights_file.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read weights count: {}", e))?;
    let num_weights = u64::from_le_bytes(buf);

    for _ in 0..num_weights {
        weights_file.read_exact(&mut buf)
            .map_err(|e| format!("Failed to read weight tensor length: {}", e))?;
        let len = u64::from_le_bytes(buf) as usize;

        let mut tensor_data = vec![0.0; len];
        let mut bytes_buf = vec![0u8; len * 8];
        weights_file.read_exact(&mut bytes_buf)
            .map_err(|e| format!("Failed to read weight tensor: {}", e))?;

        for i in 0..len {
            let start = i * 8;
            tensor_data[i] = Float::from_le_bytes([
                bytes_buf[start], bytes_buf[start + 1], bytes_buf[start + 2], bytes_buf[start + 3],
                bytes_buf[start + 4], bytes_buf[start + 5], bytes_buf[start + 6], bytes_buf[start + 7],
            ]);
        }

        weights_data.push(tensor_data);
    }

    // Read biases
    let mut biases_data = Vec::new();
    weights_file.read_exact(&mut buf)
        .map_err(|e| format!("Failed to read biases count: {}", e))?;
    let num_biases = u64::from_le_bytes(buf);

    for _ in 0..num_biases {
        weights_file.read_exact(&mut buf)
            .map_err(|e| format!("Failed to read bias tensor length: {}", e))?;
        let len = u64::from_le_bytes(buf) as usize;

        let mut tensor_data = vec![0.0; len];
        let mut bytes_buf = vec![0u8; len * 8];
        weights_file.read_exact(&mut bytes_buf)
            .map_err(|e| format!("Failed to read bias tensor: {}", e))?;

        for i in 0..len {
            let start = i * 8;
            tensor_data[i] = Float::from_le_bytes([
                bytes_buf[start], bytes_buf[start + 1], bytes_buf[start + 2], bytes_buf[start + 3],
                bytes_buf[start + 4], bytes_buf[start + 5], bytes_buf[start + 6], bytes_buf[start + 7],
            ]);
        }

        biases_data.push(tensor_data);
    }

    // Try to read optional BN running stats (backward compatible)
    let mut bn_means: Vec<Vec<Float>> = Vec::new();
    let mut bn_vars: Vec<Vec<Float>> = Vec::new();
    let mut tmp = [0u8; 8];
    if let Ok(()) = weights_file.read_exact(&mut tmp) {
        let num_bn = u64::from_le_bytes(tmp) as usize;
        for _ in 0..num_bn {
            weights_file.read_exact(&mut buf)
                .map_err(|e| format!("Failed to read BN mean length: {}", e))?;
            let len = u64::from_le_bytes(buf) as usize;
            let mut bytes_buf = vec![0u8; len * 8];
            weights_file.read_exact(&mut bytes_buf)
                .map_err(|e| format!("Failed to read BN mean: {}", e))?;
            let mut vecf = vec![0.0; len];
            for i in 0..len {
                let s = i * 8;
                vecf[i] = Float::from_le_bytes([
                    bytes_buf[s], bytes_buf[s+1], bytes_buf[s+2], bytes_buf[s+3],
                    bytes_buf[s+4], bytes_buf[s+5], bytes_buf[s+6], bytes_buf[s+7],
                ]);
            }
            bn_means.push(vecf);
        }

        // Vars count
        weights_file.read_exact(&mut buf)
            .map_err(|e| format!("Failed to read BN vars count: {}", e))?;
        let num_bn_vars = u64::from_le_bytes(buf) as usize;
        for _ in 0..num_bn_vars {
            weights_file.read_exact(&mut buf)
                .map_err(|e| format!("Failed to read BN var length: {}", e))?;
            let len = u64::from_le_bytes(buf) as usize;
            let mut bytes_buf = vec![0u8; len * 8];
            weights_file.read_exact(&mut bytes_buf)
                .map_err(|e| format!("Failed to read BN var: {}", e))?;
            let mut vecf = vec![0.0; len];
            for i in 0..len {
                let s = i * 8;
                vecf[i] = Float::from_le_bytes([
                    bytes_buf[s], bytes_buf[s+1], bytes_buf[s+2], bytes_buf[s+3],
                    bytes_buf[s+4], bytes_buf[s+5], bytes_buf[s+6], bytes_buf[s+7],
                ]);
            }
            bn_vars.push(vecf);
        }
    }

    // Reconstruct network from metadata and weights
    let mut network = crate::gpu_layer::GpuNetwork::new(device.clone());
    let mut weight_idx: usize = 0;
    let mut bias_idx: usize = 0;
    let mut bn_stat_idx: usize = 0;

    for layer_cfg in &metadata.layers {
        match layer_cfg.layer_type.as_str() {
            "Conv2D" => {
                // Assume defaults used during training: kernel_size=3, stride=1, padding=1
                let in_c = layer_cfg.input_size;
                let out_c = layer_cfg.output_size;
                let mut conv = crate::gpu_layer::GpuConv2D::new(in_c, out_c, 3, 1, 1, device.clone())?;

                // Restore weights (rows>1) then bias (rows==1)
                for param in conv.parameters_mut() {
                    let rows = param.rows();
                    let cols = param.cols();
                    let expected_len = rows * cols;
                    if rows > 1 {
                        if weight_idx >= weights_data.len() { return Err("Weights data exhausted while loading Conv2D".to_string()); }
                        let src = &weights_data[weight_idx];
                        if src.len() != expected_len { return Err(format!("Conv2D weight size mismatch: expected {} for {}x{}, got {}", expected_len, rows, cols, src.len())); }
                        *param = GpuTensor::from_slice(src, rows, cols, device.clone())?;
                        weight_idx += 1;
                    } else {
                        if bias_idx >= biases_data.len() { return Err("Biases data exhausted while loading Conv2D".to_string()); }
                        let src = &biases_data[bias_idx];
                        if src.len() != expected_len { return Err(format!("Conv2D bias size mismatch: expected {} for {}x{}, got {}", expected_len, rows, cols, src.len())); }
                        *param = GpuTensor::from_slice(src, rows, cols, device.clone())?;
                        bias_idx += 1;
                    }
                }

                network.add_layer(Box::new(conv));
            }
            "MaxPool2D" => {
                // Assume defaults used during training: pool=2, stride=2
                let pool = 2usize;
                let stride = 2usize;
                let pool = crate::gpu_layer::GpuMaxPool2D::new(pool, stride, device.clone())?;
                network.add_layer(Box::new(pool));
            }
            "Flatten" => {
                let flat = crate::gpu_layer::GpuFlatten::new();
                network.add_layer(Box::new(flat));
            }
            "Dense" => {
                let activation = match layer_cfg.activation.as_deref() {
                    Some("ReLU") => ActivationType::ReLU,
                    Some("Linear") => ActivationType::Linear,
                    Some("Sigmoid") => ActivationType::Sigmoid,
                    Some("Tanh") => ActivationType::Tanh,
                    _ => ActivationType::ReLU,
                };

                let mut dense = crate::gpu_layer::GpuDense::new(
                    layer_cfg.input_size,
                    layer_cfg.output_size,
                    activation,
                    device.clone(),
                )?;

                // Assign saved weights (rows>1) and biases (rows==1)
                for param in dense.parameters_mut() {
                    let rows = param.rows();
                    let cols = param.cols();
                    let expected_len = rows * cols;
                    if rows > 1 {
                        if weight_idx >= weights_data.len() {
                            return Err("Weights data exhausted while loading model".to_string());
                        }
                        let src = &weights_data[weight_idx];
                        if src.len() != expected_len {
                            return Err(format!(
                                "Weight tensor size mismatch: expected {} elements for {}x{}, got {}",
                                expected_len, rows, cols, src.len()
                            ));
                        }
                        *param = GpuTensor::from_slice(src, rows, cols, device.clone())?;
                        weight_idx += 1;
                    } else {
                        if bias_idx >= biases_data.len() {
                            return Err("Biases data exhausted while loading model".to_string());
                        }
                        let src = &biases_data[bias_idx];
                        if src.len() != expected_len {
                            return Err(format!(
                                "Bias tensor size mismatch: expected {} elements for {}x{}, got {}",
                                expected_len, rows, cols, src.len()
                            ));
                        }
                        *param = GpuTensor::from_slice(src, rows, cols, device.clone())?;
                        bias_idx += 1;
                    }
                }

                network.add_layer(Box::new(dense));
            }
            "BatchNorm" => {
                let mut bn = crate::gpu_layer::GpuBatchNorm::new(layer_cfg.output_size, device.clone())?;
                // BatchNorm learnables (gamma, beta) are row vectors; restore from biases bucket
                for param in bn.parameters_mut() {
                    let rows = param.rows();
                    let cols = param.cols();
                    let expected_len = rows * cols;
                    if bias_idx >= biases_data.len() {
                        return Err("Biases data exhausted while loading BatchNorm params".to_string());
                    }
                    let src = &biases_data[bias_idx];
                    if src.len() != expected_len {
                        return Err(format!(
                            "BatchNorm param size mismatch: expected {} elements for {}x{}, got {}",
                            expected_len, rows, cols, src.len()
                        ));
                    }
                    *param = GpuTensor::from_slice(src, rows, cols, device.clone())?;
                    bias_idx += 1;
                }
                // Restore running stats if present
                if bn_stat_idx < bn_means.len() && bn_stat_idx < bn_vars.len() {
                    let mean_vec = &bn_means[bn_stat_idx];
                    let var_vec = &bn_vars[bn_stat_idx];
                    let mean = GpuTensor::from_slice(mean_vec, 1, layer_cfg.output_size, device.clone())?;
                    let var = GpuTensor::from_slice(var_vec, 1, layer_cfg.output_size, device.clone())?;
                    bn.set_running_stats(&mean, &var)?;
                    bn_stat_idx += 1;
                }
                network.add_layer(Box::new(bn));
            }
            "Dropout" => {
                let dropout = crate::gpu_layer::GpuDropout::new(metadata.dropout_rate, device.clone())?;
                network.add_layer(Box::new(dropout));
            }
            _ => {
                return Err(format!("Unknown layer type: {}", layer_cfg.layer_type));
            }
        }
    }

    Ok((network, metadata))
}

/// Helper to create metadata from network architecture
pub fn create_metadata(
    input_size: usize,
    num_classes: usize,
    dropout_rate: Float,
    layers: Vec<LayerConfig>,
) -> ModelMetadata {
    ModelMetadata {
        input_size,
        num_classes,
        layers,
        dropout_rate,
    }
}

