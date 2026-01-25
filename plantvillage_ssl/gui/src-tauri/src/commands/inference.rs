//! Inference Commands
//!
//! Commands for running inference on images.

use std::path::Path;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;

use burn::module::Module;
use burn::module::AutodiffModule;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;

use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

use crate::state::{AppState, AppBackend};
use crate::backend::InferenceBackend;

use image::{DynamicImage, RgbImage, Rgb};

/// PIL-compatible bilinear resize with proper anti-aliasing.
/// This matches PIL's Image.resize(size, BILINEAR) exactly.
/// 
/// For downscaling, uses a scaled triangle filter with support radius
/// equal to the scale factor, which provides proper anti-aliasing.
fn pil_bilinear_resize(img: &DynamicImage, target_width: u32, target_height: u32) -> RgbImage {
    let src = img.to_rgb8();
    let src_width = src.width() as usize;
    let src_height = src.height() as usize;
    let target_width = target_width as usize;
    let target_height = target_height as usize;
    
    let mut dst = RgbImage::new(target_width as u32, target_height as u32);
    
    // Scale factors
    let x_scale = src_width as f32 / target_width as f32;
    let y_scale = src_height as f32 / target_height as f32;
    
    // Support radius (for downscaling, use scale factor for anti-aliasing)
    let support_x = x_scale.max(1.0);
    let support_y = y_scale.max(1.0);
    
    for dy in 0..target_height {
        for dx in 0..target_width {
            // Center of output pixel in source coordinates
            let src_cx = (dx as f32 + 0.5) * x_scale;
            let src_cy = (dy as f32 + 0.5) * y_scale;
            
            // Determine contributing source pixels
            let x_min = (src_cx - support_x).floor().max(0.0) as usize;
            let x_max = (src_cx + support_x).ceil().min(src_width as f32 - 1.0) as usize;
            let y_min = (src_cy - support_y).floor().max(0.0) as usize;
            let y_max = (src_cy + support_y).ceil().min(src_height as f32 - 1.0) as usize;
            
            let mut total_weight = 0.0f32;
            let mut weighted_sum = [0.0f32; 3]; // RGB
            
            for sy in y_min..=y_max {
                for sx in x_min..=x_max {
                    // Normalized distance from center
                    let dist_x = ((sx as f32 + 0.5) - src_cx).abs() / support_x;
                    let dist_y = ((sy as f32 + 0.5) - src_cy).abs() / support_y;
                    
                    // Triangle (bilinear) kernel
                    if dist_x < 1.0 && dist_y < 1.0 {
                        let weight_x = 1.0 - dist_x;
                        let weight_y = 1.0 - dist_y;
                        let weight = weight_x * weight_y;
                        
                        let pixel = src.get_pixel(sx as u32, sy as u32);
                        weighted_sum[0] += pixel[0] as f32 * weight;
                        weighted_sum[1] += pixel[1] as f32 * weight;
                        weighted_sum[2] += pixel[2] as f32 * weight;
                        total_weight += weight;
                    }
                }
            }
            
            if total_weight > 0.0 {
                dst.put_pixel(
                    dx as u32,
                    dy as u32,
                    Rgb([
                        (weighted_sum[0] / total_weight).round() as u8,
                        (weighted_sum[1] / total_weight).round() as u8,
                        (weighted_sum[2] / total_weight).round() as u8,
                    ])
                );
            }
        }
    }
    
    dst
}

/// Prediction result for a single image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predicted_class: usize,
    pub predicted_class_name: String,
    pub confidence: f32,
    pub probabilities: Vec<f32>,
    pub top_5: Vec<ClassPrediction>,
    pub inference_time_ms: f64,
}

/// A single class prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassPrediction {
    pub class_id: usize,
    pub class_name: String,
    pub probability: f32,
}

/// PlantVillage class names - MUST match dataset loader sort order (case-sensitive).
pub const CLASS_NAMES: [&str; 38] = [
    "Apple___Apple_scab",                                   // 0
    "Apple___Black_rot",                                    // 1
    "Apple___Cedar_apple_rust",                             // 2
    "Apple___healthy",                                      // 3
    "Blueberry___healthy",                                  // 4
    "Cherry_(including_sour)___Powdery_mildew",             // 5
    "Cherry_(including_sour)___healthy",                    // 6
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",   // 7
    "Corn_(maize)___Common_rust_",                          // 8
    "Corn_(maize)___Northern_Leaf_Blight",                  // 9
    "Corn_(maize)___healthy",                               // 10
    "Grape___Black_rot",                                    // 11
    "Grape___Esca_(Black_Measles)",                         // 12
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",           // 13
    "Grape___healthy",                                      // 14
    "Orange___Haunglongbing_(Citrus_greening)",             // 15
    "Peach___Bacterial_spot",                               // 16
    "Peach___healthy",                                      // 17
    "Pepper,_bell___Bacterial_spot",                        // 18
    "Pepper,_bell___healthy",                               // 19
    "Potato___Early_blight",                                // 20
    "Potato___Late_blight",                                 // 21
    "Potato___healthy",                                     // 22
    "Raspberry___healthy",                                  // 23
    "Soybean___healthy",                                    // 24
    "Squash___Powdery_mildew",                              // 25
    "Strawberry___Leaf_scorch",                             // 26
    "Strawberry___healthy",                                 // 27
    "Tomato___Bacterial_spot",                              // 28
    "Tomato___Early_blight",                                // 29
    "Tomato___Late_blight",                                 // 30
    "Tomato___Leaf_Mold",                                   // 31
    "Tomato___Septoria_leaf_spot",                          // 32
    "Tomato___Spider_mites Two-spotted_spider_mite",        // 33
    "Tomato___Target_Spot",                                 // 34
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",               // 35
    "Tomato___Tomato_mosaic_virus",                         // 36
    "Tomato___healthy",                                     // 37
];

fn get_class_name(class_id: usize) -> String {
    if class_id < CLASS_NAMES.len() {
        CLASS_NAMES[class_id].to_string()
    } else {
        format!("Unknown_{}", class_id)
    }
}

/// Helper to load model from path (for inference - no autodiff, no dropout)
fn load_model_from_path(model_path: &Path) -> Result<PlantClassifier<InferenceBackend>, String> {
    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();

    let config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    let model: PlantClassifier<InferenceBackend> = PlantClassifier::new(&config, &device);
    let recorder = CompactRecorder::new();

    model
        .load_file(model_path, &recorder, &device)
        .map_err(|e| format!("Failed to load model: {:?}", e))
}

/// Run inference on a single image
#[tauri::command]
pub async fn run_inference(
    image_path: String,
    state: State<'_, Arc<AppState>>,
) -> Result<PredictionResult, String> {
    let model_path = state.model_path.read().await;
    let model_path = model_path
        .as_ref()
        .ok_or("No model loaded. Please load a model first.")?
        .clone();

    let path = Path::new(&image_path);
    if !path.exists() {
        return Err(format!("Image not found: {}", image_path));
    }

    // Load model on-demand (required due to CUDA threading)
    let model = load_model_from_path(&model_path)?;
    let input_size = 128usize;

    // Load and preprocess image
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image: {:?}", e))?;

    let img = pil_bilinear_resize(&img, input_size as u32, input_size as u32);

    // Convert to tensor [1, 3, H, W]
    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
    let mut pixels: Vec<f32> = Vec::with_capacity(3 * input_size * input_size);

    // Normalize to [0, 1] and arrange as CHW
    for c in 0..3 {
        for y in 0..input_size {
            for x in 0..input_size {
                let pixel = img.get_pixel(x as u32, y as u32);
                pixels.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    let tensor = Tensor::<InferenceBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, 3, input_size, input_size]);

    // Debug: Print raw pixel stats
    eprintln!("DEBUG: Raw pixel values (0-1) - min={:.3}, max={:.3}, mean={:.3}", 
        pixels.iter().cloned().fold(f32::INFINITY, f32::min),
        pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        pixels.iter().sum::<f32>() / pixels.len() as f32
    );

    // Apply ImageNet normalization: (x - mean) / std
    let mean = Tensor::<InferenceBackend, 4>::from_floats(
        burn::tensor::TensorData::new(vec![0.485f32, 0.456, 0.406], [1, 3, 1, 1]),
        &device,
    );
    let std = Tensor::<InferenceBackend, 4>::from_floats(
        burn::tensor::TensorData::new(vec![0.229f32, 0.224, 0.225], [1, 3, 1, 1]),
        &device,
    );
    let tensor = (tensor - mean) / std;

    // Debug: Print tensor stats  
    eprintln!("DEBUG: Tensor shape: {:?}", tensor.dims());
    let tensor_clone = tensor.clone();
    let min_val = tensor_clone.clone().min().into_scalar();
    let max_val = tensor_clone.clone().max().into_scalar();
    let mean_val = tensor_clone.clone().mean().into_scalar();
    eprintln!("DEBUG: Normalized tensor - min={:.3}, max={:.3}, mean={:.3}", min_val, max_val, mean_val);

    // Run inference
    let start = Instant::now();
    let output = model.forward_softmax(tensor);
    let inference_time = start.elapsed();

    // Extract probabilities
    let output_data = output.into_data();
    let probs: Vec<f32> = output_data.to_vec()
        .map_err(|e| format!("Failed to extract probabilities: {:?}", e))?;

    // Find top prediction
    let (predicted_class, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &conf)| (idx, conf))
        .unwrap_or((0, 0.0));

    // Get top 5 predictions
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let top_5: Vec<ClassPrediction> = indexed_probs
        .iter()
        .take(5)
        .map(|(idx, prob)| ClassPrediction {
            class_id: *idx,
            class_name: get_class_name(*idx),
            probability: *prob,
        })
        .collect();

    Ok(PredictionResult {
        predicted_class,
        predicted_class_name: get_class_name(predicted_class),
        confidence,
        probabilities: probs,
        top_5,
        inference_time_ms: inference_time.as_secs_f64() * 1000.0,
    })
}

/// Run inference on image bytes (for drag and drop)
#[tauri::command]
pub async fn run_inference_bytes(
    image_bytes: Vec<u8>,
    state: State<'_, Arc<AppState>>,
) -> Result<PredictionResult, String> {
    let model_path = state.model_path.read().await;
    let model_path = model_path
        .as_ref()
        .ok_or("No model loaded. Please load a model first.")?
        .clone();

    // Load model on-demand
    let model = load_model_from_path(&model_path)?;
    let input_size = 128usize;

    // Load image from bytes
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("Failed to load image: {:?}", e))?;

    let img = pil_bilinear_resize(&img, input_size as u32, input_size as u32);

    // Convert to tensor [1, 3, H, W]
    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
    let mut pixels: Vec<f32> = Vec::with_capacity(3 * input_size * input_size);

    for c in 0..3 {
        for y in 0..input_size {
            for x in 0..input_size {
                let pixel = img.get_pixel(x as u32, y as u32);
                pixels.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    let tensor = Tensor::<InferenceBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, 3, input_size, input_size]);

    // Debug: Print raw pixel stats
    eprintln!("DEBUG: Raw pixel values (0-1) - min={:.3}, max={:.3}, mean={:.3}", 
        pixels.iter().cloned().fold(f32::INFINITY, f32::min),
        pixels.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        pixels.iter().sum::<f32>() / pixels.len() as f32
    );

    // Apply ImageNet normalization: (x - mean) / std
    let mean = Tensor::<InferenceBackend, 4>::from_floats(
        burn::tensor::TensorData::new(vec![0.485f32, 0.456, 0.406], [1, 3, 1, 1]),
        &device,
    );
    let std = Tensor::<InferenceBackend, 4>::from_floats(
        burn::tensor::TensorData::new(vec![0.229f32, 0.224, 0.225], [1, 3, 1, 1]),
        &device,
    );
    let tensor = (tensor - mean) / std;

    // Debug: Print tensor stats  
    eprintln!("DEBUG: Tensor shape: {:?}", tensor.dims());
    let tensor_clone = tensor.clone();
    let min_val = tensor_clone.clone().min().into_scalar();
    let max_val = tensor_clone.clone().max().into_scalar();
    let mean_val = tensor_clone.clone().mean().into_scalar();
    eprintln!("DEBUG: Normalized tensor - min={:.3}, max={:.3}, mean={:.3}", min_val, max_val, mean_val);

    // Run inference
    let start = Instant::now();
    let output = model.forward_softmax(tensor);
    let inference_time = start.elapsed();

    // Extract probabilities
    let output_data = output.into_data();
    let probs: Vec<f32> = output_data.to_vec()
        .map_err(|e| format!("Failed to extract probabilities: {:?}", e))?;

    let (predicted_class, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, &conf)| (idx, conf))
        .unwrap_or((0, 0.0));

    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

    let top_5: Vec<ClassPrediction> = indexed_probs
        .iter()
        .take(5)
        .map(|(idx, prob)| ClassPrediction {
            class_id: *idx,
            class_name: get_class_name(*idx),
            probability: *prob,
        })
        .collect();

    Ok(PredictionResult {
        predicted_class,
        predicted_class_name: get_class_name(predicted_class),
        confidence,
        probabilities: probs,
        top_5,
        inference_time_ms: inference_time.as_secs_f64() * 1000.0,
    })
}
