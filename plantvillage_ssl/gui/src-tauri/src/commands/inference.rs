//! Inference Commands
//!
//! Commands for running inference on images.

use std::path::Path;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use image::imageops::FilterType;

use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

use crate::state::{AppState, AppBackend};

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

/// PlantVillage class names
pub const CLASS_NAMES: [&str; 38] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
];

fn get_class_name(class_id: usize) -> String {
    if class_id < CLASS_NAMES.len() {
        CLASS_NAMES[class_id].to_string()
    } else {
        format!("Unknown_{}", class_id)
    }
}

/// Helper to load model from path
fn load_model_from_path(model_path: &Path) -> Result<PlantClassifier<AppBackend>, String> {
    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
    
    let config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.6,
        in_channels: 3,
        base_filters: 8,
    };
    
    let model: PlantClassifier<AppBackend> = PlantClassifier::new(&config, &device);
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
    
    let img = img.resize_exact(input_size as u32, input_size as u32, FilterType::Triangle);
    let img = img.to_rgb8();

    // Convert to tensor [1, 3, H, W]
    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
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

    let tensor = Tensor::<AppBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, 3, input_size, input_size]);

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
    
    let img = img.resize_exact(input_size as u32, input_size as u32, FilterType::Triangle);
    let img = img.to_rgb8();

    // Convert to tensor [1, 3, H, W]
    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
    let mut pixels: Vec<f32> = Vec::with_capacity(3 * input_size * input_size);
    
    for c in 0..3 {
        for y in 0..input_size {
            for x in 0..input_size {
                let pixel = img.get_pixel(x as u32, y as u32);
                pixels.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    let tensor = Tensor::<AppBackend, 1>::from_floats(pixels.as_slice(), &device)
        .reshape([1, 3, input_size, input_size]);

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
