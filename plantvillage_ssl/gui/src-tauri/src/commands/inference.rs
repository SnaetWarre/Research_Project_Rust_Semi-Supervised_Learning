//! Inference Commands
//!
//! Commands for running inference on images.

use std::path::Path;
use std::time::Instant;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::commands::shared::{get_class_name, load_inference_model, preprocess_image_for_inference};
use crate::state::AppState;

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
    let model = load_inference_model(&model_path)?;
    let input_size = 128usize;
    let img = image::open(path)
        .map_err(|e| format!("Failed to load image: {:?}", e))?;
    let tensor = preprocess_image_for_inference(&img, input_size);

    // Run inference
    let start = Instant::now();
    let output = model.forward_softmax(tensor);
    let inference_time = start.elapsed();

    // Extract probabilities
    let output_data = output.into_data();
    let probs = output_data.to_vec::<f32>()
        .map_err(|e| format!("Failed to extract probabilities: {:?}", e))?;

    // Find top prediction
    let (predicted_class, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, &conf)| (idx, conf))
        .unwrap_or((0, 0.0));

    // Get top 5 predictions
    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.total_cmp(&a.1));

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
    let model = load_inference_model(&model_path)?;
    let input_size = 128usize;
    let img = image::load_from_memory(&image_bytes)
        .map_err(|e| format!("Failed to load image: {:?}", e))?;
    let tensor = preprocess_image_for_inference(&img, input_size);

    // Run inference
    let start = Instant::now();
    let output = model.forward_softmax(tensor);
    let inference_time = start.elapsed();

    // Extract probabilities
    let output_data = output.into_data();
    let probs = output_data.to_vec::<f32>()
        .map_err(|e| format!("Failed to extract probabilities: {:?}", e))?;

    let (predicted_class, confidence) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, &conf)| (idx, conf))
        .unwrap_or((0, 0.0));

    let mut indexed_probs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    indexed_probs.sort_by(|a, b| b.1.total_cmp(&a.1));

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
