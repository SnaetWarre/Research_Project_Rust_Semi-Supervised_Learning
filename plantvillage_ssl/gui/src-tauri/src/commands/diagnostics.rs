//! Model Diagnostics Commands
//!
//! Commands for analyzing model behavior, detecting prediction bias,
//! and providing insights for model improvement.

use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use image::imageops::FilterType;
use rand::seq::SliceRandom;

use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::dataset::loader::PlantVillageDataset;

use crate::state::{AppState, AppBackend};

/// Diagnostic results for model analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    /// Number of times each class was predicted
    pub class_predictions: HashMap<usize, usize>,
    /// Confidence scores for each class
    pub class_confidences: HashMap<usize, Vec<f32>>,
    /// Total number of predictions made
    pub total_predictions: usize,
    /// Most frequently predicted class
    pub most_predicted_class: usize,
    /// Name of most predicted class
    pub most_predicted_class_name: String,
    /// Bias score (0-1, higher = more biased)
    pub prediction_bias_score: f64,
    /// Number of low confidence predictions
    pub low_confidence_count: usize,
    /// Expected class distribution from dataset
    pub class_distribution: HashMap<usize, usize>,
}

/// PlantVillage class names (must match inference.rs)
pub const CLASS_NAMES: [&str; 38] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
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
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    let model: PlantClassifier<AppBackend> = PlantClassifier::new(&config, &device);
    let recorder = CompactRecorder::new();

    model
        .load_file(model_path, &recorder, &device)
        .map_err(|e| format!("Failed to load model: {:?}", e))
}

/// Run comprehensive model diagnostics
#[tauri::command]
pub async fn run_model_diagnostics(
    data_dir: String,
    num_samples: usize,
    confidence_threshold: f64,
    state: State<'_, Arc<AppState>>,
) -> Result<DiagnosticResult, String> {
    let model_path = state.model_path.read().await;
    let model_path = model_path
        .as_ref()
        .ok_or("No model loaded. Please load a model first.")?
        .clone();

    // Load model
    let model = load_model_from_path(&model_path)?;
    let input_size = 128usize;

    // Load dataset to get samples
    let dataset = PlantVillageDataset::new(&data_dir)
        .map_err(|e| format!("Failed to load dataset: {:?}", e))?;

    if dataset.samples.is_empty() {
        return Err("Dataset is empty".to_string());
    }

    // Collect class distribution
    let mut class_distribution: HashMap<usize, usize> = HashMap::new();
    for sample in &dataset.samples {
        *class_distribution.entry(sample.label).or_insert(0) += 1;
    }

    // Sample a subset for diagnostics
    let samples_to_test = num_samples.min(dataset.samples.len());
    
    // Create random indices to ensure balanced sampling across classes
    let mut indices: Vec<usize> = (0..dataset.samples.len()).collect();
    let mut rng = rand::thread_rng();
    indices.shuffle(&mut rng);
    
    // Take the first N random indices
    let selected_indices = &indices[0..samples_to_test];

    let mut class_predictions: HashMap<usize, usize> = HashMap::new();
    let mut class_confidences: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut total_predictions = 0;
    let mut low_confidence_count = 0;

    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();

    // Run predictions on sampled images
    for &i in selected_indices {
        let sample = &dataset.samples[i];
        let path = &sample.path;

        if !path.exists() {
            continue;
        }

        // Load and preprocess image
        let img = match image::open(path) {
            Ok(img) => img,
            Err(_) => continue,
        };

        let img = img.resize_exact(input_size as u32, input_size as u32, FilterType::Triangle);
        let img = img.to_rgb8();

        // Convert to tensor
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
        let output = model.forward_softmax(tensor);
        let output_data = output.into_data();
        let probs: Vec<f32> = match output_data.to_vec() {
            Ok(p) => p,
            Err(_) => continue,
        };

        // Find predicted class and confidence
        let (predicted_class, confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &conf)| (idx, conf))
            .unwrap_or((0, 0.0));

        // Record prediction
        *class_predictions.entry(predicted_class).or_insert(0) += 1;
        class_confidences
            .entry(predicted_class)
            .or_insert_with(Vec::new)
            .push(confidence);

        total_predictions += 1;

        if confidence < confidence_threshold as f32 {
            low_confidence_count += 1;
        }
    }

    if total_predictions == 0 {
        return Err("No valid predictions were made".to_string());
    }

    // Find most predicted class
    let (most_predicted_class, most_predicted_count) = class_predictions
        .iter()
        .max_by_key(|(_, &count)| count)
        .map(|(&class, &count)| (class, count))
        .unwrap_or((0, 0));

    // Calculate bias score using Gini coefficient
    // Higher score = more concentrated predictions (more bias)
    let prediction_bias_score = calculate_gini_coefficient(&class_predictions, total_predictions);

    Ok(DiagnosticResult {
        class_predictions,
        class_confidences,
        total_predictions,
        most_predicted_class,
        most_predicted_class_name: get_class_name(most_predicted_class),
        prediction_bias_score,
        low_confidence_count,
        class_distribution,
    })
}

/// Calculate Gini coefficient to measure prediction concentration
/// Returns value between 0 (perfectly distributed) and 1 (all predictions in one class)
fn calculate_gini_coefficient(predictions: &HashMap<usize, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let mut counts: Vec<f64> = predictions.values().map(|&c| c as f64).collect();
    counts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = counts.len() as f64;
    let total = total as f64;

    let mut sum_weighted = 0.0;
    for (i, count) in counts.iter().enumerate() {
        sum_weighted += (2.0 * (i as f64 + 1.0) - n - 1.0) * count;
    }

    let gini = sum_weighted / (n * total);
    gini.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gini_coefficient() {
        // Perfect distribution
        let mut perfect = HashMap::new();
        perfect.insert(0, 10);
        perfect.insert(1, 10);
        perfect.insert(2, 10);
        let gini = calculate_gini_coefficient(&perfect, 30);
        assert!(gini < 0.01); // Should be very low

        // All in one class
        let mut concentrated = HashMap::new();
        concentrated.insert(0, 30);
        let gini = calculate_gini_coefficient(&concentrated, 30);
        assert!(gini > 0.9); // Should be very high
    }
}
