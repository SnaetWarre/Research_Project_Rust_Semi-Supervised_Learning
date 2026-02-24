//! Model Diagnostics Commands
//!
//! Commands for analyzing model behavior, detecting prediction bias,
//! and providing insights for model improvement.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::State;
use tracing::info;

use rand::seq::SliceRandom;

use plantvillage_ssl::dataset::loader::PlantVillageDataset;

use crate::commands::shared::{get_class_name, load_inference_model, preprocess_image_for_inference};
use crate::state::AppState;

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
    /// Distribution of input classes used for diagnostics (class_name -> count)
    pub input_distribution: HashMap<String, usize>,
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
    let model = load_inference_model(&model_path)?;
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
    let mut input_class_counts: HashMap<String, usize> = HashMap::new();

    // Run predictions on sampled images
    for &i in selected_indices {
        let sample = &dataset.samples[i];
        let path = &sample.path;

        if !path.exists() {
            continue;
        }

        // Count input distribution for debugging bias
        if let Some(parent) = path.parent() {
            if let Some(name) = parent.file_name() {
                let class_name = name.to_string_lossy().to_string();
                *input_class_counts.entry(class_name).or_insert(0) += 1;
            }
        }

        // Load and preprocess image using PIL-compatible resize
        let img = match image::open(path) {
            Ok(img) => img,
            Err(_) => continue,
        };

        let tensor = preprocess_image_for_inference(&img, input_size);

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

    // Log input distribution for debugging bias
    info!("Diagnostics input distribution ({} samples):", total_predictions);
    let mut sorted_inputs: Vec<_> = input_class_counts.iter().collect();
    sorted_inputs.sort_by(|a, b| b.1.cmp(a.1));
    for (class_name, count) in sorted_inputs {
        info!("  {}: {}", class_name, count);
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
        input_distribution: input_class_counts,
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
