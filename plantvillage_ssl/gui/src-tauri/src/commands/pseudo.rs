//! Pseudo-Labeling Demo Commands
//!
//! Commands for demonstrating pseudo-labeling with interactive confidence thresholds.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::State;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::Tensor;
use image::imageops::FilterType;

use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

use crate::state::{AppState, AppBackend};
use crate::commands::inference::CLASS_NAMES;

/// Single sample for pseudo-labeling demo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLabelSample {
    pub image_path: String,
    pub predicted_class: usize,
    pub predicted_class_name: String,
    pub confidence: f32,
    pub ground_truth: Option<usize>,
    pub ground_truth_name: Option<String>,
    pub is_correct: Option<bool>,
    pub accepted: bool,
}

/// Results from pseudo-labeling demo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLabelResults {
    pub samples: Vec<PseudoLabelSample>,
    pub total_processed: usize,
    pub total_accepted: usize,
    pub total_rejected: usize,
    pub precision: f64,
    pub acceptance_rate: f64,
    pub class_distribution: Vec<ClassCount>,
}

/// Class count for distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassCount {
    pub class_id: usize,
    pub class_name: String,
    pub count: usize,
}

fn get_class_name(class_id: usize) -> String {
    if class_id < CLASS_NAMES.len() {
        CLASS_NAMES[class_id].to_string()
    } else {
        format!("Unknown_{}", class_id)
    }
}

/// Helper to load model from path
fn load_model_from_path(model_path: &std::path::Path) -> Result<PlantClassifier<AppBackend>, String> {
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

/// Run pseudo-labeling demo on a set of images
#[tauri::command]
pub async fn run_pseudo_label_demo(
    image_paths: Vec<String>,
    confidence_threshold: f64,
    state: State<'_, Arc<AppState>>,
) -> Result<PseudoLabelResults, String> {
    let model_path = state.model_path.read().await;
    let model_path = model_path
        .as_ref()
        .ok_or("No model loaded. Please load a model first.")?
        .clone();

    // Load model on-demand
    let model = load_model_from_path(&model_path)?;

    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
    let input_size = 128usize;

    let mut samples = Vec::new();
    let mut class_counts = std::collections::HashMap::new();
    let mut correct_count = 0usize;
    let mut accepted_count = 0usize;

    for image_path in &image_paths {
        let path = std::path::Path::new(image_path);
        if !path.exists() {
            continue;
        }

        // Try to extract ground truth from path (PlantVillage format: class_name/image.jpg)
        let ground_truth = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .and_then(|class_name| {
                CLASS_NAMES.iter().position(|&n| n == class_name)
            });

        // Load and preprocess image
        let img = match image::open(path) {
            Ok(img) => img,
            Err(_) => continue,
        };

        let img = img.resize_exact(input_size as u32, input_size as u32, FilterType::Triangle);
        let img = img.to_rgb8();

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

        let output = model.forward_softmax(tensor);
        let output_data = output.into_data();
        let probs: Vec<f32> = match output_data.to_vec() {
            Ok(p) => p,
            Err(_) => continue,
        };

        let (predicted_class, confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &conf)| (idx, conf))
            .unwrap_or((0, 0.0));

        let accepted = confidence >= confidence_threshold as f32;
        let is_correct = ground_truth.map(|gt| gt == predicted_class);

        if accepted {
            accepted_count += 1;
            *class_counts.entry(predicted_class).or_insert(0) += 1;

            if is_correct == Some(true) {
                correct_count += 1;
            }
        }

        samples.push(PseudoLabelSample {
            image_path: image_path.clone(),
            predicted_class,
            predicted_class_name: get_class_name(predicted_class),
            confidence,
            ground_truth,
            ground_truth_name: ground_truth.map(get_class_name),
            is_correct,
            accepted,
        });
    }

    let total_processed = samples.len();
    let total_rejected = total_processed - accepted_count;

    // Calculate precision (only on accepted samples with known ground truth)
    let accepted_with_gt: Vec<_> = samples.iter()
        .filter(|s| s.accepted && s.is_correct.is_some())
        .collect();
    let precision = if !accepted_with_gt.is_empty() {
        correct_count as f64 / accepted_with_gt.len() as f64
    } else {
        0.0
    };

    let acceptance_rate = if total_processed > 0 {
        accepted_count as f64 / total_processed as f64
    } else {
        0.0
    };

    // Build class distribution
    let mut class_distribution: Vec<ClassCount> = class_counts
        .into_iter()
        .map(|(class_id, count)| ClassCount {
            class_id,
            class_name: get_class_name(class_id),
            count,
        })
        .collect();
    class_distribution.sort_by(|a, b| b.count.cmp(&a.count));

    Ok(PseudoLabelResults {
        samples,
        total_processed,
        total_accepted: accepted_count,
        total_rejected,
        precision: precision * 100.0,
        acceptance_rate: acceptance_rate * 100.0,
        class_distribution,
    })
}

/// Get sample images from dataset for demo
#[tauri::command]
pub async fn get_sample_images(
    data_dir: String,
    count: usize,
) -> Result<Vec<String>, String> {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use walkdir::WalkDir;

    let path = std::path::Path::new(&data_dir);
    if !path.exists() {
        return Err(format!("Dataset directory not found: {}", data_dir));
    }

    let mut image_paths: Vec<String> = Vec::new();

    for entry in WalkDir::new(&data_dir).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            if ext == "jpg" || ext == "jpeg" || ext == "png" {
                image_paths.push(path.to_string_lossy().to_string());
            }
        }
    }

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    image_paths.shuffle(&mut rng);

    Ok(image_paths.into_iter().take(count).collect())
}
