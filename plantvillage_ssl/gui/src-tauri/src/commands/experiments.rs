//! Experiment Results Commands
//!
//! Commands for loading and displaying pre-run experiment results.
//! This allows the GUI to demo results without running live training.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Label efficiency experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelEfficiencyResults {
    pub images_per_class: Vec<usize>,
    pub accuracies: Vec<f64>,
    pub training_times: Vec<f64>,
    pub best_accuracy: f64,
    pub best_images_per_class: usize,
    pub min_acceptable_images: Option<usize>,
}

/// Class scaling experiment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassScalingResults {
    pub small_base: ScalingResult,
    pub large_base: ScalingResult,
    pub relative_difficulty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingResult {
    pub base_classes: usize,
    pub total_classes: usize,
    pub base_accuracy_before: f64,
    pub base_accuracy_after: f64,
    pub new_class_accuracy: f64,
    pub overall_accuracy: f64,
    pub forgetting: f64,
    pub training_time: f64,
}

/// SSL + Incremental Learning results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSLIncrementalResults {
    pub base_accuracy: f64,
    pub with_ssl: IncrementalStepResult,
    pub without_ssl: IncrementalStepResult,
    pub ssl_improvement: f64,
    pub pseudo_labels_generated: usize,
    pub pseudo_label_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalStepResult {
    pub old_class_accuracy: f64,
    pub new_class_accuracy: f64,
    pub overall_accuracy: f64,
    pub forgetting: f64,
    pub training_time: f64,
}

/// All experiment results combined
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllExperimentResults {
    pub label_efficiency: Option<LabelEfficiencyResults>,
    pub class_scaling: Option<ClassScalingResults>,
    pub ssl_incremental: Option<SSLIncrementalResults>,
    pub has_results: bool,
}

/// Load label efficiency experiment results
#[tauri::command]
pub async fn load_label_efficiency_results(
    results_dir: Option<String>,
) -> Result<LabelEfficiencyResults, String> {
    let base_path = results_dir.unwrap_or_else(|| "output/experiments/label_efficiency".to_string());
    let results_path = Path::new(&base_path).join("results.json");

    let json = fs::read_to_string(&results_path)
        .map_err(|e| format!("Failed to read results: {:?}", e))?;
    
    let results: LabelEfficiencyResults = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse results: {:?}", e))?;

    Ok(results)
}

/// Load class scaling experiment results
#[tauri::command]
pub async fn load_class_scaling_results(
    results_dir: Option<String>,
) -> Result<ClassScalingResults, String> {
    let base_path = results_dir.unwrap_or_else(|| "output/experiments/class_scaling".to_string());
    let results_path = Path::new(&base_path).join("results.json");

    let json = fs::read_to_string(&results_path)
        .map_err(|e| format!("Failed to read results: {:?}", e))?;
    
    let results: ClassScalingResults = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse results: {:?}", e))?;

    Ok(results)
}

/// Load SSL + Incremental Learning experiment results
#[tauri::command]
pub async fn load_ssl_incremental_results(
    results_dir: Option<String>,
) -> Result<SSLIncrementalResults, String> {
    let base_path = results_dir.unwrap_or_else(|| "output/experiments/ssl_incremental".to_string());
    let results_path = Path::new(&base_path).join("results.json");

    let json = fs::read_to_string(&results_path)
        .map_err(|e| format!("Failed to read results: {:?}", e))?;
    
    let results: SSLIncrementalResults = serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse results: {:?}", e))?;

    Ok(results)
}

/// Load all available experiment results
#[tauri::command]
pub async fn load_all_experiment_results(
    base_dir: Option<String>,
) -> Result<AllExperimentResults, String> {
    let mut base_path = base_dir.unwrap_or_else(|| "output/experiments".to_string());
    
    // Auto-discovery of experiments directory if default doesn't exist
    if !Path::new(&base_path).exists() {
        if Path::new("../output/experiments").exists() {
            base_path = "../output/experiments".to_string();
        } else if Path::new("../../output/experiments").exists() {
             base_path = "../../output/experiments".to_string();
        }
    }

    let label_efficiency = load_label_efficiency_results(
        Some(format!("{}/label_efficiency", base_path))
    ).await.ok();

    let class_scaling = load_class_scaling_results(
        Some(format!("{}/class_scaling", base_path))
    ).await.ok();

    let ssl_incremental = load_ssl_incremental_results(
        Some(format!("{}/ssl_incremental", base_path))
    ).await.ok();

    let has_results = label_efficiency.is_some() 
        || class_scaling.is_some() 
        || ssl_incremental.is_some();

    Ok(AllExperimentResults {
        label_efficiency,
        class_scaling,
        ssl_incremental,
        has_results,
    })
}

/// Get conclusions text for an experiment
#[tauri::command]
pub async fn load_experiment_conclusions(
    experiment: String,
    results_dir: Option<String>,
) -> Result<String, String> {
    let base_path = results_dir.unwrap_or_else(|| "output/experiments".to_string());
    let conclusions_path = Path::new(&base_path)
        .join(&experiment)
        .join("conclusions.txt");

    fs::read_to_string(&conclusions_path)
        .map_err(|e| format!("Failed to read conclusions: {:?}", e))
}

/// Check which experiments have results available
#[tauri::command]
pub async fn get_available_experiments(
    base_dir: Option<String>,
) -> Result<Vec<String>, String> {
    let base_path = base_dir.unwrap_or_else(|| "output/experiments".to_string());
    let base = Path::new(&base_path);

    let mut available = Vec::new();

    if base.join("label_efficiency/results.json").exists() {
        available.push("label_efficiency".to_string());
    }
    if base.join("class_scaling/results.json").exists() {
        available.push("class_scaling".to_string());
    }
    if base.join("ssl_incremental/results.json").exists() {
        available.push("ssl_incremental".to_string());
    }

    Ok(available)
}
