//! Dataset Commands
//!
//! Commands for loading datasets and getting statistics.

use std::path::PathBuf;
use std::fs;
use serde::{Deserialize, Serialize};
use tauri::State;
use std::sync::Arc;

use plantvillage_ssl::PlantVillageDataset;

use crate::state::{AppState, DatasetInfo};

/// Response for model info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub loaded: bool,
    pub path: Option<String>,
    pub num_classes: usize,
    pub input_size: usize,
}

/// Helper function to find the newest model file in likely directories
fn find_newest_model() -> Option<PathBuf> {
    let search_paths = vec![
        PathBuf::from("output/research_pipeline/ssl"),
        PathBuf::from("output/research_pipeline/burn"),
        PathBuf::from("plantvillage_ssl/output/research_pipeline/ssl"),
        PathBuf::from("plantvillage_ssl/output/research_pipeline/burn"),
        PathBuf::from("../output/research_pipeline/ssl"),
        PathBuf::from("../output/research_pipeline/burn"),
        PathBuf::from("../../output/research_pipeline/ssl"),
        PathBuf::from("../../output/research_pipeline/burn"),
    ];

    let mut newest_file: Option<PathBuf> = None;
    let mut newest_time = std::time::SystemTime::UNIX_EPOCH;

    for dir in search_paths {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("mpk") {
                    if let Ok(metadata) = fs::metadata(&path) {
                        if let Ok(modified) = metadata.modified() {
                            if modified > newest_time {
                                newest_time = modified;
                                newest_file = Some(path);
                            }
                        }
                    }
                }
            }
        }
    }

    newest_file
}

/// Get dataset statistics
#[tauri::command]
pub async fn get_dataset_stats(
    data_dir: String,
    state: State<'_, Arc<AppState>>,
) -> Result<DatasetInfo, String> {
    let mut path = PathBuf::from(&data_dir);

    // Auto-discovery of dataset directory
    if !path.exists() {
        let candidates = vec![
            PathBuf::from(&data_dir),
            PathBuf::from(format!("../{}", data_dir)),
            PathBuf::from(format!("../../{}", data_dir)),
            PathBuf::from("data/plantvillage/balanced"),
            PathBuf::from("../data/plantvillage/balanced"),
        ];

        for candidate in candidates {
            if candidate.exists() {
                path = candidate;
                break;
            }
        }
    }

    if !path.exists() {
        return Err(format!("Dataset directory not found: {}", data_dir));
    }

    let path_str = path.to_string_lossy().to_string();

    let dataset = PlantVillageDataset::new(&path_str)
        .map_err(|e| format!("Failed to load dataset: {:?}", e))?;

    let stats = dataset.get_stats();

    let class_names: Vec<String> = (0..stats.num_classes)
        .map(|i| stats.class_names.get(&i).cloned().unwrap_or_else(|| format!("Class {}", i)))
        .collect();

    let class_counts: Vec<usize> = (0..stats.num_classes)
        .map(|i| stats.class_counts.get(i).copied().unwrap_or(0))
        .collect();

    let info = DatasetInfo {
        path: path_str,
        total_samples: stats.total_samples,
        num_classes: stats.num_classes,
        class_names,
        class_counts,
    };

    // Cache the dataset info
    let mut dataset_info = state.dataset_info.write().await;
    *dataset_info = Some(info.clone());

    Ok(info)
}

/// Verify and store model path (actual loading happens on-demand due to CUDA threading)
#[tauri::command]
pub async fn load_model(
    model_path: String,
    state: State<'_, Arc<AppState>>,
) -> Result<ModelInfo, String> {
    let mut path = PathBuf::from(&model_path);

    // Special case for auto-discovery "best_model.mpk" or "auto"
    if model_path == "best_model.mpk" || model_path == "auto" {
        // First priority: Try to find the newest model from experiment outputs
        if let Some(newest) = find_newest_model() {
            path = newest;
        }
        // Fallback: Check if best_model.mpk actually exists as a file/symlink
        else if !path.exists() {
             // Fallback search logic for specific paths if find_newest didn't work
             let candidates = vec![
                PathBuf::from("best_model.mpk"),
                PathBuf::from("../best_model.mpk"),
                PathBuf::from("plantvillage_ssl/best_model.mpk"),
            ];
            for candidate in candidates {
                if candidate.exists() {
                    path = candidate;
                    break;
                }
            }
        }
    }
    else if !path.exists() {
         // Standard relative path search
         let candidates = vec![
            PathBuf::from(&model_path),
            PathBuf::from(format!("../{}", model_path)),
            PathBuf::from(format!("output/checkpoints/{}", model_path)),
            PathBuf::from(format!("../output/checkpoints/{}", model_path)),
        ];

        for candidate in candidates {
            if candidate.exists() {
                path = candidate;
                break;
            }
             // Try with .mpk extension
             let with_ext = PathBuf::from(format!("{}.mpk", candidate.to_string_lossy()));
             if with_ext.exists() {
                path = with_ext;
                break;
            }
        }
    }

    if !path.exists() {
        return Err(format!("Model file not found: {}", model_path));
    }

    // Store the path in state (model will be loaded on-demand)
    let mut model_path_state = state.model_path.write().await;
    *model_path_state = Some(path.clone());

    Ok(ModelInfo {
        loaded: true,
        path: Some(path.to_string_lossy().to_string()),
        num_classes: 38,
        input_size: 128,
    })
}

/// Check if model is loaded
#[tauri::command]
pub async fn get_model_status(
    state: State<'_, Arc<AppState>>,
) -> Result<ModelInfo, String> {
    let model_path = state.model_path.read().await;

    match &*model_path {
        Some(path) => Ok(ModelInfo {
            loaded: true,
            path: Some(path.to_string_lossy().to_string()),
            num_classes: 38,
            input_size: 128,
        }),
        None => Ok(ModelInfo {
            loaded: false,
            path: None,
            num_classes: 0,
            input_size: 0,
        }),
    }
}

/// Get cached dataset info
#[tauri::command]
pub async fn get_cached_dataset_info(
    state: State<'_, Arc<AppState>>,
) -> Result<Option<DatasetInfo>, String> {
    let info = state.dataset_info.read().await;
    Ok(info.clone())
}
