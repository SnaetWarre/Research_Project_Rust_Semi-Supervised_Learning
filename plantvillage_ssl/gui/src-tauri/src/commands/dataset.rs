//! Dataset Commands
//!
//! Commands for loading datasets and getting statistics.

use std::path::{Path, PathBuf};
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

/// Get dataset statistics
#[tauri::command]
pub async fn get_dataset_stats(
    data_dir: String,
    state: State<'_, Arc<AppState>>,
) -> Result<DatasetInfo, String> {
    let path = Path::new(&data_dir);
    
    if !path.exists() {
        return Err(format!("Dataset directory not found: {}", data_dir));
    }

    let dataset = PlantVillageDataset::new(&data_dir)
        .map_err(|e| format!("Failed to load dataset: {:?}", e))?;
    
    let stats = dataset.get_stats();
    
    let class_names: Vec<String> = (0..stats.num_classes)
        .map(|i| stats.class_names.get(&i).cloned().unwrap_or_else(|| format!("Class {}", i)))
        .collect();
    
    let class_counts: Vec<usize> = (0..stats.num_classes)
        .map(|i| stats.class_counts.get(i).copied().unwrap_or(0))
        .collect();

    let info = DatasetInfo {
        path: data_dir.clone(),
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
    let path = std::path::Path::new(&model_path);
    
    // Check if path exists (with or without .mpk extension)
    let actual_path = if path.exists() {
        path.to_path_buf()
    } else {
        let with_ext = PathBuf::from(format!("{}.mpk", model_path));
        if with_ext.exists() {
            path.to_path_buf()
        } else {
            return Err(format!("Model file not found: {}", model_path));
        }
    };

    // Store the path in state (model will be loaded on-demand)
    let mut model_path_state = state.model_path.write().await;
    *model_path_state = Some(actual_path.clone());

    Ok(ModelInfo {
        loaded: true,
        path: Some(model_path),
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
