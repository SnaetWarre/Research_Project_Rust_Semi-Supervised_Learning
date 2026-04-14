//! Remote Training Commands
//!
//! Commands for running training on the remote Jetson device.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::client::{TrainRequest, TrainStartResponse, TrainStatusResponse};
use crate::state::{AppState, ConnectionMode};

/// Remote training parameters (matches server API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteTrainingParams {
    pub data_dir: Option<String>,
    pub epochs: Option<usize>,
    pub batch_size: Option<usize>,
    pub learning_rate: Option<f64>,
    pub labeled_ratio: Option<f64>,
    pub confidence_threshold: Option<f64>,
    pub output_dir: Option<String>,
    pub seed: Option<u64>,
    #[serde(default)]
    pub quick: bool,
    #[serde(default)]
    pub class_weighted: bool,
}

impl Default for RemoteTrainingParams {
    fn default() -> Self {
        Self {
            data_dir: Some("data/plantvillage/balanced".to_string()),
            epochs: Some(50),
            batch_size: Some(32),
            learning_rate: Some(0.0001),
            labeled_ratio: Some(0.2),
            confidence_threshold: Some(0.9),
            output_dir: Some("output/models".to_string()),
            seed: Some(42),
            quick: false,
            class_weighted: false,
        }
    }
}

/// Start training on Jetson
#[tauri::command]
pub async fn start_remote_training(
    params: RemoteTrainingParams,
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<TrainStartResponse, String> {
    // Check we're in remote mode
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode. Connect to Jetson first.".to_string());
    }
    
    let client = state.jetson_client.read().await;
    
    // Convert to API request
    let req = TrainRequest {
        subcommand: "train".to_string(),
        data_dir: params.data_dir,
        epochs: params.epochs,
        batch_size: params.batch_size,
        learning_rate: params.learning_rate,
        labeled_ratio: params.labeled_ratio,
        confidence_threshold: params.confidence_threshold,
        output_dir: params.output_dir,
        seed: params.seed,
        quick: params.quick,
        class_weighted: params.class_weighted,
        model: None,
        extra_args: Vec::new(),
    };
    
    let response = client.start_training(req).await?;
    
    // Emit event
    let _ = app.emit("remote_training:started", &response);
    
    Ok(response)
}

/// Get remote training status
#[tauri::command]
pub async fn get_remote_training_status(
    state: State<'_, Arc<AppState>>,
) -> Result<TrainStatusResponse, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    client.training_status().await
}

/// Stop remote training
#[tauri::command]
pub async fn stop_remote_training(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<serde_json::Value, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    let result = client.stop_training().await?;
    
    let _ = app.emit("remote_training:stopped", &result);
    
    Ok(result)
}

/// Get SSE stream URL for training output
#[tauri::command]
pub async fn get_training_stream_url(
    state: State<'_, Arc<AppState>>,
) -> Result<String, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    Ok(client.stream_url())
}

/// List remote experiments
#[tauri::command]
pub async fn list_remote_experiments(
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<crate::client::ExperimentSummary>, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    client.list_experiments().await
}

/// List remote models
#[tauri::command]
pub async fn list_remote_models(
    state: State<'_, Arc<AppState>>,
) -> Result<crate::client::FileListResponse, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    client.list_models().await
}

/// Download a file from Jetson
#[tauri::command]
pub async fn download_remote_file(
    path: String,
    state: State<'_, Arc<AppState>>,
) -> Result<Vec<u8>, String> {
    let mode = *state.connection_mode.read().await;
    if mode != ConnectionMode::Remote {
        return Err("Not in remote mode".to_string());
    }
    
    let client = state.jetson_client.read().await;
    client.download_file(&path).await
}
