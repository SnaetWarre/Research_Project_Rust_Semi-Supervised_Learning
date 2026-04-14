//! Experiments endpoints - list and retrieve experiment results

use std::path::PathBuf;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use tokio::fs;
use tracing::error;

use crate::state::SharedState;

/// Summary of an experiment
#[derive(Debug, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: Option<String>,
    pub status: String,
    /// Final accuracy if available
    pub final_accuracy: Option<f64>,
    /// Final loss if available
    pub final_loss: Option<f64>,
}

/// Detailed experiment results
#[derive(Debug, Serialize, Deserialize)]
pub struct ExperimentDetails {
    #[serde(flatten)]
    pub summary: ExperimentSummary,
    /// Training metrics over epochs (if available)
    pub metrics: Option<serde_json::Value>,
    /// Configuration used
    pub config: Option<serde_json::Value>,
    /// Available files in this experiment
    pub files: Vec<String>,
}

/// GET /experiments - List all experiments
pub async fn list_experiments(State(state): State<SharedState>) -> Result<Json<Vec<ExperimentSummary>>, (StatusCode, String)> {
    let results_dir = &state.config.results_dir;
    
    if !results_dir.exists() {
        return Ok(Json(Vec::new()));
    }
    
    let mut experiments = Vec::new();
    let mut entries = fs::read_dir(results_dir).await.map_err(|e| {
        error!("Failed to read results directory: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "Failed to list experiments".to_string())
    })?;
    
    while let Some(entry) = entries.next_entry().await.map_err(|e| {
        error!("Failed to read entry: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "Failed to list experiments".to_string())
    })? {
        let path = entry.path();
        
        // Skip non-directories and special directories
        if !path.is_dir() {
            continue;
        }
        
        let name = entry.file_name().to_string_lossy().to_string();
        
        // Try to get metadata
        let metadata = entry.metadata().await.ok();
        let created_at = metadata.as_ref().and_then(|m| {
            m.created().ok().map(|t| {
                chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339()
            })
        });
        
        // Try to read metrics from common locations
        let (status, final_accuracy, final_loss) = read_experiment_metrics(&path).await;
        
        experiments.push(ExperimentSummary {
            id: name.clone(),
            name: name.clone(),
            path: format!("output/{}", name),
            created_at,
            status,
            final_accuracy,
            final_loss,
        });
    }
    
    // Sort by creation date (newest first)
    experiments.sort_by(|a, b| {
        b.created_at.cmp(&a.created_at)
    });
    
    Ok(Json(experiments))
}

/// GET /experiments/:id - Get details for a specific experiment
pub async fn get_experiment(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<ExperimentDetails>, (StatusCode, String)> {
    let experiment_path = state.config.results_dir.join(&id);
    
    if !experiment_path.exists() || !experiment_path.is_dir() {
        return Err((StatusCode::NOT_FOUND, format!("Experiment not found: {}", id)));
    }
    
    let metadata = fs::metadata(&experiment_path).await.ok();
    let created_at = metadata.as_ref().and_then(|m| {
        m.created().ok().map(|t| {
            chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339()
        })
    });
    
    let (status, final_accuracy, final_loss) = read_experiment_metrics(&experiment_path).await;
    
    // List files in the experiment directory
    let files = list_experiment_files(&experiment_path).await;
    
    // Try to read config
    let config = read_json_file(&experiment_path.join("config.json")).await;
    
    // Try to read metrics - try multiple possible file names
    let metrics = if let Some(m) = read_json_file(&experiment_path.join("metrics.json")).await {
        Some(m)
    } else {
        read_json_file(&experiment_path.join("training_metrics.json")).await
    };
    
    Ok(Json(ExperimentDetails {
        summary: ExperimentSummary {
            id: id.clone(),
            name: id.clone(),
            path: format!("output/{}", id),
            created_at,
            status,
            final_accuracy,
            final_loss,
        },
        metrics,
        config,
        files,
    }))
}

/// Read experiment metrics from common file locations
async fn read_experiment_metrics(path: &PathBuf) -> (String, Option<f64>, Option<f64>) {
    // Try to read from metrics.json
    if let Some(metrics) = read_json_file(&path.join("metrics.json")).await {
        let accuracy = metrics.get("final_accuracy")
            .or_else(|| metrics.get("accuracy"))
            .or_else(|| metrics.get("test_accuracy"))
            .and_then(|v| v.as_f64());
        
        let loss = metrics.get("final_loss")
            .or_else(|| metrics.get("loss"))
            .or_else(|| metrics.get("test_loss"))
            .and_then(|v| v.as_f64());
        
        let status = if accuracy.is_some() { "completed" } else { "unknown" };
        return (status.to_string(), accuracy, loss);
    }
    
    // Try to read from training_log.json or similar
    if let Some(log) = read_json_file(&path.join("training_log.json")).await {
        if let Some(epochs) = log.as_array() {
            if let Some(last) = epochs.last() {
                let accuracy = last.get("accuracy").and_then(|v| v.as_f64());
                let loss = last.get("loss").and_then(|v| v.as_f64());
                return ("completed".to_string(), accuracy, loss);
            }
        }
    }
    
    // Check if there's a model file (indicates completion)
    let has_model = path.join("model.mpk").exists() 
        || path.join("best_model.mpk").exists()
        || path.join("final_model.mpk").exists();
    
    let status = if has_model { "completed" } else { "unknown" };
    (status.to_string(), None, None)
}

/// Read a JSON file and return its contents
async fn read_json_file(path: &PathBuf) -> Option<serde_json::Value> {
    let content = fs::read_to_string(path).await.ok()?;
    serde_json::from_str(&content).ok()
}

/// List all files in an experiment directory
async fn list_experiment_files(path: &PathBuf) -> Vec<String> {
    let mut files = Vec::new();
    
    if let Ok(mut entries) = fs::read_dir(path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            files.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    
    files.sort();
    files
}
