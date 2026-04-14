//! Jetson HTTP Client
//!
//! Client for communicating with the PlantVillage server running on Jetson.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default Jetson server URL
pub const DEFAULT_JETSON_URL: &str = "http://10.42.0.10:8080";

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
    pub version: String,
}

/// Training request to send to the server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainRequest {
    #[serde(default = "default_subcommand")]
    pub subcommand: String,
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
    pub model: Option<String>,
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_subcommand() -> String {
    "train".to_string()
}

/// Training start response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStartResponse {
    pub id: String,
    pub status: String,
    pub message: String,
}

/// Training status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RemoteTrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Training run info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRun {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: RemoteTrainingStatus,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub exit_code: Option<i32>,
}

/// Training status response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainStatusResponse {
    pub running: bool,
    pub current_run: Option<TrainingRun>,
    pub recent_output: Vec<OutputLine>,
}

/// Output line from training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputLine {
    pub timestamp: String,
    pub stream: String,
    pub content: String,
}

/// File info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub is_dir: bool,
    pub modified: Option<String>,
}

/// File list response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileListResponse {
    pub path: String,
    pub files: Vec<FileInfo>,
}

/// Experiment summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub id: String,
    pub name: String,
    pub path: String,
    pub created_at: Option<String>,
    pub status: String,
    pub final_accuracy: Option<f64>,
    pub final_loss: Option<f64>,
}

/// Jetson API client
#[derive(Debug, Clone)]
pub struct JetsonClient {
    base_url: String,
    client: reqwest::Client,
}

impl JetsonClient {
    /// Create a new client with default URL
    pub fn new() -> Self {
        Self::with_url(DEFAULT_JETSON_URL)
    }

    /// Create a new client with custom URL
    pub fn with_url(url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url: url.trim_end_matches('/').to_string(),
            client,
        }
    }

    /// Set the base URL
    pub fn set_url(&mut self, url: &str) {
        self.base_url = url.trim_end_matches('/').to_string();
    }

    /// Get the base URL
    pub fn url(&self) -> &str {
        &self.base_url
    }

    /// Check if the server is healthy
    pub async fn health(&self) -> Result<HealthResponse, String> {
        let url = format!("{}/health", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Connection failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// Check if the server is reachable (quick ping)
    pub async fn is_reachable(&self) -> bool {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap_or_else(|_| self.client.clone());
        
        let url = format!("{}/health", self.base_url);
        client.get(&url).send().await.is_ok()
    }

    /// Start training
    pub async fn start_training(&self, req: TrainRequest) -> Result<TrainStartResponse, String> {
        let url = format!("{}/train", self.base_url);
        self.client
            .post(&url)
            .json(&req)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// Get training status
    pub async fn training_status(&self) -> Result<TrainStatusResponse, String> {
        let url = format!("{}/train/status", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// Stop training
    pub async fn stop_training(&self) -> Result<serde_json::Value, String> {
        let url = format!("{}/train/stop", self.base_url);
        self.client
            .post(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// Get training history
    pub async fn training_history(&self) -> Result<Vec<TrainingRun>, String> {
        let url = format!("{}/train/history", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// List experiments
    pub async fn list_experiments(&self) -> Result<Vec<ExperimentSummary>, String> {
        let url = format!("{}/experiments", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// List models
    pub async fn list_models(&self) -> Result<FileListResponse, String> {
        let url = format!("{}/models", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// List output files
    pub async fn list_output(&self) -> Result<FileListResponse, String> {
        let url = format!("{}/output", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .json()
            .await
            .map_err(|e| format!("Invalid response: {}", e))
    }

    /// Get the SSE stream URL for training output
    pub fn stream_url(&self) -> String {
        format!("{}/train/stream", self.base_url)
    }

    /// Download a file from the server
    pub async fn download_file(&self, path: &str) -> Result<Vec<u8>, String> {
        let url = format!("{}/files/{}", self.base_url, path);
        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Request failed: {}", e))?
            .bytes()
            .await
            .map(|b| b.to_vec())
            .map_err(|e| format!("Failed to read response: {}", e))
    }
}

impl Default for JetsonClient {
    fn default() -> Self {
        Self::new()
    }
}
