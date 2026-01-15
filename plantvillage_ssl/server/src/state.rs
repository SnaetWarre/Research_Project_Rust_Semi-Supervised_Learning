//! Application state for the PlantVillage server
//!
//! Manages running processes, training state, and shared resources.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::process::Child;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// A line of output from a running process
#[derive(Clone, Debug, Serialize)]
pub struct OutputLine {
    pub timestamp: DateTime<Utc>,
    pub stream: String, // "stdout" or "stderr"
    pub content: String,
}

/// Status of a training run
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TrainingStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Information about a training run
#[derive(Clone, Debug, Serialize)]
pub struct TrainingRun {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: TrainingStatus,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub exit_code: Option<i32>,
    /// Recent output lines (kept in memory for quick access)
    #[serde(skip)]
    pub output_buffer: Vec<OutputLine>,
}

/// A managed process that can be monitored and controlled
pub struct ManagedProcess {
    pub run: TrainingRun,
    /// The child process handle
    pub child: Option<Child>,
    /// Broadcast sender for output streaming
    pub output_tx: broadcast::Sender<OutputLine>,
}

/// Server configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Base directory for the plantvillage_ssl project
    pub project_dir: PathBuf,
    /// Path to the plantvillage_ssl binary
    pub binary_path: PathBuf,
    /// Directory containing experiment results
    pub results_dir: PathBuf,
    /// Directory containing models
    pub models_dir: PathBuf,
    /// Directory containing data
    pub data_dir: PathBuf,
    /// Maximum output lines to keep in memory per process
    pub max_output_lines: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Try to detect paths - can be overridden
        let home = std::env::var("HOME").unwrap_or_else(|_| "/home/nvidia-user".to_string());
        let project_dir = PathBuf::from(&home).join("plantvillage_ssl");
        
        Self {
            binary_path: project_dir.join("target/release-jetson/plantvillage_ssl"),
            results_dir: project_dir.join("output"),
            models_dir: project_dir.join("output/models"),
            data_dir: project_dir.join("data"),
            project_dir,
            max_output_lines: 1000,
        }
    }
}

/// Shared application state
pub struct AppState {
    /// Server configuration
    pub config: ServerConfig,
    /// Currently running process (only one at a time for now)
    pub current_process: RwLock<Option<ManagedProcess>>,
    /// History of completed training runs
    pub run_history: RwLock<Vec<TrainingRun>>,
    /// Server start time
    pub started_at: Instant,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            current_process: RwLock::new(None),
            run_history: RwLock::new(Vec::new()),
            started_at: Instant::now(),
        }
    }

    /// Create a new training run
    pub fn create_run(&self, command: &str, args: Vec<String>) -> (TrainingRun, broadcast::Sender<OutputLine>) {
        let id = Uuid::new_v4().to_string();
        let (tx, _rx) = broadcast::channel(1024); // Buffer up to 1024 messages
        
        let run = TrainingRun {
            id,
            command: command.to_string(),
            args,
            status: TrainingStatus::Pending,
            started_at: Utc::now(),
            finished_at: None,
            exit_code: None,
            output_buffer: Vec::new(),
        };
        
        (run, tx)
    }

    /// Get uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.started_at.elapsed().as_secs()
    }
}

pub type SharedState = Arc<AppState>;
