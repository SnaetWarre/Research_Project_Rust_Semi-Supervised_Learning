//! Training endpoints with process management and SSE streaming

use std::convert::Infallible;
use std::process::Stdio;

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use chrono::Utc;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio_stream::StreamExt;
use tracing::{error, info, warn};

use crate::state::{ManagedProcess, OutputLine, SharedState, TrainingRun, TrainingStatus};

/// Request to start training
#[derive(Debug, Deserialize)]
pub struct TrainRequest {
    /// Subcommand to run (train, simulate, benchmark, etc.)
    #[serde(default = "default_subcommand")]
    pub subcommand: String,
    
    /// Data directory
    #[serde(default = "default_data_dir")]
    pub data_dir: Option<String>,
    
    /// Number of epochs
    pub epochs: Option<usize>,
    
    /// Batch size
    pub batch_size: Option<usize>,
    
    /// Learning rate
    pub learning_rate: Option<f64>,
    
    /// Labeled ratio for SSL
    pub labeled_ratio: Option<f64>,
    
    /// Confidence threshold for pseudo-labeling
    pub confidence_threshold: Option<f64>,
    
    /// Output directory
    pub output_dir: Option<String>,
    
    /// Random seed
    pub seed: Option<u64>,
    
    /// Quick mode (500 samples)
    #[serde(default)]
    pub quick: bool,
    
    /// Class-weighted loss
    #[serde(default)]
    pub class_weighted: bool,
    
    /// Model path (for simulate, infer, benchmark)
    pub model: Option<String>,
    
    /// Additional raw arguments to pass to the CLI
    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_subcommand() -> String {
    "train".to_string()
}

fn default_data_dir() -> Option<String> {
    Some("data/plantvillage/balanced".to_string())
}

/// Response after starting training
#[derive(Serialize)]
pub struct TrainStartResponse {
    pub id: String,
    pub status: TrainingStatus,
    pub message: String,
}

/// Training status response
#[derive(Serialize)]
pub struct TrainStatusResponse {
    pub running: bool,
    pub current_run: Option<TrainingRun>,
    pub recent_output: Vec<OutputLine>,
}

/// Build CLI arguments from the request
fn build_args(req: &TrainRequest, _config: &crate::state::ServerConfig) -> Vec<String> {
    let mut args = vec![req.subcommand.clone()];
    
    // Add data directory
    if let Some(ref data_dir) = req.data_dir {
        args.push("--data-dir".to_string());
        args.push(data_dir.clone());
    }
    
    // Add common training arguments
    if let Some(epochs) = req.epochs {
        args.push("--epochs".to_string());
        args.push(epochs.to_string());
    }
    
    if let Some(batch_size) = req.batch_size {
        args.push("--batch-size".to_string());
        args.push(batch_size.to_string());
    }
    
    if let Some(lr) = req.learning_rate {
        args.push("--learning-rate".to_string());
        args.push(lr.to_string());
    }
    
    if let Some(labeled_ratio) = req.labeled_ratio {
        args.push("--labeled-ratio".to_string());
        args.push(labeled_ratio.to_string());
    }
    
    if let Some(confidence) = req.confidence_threshold {
        args.push("--confidence-threshold".to_string());
        args.push(confidence.to_string());
    }
    
    if let Some(ref output_dir) = req.output_dir {
        args.push("--output-dir".to_string());
        args.push(output_dir.clone());
    }
    
    if let Some(seed) = req.seed {
        args.push("--seed".to_string());
        args.push(seed.to_string());
    }
    
    if req.quick {
        args.push("--quick".to_string());
    }
    
    if req.class_weighted {
        args.push("--class-weighted".to_string());
    }
    
    if let Some(ref model) = req.model {
        args.push("--model".to_string());
        args.push(model.clone());
    }
    
    // Add any extra arguments
    args.extend(req.extra_args.iter().cloned());
    
    args
}

/// POST /train - Start a training run
pub async fn start_training(
    State(state): State<SharedState>,
    Json(req): Json<TrainRequest>,
) -> Result<Json<TrainStartResponse>, (StatusCode, String)> {
    // Check if a process is already running
    {
        let current = state.current_process.read().await;
        if let Some(ref proc) = *current {
            if proc.run.status == TrainingStatus::Running {
                return Err((
                    StatusCode::CONFLICT,
                    format!("Training already in progress: {}", proc.run.id),
                ));
            }
        }
    }

    let args = build_args(&req, &state.config);
    let (mut run, output_tx) = state.create_run(&req.subcommand, args.clone());
    
    info!("Starting training: {} {:?}", state.config.binary_path.display(), args);
    
    // Spawn the process
    let mut child = match Command::new(&state.config.binary_path)
        .args(&args)
        .current_dir(&state.config.project_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            error!("Failed to spawn process: {}", e);
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to start training: {}", e),
            ));
        }
    };
    
    run.status = TrainingStatus::Running;
    let run_id = run.id.clone();
    
    // Take ownership of stdout/stderr
    let stdout = child.stdout.take().expect("stdout should be captured");
    let stderr = child.stderr.take().expect("stderr should be captured");
    
    // Create managed process
    let managed = ManagedProcess {
        run: run.clone(),
        child: Some(child),
        output_tx: output_tx.clone(),
    };
    
    // Store the managed process
    {
        let mut current = state.current_process.write().await;
        *current = Some(managed);
    }
    
    // Spawn task to read stdout
    let stdout_tx = output_tx.clone();
    tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        
        while let Ok(Some(line)) = lines.next_line().await {
            let output_line = OutputLine {
                timestamp: Utc::now(),
                stream: "stdout".to_string(),
                content: line,
            };
            let _ = stdout_tx.send(output_line);
        }
    });
    
    // Spawn task to read stderr
    let stderr_tx = output_tx.clone();
    tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        
        while let Ok(Some(line)) = lines.next_line().await {
            let output_line = OutputLine {
                timestamp: Utc::now(),
                stream: "stderr".to_string(),
                content: line,
            };
            let _ = stderr_tx.send(output_line);
        }
    });
    
    // Spawn task to wait for process completion
    let state_clone = state.clone();
    let run_id_clone = run_id.clone();
    tokio::spawn(async move {
        // Get the child process
        let mut child = {
            let mut current = state_clone.current_process.write().await;
            current.as_mut().and_then(|p| p.child.take())
        };
        
        if let Some(ref mut child) = child {
            let status = child.wait().await;
            
            // Update the run status
            let mut current = state_clone.current_process.write().await;
            if let Some(ref mut proc) = *current {
                if proc.run.id == run_id_clone {
                    proc.run.finished_at = Some(Utc::now());
                    match status {
                        Ok(exit_status) => {
                            proc.run.exit_code = exit_status.code();
                            proc.run.status = if exit_status.success() {
                                TrainingStatus::Completed
                            } else {
                                TrainingStatus::Failed
                            };
                            info!("Training {} completed with status: {:?}", run_id_clone, exit_status);
                        }
                        Err(e) => {
                            proc.run.status = TrainingStatus::Failed;
                            error!("Training {} failed: {}", run_id_clone, e);
                        }
                    }
                    
                    // Move to history
                    let run = proc.run.clone();
                    drop(current);
                    state_clone.run_history.write().await.push(run);
                }
            }
        }
    });
    
    Ok(Json(TrainStartResponse {
        id: run_id,
        status: TrainingStatus::Running,
        message: "Training started".to_string(),
    }))
}

/// GET /train/status - Get current training status
pub async fn get_training_status(State(state): State<SharedState>) -> Json<TrainStatusResponse> {
    let current = state.current_process.read().await;
    
    match &*current {
        Some(proc) => {
            Json(TrainStatusResponse {
                running: proc.run.status == TrainingStatus::Running,
                current_run: Some(proc.run.clone()),
                recent_output: proc.run.output_buffer.clone(),
            })
        }
        None => {
            Json(TrainStatusResponse {
                running: false,
                current_run: None,
                recent_output: Vec::new(),
            })
        }
    }
}

/// GET /train/stream - SSE stream of training output
pub async fn stream_training_output(
    State(state): State<SharedState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    let current = state.current_process.read().await;
    
    let receiver = match &*current {
        Some(proc) => proc.output_tx.subscribe(),
        None => {
            return Err((
                StatusCode::NOT_FOUND,
                "No training in progress".to_string(),
            ));
        }
    };
    
    drop(current); // Release the lock
    
    let stream = tokio_stream::wrappers::BroadcastStream::new(receiver)
        .filter_map(|result| {
            match result {
                Ok(line) => {
                    let data = serde_json::to_string(&line).unwrap_or_default();
                    Some(Ok(Event::default().data(data)))
                }
                Err(_) => None, // Skip lagged messages
            }
        });
    
    Ok(Sse::new(stream))
}

/// POST /train/stop - Stop the current training
pub async fn stop_training(State(state): State<SharedState>) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut current = state.current_process.write().await;
    
    match &mut *current {
        Some(proc) if proc.run.status == TrainingStatus::Running => {
            // Kill the process if we still have a handle
            if let Some(ref mut child) = proc.child {
                if let Err(e) = child.kill().await {
                    warn!("Failed to kill process: {}", e);
                }
            }
            
            proc.run.status = TrainingStatus::Cancelled;
            proc.run.finished_at = Some(Utc::now());
            
            let run = proc.run.clone();
            drop(current);
            state.run_history.write().await.push(run);
            
            Ok(Json(serde_json::json!({
                "success": true,
                "message": "Training cancelled"
            })))
        }
        Some(_) => {
            Err((
                StatusCode::BAD_REQUEST,
                "No training currently running".to_string(),
            ))
        }
        None => {
            Err((
                StatusCode::NOT_FOUND,
                "No training in progress".to_string(),
            ))
        }
    }
}

/// GET /train/history - Get training history
pub async fn get_training_history(State(state): State<SharedState>) -> Json<Vec<TrainingRun>> {
    let history = state.run_history.read().await;
    Json(history.clone())
}
