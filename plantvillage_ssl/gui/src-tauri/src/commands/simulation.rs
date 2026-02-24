//! Simulation Commands
//!
//! Commands for running the stream simulation with progress events.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::state::{AppState, SimulationStatus};

/// Simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationParams {
    pub data_dir: String,
    pub model_path: String,
    pub days: usize,
    pub images_per_day: usize,
    pub confidence_threshold: f64,
    pub retrain_threshold: usize,
    pub output_dir: String,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            data_dir: "data/plantvillage".to_string(),
            model_path: "output/models/plant_classifier".to_string(),
            days: 30,
            images_per_day: 50,
            confidence_threshold: 0.9,
            retrain_threshold: 200,
            output_dir: "output/simulation".to_string(),
        }
    }
}

/// Event payload for day updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayUpdate {
    pub day: usize,
    pub total_days: usize,
    pub images_processed: usize,
    pub pseudo_labels_accepted: usize,
    pub pseudo_labels_total: usize,
    pub pseudo_label_precision: f64,
    pub current_accuracy: f64,
}

/// Event payload for retrain events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainEvent {
    pub retrain_count: usize,
    pub pseudo_labels_used: usize,
    pub accuracy_before: f64,
    pub accuracy_after: f64,
}

/// Simulation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub initial_accuracy: f64,
    pub final_accuracy: f64,
    pub improvement: f64,
    pub days_simulated: usize,
    pub total_pseudo_labels: usize,
    pub pseudo_label_precision: f64,
    pub retrain_count: usize,
    pub accuracy_history: Vec<(usize, f64)>,
}

/// Get current simulation status
#[tauri::command]
pub async fn get_simulation_status(
    state: State<'_, Arc<AppState>>,
) -> Result<SimulationStatus, String> {
    let status = state.simulation_state.read().await;
    Ok(status.clone())
}

/// Start simulation
#[tauri::command]
pub async fn start_simulation(
    params: SimulationParams,
    app: AppHandle,
    state: State<'_, Arc<AppState>>,
) -> Result<SimulationResult, String> {
    use crate::backend::AdaptiveBackend;
    
    // Update state to running
    {
        let mut status = state.simulation_state.write().await;
        *status = SimulationStatus::Running {
            day: 0,
            total_days: params.days,
            pseudo_labels: 0,
            current_accuracy: 0.0,
        };
    }

    // Emit start event
    let _ = app.emit("simulation:started", &params);

    // Run simulation in blocking task
    let result = tokio::task::spawn_blocking(move || {
        run_simulation_inner::<AdaptiveBackend>(&params, &app)
    })
    .await
    .map_err(|e| format!("Simulation task failed: {:?}", e))?;

    // Update state based on result
    {
        let mut status = state.simulation_state.write().await;
        match &result {
            Ok(res) => {
                *status = SimulationStatus::Completed {
                    initial_accuracy: res.initial_accuracy,
                    final_accuracy: res.final_accuracy,
                    total_pseudo_labels: res.total_pseudo_labels,
                };
            }
            Err(e) => {
                *status = SimulationStatus::Error(e.clone());
            }
        }
    }

    result
}

/// Inner simulation function
fn run_simulation_inner<B>(
    params: &SimulationParams,
    app: &AppHandle,
) -> Result<SimulationResult, String>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    use plantvillage_ssl::training::{run_simulation, SimulationConfig};
    
    let config = SimulationConfig {
        data_dir: params.data_dir.clone(),
        model_path: params.model_path.clone(),
        days: params.days,
        images_per_day: params.images_per_day,
        confidence_threshold: params.confidence_threshold,
        retrain_threshold: params.retrain_threshold,
        labeled_ratio: 0.2,   // 20% labeled, 60% for SSL stream (plus 10% val/test)
        output_dir: params.output_dir.clone(),
        seed: 42,
        batch_size: 32,
        learning_rate: 0.0001,
        retrain_epochs: 5,
    };

    // Run the actual simulation from plantvillage_ssl
    let results = run_simulation::<B>(config)
        .map_err(|e| format!("Simulation failed: {:?}", e))?;

    let result = SimulationResult {
        initial_accuracy: results.initial_accuracy,
        final_accuracy: results.final_accuracy,
        improvement: results.improvement(),
        days_simulated: results.days_simulated,
        total_pseudo_labels: results.total_pseudo_labels,
        pseudo_label_precision: results.pseudo_label_precision * 100.0,
        retrain_count: results.retraining_count,
        accuracy_history: results.accuracy_history,
    };

    // Emit completion
    let _ = app.emit("simulation:complete", &result);

    Ok(result)
}

/// Stop simulation
#[tauri::command]
pub async fn stop_simulation(
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut status = state.simulation_state.write().await;
    if matches!(*status, SimulationStatus::Running { .. }) {
        *status = SimulationStatus::Idle;
    }
    Ok(())
}
