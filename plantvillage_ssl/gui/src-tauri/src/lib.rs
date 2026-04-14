//! PlantVillage SSL Dashboard - Tauri Backend
//!
//! This is the Tauri backend for the PlantVillage SSL Dashboard,
//! providing commands and state management for the GUI.

mod commands;
mod state;
mod device;
mod backend;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::Manager;
use state::AppState;
use commands::incremental::IncrementalProgress;
use commands::demo::DemoSessionGlobal;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize tracing for console logging
    tracing_subscriber::fmt()
        .with_env_filter("plantvillage_gui=info,demo=info")
        .with_target(false)
        .init();
    
    tracing::info!("=== PlantVillage SSL Dashboard Starting ===");

    // Initialize application state
    let app_state = Arc::new(AppState::new());

    // Initialize incremental learning progress state
    let incremental_progress_state: Arc<Mutex<Option<IncrementalProgress>>> = Arc::new(Mutex::new(None));

    // Initialize demo session state
    let demo_session_state: DemoSessionGlobal = Arc::new(Mutex::new(None));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .manage(incremental_progress_state)
        .manage(demo_session_state)
        .setup(|app| {
            // Always open devtools for debugging
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.open_devtools();
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event {
                let _ = window.destroy();
                std::process::exit(0);
            }
        })
        .invoke_handler(tauri::generate_handler![
            // System info commands
            get_device_info,
            get_backend_info,
            // Dataset commands
            commands::get_dataset_stats,
            commands::load_model,
            commands::get_model_status,
            commands::get_cached_dataset_info,
            // Inference commands
            commands::run_inference,
            commands::run_inference_bytes,
            // Training commands
            commands::get_training_status,
            commands::start_training,
            commands::stop_training,
            // Pseudo-labeling commands
            commands::run_pseudo_label_demo,
            commands::get_sample_images,
            // Simulation commands
            commands::get_simulation_status,
            commands::start_simulation,
            commands::stop_simulation,
            // Benchmark commands
            commands::run_benchmark,
            commands::load_benchmark_results,
            // Incremental learning commands
            commands::train_incremental,
            commands::get_incremental_progress,
            commands::stop_incremental_training,
            commands::run_experiment,
            commands::get_incremental_methods,
            // Diagnostics commands
            commands::run_model_diagnostics,
            // Experiment results commands
            commands::load_label_efficiency_results,
            commands::load_class_scaling_results,
            commands::load_ssl_incremental_results,
            commands::load_new_class_position_results,
            commands::load_all_experiment_results,
            commands::load_experiment_conclusions,
            commands::get_available_experiments,
            // Demo commands
            commands::init_demo_session,
            commands::advance_demo_day,
            commands::get_demo_session_state,
            commands::reset_demo_session,
            commands::process_farmer_images,
            commands::manual_retrain_demo,
            // Mobile SSL commands
            commands::start_ssl_retraining,
            commands::get_ssl_retraining_config,
            // Dataset bundling commands
            commands::create_dataset_bundle,
            commands::load_bundle_metadata,
            // Exit app command
            exit_app,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
fn exit_app() {
    std::process::exit(0);
}

/// Get device information (Desktop vs Mobile, capabilities)
#[tauri::command]
fn get_device_info() -> Result<device::DeviceCapabilities, String> {
    let device_type = device::DeviceType::detect();
    Ok(device_type.capabilities())
}

/// Get backend information (CUDA vs NdArray)
#[tauri::command]
fn get_backend_info() -> Result<serde_json::Value, String> {
    Ok(serde_json::json!({
        "backend": backend::backend_name(),
        "has_gpu": backend::has_gpu(),
        "device": format!("{:?}", backend::default_device()),
    }))
}
