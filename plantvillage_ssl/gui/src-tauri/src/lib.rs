//! PlantVillage SSL Dashboard - Tauri Backend
//!
//! This is the Tauri backend for the PlantVillage SSL Dashboard,
//! providing commands and state management for the GUI.

mod commands;
mod state;

use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::Manager;
use state::AppState;
use commands::incremental::IncrementalProgress;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize application state
    let app_state = Arc::new(AppState::new());

    // Initialize incremental learning progress state
    let incremental_progress_state: Arc<Mutex<Option<IncrementalProgress>>> = Arc::new(Mutex::new(None));

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .manage(incremental_progress_state)
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools();
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
