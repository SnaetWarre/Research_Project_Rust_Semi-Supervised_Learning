//! PlantVillage SSL Dashboard - Tauri Backend
//!
//! This is the Tauri backend for the PlantVillage SSL Dashboard,
//! providing commands and state management for the GUI.

mod commands;
mod state;

use std::sync::Arc;
use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize application state
    let app_state = Arc::new(AppState::new());

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
