use std::path::{Path, PathBuf};
use tauri::Manager;

/// Get the path to bundled assets
/// On desktop: returns the assets directory in development, or the resource directory in production
/// On mobile: returns the app's resource directory
pub fn get_bundled_assets_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource directory: {}", e))
        .map(|p| p.join("assets"))
}

/// Get the path to the bundled model
pub fn get_bundled_model_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let assets_path = get_bundled_assets_path(app)?;
    let model_path = assets_path.join("models").join("base_model.mpk");

    if !model_path.exists() {
        return Err(format!("Bundled model not found at: {:?}", model_path));
    }

    Ok(model_path)
}

/// Get the path to the bundled training data
pub fn get_bundled_data_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let assets_path = get_bundled_assets_path(app)?;
    let data_path = assets_path.join("data").join("farmer_demo");

    if !data_path.exists() {
        return Err(format!("Bundled data not found at: {:?}", data_path));
    }

    Ok(data_path)
}

/// List available bundled resources
pub fn list_bundled_resources(app: &tauri::AppHandle) -> Result<Vec<String>, String> {
    let assets_path = get_bundled_assets_path(app)?;
    let mut resources = Vec::new();

    // Check for model
    if let Ok(model_path) = get_bundled_model_path(app) {
        resources.push(format!("Model: {:?}", model_path));
    }

    // Check for data
    if let Ok(data_path) = get_bundled_data_path(app) {
        resources.push(format!("Data: {:?}", data_path));
    }

    Ok(resources)
}
