//! File serving endpoints

use std::path::PathBuf;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use serde::Serialize;
use tokio::fs;
use tokio_util::io::ReaderStream;
use tracing::error;

use crate::state::SharedState;

/// Information about a file
#[derive(Serialize)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub size: u64,
    pub is_dir: bool,
    pub modified: Option<String>,
}

/// Response for listing files
#[derive(Serialize)]
pub struct FileListResponse {
    pub path: String,
    pub files: Vec<FileInfo>,
}

/// GET /models - List available model files
pub async fn list_models(State(state): State<SharedState>) -> Result<Json<FileListResponse>, (StatusCode, String)> {
    list_directory(&state.config.models_dir, "models").await
}

/// GET /files/* - Serve a file from allowed directories
pub async fn serve_file(
    State(state): State<SharedState>,
    Path(path): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    // Determine which base directory this file is in
    let (base_dir, relative_path) = if path.starts_with("models/") {
        (&state.config.models_dir, path.strip_prefix("models/").unwrap())
    } else if path.starts_with("output/") || path.starts_with("results/") {
        (&state.config.results_dir, path.strip_prefix("output/").or_else(|| path.strip_prefix("results/")).unwrap())
    } else if path.starts_with("data/") {
        (&state.config.data_dir, path.strip_prefix("data/").unwrap())
    } else {
        // Default to project directory for other paths
        (&state.config.project_dir, path.as_str())
    };
    
    let file_path = base_dir.join(relative_path);
    
    // Security: Ensure the path doesn't escape the base directory
    let canonical_base = base_dir.canonicalize().map_err(|e| {
        error!("Failed to canonicalize base dir: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "Server error".to_string())
    })?;
    
    let canonical_file = file_path.canonicalize().map_err(|_| {
        (StatusCode::NOT_FOUND, "File not found".to_string())
    })?;
    
    if !canonical_file.starts_with(&canonical_base) && !canonical_file.starts_with(&state.config.project_dir) {
        return Err((StatusCode::FORBIDDEN, "Access denied".to_string()));
    }
    
    // Check if it's a directory
    let metadata = fs::metadata(&canonical_file).await.map_err(|_| {
        (StatusCode::NOT_FOUND, "File not found".to_string())
    })?;
    
    if metadata.is_dir() {
        // Return directory listing as JSON
        let listing = list_directory(&canonical_file, &path).await?;
        // Extract the inner FileListResponse from Json wrapper
        let response = serde_json::to_string(&listing.0).unwrap();
        return Err((StatusCode::OK, response));
    }
    
    // Open the file
    let file = fs::File::open(&canonical_file).await.map_err(|e| {
        error!("Failed to open file: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "Failed to open file".to_string())
    })?;
    
    // Get file name for Content-Disposition
    let filename = canonical_file
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("download");
    
    // Determine content type
    let content_type = match canonical_file.extension().and_then(|e| e.to_str()) {
        Some("json") => "application/json",
        Some("csv") => "text/csv",
        Some("txt") | Some("log") => "text/plain",
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("mpk") => "application/octet-stream", // Burn model format
        Some("bin") => "application/octet-stream",
        _ => "application/octet-stream",
    };
    
    // Stream the file
    let stream = ReaderStream::new(file);
    let body = Body::from_stream(stream);
    
    // Build content disposition header
    let content_disposition = format!("attachment; filename=\"{}\"", filename);
    
    Ok((
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            (header::CONTENT_DISPOSITION, content_disposition),
        ],
        body,
    ))
}

/// List files in a directory
async fn list_directory(dir: &PathBuf, display_path: &str) -> Result<Json<FileListResponse>, (StatusCode, String)> {
    let mut entries = fs::read_dir(dir).await.map_err(|e| {
        error!("Failed to read directory {:?}: {}", dir, e);
        (StatusCode::NOT_FOUND, format!("Directory not found: {}", display_path))
    })?;
    
    let mut files = Vec::new();
    
    while let Some(entry) = entries.next_entry().await.map_err(|e| {
        error!("Failed to read entry: {}", e);
        (StatusCode::INTERNAL_SERVER_ERROR, "Failed to list directory".to_string())
    })? {
        let metadata = entry.metadata().await.ok();
        let name = entry.file_name().to_string_lossy().to_string();
        
        let (size, is_dir, modified) = match metadata {
            Some(m) => {
                let modified = m.modified().ok().map(|t| {
                    chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339()
                });
                (m.len(), m.is_dir(), modified)
            }
            None => (0, false, None),
        };
        
        files.push(FileInfo {
            path: format!("{}/{}", display_path, name),
            name,
            size,
            is_dir,
            modified,
        });
    }
    
    // Sort: directories first, then by name
    files.sort_by(|a, b| {
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.cmp(&b.name),
        }
    });
    
    Ok(Json(FileListResponse {
        path: display_path.to_string(),
        files,
    }))
}

/// GET /output/* or /results/* - List or serve output files
pub async fn list_output(State(state): State<SharedState>) -> Result<Json<FileListResponse>, (StatusCode, String)> {
    list_directory(&state.config.results_dir, "output").await
}
