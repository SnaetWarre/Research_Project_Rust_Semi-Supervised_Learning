//! Connection Commands
//!
//! Commands for managing the connection to Jetson and switching modes.

use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};

use crate::client::{JetsonClient, DEFAULT_JETSON_URL};
use crate::state::{AppState, ConnectionMode, ConnectionStatus};

/// Connection info response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub mode: ConnectionMode,
    pub status: ConnectionStatus,
    pub jetson_url: String,
}

/// Get current connection info
#[tauri::command]
pub async fn get_connection_info(
    state: State<'_, Arc<AppState>>,
) -> Result<ConnectionInfo, String> {
    let mode = *state.connection_mode.read().await;
    let status = state.connection_status.read().await.clone();
    let client = state.jetson_client.read().await;
    
    Ok(ConnectionInfo {
        mode,
        status,
        jetson_url: client.url().to_string(),
    })
}

/// Set connection mode (local or remote)
#[tauri::command]
pub async fn set_connection_mode(
    mode: ConnectionMode,
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<(), String> {
    {
        let mut current_mode = state.connection_mode.write().await;
        *current_mode = mode;
    }
    
    // If switching to remote, try to connect
    if mode == ConnectionMode::Remote {
        let client = state.jetson_client.read().await;
        match client.health().await {
            Ok(health) => {
                let mut status = state.connection_status.write().await;
                *status = ConnectionStatus::Connected {
                    url: client.url().to_string(),
                    uptime_seconds: health.uptime_seconds,
                    version: health.version,
                };
            }
            Err(e) => {
                let mut status = state.connection_status.write().await;
                *status = ConnectionStatus::Error(e);
            }
        }
    } else {
        let mut status = state.connection_status.write().await;
        *status = ConnectionStatus::Disconnected;
    }
    
    // Emit connection change event
    let info = get_connection_info_inner(&state).await;
    let _ = app.emit("connection:changed", &info);
    
    Ok(())
}

/// Set Jetson URL
#[tauri::command]
pub async fn set_jetson_url(
    url: String,
    state: State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut client = state.jetson_client.write().await;
    client.set_url(&url);
    Ok(())
}

/// Test connection to Jetson
#[tauri::command]
pub async fn test_jetson_connection(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<ConnectionStatus, String> {
    let client = state.jetson_client.read().await;
    
    let new_status = match client.health().await {
        Ok(health) => ConnectionStatus::Connected {
            url: client.url().to_string(),
            uptime_seconds: health.uptime_seconds,
            version: health.version,
        },
        Err(e) => ConnectionStatus::Error(e),
    };
    
    // Update status
    {
        let mut status = state.connection_status.write().await;
        *status = new_status.clone();
    }
    
    // Emit status change
    let _ = app.emit("connection:status", &new_status);
    
    Ok(new_status)
}

/// Connect to Jetson (set remote mode and test connection)
#[tauri::command]
pub async fn connect_to_jetson(
    url: Option<String>,
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<ConnectionStatus, String> {
    // Set URL if provided
    if let Some(ref url) = url {
        let mut client = state.jetson_client.write().await;
        client.set_url(url);
    }
    
    // Set mode to remote
    {
        let mut mode = state.connection_mode.write().await;
        *mode = ConnectionMode::Remote;
    }
    
    // Test connection
    let client = state.jetson_client.read().await;
    let new_status = match client.health().await {
        Ok(health) => ConnectionStatus::Connected {
            url: client.url().to_string(),
            uptime_seconds: health.uptime_seconds,
            version: health.version,
        },
        Err(e) => ConnectionStatus::Error(e),
    };
    
    // Update status
    {
        let mut status = state.connection_status.write().await;
        *status = new_status.clone();
    }
    
    // Emit connection change
    let info = get_connection_info_inner(&state).await;
    let _ = app.emit("connection:changed", &info);
    
    Ok(new_status)
}

/// Disconnect from Jetson (set local mode)
#[tauri::command]
pub async fn disconnect_from_jetson(
    state: State<'_, Arc<AppState>>,
    app: AppHandle,
) -> Result<(), String> {
    {
        let mut mode = state.connection_mode.write().await;
        *mode = ConnectionMode::Local;
    }
    
    {
        let mut status = state.connection_status.write().await;
        *status = ConnectionStatus::Disconnected;
    }
    
    // Emit connection change
    let info = get_connection_info_inner(&state).await;
    let _ = app.emit("connection:changed", &info);
    
    Ok(())
}

/// Helper to get connection info without State wrapper
async fn get_connection_info_inner(state: &AppState) -> ConnectionInfo {
    let mode = *state.connection_mode.read().await;
    let status = state.connection_status.read().await.clone();
    let client = state.jetson_client.read().await;
    
    ConnectionInfo {
        mode,
        status,
        jetson_url: client.url().to_string(),
    }
}
