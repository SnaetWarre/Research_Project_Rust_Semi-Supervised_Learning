//! Health check endpoint

use axum::{extract::State, Json};
use serde::Serialize;

use crate::state::SharedState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
    pub version: String,
}

/// GET /health - Health check endpoint
pub async fn health_check(State(state): State<SharedState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        uptime_seconds: state.uptime_seconds(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
