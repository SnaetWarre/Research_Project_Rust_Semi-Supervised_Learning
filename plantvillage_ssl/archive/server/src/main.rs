//! PlantVillage Training Server
//!
//! HTTP API server for running PlantVillage SSL training on Jetson devices.
//! Provides endpoints for training control, status monitoring, file access,
//! and real-time output streaming.

mod routes;
mod state;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use crate::state::{AppState, ServerConfig};

/// PlantVillage Training Server
#[derive(Parser, Debug)]
#[command(name = "plantvillage-server")]
#[command(author = "Warre Snaet")]
#[command(version = "0.1.0")]
#[command(about = "HTTP API server for PlantVillage SSL training")]
struct Cli {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Project directory (where plantvillage_ssl is located)
    #[arg(long, env = "PLANTVILLAGE_PROJECT_DIR")]
    project_dir: Option<PathBuf>,

    /// Path to plantvillage_ssl binary
    #[arg(long, env = "PLANTVILLAGE_BINARY")]
    binary: Option<PathBuf>,

    /// Output/results directory
    #[arg(long, env = "PLANTVILLAGE_OUTPUT_DIR")]
    output_dir: Option<PathBuf>,

    /// Models directory
    #[arg(long, env = "PLANTVILLAGE_MODELS_DIR")]
    models_dir: Option<PathBuf>,

    /// Data directory
    #[arg(long, env = "PLANTVILLAGE_DATA_DIR")]
    data_dir: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .compact()
        .init();

    // Build configuration
    let mut config = ServerConfig::default();
    
    if let Some(project_dir) = cli.project_dir {
        config.project_dir = project_dir.clone();
        config.binary_path = project_dir.join("target/release-jetson/plantvillage_ssl");
        config.results_dir = project_dir.join("output");
        config.models_dir = project_dir.join("output/models");
        config.data_dir = project_dir.join("data");
    }
    
    if let Some(binary) = cli.binary {
        config.binary_path = binary;
    }
    
    if let Some(output_dir) = cli.output_dir {
        config.results_dir = output_dir;
    }
    
    if let Some(models_dir) = cli.models_dir {
        config.models_dir = models_dir;
    }
    
    if let Some(data_dir) = cli.data_dir {
        config.data_dir = data_dir;
    }

    info!("PlantVillage Training Server v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration:");
    info!("  Project dir: {:?}", config.project_dir);
    info!("  Binary path: {:?}", config.binary_path);
    info!("  Results dir: {:?}", config.results_dir);
    info!("  Models dir:  {:?}", config.models_dir);
    info!("  Data dir:    {:?}", config.data_dir);

    // Check if binary exists
    if !config.binary_path.exists() {
        tracing::warn!(
            "Binary not found at {:?}. Training commands will fail. \
            Build with: cargo build --release",
            config.binary_path
        );
    }

    // Create shared state
    let state = Arc::new(AppState::new(config));

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(routes::health::health_check))
        
        // Training endpoints
        .route("/train", post(routes::training::start_training))
        .route("/train/status", get(routes::training::get_training_status))
        .route("/train/stream", get(routes::training::stream_training_output))
        .route("/train/stop", post(routes::training::stop_training))
        .route("/train/history", get(routes::training::get_training_history))
        
        // Experiments
        .route("/experiments", get(routes::experiments::list_experiments))
        .route("/experiments/:id", get(routes::experiments::get_experiment))
        
        // Files
        .route("/models", get(routes::files::list_models))
        .route("/output", get(routes::files::list_output))
        .route("/files/*path", get(routes::files::serve_file))
        
        // Add state
        .with_state(state)
        
        // Add middleware
        .layer(TraceLayer::new_for_http())
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Start server
    let addr: SocketAddr = format!("{}:{}", cli.host, cli.port).parse()?;
    info!("Starting server on http://{}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
