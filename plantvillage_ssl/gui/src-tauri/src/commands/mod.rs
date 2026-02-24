//! Tauri Command Handlers
//!
//! This module contains all the Tauri commands that bridge the frontend
//! to the plantvillage_ssl Rust backend.

pub mod dataset;
pub mod inference;
pub mod training;
pub mod pseudo;
pub mod simulation;
pub mod benchmark;
pub mod incremental;
pub mod diagnostics;
pub mod experiments;
pub mod demo;
pub mod ssl_mobile;
pub mod dataset_bundle;
pub mod shared;

// Re-export all commands for registration
pub use dataset::*;
pub use inference::*;
pub use training::*;
pub use pseudo::*;
pub use simulation::*;
pub use benchmark::*;
pub use incremental::*;
pub use diagnostics::*;
pub use experiments::*;
pub use demo::*;
pub use ssl_mobile::*;
pub use dataset_bundle::*;
