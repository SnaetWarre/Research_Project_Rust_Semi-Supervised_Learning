//! Tauri Command Handlers
//!
//! This module contains all the Tauri commands that bridge the frontend
//! to the plantvillage_ssl Rust backend.

pub mod benchmark;
pub mod dataset;
pub mod dataset_bundle;
pub mod demo;
pub mod diagnostics;
pub mod experiments;
pub mod incremental;
pub mod inference;
pub mod pseudo;
pub mod shared;
pub mod simulation;
pub mod ssl_mobile;
pub mod training;

// Re-export all commands for registration
pub use benchmark::*;
pub use dataset::*;
pub use dataset_bundle::*;
pub use demo::*;
pub use diagnostics::*;
pub use experiments::*;
pub use incremental::*;
pub use inference::*;
pub use pseudo::*;
pub use simulation::*;
pub use ssl_mobile::*;
pub use training::*;
