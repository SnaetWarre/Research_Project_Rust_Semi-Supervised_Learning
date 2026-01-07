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

// Re-export all commands for registration
pub use dataset::*;
pub use inference::*;
pub use training::*;
pub use pseudo::*;
pub use simulation::*;
pub use benchmark::*;
