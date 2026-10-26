//! Core types and utilities for plant incremental learning research.
//!
//! This crate provides the foundational types, traits, and utilities used
//! across the plant incremental learning project.

pub mod backend;
pub mod cli;
pub mod config;
pub mod error;
pub mod metrics;
pub mod types;

pub use backend::*;
pub use cli::*;
pub use config::*;
pub use error::{Error, Result};
pub use metrics::*;
pub use types::*;

/// Re-export commonly used burn types
pub mod prelude {
    pub use crate::backend::*;
    // Prefer our backend `Device` enum over the one in burn's prelude
    pub use crate::backend::Device;
    pub use crate::cli::*;
    pub use crate::config::*;
    pub use crate::error::{Error, Result};
    pub use crate::metrics::*;
    pub use crate::types::*;
    pub use burn::prelude::*;
}
