//! Core types and utilities for plant incremental learning research.
//!
//! This crate provides the foundational types, traits, and utilities used
//! across the plant incremental learning project.

pub mod error;
pub mod types;
pub mod config;
pub mod metrics;
pub mod backend;
pub mod cli;

pub use error::{Error, Result};
pub use types::*;
pub use config::*;
pub use metrics::*;
pub use backend::*;
pub use cli::*;

/// Re-export commonly used burn types
pub mod prelude {
    pub use burn::prelude::*;
    pub use crate::error::{Error, Result};
    pub use crate::types::*;
    pub use crate::config::*;
    pub use crate::metrics::*;
    pub use crate::backend::*;
    pub use crate::cli::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_compiles() {
        // Basic smoke test
        assert!(true);
    }
}
