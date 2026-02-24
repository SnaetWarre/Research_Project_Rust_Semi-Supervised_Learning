//! Shared CLI helpers for workspace tools.

use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use crate::{Error, Result};

pub fn setup_cli_logging(verbose: bool) -> Result<()> {
    let filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::new("info")
    };

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .try_init()
        .map_err(|e| Error::Config(format!("Failed to initialize logger: {e}")))?;

    Ok(())
}

pub fn load_toml_config<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned,
{
    let content = fs::read_to_string(path)
        .map_err(|e| Error::Config(format!("Failed to read config {}: {e}", path.display())))?;

    toml::from_str(&content)
        .map_err(|e| Error::Config(format!("Failed to parse config {}: {e}", path.display())))
}
