//! Semi-supervised plant disease classification with pseudo-labeling, built on Burn.
//!
//! The pipeline has four steps, each a CLI command:
//!
//! 1. [`data`]: split the dataset once into labeled, stream, validation and test pools;
//! 2. [`train`]: train a supervised model on the labeled pool;
//! 3. [`ssl`]: improve it with pseudo-labels from the unlabeled stream;
//! 4. [`eval`]: score a model on the test pool.
//!
//! [`predict`] loads a saved model for single-image classification, which the desktop app uses.

pub mod data;
pub mod eval;
pub mod images;
pub mod model;
pub mod predict;
pub mod progress;
pub mod ssl;
pub mod train;

use std::path::Path;

use anyhow::{Result, bail};
use burn::prelude::Device;
use serde::Serialize;

/// Picks a compute device by name: `cuda`, `wgpu` or `cpu`.
pub fn device(name: &str) -> Result<Device> {
    Ok(match name {
        #[cfg(feature = "cuda")]
        "cuda" => Device::cuda(0),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(Default::default()),
        "cpu" => Device::flex(),
        #[cfg(not(feature = "cuda"))]
        "cuda" => bail!("this build has no CUDA support; rebuild with `--features cuda`"),
        #[cfg(not(feature = "wgpu"))]
        "wgpu" => bail!("this build has no wgpu support; rebuild with `--features wgpu`"),
        other => bail!("unknown device `{other}`; use cuda, wgpu or cpu"),
    })
}

/// The best device this build supports.
pub fn default_device_name() -> &'static str {
    if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "wgpu") {
        "wgpu"
    } else {
        "cpu"
    }
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)?)?;
    Ok(())
}
