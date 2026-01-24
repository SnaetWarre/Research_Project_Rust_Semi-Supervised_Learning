//! Error Handling Module
//!
//! Defines custom error types for the PlantVillage SSL library.
//! Uses thiserror for ergonomic error definitions.

use std::path::PathBuf;

use thiserror::Error;

/// Main error type for PlantVillage SSL operations
#[derive(Error, Debug)]
pub enum PlantVillageError {
    /// Error loading or processing an image
    #[error("Failed to load image at '{0}': {1}")]
    ImageLoadError(PathBuf, String),

    /// Error with dataset operations
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Error with model operations
    #[error("Model error: {0}")]
    Model(String),

    /// Error with training
    #[error("Training error: {0}")]
    Training(String),

    /// Error with inference
    #[error("Inference error: {0}")]
    Inference(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Device/hardware error
    #[error("Device error: {0}")]
    Device(String),

    /// Path not found
    #[error("Path not found: {0}")]
    PathNotFound(PathBuf),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Convenience Result type for PlantVillage SSL operations
pub type Result<T> = std::result::Result<T, PlantVillageError>;

/// Extension trait for adding context to errors
pub trait ResultExt<T> {
    /// Add context to an error
    fn context(self, msg: &str) -> Result<T>;

    /// Add context with a closure (lazy evaluation)
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T, E: std::error::Error> ResultExt<T> for std::result::Result<T, E> {
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| PlantVillageError::InvalidInput(format!("{}: {}", msg, e)))
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| PlantVillageError::InvalidInput(format!("{}: {}", f(), e)))
    }
}

impl<T> ResultExt<T> for Option<T> {
    fn context(self, msg: &str) -> Result<T> {
        self.ok_or_else(|| PlantVillageError::InvalidInput(msg.to_string()))
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.ok_or_else(|| PlantVillageError::InvalidInput(f()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = PlantVillageError::Dataset("test error".to_string());
        assert_eq!(format!("{}", err), "Dataset error: test error");
    }

    #[test]
    fn test_image_load_error() {
        let path = PathBuf::from("/path/to/image.jpg");
        let err = PlantVillageError::ImageLoadError(path.clone(), "file not found".to_string());
        assert!(format!("{}", err).contains("image.jpg"));
    }

    #[test]
    fn test_result_context() {
        let result: std::result::Result<i32, std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));

        let with_context = result.context("Failed to read file");
        assert!(with_context.is_err());
    }

    #[test]
    fn test_option_context() {
        let opt: Option<i32> = None;
        let with_context = opt.context("Value was None");
        assert!(with_context.is_err());
    }
}
