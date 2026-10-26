//! Error types for the plant incremental learning project.

use thiserror::Error;

/// Main error type for the plant learning project.
#[derive(Error, Debug)]
pub enum Error {
    /// IO error occurred
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Image processing error
    #[error("Image processing error: {0}")]
    Image(String),

    /// Model error
    #[error("Model error: {0}")]
    Model(String),

    /// Dataset error
    #[error("Dataset error: {0}")]
    Dataset(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Training error
    #[error("Training error: {0}")]
    Training(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Invalid argument error
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Not found error
    #[error("Not found: {0}")]
    NotFound(String),

    /// Already exists error
    #[error("Already exists: {0}")]
    AlreadyExists(String),

    /// Incremental learning error
    #[error("Incremental learning error: {0}")]
    IncrementalLearning(String),

    /// Device error
    #[error("Device error: {0}")]
    Device(String),

    /// Generic error with context
    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Serialization(err.to_string())
    }
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::Image(err.to_string())
    }
}

/// Specialized Result type for plant learning operations.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::Model("test error".to_string());
        assert_eq!(err.to_string(), "Model error: test error");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
    }

    #[test]
    fn test_result_type() {
        let success: Result<i32> = Ok(42);
        assert!(success.is_ok());

        let failure: Result<i32> = Err(Error::Other("test".to_string()));
        assert!(failure.is_err());
    }
}
