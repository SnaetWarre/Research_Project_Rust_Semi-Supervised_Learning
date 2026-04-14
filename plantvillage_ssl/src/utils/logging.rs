//! Logging Module
//!
//! Provides structured logging utilities using the `tracing` crate.
//! Supports various output formats and log levels for debugging and production use.

use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Minimum log level to display
    pub level: LogLevel,
    /// Whether to include timestamps
    pub timestamps: bool,
    /// Whether to include target (module path)
    pub include_target: bool,
    /// Whether to include thread IDs
    pub include_thread_ids: bool,
    /// Whether to use ANSI colors
    pub ansi_colors: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            timestamps: true,
            include_target: false,
            include_thread_ids: false,
            ansi_colors: true,
        }
    }
}

impl LogConfig {
    /// Create a verbose logging config for debugging
    pub fn verbose() -> Self {
        Self {
            level: LogLevel::Debug,
            timestamps: true,
            include_target: true,
            include_thread_ids: true,
            ansi_colors: true,
        }
    }

    /// Create a quiet logging config (errors only)
    pub fn quiet() -> Self {
        Self {
            level: LogLevel::Error,
            timestamps: false,
            include_target: false,
            include_thread_ids: false,
            ansi_colors: true,
        }
    }

    /// Create a production logging config
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            timestamps: true,
            include_target: false,
            include_thread_ids: false,
            ansi_colors: false,
        }
    }
}

/// Log level enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    /// Convert to tracing Level
    pub fn to_tracing_level(&self) -> Level {
        match self {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        }
    }

    /// Create from string
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "trace" => LogLevel::Trace,
            "debug" => LogLevel::Debug,
            "info" => LogLevel::Info,
            "warn" | "warning" => LogLevel::Warn,
            "error" => LogLevel::Error,
            _ => LogLevel::Info,
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// Initialize logging with the given configuration
///
/// # Arguments
/// * `config` - Logging configuration
///
/// # Returns
/// * `Result<(), String>` - Ok if logging was initialized, Err with message otherwise
pub fn init_logging(config: &LogConfig) -> Result<(), String> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(config.level.to_tracing_level())
        .with_ansi(config.ansi_colors)
        .with_target(config.include_target)
        .with_thread_ids(config.include_thread_ids)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| format!("Failed to initialize logging: {}", e))?;

    Ok(())
}

/// Initialize logging with default settings
pub fn init_default_logging() -> Result<(), String> {
    init_logging(&LogConfig::default())
}

/// Initialize verbose logging for debugging
pub fn init_verbose_logging() -> Result<(), String> {
    init_logging(&LogConfig::verbose())
}

/// Progress logger for long-running operations
pub struct ProgressLogger {
    /// Operation name
    operation: String,
    /// Total items to process
    total: usize,
    /// Current progress
    current: usize,
    /// Log interval (log every N items)
    log_interval: usize,
    /// Start time
    start_time: std::time::Instant,
}

impl ProgressLogger {
    /// Create a new progress logger
    pub fn new(operation: &str, total: usize) -> Self {
        Self {
            operation: operation.to_string(),
            total,
            current: 0,
            log_interval: (total / 10).max(1),
            start_time: std::time::Instant::now(),
        }
    }

    /// Create with custom log interval
    pub fn with_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval.max(1);
        self
    }

    /// Update progress
    pub fn update(&mut self, count: usize) {
        self.current = count;

        if self.current % self.log_interval == 0 || self.current == self.total {
            let percentage = 100.0 * self.current as f64 / self.total as f64;
            let elapsed = self.start_time.elapsed();
            let items_per_sec = self.current as f64 / elapsed.as_secs_f64();

            let eta = if items_per_sec > 0.0 {
                let remaining = self.total - self.current;
                let eta_secs = remaining as f64 / items_per_sec;
                format!("{:.0}s", eta_secs)
            } else {
                "N/A".to_string()
            };

            tracing::info!(
                "{}: {}/{} ({:.1}%) - {:.1} items/s - ETA: {}",
                self.operation,
                self.current,
                self.total,
                percentage,
                items_per_sec,
                eta
            );
        }
    }

    /// Increment progress by 1
    pub fn increment(&mut self) {
        self.update(self.current + 1);
    }

    /// Log completion
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed();
        let items_per_sec = self.total as f64 / elapsed.as_secs_f64();

        tracing::info!(
            "{}: Completed {} items in {:.2}s ({:.1} items/s)",
            self.operation,
            self.total,
            elapsed.as_secs_f64(),
            items_per_sec
        );
    }
}

/// Training progress logger
pub struct TrainingLogger {
    /// Current epoch
    epoch: usize,
    /// Total epochs
    total_epochs: usize,
    /// Epoch start time
    epoch_start: std::time::Instant,
    /// Training start time
    training_start: std::time::Instant,
}

impl TrainingLogger {
    /// Create a new training logger
    pub fn new(total_epochs: usize) -> Self {
        Self {
            epoch: 0,
            total_epochs,
            epoch_start: std::time::Instant::now(),
            training_start: std::time::Instant::now(),
        }
    }

    /// Log start of an epoch
    pub fn start_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        self.epoch_start = std::time::Instant::now();

        tracing::info!(
            "Epoch {}/{} started",
            epoch + 1,
            self.total_epochs
        );
    }

    /// Log end of an epoch with metrics
    pub fn end_epoch(&self, train_loss: f64, val_accuracy: f64, learning_rate: f64) {
        let epoch_time = self.epoch_start.elapsed();
        let total_time = self.training_start.elapsed();

        let epochs_remaining = self.total_epochs - self.epoch - 1;
        let avg_epoch_time = total_time.as_secs_f64() / (self.epoch + 1) as f64;
        let eta_secs = epochs_remaining as f64 * avg_epoch_time;

        tracing::info!(
            "Epoch {}/{} completed in {:.1}s | Loss: {:.4} | Val Acc: {:.2}% | LR: {:.6} | ETA: {:.0}s",
            self.epoch + 1,
            self.total_epochs,
            epoch_time.as_secs_f64(),
            train_loss,
            val_accuracy * 100.0,
            learning_rate,
            eta_secs
        );
    }

    /// Log a new best model
    pub fn log_new_best(&self, accuracy: f64) {
        tracing::info!(
            "🎉 New best model! Accuracy: {:.2}%",
            accuracy * 100.0
        );
    }

    /// Log early stopping
    pub fn log_early_stop(&self, patience: usize) {
        tracing::warn!(
            "Early stopping triggered after {} epochs without improvement",
            patience
        );
    }

    /// Log training completion
    pub fn log_complete(&self, best_accuracy: f64) {
        let total_time = self.training_start.elapsed();

        tracing::info!(
            "Training complete! {} epochs in {:.1}s | Best accuracy: {:.2}%",
            self.total_epochs,
            total_time.as_secs_f64(),
            best_accuracy * 100.0
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_str() {
        assert_eq!(LogLevel::from_str("debug"), LogLevel::Debug);
        assert_eq!(LogLevel::from_str("INFO"), LogLevel::Info);
        assert_eq!(LogLevel::from_str("Warning"), LogLevel::Warn);
        assert_eq!(LogLevel::from_str("unknown"), LogLevel::Info);
    }

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.timestamps);
    }

    #[test]
    fn test_progress_logger() {
        let mut logger = ProgressLogger::new("Test", 100);
        logger.update(50);
        assert_eq!(logger.current, 50);
    }
}
