//! Utilities module for logging, metrics, and helper functions
//!
//! This module provides:
//! - Structured logging with tracing
//! - Metrics computation (accuracy, F1-score, confusion matrix)
//! - Error handling types
//! - General helper functions
//!
//! ## Metrics
//!
//! The metrics module provides comprehensive evaluation utilities:
//! - Per-class precision, recall, and F1-score
//! - Confusion matrix visualization
//! - Accuracy tracking over time

pub mod charts;
pub mod error;
pub mod logging;
pub mod metrics;

// Re-export main types for convenience
pub use error::{PlantVillageError, Result};
pub use logging::init_logging;
pub use metrics::{Metrics, ConfusionMatrix};

/// Format a duration in a human-readable way
pub fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else if seconds < 3600.0 {
        let minutes = (seconds / 60.0).floor();
        let secs = seconds % 60.0;
        format!("{}m {:.0}s", minutes as u32, secs)
    } else {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds % 3600.0) / 60.0).floor();
        format!("{}h {}m", hours as u32, minutes as u32)
    }
}

/// Format a number with thousands separator
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }

    result
}

/// Format a percentage with a progress bar
pub fn format_progress_bar(progress: f64, width: usize) -> String {
    let filled = (progress * width as f64).round() as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "[{}{}] {:.1}%",
        "█".repeat(filled),
        "░".repeat(empty),
        progress * 100.0
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(30.5), "30.5s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3661.0), "1h 1m");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(42), "42");
    }

    #[test]
    fn test_format_progress_bar() {
        let bar = format_progress_bar(0.5, 10);
        assert!(bar.contains("50.0%"));
        assert!(bar.contains("█████"));
    }
}
