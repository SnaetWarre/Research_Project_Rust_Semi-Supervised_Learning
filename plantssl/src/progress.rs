//! Progress reporting for long runs. The CLI draws a progress bar from these calls and the app
//! streams them to its live SSL screen.

use crate::ssl::RoundLog;
use crate::train::EpochLog;

/// Receives progress from [`train::fit`](crate::train::fit), [`ssl::run`](crate::ssl::run) and
/// [`eval::predict`](crate::eval::predict). Every method has a no-op default.
pub trait Observer {
    /// A new pass over the data starts, such as an epoch or labeling the stream.
    fn pass(&mut self, _label: &str, _images: usize) {}
    /// Images finished so far in the current pass.
    fn images_done(&mut self, _done: usize) {}
    fn epoch(&mut self, _log: &EpochLog) {}
    fn round(&mut self, _log: &RoundLog) {}
    /// Checked between batches. Returning true ends the run with a [`Stopped`] error.
    fn should_stop(&self) -> bool {
        false
    }
}

/// Ignores all progress.
pub struct Silent;

impl Observer for Silent {}

/// The run ended because [`Observer::should_stop`] returned true.
#[derive(Debug)]
pub struct Stopped;

impl std::fmt::Display for Stopped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("stopped on request")
    }
}

impl std::error::Error for Stopped {}
