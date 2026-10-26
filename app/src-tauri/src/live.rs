//! Runs SSL inside the app on a background thread and streams its progress as `ssl` events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use plantssl::data::Split;
use plantssl::model::PlantCnn;
use plantssl::progress::{Observer, Stopped};
use plantssl::ssl::{self, Mode, RoundLog, SslConfig};
use plantssl::train::EpochLog;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use crate::{CmdResult, err};

#[derive(Debug, Deserialize)]
pub struct SslRequest {
    split: PathBuf,
    init: PathBuf,
    out: PathBuf,
    rounds: usize,
    keep: f32,
    mode: Mode,
    epochs_per_round: usize,
}

#[derive(Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SslEvent {
    Pass { label: String, images: usize },
    Progress { done: usize },
    Epoch(EpochLog),
    Round(RoundLog),
    Finished { best_round: usize, out: PathBuf },
    Stopped,
    Failed { message: String },
}

/// Set while a run is going; holds the flag that asks it to stop.
#[derive(Default)]
pub struct LiveState(Mutex<Option<Arc<AtomicBool>>>);

#[tauri::command]
pub fn start_ssl<R: Runtime>(
    app: AppHandle<R>,
    request: SslRequest,
    state: State<'_, LiveState>,
) -> CmdResult<()> {
    let mut running = state.0.lock().map_err(err)?;
    if running.is_some() {
        return Err("SSL is already running".into());
    }
    let stop = Arc::new(AtomicBool::new(false));
    *running = Some(stop.clone());

    std::thread::spawn(move || {
        // Clears the running flag even if the run panics, so Start works again afterwards
        let guard = ClearOnDrop(app.clone());
        let mut emitter = EventEmitter {
            app: app.clone(),
            stop,
            pass_images: 0,
            last_progress: Instant::now(),
        };
        let outcome =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&request, &mut emitter)));
        let event = match outcome.unwrap_or_else(|_| Err(anyhow::anyhow!("the run crashed"))) {
            Ok(best_round) => SslEvent::Finished {
                best_round,
                out: request.out.clone(),
            },
            Err(e) if e.is::<Stopped>() => SslEvent::Stopped,
            Err(e) => SslEvent::Failed {
                message: format!("{e:#}"),
            },
        };
        // Clear before telling the UI, so a Start right after this event is accepted
        drop(guard);
        emitter.send(event);
    });
    Ok(())
}

struct ClearOnDrop<R: Runtime>(AppHandle<R>);

impl<R: Runtime> Drop for ClearOnDrop<R> {
    fn drop(&mut self) {
        if let Ok(mut running) = self.0.state::<LiveState>().0.lock() {
            *running = None;
        }
    }
}

#[tauri::command]
pub fn stop_ssl(state: State<'_, LiveState>) -> CmdResult<()> {
    if let Some(stop) = state.0.lock().map_err(err)?.as_ref() {
        stop.store(true, Ordering::Relaxed);
    }
    Ok(())
}

/// Same settings as `plantssl ssl` (both use `SslConfig::new`), saved the same way so the Runs
/// screen can show the result.
fn run(request: &SslRequest, observer: &mut dyn Observer) -> Result<usize> {
    let split = Split::load(&request.split)?;
    // Training needs the GPU; the CPU backend is far too slow for this
    let device = plantssl::device(plantssl::default_device_name())?;
    let (model, meta) = PlantCnn::load(&request.init, &device)?;
    ensure!(
        meta.classes == split.classes,
        "the model was trained on other classes than this split has"
    );
    let config = SslConfig::new(
        request.rounds,
        request.keep,
        request.mode,
        request.epochs_per_round,
        meta.image_size,
    );
    let result = ssl::run(model, &meta, &split, &config, &device, observer)?;
    result.model.save(&request.out, &meta)?;
    plantssl::write_json(
        &request.out.join("ssl.json"),
        &serde_json::json!({ "best_round": result.best_round, "rounds": result.rounds }),
    )?;
    Ok(result.best_round)
}

struct EventEmitter<R: Runtime> {
    app: AppHandle<R>,
    stop: Arc<AtomicBool>,
    pass_images: usize,
    last_progress: Instant,
}

impl<R: Runtime> EventEmitter<R> {
    fn send(&self, event: SslEvent) {
        // Nothing to do if the window is gone
        let _ = self.app.emit("ssl", event);
    }
}

impl<R: Runtime> Observer for EventEmitter<R> {
    fn pass(&mut self, label: &str, images: usize) {
        self.pass_images = images;
        self.send(SslEvent::Pass {
            label: label.to_string(),
            images,
        });
    }

    fn images_done(&mut self, done: usize) {
        // A few updates per second are enough for a progress bar, plus the final one
        if done >= self.pass_images || self.last_progress.elapsed() >= Duration::from_millis(200) {
            self.last_progress = Instant::now();
            self.send(SslEvent::Progress { done });
        }
    }

    fn epoch(&mut self, log: &EpochLog) {
        self.send(SslEvent::Epoch(log.clone()));
    }

    fn round(&mut self, log: &RoundLog) {
        self.send(SslEvent::Round(log.clone()));
    }

    fn should_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }
}
