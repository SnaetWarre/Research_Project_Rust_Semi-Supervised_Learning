//! Runs SSL inside the app on a background thread and streams its progress as `ssl` events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use plantssl::data::Split;
use plantssl::model::PlantCnn;
use plantssl::progress::{Observer, Stopped};
use plantssl::ssl::{self, RoundLog, SslConfig};
use plantssl::train::{EpochLog, TrainConfig};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};

use crate::{CmdResult, err};

#[derive(Debug, Deserialize)]
pub struct SslRequest {
    split: PathBuf,
    init: PathBuf,
    out: PathBuf,
    rounds: usize,
    threshold: f32,
    epochs_per_round: usize,
    fresh_student: bool,
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
        let mut emitter = EventEmitter {
            app: app.clone(),
            stop,
            last_progress: Instant::now(),
        };
        let event = match run(&request, &mut emitter) {
            Ok(best_round) => SslEvent::Finished {
                best_round,
                out: request.out.clone(),
            },
            Err(e) if e.is::<Stopped>() => SslEvent::Stopped,
            Err(e) => SslEvent::Failed {
                message: format!("{e:#}"),
            },
        };
        emitter.send(event);
        if let Ok(mut running) = app.state::<LiveState>().0.lock() {
            *running = None;
        }
    });
    Ok(())
}

#[tauri::command]
pub fn stop_ssl(state: State<'_, LiveState>) -> CmdResult<()> {
    if let Some(stop) = state.0.lock().map_err(err)?.as_ref() {
        stop.store(true, Ordering::Relaxed);
    }
    Ok(())
}

/// Same settings as `plantssl ssl`, saved the same way so the Runs screen can show the result.
fn run(request: &SslRequest, observer: &mut dyn Observer) -> Result<usize> {
    let split = Split::load(&request.split)?;
    // Training needs the GPU; the CPU backend is far too slow for this
    let device = plantssl::device(plantssl::default_device_name())?;
    let (model, meta) = PlantCnn::load(&request.init, &device)?;
    ensure!(
        meta.classes == split.classes,
        "the model was trained on other classes than this split has"
    );
    let config = SslConfig {
        rounds: request.rounds,
        threshold: request.threshold,
        fresh_student: request.fresh_student,
        train: TrainConfig {
            epochs: request.epochs_per_round,
            learning_rate: if request.fresh_student { 1e-3 } else { 3e-4 },
            warmup_epochs: usize::from(request.fresh_student),
            image_size: meta.image_size,
            ..TrainConfig::default()
        },
    };
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
        self.send(SslEvent::Pass {
            label: label.to_string(),
            images,
        });
    }

    fn images_done(&mut self, done: usize) {
        // A few updates per second are enough for a progress bar
        if self.last_progress.elapsed() >= Duration::from_millis(200) {
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
