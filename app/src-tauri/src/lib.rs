//! Tauri backend: loads a model, classifies photos and reads the JSON files a run writes.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use plantssl::predict::{ClassScore, Predictor};
use serde::Serialize;
use serde_json::Value;
use tauri::State;
use tauri::ipc::{InvokeBody, Request};

/// Errors cross the IPC boundary as plain strings.
type CmdResult<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Default)]
struct AppState {
    predictor: Mutex<Option<Predictor>>,
}

#[derive(Serialize)]
struct ModelInfo {
    dir: PathBuf,
    classes: usize,
    image_size: u32,
}

#[tauri::command]
fn load_model(dir: PathBuf, state: State<'_, AppState>) -> CmdResult<ModelInfo> {
    let predictor = Predictor::load(&dir, plantssl::device("cpu").map_err(err)?).map_err(err)?;
    let info = ModelInfo {
        dir,
        classes: predictor.meta().classes.len(),
        image_size: predictor.meta().image_size,
    };
    *state.predictor.lock().map_err(err)? = Some(predictor);
    Ok(info)
}

#[derive(Serialize)]
struct Classification {
    scores: Vec<ClassScore>,
    millis: f64,
}

/// Classifies the raw bytes of an image file sent as the request body.
#[tauri::command]
fn classify(request: Request<'_>, state: State<'_, AppState>) -> CmdResult<Classification> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("expected the image bytes as the request body".into());
    };
    let image = image::load_from_memory(bytes).map_err(err)?;
    let guard = state.predictor.lock().map_err(err)?;
    let predictor = guard.as_ref().ok_or("load a model first")?;
    let started = Instant::now();
    let mut scores = predictor.classify(&image);
    let millis = started.elapsed().as_secs_f64() * 1000.0;
    scores.truncate(5);
    Ok(Classification { scores, millis })
}

#[derive(Serialize)]
struct Run {
    name: String,
    train: Option<Value>,
    ssl: Option<Value>,
    test: Option<Value>,
}

/// Every model folder directly inside `dir`, with the JSON each command left there.
#[tauri::command]
fn list_runs(dir: PathBuf) -> CmdResult<Vec<Run>> {
    let mut runs = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(err)? {
        let path = entry.map_err(err)?.path();
        if !path.join("model.json").is_file() {
            continue;
        }
        runs.push(Run {
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            train: read_json(&path.join("train.json")),
            ssl: read_json(&path.join("ssl.json")),
            test: read_json(&path.join("eval-test.json")).map(|mut report| {
                // The confusion matrix is large and the app does not draw it
                if let Some(obj) = report.as_object_mut() {
                    obj.remove("confusion");
                }
                report
            }),
        });
    }
    runs.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(runs)
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![load_model, classify, list_runs])
        .run(tauri::generate_context!())
        .expect("error while running the app");
}
