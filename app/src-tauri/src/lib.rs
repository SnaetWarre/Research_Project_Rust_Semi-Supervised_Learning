//! Tauri backend: loads a model, classifies photos, runs SSL live and reads the JSON files a run
//! writes.

mod live;

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

// `async` runs the command off the main thread, so the window keeps painting meanwhile
#[tauri::command(async)]
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
#[tauri::command(async)]
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
#[tauri::command(async)]
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

/// State and commands, shared by the real app and the tests.
fn with_commands<R: tauri::Runtime>(builder: tauri::Builder<R>) -> tauri::Builder<R> {
    builder
        .manage(AppState::default())
        .manage(live::LiveState::default())
        .invoke_handler(tauri::generate_handler![
            load_model,
            classify,
            list_runs,
            live::start_ssl,
            live::stop_ssl
        ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    with_commands(tauri::Builder::default())
        .plugin(tauri_plugin_dialog::init())
        .run(tauri::generate_context!())
        .expect("error while running the app");
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{ImageFormat, Rgb, RgbImage};
    use plantssl::model::{ModelMeta, PlantCnn};
    use serde_json::json;
    use tauri::ipc::{CallbackFn, InvokeBody};
    use tauri::test::{INVOKE_KEY, get_ipc_response, mock_builder, mock_context, noop_assets};
    use tauri::webview::InvokeRequest;

    use super::*;

    fn call(
        webview: &tauri::WebviewWindow<tauri::test::MockRuntime>,
        cmd: &str,
        body: InvokeBody,
    ) -> Value {
        let request = InvokeRequest {
            cmd: cmd.into(),
            callback: CallbackFn(0),
            error: CallbackFn(1),
            url: "tauri://localhost".parse().unwrap(),
            body,
            headers: Default::default(),
            invoke_key: INVOKE_KEY.to_string(),
        };
        get_ipc_response(webview, request)
            .unwrap_or_else(|e| panic!("{cmd} failed: {e}"))
            .deserialize()
            .unwrap()
    }

    #[test]
    fn commands_load_a_model_classify_bytes_and_list_runs() {
        let runs = tempfile::tempdir().unwrap();
        let meta = ModelMeta {
            classes: vec!["a".into(), "b".into(), "c".into()],
            image_size: 32,
            width: 8,
        };
        let device = plantssl::device("cpu").unwrap();
        PlantCnn::new(&meta, &device)
            .save(&runs.path().join("tiny"), &meta)
            .unwrap();

        let app = with_commands(mock_builder())
            .build(mock_context(noop_assets()))
            .unwrap();
        let webview = tauri::WebviewWindowBuilder::new(&app, "main", Default::default())
            .build()
            .unwrap();

        let info = call(
            &webview,
            "load_model",
            InvokeBody::Json(json!({ "dir": runs.path().join("tiny") })),
        );
        assert_eq!(info["classes"], 3);

        // A wide photo, sent the way the frontend sends it: raw file bytes as the body
        let mut png = Vec::new();
        RgbImage::from_pixel(60, 40, Rgb([40, 140, 60]))
            .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
            .unwrap();
        let result = call(&webview, "classify", InvokeBody::Raw(png));
        let scores = result["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 3);
        let total: f64 = scores
            .iter()
            .map(|s| s["probability"].as_f64().unwrap())
            .sum();
        assert!((total - 1.0).abs() < 1e-4);

        let listed = call(
            &webview,
            "list_runs",
            InvokeBody::Json(json!({ "dir": runs.path() })),
        );
        assert_eq!(listed[0]["name"], "tiny");
    }
}
