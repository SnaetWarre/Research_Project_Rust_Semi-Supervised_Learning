//! Interactive SSL Demo Commands
//!
//! Commands for running the day-by-day interactive SSL demo
//! that demonstrates the "farmer's edge device" narrative.

use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::Mutex;

use burn::data::dataset::Dataset;
use burn::{
    data::dataloader::batcher::Batcher,
    module::{AutodiffModule, Module},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, ElementConversion},
};
use image::ImageFormat;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use plantvillage_ssl::{
    dataset::{
        burn_dataset::{
            AugmentingBatcher, PlantVillageBurnDataset, PlantVillageItem, RawPlantVillageDataset,
        },
        split::{DatasetSplits, HiddenLabelImage, SplitConfig},
    },
    model::cnn::PlantClassifierConfig,
    training::pseudo_label::{Prediction, PseudoLabelConfig, PseudoLabeler, StreamSimulator},
    PlantClassifier, PlantVillageBatcher, PlantVillageDataset,
};

use crate::state::AppState;

/// Configuration for demo session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoConfig {
    pub data_dir: String,
    pub model_path: String,
    pub images_per_day: usize,
    pub confidence_threshold: f64,
    pub retrain_threshold: usize,
    pub labeled_ratio: f64,
    pub seed: u64,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            data_dir: "data/plantvillage".to_string(),
            model_path: "output/models/plant_classifier".to_string(),
            images_per_day: 100,
            confidence_threshold: 0.9,
            retrain_threshold: 200,
            labeled_ratio: 0.25,
            seed: 42,
        }
    }
}

/// The state of a demo session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoSessionState {
    pub current_day: usize,
    pub total_images_available: usize,
    pub images_processed: usize,
    pub pseudo_labels_accumulated: usize,
    pub total_pseudo_labels_generated: usize,
    pub retraining_count: usize,
    pub current_accuracy: f64,
    pub initial_accuracy: f64,
    pub pseudo_label_precision: f64,
    pub accuracy_history: Vec<(usize, f64)>,
    pub config: DemoConfig,
}

/// Result of advancing one day
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayResult {
    pub day: usize,
    pub images_processed_today: usize,
    pub pseudo_labels_accepted_today: usize,
    pub pseudo_labels_accumulated: usize,
    pub did_retrain: bool,
    pub accuracy_before_retrain: Option<f64>,
    pub accuracy_after_retrain: Option<f64>,
    pub current_accuracy: f64,
    pub pseudo_label_precision: f64,
    pub sample_images: Vec<DayImage>,
    pub remaining_images: usize,
}

/// A single image from today's batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DayImage {
    pub path: String,
    pub predicted_label: usize,
    pub confidence: f32,
    pub accepted: bool,
    pub ground_truth: usize,
    pub is_correct: bool,
    pub base64_thumbnail: Option<String>, // Base64 encoded thumbnail for display
    pub is_farmer_image: bool,            // Whether this is from farmer demo upload
}

/// Helper function to encode an image to base64 thumbnail (96x96)
fn encode_image_thumbnail(path: &Path) -> Option<String> {
    use base64::Engine;

    // Load and resize image
    let img = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            tracing::warn!("Failed to load image {:?}: {:?}", path, e);
            return None;
        }
    };

    let thumbnail = img.resize(96, 96, image::imageops::FilterType::Lanczos3);

    // Encode to JPEG in memory
    let mut buffer = Cursor::new(Vec::new());
    if let Err(e) = thumbnail.write_to(&mut buffer, ImageFormat::Jpeg) {
        tracing::warn!("Failed to encode thumbnail for {:?}: {:?}", path, e);
        return None;
    }

    // Convert to base64
    let base64_str = base64::engine::general_purpose::STANDARD.encode(buffer.into_inner());
    Some(format!("data:image/jpeg;base64,{}", base64_str))
}

/// The persistent demo session
pub struct DemoSession<B: AutodiffBackend> {
    model: PlantClassifier<B>,
    optimizer: burn::optim::adaptor::OptimizerAdaptor<burn::optim::Adam, PlantClassifier<B>, B>,
    stream_simulator: StreamSimulator,
    farmer_simulator: Option<StreamSimulator>, // Farmer images processed like daily stream
    pseudo_labeler: PseudoLabeler,
    splits: DatasetSplits,
    val_dataset: PlantVillageBurnDataset,
    batcher: PlantVillageBatcher<B::InnerBackend>,
    state: DemoSessionState,
    device: B::Device,
}

/// Global demo session state
pub type DemoSessionGlobal =
    Arc<Mutex<Option<DemoSession<crate::backend::AdaptiveBackend>>>>;

/// Initialize a new demo session
#[tauri::command]
pub async fn init_demo_session(
    config: DemoConfig,
    demo_session: State<'_, DemoSessionGlobal>,
    _app_state: State<'_, Arc<AppState>>,
) -> Result<DemoSessionState, String> {
    use crate::backend::AdaptiveBackend;

    // Run initialization in blocking task
    let session = tokio::task::spawn_blocking(move || init_session_inner::<AdaptiveBackend>(config))
        .await
        .map_err(|e| format!("Failed to spawn task: {:?}", e))??;

    let state = session.state.clone();

    // Store the session globally
    let mut global = demo_session.lock().await;
    *global = Some(session);

    Ok(state)
}

/// Advance the simulation by one day
#[tauri::command]
pub async fn advance_demo_day(
    demo_session: State<'_, DemoSessionGlobal>,
    app: AppHandle,
) -> Result<DayResult, String> {
    use crate::backend::AdaptiveBackend;
    use std::time::Instant;

    let total_start = Instant::now();
    tracing::info!("=== advance_demo_day called ===");
    
    // Emit progress event
    let _ = app.emit("demo:progress", serde_json::json!({"step": "starting", "message": "Starting day processing..."}));

    // Take ownership of session temporarily for processing
    tracing::info!("Step 1: Acquiring session lock...");
    let lock_start = Instant::now();
    let mut session_opt = {
        let mut global = demo_session.lock().await;
        global.take()
    };
    tracing::info!("Step 1: Lock acquired in {:?}", lock_start.elapsed());

    let session = session_opt.as_mut().ok_or("No demo session initialized")?;
    tracing::info!("Step 2: Session retrieved, current day: {}", session.state.current_day);

    // Check if retraining will happen and emit event
    let will_retrain = session.pseudo_labeler.should_retrain();
    tracing::info!("Step 3: Will retrain? {}", will_retrain);
    
    if will_retrain {
        tracing::info!("Step 3: Emitting retraining_started event");
        let _ = app.emit("demo:progress", serde_json::json!({"step": "retraining", "message": "Retraining model..."}));
        let _ = app.emit(
            "demo:retraining_started",
            serde_json::json!({
                "message": "Retraining model with pseudo-labels...",
                "pseudo_labels": session.pseudo_labeler.stats().total_accepted,
            }),
        );
    }

    // Process day
    tracing::info!("Step 4: Calling process_demo_day...");
    let _ = app.emit("demo:progress", serde_json::json!({"step": "processing", "message": "Processing images..."}));
    let process_start = Instant::now();
    
    let result = process_demo_day::<AdaptiveBackend>(session)
        .map_err(|e| {
            tracing::error!("process_demo_day FAILED: {}", e);
            format!("Failed to process day: {}", e)
        })?;
    
    tracing::info!("Step 4: process_demo_day completed in {:?}", process_start.elapsed());

    // Restore session
    tracing::info!("Step 5: Restoring session...");
    {
        let mut global = demo_session.lock().await;
        *global = session_opt;
    }
    tracing::info!("Step 5: Session restored");

    // Emit event
    tracing::info!("Step 6: Emitting day_complete event for day {}", result.day);
    let _ = app.emit("demo:day_complete", &result);

    tracing::info!("=== advance_demo_day COMPLETED in {:?} ===", total_start.elapsed());
    Ok(result)
}

/// Get current demo session state
#[tauri::command]
pub async fn get_demo_session_state(
    demo_session: State<'_, DemoSessionGlobal>,
) -> Result<Option<DemoSessionState>, String> {
    let global = demo_session.lock().await;
    Ok(global.as_ref().map(|s| s.state.clone()))
}

/// Reset the demo session
#[tauri::command]
pub async fn reset_demo_session(demo_session: State<'_, DemoSessionGlobal>) -> Result<(), String> {
    let mut global = demo_session.lock().await;
    *global = None;
    Ok(())
}

/// Result of processing farmer images
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarmerImportResult {
    pub images_processed: usize,
    pub pseudo_labels_accepted: usize,
    pub pseudo_labels_accumulated: usize,
    pub sample_images: Vec<DayImage>,
    pub current_accuracy: f64,
    pub pseudo_label_precision: f64,
}

/// Load farmer demo images from the farmer_demo directory
fn load_farmer_demo_images(data_dir: &str) -> Result<Vec<HiddenLabelImage>, String> {
    let farmer_demo_dir = PathBuf::from(data_dir)
        .parent()
        .ok_or("Invalid data_dir path")?
        .join("farmer_demo");

    if !farmer_demo_dir.exists() {
        return Err(format!(
            "Farmer demo directory not found at {:?}. Run scripts/setup_farmer_demo.sh first.",
            farmer_demo_dir
        ));
    }

    tracing::info!("Loading farmer demo images from {:?}", farmer_demo_dir);

    // Get the class name to label mapping by reading the plantvillage dataset classes
    let plantvillage_dir = PathBuf::from(data_dir);
    let train_dir = plantvillage_dir.join("train");
    
    let mut class_to_label: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if train_dir.exists() {
        let mut classes: Vec<String> = std::fs::read_dir(&train_dir)
            .map_err(|e| format!("Failed to read train dir: {:?}", e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .collect();
        classes.sort();
        for (idx, class_name) in classes.iter().enumerate() {
            class_to_label.insert(class_name.clone(), idx);
        }
    }

    // Load images from farmer_demo directory
    let mut images = Vec::new();
    let mut image_id = 1_000_000; // Start with high ID to distinguish from regular stream

    let class_dirs = std::fs::read_dir(&farmer_demo_dir)
        .map_err(|e| format!("Failed to read farmer demo dir: {:?}", e))?;

    for class_entry in class_dirs.filter_map(|e| e.ok()) {
        if !class_entry.path().is_dir() {
            continue;
        }

        let class_name = class_entry.file_name().to_string_lossy().to_string();
        let label = class_to_label
            .get(&class_name)
            .copied()
            .unwrap_or(0); // Default to 0 if class not found

        let image_files = std::fs::read_dir(class_entry.path())
            .map_err(|e| format!("Failed to read class dir: {:?}", e))?;

        for img_entry in image_files.filter_map(|e| e.ok()) {
            let path = img_entry.path();
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if ext_lower == "jpg" || ext_lower == "jpeg" || ext_lower == "png" {
                    images.push(HiddenLabelImage {
                        image_path: path,
                        hidden_label: label,
                        hidden_class_name: class_name.clone(),
                        image_id,
                    });
                    image_id += 1;
                }
            }
        }
    }

    tracing::info!("Loaded {} farmer demo images", images.len());
    Ok(images)
}

/// Process farmer demo images (simulating a farmer uploading images)
#[tauri::command]
pub async fn process_farmer_images(
    demo_session: State<'_, DemoSessionGlobal>,
    app: AppHandle,
) -> Result<FarmerImportResult, String> {
    use crate::backend::AdaptiveBackend;
    use std::time::Instant;

    let total_start = Instant::now();
    tracing::info!("=== process_farmer_images called ===");

    let _ = app.emit(
        "demo:progress",
        serde_json::json!({"step": "farmer_import", "message": "Processing farmer images..."}),
    );

    // Take session for processing
    let mut session_opt = {
        let mut global = demo_session.lock().await;
        global.take()
    };

    let session = session_opt
        .as_mut()
        .ok_or("No demo session initialized. Please start the demo first.")?;

    // Load farmer demo images
    let farmer_images = load_farmer_demo_images(&session.state.config.data_dir)?;

    if farmer_images.is_empty() {
        // Restore session
        let mut global = demo_session.lock().await;
        *global = session_opt;
        return Err("No farmer demo images found".to_string());
    }

    tracing::info!(
        "Processing {} farmer images...",
        farmer_images.len()
    );

    // Run inference on farmer images
    let inference_start = Instant::now();
    let predictions = run_inference_on_images::<AdaptiveBackend>(
        &session.model,
        &farmer_images,
        &session.batcher,
        &session.device,
    );
    tracing::info!(
        "Farmer inference completed in {:?}, got {} predictions",
        inference_start.elapsed(),
        predictions.len()
    );

    // Process predictions for pseudo-labeling
    let new_pseudo_labels = session
        .pseudo_labeler
        .process_predictions(&predictions, &farmer_images);

    let images_processed = farmer_images.len();
    let pseudo_labels_accepted = new_pseudo_labels.len();

    session.state.images_processed += images_processed;
    session.state.total_pseudo_labels_generated += pseudo_labels_accepted;
    session.state.pseudo_labels_accumulated += pseudo_labels_accepted;

    // Update precision stats
    let stats = session.pseudo_labeler.stats();
    session.state.pseudo_label_precision = stats.accuracy();

    tracing::info!(
        "Farmer images: {} processed, {} pseudo-labels accepted, {} accumulated",
        images_processed,
        pseudo_labels_accepted,
        session.state.pseudo_labels_accumulated
    );

    // Create thumbnails for all farmer images (show them all since there's only ~50)
    let sample_images: Vec<DayImage> = predictions
        .iter()
        .zip(farmer_images.iter())
        .map(|(pred, hidden)| {
            let accepted = new_pseudo_labels
                .iter()
                .any(|pl| pl.image_path == pred.image_path);
            let base64_thumbnail = encode_image_thumbnail(&pred.image_path);
            DayImage {
                path: pred.image_path.to_string_lossy().to_string(),
                predicted_label: pred.predicted_label,
                confidence: pred.confidence,
                accepted,
                ground_truth: hidden.hidden_label,
                is_correct: pred.predicted_label == hidden.hidden_label,
                base64_thumbnail,
                is_farmer_image: true,
            }
        })
        .collect();

    let result = FarmerImportResult {
        images_processed,
        pseudo_labels_accepted,
        pseudo_labels_accumulated: session.state.pseudo_labels_accumulated,
        sample_images,
        current_accuracy: session.state.current_accuracy,
        pseudo_label_precision: session.state.pseudo_label_precision,
    };

    // Restore session
    {
        let mut global = demo_session.lock().await;
        *global = session_opt;
    }

    // Emit event
    let _ = app.emit("demo:farmer_complete", &result);

    tracing::info!(
        "=== process_farmer_images COMPLETED in {:?} ===",
        total_start.elapsed()
    );
    Ok(result)
}

/// Result of manual retraining
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainResult {
    pub did_retrain: bool,
    pub accuracy_before: f64,
    pub accuracy_after: f64,
    pub pseudo_labels_used: usize,
}

/// Manually trigger retraining with accumulated pseudo-labels
#[tauri::command]
pub async fn manual_retrain_demo(
    demo_session: State<'_, DemoSessionGlobal>,
    app: AppHandle,
) -> Result<RetrainResult, String> {
    use std::time::Instant;

    let total_start = Instant::now();
    tracing::info!("=== manual_retrain_demo called ===");

    let _ = app.emit(
        "demo:progress",
        serde_json::json!({"step": "manual_retrain", "message": "Starting manual retraining..."}),
    );

    // Take session for processing
    let mut session_opt = {
        let mut global = demo_session.lock().await;
        global.take()
    };

    let session = session_opt
        .as_mut()
        .ok_or("No demo session initialized. Please start the demo first.")?;

    // Check if we have enough pseudo-labels
    if !session.pseudo_labeler.should_retrain() {
        let num_labels = session.pseudo_labeler.num_pseudo_labels();
        let threshold = session.pseudo_labeler.retrain_threshold();
        let needed = threshold - num_labels;
        
        // Restore session before returning error
        let mut global = demo_session.lock().await;
        *global = session_opt;
        
        return Err(format!(
            "Not enough pseudo-labels for retraining. Need {} more (current: {}, threshold: {})",
            needed,
            num_labels,
            threshold
        ));
    }

    let accuracy_before = session.state.current_accuracy;

    tracing::info!("Starting manual retraining...");
    let _ = app.emit(
        "demo:retraining_started",
        serde_json::json!({
            "message": "Manual retraining started...",
            "pseudo_labels": session.pseudo_labeler.stats().total_accepted,
        }),
    );

    // Get pseudo-labels and convert to training data
    let pseudo_labels = session.pseudo_labeler.get_and_clear_pseudo_labels();
    let pseudo_labels_used = pseudo_labels.len();
    
    tracing::info!(
        "Collected {} pseudo-labels for manual retraining",
        pseudo_labels_used
    );

    let pseudo_samples: Vec<(PathBuf, usize)> = pseudo_labels
        .iter()
        .map(|p| (p.image_path.clone(), p.predicted_label))
        .collect();

    // Combine with labeled pool
    let labeled_samples: Vec<(PathBuf, usize)> = session
        .splits
        .labeled_pool
        .iter()
        .map(|img| (img.image_path.clone(), img.label))
        .collect();

    tracing::info!(
        "Labeled pool size: {}, pseudo-labels: {}",
        labeled_samples.len(),
        pseudo_samples.len()
    );

    let mut combined_samples = labeled_samples;
    combined_samples.extend(pseudo_samples);

    // Create combined dataset with augmentation
    let combined_dataset = RawPlantVillageDataset::new_cached(combined_samples)
        .map_err(|e| format!("Failed to load combined dataset: {:?}", e))?;

    tracing::info!(
        "Starting retraining with {} total samples, 5 epochs",
        combined_dataset.len()
    );

    // Retrain model (5 epochs for better convergence)
    let retrain_start = Instant::now();
    retrain_model_with_augmentation(
        &mut session.model,
        &mut session.optimizer,
        &combined_dataset,
        5,      // retrain_epochs (5 for stable improvement)
        32,     // batch_size
        0.0001, // learning_rate
        session.state.config.seed + 9999, // Different seed for manual retrain
    );
    tracing::info!("Manual retraining completed in {:?}", retrain_start.elapsed());

    // Evaluate after retraining
    let new_acc = evaluate_model(&session.model, &session.val_dataset, &session.batcher, 32);
    
    session.state.current_accuracy = new_acc;
    session.state.retraining_count += 1;
    session.state.accuracy_history.push((session.state.current_day, new_acc));
    session.state.pseudo_labels_accumulated = 0; // Reset counter after retraining

    let result = RetrainResult {
        did_retrain: true,
        accuracy_before,
        accuracy_after: new_acc,
        pseudo_labels_used,
    };

    // Restore session
    {
        let mut global = demo_session.lock().await;
        *global = session_opt;
    }

    // Emit event
    let _ = app.emit("demo:manual_retrain_complete", &result);

    tracing::info!(
        "=== manual_retrain_demo COMPLETED in {:?} ===",
        total_start.elapsed()
    );
    Ok(result)
}

/// Internal function to initialize session
fn init_session_inner<B>(config: DemoConfig) -> Result<DemoSession<B>, String>
where
    B: AutodiffBackend,
{
    let device = B::Device::default();

    // Load dataset
    let dataset = PlantVillageDataset::new(&config.data_dir)
        .map_err(|e| format!("Failed to load dataset: {:?}", e))?;

    let stats = dataset.get_stats();
    if stats.total_samples == 0 {
        return Err("No images found in dataset".to_string());
    }

    // Create splits
    let all_images: Vec<(PathBuf, usize, String)> = dataset
        .samples
        .iter()
        .map(|s| (s.path.clone(), s.label, s.class_name.clone()))
        .collect();

    let stream_fraction = 1.0 - config.labeled_ratio;
    let split_config = SplitConfig {
        test_fraction: 0.10,
        validation_fraction: 0.10,
        labeled_fraction: config.labeled_ratio,
        stream_fraction,
        seed: config.seed,
        stratified: true,
    };

    let splits = DatasetSplits::from_images(all_images, split_config)
        .map_err(|e| format!("Failed to create splits: {:?}", e))?;

    let unlabeled_pool = splits.stream_pool.clone();

    // Load or create model
    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    let mut model: PlantClassifier<B> = PlantClassifier::new(&model_config, &device);

    // Try to load checkpoint
    let model_path = Path::new(&config.model_path);
    let model_path_mpk = PathBuf::from(format!("{}.mpk", config.model_path));

    let actual_model_path = if model_path_mpk.exists() {
        Some(model_path.to_path_buf())
    } else if model_path.exists() {
        Some(model_path.to_path_buf())
    } else {
        None
    };

    if let Some(load_path) = actual_model_path {
        let recorder = CompactRecorder::new();
        model = model
            .load_file(&load_path, &recorder, &device)
            .map_err(|e| format!("Failed to load model: {:?}", e))?;
    } else {
        return Err(format!("No model found at {:?}", model_path));
    }

    // Create optimizer
    let optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    // Create stream simulator
    let stream_simulator = StreamSimulator::new(unlabeled_pool, config.seed, config.images_per_day);

    // Load farmer demo images if available
    let farmer_simulator = match load_farmer_demo_images(&config.data_dir) {
        Ok(farmer_images) if !farmer_images.is_empty() => {
            tracing::info!("Loaded {} farmer demo images for day-by-day processing", farmer_images.len());
            Some(StreamSimulator::new(farmer_images, config.seed + 1000, config.images_per_day))
        }
        Ok(_) => {
            tracing::warn!("No farmer demo images found");
            None
        }
        Err(e) => {
            tracing::warn!("Failed to load farmer images: {}. Continuing without farmer images.", e);
            None
        }
    };

    // Create pseudo-labeler
    let pseudo_config = PseudoLabelConfig {
        confidence_threshold: config.confidence_threshold,
        max_per_class: Some(500),
        retrain_threshold: config.retrain_threshold,
        curriculum_learning: false,
        ..Default::default()
    };
    let pseudo_labeler = PseudoLabeler::new(pseudo_config);

    // Create batcher
    let batcher = PlantVillageBatcher::<B::InnerBackend>::with_image_size(
        <B::InnerBackend as burn::tensor::backend::Backend>::Device::default(),
        128,
    );

    // Prepare validation dataset
    let val_samples: Vec<(PathBuf, usize)> = splits
        .validation_set
        .iter()
        .map(|img| (img.image_path.clone(), img.label))
        .collect();
    let val_dataset = PlantVillageBurnDataset::new_cached(val_samples.clone(), 128)
        .map_err(|e| format!("Failed to load validation dataset: {:?}", e))?;

    // Initial validation accuracy
    let initial_accuracy = evaluate_model(&model, &val_dataset, &batcher, 32);

    let state = DemoSessionState {
        current_day: 0,
        total_images_available: stream_simulator.total(),
        images_processed: 0,
        pseudo_labels_accumulated: 0,
        total_pseudo_labels_generated: 0,
        retraining_count: 0,
        current_accuracy: initial_accuracy,
        initial_accuracy,
        pseudo_label_precision: 0.0,
        accuracy_history: vec![(0, initial_accuracy)],
        config,
    };

    Ok(DemoSession {
        model,
        optimizer,
        stream_simulator,
        farmer_simulator,
        pseudo_labeler,
        splits,
        val_dataset,
        batcher,
        state,
        device,
    })
}

/// Process a single day in the simulation
fn process_demo_day<B: AutodiffBackend>(session: &mut DemoSession<B>) -> Result<DayResult, String> {
    use std::time::Instant;
    
    let day_start = Instant::now();
    tracing::info!("  [process_demo_day] START");
    
    // Try farmer images first, then fall back to stream
    let (daily_images, is_farmer_batch) = if let Some(ref mut farmer_sim) = session.farmer_simulator {
        if let Some(images) = farmer_sim.next_day() {
            tracing::info!("  [process_demo_day] Processing {} FARMER images", images.len());
            (images, true)
        } else {
            // No more farmer images, use regular stream
            tracing::info!("  [process_demo_day] No more farmer images, using stream");
            let stream_images = session
                .stream_simulator
                .next_day()
                .ok_or("No more images available in stream")?;
            tracing::info!("  [process_demo_day] Got {} stream images", stream_images.len());
            (stream_images, false)
        }
    } else {
        // No farmer simulator, use regular stream
        let stream_images = session
            .stream_simulator
            .next_day()
            .ok_or("No more images available in stream")?;
        tracing::info!("  [process_demo_day] Got {} stream images", stream_images.len());
        (stream_images, false)
    };

    session.state.current_day += 1;
    let day = session.state.current_day;
    tracing::info!("  [process_demo_day] Day {} starting ({})", day, if is_farmer_batch { "FARMER" } else { "STREAM" });

    session.pseudo_labeler.set_day(day);

    // Run inference on daily images
    tracing::info!("  [process_demo_day] Running inference on {} images...", daily_images.len());
    let inference_start = Instant::now();
    let predictions = run_inference_on_images(
        &session.model,
        &daily_images,
        &session.batcher,
        &session.device,
    );
    tracing::info!("  [process_demo_day] Inference completed in {:?}, got {} predictions", inference_start.elapsed(), predictions.len());

    // Process predictions for pseudo-labeling
    tracing::info!("  [process_demo_day] Processing predictions for pseudo-labeling...");
    let new_pseudo_labels = session
        .pseudo_labeler
        .process_predictions(&predictions, &daily_images);
    tracing::info!("  [process_demo_day] {} new pseudo-labels accepted", new_pseudo_labels.len());

    let images_processed_today = daily_images.len();
    let pseudo_labels_accepted_today = new_pseudo_labels.len();

    session.state.images_processed += images_processed_today;
    session.state.total_pseudo_labels_generated += pseudo_labels_accepted_today;
    session.state.pseudo_labels_accumulated += pseudo_labels_accepted_today;

    // Update precision stats
    let stats = session.pseudo_labeler.stats();
    session.state.pseudo_label_precision = stats.accuracy();
    tracing::info!("  [process_demo_day] Accumulated pseudo-labels: {}", session.state.pseudo_labels_accumulated);

    // Sample images for display (show first 20)
    tracing::info!("  [process_demo_day] Creating thumbnails for display...");
    let thumbnail_start = Instant::now();
    let sample_images: Vec<DayImage> = predictions
        .iter()
        .zip(daily_images.iter())
        .take(20)
        .map(|(pred, hidden)| {
            let accepted = new_pseudo_labels
                .iter()
                .any(|pl| pl.image_path == pred.image_path);
            let base64_thumbnail = encode_image_thumbnail(&pred.image_path);
            DayImage {
                path: pred.image_path.to_string_lossy().to_string(),
                predicted_label: pred.predicted_label,
                confidence: pred.confidence,
                accepted,
                ground_truth: hidden.hidden_label,
                is_correct: pred.predicted_label == hidden.hidden_label,
                base64_thumbnail,
                is_farmer_image: is_farmer_batch, // Mark if this batch is from farmer
            }
        })
        .collect();
    tracing::info!("  [process_demo_day] Thumbnails created in {:?}", thumbnail_start.elapsed());

    // Check if we should retrain
    let should_retrain = session.pseudo_labeler.should_retrain();
    tracing::info!("  [process_demo_day] Should retrain? {} (threshold check)", should_retrain);
    
    let mut did_retrain = false;
    let mut accuracy_before_retrain = None;
    let mut accuracy_after_retrain = None;

    if should_retrain {
        did_retrain = true;
        accuracy_before_retrain = Some(session.state.current_accuracy);

        tracing::info!("  [process_demo_day] >>> RETRAINING STARTING at day {}", day);

        // Get pseudo-labels and convert to training data
        let pseudo_labels = session.pseudo_labeler.get_and_clear_pseudo_labels();
        tracing::info!(
            "Collected {} pseudo-labels for retraining",
            pseudo_labels.len()
        );

        let pseudo_samples: Vec<(PathBuf, usize)> = pseudo_labels
            .iter()
            .map(|p| (p.image_path.clone(), p.predicted_label))
            .collect();

        // Combine with labeled pool
        let labeled_samples: Vec<(PathBuf, usize)> = session
            .splits
            .labeled_pool
            .iter()
            .map(|img| (img.image_path.clone(), img.label))
            .collect();

        tracing::info!(
            "Labeled pool size: {}, pseudo-labels: {}",
            labeled_samples.len(),
            pseudo_samples.len()
        );

        let mut combined_samples = labeled_samples;
        combined_samples.extend(pseudo_samples);

        // Create combined dataset with augmentation
        let combined_dataset = RawPlantVillageDataset::new_cached(combined_samples)
            .map_err(|e| format!("Failed to load combined dataset: {:?}", e))?;

        tracing::info!(
            "Starting retraining with {} total samples, 5 epochs",
            combined_dataset.len()
        );

        // Retrain model (5 epochs for better convergence)
        let retrain_start = Instant::now();
        retrain_model_with_augmentation(
            &mut session.model,
            &mut session.optimizer,
            &combined_dataset,
            5,      // retrain_epochs (5 for stable improvement)
            32,     // batch_size
            0.0001, // learning_rate
            session.state.config.seed + day as u64,
        );
        tracing::info!("  [process_demo_day] Retraining completed in {:?}", retrain_start.elapsed());

        tracing::info!("  [process_demo_day] Evaluating model...");
        let eval_start = Instant::now();

        // Evaluate after retraining
        let new_acc = evaluate_model(&session.model, &session.val_dataset, &session.batcher, 32);
        tracing::info!("  [process_demo_day] Evaluation completed in {:?}", eval_start.elapsed());
        
        session.state.current_accuracy = new_acc;
        session.state.retraining_count += 1;
        session.state.accuracy_history.push((day, new_acc));
        session.state.pseudo_labels_accumulated = 0; // Reset counter after retraining

        accuracy_after_retrain = Some(new_acc);

        tracing::info!("  [process_demo_day] >>> RETRAINING COMPLETE! New accuracy: {:.2}%", new_acc);
    }

    tracing::info!("  [process_demo_day] END - total time: {:?}", day_start.elapsed());
    
    // Calculate total remaining images (farmer + stream)
    let farmer_remaining = session.farmer_simulator.as_ref().map(|s| s.remaining()).unwrap_or(0);
    let stream_remaining = session.stream_simulator.remaining();
    let total_remaining = farmer_remaining + stream_remaining;
    
    Ok(DayResult {
        day,
        images_processed_today,
        pseudo_labels_accepted_today,
        pseudo_labels_accumulated: session.state.pseudo_labels_accumulated,
        did_retrain,
        accuracy_before_retrain,
        accuracy_after_retrain,
        current_accuracy: session.state.current_accuracy,
        pseudo_label_precision: session.state.pseudo_label_precision,
        sample_images,
        remaining_images: total_remaining,
    })
}

/// Run inference on a batch of hidden-label images
fn run_inference_on_images<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    images: &[HiddenLabelImage],
    batcher: &PlantVillageBatcher<B::InnerBackend>,
    _device: &B::Device,
) -> Vec<Prediction> {
    const INFERENCE_BATCH_SIZE: usize = 32;
    let inner_model = model.clone().valid();
    let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
    let mut predictions = Vec::new();
    let num_classes = 38;

    for chunk_start in (0..images.len()).step_by(INFERENCE_BATCH_SIZE) {
        let chunk_end = (chunk_start + INFERENCE_BATCH_SIZE).min(images.len());
        let chunk_images = &images[chunk_start..chunk_end];

        let mut loaded_items: Vec<(usize, PlantVillageItem)> = Vec::new();
        for (idx, hidden) in chunk_images.iter().enumerate() {
            if let Ok(item) =
                PlantVillageItem::from_path(&hidden.image_path, hidden.hidden_label, 128)
            {
                loaded_items.push((idx, item));
            }
        }

        if loaded_items.is_empty() {
            continue;
        }

        let items: Vec<PlantVillageItem> =
            loaded_items.iter().map(|(_, item)| item.clone()).collect();
        let batch = batcher.batch(items, &inner_device);
        let output = inner_model.forward_softmax(batch.images);
        let output_data = output.into_data();
        let probs: Vec<f32> = output_data.to_vec().unwrap();

        for (i, (chunk_idx, _)) in loaded_items.iter().enumerate() {
            let hidden = &chunk_images[*chunk_idx];
            let start = i * num_classes;
            let end = start + num_classes;

            if end > probs.len() {
                break;
            }

            let item_probs: Vec<f32> = probs[start..end].to_vec();

            let (predicted_label, confidence) = item_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, &conf)| (idx, conf))
                .unwrap_or((0, 0.0));

            predictions.push(Prediction {
                image_path: hidden.image_path.clone(),
                predicted_label,
                confidence,
                probabilities: item_probs,
                image_id: hidden.image_id,
                ground_truth: Some(hidden.hidden_label),
            });
        }
    }

    predictions
}

/// Retrain model with augmentation
fn retrain_model_with_augmentation<B: AutodiffBackend>(
    model: &mut PlantClassifier<B>,
    optimizer: &mut burn::optim::adaptor::OptimizerAdaptor<
        burn::optim::Adam,
        PlantClassifier<B>,
        B,
    >,
    dataset: &RawPlantVillageDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
) {
    let device = B::Device::default();
    let aug_batcher = AugmentingBatcher::<B>::new(device.clone(), 128, seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for epoch in 0..epochs {
        let len = dataset.len();
        let mut indices: Vec<usize> = (0..len).collect();
        indices.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        let total_batches = (len + batch_size - 1) / batch_size;
        tracing::info!(
            "Epoch {}/{}: {} samples, {} batches",
            epoch + 1,
            epochs,
            len,
            total_batches
        );

        for start in (0..len).step_by(batch_size) {
            let end = (start + batch_size).min(len);
            let batch_indices = &indices[start..end];
            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = aug_batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());

            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            
            // CRITICAL: Detect NaN/Inf immediately
            if loss_value.is_nan() {
                tracing::error!(
                    "Loss became NaN at epoch {} batch {}. Aborting retrain.",
                    epoch + 1, batch_count + 1
                );
                return; // Abort this retraining cycle
            }
            if loss_value.is_infinite() {
                tracing::error!(
                    "Loss became infinite at epoch {} batch {}. Aborting retrain.",
                    epoch + 1, batch_count + 1
                );
                return; // Abort this retraining cycle
            }
            
            epoch_loss += loss_value;
            batch_count += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &*model);

            *model = optimizer.step(learning_rate, model.clone(), grads);
        }

        let avg_loss = if batch_count > 0 {
            epoch_loss / batch_count as f64
        } else {
            0.0
        };
        tracing::info!(
            "Epoch {}/{} complete: avg loss = {:.4}",
            epoch + 1,
            epochs,
            avg_loss
        );
    }
}

/// Evaluate model on validation set
fn evaluate_model<B: AutodiffBackend>(
    model: &PlantClassifier<B>,
    dataset: &PlantVillageBurnDataset,
    batcher: &PlantVillageBatcher<B::InnerBackend>,
    batch_size: usize,
) -> f64 {
    let inner_model = model.clone().valid();
    let inner_device = <B::InnerBackend as burn::tensor::backend::Backend>::Device::default();
    let len = dataset.len();
    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch = batcher.batch(items, &inner_device);
        let output = inner_model.forward(batch.images);
        let predictions = output.argmax(1).squeeze::<1>();

        let batch_correct: i64 = predictions
            .equal(batch.targets)
            .int()
            .sum()
            .into_scalar()
            .elem();

        correct += batch_correct as usize;
        total += end - start;
    }

    if total == 0 {
        0.0
    } else {
        100.0 * correct as f64 / total as f64
    }
}
