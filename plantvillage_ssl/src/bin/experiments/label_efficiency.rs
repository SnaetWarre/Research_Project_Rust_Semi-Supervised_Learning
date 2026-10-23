//! Experiment 1: label efficiency curve.
//!
//! Trains on 38 classes with a growing number of labeled images per class
//! and reports validation accuracy for each size.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::ElementConversion;
use chrono::Local;
use colored::Colorize;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use plantvillage_ssl::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageBurnDataset};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};
use plantvillage_ssl::training::evaluate_accuracy;
use plantvillage_ssl::PlantVillageDataset;

use super::{experiment_verbose, Backend};

/// Results from label efficiency experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LabelEfficiencyResults {
    /// Images per class tested
    images_per_class: Vec<usize>,
    /// Accuracy for each configuration
    accuracies: Vec<f64>,
    /// Training time in seconds
    training_times: Vec<f64>,
    /// Best accuracy achieved
    best_accuracy: f64,
    /// Images per class that achieved best accuracy
    best_images_per_class: usize,
    /// Minimum images for acceptable accuracy (>80%)
    min_acceptable_images: Option<usize>,
}

/// Run label efficiency experiment
pub(crate) fn run_label_efficiency(
    data_dir: &str,
    output_dir: &str,
    epochs: usize,
    seed: u64,
) -> Result<()> {
    println!("{}", "EXPERIMENT 1: Label Efficiency Curve".yellow().bold());
    println!("Question: How many labeled images per new class are needed for good accuracy?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Test different numbers of images per class
    let images_per_class_tests = vec![5, 10, 25, 50, 100, 200, 500]; // Dataset has ~2K images per class
    let mut results = LabelEfficiencyResults {
        images_per_class: Vec::new(),
        accuracies: Vec::new(),
        training_times: Vec::new(),
        best_accuracy: 0.0,
        best_images_per_class: 0,
        min_acceptable_images: None,
    };

    for &n_images in &images_per_class_tests {
        println!(
            "{}",
            format!("Testing with {} images per class...", n_images)
                .yellow()
                .bold()
        );

        let start = Instant::now();
        let accuracy = train_with_n_images_per_class::<Backend>(&dataset, n_images, epochs, seed)?;
        let training_time = start.elapsed().as_secs_f64();

        results.images_per_class.push(n_images);
        results.accuracies.push(accuracy);
        results.training_times.push(training_time);

        println!(
            "  {} images/class → {:.2}% accuracy ({:.1}s)",
            n_images, accuracy, training_time
        );

        if accuracy > results.best_accuracy {
            results.best_accuracy = accuracy;
            results.best_images_per_class = n_images;
        }

        if results.min_acceptable_images.is_none() && accuracy >= 80.0 {
            results.min_acceptable_images = Some(n_images);
        }

        println!();
    }

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_label_efficiency_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG chart
    generate_label_efficiency_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Train model with n images per class and return validation accuracy
fn train_with_n_images_per_class<B: AutodiffBackend>(
    dataset: &PlantVillageDataset,
    n_images: usize,
    epochs: usize,
    seed: u64,
) -> Result<f64> {
    let device = B::Device::default();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Sample n images per class
    let mut by_class: HashMap<usize, Vec<&plantvillage_ssl::dataset::loader::ImageSample>> =
        HashMap::new();
    for sample in &dataset.samples {
        by_class.entry(sample.label).or_default().push(sample);
    }

    let mut train_samples: Vec<(PathBuf, usize)> = Vec::new();
    let mut val_samples: Vec<(PathBuf, usize)> = Vec::new();

    for (label, samples) in by_class.iter_mut() {
        samples.shuffle(&mut rng);
        let take_n = n_images.min(samples.len());
        let val_n = (take_n as f64 * 0.2).ceil() as usize;
        let train_n = take_n - val_n;

        for s in samples.iter().take(train_n) {
            train_samples.push((s.path.clone(), *label));
        }
        for s in samples.iter().skip(train_n).take(val_n) {
            val_samples.push((s.path.clone(), *label));
        }
    }

    // Train
    let image_size = 128;
    let batch_size = 64;
    let learning_rate = 0.0001;

    let train_dataset = PlantVillageBurnDataset::new(train_samples, image_size);
    let val_dataset = PlantVillageBurnDataset::new(val_samples, image_size);

    let batcher = PlantVillageBatcher::with_image_size(image_size);

    let model_config = PlantClassifierConfig {
        num_classes: 38,
        input_size: image_size,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };
    let mut model = PlantClassifier::<B>::new(&model_config, &device);
    let mut optimizer = AdamConfig::new()
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-4f32)))
        .init();

    let mut epoch_rng = ChaCha8Rng::seed_from_u64(seed);

    for epoch in 0..epochs {
        let mut indices: Vec<usize> = (0..train_dataset.len()).collect();
        indices.shuffle(&mut epoch_rng);
        let num_batches = indices.len().div_ceil(batch_size);

        let mut epoch_loss = 0.0f64;
        let mut train_batches = 0usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = (start + batch_size).min(indices.len());
            let batch_indices = &indices[start..end];

            let items: Vec<_> = batch_indices
                .iter()
                .filter_map(|&i| train_dataset.get(i))
                .collect();

            if items.is_empty() {
                continue;
            }

            let batch = batcher.batch(items, &device);
            let output = model.forward(batch.images.clone());
            let loss = CrossEntropyLossConfig::new()
                .init(&output.device())
                .forward(output, batch.targets);

            let loss_value: f64 = loss.clone().into_scalar().elem();
            epoch_loss += loss_value;
            train_batches += 1;

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(learning_rate, model, grads);
        }

        if experiment_verbose() {
            let avg = if train_batches > 0 {
                epoch_loss / train_batches as f64
            } else {
                0.0
            };
            println!(
                "      [label efficiency, {} img/class] epoch {}/{}  train_loss={:.4}  batches={}",
                n_images,
                epoch + 1,
                epochs,
                avg,
                train_batches
            );
            let _ = io::stdout().flush();
        }
    }

    let val_acc = evaluate_accuracy(&model.valid(), &val_dataset, batch_size, image_size);
    if experiment_verbose() {
        println!(
            "      [label efficiency, {} img/class] validation accuracy: {:.2}%",
            n_images, val_acc
        );
        let _ = io::stdout().flush();
    }

    Ok(val_acc)
}

/// Generate conclusions for label efficiency experiment
fn generate_label_efficiency_conclusions(results: &LabelEfficiencyResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str("========================================================================\n");
    text.push_str("EXPERIMENT 1: Label Efficiency Curve - Conclusions\n");
    text.push_str(&format!("Generated: {}\n", timestamp));
    text.push_str("========================================================================\n\n");

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str("How many labeled images per class are needed for acceptable accuracy?\n\n");

    text.push_str("RESULTS:\n");
    text.push_str(&format!(
        "{:>12} | {:>12} | {:>12}\n",
        "Images/Class", "Accuracy (%)", "Time (s)"
    ));
    text.push_str(&format!("{}\n", "-".repeat(42)));

    for i in 0..results.images_per_class.len() {
        text.push_str(&format!(
            "{:>12} | {:>12.2} | {:>12.1}\n",
            results.images_per_class[i], results.accuracies[i], results.training_times[i]
        ));
    }

    text.push_str("\nKEY FINDINGS:\n");
    text.push_str(&format!(
        "1. Best accuracy: {:.2}% with {} images per class\n",
        results.best_accuracy, results.best_images_per_class
    ));

    if let Some(min) = results.min_acceptable_images {
        text.push_str(&format!(
            "2. Minimum for >80% accuracy: {} images per class\n",
            min
        ));
    } else {
        text.push_str("2. No configuration achieved >80% accuracy\n");
    }

    // Calculate efficiency gain
    if results.accuracies.len() >= 2 {
        let first_acc = results.accuracies[0];
        let last_acc = results.accuracies[results.accuracies.len() - 1];
        let improvement = last_acc - first_acc;
        text.push_str(&format!(
            "3. Improvement from {} to {} images: {:.2}% points\n",
            results.images_per_class[0],
            results.images_per_class[results.images_per_class.len() - 1],
            improvement
        ));
    }

    text
}

/// Generate SVG chart for label efficiency results
fn generate_label_efficiency_chart(
    results: &LabelEfficiencyResults,
    output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::utils::charts::{
        generate_bar_chart, generate_line_chart, BarData, DataPoint, DataSeries,
    };

    // Line chart: accuracy vs images per class
    let series = vec![DataSeries {
        name: "Accuracy".to_string(),
        points: results
            .images_per_class
            .iter()
            .zip(results.accuracies.iter())
            .map(|(&x, &y)| DataPoint {
                x: x as f64,
                y,
                label: Some(format!("{:.1}%", y)),
            })
            .collect(),
        color: "#3498db".to_string(),
    }];

    let chart_path = Path::new(output_dir).join("label_efficiency_curve.svg");
    generate_line_chart(
        "Label Efficiency: Accuracy vs Images per Class",
        "Images per Class",
        "Validation Accuracy (%)",
        &series,
        &chart_path,
    )?;
    println!("Chart saved to: {:?}", chart_path);

    // Bar chart version
    let bars: Vec<BarData> = results
        .images_per_class
        .iter()
        .zip(results.accuracies.iter())
        .map(|(&images, &acc)| BarData {
            label: format!("{}", images),
            value: acc,
            color: if acc >= 80.0 {
                "#2ecc71".to_string()
            } else {
                "#3498db".to_string()
            },
        })
        .collect();

    let bar_chart_path = Path::new(output_dir).join("label_efficiency_bars.svg");
    generate_bar_chart(
        "Accuracy by Number of Labeled Images",
        "Validation Accuracy (%)",
        &bars,
        &bar_chart_path,
    )?;
    println!("Bar chart saved to: {:?}", bar_chart_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_efficiency_text_has_no_canned_advice() {
        let results = LabelEfficiencyResults {
            images_per_class: vec![10, 50],
            accuracies: vec![40.0, 60.0],
            training_times: vec![1.0, 2.0],
            best_accuracy: 60.0,
            best_images_per_class: 50,
            min_acceptable_images: None,
        };
        let text = generate_label_efficiency_conclusions(&results);
        assert!(text.contains("No configuration achieved >80% accuracy"));
        assert!(!text.contains("semi-supervised"));
        assert!(!text.contains("SSL"));
    }
}
