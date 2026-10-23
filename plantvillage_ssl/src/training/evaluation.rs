//! Model evaluation shared by training, simulation, experiments and the CLI.
//!
//! This module provides:
//! - `predict_dataset`: top-1 predictions and labels for every loadable item
//! - `evaluate_accuracy`: top-1 accuracy in percent
//! - `evaluate_accuracy_and_macro_f1`: accuracy and macro-F1 in percent

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;

use crate::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageItem};
use crate::model::cnn::PlantClassifier;
use crate::utils::metrics::{ClassMetrics, ConfusionMatrix};

/// Run `model` over `dataset` and return `(predictions, labels)`.
///
/// Pass an inference model, e.g. `&model.valid()` for an autodiff model.
/// Items that fail to load are skipped, so both vectors only cover the
/// items that were actually evaluated.
pub fn predict_dataset<B: Backend>(
    model: &PlantClassifier<B>,
    dataset: &impl Dataset<PlantVillageItem>,
    batch_size: usize,
    image_size: usize,
) -> (Vec<usize>, Vec<usize>) {
    let device = Device::<B>::default();
    let batcher = PlantVillageBatcher::with_image_size(image_size);
    let len = dataset.len();

    let mut predictions = Vec::with_capacity(len);
    let mut labels = Vec::with_capacity(len);

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch_len = items.len();
        labels.extend(items.iter().map(|item| item.label));

        let batch = batcher.batch(items, &device);
        let batch_predictions: Tensor<B, 1, Int> =
            model.forward(batch.images).argmax(1).reshape([batch_len]);
        predictions.extend(
            batch_predictions
                .into_data()
                .iter::<i64>()
                .map(|p| p as usize),
        );
    }

    (predictions, labels)
}

/// Top-1 accuracy (in percent) of `model` on `dataset`.
///
/// Items that fail to load are skipped and do not count towards the total.
/// Returns 0.0 when no item could be loaded.
pub fn evaluate_accuracy<B: Backend>(
    model: &PlantClassifier<B>,
    dataset: &impl Dataset<PlantVillageItem>,
    batch_size: usize,
    image_size: usize,
) -> f64 {
    let (predictions, labels) = predict_dataset(model, dataset, batch_size, image_size);
    accuracy_percent(&predictions, &labels)
}

/// Top-1 accuracy and macro-F1 (both in percent) of `model` on `dataset`.
///
/// Macro-F1 averages over all `num_classes`, so a class with no samples
/// counts as an F1 of 0.
pub fn evaluate_accuracy_and_macro_f1<B: Backend>(
    model: &PlantClassifier<B>,
    dataset: &impl Dataset<PlantVillageItem>,
    batch_size: usize,
    image_size: usize,
    num_classes: usize,
) -> (f64, f64) {
    if num_classes == 0 {
        return (0.0, 0.0);
    }

    let (predictions, labels) = predict_dataset(model, dataset, batch_size, image_size);
    let confusion = ConfusionMatrix::from_predictions(&predictions, &labels, num_classes);
    let f1_sum: f64 = (0..num_classes)
        .map(|class_idx| ClassMetrics::from_confusion_matrix(&confusion, class_idx).f1)
        .sum();

    (
        accuracy_percent(&predictions, &labels),
        100.0 * (f1_sum / num_classes as f64),
    )
}

fn accuracy_percent(predictions: &[usize], labels: &[usize]) -> f64 {
    let correct = predictions
        .iter()
        .zip(labels)
        .filter(|(prediction, label)| prediction == label)
        .count();

    if labels.is_empty() {
        0.0
    } else {
        100.0 * correct as f64 / labels.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use burn_ndarray::NdArray;

    use super::*;
    use crate::dataset::burn_dataset::PlantVillageBurnDataset;
    use crate::model::cnn::PlantClassifierConfig;

    type TestBackend = NdArray;

    const IMAGE_SIZE: usize = 32;

    /// A model with a single class always predicts class 0
    fn single_class_model() -> PlantClassifier<TestBackend> {
        let config = PlantClassifierConfig::new()
            .with_num_classes(1)
            .with_input_size(IMAGE_SIZE)
            .with_base_filters(8);
        PlantClassifier::new(&config, &Default::default())
    }

    fn write_test_images(dir_name: &str, count: usize) -> Vec<PathBuf> {
        let dir = std::env::temp_dir().join(format!("{}_{}", dir_name, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        (0..count)
            .map(|i| {
                let path = dir.join(format!("img_{i}.png"));
                image::RgbImage::new(IMAGE_SIZE as u32, IMAGE_SIZE as u32)
                    .save(&path)
                    .unwrap();
                path
            })
            .collect()
    }

    #[test]
    fn test_unloadable_items_are_not_counted() {
        let model = single_class_model();
        let mut samples: Vec<_> = write_test_images("pv_eval_unloadable", 3)
            .into_iter()
            .map(|p| (p, 0))
            .collect();
        samples.push((PathBuf::from("/nonexistent/a.png"), 0));
        samples.push((PathBuf::from("/nonexistent/b.png"), 0));
        let dataset = PlantVillageBurnDataset::new(samples, IMAGE_SIZE);

        // All 3 loadable images are correct; the 2 missing files must not
        // lower the score (the old code reported 75%).
        let accuracy = evaluate_accuracy(&model, &dataset, 4, IMAGE_SIZE);
        assert_eq!(accuracy, 100.0);
    }

    #[test]
    fn test_no_loadable_items_gives_zero() {
        let model = single_class_model();
        let empty = PlantVillageBurnDataset::new(Vec::new(), IMAGE_SIZE);
        assert_eq!(evaluate_accuracy(&model, &empty, 4, IMAGE_SIZE), 0.0);

        let missing = PlantVillageBurnDataset::new(
            vec![(PathBuf::from("/nonexistent/c.png"), 0)],
            IMAGE_SIZE,
        );
        assert_eq!(evaluate_accuracy(&model, &missing, 4, IMAGE_SIZE), 0.0);
    }

    #[test]
    fn test_macro_f1_counts_every_class() {
        let model = single_class_model();
        // Two images of class 0 (always predicted) and one of class 1
        let paths = write_test_images("pv_eval_macro_f1", 3);
        let samples = vec![
            (paths[0].clone(), 0),
            (paths[1].clone(), 0),
            (paths[2].clone(), 1),
        ];
        let dataset = PlantVillageBurnDataset::new(samples, IMAGE_SIZE);

        // Class 0: precision 2/3, recall 1, F1 0.8. Class 1: F1 0.
        // Class 2 has no samples and also counts as 0.
        let (accuracy, macro_f1) =
            evaluate_accuracy_and_macro_f1(&model, &dataset, 2, IMAGE_SIZE, 3);
        assert!((accuracy - 200.0 / 3.0).abs() < 1e-9);
        assert!((macro_f1 - 80.0 / 3.0).abs() < 1e-9);
    }
}
