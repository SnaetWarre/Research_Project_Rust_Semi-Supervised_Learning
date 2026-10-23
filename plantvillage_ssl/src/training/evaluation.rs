//! Accuracy evaluation shared by training, simulation and experiments.
//!
//! This module provides:
//! - `evaluate_accuracy`: top-1 accuracy of a classifier on a dataset

use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;

use crate::dataset::burn_dataset::{PlantVillageBatcher, PlantVillageItem};
use crate::model::cnn::PlantClassifier;

/// Top-1 accuracy (in percent) of `model` on `dataset`.
///
/// Pass an inference model, e.g. `&model.valid()` for an autodiff model.
/// Items that fail to load are skipped and do not count towards the total.
/// Returns 0.0 when no item could be loaded.
pub fn evaluate_accuracy<B: Backend>(
    model: &PlantClassifier<B>,
    dataset: &impl Dataset<PlantVillageItem>,
    batch_size: usize,
    image_size: usize,
) -> f64 {
    let device = Device::<B>::default();
    let batcher = PlantVillageBatcher::<B>::with_image_size(device.clone(), image_size);
    let len = dataset.len();

    let mut correct = 0usize;
    let mut total = 0usize;

    for start in (0..len).step_by(batch_size) {
        let end = (start + batch_size).min(len);
        let items: Vec<_> = (start..end).filter_map(|i| dataset.get(i)).collect();

        if items.is_empty() {
            continue;
        }

        let batch_len = items.len();
        let batch = batcher.batch(items, &device);
        let predictions: Tensor<B, 1, Int> =
            model.forward(batch.images).argmax(1).reshape([batch_len]);

        let batch_correct: i64 = predictions
            .equal(batch.targets)
            .int()
            .sum()
            .into_scalar()
            .elem();

        correct += batch_correct as usize;
        total += batch_len;
    }

    if total == 0 {
        0.0
    } else {
        100.0 * correct as f64 / total as f64
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
}
