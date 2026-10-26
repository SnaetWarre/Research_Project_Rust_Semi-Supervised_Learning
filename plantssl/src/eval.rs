//! Batch prediction and classification metrics.

use std::path::PathBuf;

use anyhow::Result;
use burn::prelude::*;
use serde::Serialize;

use crate::images::Loader;
use crate::model::PlantCnn;

/// The most likely class for one image and its softmax probability.
#[derive(Debug, Clone, Copy)]
pub struct Prediction {
    pub label: usize,
    pub confidence: f32,
}

/// Predicts every sample in order. The labels in `samples` are ignored.
pub fn predict(
    model: &PlantCnn,
    samples: &[(PathBuf, usize)],
    image_size: u32,
    batch_size: usize,
    device: &Device,
) -> Result<Vec<Prediction>> {
    let model = model.valid();
    let device = device.clone().without_autodiff();
    let loader = Loader {
        batch_size,
        image_size,
        augment_seed: None,
    };
    let mut predictions = Vec::with_capacity(samples.len());
    loader.for_each(samples, &device, |batch| {
        let probabilities = model.probabilities(batch.images);
        let (confidence, label) = probabilities.max_dim_with_indices(1);
        let confidence: Vec<f32> = confidence
            .into_data()
            .try_to_vec_as::<f32>()
            .expect("f32 tensor");
        let label: Vec<i64> = label
            .into_data()
            .try_to_vec_as::<i64>()
            .expect("int tensor");
        predictions.extend(
            confidence
                .into_iter()
                .zip(label)
                .map(|(confidence, label)| Prediction {
                    label: label as usize,
                    confidence,
                }),
        );
        Ok(())
    })?;
    Ok(predictions)
}

/// Share of predictions that match the labels in `samples`.
pub fn accuracy(predictions: &[Prediction], samples: &[(PathBuf, usize)]) -> f64 {
    let correct = predictions
        .iter()
        .zip(samples)
        .filter(|(p, (_, label))| p.label == *label)
        .count();
    correct as f64 / samples.len().max(1) as f64
}

#[derive(Debug, Clone, Serialize)]
pub struct ClassReport {
    pub class: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub images: usize,
    pub accuracy: f64,
    pub macro_f1: f64,
    pub classes: Vec<ClassReport>,
    /// `confusion[truth][predicted]`
    pub confusion: Vec<Vec<usize>>,
}

impl Report {
    pub fn new(classes: &[String], truth: &[usize], predicted: &[usize]) -> Self {
        let n = classes.len();
        let mut confusion = vec![vec![0; n]; n];
        for (&t, &p) in truth.iter().zip(predicted) {
            confusion[t][p] += 1;
        }

        let per_class: Vec<ClassReport> = (0..n)
            .map(|c| {
                let tp = confusion[c][c] as f64;
                let support: usize = confusion[c].iter().sum();
                let predicted: usize = confusion.iter().map(|row| row[c]).sum();
                let precision = if predicted > 0 {
                    tp / predicted as f64
                } else {
                    0.0
                };
                let recall = if support > 0 {
                    tp / support as f64
                } else {
                    0.0
                };
                let f1 = if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                };
                ClassReport {
                    class: classes[c].clone(),
                    precision,
                    recall,
                    f1,
                    support,
                }
            })
            .collect();

        let correct: usize = (0..n).map(|c| confusion[c][c]).sum();
        let present: Vec<&ClassReport> = per_class.iter().filter(|c| c.support > 0).collect();
        Self {
            images: truth.len(),
            accuracy: correct as f64 / truth.len().max(1) as f64,
            macro_f1: present.iter().map(|c| c.f1).sum::<f64>() / present.len().max(1) as f64,
            classes: per_class,
            confusion,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_matches_hand_computed_values() {
        let classes = vec!["a".to_string(), "b".to_string()];
        // a: 2 right, 1 called b. b: 1 right.
        let report = Report::new(&classes, &[0, 0, 0, 1], &[0, 0, 1, 1]);
        assert_eq!(report.accuracy, 0.75);
        assert_eq!(report.confusion, vec![vec![2, 1], vec![0, 1]]);
        let a = &report.classes[0];
        assert_eq!((a.precision, a.support), (1.0, 3));
        assert!((a.recall - 2.0 / 3.0).abs() < 1e-12);
        let b = &report.classes[1];
        assert_eq!((b.precision, b.recall), (0.5, 1.0));
        let expected_f1 = (0.8 + 2.0 / 3.0) / 2.0;
        assert!((report.macro_f1 - expected_f1).abs() < 1e-12);
    }
}
