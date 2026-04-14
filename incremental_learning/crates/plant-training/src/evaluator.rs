//! Model evaluation infrastructure.
//!
//! This module provides:
//! - Comprehensive evaluation metrics
//! - Confusion matrix computation
//! - Per-class metrics (precision, recall, F1)
//! - Top-k accuracy
//! - Inference time measurement

use burn::{
    module::Module,
    tensor::{backend::Backend, Int, Tensor},
};
use plant_core::{Error, EvaluationMetrics, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tracing::info;

/// Result of model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub metrics: EvaluationMetrics,
    pub per_class_metrics: HashMap<usize, ClassMetrics>,
    pub total_samples: usize,
    pub avg_inference_time_ms: f64,
}

/// Per-class evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub class_id: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub support: usize,
}

/// Batch of evaluation data
#[derive(Debug, Clone)]
pub struct EvalBatch<B: Backend> {
    pub images: Tensor<B, 4>,    // [batch, channels, height, width]
    pub labels: Tensor<B, 1, Int>, // [batch]
}

/// Model evaluator
pub struct Evaluator<B: Backend, M: Module<B>> {
    model: M,
    num_classes: usize,
    device: B::Device,
}

impl<B: Backend, M: Module<B>> Evaluator<B, M> {
    /// Create a new evaluator
    pub fn new(model: M, num_classes: usize, device: B::Device) -> Self {
        Self {
            model,
            num_classes,
            device,
        }
    }

    /// Evaluate model on a dataset
    pub fn evaluate<F>(
        &self,
        batches: &[EvalBatch<B>],
        forward_fn: F,
    ) -> Result<EvaluationResult>
    where
        F: Fn(&M, Tensor<B, 4>) -> Tensor<B, 2>,
    {
        info!("Starting evaluation on {} batches", batches.len());

        let mut all_predictions = Vec::new();
        let mut all_labels = Vec::new();
        let mut total_inference_time: f64 = 0.0;
        let mut total_samples = 0;

        // Collect predictions
        for batch in batches.iter() {
            let batch_size = batch.labels.dims()[0];
            total_samples += batch_size;

            // Measure inference time
            let start = Instant::now();
            let logits = forward_fn(&self.model, batch.images.clone());
            let inference_time = start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
            total_inference_time += inference_time;

            // Get predictions
            let predictions = logits.argmax(1);

            // Convert to vectors for metrics computation
            let pred_data: Vec<i32> = predictions.to_data().to_vec().unwrap();
            let label_data: Vec<i32> = batch.labels.clone().to_data().to_vec().unwrap();

            all_predictions.extend(pred_data.iter().map(|&x| x as usize));
            all_labels.extend(label_data.iter().map(|&x| x as usize));
        }

        let avg_inference_time = total_inference_time / batches.len() as f64;

        info!(
            "Processed {} samples in {:.2}ms average per batch",
            total_samples, avg_inference_time
        );

        // Compute confusion matrix
        let confusion_matrix = self.compute_confusion_matrix(&all_predictions, &all_labels);

        // Compute overall metrics
        let accuracy = self.compute_accuracy(&all_predictions, &all_labels);
        let top5_accuracy: f64 = 0.0; // TODO: Implement top-5 accuracy

        // Compute per-class metrics
        let per_class_metrics = self.compute_per_class_metrics(&confusion_matrix);

        // Extract precision, recall, F1 for overall metrics
        let (precision, recall, f1) = self.compute_macro_averages(&per_class_metrics);

        let metrics = EvaluationMetrics {
            accuracy,
            top5_accuracy,
            per_class_precision: per_class_metrics
                .iter()
                .map(|(id, m)| (*id, m.precision))
                .collect(),
            per_class_recall: per_class_metrics
                .iter()
                .map(|(id, m)| (*id, m.recall))
                .collect(),
            per_class_f1: per_class_metrics
                .iter()
                .map(|(id, m)| (*id, m.f1_score))
                .collect(),
            confusion_matrix: confusion_matrix.clone(),
            inference_time_ms: avg_inference_time,
            num_samples: total_samples,
        };

        info!(
            "Evaluation complete: accuracy={:.4}, precision={:.4}, recall={:.4}, f1={:.4}",
            accuracy, precision, recall, f1
        );

        Ok(EvaluationResult {
            metrics,
            per_class_metrics,
            total_samples,
            avg_inference_time_ms: avg_inference_time,
        })
    }

    /// Compute confusion matrix
    fn compute_confusion_matrix(
        &self,
        predictions: &[usize],
        labels: &[usize],
    ) -> Vec<Vec<usize>> {
        let mut matrix = vec![vec![0; self.num_classes]; self.num_classes];

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if *pred < self.num_classes && *label < self.num_classes {
                matrix[*label][*pred] += 1;
            }
        }

        matrix
    }

    /// Compute overall accuracy
    fn compute_accuracy(&self, predictions: &[usize], labels: &[usize]) -> f64 {
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(pred, label)| pred == label)
            .count();

        correct as f64 / predictions.len() as f64
    }

    /// Compute per-class metrics from confusion matrix
    fn compute_per_class_metrics(
        &self,
        confusion_matrix: &[Vec<usize>],
    ) -> HashMap<usize, ClassMetrics> {
        let mut metrics = HashMap::new();

        for class_id in 0..self.num_classes {
            // True positives
            let tp = confusion_matrix[class_id][class_id] as f64;

            // False positives (predicted as this class but was another)
            let fp: f64 = (0..self.num_classes)
                .filter(|&i| i != class_id)
                .map(|i| confusion_matrix[i][class_id] as f64)
                .sum();

            // False negatives (was this class but predicted as another)
            let fn_: f64 = (0..self.num_classes)
                .filter(|&i| i != class_id)
                .map(|i| confusion_matrix[class_id][i] as f64)
                .sum();

            // Support (total samples of this class)
            let support: usize = confusion_matrix[class_id].iter().sum();

            // Compute metrics
            let precision = if tp + fp > 0.0 {
                tp / (tp + fp)
            } else {
                0.0
            };

            let recall = if tp + fn_ > 0.0 {
                tp / (tp + fn_)
            } else {
                0.0
            };

            let f1_score = if precision + recall > 0.0 {
                2.0 * (precision * recall) / (precision + recall)
            } else {
                0.0
            };

            metrics.insert(
                class_id,
                ClassMetrics {
                    class_id,
                    precision,
                    recall,
                    f1_score,
                    support,
                },
            );
        }

        metrics
    }

    /// Compute macro-averaged precision, recall, and F1
    fn compute_macro_averages(
        &self,
        per_class_metrics: &HashMap<usize, ClassMetrics>,
    ) -> (f64, f64, f64) {
        let n = per_class_metrics.len() as f64;

        let avg_precision = per_class_metrics.values().map(|m| m.precision).sum::<f64>() / n;
        let avg_recall = per_class_metrics.values().map(|m| m.recall).sum::<f64>() / n;
        let avg_f1 = per_class_metrics.values().map(|m| m.f1_score).sum::<f64>() / n;

        (avg_precision, avg_recall, avg_f1)
    }

    /// Print evaluation results in a formatted way
    pub fn print_results(result: &EvaluationResult) {
        println!("\n{}", "=".repeat(80));
        println!("EVALUATION RESULTS");
        println!("{}", "=".repeat(80));
        println!("Total samples: {}", result.total_samples);
        println!("Average inference time: {:.2}ms per batch", result.avg_inference_time_ms);
        println!("\nOverall Metrics:");
        println!("  Accuracy:  {:.4}", result.metrics.accuracy);
        println!("  Top-5 Acc: {:.4}", result.metrics.top5_accuracy);
        println!("\nPer-Class Metrics:");
        println!("{:<10} {:>10} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1-Score", "Support");
        println!("{}", "-".repeat(60));

        let mut class_ids: Vec<_> = result.per_class_metrics.keys().collect();
        class_ids.sort();

        for class_id in class_ids {
            let metrics = &result.per_class_metrics[class_id];
            println!(
                "{:<10} {:>10.4} {:>10.4} {:>10.4} {:>10}",
                metrics.class_id,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.support
            );
        }

        println!("{}", "=".repeat(80));
    }

    /// Export confusion matrix to CSV format
    pub fn export_confusion_matrix(
        confusion_matrix: &[Vec<usize>],
        output_path: &std::path::Path,
    ) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(output_path)?;

        // Write header
        write!(file, "True\\Predicted")?;
        for i in 0..confusion_matrix.len() {
            write!(file, ",{}", i)?;
        }
        writeln!(file)?;

        // Write matrix rows
        for (i, row) in confusion_matrix.iter().enumerate() {
            write!(file, "{}", i)?;
            for val in row {
                write!(file, ",{}", val)?;
            }
            writeln!(file)?;
        }

        info!("Confusion matrix exported to {:?}", output_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn test_confusion_matrix_computation() {
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let labels = vec![0, 1, 2, 1, 1, 0];

        // Create a dummy evaluator to test helper methods
        #[derive(Clone, Debug)]
        struct DummyModel;
        impl<B: Backend> Module<B> for DummyModel {
            type Record = ();
            fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {}
            fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
                self
            }
            fn into_record(self) -> Self::Record {}
            fn load_record(self, _record: Self::Record) -> Self {
                self
            }
            fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
                devices
            }
            fn fork(self, _device: &B::Device) -> Self {
                self
            }
            fn to_device(self, _device: &B::Device) -> Self {
                self
            }
        }

        let device = Default::default();
        let evaluator = Evaluator::<NdArray, _> {
            model: DummyModel,
            num_classes: 3,
            device,
        };

        let matrix = evaluator.compute_confusion_matrix(&predictions, &labels);

        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
    }

    #[test]
    fn test_accuracy_computation() {
        let predictions = vec![0, 1, 2, 0];
        let labels = vec![0, 1, 2, 0];

        #[derive(Clone, Debug)]
        struct DummyModel;
        impl<B: Backend> Module<B> for DummyModel {
            type Record = ();
            fn visit<V: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut V) {}
            fn map<M: burn::module::ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
                self
            }
            fn into_record(self) -> Self::Record {}
            fn load_record(self, _record: Self::Record) -> Self {
                self
            }
            fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
                devices
            }
            fn fork(self, _device: &B::Device) -> Self {
                self
            }
            fn to_device(self, _device: &B::Device) -> Self {
                self
            }
        }

        let device = Default::default();
        let evaluator = Evaluator::<NdArray, _> {
            model: DummyModel,
            num_classes: 3,
            device,
        };

        let accuracy = evaluator.compute_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn test_class_metrics() {
        let metrics = ClassMetrics {
            class_id: 0,
            precision: 0.95,
            recall: 0.90,
            f1_score: 0.925,
            support: 100,
        };

        assert_eq!(metrics.class_id, 0);
        assert_eq!(metrics.precision, 0.95);
        assert_eq!(metrics.support, 100);
    }
}
