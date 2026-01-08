//! Evaluation metrics for the plant incremental learning project.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    /// Overall accuracy
    pub accuracy: f64,
    /// Top-5 accuracy
    pub top5_accuracy: f64,
    /// Per-class precision
    pub per_class_precision: HashMap<usize, f64>,
    /// Per-class recall
    pub per_class_recall: HashMap<usize, f64>,
    /// Per-class F1 score
    pub per_class_f1: HashMap<usize, f64>,
    /// Confusion matrix (actual x predicted)
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Average inference time in milliseconds
    pub inference_time_ms: f64,
    /// Total number of samples evaluated
    pub num_samples: usize,
}

impl EvaluationMetrics {
    /// Creates a new evaluation metrics instance
    pub fn new(num_classes: usize) -> Self {
        Self {
            accuracy: 0.0,
            top5_accuracy: 0.0,
            per_class_precision: HashMap::new(),
            per_class_recall: HashMap::new(),
            per_class_f1: HashMap::new(),
            confusion_matrix: vec![vec![0; num_classes]; num_classes],
            inference_time_ms: 0.0,
            num_samples: 0,
        }
    }

    /// Calculates macro-averaged precision
    pub fn macro_precision(&self) -> f64 {
        if self.per_class_precision.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.per_class_precision.values().sum();
        sum / self.per_class_precision.len() as f64
    }

    /// Calculates macro-averaged recall
    pub fn macro_recall(&self) -> f64 {
        if self.per_class_recall.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.per_class_recall.values().sum();
        sum / self.per_class_recall.len() as f64
    }

    /// Calculates macro-averaged F1 score
    pub fn macro_f1(&self) -> f64 {
        if self.per_class_f1.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.per_class_f1.values().sum();
        sum / self.per_class_f1.len() as f64
    }

    /// Updates confusion matrix with a prediction
    pub fn update_confusion_matrix(&mut self, actual: usize, predicted: usize) {
        if actual < self.confusion_matrix.len() && predicted < self.confusion_matrix[0].len() {
            self.confusion_matrix[actual][predicted] += 1;
        }
    }

    /// Computes metrics from confusion matrix
    pub fn compute_from_confusion_matrix(&mut self) {
        let num_classes = self.confusion_matrix.len();
        self.num_samples = self.confusion_matrix.iter().flatten().sum();

        if self.num_samples == 0 {
            return;
        }

        // Calculate overall accuracy
        let correct: usize = (0..num_classes)
            .map(|i| self.confusion_matrix[i][i])
            .sum();
        self.accuracy = correct as f64 / self.num_samples as f64;

        // Calculate per-class metrics
        for class_id in 0..num_classes {
            let true_positives = self.confusion_matrix[class_id][class_id] as f64;
            let false_positives: f64 = (0..num_classes)
                .filter(|&i| i != class_id)
                .map(|i| self.confusion_matrix[i][class_id] as f64)
                .sum();
            let false_negatives: f64 = (0..num_classes)
                .filter(|&i| i != class_id)
                .map(|i| self.confusion_matrix[class_id][i] as f64)
                .sum();

            // Precision
            let precision = if true_positives + false_positives > 0.0 {
                true_positives / (true_positives + false_positives)
            } else {
                0.0
            };
            self.per_class_precision.insert(class_id, precision);

            // Recall
            let recall = if true_positives + false_negatives > 0.0 {
                true_positives / (true_positives + false_negatives)
            } else {
                0.0
            };
            self.per_class_recall.insert(class_id, recall);

            // F1 Score
            let f1 = if precision + recall > 0.0 {
                2.0 * (precision * recall) / (precision + recall)
            } else {
                0.0
            };
            self.per_class_f1.insert(class_id, f1);
        }
    }
}

/// Metrics specific to incremental learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalMetrics {
    /// Accuracy on original classes before incremental learning
    pub original_classes_accuracy_before: f64,
    /// Accuracy on original classes after incremental learning
    pub original_classes_accuracy_after: f64,
    /// Accuracy on new class
    pub new_class_accuracy: f64,
    /// Overall accuracy after incremental learning
    pub overall_accuracy: f64,
    /// Catastrophic forgetting measure (drop in original classes accuracy)
    pub catastrophic_forgetting: f64,
    /// Per-class accuracy changes
    pub per_class_accuracy_change: HashMap<usize, f64>,
    /// Training time in seconds
    pub training_time_seconds: f64,
}

impl IncrementalMetrics {
    /// Creates a new incremental metrics instance
    pub fn new() -> Self {
        Self {
            original_classes_accuracy_before: 0.0,
            original_classes_accuracy_after: 0.0,
            new_class_accuracy: 0.0,
            overall_accuracy: 0.0,
            catastrophic_forgetting: 0.0,
            per_class_accuracy_change: HashMap::new(),
            training_time_seconds: 0.0,
        }
    }

    /// Computes catastrophic forgetting metric
    pub fn compute_forgetting(&mut self) {
        self.catastrophic_forgetting =
            self.original_classes_accuracy_before - self.original_classes_accuracy_after;
    }

    /// Checks if forgetting occurred (accuracy dropped)
    pub fn has_forgetting(&self) -> bool {
        self.catastrophic_forgetting > 0.0
    }

    /// Gets forgetting percentage
    pub fn forgetting_percentage(&self) -> f64 {
        if self.original_classes_accuracy_before > 0.0 {
            (self.catastrophic_forgetting / self.original_classes_accuracy_before) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for IncrementalMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Training metrics tracked during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_loss: Vec<f64>,
    /// Validation loss history
    pub val_loss: Vec<f64>,
    /// Training accuracy history
    pub train_accuracy: Vec<f64>,
    /// Validation accuracy history
    pub val_accuracy: Vec<f64>,
    /// Learning rate history
    pub learning_rate: Vec<f64>,
    /// Epoch numbers
    pub epochs: Vec<usize>,
}

impl TrainingMetrics {
    /// Creates a new training metrics instance
    pub fn new() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_accuracy: Vec::new(),
            val_accuracy: Vec::new(),
            learning_rate: Vec::new(),
            epochs: Vec::new(),
        }
    }

    /// Adds metrics for an epoch
    pub fn add_epoch(
        &mut self,
        epoch: usize,
        train_loss: f64,
        val_loss: f64,
        train_acc: f64,
        val_acc: f64,
        lr: f64,
    ) {
        self.epochs.push(epoch);
        self.train_loss.push(train_loss);
        self.val_loss.push(val_loss);
        self.train_accuracy.push(train_acc);
        self.val_accuracy.push(val_acc);
        self.learning_rate.push(lr);
    }

    /// Gets the best validation accuracy
    pub fn best_val_accuracy(&self) -> Option<f64> {
        self.val_accuracy.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Gets the epoch with best validation accuracy
    pub fn best_epoch(&self) -> Option<usize> {
        self.val_accuracy
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| self.epochs[idx])
    }

    /// Checks if training has converged (validation loss not improving)
    pub fn has_converged(&self, patience: usize) -> bool {
        if self.val_loss.len() < patience {
            return false;
        }

        let recent = &self.val_loss[self.val_loss.len() - patience..];
        let min_recent = recent.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap());

        if let Some(min_val) = min_recent {
            // Check if minimum is at the beginning of the window (no improvement)
            min_val == recent[0]
        } else {
            false
        }
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Prediction result from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted class ID
    pub class_id: usize,
    /// Confidence score (probability)
    pub confidence: f64,
    /// Class name (optional)
    pub class_name: Option<String>,
    /// Top-k predictions
    pub top_k: Vec<(usize, f64)>,
}

impl PredictionResult {
    /// Creates a new prediction result
    pub fn new(class_id: usize, confidence: f64) -> Self {
        Self {
            class_id,
            confidence,
            class_name: None,
            top_k: vec![(class_id, confidence)],
        }
    }

    /// Creates a prediction result with class name
    pub fn with_class_name(class_id: usize, confidence: f64, class_name: String) -> Self {
        Self {
            class_id,
            confidence,
            class_name: Some(class_name),
            top_k: vec![(class_id, confidence)],
        }
    }

    /// Sets top-k predictions
    pub fn with_top_k(mut self, top_k: Vec<(usize, f64)>) -> Self {
        self.top_k = top_k;
        self
    }
}

/// Batch of prediction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionResults {
    /// Individual predictions
    pub predictions: Vec<PredictionResult>,
    /// Average confidence
    pub avg_confidence: f64,
    /// Total inference time in milliseconds
    pub total_time_ms: f64,
}

impl BatchPredictionResults {
    /// Creates a new batch prediction results
    pub fn new(predictions: Vec<PredictionResult>, total_time_ms: f64) -> Self {
        let avg_confidence = if !predictions.is_empty() {
            predictions.iter().map(|p| p.confidence).sum::<f64>() / predictions.len() as f64
        } else {
            0.0
        };

        Self {
            predictions,
            avg_confidence,
            total_time_ms,
        }
    }

    /// Gets average inference time per sample
    pub fn avg_time_per_sample(&self) -> f64 {
        if !self.predictions.is_empty() {
            self.total_time_ms / self.predictions.len() as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_metrics_creation() {
        let metrics = EvaluationMetrics::new(5);
        assert_eq!(metrics.confusion_matrix.len(), 5);
        assert_eq!(metrics.accuracy, 0.0);
        assert_eq!(metrics.num_samples, 0);
    }

    #[test]
    fn test_confusion_matrix_update() {
        let mut metrics = EvaluationMetrics::new(3);
        metrics.update_confusion_matrix(0, 0);
        metrics.update_confusion_matrix(0, 1);
        metrics.update_confusion_matrix(1, 1);

        assert_eq!(metrics.confusion_matrix[0][0], 1);
        assert_eq!(metrics.confusion_matrix[0][1], 1);
        assert_eq!(metrics.confusion_matrix[1][1], 1);
    }

    #[test]
    fn test_compute_from_confusion_matrix() {
        let mut metrics = EvaluationMetrics::new(2);
        // Perfect predictions for class 0, 2/3 correct for class 1
        metrics.confusion_matrix = vec![
            vec![2, 0],  // Class 0: 2 correct, 0 wrong
            vec![1, 2],  // Class 1: 1 wrong as class 0, 2 correct
        ];

        metrics.compute_from_confusion_matrix();

        assert_eq!(metrics.num_samples, 5);
        assert_eq!(metrics.accuracy, 0.8); // 4/5 correct
        assert!(metrics.per_class_precision.contains_key(&0));
        assert!(metrics.per_class_recall.contains_key(&0));
    }

    #[test]
    fn test_incremental_metrics() {
        let mut metrics = IncrementalMetrics::new();
        metrics.original_classes_accuracy_before = 0.95;
        metrics.original_classes_accuracy_after = 0.85;
        metrics.compute_forgetting();

        assert!((metrics.catastrophic_forgetting - 0.10).abs() < 1e-10);
        assert!(metrics.has_forgetting());
        assert!((metrics.forgetting_percentage() - 10.526).abs() < 0.01);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();
        metrics.add_epoch(0, 1.0, 0.9, 0.7, 0.75, 0.001);
        metrics.add_epoch(1, 0.8, 0.85, 0.75, 0.78, 0.001);
        metrics.add_epoch(2, 0.6, 0.82, 0.8, 0.80, 0.001);

        assert_eq!(metrics.epochs.len(), 3);
        assert_eq!(metrics.best_val_accuracy(), Some(0.80));
        assert_eq!(metrics.best_epoch(), Some(2));
    }

    #[test]
    fn test_prediction_result() {
        let pred = PredictionResult::new(2, 0.95);
        assert_eq!(pred.class_id, 2);
        assert_eq!(pred.confidence, 0.95);
        assert!(pred.class_name.is_none());
    }

    #[test]
    fn test_batch_prediction_results() {
        let predictions = vec![
            PredictionResult::new(0, 0.9),
            PredictionResult::new(1, 0.8),
            PredictionResult::new(2, 0.7),
        ];

        let batch = BatchPredictionResults::new(predictions, 30.0);
        assert_eq!(batch.predictions.len(), 3);
        assert!((batch.avg_confidence - 0.8).abs() < 0.01);
        assert_eq!(batch.avg_time_per_sample(), 10.0);
    }

    #[test]
    fn test_macro_metrics() {
        let mut metrics = EvaluationMetrics::new(3);
        metrics.per_class_precision.insert(0, 0.9);
        metrics.per_class_precision.insert(1, 0.8);
        metrics.per_class_precision.insert(2, 0.7);

        assert!((metrics.macro_precision() - 0.8).abs() < 0.01);
    }
}
