//! Metrics Module for Model Evaluation
//!
//! Provides comprehensive metrics for evaluating plant disease classification models:
//! - Accuracy (overall and per-class)
//! - Precision, Recall, F1-score
//! - Confusion Matrix
//! - Top-k accuracy


use serde::{Deserialize, Serialize};

/// Comprehensive metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// Total number of samples evaluated
    pub total_samples: usize,

    /// Number of correct predictions
    pub correct_predictions: usize,

    /// Overall accuracy (correct / total)
    pub accuracy: f64,

    /// Top-5 accuracy
    pub top5_accuracy: f64,

    /// Average loss over all samples
    pub average_loss: f64,

    /// Optional loss value (for trainer compatibility)
    pub loss: Option<f64>,

    /// Macro-averaged precision (average of per-class precisions)
    pub macro_precision: f64,

    /// Macro-averaged recall
    pub macro_recall: f64,

    /// Macro-averaged F1-score
    pub macro_f1: f64,

    /// Weighted F1-score (weighted by class frequency)
    pub weighted_f1: f64,

    /// Per-class metrics
    pub per_class: Vec<ClassMetrics>,

    /// Confusion matrix
    pub confusion_matrix: ConfusionMatrix,
}

impl Metrics {
    /// Create new metrics from predictions and ground truth labels
    pub fn from_predictions(
        predictions: &[usize],
        ground_truth: &[usize],
        num_classes: usize,
    ) -> Self {
        assert_eq!(
            predictions.len(),
            ground_truth.len(),
            "Predictions and ground truth must have same length"
        );

        let total_samples = predictions.len();
        if total_samples == 0 {
            return Self::default();
        }

        // Build confusion matrix
        let confusion_matrix = ConfusionMatrix::from_predictions(predictions, ground_truth, num_classes);

        // Calculate correct predictions
        let correct_predictions = predictions
            .iter()
            .zip(ground_truth.iter())
            .filter(|(p, g)| p == g)
            .count();

        let accuracy = correct_predictions as f64 / total_samples as f64;

        // Calculate per-class metrics
        let per_class: Vec<ClassMetrics> = (0..num_classes)
            .map(|class_idx| {
                ClassMetrics::from_confusion_matrix(&confusion_matrix, class_idx)
            })
            .collect();

        // Calculate macro-averaged metrics
        let valid_classes: Vec<&ClassMetrics> = per_class
            .iter()
            .filter(|m| m.support > 0)
            .collect();

        let num_valid = valid_classes.len() as f64;

        let macro_precision = if num_valid > 0.0 {
            valid_classes.iter().map(|m| m.precision).sum::<f64>() / num_valid
        } else {
            0.0
        };

        let macro_recall = if num_valid > 0.0 {
            valid_classes.iter().map(|m| m.recall).sum::<f64>() / num_valid
        } else {
            0.0
        };

        let macro_f1 = if num_valid > 0.0 {
            valid_classes.iter().map(|m| m.f1).sum::<f64>() / num_valid
        } else {
            0.0
        };

        // Calculate weighted F1
        let total_support: usize = per_class.iter().map(|m| m.support).sum();
        let weighted_f1 = if total_support > 0 {
            per_class
                .iter()
                .map(|m| m.f1 * m.support as f64)
                .sum::<f64>()
                / total_support as f64
        } else {
            0.0
        };

        Self {
            total_samples,
            correct_predictions,
            accuracy,
            top5_accuracy: 0.0, // Requires probability outputs
            average_loss: 0.0,  // Requires loss values
            loss: None,         // Set externally by trainer
            macro_precision,
            macro_recall,
            macro_f1,
            weighted_f1,
            per_class,
            confusion_matrix,
        }
    }

    /// Create metrics from predictions with top-k probabilities
    pub fn from_predictions_with_probs(
        predictions: &[usize],
        probabilities: &[Vec<f32>],
        ground_truth: &[usize],
        num_classes: usize,
    ) -> Self {
        let mut metrics = Self::from_predictions(predictions, ground_truth, num_classes);

        // Calculate top-5 accuracy
        let top5_correct = probabilities
            .iter()
            .zip(ground_truth.iter())
            .filter(|(probs, &gt)| {
                let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.iter().take(5).any(|(idx, _)| *idx == gt)
            })
            .count();

        metrics.top5_accuracy = top5_correct as f64 / metrics.total_samples as f64;

        metrics
    }

    /// Pretty print metrics
    pub fn display(&self) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        output.push_str("║                    Evaluation Metrics                        ║\n");
        output.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        output.push_str(&format!("║ Accuracy:          {:6.2}%                                  ║\n", self.accuracy * 100.0));
        output.push_str(&format!("║ Top-5 Accuracy:    {:6.2}%                                  ║\n", self.top5_accuracy * 100.0));
        output.push_str(&format!("║ Macro Precision:   {:6.2}%                                  ║\n", self.macro_precision * 100.0));
        output.push_str(&format!("║ Macro Recall:      {:6.2}%                                  ║\n", self.macro_recall * 100.0));
        output.push_str(&format!("║ Macro F1:          {:6.2}%                                  ║\n", self.macro_f1 * 100.0));
        output.push_str(&format!("║ Weighted F1:       {:6.2}%                                  ║\n", self.weighted_f1 * 100.0));
        output.push_str(&format!("║ Total Samples:     {:6}                                    ║\n", self.total_samples));
        output.push_str("╚══════════════════════════════════════════════════════════════╝\n");

        output
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            total_samples: 0,
            correct_predictions: 0,
            accuracy: 0.0,
            top5_accuracy: 0.0,
            average_loss: 0.0,
            loss: None,
            macro_precision: 0.0,
            macro_recall: 0.0,
            macro_f1: 0.0,
            weighted_f1: 0.0,
            per_class: Vec::new(),
            confusion_matrix: ConfusionMatrix::default(),
        }
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display())
    }
}

/// Per-class metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassMetrics {
    /// Class index
    pub class_idx: usize,

    /// Class name (if available)
    pub class_name: Option<String>,

    /// True positives
    pub true_positives: usize,

    /// False positives
    pub false_positives: usize,

    /// False negatives
    pub false_negatives: usize,

    /// True negatives
    pub true_negatives: usize,

    /// Precision = TP / (TP + FP)
    pub precision: f64,

    /// Recall = TP / (TP + FN)
    pub recall: f64,

    /// F1 = 2 * (precision * recall) / (precision + recall)
    pub f1: f64,

    /// Support = number of actual samples of this class
    pub support: usize,
}

impl ClassMetrics {
    /// Calculate metrics for a class from confusion matrix
    pub fn from_confusion_matrix(cm: &ConfusionMatrix, class_idx: usize) -> Self {
        let true_positives = cm.get(class_idx, class_idx);

        // False positives: predicted as this class but actually other classes
        let false_positives: usize = (0..cm.num_classes)
            .filter(|&i| i != class_idx)
            .map(|i| cm.get(i, class_idx))
            .sum();

        // False negatives: actually this class but predicted as other classes
        let false_negatives: usize = (0..cm.num_classes)
            .filter(|&i| i != class_idx)
            .map(|i| cm.get(class_idx, i))
            .sum();

        // True negatives: not this class and not predicted as this class
        let total: usize = cm.matrix.iter().sum();
        let true_negatives = total - true_positives - false_positives - false_negatives;

        let support = true_positives + false_negatives;

        let precision = if true_positives + false_positives > 0 {
            true_positives as f64 / (true_positives + false_positives) as f64
        } else {
            0.0
        };

        let recall = if true_positives + false_negatives > 0 {
            true_positives as f64 / (true_positives + false_negatives) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Self {
            class_idx,
            class_name: None,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
            precision,
            recall,
            f1,
            support,
        }
    }

    /// Set the class name
    pub fn with_name(mut self, name: &str) -> Self {
        self.class_name = Some(name.to_string());
        self
    }
}

/// Confusion Matrix for multi-class classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Number of classes
    pub num_classes: usize,

    /// Matrix data (row = actual, column = predicted)
    /// Stored as a flat vector in row-major order
    pub matrix: Vec<usize>,
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self::new(0)
    }
}

impl ConfusionMatrix {
    /// Create a new empty confusion matrix
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![0; num_classes * num_classes],
        }
    }

    /// Create confusion matrix from predictions and ground truth
    pub fn from_predictions(
        predictions: &[usize],
        ground_truth: &[usize],
        num_classes: usize,
    ) -> Self {
        let mut cm = Self::new(num_classes);

        for (&pred, &actual) in predictions.iter().zip(ground_truth.iter()) {
            cm.add(actual, pred);
        }

        cm
    }

    /// Add a single prediction to the matrix
    pub fn add(&mut self, actual: usize, predicted: usize) {
        if actual < self.num_classes && predicted < self.num_classes {
            let idx = actual * self.num_classes + predicted;
            self.matrix[idx] += 1;
        }
    }

    /// Get the count at (actual, predicted)
    pub fn get(&self, actual: usize, predicted: usize) -> usize {
        if actual < self.num_classes && predicted < self.num_classes {
            self.matrix[actual * self.num_classes + predicted]
        } else {
            0
        }
    }

    /// Get the total count
    pub fn total(&self) -> usize {
        self.matrix.iter().sum()
    }

    /// Get the number of correct predictions (diagonal sum)
    pub fn correct(&self) -> usize {
        (0..self.num_classes)
            .map(|i| self.get(i, i))
            .sum()
    }

    /// Get overall accuracy
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total > 0 {
            self.correct() as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Get the row sums (actual class counts)
    pub fn row_sums(&self) -> Vec<usize> {
        (0..self.num_classes)
            .map(|row| {
                (0..self.num_classes)
                    .map(|col| self.get(row, col))
                    .sum()
            })
            .collect()
    }

    /// Get the column sums (predicted class counts)
    pub fn col_sums(&self) -> Vec<usize> {
        (0..self.num_classes)
            .map(|col| {
                (0..self.num_classes)
                    .map(|row| self.get(row, col))
                    .sum()
            })
            .collect()
    }

    /// Normalize the matrix (rows sum to 1)
    pub fn normalize_rows(&self) -> Vec<Vec<f64>> {
        let row_sums = self.row_sums();

        (0..self.num_classes)
            .map(|row| {
                let sum = row_sums[row] as f64;
                (0..self.num_classes)
                    .map(|col| {
                        if sum > 0.0 {
                            self.get(row, col) as f64 / sum
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Pretty print the confusion matrix (for small number of classes)
    pub fn display(&self, class_names: Option<&[&str]>) -> String {
        let mut output = String::new();

        // Header
        output.push_str("\nConfusion Matrix (rows=actual, cols=predicted):\n\n");

        // Limit display for large matrices
        let max_display = 20;
        if self.num_classes > max_display {
            output.push_str(&format!(
                "(Matrix too large to display: {}x{})\n",
                self.num_classes, self.num_classes
            ));
            output.push_str(&format!("Total samples: {}\n", self.total()));
            output.push_str(&format!("Accuracy: {:.2}%\n", self.accuracy() * 100.0));
            return output;
        }

        // Column headers
        output.push_str("          ");
        for col in 0..self.num_classes {
            if let Some(names) = class_names {
                let name = names.get(col).unwrap_or(&"?");
                output.push_str(&format!("{:>6}", &name[..name.len().min(6)]));
            } else {
                output.push_str(&format!("{:>6}", col));
            }
        }
        output.push('\n');

        // Rows
        for row in 0..self.num_classes {
            if let Some(names) = class_names {
                let name = names.get(row).unwrap_or(&"?");
                output.push_str(&format!("{:>8} ", &name[..name.len().min(8)]));
            } else {
                output.push_str(&format!("{:>8} ", row));
            }

            for col in 0..self.num_classes {
                let count = self.get(row, col);
                if row == col {
                    output.push_str(&format!("[{:>4}]", count)); // Highlight diagonal
                } else if count > 0 {
                    output.push_str(&format!(" {:>4} ", count));
                } else {
                    output.push_str("    . ");
                }
            }
            output.push('\n');
        }

        output.push_str(&format!("\nAccuracy: {:.2}%\n", self.accuracy() * 100.0));

        output
    }

    /// Save confusion matrix to CSV
    pub fn save_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        let mut content = String::new();

        // Header
        content.push_str("actual\\predicted");
        for col in 0..self.num_classes {
            content.push_str(&format!(",{}", col));
        }
        content.push('\n');

        // Rows
        for row in 0..self.num_classes {
            content.push_str(&format!("{}", row));
            for col in 0..self.num_classes {
                content.push_str(&format!(",{}", self.get(row, col)));
            }
            content.push('\n');
        }

        std::fs::write(path, content)
    }
}

impl std::fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display(None))
    }
}

/// Running average for tracking metrics during training
#[derive(Debug, Clone, Default)]
pub struct RunningAverage {
    sum: f64,
    count: usize,
}

impl RunningAverage {
    /// Create a new running average
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value
    pub fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
    }

    /// Get the current average
    pub fn average(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }

    /// Get the count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset the running average
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

/// Accuracy tracker for training
#[derive(Debug, Clone, Default)]
pub struct AccuracyTracker {
    correct: usize,
    total: usize,
}

impl AccuracyTracker {
    /// Create a new accuracy tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a batch of predictions
    pub fn add_batch(&mut self, predictions: &[usize], ground_truth: &[usize]) {
        for (pred, gt) in predictions.iter().zip(ground_truth.iter()) {
            self.total += 1;
            if pred == gt {
                self.correct += 1;
            }
        }
    }

    /// Get the current accuracy
    pub fn accuracy(&self) -> f64 {
        if self.total > 0 {
            self.correct as f64 / self.total as f64
        } else {
            0.0
        }
    }

    /// Get the count
    pub fn count(&self) -> usize {
        self.total
    }

    /// Reset the tracker
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        // predictions: [0, 1, 2, 0, 1, 2, 0, 0, 2, 2]
        // ground_truth: [0, 1, 2, 0, 2, 2, 1, 0, 1, 2]
        // Comparing: (pred, gt)
        // (0,0)=correct, (1,1)=correct, (2,2)=correct, (0,0)=correct,
        // (1,2)=wrong, (2,2)=correct, (0,1)=wrong, (0,0)=correct, (2,1)=wrong, (2,2)=correct
        // Correct: 7 (indices 0,1,2,3,5,7,9)
        let predictions = vec![0, 1, 2, 0, 1, 2, 0, 0, 2, 2];
        let ground_truth = vec![0, 1, 2, 0, 2, 2, 1, 0, 1, 2];

        let cm = ConfusionMatrix::from_predictions(&predictions, &ground_truth, 3);

        // Check diagonal (correct predictions)
        assert_eq!(cm.get(0, 0), 3); // Class 0 actual, predicted 0: indices 0, 3, 7
        assert_eq!(cm.get(1, 1), 1); // Class 1 actual, predicted 1: index 1
        assert_eq!(cm.get(2, 2), 3); // Class 2 actual, predicted 2: indices 2, 5, 9

        // Check total
        assert_eq!(cm.total(), 10);
        assert_eq!(cm.correct(), 7);
        assert!((cm.accuracy() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_metrics_from_predictions() {
        let predictions = vec![0, 1, 2, 0, 1, 2, 0, 0, 2, 2];
        let ground_truth = vec![0, 1, 2, 0, 2, 2, 1, 0, 1, 2];

        let metrics = Metrics::from_predictions(&predictions, &ground_truth, 3);

        assert_eq!(metrics.total_samples, 10);
        assert_eq!(metrics.correct_predictions, 7);
        assert!((metrics.accuracy - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_class_metrics() {
        let predictions = vec![0, 0, 0, 1, 1];
        let ground_truth = vec![0, 0, 1, 1, 0];

        let cm = ConfusionMatrix::from_predictions(&predictions, &ground_truth, 2);
        let class0 = ClassMetrics::from_confusion_matrix(&cm, 0);

        // Class 0: TP=2, FP=1, FN=1, TN=1
        assert_eq!(class0.true_positives, 2);
        assert_eq!(class0.false_positives, 1);
        assert_eq!(class0.false_negatives, 1);
        assert!((class0.precision - 2.0 / 3.0).abs() < 0.001);
        assert!((class0.recall - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_running_average() {
        let mut avg = RunningAverage::new();

        avg.add(1.0);
        avg.add(2.0);
        avg.add(3.0);

        assert_eq!(avg.count(), 3);
        assert!((avg.average() - 2.0).abs() < 0.001);

        avg.reset();
        assert_eq!(avg.count(), 0);
        assert_eq!(avg.average(), 0.0);
    }

    #[test]
    fn test_accuracy_tracker() {
        let mut tracker = AccuracyTracker::new();

        tracker.add_batch(&[0, 1, 2], &[0, 1, 0]); // 2 correct out of 3

        assert_eq!(tracker.count(), 3);
        assert!((tracker.accuracy() - 2.0 / 3.0).abs() < 0.001);
    }
}
