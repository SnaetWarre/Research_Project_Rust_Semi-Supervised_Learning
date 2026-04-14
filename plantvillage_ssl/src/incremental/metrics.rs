//! Metrics for evaluating incremental learning performance
//!
//! This module provides utilities for computing and analyzing metrics
//! specific to incremental learning scenarios.

use super::IncrementalResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compute average accuracy across all tasks
pub fn average_accuracy(task_accuracies: &[f32]) -> f32 {
    if task_accuracies.is_empty() {
        return 0.0;
    }
    task_accuracies.iter().sum::<f32>() / task_accuracies.len() as f32
}

/// Compute backward transfer (change in performance on old tasks)
///
/// BWT = (1/T-1) * Σ_{i=1}^{T-1} (R_{T,i} - R_{i,i})
/// where R_{i,j} is the accuracy on task i after training on task j
pub fn backward_transfer(accuracy_matrix: &[Vec<f32>]) -> f32 {
    if accuracy_matrix.len() < 2 {
        return 0.0;
    }

    let num_tasks = accuracy_matrix.len();
    let mut sum = 0.0;

    for i in 0..(num_tasks - 1) {
        // Final accuracy on task i (after all training)
        let final_acc = accuracy_matrix[num_tasks - 1][i];
        // Accuracy on task i right after training on it
        let initial_acc = accuracy_matrix[i][i];
        sum += final_acc - initial_acc;
    }

    sum / (num_tasks - 1) as f32
}

/// Compute forward transfer (knowledge transfer to new tasks)
///
/// FWT = (1/T-1) * Σ_{i=2}^{T} (R_{i-1,i} - b_i)
/// where b_i is the random baseline accuracy for task i
pub fn forward_transfer(accuracy_matrix: &[Vec<f32>], random_baseline: f32) -> f32 {
    if accuracy_matrix.len() < 2 {
        return 0.0;
    }

    let num_tasks = accuracy_matrix.len();
    let mut sum = 0.0;

    for i in 1..num_tasks {
        // Accuracy on task i before training on it (using previous model)
        let pre_training_acc = if i > 0 && accuracy_matrix[i - 1].len() > i {
            accuracy_matrix[i - 1][i]
        } else {
            random_baseline
        };
        sum += pre_training_acc - random_baseline;
    }

    sum / (num_tasks - 1) as f32
}

/// Compute forgetting measure
///
/// Forgetting = (1/T-1) * Σ_{i=1}^{T-1} max_{j∈{1,...,T-1}} (R_{j,i} - R_{T,i})
pub fn forgetting_measure(accuracy_matrix: &[Vec<f32>]) -> f32 {
    if accuracy_matrix.len() < 2 {
        return 0.0;
    }

    let num_tasks = accuracy_matrix.len();
    let mut sum = 0.0;

    for i in 0..(num_tasks - 1) {
        let mut max_diff = 0.0f32;
        let final_acc = accuracy_matrix[num_tasks - 1][i];

        // Find maximum accuracy achieved on task i during training
        for j in i..num_tasks {
            let acc_at_j = accuracy_matrix[j][i];
            let diff = acc_at_j - final_acc;
            max_diff = max_diff.max(diff);
        }

        sum += max_diff;
    }

    sum / (num_tasks - 1) as f32
}

/// Compute intransigence (inability to learn new tasks)
pub fn intransigence(accuracy_matrix: &[Vec<f32>], random_baseline: f32) -> f32 {
    if accuracy_matrix.is_empty() {
        return 0.0;
    }

    let num_tasks = accuracy_matrix.len();
    let mut sum = 0.0;

    for i in 0..num_tasks {
        if accuracy_matrix[i].len() > i {
            sum += accuracy_matrix[i][i] - random_baseline;
        }
    }

    sum / num_tasks as f32
}

/// Analyze incremental learning results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalAnalysis {
    /// Average accuracy across all tasks
    pub average_accuracy: f32,
    /// Backward transfer (forgetting)
    pub backward_transfer: f32,
    /// Forward transfer
    pub forward_transfer: f32,
    /// Forgetting measure
    pub forgetting: f32,
    /// Intransigence measure
    pub intransigence: f32,
    /// Per-task final accuracies
    pub task_accuracies: Vec<f32>,
    /// Accuracy matrix (task x step)
    pub accuracy_matrix: Vec<Vec<f32>>,
}

impl IncrementalAnalysis {
    /// Analyze incremental learning results
    pub fn analyze(result: &IncrementalResult, random_baseline: f32) -> Self {
        // Build accuracy matrix
        let num_steps = result.step_metrics.len();
        let mut accuracy_matrix = vec![vec![0.0; num_steps]; num_steps];

        for (step_idx, step_metrics) in result.step_metrics.iter().enumerate() {
            for (task_idx, &acc) in step_metrics.task_accuracies.iter().enumerate() {
                if task_idx < num_steps && step_idx < num_steps {
                    accuracy_matrix[step_idx][task_idx] = acc;
                }
            }
        }

        // Get final task accuracies
        let task_accuracies = if let Some(last_step) = result.step_metrics.last() {
            last_step.task_accuracies.clone()
        } else {
            Vec::new()
        };

        Self {
            average_accuracy: average_accuracy(&task_accuracies),
            backward_transfer: backward_transfer(&accuracy_matrix),
            forward_transfer: forward_transfer(&accuracy_matrix, random_baseline),
            forgetting: forgetting_measure(&accuracy_matrix),
            intransigence: intransigence(&accuracy_matrix, random_baseline),
            task_accuracies,
            accuracy_matrix,
        }
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== Incremental Learning Analysis ===");
        println!("Average Accuracy:    {:.4}", self.average_accuracy);
        println!("Backward Transfer:   {:.4}", self.backward_transfer);
        println!("Forward Transfer:    {:.4}", self.forward_transfer);
        println!("Forgetting:          {:.4}", self.forgetting);
        println!("Intransigence:       {:.4}", self.intransigence);
        println!("\nPer-Task Accuracies:");
        for (i, acc) in self.task_accuracies.iter().enumerate() {
            println!("  Task {}: {:.4}", i, acc);
        }
    }
}

/// Compute class-incremental metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassIncrementalMetrics {
    /// Accuracy on old classes only
    pub old_class_accuracy: f32,
    /// Accuracy on new classes only
    pub new_class_accuracy: f32,
    /// Overall accuracy (all classes)
    pub overall_accuracy: f32,
    /// Number of old classes
    pub num_old_classes: usize,
    /// Number of new classes
    pub num_new_classes: usize,
}

impl ClassIncrementalMetrics {
    /// Compute metrics from predictions and labels
    pub fn compute(predictions: &[usize], labels: &[usize], num_old_classes: usize) -> Self {
        if predictions.len() != labels.len() {
            return Self::default();
        }

        let mut old_correct = 0;
        let mut old_total: usize = 0;
        let mut new_correct = 0;
        let mut new_total: usize = 0;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if *label < num_old_classes {
                old_total += 1;
                if pred == label {
                    old_correct += 1;
                }
            } else {
                new_total += 1;
                if pred == label {
                    new_correct += 1;
                }
            }
        }

        let old_acc = if old_total > 0 {
            old_correct as f32 / old_total as f32
        } else {
            0.0
        };

        let new_acc = if new_total > 0 {
            new_correct as f32 / new_total as f32
        } else {
            0.0
        };

        let overall_acc = if predictions.len() > 0 {
            (old_correct + new_correct) as f32 / predictions.len() as f32
        } else {
            0.0
        };

        Self {
            old_class_accuracy: old_acc,
            new_class_accuracy: new_acc,
            overall_accuracy: overall_acc,
            num_old_classes,
            num_new_classes: new_total.saturating_sub(old_total),
        }
    }
}

impl Default for ClassIncrementalMetrics {
    fn default() -> Self {
        Self {
            old_class_accuracy: 0.0,
            new_class_accuracy: 0.0,
            overall_accuracy: 0.0,
            num_old_classes: 0,
            num_new_classes: 0,
        }
    }
}

/// Export results to CSV format
pub fn export_to_csv(result: &IncrementalResult) -> String {
    let mut csv =
        String::from("step,task,accuracy,avg_accuracy,backward_transfer,forward_transfer\n");

    for step_metrics in &result.step_metrics {
        for (task_idx, &acc) in step_metrics.task_accuracies.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{:.4},{:.4},{},{}\n",
                step_metrics.step,
                task_idx,
                acc,
                step_metrics.average_accuracy,
                step_metrics
                    .backward_transfer
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_default(),
                step_metrics
                    .forward_transfer
                    .map(|v| format!("{:.4}", v))
                    .unwrap_or_default(),
            ));
        }
    }

    csv
}

/// Compare multiple incremental learning methods
#[derive(Debug)]
pub struct MethodComparison {
    /// Method name -> Analysis
    pub results: HashMap<String, IncrementalAnalysis>,
}

impl MethodComparison {
    /// Create new comparison
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Add a method's results
    pub fn add_method(&mut self, name: String, analysis: IncrementalAnalysis) {
        self.results.insert(name, analysis);
    }

    /// Print comparison table
    pub fn print_comparison(&self) {
        if self.results.is_empty() {
            println!("No results to compare");
            return;
        }

        println!("\n=== Method Comparison ===");
        println!(
            "{:<15} {:>12} {:>12} {:>12} {:>12} {:>12}",
            "Method", "Avg Acc", "BWT", "FWT", "Forgetting", "Intransig."
        );
        println!("{}", "-".repeat(77));

        for (name, analysis) in &self.results {
            println!(
                "{:<15} {:>12.4} {:>12.4} {:>12.4} {:>12.4} {:>12.4}",
                name,
                analysis.average_accuracy,
                analysis.backward_transfer,
                analysis.forward_transfer,
                analysis.forgetting,
                analysis.intransigence
            );
        }
    }

    /// Get best method by average accuracy
    pub fn best_by_accuracy(&self) -> Option<(&String, &IncrementalAnalysis)> {
        self.results
            .iter()
            .max_by(|(_, a), (_, b)| a.average_accuracy.partial_cmp(&b.average_accuracy).unwrap())
    }

    /// Get best method by backward transfer (least forgetting)
    pub fn best_by_backward_transfer(&self) -> Option<(&String, &IncrementalAnalysis)> {
        self.results.iter().max_by(|(_, a), (_, b)| {
            a.backward_transfer
                .partial_cmp(&b.backward_transfer)
                .unwrap()
        })
    }
}

impl Default for MethodComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ExperimentMetadata, IncrementalConfig, IncrementalMethod, TrainingMetrics};
    use super::*;

    #[test]
    fn test_average_accuracy() {
        let accs = vec![0.8, 0.7, 0.9];
        let avg = average_accuracy(&accs);
        assert!((avg - 0.8).abs() < 1e-6);

        assert_eq!(average_accuracy(&[]), 0.0);
    }

    #[test]
    fn test_backward_transfer() {
        // Perfect retention
        let matrix = vec![vec![0.9, 0.0], vec![0.9, 0.8]];
        let bwt = backward_transfer(&matrix);
        assert_eq!(bwt, 0.0); // No change in task 0 accuracy

        // Forgetting
        let matrix = vec![vec![0.9, 0.0], vec![0.7, 0.8]];
        let bwt = backward_transfer(&matrix);
        assert!((bwt - (-0.2)).abs() < 1e-6); // Task 0 dropped by 0.2
    }

    #[test]
    fn test_forward_transfer() {
        let matrix = vec![
            vec![0.8, 0.3], // Task 0 trained, task 1 gets 0.3 (vs 0.2 baseline)
            vec![0.7, 0.9],
        ];
        let fwt = forward_transfer(&matrix, 0.2);
        assert!((fwt - 0.1).abs() < 1e-6); // 0.3 - 0.2 = 0.1
    }

    #[test]
    fn test_forgetting_measure() {
        let matrix = vec![vec![0.9, 0.0], vec![0.9, 0.8], vec![0.7, 0.85]];
        let forgetting = forgetting_measure(&matrix);
        // Max acc on task 0 was 0.9 (at step 0 and 1), final is 0.7, diff = 0.2
        // Max acc on task 1 was 0.85 (at step 2), final is 0.85, diff = 0.0
        // Forgetting = (0.2 + 0.0) / 2 = 0.1
        assert!((forgetting - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_class_incremental_metrics() {
        let predictions = vec![0, 1, 2, 3, 4, 5];
        let labels = vec![0, 1, 2, 3, 3, 5];
        let metrics = ClassIncrementalMetrics::compute(&predictions, &labels, 3);

        // Old classes (0,1,2): 3/3 correct
        assert!((metrics.old_class_accuracy - 1.0).abs() < 1e-6);

        // New classes (3,5): 2/3 correct (labels 3,3,5 -> predictions correct for indices 3,5)
        // Actually: new class labels are at indices 3,4,5 with labels [3,3,5]
        // predictions are [3,4,5], so we have: 3==3 (correct), 4!=3 (wrong), 5==5 (correct)
        // That's 2/3 correct = 0.666...
        assert!((metrics.new_class_accuracy - 2.0 / 3.0).abs() < 1e-6);

        // Overall: 4/6 correct
        // Overall: 5/6 correct (0,1,2,3,5 are correct, only index 4 is wrong)
        assert!((metrics.overall_accuracy - 5.0 / 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_incremental_analysis() {
        let result = IncrementalResult {
            step_metrics: vec![
                StepMetrics {
                    step: 0,
                    training: TrainingMetrics::new(),
                    task_accuracies: vec![0.85],
                    average_accuracy: 0.85,
                    backward_transfer: None,
                    forward_transfer: None,
                },
                StepMetrics {
                    step: 1,
                    training: TrainingMetrics::new(),
                    task_accuracies: vec![0.80, 0.90],
                    average_accuracy: 0.85,
                    backward_transfer: Some(-0.05),
                    forward_transfer: Some(0.1),
                },
            ],
            metadata: ExperimentMetadata {
                name: "test".to_string(),
                config: IncrementalConfig {
                    initial_classes: 5,
                    classes_per_step: 5,
                    num_steps: 1,
                    method: IncrementalMethod::FineTuning {
                        freeze_backbone: false,
                        freeze_layers: 0,
                    },
                    seed: 42,
                },
                total_time: 100.0,
                final_accuracy: 0.85,
                avg_backward_transfer: -0.05,
                avg_forward_transfer: 0.1,
            },
        };

        let analysis = IncrementalAnalysis::analyze(&result, 0.2);
        assert!(analysis.average_accuracy > 0.0);
        assert_eq!(analysis.task_accuracies.len(), 2);
    }

    #[test]
    fn test_method_comparison() {
        let mut comparison = MethodComparison::new();

        let analysis1 = IncrementalAnalysis {
            average_accuracy: 0.85,
            backward_transfer: -0.1,
            forward_transfer: 0.05,
            forgetting: 0.1,
            intransigence: 0.05,
            task_accuracies: vec![0.85, 0.85],
            accuracy_matrix: vec![vec![0.85, 0.0], vec![0.75, 0.85]],
        };

        let analysis2 = IncrementalAnalysis {
            average_accuracy: 0.90,
            backward_transfer: -0.05,
            forward_transfer: 0.1,
            forgetting: 0.05,
            intransigence: 0.03,
            task_accuracies: vec![0.90, 0.90],
            accuracy_matrix: vec![vec![0.90, 0.0], vec![0.85, 0.90]],
        };

        comparison.add_method("Method1".to_string(), analysis1);
        comparison.add_method("Method2".to_string(), analysis2);

        let best = comparison.best_by_accuracy();
        assert!(best.is_some());
        assert_eq!(best.unwrap().0, "Method2");
    }

    #[test]
    fn test_export_to_csv() {
        let result = IncrementalResult {
            step_metrics: vec![StepMetrics {
                step: 0,
                training: TrainingMetrics::new(),
                task_accuracies: vec![0.85],
                average_accuracy: 0.85,
                backward_transfer: None,
                forward_transfer: None,
            }],
            metadata: ExperimentMetadata {
                name: "test".to_string(),
                config: IncrementalConfig {
                    initial_classes: 5,
                    classes_per_step: 5,
                    num_steps: 1,
                    method: IncrementalMethod::FineTuning {
                        freeze_backbone: false,
                        freeze_layers: 0,
                    },
                    seed: 42,
                },
                total_time: 100.0,
                final_accuracy: 0.85,
                avg_backward_transfer: 0.0,
                avg_forward_transfer: 0.0,
            },
        };

        let csv = export_to_csv(&result);
        assert!(csv.contains("step,task,accuracy"));
        assert!(csv.contains("0,0,0.8500"));
    }
}
