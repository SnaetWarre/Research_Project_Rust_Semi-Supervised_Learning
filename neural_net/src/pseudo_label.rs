//! Pseudo-labeling module for semi-supervised learning
//!
//! This module provides functionality for pseudo-labeling unlabeled data
//! based on GPU model predictions with confidence thresholding.

use crate::{Float, GpuNetwork, GpuTensor};
use serde::{Deserialize, Serialize};

/// Configuration for pseudo-labeling
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PseudoLabelConfig {
    /// Minimum confidence threshold to accept a pseudo-label (0.0 to 1.0)
    pub confidence_threshold: Float,
    /// Maximum number of pseudo-labels to add per iteration
    pub max_per_iteration: Option<usize>,
    /// Whether to use only the most confident predictions
    pub use_top_k: bool,
    /// Number of top predictions to use if use_top_k is true
    pub top_k: usize,
    /// Whether to balance classes when pseudo-labeling
    pub balance_classes: bool,
}

impl Default for PseudoLabelConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            max_per_iteration: None,
            use_top_k: false,
            top_k: 1000,
            balance_classes: true,
        }
    }
}

impl PseudoLabelConfig {
    /// Create new config with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: Float) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set maximum per iteration
    pub fn with_max_per_iteration(mut self, max: usize) -> Self {
        self.max_per_iteration = Some(max);
        self
    }

    /// Enable top-k selection
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.use_top_k = true;
        self.top_k = k;
        self
    }

    /// Enable class balancing
    pub fn with_balance_classes(mut self, balance: bool) -> Self {
        self.balance_classes = balance;
        self
    }
}

/// Result of pseudo-labeling
#[derive(Clone, Debug)]
pub struct PseudoLabelResult {
    /// Indices of data points that were pseudo-labeled
    pub labeled_indices: Vec<usize>,
    /// Predicted labels for those data points
    pub predicted_labels: Vec<usize>,
    /// Confidence scores for each prediction
    pub confidences: Vec<Float>,
    /// Number of predictions above threshold
    pub num_above_threshold: usize,
    /// Average confidence of accepted predictions
    pub average_confidence: Float,
}

impl PseudoLabelResult {
    /// Check if any data was labeled
    pub fn has_labels(&self) -> bool {
        !self.labeled_indices.is_empty()
    }

    /// Get number of pseudo-labels
    pub fn num_labels(&self) -> usize {
        self.labeled_indices.len()
    }

    /// Get class distribution of pseudo-labels
    pub fn class_distribution(&self, num_classes: usize) -> Vec<usize> {
        let mut counts = vec![0; num_classes];
        for &label in &self.predicted_labels {
            if label < num_classes {
                counts[label] += 1;
            }
        }
        counts
    }
}

/// Pseudo-labeling engine
pub struct PseudoLabeler {
    pub config: PseudoLabelConfig,
}

impl PseudoLabeler {
    /// Create new pseudo-labeler with config
    pub fn new(config: PseudoLabelConfig) -> Self {
        Self { config }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(PseudoLabelConfig::default())
    }

    /// Pseudo-label unlabeled data using GPU network
    ///
    /// # Arguments
    ///
    /// * `network` - Trained GPU network to make predictions
    /// * `unlabeled_data` - GPU tensor of unlabeled data
    /// * `unlabeled_indices` - Original indices of unlabeled data
    /// * `num_classes` - Number of classes in the dataset
    ///
    /// # Returns
    ///
    /// PseudoLabelResult containing indices and labels of confident predictions
    pub fn label(
        &self,
        network: &mut GpuNetwork,
        unlabeled_data: &GpuTensor,
        unlabeled_indices: &[usize],
        num_classes: usize,
    ) -> Result<PseudoLabelResult, String> {
        // Set network to evaluation mode
        network.set_training(false);

        // Get predictions (all on GPU)
        let predictions = network.forward(unlabeled_data)?;
        let softmax_preds = predictions.softmax()?;

        // Get predicted classes using GPU argmax
        let pred_classes = softmax_preds.argmax_rows()?;
        
        // Get confidence scores (transfer only softmax probabilities to CPU)
        let pred_vec = softmax_preds.to_vec()?;

        // Extract confidence scores and labels
        let mut candidates = Vec::new();
        let batch_size = unlabeled_indices.len();

        for i in 0..batch_size {
            let pred_class = pred_classes[i] as usize;
            let start_idx = i * num_classes;
            let conf = pred_vec[start_idx + pred_class];

            // Check if above threshold
            if conf >= self.config.confidence_threshold {
                candidates.push((unlabeled_indices[i], pred_class, conf));
            }
        }

        let num_above_threshold = candidates.len();

        // Sort by confidence (descending)
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        // Apply top-k or max_per_iteration filtering
        if self.config.use_top_k {
            candidates.truncate(self.config.top_k);
        }

        if let Some(max) = self.config.max_per_iteration {
            candidates.truncate(max);
        }

        // Apply class balancing if enabled
        if self.config.balance_classes && !candidates.is_empty() {
            candidates = self.balance_classes_internal(&candidates, num_classes);
        }

        // Extract results
        let labeled_indices: Vec<usize> = candidates.iter().map(|(idx, _, _)| *idx).collect();
        let predicted_labels: Vec<usize> = candidates.iter().map(|(_, label, _)| *label).collect();
        let confidences: Vec<Float> = candidates.iter().map(|(_, _, conf)| *conf).collect();

        let average_confidence = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<Float>() / confidences.len() as Float
        };

        Ok(PseudoLabelResult {
            labeled_indices,
            predicted_labels,
            confidences,
            num_above_threshold,
            average_confidence,
        })
    }

    /// Balance classes in pseudo-labeled data (public for batch processing)
    pub fn balance_classes_internal(
        &self,
        candidates: &[(usize, usize, Float)],
        num_classes: usize,
    ) -> Vec<(usize, usize, Float)> {
        // Group by class
        let mut class_groups: Vec<Vec<(usize, usize, Float)>> = vec![Vec::new(); num_classes];

        for &candidate in candidates {
            let class_idx = candidate.1;
            if class_idx < num_classes {
                class_groups[class_idx].push(candidate);
            }
        }

        // Find minimum class size
        let min_size = class_groups.iter()
            .filter(|g| !g.is_empty())
            .map(|g| g.len())
            .min()
            .unwrap_or(0);

        // Take min_size from each class
        let mut balanced = Vec::new();
        for group in class_groups {
            balanced.extend_from_slice(&group[..min_size.min(group.len())]);
        }

        balanced
    }
}

/// Pseudo-labeling history tracker
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PseudoLabelingHistory {
    iterations: Vec<PseudoLabelingIteration>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PseudoLabelingIteration {
    iteration: usize,
    num_labels: usize,
    average_confidence: Float,
    class_distribution: Vec<usize>,
}

impl PseudoLabelingHistory {
    /// Create new history tracker
    pub fn new() -> Self {
        Self {
            iterations: Vec::new(),
        }
    }

    /// Add iteration result
    pub fn add_iteration(&mut self, result: &PseudoLabelResult, num_classes: usize) {
        self.iterations.push(PseudoLabelingIteration {
            iteration: self.iterations.len(),
            num_labels: result.num_labels(),
            average_confidence: result.average_confidence,
            class_distribution: result.class_distribution(num_classes),
        });
    }

    /// Save to JSON file
    pub fn save_to_json(&self, path: impl AsRef<std::path::Path>) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Failed to write file: {}", e))
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("Pseudo-labeling History:");
        for iter in &self.iterations {
            println!("  Iteration {}: {} labels, avg confidence: {:.4}",
                iter.iteration + 1, iter.num_labels, iter.average_confidence);
        }
    }
}

impl Default for PseudoLabelingHistory {
    fn default() -> Self {
        Self::new()
    }
}
