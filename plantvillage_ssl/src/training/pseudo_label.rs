//! Pseudo-Labeling Algorithm for Semi-Supervised Learning
//!
//! This module implements the pseudo-labeling strategy where high-confidence
//! predictions on unlabeled data are used as training labels. This is a key
//! component of our semi-supervised learning pipeline.
//!
//! ## Algorithm Overview
//!
//! 1. Run inference on unlabeled images
//! 2. Filter predictions by confidence threshold
//! 3. Assign pseudo-labels to high-confidence predictions
//! 4. Track pseudo-label quality for evaluation
//! 5. Periodically retrain with combined labeled + pseudo-labeled data

use std::collections::HashMap;
use std::path::PathBuf;

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::dataset::split::{HiddenLabelImage, PseudoLabeledImage};

/// Configuration for pseudo-labeling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PseudoLabelConfig {
    /// Confidence threshold for accepting predictions (0.0 to 1.0)
    pub confidence_threshold: f64,

    /// Maximum number of pseudo-labels per class (to prevent class imbalance)
    pub max_per_class: Option<usize>,

    /// Minimum number of pseudo-labels before retraining
    pub retrain_threshold: usize,

    /// Whether to use curriculum learning (start with higher threshold)
    pub curriculum_learning: bool,

    /// Initial threshold for curriculum learning
    pub curriculum_initial_threshold: f64,

    /// Final threshold for curriculum learning
    pub curriculum_final_threshold: f64,

    /// Number of epochs over which to reduce threshold
    pub curriculum_epochs: usize,
}

impl Default for PseudoLabelConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.9,
            max_per_class: Some(500),
            retrain_threshold: 200,
            curriculum_learning: false,
            curriculum_initial_threshold: 0.95,
            curriculum_final_threshold: 0.8,
            curriculum_epochs: 20,
        }
    }
}

impl PseudoLabelConfig {
    /// Get the effective threshold based on current epoch (for curriculum learning)
    pub fn get_threshold(&self, current_epoch: usize) -> f64 {
        if !self.curriculum_learning {
            return self.confidence_threshold;
        }

        if current_epoch >= self.curriculum_epochs {
            return self.curriculum_final_threshold;
        }

        // Linear interpolation from initial to final threshold
        let progress = current_epoch as f64 / self.curriculum_epochs as f64;
        let threshold = self.curriculum_initial_threshold
            - progress * (self.curriculum_initial_threshold - self.curriculum_final_threshold);

        threshold
    }
}

/// A single prediction result from the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    /// Image path
    pub image_path: PathBuf,

    /// Predicted class label
    pub predicted_label: usize,

    /// Confidence score (probability)
    pub confidence: f32,

    /// Full probability distribution over classes
    pub probabilities: Vec<f32>,

    /// Image ID for tracking
    pub image_id: u64,

    /// Ground truth label (if known, for evaluation)
    pub ground_truth: Option<usize>,
}

impl Prediction {
    /// Check if the prediction is correct (if ground truth is known)
    pub fn is_correct(&self) -> Option<bool> {
        self.ground_truth.map(|gt| gt == self.predicted_label)
    }

    /// Get the entropy of the prediction (measure of uncertainty)
    pub fn entropy(&self) -> f32 {
        self.probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Get the margin between top-1 and top-2 predictions
    pub fn margin(&self) -> f32 {
        let mut sorted = self.probabilities.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if sorted.len() >= 2 {
            sorted[0] - sorted[1]
        } else {
            sorted.first().copied().unwrap_or(0.0)
        }
    }
}

/// Pseudo-labeler for semi-supervised learning
#[derive(Debug)]
pub struct PseudoLabeler {
    /// Configuration
    config: PseudoLabelConfig,

    /// Accumulated pseudo-labels
    pseudo_labels: Vec<PseudoLabeledImage>,

    /// Count of pseudo-labels per class
    class_counts: HashMap<usize, usize>,

    /// Statistics about pseudo-labeling quality
    stats: PseudoLabelStats,

    /// Current training epoch (for curriculum learning)
    current_epoch: usize,

    /// Simulation day (for stream simulation)
    current_day: usize,
}

impl PseudoLabeler {
    /// Create a new pseudo-labeler with the given configuration
    pub fn new(config: PseudoLabelConfig) -> Self {
        Self {
            config,
            pseudo_labels: Vec::new(),
            class_counts: HashMap::new(),
            stats: PseudoLabelStats::default(),
            current_epoch: 0,
            current_day: 0,
        }
    }

    /// Create with default configuration
    pub fn default_labeler() -> Self {
        Self::new(PseudoLabelConfig::default())
    }

    /// Set the current epoch (for curriculum learning)
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }

    /// Set the current simulation day
    pub fn set_day(&mut self, day: usize) {
        self.current_day = day;
    }

    /// Get the current confidence threshold
    pub fn current_threshold(&self) -> f64 {
        self.config.get_threshold(self.current_epoch)
    }

    /// Process a batch of predictions and generate pseudo-labels
    pub fn process_predictions(
        &mut self,
        predictions: &[Prediction],
        hidden_labels: &[HiddenLabelImage],
    ) -> Vec<PseudoLabeledImage> {
        let threshold = self.current_threshold();
        let mut new_pseudo_labels = Vec::new();

        for (pred, hidden) in predictions.iter().zip(hidden_labels.iter()) {
            self.stats.total_processed += 1;

            // Check confidence threshold
            if pred.confidence < threshold as f32 {
                self.stats.rejected_low_confidence += 1;
                continue;
            }

            // Check class count limit
            if let Some(max_per_class) = self.config.max_per_class {
                let current_count = *self.class_counts.get(&pred.predicted_label).unwrap_or(&0);
                if current_count >= max_per_class {
                    self.stats.rejected_class_limit += 1;
                    continue;
                }
            }

            // Create pseudo-label
            let is_correct = pred.predicted_label == hidden.hidden_label;
            let pseudo_label = PseudoLabeledImage {
                image_path: hidden.image_path.clone(),
                predicted_label: pred.predicted_label,
                confidence: pred.confidence,
                ground_truth: hidden.hidden_label,
                is_correct,
                image_id: hidden.image_id,
                assigned_day: self.current_day,
            };

            // Update statistics
            self.stats.total_accepted += 1;
            if is_correct {
                self.stats.correct_predictions += 1;
            } else {
                self.stats.incorrect_predictions += 1;
            }

            // Update class counts
            *self.class_counts.entry(pred.predicted_label).or_insert(0) += 1;

            new_pseudo_labels.push(pseudo_label.clone());
            self.pseudo_labels.push(pseudo_label);
        }

        debug!(
            "Processed {} predictions, accepted {} pseudo-labels (threshold: {:.2})",
            predictions.len(),
            new_pseudo_labels.len(),
            threshold
        );

        new_pseudo_labels
    }

    /// Process a single prediction
    pub fn process_single(
        &mut self,
        prediction: &Prediction,
        hidden: &HiddenLabelImage,
    ) -> Option<PseudoLabeledImage> {
        let results = self.process_predictions(&[prediction.clone()], &[hidden.clone()]);
        results.into_iter().next()
    }

    /// Check if retraining should be triggered
    pub fn should_retrain(&self) -> bool {
        self.pseudo_labels.len() >= self.config.retrain_threshold
    }

    /// Get all accumulated pseudo-labels and clear the buffer
    pub fn get_and_clear_pseudo_labels(&mut self) -> Vec<PseudoLabeledImage> {
        let labels = std::mem::take(&mut self.pseudo_labels);
        // Keep class counts for balance tracking
        labels
    }

    /// Get all accumulated pseudo-labels without clearing
    pub fn get_pseudo_labels(&self) -> &[PseudoLabeledImage] {
        &self.pseudo_labels
    }

    /// Get the current number of pseudo-labels
    pub fn num_pseudo_labels(&self) -> usize {
        self.pseudo_labels.len()
    }

    /// Get pseudo-labeling statistics
    pub fn stats(&self) -> &PseudoLabelStats {
        &self.stats
    }

    /// Reset statistics (but keep pseudo-labels)
    pub fn reset_stats(&mut self) {
        self.stats = PseudoLabelStats::default();
    }

    /// Clear all pseudo-labels and reset
    pub fn clear(&mut self) {
        self.pseudo_labels.clear();
        self.class_counts.clear();
        self.stats = PseudoLabelStats::default();
    }

    /// Get class distribution of pseudo-labels
    pub fn class_distribution(&self) -> &HashMap<usize, usize> {
        &self.class_counts
    }

    /// Calculate pseudo-label accuracy (requires ground truth)
    pub fn pseudo_label_accuracy(&self) -> f64 {
        if self.stats.total_accepted == 0 {
            return 0.0;
        }
        self.stats.correct_predictions as f64 / self.stats.total_accepted as f64
    }
}

/// Statistics about pseudo-labeling quality
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PseudoLabelStats {
    /// Total images processed
    pub total_processed: usize,

    /// Total pseudo-labels accepted
    pub total_accepted: usize,

    /// Rejected due to low confidence
    pub rejected_low_confidence: usize,

    /// Rejected due to class limit
    pub rejected_class_limit: usize,

    /// Correct predictions (matches ground truth)
    pub correct_predictions: usize,

    /// Incorrect predictions
    pub incorrect_predictions: usize,
}

impl PseudoLabelStats {
    /// Calculate acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_processed == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_processed as f64
    }

    /// Calculate accuracy of accepted pseudo-labels
    pub fn accuracy(&self) -> f64 {
        if self.total_accepted == 0 {
            return 0.0;
        }
        self.correct_predictions as f64 / self.total_accepted as f64
    }
}

impl std::fmt::Display for PseudoLabelStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Pseudo-Label Statistics:")?;
        writeln!(f, "  Total processed: {}", self.total_processed)?;
        writeln!(
            f,
            "  Accepted: {} ({:.1}%)",
            self.total_accepted,
            self.acceptance_rate() * 100.0
        )?;
        writeln!(f, "  Rejected (low confidence): {}", self.rejected_low_confidence)?;
        writeln!(f, "  Rejected (class limit): {}", self.rejected_class_limit)?;
        writeln!(
            f,
            "  Accuracy: {:.1}% ({} correct, {} incorrect)",
            self.accuracy() * 100.0,
            self.correct_predictions,
            self.incorrect_predictions
        )?;
        Ok(())
    }
}

/// Stream simulation for semi-supervised learning demo
#[derive(Debug)]
pub struct StreamSimulator {
    /// Images available for streaming
    stream_pool: Vec<HiddenLabelImage>,

    /// Current index into the stream
    current_index: usize,

    /// Random number generator for shuffling
    rng: ChaCha8Rng,

    /// Images per day simulation
    images_per_day: usize,
}

impl StreamSimulator {
    /// Create a new stream simulator
    pub fn new(mut stream_pool: Vec<HiddenLabelImage>, seed: u64, images_per_day: usize) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        stream_pool.shuffle(&mut rng);

        Self {
            stream_pool,
            current_index: 0,
            rng,
            images_per_day,
        }
    }

    /// Get the next batch of "streaming" images for a simulated day
    pub fn next_day(&mut self) -> Option<Vec<HiddenLabelImage>> {
        if self.current_index >= self.stream_pool.len() {
            return None;
        }

        let end_index = (self.current_index + self.images_per_day).min(self.stream_pool.len());
        let batch = self.stream_pool[self.current_index..end_index].to_vec();
        self.current_index = end_index;

        Some(batch)
    }

    /// Get remaining images in the stream
    pub fn remaining(&self) -> usize {
        self.stream_pool.len().saturating_sub(self.current_index)
    }

    /// Get total images in the stream
    pub fn total(&self) -> usize {
        self.stream_pool.len()
    }

    /// Reset the stream to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
        self.stream_pool.shuffle(&mut self.rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_prediction(label: usize, confidence: f32, ground_truth: usize) -> Prediction {
        let mut probabilities = vec![0.0; 39];
        probabilities[label] = confidence;
        let remaining = 1.0 - confidence;
        for (i, p) in probabilities.iter_mut().enumerate() {
            if i != label {
                *p = remaining / 38.0;
            }
        }

        Prediction {
            image_path: PathBuf::from(format!("test_image_{}.jpg", label)),
            predicted_label: label,
            confidence,
            probabilities,
            image_id: label as u64,
            ground_truth: Some(ground_truth),
        }
    }

    fn create_test_hidden(label: usize, id: u64) -> HiddenLabelImage {
        HiddenLabelImage {
            image_path: PathBuf::from(format!("test_image_{}.jpg", id)),
            hidden_label: label,
            hidden_class_name: format!("Class_{}", label),
            image_id: id,
        }
    }

    #[test]
    fn test_threshold_filtering() {
        let config = PseudoLabelConfig {
            confidence_threshold: 0.9,
            max_per_class: None,
            retrain_threshold: 100,
            ..Default::default()
        };

        let mut labeler = PseudoLabeler::new(config);

        // High confidence prediction should be accepted
        let pred_high = create_test_prediction(5, 0.95, 5);
        let hidden_high = create_test_hidden(5, 1);
        let result = labeler.process_single(&pred_high, &hidden_high);
        assert!(result.is_some());

        // Low confidence prediction should be rejected
        let pred_low = create_test_prediction(5, 0.7, 5);
        let hidden_low = create_test_hidden(5, 2);
        let result = labeler.process_single(&pred_low, &hidden_low);
        assert!(result.is_none());
    }

    #[test]
    fn test_class_limit() {
        let config = PseudoLabelConfig {
            confidence_threshold: 0.8,
            max_per_class: Some(2),
            retrain_threshold: 100,
            ..Default::default()
        };

        let mut labeler = PseudoLabeler::new(config);

        // First two should be accepted
        for i in 0..2 {
            let pred = create_test_prediction(5, 0.95, 5);
            let hidden = create_test_hidden(5, i);
            let result = labeler.process_single(&pred, &hidden);
            assert!(result.is_some());
        }

        // Third should be rejected due to class limit
        let pred = create_test_prediction(5, 0.95, 5);
        let hidden = create_test_hidden(5, 3);
        let result = labeler.process_single(&pred, &hidden);
        assert!(result.is_none());
    }

    #[test]
    fn test_accuracy_tracking() {
        let config = PseudoLabelConfig {
            confidence_threshold: 0.8,
            max_per_class: None,
            retrain_threshold: 100,
            ..Default::default()
        };

        let mut labeler = PseudoLabeler::new(config);

        // Correct prediction
        let pred1 = create_test_prediction(5, 0.95, 5);
        let hidden1 = create_test_hidden(5, 1);
        labeler.process_single(&pred1, &hidden1);

        // Incorrect prediction (model predicts 3, true label is 5)
        let pred2 = create_test_prediction(3, 0.95, 5);
        let hidden2 = create_test_hidden(5, 2);
        labeler.process_single(&pred2, &hidden2);

        assert_eq!(labeler.stats().correct_predictions, 1);
        assert_eq!(labeler.stats().incorrect_predictions, 1);
        assert!((labeler.pseudo_label_accuracy() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_curriculum_learning() {
        let config = PseudoLabelConfig {
            curriculum_learning: true,
            curriculum_initial_threshold: 0.95,
            curriculum_final_threshold: 0.80,
            curriculum_epochs: 10,
            ..Default::default()
        };

        let mut labeler = PseudoLabeler::new(config);

        // At epoch 0, threshold should be initial
        labeler.set_epoch(0);
        assert!((labeler.current_threshold() - 0.95).abs() < 0.001);

        // At epoch 5, threshold should be midpoint
        labeler.set_epoch(5);
        assert!((labeler.current_threshold() - 0.875).abs() < 0.001);

        // At epoch 10+, threshold should be final
        labeler.set_epoch(10);
        assert!((labeler.current_threshold() - 0.80).abs() < 0.001);

        labeler.set_epoch(20);
        assert!((labeler.current_threshold() - 0.80).abs() < 0.001);
    }

    #[test]
    fn test_prediction_entropy() {
        let pred = create_test_prediction(5, 0.9, 5);
        let entropy = pred.entropy();
        // High confidence should have low entropy
        assert!(entropy > 0.0);
        assert!(entropy < 1.0);
    }

    #[test]
    fn test_stream_simulator() {
        let images: Vec<HiddenLabelImage> = (0..100)
            .map(|i| create_test_hidden(i % 10, i as u64))
            .collect();

        let mut simulator = StreamSimulator::new(images, 42, 25);

        assert_eq!(simulator.total(), 100);
        assert_eq!(simulator.remaining(), 100);

        // First day
        let day1 = simulator.next_day().unwrap();
        assert_eq!(day1.len(), 25);
        assert_eq!(simulator.remaining(), 75);

        // Second day
        let day2 = simulator.next_day().unwrap();
        assert_eq!(day2.len(), 25);
        assert_eq!(simulator.remaining(), 50);

        // Continue until exhausted
        simulator.next_day();
        simulator.next_day();
        assert_eq!(simulator.remaining(), 0);

        // Should return None when exhausted
        assert!(simulator.next_day().is_none());
    }
}
