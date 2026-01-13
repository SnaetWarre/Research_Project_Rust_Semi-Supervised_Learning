//! Learning without Forgetting (LwF)
//!
//! This module implements the Learning without Forgetting approach using
//! knowledge distillation to preserve knowledge of old tasks while learning new ones.

use super::{IncrementalConfig, IncrementalLearner, TrainingMetrics};
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LwF configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LwFConfig {
    /// Temperature for softmax distillation
    pub temperature: f32,
    /// Weight for distillation loss (lambda)
    pub lambda: f32,
    /// Weight for classification loss on new data
    pub alpha: f32,
    /// Learning rate
    pub learning_rate: f64,
    /// Whether to store teacher outputs or compute on-the-fly
    pub store_teacher_outputs: bool,
}

impl Default for LwFConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            lambda: 1.0,
            alpha: 1.0,
            learning_rate: 0.001,
            store_teacher_outputs: true,
        }
    }
}

/// Learning without Forgetting learner
#[derive(Debug)]
pub struct LwFLearner<B: Backend> {
    /// Current step/task number
    current_step: usize,
    /// Total classes learned
    num_classes: usize,
    /// Configuration
    config: LwFConfig,
    /// Teacher model outputs (soft targets) for distillation
    teacher_outputs: HashMap<String, Vec<f32>>,
    /// Number of old classes (for distillation)
    old_classes: usize,
    /// Phantom data for backend
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> LwFLearner<B> {
    /// Create a new LwF learner
    pub fn new(config: LwFConfig) -> Self {
        Self {
            current_step: 0,
            num_classes: 0,
            config,
            teacher_outputs: HashMap::new(),
            old_classes: 0,
            _backend: std::marker::PhantomData,
        }
    }

    /// Store teacher outputs for a sample
    pub fn store_teacher_output(&mut self, sample_id: String, outputs: Vec<f32>) {
        self.teacher_outputs.insert(sample_id, outputs);
    }

    /// Get teacher outputs for a sample
    pub fn get_teacher_output(&self, sample_id: &str) -> Option<&Vec<f32>> {
        self.teacher_outputs.get(sample_id)
    }

    /// Clear stored teacher outputs
    pub fn clear_teacher_outputs(&mut self) {
        self.teacher_outputs.clear();
    }

    /// Compute distillation loss
    fn compute_distillation_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: &[f32],
    ) -> f32 {
        if student_logits.len() != teacher_logits.len() {
            return 0.0;
        }

        let temp = self.config.temperature;

        // Apply temperature and compute softmax
        let student_soft = Self::softmax_with_temperature(student_logits, temp);
        let teacher_soft = Self::softmax_with_temperature(teacher_logits, temp);

        // KL divergence loss
        let mut loss = 0.0;
        for (s, t) in student_soft.iter().zip(teacher_soft.iter()) {
            if *t > 1e-10 {
                loss -= t * (s / t).ln();
            }
        }

        // Scale by temperature squared (standard practice in distillation)
        loss * temp * temp
    }

    /// Apply softmax with temperature
    fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
        let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_vals: Vec<f32> = scaled.iter().map(|x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();

        exp_vals.iter().map(|x| x / sum).collect()
    }

    /// Compute combined loss (classification + distillation)
    fn compute_combined_loss(
        &self,
        student_logits: &[f32],
        teacher_logits: Option<&[f32]>,
        target: usize,
        _num_classes: usize,
    ) -> f32 {
        // Classification loss (cross-entropy)
        let student_soft = Self::softmax_with_temperature(student_logits, 1.0);
        let class_loss = if target < student_soft.len() {
            -student_soft[target].ln()
        } else {
            0.0
        };

        // Distillation loss (if we have teacher outputs)
        let distill_loss = if let Some(teacher_logits) = teacher_logits {
            // Only distill on old classes
            let old_student = &student_logits[..self.old_classes.min(student_logits.len())];
            let old_teacher = &teacher_logits[..self.old_classes.min(teacher_logits.len())];
            self.compute_distillation_loss(old_student, old_teacher)
        } else {
            0.0
        };

        // Combined loss
        self.config.alpha * class_loss + self.config.lambda * distill_loss
    }

    /// Capture teacher outputs before training on new task
    fn capture_teacher_outputs(&mut self, data: &[(Vec<f32>, usize)]) -> Result<()> {
        if !self.config.store_teacher_outputs || self.current_step == 0 {
            return Ok(());
        }

        println!("Capturing teacher outputs for {} samples...", data.len());

        for (idx, (features, _)) in data.iter().enumerate() {
            // Simulate teacher model inference
            // In real implementation, this would run the current model
            let teacher_logits = self.simulate_model_output(features);

            let sample_id = format!("sample_{}", idx);
            self.store_teacher_output(sample_id, teacher_logits);
        }

        Ok(())
    }

    /// Simulate model output (placeholder)
    fn simulate_model_output(&self, _features: &[f32]) -> Vec<f32> {
        // In real implementation, this would run the actual model
        vec![0.1; self.old_classes]
    }
}

impl<B: Backend> IncrementalLearner<B> for LwFLearner<B> {
    fn prepare_for_new_classes(&mut self, num_new_classes: usize) -> Result<()> {
        if num_new_classes == 0 {
            return Err(anyhow!("Cannot add zero classes"));
        }

        // Store the number of old classes for distillation
        self.old_classes = self.num_classes;

        // Update total class count
        self.num_classes += num_new_classes;

        println!(
            "LwF: Prepared for {} new classes (total: {}, old: {})",
            num_new_classes, self.num_classes, self.old_classes
        );

        Ok(())
    }

    fn train_incremental(
        &mut self,
        train_data: &[(Vec<f32>, usize)],
        val_data: &[(Vec<f32>, usize)],
        config: &IncrementalConfig,
    ) -> Result<TrainingMetrics> {
        if train_data.is_empty() {
            return Err(anyhow!("Training data is empty"));
        }
        if val_data.is_empty() {
            return Err(anyhow!("Validation data is empty"));
        }

        // Capture teacher outputs if this is an incremental step
        if self.current_step > 0 {
            self.capture_teacher_outputs(train_data)?;
        }

        let mut metrics = TrainingMetrics::new();
        let start_time = std::time::Instant::now();

        let num_epochs = 15; // Typically LwF needs more epochs

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut correct = 0;
            let total = train_data.len();

            // Training loop
            for (idx, (features, target)) in train_data.iter().enumerate() {
                // Simulate student model output
                let student_logits = self.simulate_model_output(features);

                // Get teacher outputs if available
                let sample_id = format!("sample_{}", idx);
                let teacher_logits = self.get_teacher_output(&sample_id);

                // Compute loss
                let loss = self.compute_combined_loss(
                    &student_logits,
                    teacher_logits.map(|v| v.as_slice()),
                    *target,
                    self.num_classes,
                );

                epoch_loss += loss;

                // Simulate prediction
                let prediction = student_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if prediction == *target {
                    correct += 1;
                }
            }

            let train_loss = epoch_loss / total as f32;
            let train_accuracy = correct as f32 / total as f32;

            // Validation
            let mut val_correct = 0;
            for (features, target) in val_data.iter() {
                let logits = self.simulate_model_output(features);
                let prediction = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if prediction == *target {
                    val_correct += 1;
                }
            }
            let val_accuracy = val_correct as f32 / val_data.len() as f32;

            metrics.add_epoch(train_loss, val_accuracy);

            if epoch % 5 == 0 {
                println!(
                    "Epoch {}/{}: loss={:.4}, train_acc={:.4}, val_acc={:.4}",
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    train_accuracy,
                    val_accuracy
                );
            }
        }

        metrics.set_training_time(start_time.elapsed().as_secs_f64());
        metrics.add_extra("distillation_lambda", self.config.lambda);
        metrics.add_extra("temperature", self.config.temperature);
        metrics.add_extra("old_classes", self.old_classes as f32);

        Ok(metrics)
    }

    fn finalize_step(&mut self, step: usize) -> Result<()> {
        self.current_step = step;

        // Optionally clear teacher outputs to save memory
        if !self.config.store_teacher_outputs {
            self.clear_teacher_outputs();
        }

        println!("LwF: Finalized step {} with {} total classes", step, self.num_classes);

        Ok(())
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_lwf_learner_creation() {
        let config = LwFConfig::default();
        let learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        assert_eq!(learner.current_step(), 0);
        assert_eq!(learner.num_classes(), 0);
        assert_eq!(learner.old_classes, 0);
    }

    #[test]
    fn test_teacher_output_storage() {
        let config = LwFConfig::default();
        let mut learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        let outputs = vec![0.1, 0.2, 0.7];
        learner.store_teacher_output("sample_1".to_string(), outputs.clone());

        assert_eq!(learner.get_teacher_output("sample_1"), Some(&outputs));
        assert_eq!(learner.get_teacher_output("sample_2"), None);

        learner.clear_teacher_outputs();
        assert_eq!(learner.get_teacher_output("sample_1"), None);
    }

    #[test]
    fn test_softmax_with_temperature() {
        let logits = vec![1.0, 2.0, 3.0];

        // T=1 (normal softmax)
        let soft1 = LwFLearner::<TestBackend>::softmax_with_temperature(&logits, 1.0);
        let sum1: f32 = soft1.iter().sum();
        assert!((sum1 - 1.0).abs() < 1e-6);

        // T=2 (softer)
        let soft2 = LwFLearner::<TestBackend>::softmax_with_temperature(&logits, 2.0);
        let sum2: f32 = soft2.iter().sum();
        assert!((sum2 - 1.0).abs() < 1e-6);

        // Higher temperature should make distribution more uniform
        assert!(soft2[0] > soft1[0]);
        assert!(soft2[2] < soft1[2]);
    }

    #[test]
    fn test_distillation_loss() {
        let config = LwFConfig {
            temperature: 2.0,
            ..Default::default()
        };
        let learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        let student = vec![1.0, 2.0, 3.0];
        let teacher = vec![1.0, 2.0, 3.0];

        // Same outputs should give very low loss
        let loss = learner.compute_distillation_loss(&student, &teacher);
        assert!(loss < 0.1);

        // Different outputs should give higher loss
        let teacher2 = vec![3.0, 2.0, 1.0];
        let loss2 = learner.compute_distillation_loss(&student, &teacher2);
        assert!(loss2 > loss);
    }

    #[test]
    fn test_prepare_for_new_classes() {
        let config = LwFConfig::default();
        let mut learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        // Initial classes
        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 5);
        assert_eq!(learner.old_classes, 0);

        learner.finalize_step(1).unwrap();

        // Add more classes
        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 10);
        assert_eq!(learner.old_classes, 5);
    }

    #[test]
    fn test_train_incremental() {
        let config = LwFConfig::default();
        let mut learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();

        let train_data = vec![(vec![0.5; 100], 0); 20];
        let val_data = vec![(vec![0.3; 100], 0); 10];

        let inc_config = IncrementalConfig {
            initial_classes: 5,
            classes_per_step: 5,
            num_steps: 1,
            method: crate::IncrementalMethod::LwF {
                temperature: 2.0,
                lambda: 1.0,
            },
            seed: 42,
        };

        let metrics = learner.train_incremental(&train_data, &val_data, &inc_config).unwrap();

        assert!(metrics.train_loss.len() > 0);
        assert!(metrics.val_accuracy.len() > 0);
        assert!(metrics.extra.contains_key("distillation_lambda"));
        assert!(metrics.extra.contains_key("temperature"));
    }

    #[test]
    fn test_combined_loss_computation() {
        let config = LwFConfig {
            lambda: 1.0,
            alpha: 1.0,
            ..Default::default()
        };
        let mut learner: LwFLearner<TestBackend> = LwFLearner::new(config);

        learner.old_classes = 3;
        learner.num_classes = 5;

        let student_logits = vec![1.0, 2.0, 3.0, 0.5, 0.5];
        let teacher_logits = vec![1.5, 2.5, 2.0];

        let loss = learner.compute_combined_loss(
            &student_logits,
            Some(&teacher_logits),
            2,
            5,
        );

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }
}
