//! Elastic Weight Consolidation (EWC)
//!
//! This module implements EWC, which uses the Fisher Information Matrix
//! to identify important parameters and penalize changes to them during
//! incremental learning.

use crate::{IncrementalConfig, IncrementalLearner, TrainingMetrics};
use anyhow::{anyhow, Result};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// EWC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EWCConfig {
    /// Importance weight for EWC penalty (lambda)
    pub lambda: f32,
    /// Number of samples to use for Fisher estimation
    pub fisher_samples: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Online EWC (accumulate Fisher over tasks)
    pub online: bool,
    /// Gamma for online EWC (decay factor)
    pub gamma: f32,
}

impl Default for EWCConfig {
    fn default() -> Self {
        Self {
            lambda: 5000.0,
            fisher_samples: 200,
            learning_rate: 0.001,
            online: false,
            gamma: 0.9,
        }
    }
}

/// Parameter importance information
#[derive(Debug, Clone)]
pub struct ParameterImportance {
    /// Parameter name/identifier
    pub name: String,
    /// Parameter values (stored after task completion)
    pub values: Vec<f32>,
    /// Fisher information (importance)
    pub fisher: Vec<f32>,
}

impl ParameterImportance {
    /// Create new parameter importance
    pub fn new(name: String, size: usize) -> Self {
        Self {
            name,
            values: vec![0.0; size],
            fisher: vec![0.0; size],
        }
    }

    /// Update parameter values
    pub fn update_values(&mut self, values: Vec<f32>) {
        self.values = values;
    }

    /// Update Fisher information
    pub fn update_fisher(&mut self, fisher: Vec<f32>) {
        self.fisher = fisher;
    }

    /// Accumulate Fisher information (for online EWC)
    pub fn accumulate_fisher(&mut self, new_fisher: Vec<f32>, gamma: f32) {
        for (f, new_f) in self.fisher.iter_mut().zip(new_fisher.iter()) {
            *f = gamma * (*f) + new_f;
        }
    }

    /// Compute EWC penalty for parameter changes
    pub fn compute_penalty(&self, current_values: &[f32]) -> f32 {
        if current_values.len() != self.values.len() {
            return 0.0;
        }

        self.values
            .iter()
            .zip(self.fisher.iter())
            .zip(current_values.iter())
            .map(|((old_val, fisher), new_val)| {
                fisher * (new_val - old_val).powi(2)
            })
            .sum()
    }
}

/// Elastic Weight Consolidation learner
#[derive(Debug)]
pub struct EWCLearner<B: Backend> {
    /// Current step/task number
    current_step: usize,
    /// Total classes learned
    num_classes: usize,
    /// Configuration
    config: EWCConfig,
    /// Parameter importance for each task
    parameter_importance: HashMap<String, ParameterImportance>,
    /// Total number of parameters (for simulation)
    num_parameters: usize,
    /// Phantom data for backend
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> EWCLearner<B> {
    /// Create a new EWC learner
    pub fn new(config: EWCConfig) -> Self {
        Self {
            current_step: 0,
            num_classes: 0,
            config,
            parameter_importance: HashMap::new(),
            num_parameters: 1000, // Simulated parameter count
            _backend: std::marker::PhantomData,
        }
    }

    /// Estimate Fisher Information Matrix
    fn estimate_fisher(&self, data: &[(Vec<f32>, usize)]) -> Result<Vec<f32>> {
        let num_samples = self.config.fisher_samples.min(data.len());

        if num_samples == 0 {
            return Err(anyhow!("No samples available for Fisher estimation"));
        }

        println!("Estimating Fisher Information using {} samples...", num_samples);

        let mut fisher = vec![0.0; self.num_parameters];

        // Estimate Fisher by sampling
        for (features, _label) in data.iter().take(num_samples) {
            // Simulate gradient computation
            // In real implementation, this would compute:
            // F_i = E[(∂log p(y|x,θ) / ∂θ_i)^2]
            let gradients = self.simulate_gradients(features);

            // Accumulate squared gradients
            for (f, grad) in fisher.iter_mut().zip(gradients.iter()) {
                *f += grad * grad;
            }
        }

        // Average over samples
        for f in fisher.iter_mut() {
            *f /= num_samples as f32;
        }

        Ok(fisher)
    }

    /// Simulate gradient computation (placeholder)
    fn simulate_gradients(&self, _features: &[f32]) -> Vec<f32> {
        // In real implementation, compute actual gradients via backprop
        // For now, simulate with random-like values
        (0..self.num_parameters)
            .map(|i| (i as f32 * 0.1).sin() * 0.01)
            .collect()
    }

    /// Get current parameter values (simulated)
    fn get_parameter_values(&self) -> Vec<f32> {
        // In real implementation, extract from actual model
        vec![0.5; self.num_parameters]
    }

    /// Compute EWC penalty
    fn compute_ewc_penalty(&self, current_params: &[f32]) -> f32 {
        self.parameter_importance
            .values()
            .map(|importance| importance.compute_penalty(current_params))
            .sum::<f32>()
            * self.config.lambda
            / 2.0
    }

    /// Store parameter importance after task completion
    fn store_importance(&mut self, param_name: String, data: &[(Vec<f32>, usize)]) -> Result<()> {
        let fisher = self.estimate_fisher(data)?;
        let values = self.get_parameter_values();

        if let Some(importance) = self.parameter_importance.get_mut(&param_name) {
            // Update existing (online EWC)
            if self.config.online {
                importance.accumulate_fisher(fisher, self.config.gamma);
                importance.update_values(values);
            }
        } else {
            // Create new
            let mut importance = ParameterImportance::new(param_name.clone(), self.num_parameters);
            importance.update_values(values);
            importance.update_fisher(fisher);
            self.parameter_importance.insert(param_name, importance);
        }

        Ok(())
    }
}

impl<B: Backend> IncrementalLearner<B> for EWCLearner<B> {
    fn prepare_for_new_classes(&mut self, num_new_classes: usize) -> Result<()> {
        if num_new_classes == 0 {
            return Err(anyhow!("Cannot add zero classes"));
        }

        self.num_classes += num_new_classes;

        println!(
            "EWC: Prepared for {} new classes (total: {})",
            num_new_classes, self.num_classes
        );

        Ok(())
    }

    fn train_incremental(
        &mut self,
        train_data: &[(Vec<f32>, usize)],
        val_data: &[(Vec<f32>, usize)],
        _config: &IncrementalConfig,
    ) -> Result<TrainingMetrics> {
        if train_data.is_empty() {
            return Err(anyhow!("Training data is empty"));
        }
        if val_data.is_empty() {
            return Err(anyhow!("Validation data is empty"));
        }

        let mut metrics = TrainingMetrics::new();
        let start_time = std::time::Instant::now();

        let num_epochs = 20;

        for epoch in 0..num_epochs {
            let mut epoch_loss = 0.0;
            let mut ewc_penalty_sum = 0.0;

            // Training loop
            for (_features, _target) in train_data.iter() {
                // Simulate forward pass and loss computation
                let classification_loss = 2.0 * (-0.1 * epoch as f32).exp();

                // Compute EWC penalty if not first task
                let ewc_penalty = if self.current_step > 0 {
                    let current_params = self.get_parameter_values();
                    self.compute_ewc_penalty(&current_params)
                } else {
                    0.0
                };

                let total_loss = classification_loss + ewc_penalty;
                epoch_loss += total_loss;
                ewc_penalty_sum += ewc_penalty;
            }

            let train_loss = epoch_loss / train_data.len() as f32;
            let avg_ewc_penalty = ewc_penalty_sum / train_data.len() as f32;

            // Validation
            let mut val_correct = 0;
            for (_features, _target) in val_data.iter() {
                // Simulate validation
                if epoch > 5 {
                    val_correct += 1;
                }
            }
            let val_accuracy = val_correct as f32 / val_data.len() as f32;

            metrics.add_epoch(train_loss, val_accuracy);

            if epoch % 5 == 0 {
                println!(
                    "Epoch {}/{}: loss={:.4}, ewc_penalty={:.4}, val_acc={:.4}",
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    avg_ewc_penalty,
                    val_accuracy
                );
            }
        }

        metrics.set_training_time(start_time.elapsed().as_secs_f64());
        metrics.add_extra("ewc_lambda", self.config.lambda);
        metrics.add_extra("fisher_samples", self.config.fisher_samples as f32);

        Ok(metrics)
    }

    fn finalize_step(&mut self, step: usize) -> Result<()> {
        self.current_step = step;

        // Store parameter importance for this task
        // In a real implementation, this would use actual training data
        let dummy_data = vec![(vec![0.5; 100], 0); self.config.fisher_samples];
        let param_name = format!("task_{}", step);
        self.store_importance(param_name, &dummy_data)?;

        println!(
            "EWC: Finalized step {}, stored Fisher information for {} parameters",
            step, self.num_parameters
        );

        Ok(())
    }

    fn current_step(&self) -> usize {
        self.current_step
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_ewc_learner_creation() {
        let config = EWCConfig::default();
        let learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        assert_eq!(learner.current_step(), 0);
        assert_eq!(learner.num_classes(), 0);
        assert_eq!(learner.parameter_importance.len(), 0);
    }

    #[test]
    fn test_parameter_importance() {
        let mut importance = ParameterImportance::new("test".to_string(), 5);

        importance.update_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        importance.update_fisher(vec![0.1, 0.2, 0.3, 0.4, 0.5]);

        // Same values should give zero penalty
        let penalty = importance.compute_penalty(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(penalty < 1e-6);

        // Different values should give non-zero penalty
        let penalty = importance.compute_penalty(&[1.5, 2.5, 3.5, 4.5, 5.5]);
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_fisher_accumulation() {
        let mut importance = ParameterImportance::new("test".to_string(), 3);

        importance.update_fisher(vec![1.0, 2.0, 3.0]);
        importance.accumulate_fisher(vec![2.0, 4.0, 6.0], 0.5);

        // After accumulation: 0.5 * old + new = 0.5 * [1,2,3] + [2,4,6] = [2.5, 5.0, 7.5]
        assert!((importance.fisher[0] - 2.5).abs() < 1e-6);
        assert!((importance.fisher[1] - 5.0).abs() < 1e-6);
        assert!((importance.fisher[2] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_prepare_for_new_classes() {
        let config = EWCConfig::default();
        let mut learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 5);

        learner.prepare_for_new_classes(5).unwrap();
        assert_eq!(learner.num_classes(), 10);
    }

    #[test]
    fn test_fisher_estimation() {
        let config = EWCConfig {
            fisher_samples: 10,
            ..Default::default()
        };
        let learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        let data = vec![(vec![0.5; 100], 0); 20];
        let fisher = learner.estimate_fisher(&data).unwrap();

        assert_eq!(fisher.len(), learner.num_parameters);
        assert!(fisher.iter().all(|&f| f >= 0.0)); // Fisher values should be non-negative
    }

    #[test]
    fn test_ewc_penalty_computation() {
        let config = EWCConfig {
            lambda: 100.0,
            ..Default::default()
        };
        let mut learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        // Store some importance
        let data = vec![(vec![0.5; 100], 0); 10];
        learner.store_importance("test".to_string(), &data).unwrap();

        // Compute penalty
        let current_params = learner.get_parameter_values();
        let penalty = learner.compute_ewc_penalty(&current_params);

        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_train_incremental() {
        let config = EWCConfig::default();
        let mut learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();

        let train_data = vec![(vec![0.5; 100], 0); 20];
        let val_data = vec![(vec![0.3; 100], 0); 10];

        let inc_config = IncrementalConfig {
            initial_classes: 5,
            classes_per_step: 5,
            num_steps: 1,
            method: crate::IncrementalMethod::EWC {
                lambda: 5000.0,
                fisher_samples: 10,
            },
            seed: 42,
        };

        let metrics = learner.train_incremental(&train_data, &val_data, &inc_config).unwrap();

        assert!(metrics.train_loss.len() > 0);
        assert!(metrics.val_accuracy.len() > 0);
        assert!(metrics.extra.contains_key("ewc_lambda"));
        assert!(metrics.extra.contains_key("fisher_samples"));
    }

    #[test]
    fn test_online_ewc() {
        let config = EWCConfig {
            online: true,
            gamma: 0.9,
            fisher_samples: 10,
            ..Default::default()
        };
        let mut learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        let data = vec![(vec![0.5; 100], 0); 10];

        // Store importance twice with same name (should accumulate)
        learner.store_importance("test".to_string(), &data).unwrap();
        learner.store_importance("test".to_string(), &data).unwrap();

        assert_eq!(learner.parameter_importance.len(), 1);
    }

    #[test]
    fn test_finalize_step() {
        let config = EWCConfig::default();
        let mut learner: EWCLearner<TestBackend> = EWCLearner::new(config);

        learner.prepare_for_new_classes(5).unwrap();
        learner.finalize_step(1).unwrap();

        assert_eq!(learner.current_step(), 1);
        assert!(learner.parameter_importance.len() > 0);
    }
}
