//! Learning rate scheduling strategies.
//!
//! This module provides various learning rate schedulers:
//! - Step decay
//! - Exponential decay
//! - Cosine annealing
//! - Reduce on plateau
//! - Warmup schedules

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Learning rate scheduler type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulerType {
    /// Constant learning rate (no scheduling)
    Constant,

    /// Step decay: multiply LR by gamma every step_size epochs
    StepLR {
        step_size: usize,
        gamma: f64,
    },

    /// Exponential decay: multiply LR by gamma every epoch
    ExponentialLR {
        gamma: f64,
    },

    /// Cosine annealing: cosine decay from initial to min LR
    CosineAnnealingLR {
        t_max: usize,
        eta_min: f64,
    },

    /// Reduce on plateau: reduce LR when metric stops improving
    ReduceLROnPlateau {
        factor: f64,
        patience: usize,
        threshold: f64,
        min_lr: f64,
    },

    /// Linear warmup followed by cosine decay
    WarmupCosine {
        warmup_epochs: usize,
        total_epochs: usize,
        eta_min: f64,
    },
}

impl Default for SchedulerType {
    fn default() -> Self {
        Self::Constant
    }
}

/// Learning rate scheduler
pub struct LearningRateScheduler {
    scheduler_type: SchedulerType,
    base_lr: f64,
    current_lr: f64,
    current_epoch: usize,

    // For ReduceLROnPlateau
    best_metric: Option<f64>,
    patience_counter: usize,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler
    pub fn new(scheduler_type: SchedulerType, base_lr: f64) -> Self {
        Self {
            scheduler_type,
            base_lr,
            current_lr: base_lr,
            current_epoch: 0,
            best_metric: None,
            patience_counter: 0,
        }
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Step the scheduler (call at the end of each epoch)
    pub fn step(&mut self) {
        self.current_epoch += 1;

        match &self.scheduler_type {
            SchedulerType::Constant => {
                // No change
            }

            SchedulerType::StepLR { step_size, gamma } => {
                if self.current_epoch % step_size == 0 {
                    self.current_lr *= gamma;
                }
            }

            SchedulerType::ExponentialLR { gamma } => {
                self.current_lr *= gamma;
            }

            SchedulerType::CosineAnnealingLR { t_max, eta_min } => {
                let progress = (self.current_epoch as f64) / (*t_max as f64);
                let progress = progress.min(1.0);
                self.current_lr = eta_min + (self.base_lr - eta_min)
                    * (1.0 + (progress * PI).cos()) / 2.0;
            }

            SchedulerType::WarmupCosine {
                warmup_epochs,
                total_epochs,
                eta_min,
            } => {
                if self.current_epoch <= *warmup_epochs {
                    // Linear warmup
                    let progress = (self.current_epoch as f64) / (*warmup_epochs as f64);
                    self.current_lr = self.base_lr * progress;
                } else {
                    // Cosine decay
                    let decay_epochs = total_epochs - warmup_epochs;
                    let progress = ((self.current_epoch - warmup_epochs) as f64)
                        / (decay_epochs as f64);
                    let progress = progress.min(1.0);
                    self.current_lr = eta_min + (self.base_lr - eta_min)
                        * (1.0 + (progress * PI).cos()) / 2.0;
                }
            }

            SchedulerType::ReduceLROnPlateau { .. } => {
                // This requires calling step_with_metric instead
            }
        }
    }

    /// Step the scheduler with a validation metric (for ReduceLROnPlateau)
    pub fn step_with_metric(&mut self, metric: f64) {
        self.current_epoch += 1;

        if let SchedulerType::ReduceLROnPlateau {
            factor,
            patience,
            threshold,
            min_lr,
        } = &self.scheduler_type
        {
            let improved = if let Some(best) = self.best_metric {
                // Check if metric improved by at least threshold
                metric < best - threshold
            } else {
                true
            };

            if improved {
                self.best_metric = Some(metric);
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;

                if self.patience_counter >= *patience {
                    // Reduce learning rate
                    let new_lr = self.current_lr * factor;
                    self.current_lr = new_lr.max(*min_lr);
                    self.patience_counter = 0;
                }
            }
        } else {
            // For other schedulers, just call regular step
            self.step();
        }
    }

    /// Reset the scheduler
    pub fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.current_epoch = 0;
        self.best_metric = None;
        self.patience_counter = 0;
    }

    /// Get the current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let mut scheduler = LearningRateScheduler::new(SchedulerType::Constant, 0.001);
        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step();
        assert_eq!(scheduler.get_lr(), 0.001);
    }

    #[test]
    fn test_step_lr() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::StepLR {
                step_size: 2,
                gamma: 0.5,
            },
            0.001,
        );

        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step(); // epoch 1
        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step(); // epoch 2
        assert!((scheduler.get_lr() - 0.0005).abs() < 1e-6);

        scheduler.step(); // epoch 3
        assert!((scheduler.get_lr() - 0.0005).abs() < 1e-6);

        scheduler.step(); // epoch 4
        assert!((scheduler.get_lr() - 0.00025).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_lr() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::ExponentialLR { gamma: 0.9 },
            0.001,
        );

        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.0009).abs() < 1e-6);

        scheduler.step();
        assert!((scheduler.get_lr() - 0.00081).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::CosineAnnealingLR {
                t_max: 10,
                eta_min: 0.0001,
            },
            0.001,
        );

        assert_eq!(scheduler.get_lr(), 0.001);

        // At halfway point, LR should be closer to eta_min
        for _ in 0..5 {
            scheduler.step();
        }

        let lr_at_5 = scheduler.get_lr();
        assert!(lr_at_5 < 0.001);
        assert!(lr_at_5 > 0.0001);

        // At end, LR should be close to eta_min
        for _ in 0..5 {
            scheduler.step();
        }

        let lr_at_10 = scheduler.get_lr();
        assert!((lr_at_10 - 0.0001).abs() < 0.0001);
    }

    #[test]
    fn test_warmup_cosine() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::WarmupCosine {
                warmup_epochs: 5,
                total_epochs: 20,
                eta_min: 0.0001,
            },
            0.001,
        );

        assert_eq!(scheduler.get_lr(), 0.001);

        // During warmup, LR should increase
        scheduler.step(); // epoch 1
        let lr_1 = scheduler.get_lr();
        assert!(lr_1 < 0.001);

        scheduler.step(); // epoch 2
        let lr_2 = scheduler.get_lr();
        assert!(lr_2 > lr_1);

        // After warmup, LR should start decreasing
        for _ in 0..4 {
            scheduler.step();
        }

        let lr_after_warmup = scheduler.get_lr();

        scheduler.step();
        let lr_after_decay_start = scheduler.get_lr();
        assert!(lr_after_decay_start < lr_after_warmup);
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::ReduceLROnPlateau {
                factor: 0.5,
                patience: 2,
                threshold: 0.001,
                min_lr: 0.00001,
            },
            0.001,
        );

        assert_eq!(scheduler.get_lr(), 0.001);

        // Improving metric - no LR change
        scheduler.step_with_metric(0.5);
        assert_eq!(scheduler.get_lr(), 0.001);

        scheduler.step_with_metric(0.4);
        assert_eq!(scheduler.get_lr(), 0.001);

        // No improvement for patience epochs - LR should reduce
        scheduler.step_with_metric(0.401); // No improvement (within threshold)
        scheduler.step_with_metric(0.402);

        let new_lr = scheduler.get_lr();
        assert!((new_lr - 0.0005).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut scheduler = LearningRateScheduler::new(
            SchedulerType::ExponentialLR { gamma: 0.9 },
            0.001,
        );

        scheduler.step();
        scheduler.step();
        assert_ne!(scheduler.get_lr(), 0.001);
        assert_eq!(scheduler.current_epoch(), 2);

        scheduler.reset();
        assert_eq!(scheduler.get_lr(), 0.001);
        assert_eq!(scheduler.current_epoch(), 0);
    }
}
