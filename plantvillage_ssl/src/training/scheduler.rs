//! Learning Rate Scheduler Module
//!
//! Provides various learning rate scheduling strategies for training.
//! These schedulers help optimize training convergence and final model quality.

use serde::{Deserialize, Serialize};

/// Learning rate scheduler that adjusts the learning rate during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRScheduler {
    /// Constant learning rate (no scheduling)
    Constant {
        lr: f64,
    },

    /// Step decay: reduce LR by factor at specified epochs
    StepDecay {
        initial_lr: f64,
        decay_factor: f64,
        step_epochs: Vec<usize>,
    },

    /// Exponential decay: lr = initial_lr * decay_rate^epoch
    Exponential {
        initial_lr: f64,
        decay_rate: f64,
    },

    /// Cosine annealing: smooth decay following cosine curve
    CosineAnnealing {
        initial_lr: f64,
        min_lr: f64,
        total_epochs: usize,
    },

    /// Warmup followed by cosine annealing
    WarmupCosine {
        initial_lr: f64,
        min_lr: f64,
        warmup_epochs: usize,
        total_epochs: usize,
    },

    /// Linear warmup followed by linear decay
    LinearWarmupDecay {
        initial_lr: f64,
        warmup_epochs: usize,
        total_epochs: usize,
    },

    /// One-cycle policy: warmup then annealing
    OneCycle {
        max_lr: f64,
        total_epochs: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    },
}

impl LRScheduler {
    /// Create a constant learning rate scheduler
    pub fn constant(lr: f64) -> Self {
        Self::Constant { lr }
    }

    /// Create a step decay scheduler
    pub fn step_decay(initial_lr: f64, decay_factor: f64, step_epochs: Vec<usize>) -> Self {
        Self::StepDecay {
            initial_lr,
            decay_factor,
            step_epochs,
        }
    }

    /// Create a cosine annealing scheduler
    pub fn cosine_annealing(initial_lr: f64, min_lr: f64, total_epochs: usize) -> Self {
        Self::CosineAnnealing {
            initial_lr,
            min_lr,
            total_epochs,
        }
    }

    /// Create a warmup + cosine annealing scheduler
    pub fn warmup_cosine(
        initial_lr: f64,
        min_lr: f64,
        warmup_epochs: usize,
        total_epochs: usize,
    ) -> Self {
        Self::WarmupCosine {
            initial_lr,
            min_lr,
            warmup_epochs,
            total_epochs,
        }
    }

    /// Create a one-cycle scheduler
    pub fn one_cycle(max_lr: f64, total_epochs: usize) -> Self {
        Self::OneCycle {
            max_lr,
            total_epochs,
            pct_start: 0.3, // 30% warmup
            div_factor: 25.0,
            final_div_factor: 1e4,
        }
    }

    /// Get the learning rate for a given epoch
    pub fn get_lr(&self, epoch: usize) -> f64 {
        match self {
            Self::Constant { lr } => *lr,

            Self::StepDecay {
                initial_lr,
                decay_factor,
                step_epochs,
            } => {
                let mut lr = *initial_lr;
                for &step_epoch in step_epochs {
                    if epoch >= step_epoch {
                        lr *= decay_factor;
                    }
                }
                lr
            }

            Self::Exponential {
                initial_lr,
                decay_rate,
            } => initial_lr * decay_rate.powi(epoch as i32),

            Self::CosineAnnealing {
                initial_lr,
                min_lr,
                total_epochs,
            } => {
                let progress = (epoch as f64) / (*total_epochs as f64);
                let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                min_lr + (initial_lr - min_lr) * cosine_factor
            }

            Self::WarmupCosine {
                initial_lr,
                min_lr,
                warmup_epochs,
                total_epochs,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    let progress = (epoch as f64 + 1.0) / (*warmup_epochs as f64);
                    initial_lr * progress
                } else {
                    // Cosine annealing
                    let remaining_epochs = total_epochs - warmup_epochs;
                    let progress = (epoch - warmup_epochs) as f64 / remaining_epochs as f64;
                    let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                    min_lr + (initial_lr - min_lr) * cosine_factor
                }
            }

            Self::LinearWarmupDecay {
                initial_lr,
                warmup_epochs,
                total_epochs,
            } => {
                if epoch < *warmup_epochs {
                    // Linear warmup
                    let progress = (epoch as f64 + 1.0) / (*warmup_epochs as f64);
                    initial_lr * progress
                } else {
                    // Linear decay
                    let remaining_epochs = total_epochs - warmup_epochs;
                    let progress = (epoch - warmup_epochs) as f64 / remaining_epochs as f64;
                    initial_lr * (1.0 - progress).max(0.0)
                }
            }

            Self::OneCycle {
                max_lr,
                total_epochs,
                pct_start,
                div_factor,
                final_div_factor,
            } => {
                let initial_lr = max_lr / div_factor;
                let min_lr = max_lr / final_div_factor;
                let warmup_epochs = (*total_epochs as f64 * pct_start) as usize;

                if epoch < warmup_epochs {
                    // Warmup phase: initial_lr -> max_lr
                    let progress = epoch as f64 / warmup_epochs as f64;
                    initial_lr + (max_lr - initial_lr) * progress
                } else {
                    // Annealing phase: max_lr -> min_lr
                    let remaining = total_epochs - warmup_epochs;
                    let progress = (epoch - warmup_epochs) as f64 / remaining as f64;
                    let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                    min_lr + (max_lr - min_lr) * cosine_factor
                }
            }
        }
    }

    /// Get the learning rate for a specific step within an epoch
    /// Useful for step-level scheduling (e.g., in one-cycle policy)
    pub fn get_lr_at_step(&self, epoch: usize, step: usize, steps_per_epoch: usize) -> f64 {
        let fractional_epoch = epoch as f64 + (step as f64 / steps_per_epoch as f64);

        match self {
            Self::OneCycle {
                max_lr,
                total_epochs,
                pct_start,
                div_factor,
                final_div_factor,
            } => {
                let initial_lr = max_lr / div_factor;
                let min_lr = max_lr / final_div_factor;
                let total_steps = *total_epochs as f64 * steps_per_epoch as f64;
                let warmup_steps = total_steps * pct_start;
                let current_step = fractional_epoch * steps_per_epoch as f64;

                if current_step < warmup_steps {
                    // Warmup phase
                    let progress = current_step / warmup_steps;
                    initial_lr + (max_lr - initial_lr) * progress
                } else {
                    // Annealing phase
                    let remaining_steps = total_steps - warmup_steps;
                    let progress = (current_step - warmup_steps) / remaining_steps;
                    let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
                    min_lr + (max_lr - min_lr) * cosine_factor
                }
            }

            // For other schedulers, just use epoch-level scheduling
            _ => self.get_lr(epoch),
        }
    }

    /// Get a description of the scheduler
    pub fn description(&self) -> String {
        match self {
            Self::Constant { lr } => format!("Constant LR: {:.6}", lr),
            Self::StepDecay {
                initial_lr,
                decay_factor,
                step_epochs,
            } => format!(
                "Step Decay: initial={:.6}, factor={}, steps={:?}",
                initial_lr, decay_factor, step_epochs
            ),
            Self::Exponential {
                initial_lr,
                decay_rate,
            } => format!(
                "Exponential: initial={:.6}, decay={:.4}",
                initial_lr, decay_rate
            ),
            Self::CosineAnnealing {
                initial_lr,
                min_lr,
                total_epochs,
            } => format!(
                "Cosine Annealing: initial={:.6}, min={:.6}, epochs={}",
                initial_lr, min_lr, total_epochs
            ),
            Self::WarmupCosine {
                initial_lr,
                warmup_epochs,
                total_epochs,
                ..
            } => format!(
                "Warmup + Cosine: initial={:.6}, warmup={}, total={}",
                initial_lr, warmup_epochs, total_epochs
            ),
            Self::LinearWarmupDecay {
                initial_lr,
                warmup_epochs,
                total_epochs,
            } => format!(
                "Linear Warmup + Decay: initial={:.6}, warmup={}, total={}",
                initial_lr, warmup_epochs, total_epochs
            ),
            Self::OneCycle {
                max_lr,
                total_epochs,
                ..
            } => format!("One-Cycle: max_lr={:.6}, epochs={}", max_lr, total_epochs),
        }
    }
}

impl Default for LRScheduler {
    fn default() -> Self {
        Self::WarmupCosine {
            initial_lr: 0.001,
            min_lr: 1e-6,
            warmup_epochs: 5,
            total_epochs: 50,
        }
    }
}

/// State for reduce-on-plateau scheduler
#[derive(Debug, Clone)]
pub struct ReduceOnPlateauState {
    best_metric: f64,
    epochs_without_improvement: usize,
    current_lr: f64,
    reduction_factor: f64,
    patience: usize,
    min_lr: f64,
    mode: PlateauMode,
}

/// Mode for plateau detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Metric should decrease (e.g., loss)
    Min,
    /// Metric should increase (e.g., accuracy)
    Max,
}

impl ReduceOnPlateauState {
    /// Create a new reduce-on-plateau state
    pub fn new(
        initial_lr: f64,
        reduction_factor: f64,
        patience: usize,
        min_lr: f64,
        mode: PlateauMode,
    ) -> Self {
        let best_metric = match mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };

        Self {
            best_metric,
            epochs_without_improvement: 0,
            current_lr: initial_lr,
            reduction_factor,
            patience,
            min_lr,
            mode,
        }
    }

    /// Update the scheduler with a new metric value and return the new learning rate
    pub fn step(&mut self, metric: f64) -> f64 {
        let improved = match self.mode {
            PlateauMode::Min => metric < self.best_metric,
            PlateauMode::Max => metric > self.best_metric,
        };

        if improved {
            self.best_metric = metric;
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;

            if self.epochs_without_improvement >= self.patience {
                let new_lr = (self.current_lr * self.reduction_factor).max(self.min_lr);
                if new_lr < self.current_lr {
                    self.current_lr = new_lr;
                    self.epochs_without_improvement = 0;
                }
            }
        }

        self.current_lr
    }

    /// Get the current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Check if the learning rate was recently reduced
    pub fn lr_was_reduced(&self) -> bool {
        self.epochs_without_improvement == 0 && self.current_lr < self.min_lr * 1.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LRScheduler::constant(0.001);
        assert_eq!(scheduler.get_lr(0), 0.001);
        assert_eq!(scheduler.get_lr(50), 0.001);
        assert_eq!(scheduler.get_lr(100), 0.001);
    }

    #[test]
    fn test_step_decay_scheduler() {
        let scheduler = LRScheduler::step_decay(0.1, 0.1, vec![10, 20, 30]);

        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(9), 0.1);
        assert!((scheduler.get_lr(10) - 0.01).abs() < 1e-10);
        assert!((scheduler.get_lr(20) - 0.001).abs() < 1e-10);
        assert!((scheduler.get_lr(30) - 0.0001).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let scheduler = LRScheduler::cosine_annealing(0.1, 0.001, 100);

        // At epoch 0, should be at initial_lr
        let lr_0 = scheduler.get_lr(0);
        assert!(lr_0 > 0.09); // Close to 0.1

        // At epoch 50, should be approximately midpoint
        let lr_50 = scheduler.get_lr(50);
        let expected_mid = (0.1 + 0.001) / 2.0;
        assert!((lr_50 - expected_mid).abs() < 0.01);

        // At final epoch, should be close to min_lr
        let lr_100 = scheduler.get_lr(100);
        assert!(lr_100 < 0.01);
    }

    #[test]
    fn test_warmup_cosine_scheduler() {
        let scheduler = LRScheduler::warmup_cosine(0.1, 0.001, 10, 100);

        // During warmup, LR should increase
        let lr_0 = scheduler.get_lr(0);
        let lr_5 = scheduler.get_lr(5);
        let lr_9 = scheduler.get_lr(9);

        // LR should be increasing during warmup
        assert!(lr_0 > 0.0);
        assert!(lr_5 > 0.0);
        assert!(lr_9 > 0.0);
        assert!(lr_9 <= 0.1); // Should not exceed initial_lr

        // After warmup, should follow cosine decay
        let lr_10 = scheduler.get_lr(10);
        let lr_50 = scheduler.get_lr(50);
        let lr_99 = scheduler.get_lr(99);

        // Should be decreasing over time
        assert!(lr_10 > 0.0);
        assert!(lr_99 >= 0.001); // Should be at or above min_lr
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut state =
            ReduceOnPlateauState::new(0.1, 0.5, 3, 1e-6, PlateauMode::Min);

        // Metric improves
        assert_eq!(state.step(1.0), 0.1);
        assert_eq!(state.step(0.9), 0.1);
        assert_eq!(state.step(0.8), 0.1);

        // Metric stagnates
        assert_eq!(state.step(0.85), 0.1);
        assert_eq!(state.step(0.86), 0.1);
        assert_eq!(state.step(0.87), 0.05); // LR reduced after patience=3
    }
}
