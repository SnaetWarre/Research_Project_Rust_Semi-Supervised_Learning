//! Supervised training loop, shared by the first model and every SSL round.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use tracing::info;

use crate::eval;
use crate::images::Loader;
use crate::model::PlantCnn;

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    /// Epochs of linear learning-rate warmup before the cosine decay.
    pub warmup_epochs: usize,
    pub image_size: u32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch_size: 64,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            warmup_epochs: 1,
            image_size: 128,
            seed: 42,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochLog {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_accuracy: f64,
    pub seconds: f64,
}

pub struct Fit {
    /// The weights from the epoch with the best validation accuracy.
    pub model: PlantCnn,
    pub best_epoch: usize,
    pub best_val_accuracy: f64,
    pub history: Vec<EpochLog>,
}

/// Trains with AdamW and a warmup-then-cosine learning rate, and keeps the epoch with the best
/// validation accuracy.
pub fn fit(
    model: PlantCnn,
    train: &[(PathBuf, usize)],
    validation: &[(PathBuf, usize)],
    config: &TrainConfig,
    device: &Device,
) -> Result<Fit> {
    let train_device = device.clone().autodiff();
    // `Option` so the closure below can move the model into `optimizer.step`
    let mut model = Some(model.to_device(&train_device).train());
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&train_device);

    let steps_per_epoch = train.len().div_ceil(config.batch_size);
    let schedule = Schedule {
        base: config.learning_rate,
        warmup: config.warmup_epochs * steps_per_epoch,
        total: config.epochs * steps_per_epoch,
    };

    let mut order = train.to_vec();
    let mut step = 0;
    let mut best: Option<(usize, f64, PlantCnn)> = None;
    let mut history = Vec::with_capacity(config.epochs);

    for epoch in 1..=config.epochs {
        let started = Instant::now();
        let seed = config.seed.wrapping_mul(1000).wrapping_add(epoch as u64);
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
        let loader = Loader {
            batch_size: config.batch_size,
            image_size: config.image_size,
            augment_seed: Some(seed),
        };

        let progress = crate::progress(order.len(), &format!("epoch {epoch}/{}", config.epochs));
        let mut loss_sum = 0.0;
        loader.for_each(&order, &train_device, |batch| {
            let targets = batch.targets(&train_device);
            let count = batch.labels.len();
            let current = model.take().expect("model is put back after every step");
            let loss = loss_fn.forward(current.forward(batch.images), targets);
            let grads = GradientsParams::from_grads(loss.backward(), &current);
            model = Some(optimizer.step(schedule.at(step), current, grads));
            step += 1;
            loss_sum += loss.into_scalar::<f64>() * count as f64;
            progress.inc(count as u64);
            Ok(())
        })?;
        progress.finish_and_clear();

        let current = model.as_ref().expect("model is put back after every step");
        let predictions = eval::predict(
            current,
            validation,
            config.image_size,
            config.batch_size * 2,
            device,
        )?;
        let log = EpochLog {
            epoch,
            train_loss: loss_sum / order.len().max(1) as f64,
            val_accuracy: eval::accuracy(&predictions, validation),
            seconds: started.elapsed().as_secs_f64(),
        };
        info!(
            "epoch {:>3}: loss {:.4}, validation accuracy {:.2}% ({:.0}s)",
            log.epoch,
            log.train_loss,
            log.val_accuracy * 100.0,
            log.seconds
        );
        if best
            .as_ref()
            .is_none_or(|(_, acc, _)| log.val_accuracy > *acc)
        {
            best = Some((epoch, log.val_accuracy, current.valid()));
        }
        history.push(log);
    }

    let (best_epoch, best_val_accuracy, model) = best.expect("at least one epoch");
    Ok(Fit {
        model,
        best_epoch,
        best_val_accuracy,
        history,
    })
}

struct Schedule {
    base: f64,
    warmup: usize,
    total: usize,
}

impl Schedule {
    fn at(&self, step: usize) -> f64 {
        if step < self.warmup {
            return self.base * (step + 1) as f64 / self.warmup as f64;
        }
        let progress = (step - self.warmup) as f64 / (self.total - self.warmup).max(1) as f64;
        let floor = self.base * 0.01;
        floor + (self.base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_warms_up_then_decays() {
        let s = Schedule {
            base: 1.0,
            warmup: 10,
            total: 110,
        };
        assert!((s.at(0) - 0.1).abs() < 1e-12);
        assert!((s.at(9) - 1.0).abs() < 1e-12);
        assert!((s.at(10) - 1.0).abs() < 1e-12);
        assert!(s.at(60) < s.at(30));
        assert!((s.at(109) - 0.01).abs() < 1e-3);
    }
}
