//! Self-training with confidence-thresholded pseudo-labels.
//!
//! The stream pool arrives in rounds, like new field photos would. Each round:
//!
//! 1. reveal the next slice of the stream;
//! 2. let the current model label every image revealed so far, so a label it got wrong in an
//!    earlier round can change or drop out;
//! 3. keep labels at or above the confidence threshold, capped per class at that class's
//!    expected share of the revealed images, so easy classes cannot flood the training set;
//! 4. fine-tune the same model on the labeled pool plus the kept pseudo-labels.
//!
//! Only one model is in memory at any time, which keeps peak VRAM at the level of plain
//! supervised training. The best round by validation accuracy is kept.

use anyhow::{Result, ensure};
use burn::prelude::*;
use serde::Serialize;
use tracing::info;

use crate::data::Split;
use crate::eval::{self, Prediction};
use crate::model::PlantCnn;
use crate::train::{self, TrainConfig};

#[derive(Debug, Clone)]
pub struct SslConfig {
    pub rounds: usize,
    /// Minimum softmax probability for a pseudo-label.
    pub threshold: f32,
    /// Training settings for the fine-tuning in each round.
    pub train: TrainConfig,
}

impl Default for SslConfig {
    fn default() -> Self {
        Self {
            rounds: 10,
            threshold: 0.95,
            train: TrainConfig {
                epochs: 3,
                learning_rate: 3e-4,
                warmup_epochs: 0,
                ..TrainConfig::default()
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundLog {
    pub round: usize,
    /// Stream images revealed so far.
    pub revealed: usize,
    /// Revealed images at or above the threshold, before the per-class cap.
    pub confident: usize,
    /// Pseudo-labels used for training this round.
    pub accepted: usize,
    /// Share of accepted pseudo-labels that match the hidden true label.
    pub precision: f64,
    pub val_accuracy: f64,
}

pub struct SslRun {
    pub model: PlantCnn,
    pub best_round: usize,
    pub rounds: Vec<RoundLog>,
}

pub fn run(model: PlantCnn, split: &Split, config: &SslConfig, device: &Device) -> Result<SslRun> {
    ensure!(config.rounds > 0, "need at least one round");
    let tc = &config.train;
    let labeled = split.resolve(&split.labeled);
    let validation = split.resolve(&split.validation);
    let stream = split.resolve(&split.stream);
    let quota = ClassQuota::from_labeled(&split.labeled, split.classes.len());

    let initial = eval::accuracy(
        &eval::predict(
            &model,
            &validation,
            tc.image_size,
            tc.batch_size * 2,
            device,
        )?,
        &validation,
    );
    info!("round 0: validation accuracy {:.2}%", initial * 100.0);
    let mut rounds = vec![RoundLog {
        round: 0,
        revealed: 0,
        confident: 0,
        accepted: 0,
        precision: 0.0,
        val_accuracy: initial,
    }];
    let mut best = (0, initial, model.clone());
    let mut model = model;

    for round in 1..=config.rounds {
        let revealed = &stream[..stream.len() * round / config.rounds];
        let predictions =
            eval::predict(&model, revealed, tc.image_size, tc.batch_size * 2, device)?;
        let confident = predictions
            .iter()
            .filter(|p| p.confidence >= config.threshold)
            .count();
        let accepted = select(&predictions, config.threshold, &quota);

        // The hidden labels are read here, and only here, to report precision.
        let correct = accepted
            .iter()
            .filter(|&&i| predictions[i].label == revealed[i].1)
            .count();

        let mut train_set = labeled.clone();
        train_set.extend(
            accepted
                .iter()
                .map(|&i| (revealed[i].0.clone(), predictions[i].label)),
        );
        let round_config = TrainConfig {
            seed: tc.seed.wrapping_add(round as u64),
            ..tc.clone()
        };
        let fit = train::fit(model, &train_set, &validation, &round_config, device)?;
        // The next round continues from this round's best epoch.
        model = fit.model;

        let log = RoundLog {
            round,
            revealed: revealed.len(),
            confident,
            accepted: accepted.len(),
            precision: correct as f64 / accepted.len().max(1) as f64,
            val_accuracy: fit.best_val_accuracy,
        };
        info!(
            "round {round}: {} revealed, {} confident, {} accepted ({:.1}% correct), validation accuracy {:.2}%",
            log.revealed,
            log.confident,
            log.accepted,
            log.precision * 100.0,
            log.val_accuracy * 100.0
        );
        if log.val_accuracy > best.1 {
            best = (round, log.val_accuracy, model.clone());
        }
        rounds.push(log);
    }

    Ok(SslRun {
        model: best.2,
        best_round: best.0,
        rounds,
    })
}

/// How many pseudo-labels each class may receive, from its share of the labeled pool.
struct ClassQuota {
    shares: Vec<f64>,
}

impl ClassQuota {
    fn from_labeled(labeled: &[crate::data::Sample], classes: usize) -> Self {
        let mut counts = vec![0usize; classes];
        for sample in labeled {
            counts[sample.label] += 1;
        }
        let total = labeled.len().max(1) as f64;
        Self {
            shares: counts.iter().map(|&c| c as f64 / total).collect(),
        }
    }

    fn limit(&self, class: usize, revealed: usize) -> usize {
        (self.shares[class] * revealed as f64).ceil() as usize
    }
}

/// Indices of predictions to use as pseudo-labels: above the threshold, and per predicted class
/// the most confident ones up to that class's quota.
fn select(predictions: &[Prediction], threshold: f32, quota: &ClassQuota) -> Vec<usize> {
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); quota.shares.len()];
    for (i, p) in predictions.iter().enumerate() {
        if p.confidence >= threshold {
            by_class[p.label].push(i);
        }
    }
    let mut accepted = Vec::new();
    for (class, mut indices) in by_class.into_iter().enumerate() {
        indices.sort_by(|&a, &b| {
            predictions[b]
                .confidence
                .total_cmp(&predictions[a].confidence)
        });
        indices.truncate(quota.limit(class, predictions.len()));
        accepted.extend(indices);
    }
    accepted.sort_unstable();
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(label: usize, confidence: f32) -> Prediction {
        Prediction { label, confidence }
    }

    #[test]
    fn select_applies_threshold_and_class_cap() {
        // Two equally common classes, so each may take half of the four revealed images.
        let quota = ClassQuota {
            shares: vec![0.5, 0.5],
        };
        let predictions = [p(0, 0.99), p(0, 0.97), p(0, 0.98), p(1, 0.5)];
        // Class 0 has three confident predictions but a cap of two: the two most confident win.
        assert_eq!(select(&predictions, 0.9, &quota), vec![0, 2]);
    }
}
