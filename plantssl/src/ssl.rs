//! Class-balanced self-training with pseudo-labels.
//!
//! The stream pool arrives in rounds, like new field photos would. Each round:
//!
//! 1. reveal the next slice of the stream;
//! 2. let the current model label every image revealed so far, so a label it got wrong in an
//!    earlier round can change or drop out;
//! 3. for every class, keep the most confident share of the images predicted as that class
//!    (class-balanced self-training, Zou et al. 2018). One global confidence threshold would
//!    mostly keep images of classes the model already knows: with 5% labels, a 0.95 threshold
//!    keeps 94% of the stream images of the easiest class and 18% of the hardest, and training
//!    on that skewed set lowers accuracy on the hard classes;
//! 4. train on the labeled pool plus the kept pseudo-labels: either fine-tune the same model, or
//!    with `fresh_student` train a new model from scratch (teacher-student self-training, as in
//!    Noisy Student). The student becomes the teacher of the next round.
//!
//! The teacher is only needed for its predictions, so one model is in memory at a time and
//! peak VRAM stays at the level of plain supervised training. The best round by validation
//! accuracy is kept.

use anyhow::{Result, ensure};
use burn::prelude::*;
use serde::Serialize;
use tracing::info;

use crate::data::Split;
use crate::eval::{self, Prediction};
use crate::model::{ModelMeta, PlantCnn};
use crate::progress::Observer;
use crate::train::{self, TrainConfig};

#[derive(Debug, Clone)]
pub struct SslConfig {
    pub rounds: usize,
    /// Share of each predicted class to keep as pseudo-labels, most confident first.
    pub keep: f32,
    /// Train a new model from scratch each round instead of fine-tuning the current one.
    pub fresh_student: bool,
    /// Training settings for each round.
    pub train: TrainConfig,
}

impl SslConfig {
    /// The settings shared by `plantssl ssl` and the app. A fresh student trains from scratch, so
    /// it gets the supervised learning rate and warmup; fine-tuning a trained model uses a lower
    /// rate and no warmup.
    pub fn new(
        rounds: usize,
        keep: f32,
        epochs_per_round: usize,
        fresh_student: bool,
        image_size: u32,
    ) -> Self {
        Self {
            rounds,
            keep,
            fresh_student,
            train: TrainConfig {
                epochs: epochs_per_round,
                learning_rate: if fresh_student { 1e-3 } else { 3e-4 },
                warmup_epochs: usize::from(fresh_student),
                image_size,
                ..TrainConfig::default()
            },
        }
    }
}

impl Default for SslConfig {
    fn default() -> Self {
        Self::new(10, 0.5, 3, false, TrainConfig::default().image_size)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RoundLog {
    pub round: usize,
    /// Stream images revealed so far.
    pub revealed: usize,
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

pub fn run(
    model: PlantCnn,
    meta: &ModelMeta,
    split: &Split,
    config: &SslConfig,
    device: &Device,
    observer: &mut dyn Observer,
) -> Result<SslRun> {
    ensure!(
        config.rounds > 0 && config.train.epochs > 0,
        "need at least one round and one epoch per round"
    );
    let tc = &config.train;
    let labeled = split.resolve(&split.labeled);
    let validation = split.resolve(&split.validation);
    let stream = split.resolve(&split.stream);
    ensure!(
        config.keep > 0.0 && config.keep <= 1.0,
        "keep must be in (0, 1]"
    );

    let initial = eval::accuracy(
        &eval::predict(
            &model,
            &validation,
            tc.image_size,
            tc.batch_size * 2,
            device,
            observer,
        )?,
        &validation,
    );
    info!("round 0: validation accuracy {:.2}%", initial * 100.0);
    let mut rounds = vec![RoundLog {
        round: 0,
        revealed: 0,
        accepted: 0,
        precision: 0.0,
        val_accuracy: initial,
    }];
    observer.round(&rounds[0]);
    let mut best = (0, initial, model.clone());
    let mut model = model;

    for round in 1..=config.rounds {
        let revealed = &stream[..stream.len() * round / config.rounds];
        let predictions = eval::predict(
            &model,
            revealed,
            tc.image_size,
            tc.batch_size * 2,
            device,
            observer,
        )?;
        let accepted = select(&predictions, split.classes.len(), config.keep);

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
        let student = if config.fresh_student {
            // The teacher's predictions are in, so free it before the student trains
            drop(model);
            PlantCnn::new(meta, device)
        } else {
            model
        };
        let fit = train::fit(
            student,
            &train_set,
            &validation,
            &round_config,
            device,
            observer,
        )?;
        // The next round's teacher is this round's best epoch.
        model = fit.model;

        let log = RoundLog {
            round,
            revealed: revealed.len(),
            accepted: accepted.len(),
            precision: correct as f64 / accepted.len().max(1) as f64,
            val_accuracy: fit.best_val_accuracy,
        };
        info!(
            "round {round}: {} revealed, {} accepted ({:.1}% correct), validation accuracy {:.2}%",
            log.revealed,
            log.accepted,
            log.precision * 100.0,
            log.val_accuracy * 100.0
        );
        observer.round(&log);
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

/// Indices of predictions to use as pseudo-labels: for each predicted class, the most confident
/// `keep` share of the images predicted as that class.
fn select(predictions: &[Prediction], classes: usize, keep: f32) -> Vec<usize> {
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); classes];
    for (i, p) in predictions.iter().enumerate() {
        by_class[p.label].push(i);
    }
    let mut accepted = Vec::new();
    for mut indices in by_class {
        indices.sort_by(|&a, &b| {
            predictions[b]
                .confidence
                .total_cmp(&predictions[a].confidence)
        });
        indices.truncate((indices.len() as f32 * keep).round() as usize);
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
    fn select_keeps_the_same_share_of_every_class() {
        // Class 0 is predicted four times with high confidence, class 1 twice with low confidence.
        let predictions = [
            p(0, 0.99),
            p(0, 0.97),
            p(1, 0.40),
            p(0, 0.98),
            p(1, 0.60),
            p(0, 0.90),
        ];
        // Half of each class: the two most confident of class 0, the more confident of class 1.
        assert_eq!(select(&predictions, 2, 0.5), vec![0, 3, 4]);
    }
}
