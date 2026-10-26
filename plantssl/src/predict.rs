//! Single-image classification with a saved model.

use std::path::Path;

use anyhow::Result;
use burn::prelude::*;
use image::DynamicImage;
use serde::Serialize;

use crate::images;
use crate::model::{ModelMeta, PlantCnn};

#[derive(Debug, Clone, Serialize)]
pub struct ClassScore {
    pub class: String,
    pub probability: f32,
}

/// A loaded model, ready to classify images one at a time.
pub struct Predictor {
    model: PlantCnn,
    meta: ModelMeta,
    device: Device,
}

impl Predictor {
    pub fn load(model_dir: &Path, device: Device) -> Result<Self> {
        let (model, meta) = PlantCnn::load(model_dir, &device)?;
        Ok(Self {
            model: model.valid(),
            meta,
            device,
        })
    }

    pub fn meta(&self) -> &ModelMeta {
        &self.meta
    }

    /// All classes, most likely first.
    pub fn classify(&self, image: &DynamicImage) -> Vec<ClassScore> {
        let input = images::to_tensor(&[images::resize(image, self.meta.image_size)], &self.device);
        let probabilities: Vec<f32> = self
            .model
            .probabilities(input)
            .into_data()
            .try_to_vec_as()
            .expect("f32 tensor");
        let mut scores: Vec<ClassScore> = self
            .meta
            .classes
            .iter()
            .zip(probabilities)
            .map(|(class, probability)| ClassScore {
                class: class.clone(),
                probability,
            })
            .collect();
        scores.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        scores
    }
}
