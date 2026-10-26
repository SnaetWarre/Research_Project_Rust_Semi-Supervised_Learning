//! A small convolutional network sized for edge devices.
//!
//! A stride-2 stem convolution halves the input first, which makes training about twice as
//! fast as running the first stage at full resolution. Three stages of two 3x3 convolutions
//! with batch norm, ReLU and 2x2 max pooling follow, then global average pooling and one
//! linear layer. About 1.2M parameters at the default width.

use std::path::Path;

use anyhow::{Context, Result};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig};
use burn::nn::{
    BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use serde::{Deserialize, Serialize};

const WEIGHTS_FILE: &str = "model.bpk";
const META_FILE: &str = "model.json";

/// Everything needed to rebuild a model and read its output, stored next to the weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    pub classes: Vec<String>,
    pub image_size: u32,
    /// Channels in the stem; each stage doubles it.
    pub width: usize,
}

fn conv3x3(channels_in: usize, channels_out: usize, stride: usize, device: &Device) -> Conv2d {
    Conv2dConfig::new([channels_in, channels_out], [3, 3])
        .with_stride([stride, stride])
        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
        .with_bias(false)
        .init(device)
}

#[derive(Module, Debug)]
struct Stage {
    conv1: Conv2d,
    norm1: BatchNorm,
    conv2: Conv2d,
    norm2: BatchNorm,
    pool: MaxPool2d,
    relu: Relu,
}

impl Stage {
    fn new(channels_in: usize, channels_out: usize, device: &Device) -> Self {
        Self {
            conv1: conv3x3(channels_in, channels_out, 1, device),
            norm1: BatchNormConfig::new(channels_out).init(device),
            conv2: conv3x3(channels_out, channels_out, 1, device),
            norm2: BatchNormConfig::new(channels_out).init(device),
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.relu.forward(self.norm1.forward(self.conv1.forward(x)));
        let x = self.relu.forward(self.norm2.forward(self.conv2.forward(x)));
        self.pool.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PlantCnn {
    stem: Conv2d,
    stem_norm: BatchNorm,
    relu: Relu,
    stages: Vec<Stage>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    head: Linear,
}

impl PlantCnn {
    pub fn new(meta: &ModelMeta, device: &Device) -> Self {
        let widths = [meta.width * 2, meta.width * 4, meta.width * 8];
        let mut channels_in = meta.width;
        let stages = widths
            .iter()
            .map(|&channels_out| {
                let stage = Stage::new(channels_in, channels_out, device);
                channels_in = channels_out;
                stage
            })
            .collect();
        Self {
            stem: conv3x3(3, meta.width, 2, device),
            stem_norm: BatchNormConfig::new(meta.width).init(device),
            relu: Relu::new(),
            stages,
            pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            dropout: DropoutConfig::new(0.2).init(),
            head: LinearConfig::new(channels_in, meta.classes.len()).init(device),
        }
    }

    /// Returns logits of shape `[batch, classes]`.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<2> {
        let x = self
            .relu
            .forward(self.stem_norm.forward(self.stem.forward(images)));
        let x = self.stages.iter().fold(x, |x, stage| stage.forward(x));
        let x: Tensor<2> = self.pool.forward(x).flatten(1, 3);
        self.head.forward(self.dropout.forward(x))
    }

    pub fn probabilities(&self, images: Tensor<4>) -> Tensor<2> {
        softmax(self.forward(images), 1)
    }

    pub fn save(&self, dir: &Path, meta: &ModelMeta) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        self.clone()
            .save_file(dir.join(WEIGHTS_FILE))
            .map_err(|e| anyhow::anyhow!("cannot save model: {e:?}"))?;
        crate::write_json(&dir.join(META_FILE), meta)
    }

    pub fn load(dir: &Path, device: &Device) -> Result<(Self, ModelMeta)> {
        let meta: ModelMeta = serde_json::from_str(
            &std::fs::read_to_string(dir.join(META_FILE))
                .with_context(|| format!("no model in {}", dir.display()))?,
        )?;
        let model = Self::new(&meta, device)
            .try_load_file(dir.join(WEIGHTS_FILE))
            .map_err(|e| anyhow::anyhow!("cannot load {}: {e:?}", dir.display()))?;
        Ok((model, meta))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ModelMeta {
        ModelMeta {
            classes: vec!["a".into(), "b".into(), "c".into()],
            image_size: 32,
            width: 8,
        }
    }

    #[test]
    fn output_has_one_logit_per_class() {
        let device = Device::flex();
        let model = PlantCnn::new(&meta(), &device);
        let logits = model.forward(Tensor::zeros([2, 3, 32, 32], &device));
        assert_eq!(logits.dims(), [2, 3]);
    }

    #[test]
    fn save_then_load_gives_the_same_output() {
        let device = Device::flex();
        let dir = tempfile::tempdir().unwrap();
        let model = PlantCnn::new(&meta(), &device).valid();
        model.save(dir.path(), &meta()).unwrap();
        let (loaded, loaded_meta) = PlantCnn::load(dir.path(), &device).unwrap();
        assert_eq!(loaded_meta.classes, meta().classes);

        let input =
            Tensor::<4>::random([1, 3, 32, 32], burn::tensor::Distribution::Default, &device);
        let a: Vec<f32> = model
            .forward(input.clone())
            .into_data()
            .try_to_vec_as()
            .unwrap();
        let b: Vec<f32> = loaded
            .valid()
            .forward(input)
            .into_data()
            .try_to_vec_as()
            .unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5);
        }
    }
}
