//! Image decoding, augmentation and batching.
//!
//! Batches are decoded on a background thread with rayon while the GPU works on the previous
//! batch, so the whole dataset never has to sit in memory.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result};
use burn::prelude::*;
use image::imageops::{self, FilterType};
use image::{DynamicImage, RgbImage};
use rand::{Rng, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// ImageNet channel statistics. Close enough for leaf photos and a common default.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A batch ready for the model: `[n, 3, size, size]` images and their class indices.
pub struct Batch {
    pub images: Tensor<4>,
    pub labels: Vec<usize>,
}

impl Batch {
    pub fn targets(&self, device: &Device) -> Tensor<1, Int> {
        let labels: Vec<i64> = self.labels.iter().map(|&l| l as i64).collect();
        Tensor::from_data(TensorData::new(labels, [self.labels.len()]), device)
    }
}

/// Decodes an image file and resizes it to `size` x `size`.
pub fn load(path: &Path, size: u32) -> Result<RgbImage> {
    let image = image::open(path).with_context(|| format!("cannot decode {}", path.display()))?;
    Ok(resize(&image, size))
}

pub fn resize(image: &DynamicImage, size: u32) -> RgbImage {
    imageops::resize(&image.to_rgb8(), size, size, FilterType::Triangle)
}

/// Label-preserving augmentation: random crop, flips, quarter turns and colour jitter.
pub fn augment(image: &RgbImage, rng: &mut impl Rng) -> RgbImage {
    let size = image.width();

    let crop = (size as f32 * rng.random_range(0.75..=1.0)) as u32;
    let x = rng.random_range(0..=size - crop);
    let y = rng.random_range(0..=size - crop);
    let mut out = imageops::crop_imm(image, x, y, crop, crop).to_image();
    if crop != size {
        out = imageops::resize(&out, size, size, FilterType::Triangle);
    }

    if rng.random_bool(0.5) {
        imageops::flip_horizontal_in_place(&mut out);
    }
    if rng.random_bool(0.5) {
        imageops::flip_vertical_in_place(&mut out);
    }
    out = match rng.random_range(0..4) {
        1 => imageops::rotate90(&out),
        2 => imageops::rotate180(&out),
        3 => imageops::rotate270(&out),
        _ => out,
    };

    let brightness: f32 = rng.random_range(-0.15..=0.15) * 255.0;
    let contrast: f32 = rng.random_range(0.8..=1.2);
    let saturation: f32 = rng.random_range(0.8..=1.2);
    for pixel in out.pixels_mut() {
        let [r, g, b] = pixel.0.map(f32::from);
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        pixel.0 = [r, g, b].map(|c| {
            let c = gray + (c - gray) * saturation;
            let c = (c - 128.0) * contrast + 128.0 + brightness;
            c.clamp(0.0, 255.0) as u8
        });
    }
    out
}

/// Appends an image to `out` in normalized channel-first layout.
fn push_chw(image: &RgbImage, out: &mut Vec<f32>) {
    for (channel, (mean, std)) in MEAN.iter().zip(STD).enumerate() {
        out.extend(
            image
                .pixels()
                .map(|p| (f32::from(p.0[channel]) / 255.0 - mean) / std),
        );
    }
}

/// Turns decoded images into a `[n, 3, size, size]` tensor.
pub fn to_tensor(images: &[RgbImage], device: &Device) -> Tensor<4> {
    let size = images.first().map_or(0, |i| i.width() as usize);
    let mut data = Vec::with_capacity(images.len() * 3 * size * size);
    for image in images {
        push_chw(image, &mut data);
    }
    Tensor::from_data(TensorData::new(data, [images.len(), 3, size, size]), device)
}

/// How to read a list of images in batches.
#[derive(Debug, Clone, Copy)]
pub struct Loader {
    pub batch_size: usize,
    pub image_size: u32,
    /// Augments every image with this seed when set.
    pub augment_seed: Option<u64>,
}

impl Loader {
    /// Calls `step` for every batch, in order. Decoding runs one batch ahead on other threads.
    pub fn for_each(
        &self,
        samples: &[(PathBuf, usize)],
        device: &Device,
        mut step: impl FnMut(Batch) -> Result<()>,
    ) -> Result<()> {
        let (sender, receiver) = mpsc::sync_channel::<Result<(Vec<RgbImage>, Vec<usize>)>>(2);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                for (index, chunk) in samples.chunks(self.batch_size).enumerate() {
                    let batch = self.decode(chunk, index);
                    if sender.send(batch).is_err() {
                        break;
                    }
                }
            });
            for batch in receiver {
                let (images, labels) = batch?;
                step(Batch {
                    images: to_tensor(&images, device),
                    labels,
                })?;
            }
            Ok(())
        })
    }

    fn decode(
        &self,
        chunk: &[(PathBuf, usize)],
        index: usize,
    ) -> Result<(Vec<RgbImage>, Vec<usize>)> {
        let images = chunk
            .par_iter()
            .enumerate()
            .map(|(i, (path, _))| {
                let image = load(path, self.image_size)?;
                Ok(match self.augment_seed {
                    Some(seed) => {
                        let item = (index * self.batch_size + i) as u64;
                        let mut rng = ChaCha8Rng::seed_from_u64(
                            seed ^ item.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                        );
                        augment(&image, &mut rng)
                    }
                    None => image,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((images, chunk.iter().map(|(_, label)| *label).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn augment_keeps_size() {
        let image = RgbImage::from_fn(32, 32, |x, y| image::Rgb([x as u8 * 8, y as u8 * 8, 100]));
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..20 {
            let out = augment(&image, &mut rng);
            assert_eq!(out.dimensions(), (32, 32));
        }
    }

    #[test]
    fn tensor_is_channel_first_and_normalized() {
        let image = RgbImage::from_pixel(4, 4, image::Rgb([255, 0, 0]));
        let tensor = to_tensor(&[image], &Device::flex());
        assert_eq!(tensor.dims(), [1, 3, 4, 4]);
        let values: Vec<f32> = tensor.into_data().try_to_vec_as().unwrap();
        assert!((values[0] - (1.0 - MEAN[0]) / STD[0]).abs() < 1e-5);
        assert!((values[16] - (0.0 - MEAN[1]) / STD[1]).abs() < 1e-5);
    }
}
