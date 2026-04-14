//! Shared utilities for inference-style command handlers.

use std::path::Path;

use burn::module::Module;
use burn::record::CompactRecorder;
use burn::tensor::{Tensor, TensorData};
use image::{DynamicImage, Rgb, RgbImage};

use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

use crate::backend::InferenceBackend;
use crate::state::AppBackend;

pub const CLASS_NAMES: [&str; 38] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
];

pub fn get_class_name(class_id: usize) -> String {
    if class_id < CLASS_NAMES.len() {
        CLASS_NAMES[class_id].to_string()
    } else {
        format!("Unknown_{}", class_id)
    }
}

pub fn load_inference_model(model_path: &Path) -> Result<PlantClassifier<InferenceBackend>, String> {
    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
    let model: PlantClassifier<InferenceBackend> = PlantClassifier::new(&model_config(), &device);
    let recorder = CompactRecorder::new();

    model
        .load_file(model_path, &recorder, &device)
        .map_err(|e| format!("Failed to load model: {:?}", e))
}

pub fn load_app_model(model_path: &Path) -> Result<PlantClassifier<AppBackend>, String> {
    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
    let model: PlantClassifier<AppBackend> = PlantClassifier::new(&model_config(), &device);
    let recorder = CompactRecorder::new();

    model
        .load_file(model_path, &recorder, &device)
        .map_err(|e| format!("Failed to load model: {:?}", e))
}

pub fn preprocess_image_for_inference(img: &DynamicImage, input_size: usize) -> Tensor<InferenceBackend, 4> {
    let device = <InferenceBackend as burn::tensor::backend::Backend>::Device::default();
    preprocess_image(img, input_size, &device)
}

pub fn preprocess_image_for_app(img: &DynamicImage, input_size: usize) -> Tensor<AppBackend, 4> {
    let device = <AppBackend as burn::tensor::backend::Backend>::Device::default();
    preprocess_image(img, input_size, &device)
}

fn model_config() -> PlantClassifierConfig {
    PlantClassifierConfig {
        num_classes: 38,
        input_size: 128,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    }
}

fn preprocess_image<B: burn::tensor::backend::Backend>(
    img: &DynamicImage,
    input_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let img = pil_bilinear_resize(img, input_size as u32, input_size as u32);
    let mut pixels: Vec<f32> = Vec::with_capacity(3 * input_size * input_size);

    for c in 0..3 {
        for y in 0..input_size {
            for x in 0..input_size {
                let pixel = img.get_pixel(x as u32, y as u32);
                pixels.push(pixel[c] as f32 / 255.0);
            }
        }
    }

    let tensor = Tensor::<B, 1>::from_floats(pixels.as_slice(), device)
        .reshape([1, 3, input_size, input_size]);
    let mean = Tensor::<B, 4>::from_floats(
        TensorData::new(vec![0.485f32, 0.456, 0.406], [1, 3, 1, 1]),
        device,
    );
    let std = Tensor::<B, 4>::from_floats(
        TensorData::new(vec![0.229f32, 0.224, 0.225], [1, 3, 1, 1]),
        device,
    );

    (tensor - mean) / std
}

fn pil_bilinear_resize(img: &DynamicImage, target_width: u32, target_height: u32) -> RgbImage {
    let src = img.to_rgb8();
    let src_width = src.width() as usize;
    let src_height = src.height() as usize;
    let target_width = target_width as usize;
    let target_height = target_height as usize;

    let mut dst = RgbImage::new(target_width as u32, target_height as u32);

    let x_scale = src_width as f32 / target_width as f32;
    let y_scale = src_height as f32 / target_height as f32;
    let support_x = x_scale.max(1.0);
    let support_y = y_scale.max(1.0);

    for dy in 0..target_height {
        for dx in 0..target_width {
            let src_cx = (dx as f32 + 0.5) * x_scale;
            let src_cy = (dy as f32 + 0.5) * y_scale;
            let x_min = (src_cx - support_x).floor().max(0.0) as usize;
            let x_max = (src_cx + support_x).ceil().min(src_width as f32 - 1.0) as usize;
            let y_min = (src_cy - support_y).floor().max(0.0) as usize;
            let y_max = (src_cy + support_y).ceil().min(src_height as f32 - 1.0) as usize;

            let mut total_weight = 0.0f32;
            let mut weighted_sum = [0.0f32; 3];

            for sy in y_min..=y_max {
                for sx in x_min..=x_max {
                    let dist_x = ((sx as f32 + 0.5) - src_cx).abs() / support_x;
                    let dist_y = ((sy as f32 + 0.5) - src_cy).abs() / support_y;

                    if dist_x < 1.0 && dist_y < 1.0 {
                        let weight_x = 1.0 - dist_x;
                        let weight_y = 1.0 - dist_y;
                        let weight = weight_x * weight_y;

                        let pixel = src.get_pixel(sx as u32, sy as u32);
                        weighted_sum[0] += pixel[0] as f32 * weight;
                        weighted_sum[1] += pixel[1] as f32 * weight;
                        weighted_sum[2] += pixel[2] as f32 * weight;
                        total_weight += weight;
                    }
                }
            }

            if total_weight > 0.0 {
                dst.put_pixel(
                    dx as u32,
                    dy as u32,
                    Rgb([
                        (weighted_sum[0] / total_weight).round() as u8,
                        (weighted_sum[1] / total_weight).round() as u8,
                        (weighted_sum[2] / total_weight).round() as u8,
                    ]),
                );
            }
        }
    }

    dst
}
