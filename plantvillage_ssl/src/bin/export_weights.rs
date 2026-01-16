//! Weight Export Tool for Mobile Inference
//!
//! Exports trained Burn model weights to JSON format for loading
//! into PyTorch and subsequent ONNX export for mobile/web inference.
//!
//! Usage:
//!   cargo run --release --bin export_weights -- --model best_model.mpk --output mobile_export/weights

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use burn::{
    module::Module,
    record::CompactRecorder,
    tensor::{backend::Backend, Tensor},
};
use clap::Parser;
use serde::Serialize;

use plantvillage_ssl::backend::{default_device, DefaultBackend};
use plantvillage_ssl::model::cnn::{PlantClassifier, PlantClassifierConfig};

/// Export Burn model weights to JSON format for PyTorch loading
#[derive(Parser, Debug)]
#[command(name = "export_weights")]
#[command(about = "Export trained model weights for PyTorch/ONNX conversion")]
struct Args {
    /// Path to the trained model (.mpk file)
    #[arg(short, long)]
    model: PathBuf,

    /// Output directory for exported weights
    #[arg(short, long, default_value = "mobile_export/weights")]
    output: PathBuf,

    /// Number of classes in the model
    #[arg(long, default_value = "39")]
    num_classes: usize,
}

/// Tensor data for JSON export
#[derive(Serialize)]
struct TensorData {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// All model weights organized by layer
#[derive(Serialize)]
struct ModelWeights {
    // Conv block 1
    conv1_conv_weight: TensorData,
    conv1_conv_bias: TensorData,
    conv1_bn_gamma: TensorData,
    conv1_bn_beta: TensorData,
    conv1_bn_running_mean: TensorData,
    conv1_bn_running_var: TensorData,

    // Conv block 2
    conv2_conv_weight: TensorData,
    conv2_conv_bias: TensorData,
    conv2_bn_gamma: TensorData,
    conv2_bn_beta: TensorData,
    conv2_bn_running_mean: TensorData,
    conv2_bn_running_var: TensorData,

    // Conv block 3
    conv3_conv_weight: TensorData,
    conv3_conv_bias: TensorData,
    conv3_bn_gamma: TensorData,
    conv3_bn_beta: TensorData,
    conv3_bn_running_mean: TensorData,
    conv3_bn_running_var: TensorData,

    // Conv block 4
    conv4_conv_weight: TensorData,
    conv4_conv_bias: TensorData,
    conv4_bn_gamma: TensorData,
    conv4_bn_beta: TensorData,
    conv4_bn_running_mean: TensorData,
    conv4_bn_running_var: TensorData,

    // FC layers
    fc1_weight: TensorData,
    fc1_bias: TensorData,
    fc2_weight: TensorData,
    fc2_bias: TensorData,
}

/// Convert a Burn tensor to TensorData for JSON serialization
fn tensor_to_data<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> TensorData {
    let shape: Vec<usize> = tensor.dims().to_vec();
    let data: Vec<f32> = tensor.into_data().to_vec().unwrap();
    TensorData { shape, data }
}

/// Extract weights from the model
fn extract_weights<B: Backend>(model: &PlantClassifier<B>) -> ModelWeights {
    ModelWeights {
        // Conv block 1
        conv1_conv_weight: tensor_to_data(model.conv1.conv.weight.val()),
        conv1_conv_bias: tensor_to_data(
            model.conv1.conv.bias.clone()
                .expect("Conv bias should exist")
                .val()
        ),
        conv1_bn_gamma: tensor_to_data(model.conv1.bn.gamma.val()),
        conv1_bn_beta: tensor_to_data(model.conv1.bn.beta.val()),
        conv1_bn_running_mean: tensor_to_data(model.conv1.bn.running_mean.value()),
        conv1_bn_running_var: tensor_to_data(model.conv1.bn.running_var.value()),

        // Conv block 2
        conv2_conv_weight: tensor_to_data(model.conv2.conv.weight.val()),
        conv2_conv_bias: tensor_to_data(
            model.conv2.conv.bias.clone()
                .expect("Conv bias should exist")
                .val()
        ),
        conv2_bn_gamma: tensor_to_data(model.conv2.bn.gamma.val()),
        conv2_bn_beta: tensor_to_data(model.conv2.bn.beta.val()),
        conv2_bn_running_mean: tensor_to_data(model.conv2.bn.running_mean.value()),
        conv2_bn_running_var: tensor_to_data(model.conv2.bn.running_var.value()),

        // Conv block 3
        conv3_conv_weight: tensor_to_data(model.conv3.conv.weight.val()),
        conv3_conv_bias: tensor_to_data(
            model.conv3.conv.bias.clone()
                .expect("Conv bias should exist")
                .val()
        ),
        conv3_bn_gamma: tensor_to_data(model.conv3.bn.gamma.val()),
        conv3_bn_beta: tensor_to_data(model.conv3.bn.beta.val()),
        conv3_bn_running_mean: tensor_to_data(model.conv3.bn.running_mean.value()),
        conv3_bn_running_var: tensor_to_data(model.conv3.bn.running_var.value()),

        // Conv block 4
        conv4_conv_weight: tensor_to_data(model.conv4.conv.weight.val()),
        conv4_conv_bias: tensor_to_data(
            model.conv4.conv.bias.clone()
                .expect("Conv bias should exist")
                .val()
        ),
        conv4_bn_gamma: tensor_to_data(model.conv4.bn.gamma.val()),
        conv4_bn_beta: tensor_to_data(model.conv4.bn.beta.val()),
        conv4_bn_running_mean: tensor_to_data(model.conv4.bn.running_mean.value()),
        conv4_bn_running_var: tensor_to_data(model.conv4.bn.running_var.value()),

        // FC layers
        fc1_weight: tensor_to_data(model.fc1.weight.val()),
        fc1_bias: tensor_to_data(
            model.fc1.bias.clone()
                .expect("FC1 bias should exist")
                .val()
        ),
        fc2_weight: tensor_to_data(model.fc2.weight.val()),
        fc2_bias: tensor_to_data(
            model.fc2.bias.clone()
                .expect("FC2 bias should exist")
                .val()
        ),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = default_device();

    println!("=== Burn Weight Export Tool ===");
    println!("Loading model from: {:?}", args.model);

    // Create model config matching the trained model
    let config = PlantClassifierConfig {
        num_classes: args.num_classes,
        input_size: 256,
        dropout_rate: 0.3,
        in_channels: 3,
        base_filters: 32,
    };

    // Load the trained model
    let recorder = CompactRecorder::new();
    let model: PlantClassifier<DefaultBackend> = PlantClassifier::new(&config, &device)
        .load_file(&args.model, &recorder, &device)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;

    println!("Model loaded successfully!");
    println!("Extracting weights...");

    // Extract all weights
    let weights = extract_weights(&model);

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Save as JSON (for Python to load)
    let json_path = args.output.join("weights.json");
    println!("Saving weights to: {:?}", json_path);
    
    let json_data = serde_json::to_string(&weights)?;
    let mut file = File::create(&json_path)?;
    file.write_all(json_data.as_bytes())?;

    println!();
    println!("Export complete!");
    println!();
    println!("Weight shapes:");
    println!("  conv1.conv.weight: {:?}", weights.conv1_conv_weight.shape);
    println!("  conv2.conv.weight: {:?}", weights.conv2_conv_weight.shape);
    println!("  conv3.conv.weight: {:?}", weights.conv3_conv_weight.shape);
    println!("  conv4.conv.weight: {:?}", weights.conv4_conv_weight.shape);
    println!("  fc1.weight: {:?}", weights.fc1_weight.shape);
    println!("  fc2.weight: {:?}", weights.fc2_weight.shape);
    println!();
    println!("Next steps:");
    println!("  1. Run: python mobile_export/load_weights.py");
    println!("  2. This will create model.onnx for web inference");

    Ok(())
}
