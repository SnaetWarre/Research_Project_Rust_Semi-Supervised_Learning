#!/usr/bin/env python3
"""
Load Burn model weights into PyTorch and export to ONNX.

This script:
1. Loads weights exported from Burn (JSON format)
2. Creates a PyTorch model with matching architecture
3. Loads the weights into the PyTorch model
4. Exports to ONNX format for web/mobile inference

Usage:
    python load_weights.py --weights weights/weights.json --output model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PlantVillage dataset configuration
NUM_CLASSES = 38
IMAGE_SIZE = 128  # Model was trained on 128x128 images (not 256!)

# Class names for the 38 disease classes - MUST be in alphabetical order to match Rust training!
CLASS_NAMES = [
    "Apple___Apple_scab",                                   # 0
    "Apple___Black_rot",                                    # 1
    "Apple___Cedar_apple_rust",                             # 2
    "Apple___healthy",                                      # 3
    "Blueberry___healthy",                                  # 4
    "Cherry_(including_sour)___Powdery_mildew",             # 5
    "Cherry_(including_sour)___healthy",                    # 6
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",   # 7
    "Corn_(maize)___Common_rust_",                          # 8
    "Corn_(maize)___Northern_Leaf_Blight",                  # 9
    "Corn_(maize)___healthy",                               # 10
    "Grape___Black_rot",                                    # 11
    "Grape___Esca_(Black_Measles)",                         # 12
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",           # 13
    "Grape___healthy",                                      # 14
    "Orange___Haunglongbing_(Citrus_greening)",             # 15
    "Peach___Bacterial_spot",                               # 16
    "Peach___healthy",                                      # 17
    "Pepper,_bell___Bacterial_spot",                        # 18
    "Pepper,_bell___healthy",                               # 19
    "Potato___Early_blight",                                # 20
    "Potato___Late_blight",                                 # 21
    "Potato___healthy",                                     # 22
    "Raspberry___healthy",                                  # 23
    "Soybean___healthy",                                    # 24
    "Squash___Powdery_mildew",                              # 25
    "Strawberry___Leaf_scorch",                             # 26
    "Strawberry___healthy",                                 # 27
    "Tomato___Bacterial_spot",                              # 28
    "Tomato___Early_blight",                                # 29
    "Tomato___Late_blight",                                 # 30
    "Tomato___Leaf_Mold",                                   # 31
    "Tomato___Septoria_leaf_spot",                          # 32
    "Tomato___Spider_mites Two-spotted_spider_mite",        # 33
    "Tomato___Target_Spot",                                 # 34
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",               # 35
    "Tomato___Tomato_mosaic_virus",                         # 36
    "Tomato___healthy",                                     # 37
]



class ConvBlock(nn.Module):
    """Convolutional block matching Burn architecture."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        with_pool: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2) if with_pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        if self.pool:
            x = self.pool(x)
        return x


class PlantClassifier(nn.Module):
    """
    Plant Disease Classifier CNN - matches Burn architecture exactly.
    
    Architecture:
    - 4 conv blocks: 3->32->64->128->256 with BatchNorm, ReLU, MaxPool
    - Global Average Pooling
    - FC: 256->256->num_classes with ReLU and Dropout
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 3,
        base_filters: int = 32,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Convolutional blocks: 3 -> 32 -> 64 -> 128 -> 256
        self.conv1 = ConvBlock(input_channels, base_filters, 3, True)
        self.conv2 = ConvBlock(base_filters, base_filters * 2, 3, True)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4, 3, True)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8, 3, True)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (matches Burn: 256 -> 256 -> num_classes)
        self.fc1 = nn.Linear(base_filters * 8, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax for inference."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def load_tensor(data: dict) -> torch.Tensor:
    """Convert JSON tensor data to PyTorch tensor."""
    shape = data["shape"]
    values = data["data"]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def load_weights_from_json(model: PlantClassifier, weights_path: Path) -> None:
    """Load weights from JSON file exported by Burn."""
    print(f"Loading weights from: {weights_path}")
    
    with open(weights_path) as f:
        weights = json.load(f)
    
    # Load conv block 1
    model.conv1.conv.weight.data = load_tensor(weights["conv1_conv_weight"])
    model.conv1.conv.bias.data = load_tensor(weights["conv1_conv_bias"])
    model.conv1.bn.weight.data = load_tensor(weights["conv1_bn_gamma"])
    model.conv1.bn.bias.data = load_tensor(weights["conv1_bn_beta"])
    model.conv1.bn.running_mean.data = load_tensor(weights["conv1_bn_running_mean"])
    model.conv1.bn.running_var.data = load_tensor(weights["conv1_bn_running_var"])

    # Load conv block 2
    model.conv2.conv.weight.data = load_tensor(weights["conv2_conv_weight"])
    model.conv2.conv.bias.data = load_tensor(weights["conv2_conv_bias"])
    model.conv2.bn.weight.data = load_tensor(weights["conv2_bn_gamma"])
    model.conv2.bn.bias.data = load_tensor(weights["conv2_bn_beta"])
    model.conv2.bn.running_mean.data = load_tensor(weights["conv2_bn_running_mean"])
    model.conv2.bn.running_var.data = load_tensor(weights["conv2_bn_running_var"])

    # Load conv block 3
    model.conv3.conv.weight.data = load_tensor(weights["conv3_conv_weight"])
    model.conv3.conv.bias.data = load_tensor(weights["conv3_conv_bias"])
    model.conv3.bn.weight.data = load_tensor(weights["conv3_bn_gamma"])
    model.conv3.bn.bias.data = load_tensor(weights["conv3_bn_beta"])
    model.conv3.bn.running_mean.data = load_tensor(weights["conv3_bn_running_mean"])
    model.conv3.bn.running_var.data = load_tensor(weights["conv3_bn_running_var"])

    # Load conv block 4
    model.conv4.conv.weight.data = load_tensor(weights["conv4_conv_weight"])
    model.conv4.conv.bias.data = load_tensor(weights["conv4_conv_bias"])
    model.conv4.bn.weight.data = load_tensor(weights["conv4_bn_gamma"])
    model.conv4.bn.bias.data = load_tensor(weights["conv4_bn_beta"])
    model.conv4.bn.running_mean.data = load_tensor(weights["conv4_bn_running_mean"])
    model.conv4.bn.running_var.data = load_tensor(weights["conv4_bn_running_var"])

    # Load FC layers (Burn stores as [in, out], PyTorch expects [out, in])
    model.fc1.weight.data = load_tensor(weights["fc1_weight"]).t()
    model.fc1.bias.data = load_tensor(weights["fc1_bias"])
    model.fc2.weight.data = load_tensor(weights["fc2_weight"]).t()
    model.fc2.bias.data = load_tensor(weights["fc2_bias"])

    print("Weights loaded successfully!")
    
    # Print weight statistics for verification
    print("\nWeight statistics:")
    print(f"  conv1.conv.weight: shape={list(model.conv1.conv.weight.shape)}, "
          f"mean={model.conv1.conv.weight.mean():.4f}, std={model.conv1.conv.weight.std():.4f}")
    print(f"  fc1.weight: shape={list(model.fc1.weight.shape)}, "
          f"mean={model.fc1.weight.mean():.4f}, std={model.fc1.weight.std():.4f}")
    print(f"  fc2.weight: shape={list(model.fc2.weight.shape)}, "
          f"mean={model.fc2.weight.mean():.4f}, std={model.fc2.weight.std():.4f}")


def export_to_onnx(model: PlantClassifier, output_path: Path, image_size: int = 128) -> None:
    """Export PyTorch model to ONNX format."""
    print(f"\nExporting to ONNX: {output_path}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, image_size, image_size)
    
    # Use legacy dynamo_export=False to avoid onnxscript dependency
    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,  # Use legacy export
    )
    
    print(f"ONNX model exported successfully!")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def verify_model(model: PlantClassifier) -> None:
    """Run a quick inference test."""
    print("\nVerifying model...")
    
    model.eval()
    with torch.no_grad():
        # Random input
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        output = model(x)
        probs = F.softmax(output, dim=1)
        
        print(f"  Input shape: {list(x.shape)}")
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Output sum (softmax): {probs.sum():.4f}")
        
        # Top-3 predictions
        top3 = torch.topk(probs, 3)
        print(f"  Top-3 predictions (random input):")
        for i, (prob, idx) in enumerate(zip(top3.values[0], top3.indices[0])):
            print(f"    {i+1}. {CLASS_NAMES[idx]}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Load Burn weights into PyTorch and export to ONNX"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/weights.json"),
        help="Path to weights.json exported by Burn",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.onnx"),
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of output classes",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification test after loading",
    )
    args = parser.parse_args()

    if not args.weights.exists():
        print(f"Error: Weights file not found: {args.weights}")
        print("\nFirst run the Rust export tool:")
        print("  cargo run --release --bin export_weights -- --model best_model.mpk")
        sys.exit(1)

    # Create model
    print("Creating PlantClassifier model...")
    model = PlantClassifier(num_classes=args.num_classes)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load weights
    load_weights_from_json(model, args.weights)

    # Verify
    if args.verify:
        verify_model(model)

    # Export to ONNX
    export_to_onnx(model, args.output)

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nNext steps:")
    print(f"  1. Convert ONNX to TensorFlow.js or ONNX.js format")
    print(f"  2. Create PWA web app for iPhone inference")


if __name__ == "__main__":
    main()
