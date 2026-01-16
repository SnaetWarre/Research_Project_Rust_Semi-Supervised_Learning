"""
PyTorch Reference Implementation - Model Architecture

This module implements the same CNN architecture as the Rust/Burn implementation
for comparison and benchmarking purposes.

Author: Warre Snaet
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# PlantVillage dataset configuration
NUM_CLASSES = 39
IMAGE_SIZE = 256


class ConvBlock(nn.Module):
    """Convolutional block with Conv2d, BatchNorm, ReLU, and optional MaxPool."""

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
    Plant Disease Classifier CNN

    Architecture matches the Rust/Burn implementation:
    - 4 convolutional blocks with increasing filter sizes
    - BatchNorm and ReLU after each convolution
    - MaxPooling after each block
    - Global Average Pooling
    - Fully connected classifier with dropout
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        input_channels: int = 3,
        base_filters: int = 32,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.num_classes = num_classes

        # Convolutional blocks: 3 -> 32 -> 64 -> 128 -> 256
        self.conv1 = ConvBlock(input_channels, base_filters, 3, True)  # 256 -> 128
        self.conv2 = ConvBlock(base_filters, base_filters * 2, 3, True)  # 128 -> 64
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4, 3, True)  # 64 -> 32
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8, 3, True)  # 32 -> 16

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (matches Burn architecture: 256 -> 256 -> num_classes)
        self.fc1 = nn.Linear(base_filters * 8, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Global pooling: [B, C, H, W] -> [B, C, 1, 1]
        x = self.global_pool(x)

        # Flatten: [B, C, 1, 1] -> [B, C]
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def forward_softmax(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax for inference."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.

        Returns:
            tuple: (predicted_classes, confidence_scores)
        """
        probs = self.forward_softmax(x)
        confidence, predicted = probs.max(dim=1)
        return predicted, confidence


class PlantClassifierLite(nn.Module):
    """
    Lightweight Plant Disease Classifier for edge deployment.

    Smaller model with fewer parameters for faster inference
    on resource-constrained devices like Jetson Orin Nano.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
        super().__init__()

        self.num_classes = num_classes

        # Smaller convolutional blocks
        self.conv1 = nn.Conv2d(3, 16, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")
        self.bn3 = nn.BatchNorm2d(64)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.fc = nn.Linear(64, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the size of a model in megabytes."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


if __name__ == "__main__":
    # Test the models
    print("=" * 60)
    print("PlantVillage PyTorch Reference Models")
    print("=" * 60)

    # Test full model
    model = PlantClassifier()
    print(f"\nPlantClassifier:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Size: {get_model_size_mb(model):.2f} MB")

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Test lite model
    model_lite = PlantClassifierLite()
    print(f"\nPlantClassifierLite:")
    print(f"  Parameters: {count_parameters(model_lite):,}")
    print(f"  Size: {get_model_size_mb(model_lite):.2f} MB")

    output_lite = model_lite(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output_lite.shape}")
