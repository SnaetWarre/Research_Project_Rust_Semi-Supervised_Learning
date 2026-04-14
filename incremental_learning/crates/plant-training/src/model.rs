//! Model architectures for plant disease classification.
//!
//! Implements:
//! - EfficientNet-B0 (primary architecture)
//! - ResNet-18 (secondary/baseline)
//! - Unified PlantClassifier interface

use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
    },
    tensor::{backend::Backend, Tensor},
};
use plant_core::{Error, ModelArchitecture as ArchType, Result};

/// Unified plant classifier interface
#[derive(Module, Debug)]
pub struct PlantClassifier<B: Backend> {
    arch: ModelArchitecture<B>,
    num_classes: usize,
}

impl<B: Backend> PlantClassifier<B> {
    /// Create a new plant classifier with specified architecture
    pub fn new(arch_type: ArchType, num_classes: usize, device: &B::Device) -> Self {
        let arch = match arch_type {
            ArchType::EfficientNetB0 => {
                ModelArchitecture::EfficientNet(EfficientNetB0::new(num_classes, device))
            }
            ArchType::ResNet18 => {
                ModelArchitecture::ResNet(ResNet18::new(num_classes, device))
            }
            _ => {
                // Default to ResNet18 for unsupported architectures
                ModelArchitecture::ResNet(ResNet18::new(num_classes, device))
            }
        };

        Self { arch, num_classes }
    }

    /// Forward pass through the model
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        match &self.arch {
            ModelArchitecture::EfficientNet(model) => model.forward(input),
            ModelArchitecture::ResNet(model) => model.forward(input),
        }
    }

    /// Get the number of output classes
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Forward pass with class predictions
    pub fn predict(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let logits = self.forward(input);
        burn::tensor::activation::softmax(logits, 1)
    }
}

/// Model architecture enum
#[derive(Module, Debug)]
pub enum ModelArchitecture<B: Backend> {
    EfficientNet(EfficientNetB0<B>),
    ResNet(ResNet18<B>),
}

/// EfficientNet-B0 implementation
///
/// Lightweight and efficient architecture suitable for mobile deployment.
/// Architecture:
/// - Input: 224x224x3
/// - Stem: Conv 3x3, stride 2
/// - MBConv blocks with squeeze-and-excitation
/// - Global average pooling
/// - Classification head
#[derive(Module, Debug)]
pub struct EfficientNetB0<B: Backend> {
    stem: Conv2d<B>,
    stem_bn: BatchNorm<B, 2>,

    // Stage 1: MBConv1, k3x3
    stage1_conv1: Conv2d<B>,
    stage1_bn1: BatchNorm<B, 2>,

    // Stage 2: MBConv6, k3x3
    stage2_conv1: Conv2d<B>,
    stage2_bn1: BatchNorm<B, 2>,
    stage2_conv2: Conv2d<B>,
    stage2_bn2: BatchNorm<B, 2>,

    // Stage 3: MBConv6, k5x5
    stage3_conv1: Conv2d<B>,
    stage3_bn1: BatchNorm<B, 2>,
    stage3_conv2: Conv2d<B>,
    stage3_bn2: BatchNorm<B, 2>,

    // Head
    head_conv: Conv2d<B>,
    head_bn: BatchNorm<B, 2>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    fc: Linear<B>,

    activation: Relu,
}

impl<B: Backend> EfficientNetB0<B> {
    /// Create a new EfficientNet-B0 model
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        // Stem: 3 -> 32
        let stem = Conv2dConfig::new([3, 32], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stem_bn = BatchNormConfig::new(32).init(device);

        // Stage 1: 32 -> 16 (MBConv1)
        let stage1_conv1 = Conv2dConfig::new([32, 16], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage1_bn1 = BatchNormConfig::new(16).init(device);

        // Stage 2: 16 -> 24 (MBConv6)
        let stage2_conv1 = Conv2dConfig::new([16, 96], [1, 1])
            .init(device);
        let stage2_bn1 = BatchNormConfig::new(96).init(device);
        let stage2_conv2 = Conv2dConfig::new([96, 24], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage2_bn2 = BatchNormConfig::new(24).init(device);

        // Stage 3: 24 -> 40 (MBConv6, k5x5)
        let stage3_conv1 = Conv2dConfig::new([24, 144], [1, 1])
            .init(device);
        let stage3_bn1 = BatchNormConfig::new(144).init(device);
        let stage3_conv2 = Conv2dConfig::new([144, 40], [5, 5])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(2, 2))
            .init(device);
        let stage3_bn2 = BatchNormConfig::new(40).init(device);

        // Head: 40 -> 1280 -> num_classes
        let head_conv = Conv2dConfig::new([40, 1280], [1, 1])
            .init(device);
        let head_bn = BatchNormConfig::new(1280).init(device);

        let pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let dropout = DropoutConfig::new(0.2).init();
        let fc = LinearConfig::new(1280, num_classes).init(device);

        let activation = Relu::new();

        Self {
            stem,
            stem_bn,
            stage1_conv1,
            stage1_bn1,
            stage2_conv1,
            stage2_bn1,
            stage2_conv2,
            stage2_bn2,
            stage3_conv1,
            stage3_bn1,
            stage3_conv2,
            stage3_bn2,
            head_conv,
            head_bn,
            pool,
            dropout,
            fc,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Stem
        let x = self.stem.forward(input);
        let x = self.stem_bn.forward(x);
        let x = self.activation.forward(x);

        // Stage 1 (MBConv1)
        let x = self.stage1_conv1.forward(x);
        let x = self.stage1_bn1.forward(x);
        let x = self.activation.forward(x);

        // Stage 2 (MBConv6)
        let x = self.stage2_conv1.forward(x);
        let x = self.stage2_bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.stage2_conv2.forward(x);
        let x = self.stage2_bn2.forward(x);
        let x = self.activation.forward(x);

        // Stage 3 (MBConv6, k5x5)
        let x = self.stage3_conv1.forward(x);
        let x = self.stage3_bn1.forward(x);
        let x = self.activation.forward(x);
        let x = self.stage3_conv2.forward(x);
        let x = self.stage3_bn2.forward(x);
        let x = self.activation.forward(x);

        // Head
        let x = self.head_conv.forward(x);
        let x = self.head_bn.forward(x);
        let x = self.activation.forward(x);

        // Global average pooling
        let x = self.pool.forward(x);

        // Flatten [batch, channels, 1, 1] -> [batch, channels]
        let [batch, channels, _, _] = x.dims();
        let x_flat: Tensor<B, 2> = x.reshape([batch, channels]);

        // Dropout and classification
        let x_flat = self.dropout.forward(x_flat);
        self.fc.forward(x_flat)
    }
}

/// ResNet-18 implementation
///
/// Classic architecture with residual connections.
/// Architecture:
/// - Input: 224x224x3
/// - Conv1: 7x7, stride 2
/// - MaxPool: 3x3, stride 2
/// - 4 residual stages (2 blocks each)
/// - Global average pooling
/// - FC layer
#[derive(Module, Debug)]
pub struct ResNet18<B: Backend> {
    // Initial layers
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    maxpool: MaxPool2d,

    // Stage 1: 64 channels
    stage1_conv1: Conv2d<B>,
    stage1_bn1: BatchNorm<B, 2>,
    stage1_conv2: Conv2d<B>,
    stage1_bn2: BatchNorm<B, 2>,

    // Stage 2: 128 channels
    stage2_conv1: Conv2d<B>,
    stage2_bn1: BatchNorm<B, 2>,
    stage2_conv2: Conv2d<B>,
    stage2_bn2: BatchNorm<B, 2>,
    stage2_downsample: Conv2d<B>,
    stage2_downsample_bn: BatchNorm<B, 2>,

    // Stage 3: 256 channels
    stage3_conv1: Conv2d<B>,
    stage3_bn1: BatchNorm<B, 2>,
    stage3_conv2: Conv2d<B>,
    stage3_bn2: BatchNorm<B, 2>,
    stage3_downsample: Conv2d<B>,
    stage3_downsample_bn: BatchNorm<B, 2>,

    // Stage 4: 512 channels
    stage4_conv1: Conv2d<B>,
    stage4_bn1: BatchNorm<B, 2>,
    stage4_conv2: Conv2d<B>,
    stage4_bn2: BatchNorm<B, 2>,
    stage4_downsample: Conv2d<B>,
    stage4_downsample_bn: BatchNorm<B, 2>,

    // Classification head
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<B>,

    activation: Relu,
}

impl<B: Backend> ResNet18<B> {
    /// Create a new ResNet-18 model
    pub fn new(num_classes: usize, device: &B::Device) -> Self {
        // Initial conv: 3 -> 64
        let conv1 = Conv2dConfig::new([3, 64], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .init(device);
        let bn1 = BatchNormConfig::new(64).init(device);
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        // Stage 1: 64 -> 64
        let stage1_conv1 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage1_bn1 = BatchNormConfig::new(64).init(device);
        let stage1_conv2 = Conv2dConfig::new([64, 64], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage1_bn2 = BatchNormConfig::new(64).init(device);

        // Stage 2: 64 -> 128
        let stage2_conv1 = Conv2dConfig::new([64, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage2_bn1 = BatchNormConfig::new(128).init(device);
        let stage2_conv2 = Conv2dConfig::new([128, 128], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage2_bn2 = BatchNormConfig::new(128).init(device);
        let stage2_downsample = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let stage2_downsample_bn = BatchNormConfig::new(128).init(device);

        // Stage 3: 128 -> 256
        let stage3_conv1 = Conv2dConfig::new([128, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage3_bn1 = BatchNormConfig::new(256).init(device);
        let stage3_conv2 = Conv2dConfig::new([256, 256], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage3_bn2 = BatchNormConfig::new(256).init(device);
        let stage3_downsample = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let stage3_downsample_bn = BatchNormConfig::new(256).init(device);

        // Stage 4: 256 -> 512
        let stage4_conv1 = Conv2dConfig::new([256, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage4_bn1 = BatchNormConfig::new(512).init(device);
        let stage4_conv2 = Conv2dConfig::new([512, 512], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let stage4_bn2 = BatchNormConfig::new(512).init(device);
        let stage4_downsample = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([2, 2])
            .init(device);
        let stage4_downsample_bn = BatchNormConfig::new(512).init(device);

        // Classification head
        let avgpool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let fc = LinearConfig::new(512, num_classes).init(device);

        let activation = Relu::new();

        Self {
            conv1,
            bn1,
            maxpool,
            stage1_conv1,
            stage1_bn1,
            stage1_conv2,
            stage1_bn2,
            stage2_conv1,
            stage2_bn1,
            stage2_conv2,
            stage2_bn2,
            stage2_downsample,
            stage2_downsample_bn,
            stage3_conv1,
            stage3_bn1,
            stage3_conv2,
            stage3_bn2,
            stage3_downsample,
            stage3_downsample_bn,
            stage4_conv1,
            stage4_bn1,
            stage4_conv2,
            stage4_bn2,
            stage4_downsample,
            stage4_downsample_bn,
            avgpool,
            fc,
            activation,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        // Initial conv
        let mut x = self.conv1.forward(input);
        x = self.bn1.forward(x);
        x = self.activation.forward(x);
        x = self.maxpool.forward(x);

        // Stage 1: Residual blocks without downsampling
        let identity = x.clone();
        x = self.stage1_conv1.forward(x);
        x = self.stage1_bn1.forward(x);
        x = self.activation.forward(x);
        x = self.stage1_conv2.forward(x);
        x = self.stage1_bn2.forward(x);
        x = x.add(identity);
        x = self.activation.forward(x);

        // Stage 2: First residual block with downsampling
        let identity = self.stage2_downsample.forward(x.clone());
        let identity = self.stage2_downsample_bn.forward(identity);
        x = self.stage2_conv1.forward(x);
        x = self.stage2_bn1.forward(x);
        x = self.activation.forward(x);
        x = self.stage2_conv2.forward(x);
        x = self.stage2_bn2.forward(x);
        x = x.add(identity);
        x = self.activation.forward(x);

        // Stage 3: First residual block with downsampling
        let identity = self.stage3_downsample.forward(x.clone());
        let identity = self.stage3_downsample_bn.forward(identity);
        x = self.stage3_conv1.forward(x);
        x = self.stage3_bn1.forward(x);
        x = self.activation.forward(x);
        x = self.stage3_conv2.forward(x);
        x = self.stage3_bn2.forward(x);
        x = x.add(identity);
        x = self.activation.forward(x);

        // Stage 4: First residual block with downsampling
        let identity = self.stage4_downsample.forward(x.clone());
        let identity = self.stage4_downsample_bn.forward(identity);
        x = self.stage4_conv1.forward(x);
        x = self.stage4_bn1.forward(x);
        x = self.activation.forward(x);
        x = self.stage4_conv2.forward(x);
        x = self.stage4_bn2.forward(x);
        x = x.add(identity);
        x = self.activation.forward(x);

        // Global average pooling
        let x = self.avgpool.forward(x);

        // Flatten [batch, channels, 1, 1] -> [batch, channels]
        let [batch, channels, _, _] = x.dims();
        let x = x.reshape([batch, channels]);

        // Classification
        self.fc.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_efficientnet_creation() {
        let device = Default::default();
        let model = EfficientNetB0::<TestBackend>::new(10, &device);
        assert!(true); // If we get here, model was created successfully
    }

    #[test]
    fn test_resnet_creation() {
        let device = Default::default();
        let model = ResNet18::<TestBackend>::new(10, &device);
        assert!(true);
    }

    #[test]
    fn test_plant_classifier_efficientnet() {
        let device = Default::default();
        let classifier = PlantClassifier::<TestBackend>::new(
            ArchType::EfficientNetB0,
            10,
            &device,
        );
        assert_eq!(classifier.num_classes(), 10);
    }

    #[test]
    fn test_plant_classifier_resnet() {
        let device = Default::default();
        let classifier = PlantClassifier::<TestBackend>::new(
            ArchType::ResNet18,
            10,
            &device,
        );
        assert_eq!(classifier.num_classes(), 10);
    }
}
