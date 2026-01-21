//! CNN Model Architecture for Plant Disease Classification
//!
//! This module implements a Convolutional Neural Network using the Burn framework
//! for classifying plant diseases from leaf images. The architecture is designed
//! to be efficient enough for edge deployment on NVIDIA Jetson Orin Nano.

use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig, MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        Relu,
    },
    tensor::{backend::Backend, Tensor},
};

/// Configuration for the PlantClassifier CNN model
#[derive(Config, Debug)]
pub struct PlantClassifierConfig {
    /// Number of output classes (default: 38 for PlantVillage)
    #[config(default = "38")]
    pub num_classes: usize,

    /// Input image size (assumes square images)
    #[config(default = "256")]
    pub input_size: usize,

    /// Dropout rate for regularization
    #[config(default = "0.3")]
    pub dropout_rate: f64,

    /// Number of input channels (3 for RGB)
    #[config(default = "3")]
    pub in_channels: usize,

    /// Base number of convolutional filters
    #[config(default = "32")]
    pub base_filters: usize,
}

/// A CNN block with Conv2d, BatchNorm, ReLU, and optional MaxPool
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: BatchNorm<B>,
    pub relu: Relu,
    pub pool: Option<MaxPool2d>,
}

impl<B: Backend> ConvBlock<B> {
    /// Create a new convolutional block
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        with_pool: bool,
        device: &B::Device,
    ) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let bn = BatchNormConfig::new(out_channels).init(device);

        let pool = if with_pool {
            Some(MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init())
        } else {
            None
        };

        Self {
            conv,
            bn,
            relu: Relu::new(),
            pool,
        }
    }

    /// Forward pass through the block
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        let x = self.relu.forward(x);

        match &self.pool {
            Some(pool) => pool.forward(x),
            None => x,
        }
    }
}

/// Plant Disease Classifier CNN
///
/// Architecture:
/// - 4 convolutional blocks with increasing filter sizes
/// - BatchNorm and ReLU after each convolution
/// - MaxPooling after each block
/// - Global Average Pooling
/// - Fully connected classifier with dropout
#[derive(Module, Debug)]
pub struct PlantClassifier<B: Backend> {
    // Convolutional blocks (public for weight export)
    pub conv1: ConvBlock<B>,
    pub conv2: ConvBlock<B>,
    pub conv3: ConvBlock<B>,
    pub conv4: ConvBlock<B>,

    // Global pooling
    pub global_pool: AdaptiveAvgPool2d,

    // Classifier head (public for weight export)
    pub fc1: Linear<B>,
    pub dropout: Dropout,
    pub fc2: Linear<B>,

    // Store config for reference
    num_classes: usize,
}

impl<B: Backend> PlantClassifier<B> {
    /// Create a new PlantClassifier from configuration
    pub fn new(config: &PlantClassifierConfig, device: &B::Device) -> Self {
        let base = config.base_filters;

        // Convolutional blocks: 3 -> 32 -> 64 -> 128 -> 256
        let conv1 = ConvBlock::new(config.in_channels, base, 3, true, device); // 256 -> 128
        let conv2 = ConvBlock::new(base, base * 2, 3, true, device); // 128 -> 64
        let conv3 = ConvBlock::new(base * 2, base * 4, 3, true, device); // 64 -> 32
        let conv4 = ConvBlock::new(base * 4, base * 8, 3, true, device); // 32 -> 16

        // Global average pooling
        let global_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        // Fully connected layers with stronger capacity
        let fc1 = LinearConfig::new(base * 8, 256).init(device);
        let dropout = DropoutConfig::new(config.dropout_rate).init();
        let fc2 = LinearConfig::new(256, config.num_classes).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            conv4,
            global_pool,
            fc1,
            dropout,
            fc2,
            num_classes: config.num_classes,
        }
    }

    /// Forward pass through the network
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, 3, height, width]
    ///
    /// # Returns
    /// * Logits tensor of shape [batch_size, num_classes]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Convolutional feature extraction
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);
        let x = self.conv4.forward(x);

        // Global pooling: [B, C, H, W] -> [B, C, 1, 1]
        let x = self.global_pool.forward(x);

        // Flatten: [B, C, 1, 1] -> [B, C]
        let [batch_size, channels, _, _] = x.dims();
        let x = x.reshape([batch_size, channels]);

        // Classifier
        let x = self.fc1.forward(x);
        let x = Relu::new().forward(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);

        x
    }

    /// Forward pass with softmax for inference
    pub fn forward_softmax(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        burn::tensor::activation::softmax(logits, 1)
    }

    /// Get the number of output classes
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// A smaller, faster model for edge deployment
#[derive(Module, Debug)]
pub struct PlantClassifierLite<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B>,
    pool1: MaxPool2d,

    conv2: Conv2d<B>,
    bn2: BatchNorm<B>,
    pool2: MaxPool2d,

    conv3: Conv2d<B>,
    bn3: BatchNorm<B>,

    global_pool: AdaptiveAvgPool2d,

    fc: Linear<B>,
    dropout: Dropout,
    classifier: Linear<B>,

    num_classes: usize,
}

impl<B: Backend> PlantClassifierLite<B> {
    /// Create a lightweight classifier for edge deployment
    pub fn new(num_classes: usize, dropout_rate: f64, device: &B::Device) -> Self {
        // Smaller filter sizes for faster inference
        let conv1 = Conv2dConfig::new([3, 16], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn1 = BatchNormConfig::new(16).init(device);
        let pool1 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv2 = Conv2dConfig::new([16, 32], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn2 = BatchNormConfig::new(32).init(device);
        let pool2 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        let conv3 = Conv2dConfig::new([32, 64], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .init(device);
        let bn3 = BatchNormConfig::new(64).init(device);

        let global_pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();

        let fc = LinearConfig::new(64, 128).init(device);
        let dropout = DropoutConfig::new(dropout_rate).init();
        let classifier = LinearConfig::new(128, num_classes).init(device);

        Self {
            conv1,
            bn1,
            pool1,
            conv2,
            bn2,
            pool2,
            conv3,
            bn3,
            global_pool,
            fc,
            dropout,
            classifier,
            num_classes,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        // Block 1
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = Relu::new().forward(x);
        let x = self.pool1.forward(x);

        // Block 2
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = Relu::new().forward(x);
        let x = self.pool2.forward(x);

        // Block 3
        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = Relu::new().forward(x);

        // Global pooling
        let x = self.global_pool.forward(x);

        // Flatten
        let [batch_size, channels, _, _] = x.dims();
        let x = x.reshape([batch_size, channels]);

        // Classifier
        let x = self.fc.forward(x);
        let x = Relu::new().forward(x);
        let x = self.dropout.forward(x);
        self.classifier.forward(x)
    }

    /// Forward with softmax
    pub fn forward_softmax(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let logits = self.forward(x);
        burn::tensor::activation::softmax(logits, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cuda::Cuda;

    type TestBackend = Cuda;

    #[test]
    fn test_plant_classifier_output_shape() {
        let device = Default::default();
        let config = PlantClassifierConfig::new();
        let model = PlantClassifier::<TestBackend>::new(&config, &device);

        // Create dummy input: [batch=2, channels=3, height=256, width=256]
        let input = Tensor::<TestBackend, 4>::zeros([2, 3, 256, 256], &device);

        let output = model.forward(input);
        let dims = output.dims();

        assert_eq!(dims[0], 2); // batch size
        assert_eq!(dims[1], 38); // num classes (PlantVillage has 38 classes)
    }

    #[test]
    fn test_plant_classifier_lite_output_shape() {
        let device = Default::default();
        let model = PlantClassifierLite::<TestBackend>::new(38, 0.5, &device);

        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 256, 256], &device);

        let output = model.forward(input);
        let dims = output.dims();

        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 38); // PlantVillage has 38 classes
    }
}
