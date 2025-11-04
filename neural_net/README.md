# Neural Network Library from Scratch

A pure Rust neural network library built from scratch using only `ndarray` for tensor operations. This library provides a clean, modular API for building, training, and using neural networks without relying on high-level ML frameworks like PyTorch or TensorFlow.

## Features

- 🧮 **Pure Rust Implementation**: Built from scratch with minimal dependencies
- 🔢 **Tensor Operations**: Efficient matrix operations using `ndarray`
- 🏗️ **Modular Architecture**: Clean separation of layers, activations, losses, and optimizers
- 🎯 **Multiple Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax, Linear
- 📉 **Various Loss Functions**: MSE, MAE, Binary Cross-Entropy, Cross-Entropy
- 🚀 **Modern Optimizers**: SGD (with momentum), Adam, RMSprop
- 🧪 **Well-Tested**: Comprehensive test coverage for all components
- 📊 **Easy to Use**: Builder pattern and intuitive API

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
neural_net = { path = "path/to/neural_net" }
```

## Quick Start

Here's a simple example solving the XOR problem:

```rust
use neural_net::*;
use std::sync::Arc;

fn main() {
    // Create training data
    let x_train = Tensor::from(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    let y_train = Tensor::from(vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ]);

    // Build the network
    let mut network = NetworkBuilder::with_name("XOR Network")
        .add(Box::new(Dense::new(2, 4, Arc::new(ReLU))))
        .add(Box::new(Dense::new(4, 1, Arc::new(Sigmoid))))
        .build();

    // Define loss and optimizer
    let loss_fn = MeanSquaredError;
    let mut optimizer = Adam::new(0.1);

    // Train the network
    network.fit(
        &x_train,
        &y_train,
        &loss_fn,
        &mut optimizer,
        1000,  // epochs
        4,     // batch_size
        true,  // verbose
    );

    // Make predictions
    let input = Tensor::from(vec![vec![1.0, 0.0]]);
    let prediction = network.predict(&input);
    println!("Prediction: {}", prediction.data[[0, 0]]);
}
```

## Architecture

The library is organized into several key modules:

### 1. Tensors (`tensor.rs`)

Wrapper around `ndarray::Array2` with convenience methods:

```rust
// Create tensors
let t1 = Tensor::zeros(3, 4);
let t2 = Tensor::ones(3, 4);
let t3 = Tensor::random(3, 4, 0.0, 1.0);
let t4 = Tensor::xavier(3, 4);  // Xavier initialization
let t5 = Tensor::he(3, 4);      // He initialization

// Operations
let result = t1.matmul(&t2);    // Matrix multiplication
let result = t1.add(&t2);       // Element-wise addition
let result = t1.mul(&t2);       // Element-wise multiplication
let result = t1.transpose();     // Transpose
```

### 2. Activation Functions (`activation.rs`)

Implements the `Activation` trait with forward and backward passes:

- **ReLU**: `f(x) = max(0, x)`
- **Leaky ReLU**: `f(x) = max(αx, x)`
- **Sigmoid**: `f(x) = 1 / (1 + exp(-x))`
- **Tanh**: `f(x) = tanh(x)`
- **Softmax**: `f(x_i) = exp(x_i) / Σ exp(x_j)`
- **Linear**: `f(x) = x`

```rust
let relu = Arc::new(ReLU);
let sigmoid = Arc::new(Sigmoid);
let leaky_relu = Arc::new(LeakyReLU::new(0.01));
```

### 3. Layers (`layer.rs`)

Currently implements **Dense (Fully Connected)** layers:

```rust
// Create a dense layer: input_size -> output_size with activation
let layer = Dense::new(10, 5, Arc::new(ReLU));

// Custom initialization
let layer = Dense::with_weights(weights, bias, Arc::new(Sigmoid));
```

### 4. Loss Functions (`loss.rs`)

Implements the `Loss` trait with loss computation and gradient calculation:

- **Mean Squared Error (MSE)**: For regression
- **Mean Absolute Error (MAE)**: Robust to outliers
- **Binary Cross-Entropy**: For binary classification
- **Cross-Entropy**: For multi-class classification
- **Softmax Cross-Entropy**: Numerically stable combined loss

```rust
let mse = MeanSquaredError;
let mae = MeanAbsoluteError;
let bce = BinaryCrossEntropy::new();
let ce = CrossEntropy::new();
```

### 5. Optimizers (`optimizer.rs`)

Implements the `Optimizer` trait for parameter updates:

- **SGD**: Stochastic Gradient Descent (with optional momentum)
- **Adam**: Adaptive Moment Estimation
- **RMSprop**: Root Mean Square Propagation

```rust
let sgd = SGD::new(0.01);
let sgd_momentum = SGD::with_momentum(0.01, 0.9);
let adam = Adam::new(0.001);
let rmsprop = RMSprop::new(0.001);
```

### 6. Network (`network.rs`)

High-level API for building and training neural networks:

```rust
// Using builder pattern
let mut network = NetworkBuilder::new()
    .add(Box::new(Dense::new(784, 128, Arc::new(ReLU))))
    .add(Box::new(Dense::new(128, 64, Arc::new(ReLU))))
    .add(Box::new(Dense::new(64, 10, Arc::new(Softmax))))
    .build();

// Or manually
let mut network = Network::new();
network.add_layer(Box::new(Dense::new(10, 5, Arc::new(ReLU))));

// Training
let losses = network.fit(
    &x_train,
    &y_train,
    &loss_fn,
    &mut optimizer,
    epochs,
    batch_size,
    verbose,
);

// Evaluation
let test_loss = network.evaluate(&x_test, &y_test, &loss_fn);
let accuracy = network.accuracy(&x_test, &y_test);

// Display architecture
network.summary();
```

## Examples

### XOR Problem

```bash
cargo run --example xor_example
```

Demonstrates solving the classic XOR problem with a 2-4-1 network.

### Sine Wave Regression

```bash
cargo run --example regression_example
```

Shows how to use the library for regression tasks by approximating the sine function.

## Project Structure

```
neural_net/
├── src/
│   ├── lib.rs           # Library entry point and re-exports
│   ├── tensor.rs        # Tensor operations
│   ├── activation.rs    # Activation functions
│   ├── layer.rs         # Layer implementations
│   ├── loss.rs          # Loss functions
│   ├── optimizer.rs     # Optimization algorithms
│   └── network.rs       # Network orchestration
├── examples/
│   ├── xor_example.rs
│   └── regression_example.rs
├── Cargo.toml
└── README.md
```

## Design Principles

1. **Modularity**: Each component (layers, activations, losses, optimizers) is independent and interchangeable
2. **Type Safety**: Leverages Rust's type system for compile-time safety
3. **Performance**: Uses efficient ndarray operations
4. **Extensibility**: Easy to add new layers, activations, losses, and optimizers
5. **Clarity**: Clean, readable code with comprehensive documentation

## How It Works

### Forward Pass

1. Input passes through each layer sequentially
2. Each layer applies: `output = activation(input * weights + bias)`
3. Intermediate outputs are cached for backpropagation

### Backward Pass

1. Compute loss gradient w.r.t. network output
2. Propagate gradients backward through each layer
3. Each layer computes:
   - Gradient w.r.t. weights: `∂L/∂W = input^T * ∂L/∂output`
   - Gradient w.r.t. bias: `∂L/∂b = sum(∂L/∂output)`
   - Gradient w.r.t. input: `∂L/∂input = ∂L/∂output * W^T`

### Parameter Update

1. Optimizer collects all parameter gradients
2. Applies optimization algorithm (SGD, Adam, etc.)
3. Updates parameters: `θ = θ - learning_rate * gradient`

## Testing

Run all tests:

```bash
cargo test
```

Run tests with output:

```bash
cargo test -- --nocapture
```

The library includes comprehensive unit tests for:
- Tensor operations
- Activation function forward/backward passes
- Layer computations
- Loss function calculations
- Optimizer updates
- Network training

## Performance Considerations

- Uses `ndarray` for efficient matrix operations
- Gradient cloning in training loop (can be optimized further)
- Batch processing for better throughput
- No GPU support yet (CPU only)

## Future Enhancements

Possible extensions to the library:

- [ ] Convolutional layers (Conv2D)
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Dropout and batch normalization
- [ ] More activation functions (ELU, GELU, Swish)
- [ ] Learning rate scheduling
- [ ] Model serialization (save/load)
- [ ] Data augmentation utilities
- [ ] GPU acceleration
- [ ] Automatic differentiation system
- [ ] Visualization tools

## Contributing

This is a learning/research project demonstrating neural network fundamentals in Rust. Feel free to:

1. Fork the repository
2. Add new features or improvements
3. Write more examples
4. Improve documentation
5. Optimize performance

## License

This project is provided as-is for educational purposes.

## Acknowledgments

Built as a learning exercise to understand:
- Neural network fundamentals
- Backpropagation algorithm
- Rust's ownership and type system
- Scientific computing in Rust

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [ndarray documentation](https://docs.rs/ndarray/)
- [Understanding Backpropagation](https://brilliant.org/wiki/backpropagation/)

---

**Note**: This is a from-scratch implementation meant for learning. For production use cases, consider using established frameworks like `tch-rs` (PyTorch bindings) or `burn` (native Rust deep learning framework).